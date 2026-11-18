"""The default scenario holds Sentinel-1A and 1C to ESA's reference orbit:
sun-synchronous at 18:00 LTAN, 175 revolutions in 12 days, frozen perigee
over the North Pole, the two satellites half an orbit apart."""

import math
from pathlib import Path

import numpy as np
import pytest

from satflight.analysis import osculating_to_mean_i
from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.integrators import Propagator
from satflight.orbitinfo import local_time_of_node
from satflight.scenario import Scenario
from satflight.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
REPEAT_PERIOD = 12 * 86400 / 175          # s, node to node


@pytest.fixture(scope="module")
def sim():
    return Simulation(Scenario.load(ROOT / "scenarios" / "default.json"))


def test_it_is_sentinel_1a_and_1c_with_j2_to_j4_and_drag(sim):
    assert [s.name for s in sim.sats] == ["Sentinel-1A", "Sentinel-1C"]
    f = sim.scenario.forces
    assert f.j2 and f.j3 and f.j4 and f.drag


def test_reference_orbit_elements(sim):
    jd = sim.clock.jd(0.0)
    for k in range(2):
        el = rv2coe(sim.y[k, :3], sim.y[k, 3:])
        i_mean = math.degrees(float(osculating_to_mean_i(el.i, el.a, el.e, el.argp, el.nu)))
        assert i_mean == pytest.approx(98.18, abs=0.01)
        assert local_time_of_node(el.raan, jd) == pytest.approx(18.0, abs=1 / 60)
        assert 7060 < el.a < 7085                       # osculating, around 7071 km mean


def test_1c_flies_half_an_orbit_behind_1a(sim):
    u = [rv2coe(sim.y[k, :3], sim.y[k, 3:]) for k in range(2)]
    lat_arg = [math.degrees(e.argp + e.nu) % 360 for e in u]
    assert (lat_arg[0] - lat_arg[1]) % 360 == pytest.approx(180.0, abs=0.05)


def test_node_to_node_period_closes_the_12_day_repeat(sim):
    """Flown with the run's own gravity field, three nodal periods each match
    12 d / 175 to 10 ms: under 2 s (15 km along track) after a whole cycle."""
    model = ForceModel(j2=True, j3=True, j4=True)

    def f(t, y):
        return np.concatenate([y[:, 3:], model.acceleration(y[:, :3], y[:, 3:],
                                                            np.zeros(len(y)))], axis=1)

    def at(y0, t0, t1):
        return Propagator("dopri5", rtol=1e-11, atol=1e-9, h_max=30.0).integrate(
            f, t0, y0, t1)[1]

    y, t, nodes = sim.y[:1].copy(), 0.0, []
    while len(nodes) < 4:
        y1 = at(y, t, t + 60.0)
        if y[0, 2] < 0 <= y1[0, 2]:
            tn, yn = t, y
            for _ in range(8):
                dt = -yn[0, 2] / yn[0, 5]
                tn += dt
                yn = at(y, t, tn)
                if abs(dt) < 1e-7:
                    break
            nodes.append(tn)
        t, y = t + 60.0, y1
    for p in np.diff(nodes):
        assert p == pytest.approx(REPEAT_PERIOD, abs=0.01)
