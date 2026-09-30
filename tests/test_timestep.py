"""The physics must not depend on the frame rate: advancing a simulation in
many small calls (one per UI frame) has to follow exactly the same trajectory,
and log exactly the same events, as advancing it in one call."""

import itertools
from datetime import datetime

import numpy as np
import pytest

from satflight.constants import MU_EARTH, R_EARTH
from satflight.elements import kepler_propagate
from satflight.forces import ForceModel
from satflight.launch import LAUNCH_SITES, LaunchSpec, vehicle_preset
from satflight.maneuvers import Maneuver
from satflight.scenario import IntegratorSettings, SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.timeutil import UTC

EPOCH = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
# frame lengths in simulated seconds; binary fractions so their sums are exact
FRAMES = (0.5, 1.25, 2.0, 0.25, 3.75, 0.125)


def _in_frames(sim: Simulation, total: float):
    """Advance ``sim`` by ``total`` seconds in uneven, frame-sized calls."""
    for dt in itertools.cycle(FRAMES):
        if sim.t >= total:
            break
        sim.advance(min(dt, total - sim.t))


def _same(a: Simulation, b: Simulation):
    assert a.t == b.t
    assert np.array_equal(a.y, b.y)
    assert [(e.t, e.text) for e in a.events] == [(e.t, e.text) for e in b.events]
    assert [s.status for s in a.sats] == [s.status for s in b.sats]
    ta, ya = a.history.series()
    tb, yb = b.history.series()
    assert np.array_equal(ta, tb) and np.array_equal(ya, yb)


def _orbits(propagator: str, method: str) -> Scenario:
    sats = [SatSpec("low", {"type": "elements", "altitude": 300, "i": 51.6}, mass=200, area=3),
            SatSpec("mid", {"type": "elements", "altitude": 1200, "i": 98, "e": 0.02})]
    sc = Scenario(name="frames", epoch=EPOCH, satellites=sats, propagator=propagator,
                  forces=ForceModel(j2=True, drag=True),
                  integrator=IntegratorSettings(method=method))
    return sc


@pytest.mark.parametrize("propagator,method", [("cowell", "dopri5"), ("cowell", "rk4"),
                                               ("cowell", "leapfrog"), ("j2mean", "dopri5"),
                                               ("kepler", "dopri5")])
def test_orbits_do_not_depend_on_the_frame_rate(propagator, method):
    sims = [Simulation(_orbits(propagator, method)) for _ in range(2)]
    for sim in sims:
        sim.schedule(Maneuver("low", "impulse", dv=(0.05, 0, 0), timing="delay", delay=700.3))
        sim.schedule(Maneuver("mid", "finite", thrust=400.0, isp=300.0, duration=120.0,
                              dv=(1, 0, 0), timing="delay", delay=1500.0))
    sims[0].advance(3000.0)
    _in_frames(sims[1], 3000.0)
    _same(*sims)


def test_launch_does_not_depend_on_the_frame_rate():
    lat, lon, alt = LAUNCH_SITES["Cape Canaveral SLC-40 (USA)"]
    launch = LaunchSpec(name="Sat", lat=lat, lon=lon, alt=alt, timing="delay", delay=60.0,
                        vehicle=vehicle_preset("Falcon 9 (approx.)"), payload_mass=5000.0)
    iss = SatSpec("ISS", {"type": "elements", "altitude": 420, "i": 51.64})
    sc = Scenario(name="launch", epoch=EPOCH, satellites=[iss], launches=[launch],
                  forces=ForceModel(j2=True, drag=True))
    one, many = Simulation(sc.copy()), Simulation(sc.copy())
    one.advance(900.0)
    _in_frames(many, 900.0)
    _same(one, many)
    assert "Sat Stage 1" in [s.name for s in one.sats]      # the flight got that far


def test_a_frame_between_steps_reports_the_state_at_its_own_time():
    """The state shown between two physics steps is the integrated state at
    the requested time, not the last step's."""
    sim = Simulation(_orbits("cowell", "rk4"))                # 10 s steps
    ref = Simulation(_orbits("kepler", "dopri5"))
    sim.forces = ref.forces = ForceModel(j2=False)
    sim.advance(3.0)
    ref.advance(3.0)
    assert sim.t == 3.0
    assert np.allclose(sim.y, ref.y, atol=1e-6)


def test_changing_a_state_between_frames_restarts_the_physics_there():
    sim = Simulation(_orbits("cowell", "dopri5"))
    sim.advance(7.0)
    sim.y[0, 3:] *= 1.01                                       # e.g. an edit from the UI
    ref = Simulation(sim.snapshot_scenario())
    sim.advance(1.0)
    ref.advance(1.0)
    assert np.allclose(sim.y, ref.y, atol=1e-5)


def _crossing(miss_km: float, t_meet: float) -> Scenario:
    """Two circular orbits, equatorial and polar, whose satellites pass
    ``miss_km`` apart over the node at ``t_meet`` (two-body motion)."""
    r = R_EARTH + 500.0
    v = float(np.sqrt(MU_EARTH / r))
    rb = r + miss_km
    sats = []
    for name, r1, v1 in (("Eq", [r, 0, 0], [0, v, 0]),
                         ("Polar", [rb, 0, 0], [0, 0, float(np.sqrt(MU_EARTH / rb))])):
        r0, v0 = kepler_propagate(np.array(r1, float), np.array(v1, float), -t_meet)
        sats.append(SatSpec(name, {"type": "state", "r": r0.tolist(), "v": v0.tolist()}))
    return Scenario(name="crossing", epoch=EPOCH, satellites=sats, forces=ForceModel(j2=False))


def test_a_close_pass_between_two_steps_is_caught():
    t_meet = 1234.567
    sim = Simulation(_crossing(3.0, t_meet))
    sim.advance(2000.0)
    passes = [e for e in sim.events if e.text.startswith("Close approach")]
    assert len(passes) == 1
    assert passes[0].t == pytest.approx(t_meet, abs=0.05)
    assert "3.00 km" in passes[0].text
    assert not any(abs(t - t_meet) < 1e-6 for t in sim.history.times)   # between steps
    framed = Simulation(_crossing(3.0, t_meet))
    _in_frames(framed, 2000.0)
    _same(sim, framed)


def test_a_wide_pass_is_not_reported():
    sim = Simulation(_crossing(25.0, 1234.567))
    sim.advance(2000.0)
    assert not any(e.text.startswith("Close approach") for e in sim.events)
