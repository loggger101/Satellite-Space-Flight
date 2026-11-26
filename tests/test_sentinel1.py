"""The default scenario holds Sentinel-1A and 1C to ESA's reference orbit:
sun-synchronous at 18:00 LTAN, 175 revolutions in 12 days, frozen perigee
over the North Pole, 1C trailing 1A by the 176 deg the precise orbits show."""

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
        # 1A on the reference 18:00; 1C's plane a minute earlier, onto 1A's tracks
        ltan = (18.0, 18.0 - 1.0 / 60)[k]
        assert local_time_of_node(el.raan, jd) == pytest.approx(ltan, abs=0.2 / 60)
        assert 7060 < el.a < 7085                       # osculating, around 7071 km mean


def test_1c_trails_1a_by_the_measured_176_degrees(sim):
    """Not the nominal 180: what the precise orbits show from July 2025 until
    1C was moved in June 2026 (phasing.csv)."""
    import csv
    with open(STUDY / "phasing.csv", newline="", encoding="utf-8") as f:
        real = np.mean([float(r["mean_deg"]) for r in csv.DictReader(f)
                        if r["date"] < "2026-06-01"])
    # measured as phasing.csv is: when 1C crosses the equator northward after
    # 1A, over 1A's node-to-node period (the osculating argument of latitude
    # differs from it by J2's short-period swing, 0.27 deg here)
    times, _ = _nodes(sim)
    a, c = times
    after = c - a[np.searchsorted(a, c) - 1]
    phase = 360.0 * after[c > a[0]] / np.diff(a).mean()
    assert phase.mean() == pytest.approx(real, abs=0.05)


def _revolution_mean_e(sim, k, revs=3, step=5.0):
    """Satellite ``k`` flown ``revs`` revolutions with the scenario's forces:
    its eccentricity vector averaged over the whole revolutions, node to node
    (the study's mean; the osculating e swings by ~1e-3 twice a revolution)."""
    from satflight.fit import fly
    s = sim.scenario.satellites[k]
    t = np.arange(0.0, (revs + 1.1) * REPEAT_PERIOD, step)
    rv = fly(sim.scenario.forces, sim.clock.jd(0.0), 0.0, sim.y[k], t,
             s.cd * s.area / s.mass, 0.0, states=True)
    z = rv[:, 0, 2]
    up = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))
    lo, hi = up[0] + 1, up[revs] + 1
    el = rv2coe(rv[lo:hi, 0], rv[lo:hi, 1])
    ex = np.trapezoid(el.e * np.cos(el.argp), t[lo:hi]) / (t[hi - 1] - t[lo])
    ey = np.trapezoid(el.e * np.sin(el.argp), t[lo:hi]) / (t[hi - 1] - t[lo])
    return math.hypot(ex, ey), math.degrees(math.atan2(ey, ex))


@pytest.mark.parametrize("k", [0, 1], ids=["S1A", "S1C"])
def test_the_default_flies_the_real_mean_eccentricity(sim, k):
    """Frozen e and argp as the precise orbits measure them (the year's mean of
    the revolution means), not the first-order J2 + J3 value (1.04e-3): each
    satellite's start is set for it, since the short-period swing depends on
    where in the orbit it starts."""
    mission = ("S1A", "S1C")[k]
    real = _daily(mission)
    e, argp = _revolution_mean_e(sim, k)
    assert e == pytest.approx(np.mean(real["e"][:335]), abs=3e-6)
    assert argp == pytest.approx(90.0, abs=0.5)


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


# --- against the truth: a year of ESA precise orbits ------------------------------------------
# docs/analysis/sentinel1/ holds what tools/sentinel1_study.py measured from the downloaded
# files, so these run without them.
STUDY = ROOT / "docs" / "analysis" / "sentinel1"


def _daily(mission):
    import csv
    with open(STUDY / f"daily_{mission}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
            for k in rows[0] if k != "date"}


@pytest.mark.parametrize("mission", ["S1A", "S1C"])
def test_the_real_orbit_flies_the_12_day_repeat(mission):
    revs = _daily(mission)["revs"][:335]          # July 2025 to May 2026, before 1C moved
    assert revs.sum() / len(revs) == pytest.approx(175 / 12, abs=0.005)


@pytest.mark.parametrize("mission", ["S1A", "S1C"])
def test_the_default_ltan_inclination_and_perigee_match_the_real_orbit(sim, mission):
    d = _daily(mission)
    k = [s.name for s in sim.sats].index(f"Sentinel-1{mission[-1]}")
    el = rv2coe(sim.y[k, :3], sim.y[k, 3:])
    ltan = local_time_of_node(el.raan, sim.clock.jd(0.0))
    assert abs(np.mean(d["ltan_h"]) - ltan) * 60 < 2.0           # minutes
    assert np.mean(d["i_deg"]) == pytest.approx(98.18, abs=0.005)
    assert np.mean(d["argp_deg"]) == pytest.approx(90.0, abs=1.0)


def test_the_real_pair_flies_176_degrees_apart_until_1c_moved_in_june_2026():
    """Not the 180 deg of the default: 1C trails 1A by 176.2 deg (measured twice:
    from node times and from the argument of latitude), and in June 2026 it was
    raised, drifted and lowered to trail by about 207 deg."""
    import csv
    with open(STUDY / "phasing.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    before = [float(r["mean_deg"]) for r in rows if r["date"] < "2026-06-01"]
    after = [float(r["mean_deg"]) for r in rows if r["date"] >= "2026-06-24"]
    assert min(before) > 175.8 and max(before) < 177.0
    assert after and all(206.0 < x < 208.0 for x in after)


# --- one set of ground tracks -----------------------------------------------------------------
# 175 tracks in 12 days lie 360/175 deg apart at the equator; each orbit's node lands
# 360 * 12/175 deg (12 tracks) farther west. Trailing by 176.3 deg alone would put 1C
# 0.12 of a track (28 km) off 1A's; ESA turns 1C's plane 0.25 deg west to close it.
GRID_DEG = 360.0 / 175


def _wrap(x, g):
    return (np.asarray(x) + g / 2) % g - g / 2


def _nodes(sim, hours=26.0, step=10.0):
    """Each satellite's ascending-node times (s) and longitudes (deg,
    Earth-fixed) over ``hours`` flown with the scenario's forces."""
    from satflight.fit import fly_batch
    from satflight.timeutil import gmst
    sats = sim.scenario.satellites
    t = np.arange(0.0, hours * 3600.0, step)
    jd0 = sim.clock.jd(0.0)
    Y = fly_batch(sim.scenario.forces, jd0, 0.0, sim.y, t,
                  [s.cd * s.area / s.mass for s in sats], [0.0] * len(sats))
    times, out = [], []
    for k in range(len(sats)):
        r = Y[k, :, :3]
        up = np.flatnonzero((r[:-1, 2] < 0) & (r[1:, 2] >= 0))
        f = -r[up, 2] / (r[up + 1, 2] - r[up, 2])
        tn = t[up] + f * step
        rn = r[up] + f[:, None] * (r[up + 1] - r[up])
        lon = np.degrees(np.arctan2(rn[:, 1], rn[:, 0]) - gmst(jd0 + tn / 86400.0))
        times.append(tn)
        out.append(lon)
    return times, out


def test_1c_flies_1a_s_ground_tracks(sim):
    """The pair's purpose: 1C passes over 1A's tracks six days later, so each
    of 1C's equator crossings lands on 1A's 175-track grid, as the precise
    orbits show (within 0.2 km, July 2025 - May 2026)."""
    _, (a, c) = _nodes(sim)
    off_km = _wrap(c - a[0], GRID_DEG) * 111.32
    assert len(off_km) >= 15
    assert np.abs(off_km).max() < 1.0


def test_the_default_turns_1c_s_plane_as_the_real_pair_does(sim):
    """1C's node runs about a minute of local time ahead of 1A's in the precise
    orbits (the 0.25 deg its 176.3 deg phasing needs); the default has the same gap."""
    real = (np.mean(_daily("S1A")["ltan_h"][:335])
            - np.mean(_daily("S1C")["ltan_h"][:335])) * 60
    jd = sim.clock.jd(0.0)
    ltan = [local_time_of_node(rv2coe(sim.y[k, :3], sim.y[k, 3:]).raan, jd)
            for k in range(2)]
    assert (ltan[0] - ltan[1]) * 60 == pytest.approx(real, abs=0.1)
