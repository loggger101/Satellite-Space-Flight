"""Flying planned burns ahead of time (satflight.preview) matches the run."""
from datetime import datetime, timezone

import numpy as np
import pytest

from satflight.forces import ForceModel
from satflight.maneuvers import Maneuver
from satflight.planner import plan_hohmann, plan_target_orbit
from satflight.preview import fly, orbit_summary, snapshot
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation

EPOCH = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
GEO = 35786.0
GTO = {"type": "elements", "perigee_alt": 185.0, "apogee_alt": GEO, "i": 28.5, "raan": 40.0,
       "argp": 180.0, "nu": 120.0}


def _one(orbit, forces=None):
    sc = Scenario(name="t", epoch=EPOCH, forces=forces or ForceModel(j2=False),
                  satellites=[SatSpec("S", orbit, mass=3000, area=12)])
    return Simulation(sc)


@pytest.mark.parametrize("j2", [False, True])
def test_aimed_preview_is_what_the_run_flies(j2):
    sim = _one(GTO, ForceModel(j2=j2))
    burns, _ = plan_target_orbit(sim, "S", GEO, GEO, 0.0, "periapsis")
    plan = fly(sim, "S", burns, aim=True)
    assert len(plan.burns) == 2 and len(plan.arcs) == 1
    first, second = (b.place for b in plan.burns)
    assert first.endswith(("perigee", "perigee, node")) and second.endswith(("apogee",
                                                                              "apogee, node"))
    if not j2:                                    # J2 moves the apsides off the nodes
        assert first.endswith(", node") and second.endswith(", node")
    assert plan.burns[1].tilt == pytest.approx(28.5, abs=0.2)
    assert plan.total_dv == pytest.approx(1.838, abs=0.003)
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[-1].t - sim.t)
    assert np.linalg.norm(sim.y[0] - plan.final) < 1e-3   # km: 1 m
    hp, ha, inc, period = orbit_summary(plan.final)
    assert abs(hp - GEO) < 15 and abs(ha - GEO) < 15 and inc < 0.2
    assert period == pytest.approx(86164, rel=1e-3)


def test_two_body_preview_of_a_hohmann_closes_on_the_target():
    sim = _one({"type": "elements", "altitude": 400, "i": 51.6})
    burns, _ = plan_hohmann(sim, "S", 1200.0, "now", aim=False)
    plan = fly(sim, "S", burns)
    assert not plan.aimed and len(plan.arcs) == 1
    assert np.linalg.norm(plan.arcs[0][-1]) == pytest.approx(6378.137 + 1200, abs=0.01)
    radii = np.linalg.norm(plan.final_track, axis=1)
    assert np.ptp(radii) < 0.01                      # circularized: one revolution, flat


def test_relative_timings_and_finite_burns_are_flown():
    sim = _one({"type": "elements", "perigee_alt": 400, "apogee_alt": 900, "i": 30, "nu": 40})
    plan = fly(sim, "S", [Maneuver("S", "circularize", timing="apoapsis")], aim=True)
    assert plan.burns[0].place.endswith(("apogee", "apogee, node"))
    assert orbit_summary(plan.final)[1] - orbit_summary(plan.final)[0] < 0.01
    fin = Maneuver("S", "finite", timing="now", thrust=400.0, duration=600.0, dv=(1, 0, 0),
                   isp=320.0)
    plan = fly(sim, "S", [fin])
    assert "drawn as one kick" in plan.burns[0].note
    assert plan.burns[0].size * 1000 == pytest.approx(80.6, abs=0.5)    # rocket equation


def test_preview_leaves_the_run_alone_and_a_snapshot_can_fly_it():
    sim = _one(GTO)
    before = sim.y.copy(), sim.t, list(sim.maneuvers)
    snap = snapshot(sim)
    burns, _ = plan_target_orbit(snap, "S", GEO, GEO, 0.0, "apoapsis")
    fly(snap, "S", burns, aim=True)
    sim.advance(60.0)
    assert snap.t == before[1] and np.array_equal(snap.y, before[0])
    assert sim.maneuvers == before[2]


def test_a_reentering_final_orbit_stops_at_the_ground():
    sim = _one({"type": "elements", "altitude": 400, "i": 51.6})
    plan = fly(sim, "S", [Maneuver("S", "impulse", dv=(-0.3, 0, 0))])
    assert np.linalg.norm(plan.final_track[-1]) < 6378.137
    assert np.all(np.linalg.norm(plan.final_track[:-1], axis=1) >= 6378.137)
