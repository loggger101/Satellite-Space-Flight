"""The "Change orbit to..." planner: perigee, apogee and inclination at once."""
import math
import re
from datetime import datetime, timezone

import numpy as np
import pytest

from satflight.constants import R_EARTH
from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.planner import _split_cost, best_split, plan_target_orbit, plane_through
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation

EPOCH = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
GEO = 35786.0
GTO = {"type": "elements", "perigee_alt": 185.0, "apogee_alt": GEO, "i": 28.5, "raan": 40.0,
       "argp": 180.0, "nu": 120.0}


def _one(orbit, forces=None, propagator="cowell"):
    sc = Scenario(name="t", epoch=EPOCH, forces=forces or ForceModel(j2=False),
                  propagator=propagator,
                  satellites=[SatSpec("S", orbit, mass=3000, area=12)])
    return Simulation(sc)


def _fly(sim, burns, after=600.0):
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[-1].t - sim.t + after)
    el = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    return el.a * (1 - el.e) - R_EARTH, el.a * (1 + el.e) - R_EARTH, math.degrees(el.i)


def test_parking_orbit_to_geo_split_matches_the_textbook():
    # the classic result from a 185 km, 28.5 deg parking orbit: ~2.2 deg of the
    # turn rides on the big injection burn, the rest at apogee, 4.27 km/s in all
    # against 4.30 km/s with the whole turn at apogee
    sim = _one({"type": "elements", "altitude": 185.0, "i": 28.5, "raan": 40.0, "nu": 0.0})
    r, v = sim.y[0, :3], sim.y[0, 3:]
    f, total = best_split(r, v, R_EARTH + GEO, R_EARTH + GEO, 28.5, 0.0)
    assert 2.0 < f * 28.5 < 2.4, f * 28.5
    assert total == pytest.approx(4.272, abs=0.002)
    assert _split_cost(r, v, R_EARTH + GEO, R_EARTH + GEO, 28.5, 0.0, 0.0)[0] > total + 0.02
    _, summary = plan_target_orbit(sim, "S", GEO, GEO, 0.0, "now", aim=False)
    assert "two-body estimate" in summary and "burn 2 takes 92%" in summary


def test_from_gto_the_whole_turn_belongs_at_apogee():
    # burn 1 at perigee changes no speed, so any tilt there is paid in full at
    # 10.2 km/s; the textbook all-at-apogee GTO -> GEO burn is 1.84 km/s
    sim = _one({**GTO, "nu": 0.0})
    r, v = sim.y[0, :3], sim.y[0, 3:]
    f, total = best_split(r, v, R_EARTH + GEO, R_EARTH + GEO, 28.5, 0.0)
    assert f == 0.0 and total == pytest.approx(1.837, abs=0.002)


@pytest.mark.parametrize("forces", [ForceModel(j2=False), ForceModel(j2=True)])
def test_gto_to_geo_lands_on_geo(forces):
    sim = _one(GTO, forces)
    burns, summary = plan_target_orbit(sim, "S", GEO, GEO, 0.0, "periapsis")
    assert len(burns) == 2, summary
    hp, ha, inc = _fly(sim, burns)
    # osculating elements under J2 swing a few km about the target; two-body exactly
    tol = 0.05 if not forces.j2 else 15.0
    assert abs(hp - GEO) < tol and abs(ha - GEO) < tol, (hp, ha)
    if forces.j2:            # the J2 perigee is off the equator: GEO's plane misses it a bit
        m = re.search(r"it reaches ([0-9.]+) deg", summary)
        assert m, summary
        assert inc == pytest.approx(float(m.group(1)), abs=0.002)
    else:
        assert inc < 1e-6, inc


def test_burning_at_a_target_apsis_takes_one_burn():
    sim = _one(GTO)
    burns, summary = plan_target_orbit(sim, "S", GEO, GEO, 0.0, "apoapsis")
    assert len(burns) == 1 and "one burn" in summary
    hp, ha, inc = _fly(sim, burns)
    assert abs(hp - GEO) < 0.05 and abs(ha - GEO) < 0.05 and inc < 1e-6


@pytest.mark.parametrize("target", [(500.0, 2000.0, 40.0), (2000.0, 20000.0, 63.4),
                                    (300.0, 300.0, 51.6)])
def test_raising_lowering_and_tilting_land_on_target(target):
    sim = _one({"type": "elements", "perigee_alt": 700, "apogee_alt": 900, "i": 51.6,
                "raan": 10, "argp": 0.0, "nu": 300.0}, ForceModel(j2=False))
    burns, _ = plan_target_orbit(sim, "S", *target, timing="ascending_node")
    hp, ha, inc = _fly(sim, burns)
    assert abs(hp - target[0]) < 0.05 and abs(ha - target[1]) < 0.05, (hp, ha)
    assert abs(inc - target[2]) < 1e-6


def test_impossible_targets_say_why():
    sim = _one({"type": "elements", "altitude": 500, "i": 51.6, "nu": 90.0})
    with pytest.raises(ValueError, match="swap them"):
        plan_target_orbit(sim, "S", 900.0, 800.0, 51.6, "now")
    with pytest.raises(ValueError, match="cannot reach"):
        plan_target_orbit(sim, "S", 500.0, 500.0, 10.0, "now")   # at 51.6 deg latitude
    with pytest.raises(ValueError, match="inclination"):
        plan_target_orbit(sim, "S", 500.0, 900.0, 200.0, "now")


def test_plane_through_keeps_the_direction_of_travel():
    r = np.array([7000.0, 0.0, 0.0])
    for inc in (0.0, 30.0, 90.0, 150.0):
        h = plane_through(r, np.array([0.0, -0.5, 1.0]), inc)
        assert abs(np.dot(h, r)) < 1e-9
        assert math.degrees(math.acos(h[2])) == pytest.approx(inc)
