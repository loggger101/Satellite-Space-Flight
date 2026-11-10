"""Planners aim with the run's own force model (audit A1, A2)."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from satflight.elements import coe2rv
from satflight.forces import ForceModel
from satflight.maneuvers import lambert
from satflight.planner import plan_rendezvous, scan_rendezvous
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation

EPOCH = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def _el(**kw):
    return {"type": "elements", **kw}


def _pair(forces, propagator="cowell", inc=51.6, draan=0.0):
    sc = Scenario(name="t", epoch=EPOCH, forces=forces, propagator=propagator, satellites=[
        SatSpec("Station", _el(altitude=420, i=inc, raan=30 + draan, nu=40), mass=420000,
                area=1600),
        SatSpec("Chaser", _el(altitude=350, i=inc, raan=30, nu=34), mass=8000, area=20)])
    sim = Simulation(sc)
    sim.advance(300)
    return sim


def _fly(sim, burns):
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[1].t - sim.t)          # lands on the arrival, before match_velocity runs
    i, j = sim.index_of("Chaser"), sim.index_of("Station")
    return float(np.linalg.norm(sim.y[i, :3] - sim.y[j, :3]))


@pytest.mark.parametrize("side", [+1, -1])
def test_lambert_goes_the_chasers_way_round_in_a_polar_orbit(side):
    # A1: at i = 90 deg the old test used the sign of h_z, which is rounding noise
    r1, v1 = coe2rv(6878.137, 0.0, math.pi / 2, 0.0, 0.0, 0.0)
    r2, _ = coe2rv(6898.137, 0.0, math.pi / 2, math.radians(0.3 * side), 0.0, 1.6)
    h = np.cross(r1, v1)
    lv1, _ = lambert(r1, r2, 1500.0, normal=h)
    back, _ = lambert(r1, r2, 1500.0, normal=-h)
    assert np.linalg.norm(lv1 - v1) < 1.0                       # km/s
    assert np.linalg.norm(back - v1) > 10.0                     # the reverse arc
    assert np.dot(np.cross(r1, lv1), h) > 0                     # same sense of motion


@pytest.mark.parametrize("propagator", ["cowell", "j2mean", "kepler"])
@pytest.mark.parametrize("tof", [3600.0, 10800.0])
def test_rendezvous_with_j2_arrives_on_the_target(propagator, tof):
    # A2: two-body aiming missed by 11 km after 1 h and 90 km after 3 h with J2 on
    sim = _pair(ForceModel(j2=True), propagator)
    burns, _ = plan_rendezvous(sim, "Chaser", "Station", tof)
    assert _fly(sim, burns) < 0.01                               # km


def test_rendezvous_aims_through_drag_and_higher_harmonics():
    sim = _pair(ForceModel(j2=True, j3=True, j4=True, drag=True))
    burns, _ = plan_rendezvous(sim, "Chaser", "Station", 4140.0)
    assert _fly(sim, burns) < 0.02


def test_rendezvous_through_the_earth_is_refused():
    sim = _pair(ForceModel(j2=True))
    with pytest.raises(ValueError, match="below the surface"):
        plan_rendezvous(sim, "Chaser", "Station", 5400.0)


@pytest.mark.parametrize("side", [+1, -1])
def test_polar_rendezvous_costs_the_same_on_either_side(side):
    sim = _pair(ForceModel(j2=True), inc=90.0, draan=0.3 * side)
    burns, summary = plan_rendezvous(sim, "Chaser", "Station", 1500.0)
    assert np.linalg.norm(burns[0].dv) < 1.0, summary
    assert _fly(sim, burns) < 0.01


def test_scan_then_plan_meets_the_target():
    sim = _pair(ForceModel(j2=True))
    tof, total = scan_rendezvous(sim, "Chaser", "Station", 1200.0, 3 * 3600.0, 40)
    assert 0 < total < 1.0
    burns, _ = plan_rendezvous(sim, "Chaser", "Station", tof)
    assert _fly(sim, burns) < 0.01


def test_coast_states_leaves_the_run_alone():
    sim = _pair(ForceModel(j2=True, drag=True))
    t, y = sim.t, sim.y.copy()
    ahead = sim.predict("Chaser", sim.t + 3000.0)
    assert sim.t == t and np.array_equal(sim.y, y)
    sim.advance(3000.0)
    assert np.linalg.norm(ahead[:3] - sim.y[sim.index_of("Chaser"), :3]) < 1e-3


def _one(orbit, forces, propagator="cowell", **props):
    sc = Scenario(name="t", epoch=EPOCH, forces=forces, propagator=propagator,
                  satellites=[SatSpec("S", orbit, **props)])
    return Simulation(sc)


def _radius_range(sim, minutes=200):
    rs = []
    for _ in range(minutes):
        sim.advance(60.0)
        rs.append(float(np.linalg.norm(sim.y[0, :3])))
    return min(rs), max(rs)


@pytest.mark.parametrize("forces", [ForceModel(j2=False), ForceModel(j2=True)])
def test_hohmann_off_an_apsis_still_reaches_the_target(forces):
    # A6: from 300 x 1500 km at nu = 90 deg, "now" used to give 2090 x 2090 km
    from satflight.planner import plan_hohmann
    sim = _one(_el(perigee_alt=300, apogee_alt=1500, i=28, nu=90), forces)
    burns, summary = plan_hohmann(sim, "S", 2000.0, "now")
    assert "levels the flight path" in summary
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[1].t - sim.t + 1.0)
    lo, hi = _radius_range(sim, 120)
    target = 6378.137 + 2000.0
    assert abs(lo - target) < 4.0 and abs(hi - target) < 4.0, (lo, hi)  # J2 ripple only


@pytest.mark.parametrize("inc", [0.0, 51.6, 98.0])
def test_circularize_under_j2_leaves_only_the_j2_ripple(inc):
    # A7: two-body circular speed left 19 / 11 / 7 km swings at these inclinations
    from satflight.maneuvers import Maneuver
    sim = _one(_el(perigee_alt=690, apogee_alt=710, i=inc, argp=40, nu=10), ForceModel(j2=True))
    sim.schedule(Maneuver("S", "circularize", timing="periapsis"))
    sim.advance(sim.maneuvers[0].t - sim.t + 1.0)
    lo, hi = _radius_range(sim)
    ripple = 1.5 * 1.0826e-3 * 6378.137 ** 2 / 7068.0 * math.sin(math.radians(inc)) ** 2
    assert hi - lo < 0.01 + ripple, (hi - lo, ripple)


def test_hohmann_to_geo_arrives_at_the_target_radius_with_j2():
    # J2 at the 300 km start pulls a two-body-aimed apogee ~180 km short of GEO
    from satflight.planner import plan_hohmann
    sim = _one(_el(altitude=300, i=0.0), ForceModel(j2=True), mass=4000, area=15)
    burns, _ = plan_hohmann(sim, "S", 35786.0)
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[1].t - sim.t + 1.0)
    lo, hi = _radius_range(sim, 30)
    assert abs(lo - 42164.137) < 0.05 and abs(hi - 42164.137) < 0.05, (lo, hi)


def test_bielliptic_lands_every_leg():
    from satflight.planner import plan_bielliptic
    sim = _one(_el(altitude=400, i=30.0), ForceModel(j2=True))
    burns, _ = plan_bielliptic(sim, "S", 30000.0, 20000.0)
    for b in burns:
        sim.schedule(b)
    sim.advance(burns[1].t - sim.t)
    assert np.linalg.norm(sim.y[0, :3]) == pytest.approx(6378.137 + 30000.0, abs=0.05)
    sim.advance(burns[2].t - sim.t + 1.0)
    lo, hi = _radius_range(sim, 30)
    assert abs(lo - 26378.137) < 0.05 and abs(hi - 26378.137) < 0.05, (lo, hi)


@pytest.mark.parametrize("propagator", ["cowell", "j2mean"])
def test_node_timing_lands_on_the_equator(propagator):
    from satflight.maneuvers import Maneuver
    sim = _one(_el(perigee_alt=400, apogee_alt=900, i=63.0, argp=50, nu=20),
               ForceModel(j2=True, j3=True), propagator)
    for timing, up in (("ascending_node", True), ("descending_node", False)):
        t = sim.resolve_time(Maneuver("S", timing=timing))
        y = sim.predict("S", t)
        assert abs(y[2]) < 1e-3 and (y[5] > 0) == up                 # within a metre


def test_apsis_timing_finds_the_real_extreme_of_a_near_circular_orbit():
    # under J2 a circular orbit's osculating perigee can sit anywhere
    from satflight.maneuvers import Maneuver
    sim = _one(_el(altitude=420, i=51.6, nu=40), ForceModel(j2=True))
    sim.advance(100.0)
    for timing, pick in (("periapsis", np.argmin), ("apoapsis", np.argmax)):
        t = sim.resolve_time(Maneuver("S", timing=timing))
        times = sim.t + np.arange(1.0, 5600.0, 1.0)
        rad = np.linalg.norm(sim.coast_track(0, times)[:, :3], axis=1)
        assert abs(t - times[pick(rad)]) < 2.0


@pytest.mark.parametrize("delay, why", [(-1.0, "between 0 s and 100 years"),
                                        (1e300, "between 0 s and 100 years"),
                                        (30 * 86400.0, "days ahead")])
@pytest.mark.parametrize("aim", [True, False])
def test_a_bad_or_far_off_delay_is_refused_before_anything_is_flown(delay, why, aim):
    # planners coast the run's forces to the burn: 1e300 s used to hang, -1 s
    # was planned in the past
    from satflight.planner import plan_bielliptic, plan_hohmann
    sim = _one(_el(altitude=420, i=51.6), ForceModel(j2=True))
    with pytest.raises(ValueError, match=why):
        plan_hohmann(sim, "S", 1000.0, "delay", delay, aim=aim)
    with pytest.raises(ValueError, match=why):
        plan_bielliptic(sim, "S", 5000.0, 1000.0, "delay", delay, aim=aim)


def test_a_rendezvous_past_the_planning_horizon_is_refused():
    sim = _pair(ForceModel(j2=True))
    with pytest.raises(ValueError, match="arrival is 7.0 days ahead"):
        plan_rendezvous(sim, "Chaser", "Station", 3600.0, "delay", 6.99 * 86400)
    with pytest.raises(ValueError, match="time of flight"):
        plan_rendezvous(sim, "Chaser", "Station", 0.0)
    with pytest.raises(ValueError, match="latest arrival searched"):
        scan_rendezvous(sim, "Chaser", "Station", timing="delay", delay=6.99 * 86400)


def test_a_far_off_plane_change_is_still_previewed():
    # single burns are worked out when they fire: no planning horizon
    from satflight.planner import plane_change_summary
    sim = _one(_el(altitude=420, i=51.6), ForceModel(j2=True))
    assert plane_change_summary(sim, "S", 5.0, "delay", 30 * 86400.0).startswith("dV ~")
