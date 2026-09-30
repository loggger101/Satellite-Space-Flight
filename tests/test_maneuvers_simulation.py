import math
from pathlib import Path

import numpy as np
import pytest

from satflight.analysis import j2_secular_rates
from satflight.constants import R_EARTH, R_GEO
from satflight.elements import coe2rv, kepler_propagate, rv2coe
from satflight.forces import ForceModel
from satflight.maneuvers import Maneuver, bielliptic, hohmann, lambert, sun_synchronous_inclination
from satflight.scenario import ConstellationSpec, SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.tle import checksum, parse_tle

ROOT = Path(__file__).resolve().parents[1]


def test_hohmann_vallado_example_6_1():
    # Vallado Example 6-1: 191.34411 km -> 35781.34857 km altitude
    dv1, dv2, tof = hohmann(R_EARTH + 191.34411, R_EARTH + 35781.34857)
    assert dv1 == pytest.approx(2.457038, abs=1e-5)
    assert dv2 == pytest.approx(1.478187, abs=1e-5)
    assert tof / 3600 == pytest.approx(5.2567, abs=1e-3)


def test_bielliptic_beats_hohmann_for_large_ratio():
    r1, r2 = 7000.0, 7000.0 * 20
    h = sum(abs(x) for x in hohmann(r1, r2)[:2])
    b = sum(abs(x) for x in bielliptic(r1, r2 * 2, r2)[:3])
    assert b < h


def test_lambert_recovers_kepler_arc():
    r1, v1 = coe2rv(9000.0, 0.2, 0.6, 0.4, 0.3, 0.2)
    for tof in (1200.0, 3000.0, 5000.0):
        r2, v2 = kepler_propagate(r1, v1, tof)
        lv1, lv2 = lambert(r1, r2, tof)
        assert np.allclose(lv1, v1, atol=1e-7)
        assert np.allclose(lv2, v2, atol=1e-7)


def test_sun_synchronous_inclination_700km():
    assert math.degrees(sun_synchronous_inclination(R_EARTH + 700)) == pytest.approx(98.19, abs=0.02)


def _scenario(sats, **kw):
    sc = Scenario(name="t", satellites=sats, forces=kw.pop("forces", ForceModel(j2=False)), **kw)
    return sc


def test_simulated_hohmann_reaches_geo():
    sc = _scenario([SatSpec("sat", {"type": "elements", "altitude": 300, "i": 0})],
                   warp=1.0)
    sim = Simulation(sc)
    r1 = np.linalg.norm(sim.y[0, :3])
    dv1, dv2, tof = hohmann(r1, R_GEO)
    sim.schedule(Maneuver("sat", "impulse", dv=(dv1, 0, 0)))
    sim.schedule(Maneuver("sat", "circularize", timing="delay", delay=tof))
    sim.advance(tof + 3600)
    el = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    assert el.a == pytest.approx(R_GEO, rel=1e-6)
    assert el.e < 1e-6
    assert sim.sats[0].dv_used == pytest.approx(dv1 + dv2, rel=1e-6)
    assert sim.sats[0].mass < 500.0


def test_plane_change_at_node():
    sc = _scenario([SatSpec("sat", {"type": "elements", "altitude": 500, "i": 28.5,
                                    "raan": 40, "nu": 90})])
    sim = Simulation(sc)
    sim.schedule(Maneuver("sat", "plane_change", timing="descending_node", delta_i=-28.5))
    sim.advance(7000)
    assert math.degrees(rv2coe(sim.y[0, :3], sim.y[0, 3:]).i) == pytest.approx(0.0, abs=1e-6)


def test_finite_burn_matches_rocket_equation():
    sc = _scenario([SatSpec("sat", {"type": "elements", "altitude": 500}, mass=1000)])
    sim = Simulation(sc)
    sim.schedule(Maneuver("sat", "finite", thrust=500.0, isp=320.0, duration=200.0,
                          dv=(1, 0, 0)))
    a0 = rv2coe(sim.y[0, :3], sim.y[0, 3:]).a
    sim.advance(400)
    sat = sim.sats[0]
    mdot = 500.0 / (320 * 9.80665)
    assert sat.mass == pytest.approx(1000 - 200 * mdot, rel=1e-9)
    assert sat.dv_used * 1000 == pytest.approx(320 * 9.80665 * math.log(1000 / sat.mass), rel=1e-9)
    assert rv2coe(sim.y[0, :3], sim.y[0, 3:]).a > a0 + 100


def test_j2_nodal_regression_matches_theory():
    sc = _scenario([SatSpec("sso", {"type": "elements", "altitude": 700, "i": "sso"})],
                   forces=ForceModel(j2=True))
    sim = Simulation(sc)
    el0 = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    days = 3.0
    sim.advance(days * 86400)
    el1 = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    drift = math.degrees(math.remainder(el1.raan - el0.raan, 2 * math.pi)) / days
    expected = math.degrees(j2_secular_rates(el0.a, el0.e, el0.i)[0]) * 86400
    assert drift == pytest.approx(0.9856, abs=0.02)      # sun-synchronous
    assert drift == pytest.approx(expected, rel=0.02)


def test_drag_decays_low_orbit_and_detects_reentry():
    sc = _scenario([SatSpec("low", {"type": "elements", "altitude": 150}, mass=50, area=4)],
                   forces=ForceModel(j2=True, drag=True))
    sim = Simulation(sc)
    sim.advance(3 * 86400)
    assert sim.sats[0].status in ("re-entered", "impacted")


def test_kepler_and_cowell_agree_for_two_body():
    sats = [SatSpec("a", {"type": "elements", "altitude": 600, "i": 40, "e": 0.01})]
    a = Simulation(_scenario(sats))
    b = Simulation(_scenario(sats, propagator="kepler"))
    a.advance(20000)
    b.advance(20000)
    assert np.linalg.norm(a.y[0, :3] - b.y[0, :3]) < 0.01


def test_walker_and_json_round_trip(tmp_path):
    sc = Scenario(name="w", constellations=[ConstellationSpec(total=24, planes=6, phasing=1)],
                  satellites=[SatSpec("x", {"type": "geo", "lon": 10}, count=3)])
    path = tmp_path / "s.json"
    sc.save(path)
    sim = Simulation(Scenario.load(path))
    assert sim.n == 27
    rs = np.linalg.norm(sim.y[3:, :3], axis=1)   # satellites first, then Walker
    assert np.allclose(rs, R_EARTH + 550)


def test_snapshot_reproduces_state():
    sim = Simulation(_scenario([SatSpec("a", {"type": "elements", "altitude": 800, "i": 70})]))
    sim.advance(1000)
    sim2 = Simulation(sim.snapshot_scenario())
    assert np.allclose(sim2.y, sim.y)
    assert sim2.clock.jd0 == pytest.approx(sim.clock.jd(sim.t))


@pytest.mark.parametrize("path", sorted((ROOT / "scenarios").glob("*.json")),
                         ids=lambda p: p.name)
def test_bundled_scenarios_run(path):
    sim = Simulation(Scenario.load(path))
    sim.advance(600)
    assert np.all(np.isfinite(sim.y))


ISS_TLE = ("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
           "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")


def test_tle_parsing():
    assert checksum(ISS_TLE[0]) == 7 and checksum(ISS_TLE[1]) == 7
    tle = parse_tle("\n".join(ISS_TLE), "ISS")
    assert tle.checksum_ok
    assert math.degrees(tle.inclination) == pytest.approx(51.6416)
    assert tle.eccentricity == pytest.approx(0.0006703)
    assert tle.bstar == pytest.approx(-0.11606e-4)
    assert tle.epoch.year == 2008 and tle.epoch.timetuple().tm_yday == 264
    r, v = tle.state_at(tle.epoch)
    assert 6600 < np.linalg.norm(r) < 6800


def test_rendezvous_scenario_arrives_at_target():
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "rendezvous.json"))
    t_match = max(m.t for m in sim.maneuvers)
    sim.advance(t_match + 1.0)
    i, j = sim.index_of("Chaser"), sim.index_of("Station")
    assert np.linalg.norm(sim.y[i, :3] - sim.y[j, :3]) < 0.05          # km
    assert np.linalg.norm(sim.y[i, 3:] - sim.y[j, 3:]) < 1e-3          # km/s


def test_launch_from_surface_is_not_flagged_as_reentry():
    sc = _scenario([SatSpec("hop", {"type": "surface", "lat": 0, "lon": 0, "alt": 0.0,
                                    "speed": 3.0, "azimuth": 90, "fpa": 60})])
    sim = Simulation(sc)
    sim.advance(60)
    assert sim.sats[0].status == "active"
    sim.advance(3600)
    assert sim.sats[0].status in ("re-entered", "impacted")


def test_sun_and_moon_forces_are_out_of_scope(tmp_path):
    """Only the Earth acts on satellites: a scenario asking for Sun/Moon
    gravity or SRP still loads, runs Earth-only and says what it ignored."""
    import json
    assert set(ForceModel.TERMS) == {"j2", "j3", "j4", "drag"}
    assert not {"sun", "moon", "srp"} & {f for f in vars(ForceModel())}
    d = Scenario(name="old", satellites=[]).to_dict()
    d["forces"].update(sun=True, moon=True, srp=False)
    d["satellites"] = [{"name": "HEO", "cr": 1.8,
                        "orbit": {"type": "elements", "perigee_alt": 1000, "apogee_alt": 120000, "i": 60}}]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    sc = Scenario.load(path)
    assert sc.out_of_scope == ["Sun third-body gravity", "Moon third-body gravity"]
    assert "out_of_scope" not in sc.to_dict() and "sun" not in sc.to_dict()["forces"]
    sim = Simulation(sc)
    assert any(e.kind == "warn" and "Moon third-body" in e.text for e in sim.events)
    ref = Simulation(Scenario(name="earth only", satellites=sc.satellites, forces=ForceModel()))
    sim.advance(6 * 3600)
    ref.advance(6 * 3600)
    assert np.array_equal(sim.y, ref.y)
