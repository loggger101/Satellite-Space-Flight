"""Bad input is refused with a reason, never read as a default.

Each case here used to load and run quietly with something the file did not
ask for (a misspelt key read as its default, an unknown frame read as ECI, a
missing file read as an empty scenario)."""

import copy
import json
from pathlib import Path

import pytest

from satflight.launch import LaunchSpec, Vehicle
from satflight.maneuvers import Maneuver
from satflight.scenario import DEFAULT_EPOCH, Scenario, orbit_state
from satflight.simulation import Simulation
from satflight.timeutil import Clock

ROOT = Path(__file__).resolve().parents[1]
ISS_TLE = ("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
           "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")

BASE = {
    "name": "t",
    "satellites": [{"name": "A", "orbit": {"type": "elements", "altitude": 500, "i": 51.6}}],
}


def scenario(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    return d


def sat_orbit(**orbit):
    return scenario(satellites=[{"name": "A", "orbit": orbit}])


@pytest.mark.parametrize("d, words", [
    (scenario(satelites=[]), "'satelites'"),
    (scenario(satellites=[{"name": "A", "orbit": {"altitude": 500}, "mas": 3}]), "'mas'"),
    (scenario(satellites=[{"orbit": {"altitude": 500}}]), "'name'"),
    (sat_orbit(type="elements", altitude=500, inclination=51.6), "'inclination'"),
    (sat_orbit(type="geo", lon=10, i=3), "'i'"),
    (sat_orbit(type="state", r=[7000, 0, 0]), "'v'"),
    (sat_orbit(type="ellipse", a=7000), "orbit type"),
    (sat_orbit(type="elements", i=10), "size"),
    (sat_orbit(type="elements", altitude=500, a=7000), "size"),
    (sat_orbit(type="elements", perigee_alt=300, apogee_alt=900, e=0.1), "'e'"),
    (sat_orbit(type="elements", altitude=500, nu=10, M=20), "'nu' or 'M'"),
    (sat_orbit(type="elements", altitude=500, i="polar"), "'polar'"),
    (sat_orbit(type="tle", line1=ISS_TLE[0], line2=ISS_TLE[1][:-1] + "0"), "checksum"),
    (scenario(forces={"j2": True, "dragg": True}), "'dragg'"),
    (scenario(integrator={"methd": "rk4"}), "'methd'"),
    (scenario(integrator={"method": "rk45"}), "'rk45'"),
    (scenario(propagator="sgp4"), "'sgp4'"),
    (scenario(constellations=[{"pattern": "Delta"}]), "'Delta'"),
    (scenario(stations=[{"name": "X", "lat": 1, "lon": 2, "elevation": 5}]), "'elevation'"),
    (scenario(maneuvers=[{"sat": "A", "kind": "impulse", "frame": "vnb"}]), "'vnb'"),
    (scenario(maneuvers=[{"sat": "A", "kind": "boost"}]), "'boost'"),
    (scenario(maneuvers=[{"sat": "A", "timing": "perigee"}]), "'perigee'"),
    (scenario(maneuvers=[{"sat": "A", "kind": "finite", "thrust": 400, "duration": 60}]),
     "direction"),
    (scenario(maneuvers=[{"sat": "A", "kind": "impulse", "isp": 0}]), "Isp"),
    (scenario(maneuvers=[{"sat": "A", "timing": "delay", "delay": -60}]), "delay"),
    (scenario(maneuvers=[{"sat": "B", "kind": "circularize"}]), "'B'"),
    (scenario(maneuvers=[{"sat": "A", "kind": "match_velocity", "target": "B"}]), "'B'"),
    (scenario(launches=[{"name": "L", "vehicle": "Falcon 9 (approx.)", "directon": "south"}]),
     "'directon'"),
    (scenario(launches=[{"name": "L", "vehicle": "Falcon 9 (approx.)", "direction": "sout"}]),
     "'sout'"),
    (scenario(launches=[{"name": "L", "vehicle": "Falcon 10"}]), "Falcon 10"),
    (scenario(launches=[{"name": "L", "vehicle": {"stages": [{"thrust": 10, "ips_vac": 300}]}}]),
     "'ips_vac'"),
])
def test_bad_scenarios_are_refused_with_the_reason(d, words):
    with pytest.raises(ValueError) as err:
        Simulation(Scenario.from_dict(d))
    assert words in str(err.value)


def test_out_of_scope_forces_are_named_not_refused():
    sim = Simulation(Scenario.from_dict(scenario(forces={"j2": True, "sun": True})))
    assert any("Sun third-body" in e.text and e.kind == "warn" for e in sim.events)


def test_every_bundled_and_generated_scenario_passes_the_strict_reader():
    for path in sorted((ROOT / "scenarios").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sc = Scenario.from_dict(d)
        again = Scenario.from_dict(json.loads(json.dumps(sc.to_dict())))   # save -> load
        assert again.to_dict() == sc.to_dict(), path.name
        Simulation(sc)


def test_round_trips_of_specs_stay_strict():
    for spec in (LaunchSpec(), Maneuver("A")):
        assert type(spec).from_dict(spec.to_dict()) == spec
    v = Vehicle.from_dict(LaunchSpec().vehicle.to_dict())
    assert v == LaunchSpec().vehicle


def test_tle_orbit_with_a_good_checksum_still_loads():
    tle = {"type": "tle", "line1": ISS_TLE[0], "line2": ISS_TLE[1]}
    r, v = orbit_state(tle, Clock(DEFAULT_EPOCH))
    assert 6500 < (r @ r) ** 0.5 < 7000
