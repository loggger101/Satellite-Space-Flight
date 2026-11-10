"""A propellant budget (audit A8, B3): burns stop at the dry mass, and say so."""

import json
import math

import numpy as np
import pytest

from satflight.forces import ForceModel
from satflight.maneuvers import G0, Maneuver
from satflight.scenario import ConstellationSpec, SatSpec, Scenario
from satflight.simulation import Simulation, dv_left

ORBIT = {"type": "elements", "altitude": 500, "i": 30}


def _sim(**props):
    return Simulation(Scenario(name="t", forces=ForceModel(j2=False),
                               satellites=[SatSpec("S", ORBIT, **props)]))


def test_an_impulse_stops_when_the_propellant_runs_out():
    sim = _sim(mass=500.0, dry_mass=400.0)
    room = dv_left(sim.sats[0])
    assert room == pytest.approx(300 * G0 * math.log(500 / 400))       # ~656 m/s
    v0 = sim.y[0, 3:].copy()
    sim.schedule(Maneuver("S", "impulse", dv=(1.0, 0.0, 0.0)))         # 1 km/s asked
    sim.advance(1.0)
    s = sim.sats[0]
    assert s.mass == pytest.approx(400.0)
    assert s.dv_used == pytest.approx(room)
    assert np.linalg.norm(sim.y[0, 3:] - v0) < room + 1e-3              # gravity over 1 s
    assert any("propellant ran out" in e.text for e in sim.events)
    sim.schedule(Maneuver("S", "impulse", dv=(0.01, 0.0, 0.0)))
    sim.advance(1.0)
    assert s.dv_used == pytest.approx(room) and s.mass == pytest.approx(400.0)


def test_a_finite_burn_ends_at_the_dry_mass():
    sim = _sim(mass=500.0, dry_mass=450.0)
    sim.schedule(Maneuver("S", "finite", dv=(1, 0, 0), thrust=1000.0, isp=300.0,
                          duration=3600.0))
    sim.advance(3700.0)
    s = sim.sats[0]
    assert s.mass == pytest.approx(450.0)
    assert s.dv_used == pytest.approx(300 * G0 * math.log(500 / 450), rel=1e-9)
    assert any("propellant for only" in e.text for e in sim.events)


def test_no_dry_mass_means_no_limit_and_files_stay_as_they_were():
    sim = _sim(mass=500.0)
    assert dv_left(sim.sats[0]) == math.inf
    sim.schedule(Maneuver("S", "impulse", dv=(3.0, 0.0, 0.0)))
    sim.advance(1.0)
    assert sim.sats[0].dv_used == pytest.approx(3.0)
    assert "dry_mass" not in SatSpec("S", ORBIT).to_dict()
    assert "dry_mass" not in ConstellationSpec().to_dict()
    assert SatSpec("S", ORBIT, dry_mass=100.0).to_dict()["dry_mass"] == 100.0


def test_dry_mass_round_trips_through_a_scenario_file(tmp_path):
    sc = Scenario(name="t", satellites=[SatSpec("S", ORBIT, mass=500.0, dry_mass=320.0)],
                  constellations=[ConstellationSpec(total=2, planes=1, dry_mass=200.0)])
    path = tmp_path / "s.json"
    sc.save(path)
    assert json.loads(path.read_text())["satellites"][0]["dry_mass"] == 320.0
    sim = Simulation(Scenario.load(path))
    assert [s.dry_mass for s in sim.sats] == [320.0, 200.0, 200.0]
    snap = sim.snapshot_scenario()
    assert snap.satellites[0].dry_mass == 320.0


@pytest.mark.parametrize("dry", [-1.0, 501.0, float("nan")])
def test_impossible_dry_masses_are_refused(dry):
    with pytest.raises(ValueError, match="dry mass"):
        SatSpec("S", ORBIT, mass=500.0, dry_mass=dry)
