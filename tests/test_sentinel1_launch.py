"""The Sentinel-1 launch scenario: both satellites lifted from Vandenberg on
Falcon 9 a few days before the study starts and flown, with three burns each,
onto the states ESA's precise orbits give them at its first moment."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from satflight import phasing
from satflight.scenario import Scenario
from satflight.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_scenarios as ms  # noqa: E402

FILE = ROOT / "scenarios" / "sentinel1_launch.json"
T = (ms.S1_DATA_START - ms.S1_LAUNCH_EPOCH).total_seconds()


@pytest.fixture(scope="module")
def doc():
    return json.loads(FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arrived():
    """The scenario flown to a minute after the study's first moment."""
    sim = Simulation(Scenario.load(FILE))
    sim.advance(T - sim.t)
    at_t = {s.name: sim.y[i].copy() for i, s in enumerate(sim.sats)}
    sim.advance(3 * 3600.0)                      # two orbits more: nodes for the phasing
    return at_t, sim


def test_the_targets_are_the_precise_orbits_at_the_first_sample():
    from satflight import poeorb, replay
    for mission, (r, v) in ms.S1_DATA_START_STATE.items():
        try:
            y = replay.for_orbit({"type": "precise", "mission": mission},
                                 ms.S1_DATA_START).state(0.0)
        except poeorb.PoeorbError as exc:
            pytest.skip(f"precise orbits not downloaded: {exc}")
        assert np.abs(y[:3] - r).max() < 1e-6 and np.abs(y[3:] - v).max() < 1e-9


def test_falcon_9_lifts_both_from_vandenberg_southbound_a_day_apart(doc):
    launches = doc["launches"]
    assert [L["name"] for L in launches] == ["Sentinel-1A", "Sentinel-1C"]
    for k, L in enumerate(launches):
        assert L["site"] == "Vandenberg SLC-4E (USA)" and L["direction"] == "south"
        assert L["vehicle"]["name"] == "Falcon 9" and L["timing"] == "absolute"
        assert k * 86400.0 < L["t0"] < (k + 1) * 86400.0
        assert L["inclination"] == 98.18 and L["perigee_alt"] == L["apogee_alt"] == 600.0
    assert doc["epoch"].startswith("2025-06-25")


def test_three_burns_each_on_their_own_thrusters_after_separation(doc):
    t0 = {L["name"]: L["t0"] for L in doc["launches"]}
    for name in ("Sentinel-1A", "Sentinel-1C"):
        burns = [m for m in doc["maneuvers"] if m["sat"] == name]
        assert [m["label"] for m in burns] == ["phasing burn", "transfer to the slot",
                                              "arrival in the slot"]
        assert all(m["timing"] == "absolute" and m["frame"] == "ECI" and m["isp"] == 220.0
                   for m in burns)
        times = [m["t"] for m in burns]
        assert t0[name] + 3600.0 < times[0] < times[1] < times[2] == pytest.approx(T)
        total = sum(np.linalg.norm(m["dv"]) for m in burns) * 1e3
        # m/s: 600 km up to 693 and the phase; 1C, launched a day later, catches up
        # with an 80 m/s transfer: 134 m/s, some 130 kg of hydrazine at Isp 220
        assert 20.0 < total < 150.0


def test_each_satellite_arrives_on_its_precise_orbit_state(arrived):
    at_t, sim = arrived
    for name, mission in (("Sentinel-1A", "S1A"), ("Sentinel-1C", "S1C")):
        r, v = (np.array(x) for x in ms.S1_DATA_START_STATE[mission])
        y = at_t[name]
        assert np.linalg.norm(y[:3] - r) * 1e3 < 50.0          # m
        assert np.linalg.norm(y[3:] - v) * 1e6 < 50.0          # mm/s
        assert sim.sats[sim.index_of(name)].launch_report["outcome"] == "orbit"
    assert not [e.text for e in sim.events if e.kind == "warn"]


def test_once_arrived_the_pair_flies_the_study_s_first_day_phasing(arrived):
    _, sim = arrived
    with open(ROOT / "docs" / "analysis" / "sentinel1" / "phasing.csv", newline="",
              encoding="utf-8") as f:
        first_day = float(next(csv.DictReader(f))["mean_deg"])      # 2025-07-01
    (c,) = phasing.companions(sim, sim.index_of("Sentinel-1A"))
    assert sim.sats[c.index].name == "Sentinel-1C"
    assert -c.ahead_deg == pytest.approx(first_day, abs=0.05)
    assert abs(c.track_km) < 1.0
