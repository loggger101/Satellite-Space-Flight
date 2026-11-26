"""satflight.phasing: how satellites sharing an orbit sit along it and over the
ground, held to the default Sentinel-1 pair, the precise orbits and node
crossings measured on flown orbits."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from satflight import phasing
from satflight.scenario import Scenario
from satflight.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "docs" / "analysis" / "sentinel1"
DEFAULT = json.loads((ROOT / "scenarios" / "default.json").read_text(encoding="utf-8"))


def _sim(d=DEFAULT, hours=4.0):
    """The scenario ``d`` flown ``hours`` (two orbits: the view reads nodes)."""
    s = Simulation(Scenario.from_dict(copy.deepcopy(d)))
    s.advance(hours * 3600.0)
    return s


@pytest.fixture(scope="module")
def sim():
    return _sim()


def test_nothing_is_known_before_the_equator_crossings():
    s = _sim(hours=0.0)
    assert phasing.repeat_cycle(s, 0) is None
    (c,) = phasing.companions(s, 0)
    assert c.ahead_deg is None and c.track_km is None
    rows = dict(phasing.rows(s, 0))
    assert rows["Repeat cycle"] == "after two equator crossings"
    assert rows["Phase angle"] == "at its next equator crossing"


def _real(name):
    with open(STUDY / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flown_track_offset_km(sim, hours=26.0, step=10.0):
    """1C's mean ascending-node distance (km, + east) from 1A's 175-track grid,
    from the node crossings of a day flown with the scenario's forces."""
    from satflight.fit import fly_batch
    from satflight.timeutil import gmst
    sats = sim.scenario.satellites
    t = np.arange(0.0, hours * 3600.0, step)
    jd0 = sim.clock.jd(0.0)
    Y = fly_batch(sim.scenario.forces, jd0, 0.0, sim.y, t,
                  [s.cd * s.area / s.mass for s in sats], [0.0] * len(sats))
    lon = []
    for k in range(2):
        r = Y[k, :, :3]
        up = np.flatnonzero((r[:-1, 2] < 0) & (r[1:, 2] >= 0))
        f = -r[up, 2] / (r[up + 1, 2] - r[up, 2])
        rn = r[up] + f[:, None] * (r[up + 1] - r[up])
        lon.append(np.degrees(np.arctan2(rn[:, 1], rn[:, 0])
                              - gmst(jd0 + (t[up] + f * step) / 86400.0)))
    g = 360.0 / 175
    return float(np.mean((lon[1] - lon[0][0] + g / 2) % g - g / 2)) * np.pi / 180 * 6378.137


def test_the_sentinel_1_orbit_repeats_in_175_orbits_and_12_days(sim):
    rep = phasing.repeat_cycle(sim, 0)
    assert (rep.orbits, rep.days) == (175, 12)
    assert rep.spacing_km == pytest.approx(229.0, abs=0.5)
    assert rep.closure_km < 5.0


def test_an_orbit_that_does_not_repeat_soon_has_no_cycle():
    d = copy.deepcopy(DEFAULT)
    d["satellites"] = d["satellites"][:1]
    d["satellites"][0]["orbit"].update(altitude=700.0, i=40.0)   # 14.55 orbits a day
    assert phasing.repeat_cycle(_sim(d), 0) is None


def test_1c_trails_1a_by_the_phasing_the_precise_orbits_show(sim):
    real = np.mean([float(r["mean_deg"]) for r in _real("phasing.csv")])
    (c,) = phasing.companions(sim, 0)
    assert c.index == 1
    assert -c.ahead_deg == pytest.approx(real, abs=0.1)
    (a,) = phasing.companions(sim, 1)
    assert a.index == 0 and a.ahead_deg == pytest.approx(real, abs=0.1)


def test_1c_s_plane_is_turned_west_by_the_real_pair_s_ltan_gap(sim):
    ltan = {m: np.mean([float(r["ltan_h"]) for r in _real(f"daily_{m}.csv")])
            for m in ("S1A", "S1C")}
    real_deg = (ltan["S1C"] - ltan["S1A"]) * 15.0          # 1 h of local time = 15 deg
    (c,) = phasing.companions(sim, 0)
    assert c.plane_deg == pytest.approx(real_deg, abs=0.03)
    # each side takes the precession between the two crossings at its own rate
    assert phasing.companions(sim, 1)[0].plane_deg == pytest.approx(-c.plane_deg, abs=1e-6)


def test_the_default_puts_1c_on_1a_s_tracks_as_flown(sim):
    (c,) = phasing.companions(sim, 0)
    flown = _flown_track_offset_km(sim)
    assert abs(flown) < 1.0
    assert c.track_km == pytest.approx(flown, abs=1.5)


def test_without_the_plane_offset_1c_flies_26_km_off_1a_s_tracks():
    """The old default (both planes on 18:00): the view sees what flying shows."""
    d = copy.deepcopy(DEFAULT)
    d["satellites"][1]["orbit"]["raan"] = d["satellites"][0]["orbit"]["raan"]
    s = _sim(d)
    flown = _flown_track_offset_km(s)
    assert 20.0 < abs(flown) < 32.0
    (c,) = phasing.companions(s, 0)
    assert c.track_km == pytest.approx(flown, abs=1.5)
    assert phasing.companions(s, 1)[0].track_km == pytest.approx(-flown, abs=1.5)


def test_satellites_on_other_orbits_are_not_companions():
    d = copy.deepcopy(DEFAULT)
    other = copy.deepcopy(d["satellites"][1])
    other["name"] = "Other plane"
    other["orbit"]["raan"] = (other["orbit"]["raan"] + 30.0) % 360.0
    higher = copy.deepcopy(d["satellites"][1])
    higher["name"] = "Higher"
    higher["orbit"]["altitude"] = 800.0
    d["satellites"] += [other, higher]
    s = _sim(d)
    assert [c.index for c in phasing.companions(s, 0)] == [1]


def test_rows_say_it_in_words(sim):
    rows = dict(phasing.rows(sim, 0))
    assert rows["Repeat cycle"].startswith("175 orbits / 12 d")
    assert rows["Companion"] == "Sentinel-1C"
    assert rows["Phase angle"].endswith("deg behind")
    assert rows["Plane offset"].startswith("-0.2") and "W" in rows["Plane offset"]
    assert rows["Track offset"].endswith("of these tracks")
    rows1 = dict(phasing.rows(sim, 1))
    assert rows1["Companion"] == "Sentinel-1A" and rows1["Phase angle"].endswith("deg ahead")


def test_a_lone_satellite_has_no_rows():
    d = copy.deepcopy(DEFAULT)
    d["satellites"] = d["satellites"][:1]
    assert phasing.rows(_sim(d), 0) == []


def test_every_phasing_row_and_the_section_are_explained(sim):
    from satflight.ui import glossary
    labels = {label for label, _ in phasing.rows(sim, 0) + phasing.rows(_sim(hours=0.0), 0)}
    assert labels == {"Repeat cycle", "Companion", "Phase angle", "Plane offset", "Track offset"}
    assert all(glossary.row(label) for label in labels)
    assert glossary.section("PHASING")


def test_an_equatorial_orbit_has_no_node_and_no_companions():
    d = json.loads((ROOT / "scenarios" / "geo_drift.json").read_text(encoding="utf-8"))
    s = _sim(d, hours=1.0)
    assert s.n > 1 and phasing.companions(s, 0) == [] and phasing.rows(s, 0) == []
