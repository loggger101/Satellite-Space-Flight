"""Replay of the Sentinel-1 precise orbits in the simulator (satflight.replay):
a satellite that follows the recorded truth instead of being integrated, a
twin that starts from the same recorded state and is flown by the force model,
and the live radial / along-track / cross-track gap between them.

The data-dependent parts run on synthetic POEORB files and a synthetic IERS
table written into a temporary data folder, so they run everywhere; the last
test uses the real files when they are on this machine."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from test_eop import _line
from test_poeorb import _states, make_eof

from satflight import replay
from satflight.elements import kepler_propagate
from satflight.scenario import SatSpec, Scenario, check_orbit
from satflight.simulation import ACTIVE, Simulation

UTC = timezone.utc
EPOCH = datetime(2025, 7, 1, 6, 0, tzinfo=UTC)
TRUTH = {"type": "replay", "mission": "S1A", "until": "2025-07-02"}
TWIN = {"type": "precise", "mission": "S1A"}


# --- interpolation ---------------------------------------------------------------------------

def _kepler_ephemeris(step=10.0, hours=2.0):
    r0, v0 = np.array([7071.0, 0.0, 0.0]), np.array([0.0, 1.06, 7.43])
    t = np.arange(0.0, hours * 3600.0 + 1e-9, step)
    r, v = kepler_propagate(np.repeat(r0[None], len(t), 0), np.repeat(v0[None], len(t), 0), t)
    return replay.Ephemeris("S1A", t, r, v), (r0, v0)


def test_hermite_interpolation_between_10_s_samples_is_millimetre_exact():
    eph, (r0, v0) = _kepler_ephemeris()
    tq = np.arange(5.0, 7000.0, 10.0)                     # every midpoint: the worst place
    rq, vq = kepler_propagate(np.repeat(r0[None], len(tq), 0), np.repeat(v0[None], len(tq), 0),
                              tq)
    got = np.array([eph.state(t) for t in tq])
    assert np.abs(got[:, :3] - rq).max() < 1e-6            # km: under 1 mm
    assert np.abs(got[:, 3:] - vq).max() < 1e-6            # km/s: under 1 mm/s
    assert np.array_equal(eph.state(20.0)[:3], eph.r[2])   # on a sample it is the sample


def test_a_time_outside_the_recording_is_refused():
    eph, _ = _kepler_ephemeris()
    for t in (-1.0, eph.end + 1.0):
        with pytest.raises(replay.ReplayError, match="outside"):
            eph.state(t)


# --- the scenario's orbit types ----------------------------------------------------------------

@pytest.mark.parametrize("orbit, msg", [
    ({"type": "replay", "mission": "S1A"}, "until"),
    ({"type": "replay", "mission": "S2A", "until": "2025-07-02"}, "S2A"),
    ({"type": "replay", "mission": "S1A", "until": "2nd July"}, "until"),
    ({"type": "precise"}, "mission"),
    ({"type": "precise", "mission": "S1A", "until": "2025-07-02"}, "until"),
])
def test_replay_and_precise_orbits_are_checked_strictly(orbit, msg):
    with pytest.raises(ValueError, match=msg):
        check_orbit(orbit)


def test_a_replayed_satellite_cannot_be_copied():
    d = _scenario().to_dict()
    d["satellites"][0]["count"] = 2
    with pytest.raises(ValueError, match="replay"):
        Scenario.from_dict(d)


# --- with (synthetic) data ----------------------------------------------------------------------

@pytest.fixture
def data(tmp_path, monkeypatch):
    """Two days of synthetic S1A files and a zero-pole IERS table."""
    monkeypatch.setenv("SATFLIGHT_DATA", str(tmp_path))
    for d in (date(2025, 7, 1), date(2025, 7, 2)):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", d)
    (tmp_path / "eop").mkdir()
    (tmp_path / "eop" / "finals2000A.all").write_text(
        "\n".join(_line(60856.0 + k, 0.0, 0.0, 0.0434) for k in range(6)) + "\n",
        encoding="ascii")
    replay.clear_cache()
    yield tmp_path
    replay.clear_cache()


def _scenario(epoch=EPOCH):
    from satflight.forces import ForceModel
    return Scenario(name="replay test", epoch=epoch, forces=ForceModel(j2=False, drag=False),
                    satellites=[SatSpec("S1A truth", dict(TRUTH)),
                                SatSpec("S1A twin", dict(TWIN), mass=2164.0, area=10.0)])


def _truth_eci(when):
    _, _, r, v = _states([when])
    return r[0], v[0]


def test_the_truth_follows_the_recording_and_the_twin_starts_on_it(data):
    sim = Simulation(_scenario())
    i, j = sim.index_of("S1A truth"), sim.index_of("S1A twin")
    r0, v0 = _truth_eci(EPOCH)
    assert np.abs(sim.y[i, :3] - r0).max() < 1e-6 and np.abs(sim.y[j, :3] - r0).max() < 1e-6
    sim.advance(5000.0)
    r1, v1 = _truth_eci(EPOCH + timedelta(seconds=5000))
    assert np.abs(sim.y[i, :3] - r1).max() < 1e-6 and np.abs(sim.y[i, 3:] - v1).max() < 1e-6
    # the synthetic truth is a two-body orbit, so the two-body twin stays on it
    gap = sim.gap_to_truth(j)
    assert gap.shape == (3,) and np.linalg.norm(gap) < 1e-3          # km
    assert sim.gap_to_truth(i) is None                                 # the truth has no twin


def test_the_gap_is_radial_along_track_and_cross_track_in_the_truths_frame(data):
    sim = Simulation(_scenario())
    i, j = sim.index_of("S1A truth"), sim.index_of("S1A twin")
    r, v = sim.y[i, :3], sim.y[i, 3:]
    h = np.cross(r, v)
    sim.y[j, :3] = r + 0.2 * r / np.linalg.norm(r) + 0.05 * h / np.linalg.norm(h)
    assert sim.gap_to_truth(j) == pytest.approx([0.2, 0.0, 0.05], abs=1e-9)



def test_the_truth_sees_sunlight_and_stations_like_its_twin_but_raises_no_approach(data):
    """The truth is the real satellite: its illumination is worked out and it
    gets ground contacts. Its twin sits metres away by design, which is no
    close approach. (The synthetic orbit is never eclipsed in July, so a stale
    'in shadow' planted on the truth must be put right.)"""
    from satflight.scenario import GroundStation
    sc = _scenario()
    sc.stations = [GroundStation("Svalbard", 78.23, 15.39, 0.5, 5.0)]
    sim = Simulation(sc)
    i, j = sim.index_of("S1A truth"), sim.index_of("S1A twin")
    sim.sats[i].shadow = 0.0
    for _ in range(40):
        sim.advance(300.0)
        assert sim.sats[i].shadow == pytest.approx(sim.sats[j].shadow, abs=1e-3) == 1.0
    contacts = {name: sum(e.kind == "station" and e.text.split(" @ ")[0][4:] == name
                          for e in sim.events) for name in ("S1A truth", "S1A twin")}
    assert contacts["S1A truth"] == contacts["S1A twin"] > 0
    assert not any(e.kind == "alert" for e in sim.events)
    assert i not in sim.closest[1:]


def test_the_truth_stops_where_the_recording_ends(data):
    sim = Simulation(_scenario())
    i = sim.index_of("S1A truth")
    end = (datetime(2025, 7, 3, tzinfo=UTC) - EPOCH).total_seconds()
    sim.advance(end - 60.0)
    assert sim.sats[i].status == ACTIVE
    sim.advance(600.0)
    assert sim.sats[i].status == replay.ENDED
    assert any("end of" in e.text and "S1A truth" in e.text for e in sim.events)
    assert sim.sats[sim.index_of("S1A twin")].status == ACTIVE         # the twin flies on
    assert sim.gap_to_truth(sim.index_of("S1A twin")) is None


def test_the_truth_stays_at_the_end_of_its_recording_after_it_stops(data):
    """Its recording ends between two steps, seen when a sample is shown: it
    stays where the recording ends through the steps after, not where the last
    step left it (a slip of hundreds of km, 2026-11-22)."""
    sim = Simulation(_scenario())
    i = sim.index_of("S1A truth")
    E = sim.sats[i].replay
    sim.advance(E.end - 5.0)
    sim.advance(6.0)             # shown 1 s past the end, inside the twin's next step
    assert sim.sats[i].status == replay.ENDED and sim._grid[0] < E.end
    for _ in range(3):
        assert np.abs(sim.y[i] - E.state(E.end)).max() < 1e-9
        sim.advance(60.0)


def test_the_truth_cannot_be_manoeuvred(data):
    from satflight.maneuvers import Maneuver
    sim = Simulation(_scenario())
    with pytest.raises(ValueError, match="recorded"):
        sim.schedule(Maneuver(sat="S1A truth", dv=(0.001, 0.0, 0.0)))


def test_a_snapshot_keeps_the_truth_on_its_recording(data):
    sim = Simulation(_scenario())
    sim.advance(1800.0)
    snap = sim.snapshot_scenario()
    truth = next(s for s in snap.satellites if s.name == "S1A truth")
    assert truth.orbit == TRUTH
    again = Simulation(snap.copy())
    k = again.index_of("S1A truth")
    assert np.abs(again.y[k] - sim.y[sim.index_of("S1A truth")]).max() < 1e-6


def test_missing_data_is_reported_not_bridged(data):
    later = _scenario(EPOCH + timedelta(days=3))                       # no file for 07-04
    later.satellites[0].orbit["until"] = "2025-07-05"
    with pytest.raises(ValueError, match="fetch_poeorb"):
        Simulation(later)
    with pytest.raises(ValueError, match="ends before"):
        Simulation(_scenario(EPOCH + timedelta(days=2)))               # starts 07-03, until 07-02
    sc = _scenario()
    sc.satellites[0].orbit["until"] = "2025-07-05"
    with pytest.raises(ValueError, match="2025-07-03"):
        Simulation(sc)



def test_a_start_at_midnight_takes_the_samples_before_it_from_the_day_before(data):
    """The files sample at :02 and :12 past the minute, so a day's own samples
    begin 2 s after midnight; the moment before comes from the previous day."""
    midnight = datetime(2025, 7, 2, tzinfo=UTC)
    sim = Simulation(_scenario(midnight))
    r0, _ = _truth_eci(midnight)
    assert np.abs(sim.y[sim.index_of("S1A truth"), :3] - r0).max() < 1e-6
    with pytest.raises(ValueError, match="2025-06-30"):                # not on disk
        Simulation(_scenario(datetime(2025, 7, 1, tzinfo=UTC)))


# --- the real files -------------------------------------------------------------------------

def test_the_bundled_replay_scenario_runs_on_the_real_files():
    from pathlib import Path

    from satflight import eop, poeorb
    path = Path(__file__).resolve().parents[1] / "scenarios" / "data" / "sentinel1_replay.json"
    sc = Scenario.load(path)
    truth = next(s for s in sc.satellites if s.orbit.get("type") == "replay")
    first = sc.epoch.date()
    last = date.fromisoformat(truth.orbit["until"])
    have = poeorb.local_files(truth.orbit["mission"])
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    if any(d not in have for d in days) or not eop.table_path().exists():
        pytest.skip("the Sentinel-1 files or the IERS table are not on this machine")
    replay.clear_cache()
    sim = Simulation(sc)
    sim.advance(3 * 3600.0)
    gaps = [sim.gap_to_truth(k) for k in range(sim.n)]
    gaps = [g for g in gaps if g is not None]
    assert len(gaps) == 2                                            # 1A and 1C twins
    for g in gaps:
        assert np.linalg.norm(g) < 0.1                               # km after 3 h
    assert math.isfinite(sim.t)


def _exported(sim, tmp_path, **kw):
    from satflight.matlab_export import export_live_script
    return export_live_script(sim, tmp_path / "replay_test.m", **kw).read_text(encoding="utf-8")


def _column(text, name):
    """The strings of a ``sim.<name> = [...]`` column in an exported script."""
    import re
    m = re.search(rf"^sim\.{name} = \[\n(.*?)\n\];", text, re.M | re.S)
    assert m, f"no sim.{name} column"
    return [re.fullmatch(r'\s*"([^"]*)"', ln).group(1) for ln in m.group(1).splitlines()]


def test_the_matlab_export_has_the_truth_read_its_recording_and_the_twin_keep_its_mission(
        data, tmp_path):
    """MATLAB reads the same POEORB files: the script names each replay's
    mission and last day, the twin's mission (for its gap to the truth), and
    where the files are."""
    text = _exported(Simulation(_scenario()), tmp_path)
    assert _column(text, "names") == ["S1A truth", "S1A twin"]
    assert _column(text, "mission") == ["S1A", "S1A"]
    assert _column(text, "replayUntil") == ["2025-07-02", ""]
    assert 'dataFolder = "";' in text and "sim.dataFolder = dataFolder;" in text


def test_a_snapshot_export_keeps_the_twins_mission(data, tmp_path):
    """From the current moment the twin is a plain state in the scenario, but
    it is still the twin of its mission."""
    sim = Simulation(_scenario())
    sim.advance(600.0)
    text = _exported(sim, tmp_path, start="now")
    assert _column(text, "mission") == ["S1A", "S1A"]
    assert _column(text, "replayUntil") == ["2025-07-02", ""]


def test_a_lone_replay_exports_its_columns_without_brackets(data, tmp_path):
    from satflight.forces import ForceModel
    sc = Scenario(name="alone", epoch=EPOCH, forces=ForceModel(j2=False, drag=False),
                  satellites=[SatSpec("S1A truth", dict(TRUTH))])
    text = _exported(Simulation(sc), tmp_path)
    assert 'sim.mission = "S1A";' in text and 'sim.replayUntil = "2025-07-02";' in text


def test_without_a_recording_the_script_reads_no_data(tmp_path):
    from satflight.forces import ForceModel
    sc = Scenario(name="plain", epoch=EPOCH, forces=ForceModel(),
                  satellites=[SatSpec("A", {"type": "elements", "a": 7000.0, "e": 0.001,
                                            "i": 98.0, "raan": 0.0, "argp": 0.0, "nu": 0.0})])
    text = _exported(Simulation(sc), tmp_path)
    assert "dataFolder = " not in text and "sim.mission = " not in text
