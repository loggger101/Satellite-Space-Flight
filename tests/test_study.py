"""The Sentinel-1 study's measurements (satflight.study), on synthetic orbits
whose answers are known: revolution means, node longitude and LTAN, the
ground-track offset per relative orbit, the 1A-1C phasing and burn sizes."""

import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from satflight import study
from satflight.constants import MU_EARTH, R_EARTH
from satflight.elements import coe2rv, mean_to_true
from satflight.frames import eci_to_ecef_state
from satflight.orbitinfo import local_time_of_node
from satflight.poeorb import Arc, julian_dates
from satflight.timeutil import gmst

DAY = date(2025, 7, 1)
A, E, INC, RAAN, ARGP = 7071.0, 1.1e-3, math.radians(98.18), math.radians(250.0), math.pi / 2


def _arc(hours=6.0, a=A, e=E, phase=0.0, jump=None, burn=None, mission="S1A"):
    """A Kepler orbit sampled every 10 s from ``DAY`` 00:00 UTC, seen Earth-fixed,
    as a poeorb.Arc. ``jump=(t_s, da_km, di_rad)`` changes a and i at t_s;
    ``burn=(t0, t1)`` s flags those samples DEGRADED-MANOEUVRE."""
    t = np.arange(0.0, hours * 3600.0, 10.0)
    utc = np.datetime64(datetime(DAY.year, DAY.month, DAY.day), "us") + (
        t * 1e6).astype("timedelta64[us]")
    aa = np.full_like(t, a)
    ii = np.full_like(t, INC)
    if jump is not None:
        aa[t >= jump[0]] += jump[1]
        ii[t >= jump[0]] += jump[2]
    n = np.sqrt(MU_EARTH / aa ** 3)
    nu = mean_to_true(n * t + phase, e)
    r, v = coe2rv(aa, e, ii, RAAN, ARGP, nu)
    theta = gmst(julian_dates(utc))
    re, ve = eci_to_ecef_state(r, v, theta)
    man = np.zeros(len(t), bool)
    if burn is not None:
        man = (t >= burn[0]) & (t <= burn[1])
    orbit = 1000 + np.floor((n * t + phase + ARGP) / (2 * np.pi)).astype(int)
    return Arc(mission, DAY, DAY, utc, np.zeros(len(t)), np.full(len(t), 37.0), orbit, re, ve,
               man, np.zeros(len(t), bool), ())


def test_each_revolution_gives_its_mean_elements_node_longitude_and_ltan():
    arc = _arc(hours=7.5)
    revs = study.revolutions(arc)
    period = 2 * np.pi * math.sqrt(A ** 3 / MU_EARTH)
    assert len(revs["node_utc"]) == 3                     # whole revolutions only
    assert np.diff(revs["node_utc"]).astype(float) / 1e6 == pytest.approx(period, abs=1e-3)
    assert revs["a_km"] == pytest.approx(A, abs=1e-6)
    assert revs["i_deg"] == pytest.approx(math.degrees(INC), abs=1e-9)
    assert np.hypot(revs["ex"], revs["ey"]) == pytest.approx(E, rel=1e-6)
    assert np.degrees(np.arctan2(revs["ey"], revs["ex"])) == pytest.approx(90.0, abs=1e-6)
    jd = julian_dates(revs["node_utc"])
    lon = np.degrees((RAAN - gmst(jd) + np.pi) % (2 * np.pi) - np.pi)
    assert revs["lon_deg"] == pytest.approx(lon, abs=1e-6)          # ~0.1 m at the equator
    ltan = [local_time_of_node(RAAN, j) for j in jd]
    assert revs["ltan_h"] == pytest.approx(ltan, abs=1e-6)
    assert list(np.diff(revs["orbit"])) == [1, 1]
    assert not revs["burn"].any()


def test_a_revolution_mean_runs_exactly_node_to_node_not_sample_to_sample():
    """Osculating a swings ~10 km twice a revolution (J2): counting a 10 s
    sample more or less at the ends biased the mean by ~10 m, alternating."""
    t = np.arange(0.0, 6000.0, 10.0)
    t0, t1 = 9.9, 9.9 + 5910.2                      # nodes between samples
    w = 2 * np.pi / (t1 - t0)
    y = 7071.0 + 10.0 * np.cos(2 * w * (t - t0))
    m = study.span_mean(t, y, t0, t1, 7071.0 + 10.0, 7071.0 + 10.0)
    assert m == pytest.approx(7071.0, abs=1e-3)     # trapezoids: 0.01 mm; samples: 17 m


def test_a_revolution_with_a_burn_or_tracking_gap_is_flagged():
    arc = _arc(hours=7.5, burn=(11000.0, 11600.0))
    revs = study.revolutions(arc)
    assert list(revs["burn"]) == [False, True, False]


def test_ground_track_offset_is_measured_from_each_relative_orbits_median():
    orbit = np.arange(1000, 1000 + 3 * 175)
    lon = np.tile(np.linspace(-170, 170, 175), 3)
    lon[175 + 10] += 0.001                                  # 111 m east at the equator
    lon[[20, 175 + 20, 2 * 175 + 20]] = [179.9995, 179.9995, -179.9995]   # date line
    lon[30] -= 0.002                       # the odd one out is the first pass
    off = study.track_offsets(orbit, lon)
    m_per_deg = R_EARTH * math.pi / 180 * 1e3
    assert off[175 + 10] == pytest.approx(0.001 * m_per_deg, rel=1e-6)
    assert off[2 * 175 + 20] == pytest.approx(0.001 * m_per_deg, rel=1e-4)
    assert off[30] == pytest.approx(-0.002 * m_per_deg, rel=1e-4)
    assert np.count_nonzero(np.abs(off) > 1e-3) == 3


def test_1c_half_an_orbit_behind_1a_is_180_degrees():
    a = study.revolutions(_arc(hours=8.0))
    c = study.revolutions(_arc(hours=8.0, phase=-math.pi, mission="S1C"))
    times, phase = study.phasing(a, c)
    assert len(times) >= 3
    assert phase == pytest.approx(180.0, abs=1e-4)
    c2 = study.revolutions(_arc(hours=8.0, phase=-math.pi + math.radians(0.5), mission="S1C"))
    assert study.phasing(a, c2)[1] == pytest.approx(179.5, abs=1e-4)


def _revs_with_daily_wobble(step_at, da_km, di_deg, burn_revs=(), pair=()):
    """Revolution means as the real ones look: drag decay, a daily wobble in
    i and a with the node's Earth-fixed longitude, and a step at ``step_at``."""
    k = np.arange(120)
    period = 5917.0
    t = np.datetime64("2025-07-01T00:00:00", "us") + study._us(k * period)
    lon = (-24.69 * k + 180.0) % 360.0 - 180.0
    after = k >= step_at
    rad = np.radians(lon)
    a = 7071.0 - 2e-5 * k + 0.004 * np.cos(rad) + da_km * after
    i = 98.18 + 0.002 * np.sin(rad + 0.4) + 0.0004 * np.cos(2 * rad) + di_deg * after
    burn = np.isin(k, list(burn_revs) + list(pair))
    return {"node_utc": t, "period_s": np.full(len(k), period), "lon_deg": lon, "a_km": a,
            "i_deg": i, "ex": np.full(len(k), 0.0), "ey": np.full(len(k), 1.18e-3),
            "orbit": 1000 + k, "burn": burn, "gap": np.zeros(len(k), bool)}


def test_a_burn_is_sized_through_the_daily_wobble_of_the_mean_elements():
    revs = _revs_with_daily_wobble(60, 0.008, 0.0, burn_revs=(59, 60), pair=())
    t = revs["node_utc"]
    out = study.burn_sizes(revs, [(t[59] + study._us(2000), t[59] + study._us(2600))])
    assert out[0]["da_m"] == pytest.approx(8.0, abs=0.2)
    assert abs(out[0]["di_mdeg"]) < 0.05            # the wobble is 2 mdeg: not a burn
    assert out[0]["dv_cross_mm_s"] < 7.0


def test_a_pair_of_burns_an_orbit_apart_is_one_manoeuvre():
    revs = _revs_with_daily_wobble(61, 0.012, 0.0, burn_revs=(59, 60, 61))
    t = revs["node_utc"]
    w = [(t[59] + study._us(2000), t[59] + study._us(2600)),
         (t[60] + study._us(4900), t[61] + study._us(500))]
    out = study.burn_sizes(revs, w)
    assert len(out) == 1 and out[0]["windows"] == 2
    assert out[0]["da_m"] == pytest.approx(12.0, abs=0.2)


def test_a_burn_is_sized_from_the_jump_in_the_mean_elements():
    tb = 6 * 5926.0
    da, di = 0.012, math.radians(0.002)
    arc = _arc(hours=24.0, jump=(tb, da, di), burn=(tb - 300.0, tb + 300.0))
    revs = study.revolutions(arc)
    out = study.burn_sizes(revs, [(arc.utc[0] + np.timedelta64(int(tb - 300) * 10 ** 6, "us"),
                                   arc.utc[0] + np.timedelta64(int(tb + 300) * 10 ** 6, "us"))])
    assert len(out) == 1
    b = out[0]
    v = math.sqrt(MU_EARTH / A)
    assert b["da_m"] == pytest.approx(da * 1e3, rel=1e-3)
    assert b["di_mdeg"] == pytest.approx(math.degrees(di) * 1e3, rel=1e-3)
    assert b["dv_along_mm_s"] == pytest.approx(v / (2 * A) * da * 1e6, rel=2e-3)
    assert b["dv_cross_mm_s"] == pytest.approx(v * di * 1e6, rel=2e-3)


def test_a_burn_too_close_to_the_data_edge_is_not_sized():
    arc = _arc(hours=6.0, burn=(3000.0, 3600.0))
    revs = study.revolutions(arc)
    t0 = arc.utc[0] + np.timedelta64(3000 * 10 ** 6, "us")
    out = study.burn_sizes(revs, [(t0, t0 + np.timedelta64(600 * 10 ** 6, "us"))])
    assert math.isnan(out[0]["da_m"])
    assert "revolutions" in out[0]["note"]


def test_phasing_skips_nodes_with_no_partner_within_a_revolution():
    a = study.revolutions(_arc(hours=8.0))
    c = study.revolutions(_arc(hours=8.0, phase=-math.pi, mission="S1C"))
    c = {k: v[[0, 1, 3]] for k, v in c.items()}       # an outage in 1C
    times, phase = study.phasing(a, c)
    assert phase == pytest.approx(180.0, abs=1e-3)
    assert len(times) == 2


def test_weekly_arc_starts_skip_spans_the_data_does_not_cover():
    starts = study.arc_starts(date(2025, 7, 1), date(2025, 7, 31), days=4)
    assert starts[0] == date(2025, 7, 1)
    assert all((b - a) == timedelta(days=7) for a, b in zip(starts, starts[1:], strict=False))
    assert starts[-1] + timedelta(days=3) <= date(2025, 7, 31)


# --- tools/sentinel1_study.py -----------------------------------------------------------------
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentinel1_study as tool  # noqa: E402


def test_the_days_split_into_runs_around_an_outage():
    have = {date(2026, 6, d) for d in (1, 2, 3, 7, 8)}
    assert tool.runs_of_days(have, date(2026, 6, 1), date(2026, 6, 9)) == [
        (date(2026, 6, 1), date(2026, 6, 3)), (date(2026, 6, 7), date(2026, 6, 8))]


def test_a_burn_read_from_two_chunks_is_one_burn():
    t = np.datetime64("2025-07-02T00:00:00", "us")
    s = np.timedelta64(1, "s")
    w = tool.merge_windows([(t - 300 * s, t), (t, t + 300 * s), (t + 5000 * s, t + 5600 * s)])
    assert w == [(t - 300 * s, t + 300 * s), (t + 5000 * s, t + 5600 * s)]


def test_orbit_numbers_must_count_the_revolutions():
    revs = study.revolutions(_arc(hours=12.0))
    tool.check_orbit_numbers("S1A", revs)
    revs["orbit"][3:] += 1
    with pytest.raises(ValueError, match="numbered"):
        tool.check_orbit_numbers("S1A", revs)


def test_chunked_reading_gives_the_same_revolutions_as_one_read(tmp_path, monkeypatch):
    from test_poeorb import make_eof
    for d in range(1, 5):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, d),
                 burn=range(4000, 4061) if d == 2 else ())
    first, last = date(2025, 7, 1), date(2025, 7, 4)
    whole, w_whole = tool.mission_revolutions("S1A", first, last, None, tmp_path, lambda s: 0)
    monkeypatch.setattr(tool, "CHUNK_DAYS", 1)
    parts, w_parts = tool.mission_revolutions("S1A", first, last, None, tmp_path, lambda s: 0)
    assert len(whole["node_utc"]) == len(parts["node_utc"]) > 50
    assert np.array_equal(whole["node_utc"], parts["node_utc"])
    assert np.allclose(whole["a_km"], parts["a_km"])
    assert w_whole == w_parts and len(w_whole) == 1


def _fake_arc(m, d):
    if m == "S1C" and d == date(2025, 7, 8):
        raise tool.poeorb.PoeorbError("S1C: no downloaded file for 1 day(s)")
    row = {c: 1.0 for c in tool.ARC_COLUMNS if c not in ("mission", "start", "error")}
    return {**row, "mission": m, "start": d.isoformat(), "day1_s_m": 10.0, "day2_s_m": 40.0,
            "day3_s_m": 90.0, "fit_rms_m": 1.5, "check_hours": 72.0,
            "fit_hours": 23.4 if (m, d) == ("S1A", date(2025, 7, 15)) else 24.0}


def test_the_arcs_resume_and_record_what_could_not_run(tmp_path):
    calls = []

    def one(m, d):
        calls.append((m, d))
        return _fake_arc(m, d)

    first, last = date(2025, 7, 1), date(2025, 7, 20)
    tool.run_arcs(first, last, limit=3, out=lambda s: 0, dest=tmp_path, one=one)
    tool.run_arcs(first, last, out=lambda s: 0, dest=tmp_path, one=one)
    assert len(calls) == len(set(calls)) == 6          # three weeks x two, none twice
    rows = tool.read_csv(tmp_path / "arcs.csv")
    assert len(rows) == 6
    bad = [r for r in rows if r["error"]]
    assert [(r["mission"], r["start"]) for r in bad] == [("S1C", "2025-07-08")]


def test_the_arcs_come_out_in_date_order_whatever_order_they_finished_in(tmp_path):
    """Workers finish out of order and rows are appended as they do; the
    figures draw lines through the rows, so the file and its reader are sorted."""
    late = {**_fake_arc("S1C", date(2025, 7, 15)), "error": ""}
    tool.write_csv(tmp_path / "arcs.csv", [late], tool.ARC_COLUMNS)
    tool.run_arcs(date(2025, 7, 1), date(2025, 7, 20), out=lambda s: 0, dest=tmp_path,
                  one=_fake_arc)
    want = [(d, m) for d in ("2025-07-01", "2025-07-08", "2025-07-15") for m in tool.MISSIONS]
    for rows in (tool.read_csv(tmp_path / "arcs.csv"), tool.read_arcs(tmp_path)):
        assert [(r["start"], r["mission"]) for r in rows] == want
    shuffled = tool.read_csv(tmp_path / "arcs.csv")[::-1]
    tool.write_csv(tmp_path / "arcs.csv", shuffled, tool.ARC_COLUMNS)
    assert [(r["start"], r["mission"]) for r in tool.read_arcs(tmp_path)] == want


def test_the_report_is_built_from_the_csvs_alone(tmp_path):
    days = [date(2025, 7, 1) + timedelta(days=k) for k in range(30)]
    for m in tool.MISSIONS:
        tool.write_csv(tmp_path / f"daily_{m}.csv", [
            {"date": d.isoformat(), "revs": 15, "burn_revs": 0, "gap_revs": 0,
             "a_km": 7077.0 - 0.01 * k, "e": 1.1e-3, "argp_deg": 90.0, "i_deg": 98.18,
             "ltan_h": 18.0 + 0.001 * k, "track_rms_m": 50.0, "track_max_m": 120.0}
            for k, d in enumerate(days)])
    tool.write_csv(tmp_path / "phasing.csv", [{"date": d.isoformat(), "mean_deg": 180.01,
                                              "min_deg": 180.0, "max_deg": 180.02} for d in days])
    tool.write_csv(tmp_path / "burns.csv", [
        {"mission": "S1A", "kind": "burn", "start_utc": "2025-07-05T10:00:00",
         "end_utc": "2025-07-05T10:10:00",
         "da_m": 8.0, "di_mdeg": 0.1, "de_1e6": 3.0, "dv_along_mm_s": 4.2,
         "dv_cross_mm_s": 13.0, "note": ""},
        {"mission": "S1A", "kind": "placebo", "start_utc": "2025-07-20T10:00:00",
         "end_utc": "2025-07-20T10:10:00", "da_m": 0.4, "di_mdeg": 0.3, "de_1e6": 1.0,
         "dv_along_mm_s": 0.2, "dv_cross_mm_s": 39.0, "note": ""}])
    tool.run_arcs(days[0], days[-1], out=lambda s: 0, dest=tmp_path, one=_fake_arc)
    path = tool.run_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Copernicus Sentinel data" in text
    assert "180.010 deg behind 1A" in text
    assert "| Sentinel-1A | 1 | 1 | 30.0 | 1 | 4.20 |" in text
    assert ("1 placebo windows (midway between manoeuvres, where nothing burned) size to "
            "a spread of 0.6 m in a (0.30 mm/s along-track) and 0.44 mdeg in i (57.8 mm/s") in text
    assert "| 50 / 120 |" in text and "off its usual track" not in text
    assert "S1C 2025-07-08: not run" in text
    assert ("- Spans cut at a burn (an impulse needs half an hour of data after its window): "
            "1 fits shorter than a day (shortest 23.4 h), 0 checks shorter than three days.\n"
            in text)
    assert "| Sentinel-1A | day 3 | 1.0 (1.0) | 90.0 (90.0) | 1.0 (1.0) |" in text
    for png in ("elements.png", "track_phasing.png", "accuracy.png"):
        assert (tmp_path / png).stat().st_size > 10_000
        assert f"]({png})" in text


def test_a_satellite_moved_to_another_track_is_reported_not_averaged(tmp_path):
    days = [date(2026, 6, 1) + timedelta(days=k) for k in range(10)]
    for m in tool.MISSIONS:
        tool.write_csv(tmp_path / f"daily_{m}.csv", [
            {"date": d.isoformat(), "revs": 15, "a_km": 7071.0, "e": 1.1e-3, "argp_deg": 90.0,
             "i_deg": 98.18, "ltan_h": 18.0, "track_rms_m": 229e3 if (m, k) >= ("S1C", 6) else 20,
             "track_max_m": 229e3 if (m, k) >= ("S1C", 6) else 60} for k, d in enumerate(days)])
    text = tool.run_report(tmp_path, figures=False).read_text(encoding="utf-8")
    assert ("Sentinel-1C is more than 1 km off its usual track on 4 days, 2026-06-07 .. "
            "2026-06-10 (up to 229 km)") in text
    assert text.count("| 20 / 60 |") == 2


def test_placebo_windows_sit_midway_between_manoeuvres_far_enough_apart():
    t = np.datetime64("2025-07-01T00:00:00", "us")
    d = np.timedelta64(86400 * 10 ** 6, "us")
    s = np.timedelta64(10 ** 6, "us")
    w = [(t, t + 600 * s), (t + 3 * d, t + 3 * d + 600 * s), (t + 4 * d, t + 4 * d + 600 * s)]
    p = study.placebo_windows(w, min_gap_days=2.0)
    assert len(p) == 1
    mid = t + 600 * s + (3 * d - 600 * s) / 2
    assert abs((p[0][0] - mid) / np.timedelta64(1, "s")) < 1.0
    assert (p[0][1] - p[0][0]) / np.timedelta64(1, "s") == pytest.approx(600.0)


def test_an_accuracy_arc_fits_a_day_and_predicts_on_from_its_end(tmp_path):
    """The synthetic files hold a two-body circle: flown two-body, the fit and
    each day of the prediction to centimetres (a wrong hand-over: kilometres)."""
    from test_poeorb import make_eof

    from satflight.forces import ForceModel
    for d in (1, 2, 3):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, d))
    row = study.accuracy_arc("S1A", date(2025, 7, 1), ForceModel(j2=False), 2164.0,
                             fit_hours=6.0, check_hours=48.0, step=120.0, root=tmp_path)
    assert row["fit_rms_m"] < 0.01
    assert row["burns_fit"] == row["burns_check"] == 0
    for d in (1, 2):
        for c in "rsw":
            assert row[f"day{d}_{c}_m"] < 0.5, (d, c, row[f"day{d}_{c}_m"])   # ~8 cm
    assert "day3_s_m" not in row
    assert row["fit_hours"] == 6.0 and row["check_hours"] == 48.0


def _flagged(t0_h, t1_h):
    """make_eof sample indices of a burn window t0_h..t1_h hours after that
    file's own 00:00 UTC (each file starts 3618 s before it, every 10 s)."""
    return set(range(math.ceil((t0_h * 3600 + 3618) / 10),
                     math.floor((t1_h * 3600 + 3618) / 10) + 1))


@pytest.mark.parametrize("window_h", [(5.9, 6.1), (5.45, 5.65)],
                         ids=["straddles the end", "too near the end"])
def test_a_burn_at_the_end_of_the_fit_day_moves_to_the_check(tmp_path, window_h):
    """The fit ends where the burn's window begins, and the prediction (which
    starts there) fits the burn: counted once, and never without data after it."""
    from test_poeorb import make_eof

    from satflight.forces import ForceModel
    make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, 1), burn=_flagged(*window_h))
    for d in (2, 3):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, d))
    row = study.accuracy_arc("S1A", date(2025, 7, 1), ForceModel(j2=False), 2164.0,
                             fit_hours=6.0, check_hours=48.0, step=120.0, root=tmp_path)
    start_h = min(_flagged(*window_h)) * 10 / 3600 - 3618 / 3600
    assert start_h - 120 / 3600 <= row["fit_hours"] <= start_h       # on the 120 s grid
    assert row["burns_fit"] == 0 and row["burns_check"] == 1
    assert row["fit_rms_m"] < 0.01 and row["check_hours"] == 48.0
    assert all(row[f"day{d}_{c}_m"] < 0.5 for d in (1, 2) for c in "rsw")


def test_a_burn_at_the_end_of_the_check_cuts_the_check_short(tmp_path):
    """With under half an hour of truth after it a burn cannot be fitted, and
    the truth after an unfitted burn is not the model's error: the check stops
    at its window, and the last day has no end-of-day value."""
    from test_poeorb import make_eof

    from satflight.forces import ForceModel
    for d in (1, 2):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, d))
    # 6 h fit + 48 h check ends at 54 h, 06:00 on 3 July
    make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, 3), burn=_flagged(5.75, 5.9))
    row = study.accuracy_arc("S1A", date(2025, 7, 1), ForceModel(j2=False), 2164.0,
                             fit_hours=6.0, check_hours=48.0, step=120.0, root=tmp_path)
    end_h = 48 + min(_flagged(5.75, 5.9)) * 10 / 3600 - 3618 / 3600 - 6.0
    assert end_h - 120 / 3600 <= row["check_hours"] <= end_h
    assert row["burns_check"] == 0 and row["fit_hours"] == 6.0
    assert row["day2_s_m"] < 0.5 and math.isfinite(row["at24h_s_m"])
    assert math.isnan(row["at48h_s_m"])


def test_a_fit_day_cut_below_half_is_refused(tmp_path):
    from test_poeorb import make_eof

    from satflight.fit import FitError
    from satflight.forces import ForceModel
    make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, 1), burn=_flagged(2.0, 6.5))
    for d in (2, 3):
        make_eof(tmp_path / "poeorb" / "S1A", "S1A", date(2025, 7, d))
    with pytest.raises(FitError, match="burn at 2.0 h leaves under half"):
        study.accuracy_arc("S1A", date(2025, 7, 1), ForceModel(j2=False), 2164.0,
                           fit_hours=6.0, check_hours=48.0, step=120.0, root=tmp_path)


def test_only_the_burns_are_ticked_on_the_elements_figure():
    """The placebo windows are where nothing burned: no tick for them."""
    burns = [{"mission": "S1A", "kind": "burn", "start_utc": "2025-07-05T10:00:00"},
             {"mission": "S1A", "kind": "placebo", "start_utc": "2025-07-20T10:00:00"},
             {"mission": "S1C", "kind": "burn", "start_utc": "2025-07-06T23:59:00"}]
    assert tool.burn_days(burns, "S1A") == [np.datetime64("2025-07-05")]
    assert tool.burn_days(burns, "S1C") == [np.datetime64("2025-07-06")]
