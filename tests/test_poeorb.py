"""ESA precise orbit files (satflight.poeorb, tools/fetch_poeorb.py): file names,
reading with every check failing loudly, stitching days, the frame conversion
and the download tool. Synthetic files stand in for ESA's, built in the same
layout as the real ones (checked against S1A 2025-07-01); a last test reads the
real downloads when this machine has them."""

from __future__ import annotations

import io
import math
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from satflight import poeorb
from satflight.elements import rv2coe
from satflight.frames import eci_to_ecef_state
from satflight.poeorb import PoeorbError, load_arc, parse_name, read_file, to_eci
from satflight.timeutil import Clock, gmst, julian_date

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fetch_poeorb  # noqa: E402

UTC = timezone.utc
DUT1 = 0.0434
MU = 398600.4418
REF = datetime(2025, 6, 1, tzinfo=UTC)


def _name(mission, day, created=None):
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(seconds=3618)
    stop = start + timedelta(hours=26)
    created = created or start + timedelta(days=21)
    return (f"{mission}_OPER_AUX_POEORB_OPOD_{created:%Y%m%dT%H%M%S}_V{start:%Y%m%dT%H%M%S}_"
            f"{stop:%Y%m%dT%H%M%S}.EOF"), start, stop


def _states(t_utc, dut1=DUT1, offset_m=0.0):
    """A circular polar orbit (ECI), seen Earth-fixed: (r_ecef km, v_ecef km/s, r_eci, v_eci)."""
    a, inc = 7071.0, math.radians(98.18)
    n = math.sqrt(MU / a ** 3)
    t = np.array([(x - REF).total_seconds() for x in t_utc])   # one orbit for every file
    u = n * t + 0.3
    p = np.stack([np.cos(u), np.sin(u) * math.cos(inc), np.sin(u) * math.sin(inc)], 1)
    q = np.stack([-np.sin(u), np.cos(u) * math.cos(inc), np.cos(u) * math.sin(inc)], 1)
    r, v = a * p, a * n * q
    r = r + offset_m / 1000.0 * p
    theta = gmst(np.array([julian_date(x) for x in t_utc]) + dut1 / 86400.0)
    re, ve = eci_to_ecef_state(r, v, theta)
    return re, ve, r, v


def make_eof(folder: Path, mission="S1A", day=date(2025, 7, 1), mutate=None, created=None,
             offset_m=0.0, zipped=True, burn=(), gap=()):
    """Write a synthetic POEORB file for ``day``; ``mutate(header, osvs)`` may
    break it first. Returns its path."""
    name, start, stop = _name(mission, day, created)
    times = [start + timedelta(seconds=10 * k) for k in range(9361)]
    re, ve, _, _ = _states(times, offset_m=offset_m)
    header = {"File_Name": name.removesuffix(".EOF"), "File_Type": "AUX_POEORB",
              "Mission": f"Sentinel-1{mission[-1]}",
              "Validity_Start": f"UTC={start:%Y-%m-%dT%H:%M:%S}",
              "Validity_Stop": f"UTC={stop:%Y-%m-%dT%H:%M:%S}", "Ref_Frame": "EARTH_FIXED",
              "Time_Reference": "UTC", "count": None}
    osvs = []
    for k, t in enumerate(times):
        osvs.append({
            "TAI": f"TAI={t + timedelta(seconds=37):%Y-%m-%dT%H:%M:%S}.000000",
            "UTC": f"UTC={t:%Y-%m-%dT%H:%M:%S}.000000",
            "UT1": f"UT1={t + timedelta(seconds=DUT1):%Y-%m-%dT%H:%M:%S.%f}",
            "Absolute_Orbit": f"+{59882 + k // 590}",
            **{c: (f"{re[k, j] * 1000:.6f}", "m") for j, c in enumerate("XYZ")},
            **{c: (f"{ve[k, j] * 1000:.6f}", "m/s") for j, c in enumerate(("VX", "VY", "VZ"))},
            "Quality": ("DEGRADED-MANOEUVRE" if k in burn else
                        "DEGRADED-GAP" if k in gap else "NOMINAL")})
    if mutate:
        mutate(header, osvs)
    count = header["count"] if header["count"] is not None else len(osvs)

    def field(tag, val):
        if isinstance(val, tuple):
            return f'<{tag} unit="{val[1]}">{val[0]}</{tag}>'
        return f"<{tag}>{val}</{tag}>"

    body = "".join("<OSV>" + "".join(field(t, v) for t, v in o.items()) + "</OSV>"
                   for o in osvs)
    h = header
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><Earth_Explorer_File><Earth_Explorer_Header>'
           f"<Fixed_Header><File_Name>{h['File_Name']}</File_Name><Mission>{h['Mission']}"
           f"</Mission><File_Type>{h['File_Type']}</File_Type><Validity_Period><Validity_Start>"
           f"{h['Validity_Start']}</Validity_Start><Validity_Stop>{h['Validity_Stop']}"
           f"</Validity_Stop></Validity_Period></Fixed_Header><Variable_Header><Ref_Frame>"
           f"{h['Ref_Frame']}</Ref_Frame><Time_Reference>{h['Time_Reference']}"
           f"</Time_Reference></Variable_Header></Earth_Explorer_Header><Data_Block type=\"xml\">"
           f'<List_of_OSVs count="{count}">{body}</List_of_OSVs></Data_Block>'
           f"</Earth_Explorer_File>")
    folder.mkdir(parents=True, exist_ok=True)
    if not zipped:
        path = folder / name
        path.write_text(xml, encoding="utf-8")
        return path
    path = folder / (name + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, xml)
    return path


# --- names and folders ----------------------------------------------------------------------

def test_a_file_name_gives_its_day_and_folder():
    f = parse_name("S1A_OPER_AUX_POEORB_OPOD_20260720T070517_V20260629T225942_"
                   "20260701T005942.EOF.zip")
    assert (f.mission, f.day, f.folder) == ("S1A", date(2026, 6, 30), "2026/06")
    assert f.created == datetime(2026, 7, 20, 7, 5, 17, tzinfo=UTC)
    for bad in ("S1A_OPER_AUX_RESORB_OPOD_20260720T070517_V20260629T225942_20260701T005942.EOF",
                "S1A_OPER_AUX_POEORB_OPOD_20260720T070517_V20260629T225942_20260629T235942.EOF",
                "notes.txt"):
        with pytest.raises(ValueError):
            parse_name(bad)


def test_the_first_of_a_month_comes_from_the_month_before():
    assert poeorb.folders_for(date(2025, 7, 1), date(2025, 7, 31)) == ["2025/06", "2025/07"]
    assert poeorb.folders_for(date(2025, 7, 2), date(2025, 8, 1)) == ["2025/07"]


def test_a_reissued_day_keeps_the_newest_file_and_listings_are_read():
    old, *_ = _name("S1C", date(2025, 7, 3), datetime(2025, 7, 24, tzinfo=UTC))
    new, *_ = _name("S1C", date(2025, 7, 3), datetime(2025, 8, 2, tzinfo=UTC))
    html = (f'<a href="../">..</a><a href="{old}.zip">x</a><a href="{new}.zip">y</a>'
            '<a href="?C=M;O=A">sort</a><a href="readme.txt">r</a>')
    names = poeorb.listing_names(html)
    assert names == [old + ".zip", new + ".zip"]
    assert poeorb.newest_per_day(names)[date(2025, 7, 3)].name == new + ".zip"


def test_the_data_folder_follows_satflight_data(monkeypatch, tmp_path):
    monkeypatch.setenv("SATFLIGHT_DATA", str(tmp_path))
    assert poeorb.mission_dir("S1C") == tmp_path / "poeorb" / "S1C"
    monkeypatch.delenv("SATFLIGHT_DATA")
    assert poeorb.data_root() == Path.home() / "satflight-data"
    with pytest.raises(ValueError):
        poeorb.mission_dir("S2A")


# --- reading ----------------------------------------------------------------------------------

def test_a_file_reads_in_km_with_ut1_and_burn_and_gap_flags(tmp_path):
    f = read_file(make_eof(tmp_path, burn=range(100, 161), gap=range(5000, 5400)))
    assert (f.mission, f.day, len(f.utc), f.tai_utc) == ("S1A", date(2025, 7, 1), 9361, 37.0)
    assert f.utc[0] == np.datetime64("2025-06-30T22:59:42")
    assert np.allclose(f.dut1, DUT1, atol=1e-6)
    assert np.allclose(np.linalg.norm(f.r, axis=1), 7071.0, atol=1e-6)     # km, not m
    assert f.manoeuvre.sum() == 61 and f.manoeuvre[100] and not f.manoeuvre[99]
    assert f.gap.sum() == 400 and not f.gap[100] and f.degraded.sum() == 461
    assert read_file(make_eof(tmp_path / "plain", zipped=False)).r.shape == (9361, 3)


def _set(field, value, k=500):
    def m(h, osvs):
        osvs[k][field] = value
    return m


def _header(field, value):
    def m(h, osvs):
        h[field] = value
    return m


def _drop(k):
    def m(h, osvs):
        del osvs[k]
    return m


def _swap_v(h, osvs):
    for o in osvs:
        o["VX"], o["VY"] = o["VY"], o["VX"]


def _back_orbit(h, osvs):
    osvs[700]["Absolute_Orbit"] = "+59000"


BROKEN = {
    "file type": (_header("File_Type", "AUX_RESORB"), "File_Type"),
    "frame": (_header("Ref_Frame", "INERTIAL"), "Ref_Frame"),
    "mission": (_header("Mission", "Sentinel-1C"), "Mission"),
    "validity": (_header("Validity_Stop", "UTC=2025-07-02T00:59:52"), "Validity_Stop"),
    "count": (_header("count", 9360), "count says"),
    "unit": (_set("X", ("7000.0", "km")), "in 'km'"),
    "quality": (_set("Quality", "DEGRADED"), "unknown quality"),
    "missing tag": (lambda h, o: o[3].pop("UT1"), "no <UT1>"),
    "time prefix": (_set("UTC", "2025-06-30T23:23:02.000000"), "lacks 'UTC='"),
    "gap": (_drop(4000), "not 10 s"),
    "leap": (_set("TAI", "TAI=2025-07-01T00:22:39.000000", 8400), "TAI-UTC changes"),
    "dut1": (_set("UT1", "UT1=2025-06-30T23:23:03.000000", 140), "UT1-UTC"),
    "orbit": (_back_orbit, "orbit number goes back"),
    "radius": (_set("Z", ("99999999.0", "m")), "|r|"),
    "number": (_set("Y", ("12,5", "m")), "unreadable number"),
    "velocity": (_swap_v, "velocity disagrees"),
}


@pytest.mark.parametrize("what", BROKEN)
def test_every_check_refuses_its_defect(tmp_path, what):
    mutate, says = BROKEN[what]
    with pytest.raises(PoeorbError, match=f"S1A_OPER.*{_re(says)}"):
        read_file(make_eof(tmp_path, mutate=mutate))


def _re(text):
    import re
    return re.escape(text)


def test_a_burn_window_is_exempt_from_the_velocity_check(tmp_path):
    def kick(h, osvs):                   # a burn: positions continue, velocity jumps 2 m/s
        for k in range(3000, 3061):
            vx = float(osvs[k]["VX"][0]) + 2.0
            osvs[k]["VX"] = (f"{vx:.6f}", "m/s")
            osvs[k]["Quality"] = "DEGRADED-MANOEUVRE"
    assert read_file(make_eof(tmp_path, mutate=kick)).manoeuvre.sum() == 61

    def unflagged(h, osvs):
        kick(h, osvs)
        for k in range(3000, 3061):
            osvs[k]["Quality"] = "NOMINAL"
    with pytest.raises(PoeorbError, match="velocity disagrees"):
        read_file(make_eof(tmp_path / "b", mutate=unflagged))


def test_unreadable_files_say_so(tmp_path):
    name, *_ = _name("S1A", date(2025, 7, 1))
    p = tmp_path / (name + ".zip")
    p.write_bytes(b"<html>503 Service Unavailable</html>")
    with pytest.raises(PoeorbError, match="cannot be read"):
        read_file(p)
    with pytest.raises(PoeorbError, match="not a POEORB file name"):
        read_file(tmp_path / "orbit.zip")


# --- frames and time --------------------------------------------------------------------------

def test_earth_fixed_states_come_back_to_the_simulators_eci(tmp_path):
    f = read_file(make_eof(tmp_path))
    r, v = to_eci(f.utc, f.dut1, f.r, f.v)
    times = [f.start + timedelta(seconds=10 * k) for k in range(9361)]
    _, _, r0, v0 = _states(times)
    # a float64 Julian date resolves ~40 us, 2 cm of Earth rotation at the satellite
    assert np.abs(r - r0).max() < 1e-4 and np.abs(v - v0).max() < 1e-7   # km, km/s
    r_utc, _ = to_eci(f.utc, 0.0, f.r, f.v)                               # UT1 taken as UTC
    assert 0.002 < np.linalg.norm(r_utc - r0, axis=1).max() < 0.03         # 0.0434 s ~ 20 m


def test_the_clock_turns_the_earth_at_ut1():
    epoch = datetime(2025, 7, 1, tzinfo=UTC)
    shift = Clock(epoch, dut1=0.5).gmst(100.0) - Clock(epoch).gmst(100.0)
    assert shift == pytest.approx(0.5 * 7.2921158553e-5, rel=1e-4)    # Earth's spin rate
    assert Clock(epoch, dut1=0.5).jd(100.0) == Clock(epoch).jd(100.0)    # UTC is unchanged
    with pytest.raises(ValueError, match="UT1-UTC"):
        Clock(epoch, dut1=0.9)


# --- several days -----------------------------------------------------------------------------

def test_days_stitch_into_one_arc_and_seams_are_measured(tmp_path):
    folder = tmp_path / "poeorb" / "S1A"
    for d in (date(2025, 7, 1), date(2025, 7, 2), date(2025, 7, 3)):
        make_eof(folder, day=d)
    arc = load_arc("S1A", date(2025, 7, 1), date(2025, 7, 3), root=tmp_path)
    # ESA's samples fall on 22:59:42 + 10 s steps: :02, :12, ... after each minute
    assert len(arc.utc) == 3 * 8640 and arc.utc[0] == np.datetime64("2025-07-01T00:00:02")
    assert arc.utc[-1] == np.datetime64("2025-07-03T23:59:52")
    assert len(arc.seams) == 2 and all(s.points == 721 for s in arc.seams)
    assert max(s.max_pos_m for s in arc.seams) < 1e-3
    assert arc.seconds()[-1] == 3 * 86400 - 10
    r, _ = arc.eci()
    assert np.allclose(np.linalg.norm(r, axis=1), 7071.0)


def test_a_missing_day_or_a_disagreeing_seam_stops_the_arc(tmp_path):
    folder = tmp_path / "poeorb" / "S1A"
    make_eof(folder, day=date(2025, 7, 1))
    make_eof(folder, day=date(2025, 7, 3))
    with pytest.raises(PoeorbError, match="1 day.*2025-07-02"):
        load_arc("S1A", date(2025, 7, 1), date(2025, 7, 3), root=tmp_path)
    make_eof(folder, day=date(2025, 7, 2), offset_m=15.0)    # 15 m off its neighbours
    with pytest.raises(PoeorbError, match="differs from the day before by 15.0"):
        load_arc("S1A", date(2025, 7, 1), date(2025, 7, 3), root=tmp_path)


def test_a_seam_inside_a_tracking_gap_is_kept_but_not_measured(tmp_path):
    folder = tmp_path / "poeorb" / "S1A"
    make_eof(folder, day=date(2025, 7, 1), gap=range(8640, 9361))   # its last two hours
    make_eof(folder, day=date(2025, 7, 2), offset_m=15.0)           # would fail if measured
    arc = load_arc("S1A", date(2025, 7, 1), date(2025, 7, 2), root=tmp_path)
    (s,) = arc.seams
    assert s.points == 0 and math.isnan(s.max_pos_m)
    assert arc.gap.sum() == 362              # 22:59:42..23:59:52 of 1 July, every 10 s


# --- the download tool ------------------------------------------------------------------------

def test_downloads_are_checked_before_they_are_kept():
    name, *_ = _name("S1A", date(2025, 7, 1))
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as z:
        z.writestr(name, "<x/>")
    fetch_poeorb.check_zip(name + ".zip", good.getvalue())
    with pytest.raises(ValueError, match="not a zip"):
        fetch_poeorb.check_zip(name + ".zip", b"<html>error</html>")
    other = io.BytesIO()
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("something_else.EOF", "<x/>")
    with pytest.raises(ValueError, match="expected just"):
        fetch_poeorb.check_zip(name + ".zip", other.getvalue())


def test_fetch_gets_missing_days_replaces_reissues_and_reports_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("SATFLIGHT_DATA", str(tmp_path))
    monkeypatch.setattr(fetch_poeorb, "PAUSE", 0.0)
    src = tmp_path / "server"
    files = {p.name: p.read_bytes() for p in (
        make_eof(src, day=date(2025, 7, 1)), make_eof(src, day=date(2025, 7, 2)))}
    stale, *_ = _name("S1A", date(2025, 7, 2), datetime(2025, 7, 20, tzinfo=UTC))
    here = tmp_path / "poeorb" / "S1A"
    here.mkdir(parents=True)
    (here / (stale + ".zip")).write_bytes(b"old")           # an earlier issue of 2 July
    asked = []

    def get(url):
        asked.append(url)
        if url.endswith("/"):
            folder = url.rstrip("/").rsplit("/", 2)
            month = f"{folder[-2]}/{folder[-1]}"
            return "".join(f'<a href="{n}">' for n in files
                           if parse_name(n).folder == month).encode()
        return files[url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(fetch_poeorb, "_get", get)
    missing = fetch_poeorb.fetch("S1A", date(2025, 7, 1), date(2025, 7, 3))
    assert missing == [date(2025, 7, 3)]                    # not on the server: reported
    assert sorted(p.name for p in here.iterdir()) == sorted(files)   # stale issue replaced
    assert any(u.endswith("/S1A/2025/06/") for u in asked)  # 1 July sits in June
    asked.clear()
    assert fetch_poeorb.fetch("S1A", date(2025, 7, 1), date(2025, 7, 2)) == []
    assert all(u.endswith("/") for u in asked)              # nothing downloaded twice
    assert fetch_poeorb.main(["--sats", "S1A", "--from", "2025-07-01", "--to",
                              "2025-07-03"]) == 1           # a gap fails the run


# --- the real files, when downloaded --------------------------------------------------------

def test_the_downloaded_sentinel_1_files_read_as_sentinel_1():
    have = poeorb.local_files("S1A")
    if not have:
        pytest.skip(f"no S1A files under {poeorb.mission_dir('S1A')} (tools/fetch_poeorb.py)")
    day = min(have)
    arc = load_arc("S1A", day, day)
    r, v = arc.eci()
    el = rv2coe(r[0], v[0])
    assert 7050 < el.a < 7090 and math.degrees(el.i) == pytest.approx(98.18, abs=0.05)
    assert np.all(np.abs(arc.dut1) < 0.9) and arc.tai_utc[0] == 37.0
