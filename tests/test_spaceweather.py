"""Space weather (satflight.spaceweather): CelesTrak's SW-All.csv read strictly,
and the F10.7 / ap inputs NRLMSIS needs at any time, held to pymsis's own
reader of the same file."""

from __future__ import annotations

import numpy as np
import pytest

from satflight import spaceweather as sw

HEADER = ("DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,"
          "AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,"
          "F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81")
DAY0 = np.datetime64("2025-01-01")


def _ap(k, j):
    """A distinct 3-hourly ap for day k, slot j, so every index tells."""
    return (7 * k + 3 * j) % 47 + 1


def _row(k, kind="OBS", f107=None, aps=None):
    day = DAY0 + np.timedelta64(k, "D")
    aps = aps if aps is not None else [_ap(k, j) for j in range(8)]
    f = f107 if f107 is not None else 100.0 + k
    return ",".join(str(x) for x in [
        day, 2600, 1, *([10] * 8), 80, *aps, round(sum(aps) / 8), 0.1, 0, 50,
        f"{f:.1f}", f"{f:.1f}", kind, f"{150.0 + k / 10:.1f}", 0.0, 0.0, 0.0])


def _file(tmp_path, rows, header=HEADER):
    p = tmp_path / "SW-All.csv"
    p.write_text("\n".join([header, *rows]) + "\n", encoding="ascii")
    return p


def _days(n, measured=None, **kw):
    """n days from DAY0; the first ``measured`` are OBS, the rest PRD."""
    measured = n if measured is None else measured
    return [_row(k, "OBS" if k < measured else "PRD", **kw) for k in range(n)]


def test_a_table_keeps_measured_days_and_stops_at_predictions(tmp_path):
    rows = _days(100, measured=90)
    rows[20] = _row(20, "INT")                     # CelesTrak's fill for a missing reading
    t = sw.read(_file(tmp_path, [*rows, "2041-10-01,2837,1,,,,,,,,,,,,,,,,,,,,,10,69.8,"
                                        "70.0,PRM,68.8,69.0,69.2,70.5"]))
    assert t.day[0] == DAY0 and len(t.day) == 90
    assert t.f107[5] == 105.0 and t.f107a[5] == 150.5
    assert t.ap3[3].tolist() == [_ap(3, j) for j in range(8)]


def test_indices_match_pymsis_on_the_same_file(tmp_path, monkeypatch):
    pymsis_utils = pytest.importorskip("pymsis.utils")
    p = _file(tmp_path, _days(150, measured=140))
    t = sw.read(p)
    when = (DAY0 + np.timedelta64(3, "D")).astype("datetime64[m]") + np.arange(
        0, 90 * 24 * 60, 97, dtype="timedelta64[m]")
    f107, f107a, ap = t.indices(when)
    # pymsis's own reader of the same file (put back as it was afterwards)
    monkeypatch.setattr(pymsis_utils._SPACE_WEATHER, "path", p)
    monkeypatch.setattr(pymsis_utils._SPACE_WEATHER, "data", None)
    ref = pymsis_utils.get_f107_ap(when)
    np.testing.assert_allclose(f107, ref[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(f107a, ref[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ap, ref[2], rtol=0, atol=1e-12)


def test_the_ap_array_by_hand(tmp_path):
    t = sw.read(_file(tmp_path, _days(100)))
    _, _, ap = t.indices(np.array(["2025-01-10T07:30"], "datetime64[m]"))
    s = [_ap(k, j) for k in range(10) for j in range(8)]       # slots from DAY0
    k = 9 * 8 + 2                                              # 06:00-09:00 on day 9
    want = [round(sum(_ap(9, j) for j in range(8)) / 8), s[k], s[k - 1], s[k - 2],
            s[k - 3], np.mean(s[k - 11:k - 3]), np.mean(s[k - 19:k - 11])]
    np.testing.assert_allclose(ap[0], want)


def test_f107_is_the_previous_days(tmp_path):
    t = sw.read(_file(tmp_path, _days(100)))
    f107, f107a, _ = t.indices(np.array(["2025-01-10T23:59"], "datetime64[m]"))
    assert f107[0] == 108.0 and f107a[0] == 150.9


@pytest.mark.parametrize("when, why", [
    ("2025-01-02T00:00", "ap history"),          # needs 57 h of ap before it
    ("2025-03-02T00:00", "81-day"),              # its centred mean reaches predictions
    ("2025-04-20T00:00", "outside"),             # after the table
])
def test_times_without_measured_inputs_are_refused(tmp_path, when, why):
    t = sw.read(_file(tmp_path, _days(110, measured=100)))
    with pytest.raises(sw.SpaceWeatherError, match=why):
        t.indices(np.array([when], "datetime64[m]"))


def test_a_radio_burst_day_is_refused_not_smoothed(tmp_path):
    rows = _days(100)
    rows[49] = _row(49, f107=563.5)
    t = sw.read(_file(tmp_path, rows))
    t.indices(np.array(["2025-02-19T12:00"], "datetime64[m]"))      # uses day 48's
    with pytest.raises(sw.SpaceWeatherError, match=r"2025-02-19.*563\.5.*radio burst"):
        t.indices(np.array(["2025-02-20T12:00"], "datetime64[m]"))


@pytest.mark.parametrize("change, why", [
    (lambda rows: [HEADER.replace("AP1", "APX"), *rows[1:]], "columns"),
    (lambda rows: [rows[0], rows[1], rows[3], *rows[4:]], "does not follow"),
    (lambda rows: [rows[0], rows[1].replace(",OBS,", ",XYZ,"), *rows[2:]], "line 2"),
    (lambda rows: [rows[0], rows[1].replace(",100.0,", ",x,", 1), *rows[2:]], "line 2"),
    (lambda rows: [rows[0], rows[1], rows[2].replace(",OBS,", ",PRD,"),
                   rows[3], *rows[4:]], "after a prediction"),
])
def test_a_broken_file_is_named(tmp_path, change, why):
    p = tmp_path / "SW-All.csv"
    p.write_text("\n".join(change([HEADER, *_days(10)])) + "\n", encoding="ascii")
    with pytest.raises(sw.SpaceWeatherError, match=why):
        sw.read(p)


def test_a_missing_table_says_where_it_looked_and_how_to_get_it(tmp_path):
    with pytest.raises(sw.SpaceWeatherError, match=r"fetch_space_weather\.py"):
        sw.read(tmp_path / "nope.csv")


def test_load_follows_the_data_root_and_a_replaced_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SATFLIGHT_DATA", str(tmp_path))
    p = sw.table_path()
    assert p == tmp_path / "spaceweather" / "SW-All.csv"
    p.parent.mkdir()
    p.write_text("\n".join([HEADER, *_days(100)]) + "\n", encoding="ascii")
    assert len(sw.load().day) == 100
    p.write_text("\n".join([HEADER, *_days(120)]) + "\n", encoding="ascii")
    assert len(sw.load().day) == 120


def test_fetch_keeps_a_good_table_and_refuses_a_broken_one(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "fetch_space_weather", Path(__file__).parents[1] / "tools" / "fetch_space_weather.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    good = ("\n".join([HEADER, *_days(100, measured=95)]) + "\n").encode()
    said = []
    dest = tool.fetch(tmp_path, get=lambda url: good, out=said.append)
    assert dest.read_bytes() == good and "2025-04-05" in said[0]
    with pytest.raises(sw.SpaceWeatherError):
        tool.fetch(tmp_path, get=lambda url: b"<html>busy</html>")
    assert dest.read_bytes() == good                 # the good table stays


def test_the_downloaded_table_has_the_storm_of_2025_11_12():
    p = sw.table_path()
    if not p.is_file():
        pytest.skip(f"no space-weather table at {p}")
    f107, f107a, ap = sw.load().indices(np.array(["2025-11-12T01:30"], "datetime64[m]"))
    assert ap[0, 1] == 300 and ap[0, 0] == 137           # CelesTrak: Kp 9-, Ap 137
    assert f107[0] == 168.0 and f107a[0] == 144.3
