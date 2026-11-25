"""The future Sun for the NRLMSIS lifetime: measured space weather where it
exists, NOAA SWPC's predicted solar cycle after it, then the mean of the
measured cycles, repeated."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from satflight import solarforecast as sf
from satflight import spaceweather as sw

MINIMA = [np.datetime64(m, "D") for m in ("1964-10-01", "1976-03-01", "1986-09-01",
                                          "1996-08-01", "2008-12-01", "2019-12-01")]
PEAKS = [150.0, 210.0, 200.0, 160.0, 110.0]        # each cycle's peak F10.7 above 70


def _cycle_f107(days):
    """70 sfu at each minimum, a sin^2 bump of the cycle's own peak between."""
    out = np.full(days.shape, 70.0)
    for k in range(5):
        a, b = MINIMA[k], MINIMA[k + 1]
        inside = (days >= a) & (days < b)
        phase = (days[inside] - a).astype(float) / float((b - a).astype(int))
        out[inside] = 70.0 + PEAKS[k] * np.sin(math.pi * phase) ** 2
    return out


def _table(last="2026-10-31"):
    """A measured table from 1960 to ``last`` with the cycles above and a daily
    Ap of 12 (3-hourly ap 12 too)."""
    day = np.arange(np.datetime64("1960-01-01"), np.datetime64(last) + np.timedelta64(1, "D"))
    f = _cycle_f107(day)
    return sw.Table(day, np.full((day.size, 8), 12.0), np.full(day.size, 12.0), f, f)


def _swpc_json(first="2026-05", months=56, base=130.0):
    m0 = np.datetime64(first, "M")
    rows = []
    for k in range(months):
        v = base - (base - 70.0) * k / (months - 1)
        rows.append({"time-tag": str(m0 + np.timedelta64(k, "M")), "predicted_f10.7": round(v, 1),
                     "high_f10.7": round(v + 7, 1), "low_f10.7": round(v - 6, 1),
                     "predicted_ssn": 0.0})
    return json.dumps(rows)


@pytest.fixture
def swpc(tmp_path):
    p = tmp_path / "forecast.json"
    p.write_text(_swpc_json(), encoding="ascii")
    return sf.read_swpc(p)


def _utc(s):
    return np.array([np.datetime64(s, "m")])


def test_the_swpc_file_is_read_strictly(tmp_path, swpc):
    assert swpc.month[0] == np.datetime64("2026-05") and swpc.month[-1] == np.datetime64("2030-12")
    assert swpc.f107["mean"][0] == 130.0 and swpc.f107["high"][0] == 137.0
    bad = tmp_path / "bad.json"
    for text, why in [
        ("not json", "not JSON"),
        (json.dumps([]), "no months"),
        (json.dumps([{"time-tag": "2026-05"}]), "predicted_f10.7"),
        (_swpc_json().replace('"2026-06"', '"2026-08"'), "does not follow"),
        (_swpc_json().replace('"predicted_f10.7": 130.0', '"predicted_f10.7": 900.0'),
         "outside"),
    ]:
        bad.write_text(text, encoding="ascii")
        with pytest.raises(sw.SpaceWeatherError, match=why):
            sf.read_swpc(bad)


def test_the_mean_cycle_averages_the_measured_cycles_by_phase():
    cyc = sf.mean_cycle(_table())
    assert cyc.period_days == pytest.approx(float((MINIMA[-1] - MINIMA[0]).astype(int)) / 5)
    mid = np.argmin(np.abs(cyc.phase - 0.5))
    assert cyc.f107["mean"][mid] == pytest.approx(70 + np.mean(PEAKS), rel=2e-3)
    assert cyc.f107["high"][mid] == pytest.approx(70 + max(PEAKS), rel=2e-3)
    assert cyc.f107["low"][mid] == pytest.approx(70 + min(PEAKS), rel=2e-3)
    assert cyc.f107["mean"][0] == pytest.approx(70.0, abs=0.5)


def test_measured_days_come_from_the_table_unchanged(swpc):
    t = _table()
    fc = sf.SolarForecast(t, swpc, sf.mean_cycle(t), ap_mean=12.0)
    when = _utc("2025-03-01T05:17")
    for got, want in zip(fc.indices(when), t.indices(when), strict=True):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(fc.at_level("high").indices(when), t.indices(when), strict=True):
        np.testing.assert_array_equal(got, want)


def test_after_the_measured_days_the_swpc_forecast_then_the_mean_cycle(swpc):
    t = _table()
    cyc = sf.mean_cycle(t)
    fc = sf.SolarForecast(t, swpc, cyc, ap_mean=12.0)
    # mid-month inside the forecast: that month's value, for F10.7 and its mean
    f, fa, ap = fc.indices(_utc("2028-03-15T12:00"))
    k = int((np.datetime64("2028-03") - swpc.month[0]).astype(int))
    assert f[0] == pytest.approx(swpc.f107["mean"][k], abs=0.2) and fa[0] == f[0]
    assert ap.shape == (1, 7) and np.all(ap == 12.0)
    hi, lo = fc.at_level("high").indices(_utc("2028-03-15"))[0], \
        fc.at_level("low").indices(_utc("2028-03-15"))[0]
    assert hi[0] > f[0] > lo[0]
    # past the forecast: the mean cycle, starting where the forecast ends
    start = (swpc.month[-1] + np.timedelta64(1, "M")).astype("datetime64[D]")
    when = start + np.timedelta64(round(0.5 * cyc.period_days), "D")
    f, _, _ = fc.indices(np.array([when.astype("datetime64[m]")]))
    assert f[0] == pytest.approx(70 + np.mean(PEAKS), rel=5e-3)
    # and again one mean cycle later
    later = when + np.timedelta64(round(cyc.period_days), "D")
    again = fc.indices(np.array([later.astype("datetime64[m]")]))[0][0]
    assert again == pytest.approx(f[0], rel=5e-3)


def test_a_forecast_that_starts_after_the_measured_days_is_refused(tmp_path):
    t = _table(last="2026-10-31")
    p = tmp_path / "late.json"
    p.write_text(_swpc_json(first="2027-06"), encoding="ascii")
    fc = sf.SolarForecast(t, sf.read_swpc(p), sf.mean_cycle(t), ap_mean=12.0)
    with pytest.raises(sw.SpaceWeatherError, match="forecast starts"):
        fc.indices(_utc("2027-01-10"))


def test_ap_mean_is_the_measured_daily_ap_over_the_cycles():
    t = _table()
    assert sf.mean_ap(t) == pytest.approx(12.0)


def test_unknown_level_is_refused(swpc):
    t = _table()
    with pytest.raises(ValueError, match="level"):
        sf.SolarForecast(t, swpc, sf.mean_cycle(t), ap_mean=12.0, level="median")


def test_fetch_keeps_a_good_forecast_and_refuses_a_broken_one(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "fetch_space_weather", Path(__file__).parents[1] / "tools" / "fetch_space_weather.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    good = _swpc_json().encode()
    said = []
    dest = tool.fetch_forecast(tmp_path, get=lambda url: good, out=said.append)
    assert dest == sf.forecast_path(tmp_path) and dest.read_bytes() == good
    assert "2026-05..2030-12" in said[0]
    with pytest.raises(sw.SpaceWeatherError):
        tool.fetch_forecast(tmp_path, get=lambda url: b"<html>busy</html>")
    assert dest.read_bytes() == good
