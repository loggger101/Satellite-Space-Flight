"""NRLMSIS 2.1 in the drag force (forces "atmosphere": "msis"): the density at
the satellite's latitude, longitude, height and time, with the measured space
weather, held to pymsis called directly with its own reader of the same table."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pymsis
import pymsis.utils
import pytest

from satflight import spaceweather as sw
from satflight.forces import ForceModel, accel_drag
from satflight.frames import OMEGA_VEC, geodetic_to_ecef
from satflight.matlab_export import unported_forces
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.timeutil import UTC, julian_date

DAY0 = np.datetime64("2025-01-01")
STORM = 100                                   # day index of the synthetic storm


def _row(k):
    day = DAY0 + np.timedelta64(k, "D")
    aps = [300 if k == STORM else (5 * k + 7 * j) % 15 + 2 for j in range(8)]
    f = 150.0 + 20.0 * math.sin(k / 9.0)
    return ",".join(str(x) for x in [
        day, 2600, 1, *([10] * 8), 80, *aps, round(sum(aps) / 8), 0.1, 0, 50,
        f"{f:.1f}", f"{f:.1f}", "OBS", f"{150.0 + k / 10:.1f}", 0.0, 0.0, 0.0])


@pytest.fixture
def table(tmp_path, monkeypatch):
    """A 200-day measured table under a temporary data root; pymsis's reader
    pointed at the same file (both put back afterwards)."""
    monkeypatch.setenv("SATFLIGHT_DATA", str(tmp_path))
    p = sw.table_path()
    p.parent.mkdir(parents=True)
    p.write_text("\n".join([sw.HEADER, *(_row(k) for k in range(200))]) + "\n",
                 encoding="ascii")
    monkeypatch.setattr(pymsis.utils._SPACE_WEATHER, "path", p)
    monkeypatch.setattr(pymsis.utils._SPACE_WEATHER, "data", None)
    return p


def _jd(when: np.datetime64) -> float:
    return julian_date(when.astype(datetime).replace(tzinfo=UTC))


def _place(lat_deg, lon_deg, alt_km, gmst):
    """An inertial position over the given ground point when the Earth has
    turned by ``gmst``, and a 7.5 km/s velocity across it."""
    e = geodetic_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_km)
    c, s = math.cos(gmst), math.sin(gmst)
    r = np.array([[c * e[0] - s * e[1], s * e[0] + c * e[1], e[2]]])
    v = np.cross([0.0, 0.0, 1.0], r[0])
    return r, (7.5 * v / np.linalg.norm(v))[None, :] + np.array([[0.0, 0.0, 1.0]])


@pytest.mark.parametrize("when, lat, lon, alt", [
    ("2025-03-01T05:17", 37.0, -120.0, 650.0),
    ("2025-04-11T13:40", -71.5, 155.0, 702.0),      # the synthetic storm day, south
    ("2025-02-14T22:05", 81.0, 10.0, 410.0),
])
def test_drag_uses_msis_at_the_satellites_place_and_time(table, when, lat, lon, alt):
    t = np.datetime64(when, "m")
    gmst = 1.234
    r, v = _place(lat, lon, alt, gmst)
    cd_am, scale = 0.02, 1.7
    got = accel_drag(r, v, np.array([cd_am]), scale, atmosphere_model="msis", gmst=gmst,
                     jd=_jd(t))
    f107, f107a, ap = pymsis.utils.get_f107_ap(np.array([t]))
    rho = pymsis.calculate(np.array([t]), [lon], [lat], [alt], f107, f107a, ap,
                           geomagnetic_activity=-1)[0, 0] * scale
    v_rel = v - np.cross(OMEGA_VEC, r)
    want = -0.5e3 * rho * cd_am * np.linalg.norm(v_rel) * v_rel
    # MSIS computes in single precision: a position that round-trips through the
    # simulator's geodetic conversion to 1e-12 can land on the next float32
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_storm_thickens_the_air(table):
    r, v = _place(0.0, 0.0, 700.0, 0.0)
    quiet, storm = (np.linalg.norm(accel_drag(
        r, v, np.array([0.02]), atmosphere_model="msis", gmst=0.0,
        jd=_jd(np.datetime64(d, "m")))) for d in ("2025-04-08T12:00", "2025-04-11T12:00"))
    assert storm > 2.0 * quiet


def test_the_switch_is_checked_and_kept_in_scenario_files():
    with pytest.raises(ValueError, match="atmosphere 'jacchia'"):
        ForceModel(atmosphere="jacchia")
    with pytest.raises(ValueError, match="atmosphere"):
        ForceModel.from_dict({"drag": True, "atmosphere": 3})
    m = ForceModel(drag=True, atmosphere="msis")
    assert ForceModel.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    assert ForceModel().atmosphere == "table"
    assert m.label() == "2B+J2+DRAG(MSIS)" and ForceModel(drag=True).label() == "2B+J2+DRAG"
    assert m.needs_time and m.needs_gmst
    off = ForceModel(drag=False, atmosphere="msis")     # no drag, nothing to look up
    assert not off.needs_time and not off.needs_gmst and off.label() == "2B+J2"


def test_msis_drag_without_the_date_is_refused():
    r, v = _place(0.0, 0.0, 700.0, 0.0)
    with pytest.raises(ValueError, match="date"):
        ForceModel(drag=True, atmosphere="msis").acceleration(r, v, np.array([0.02]))


def test_the_breakdown_shows_msis_drag(table):
    r, v = _place(10.0, 20.0, 700.0, 0.5)
    jd = _jd(np.datetime64("2025-04-11T12:00", "m"))
    m = ForceModel(drag=True, atmosphere="msis")
    want = np.linalg.norm(m.acceleration(r, v, np.array([0.02]), 0.5, jd)
                          - ForceModel(drag=False).acceleration(r, v, np.array([0.02])))
    assert m.breakdown(r, v, 0.02, 0.5, jd)["drag"][0] == pytest.approx(want, rel=1e-12)


def _sim(epoch, atmosphere):
    return Simulation(Scenario(
        name="x", epoch=epoch, forces=ForceModel(drag=True, atmosphere=atmosphere),
        satellites=[SatSpec("S1", {"type": "elements", "altitude": 400, "i": 51.6},
                            mass=500, area=10)]))


def test_a_run_flies_through_the_storm(table):
    def drop(epoch, atmosphere):
        sim = _sim(epoch, atmosphere)
        r0 = np.linalg.norm(sim.y[0, :3])
        sim.advance(6 * 3600)
        return r0 - np.linalg.norm(sim.y[0, :3]) + 0.0, sim
    quiet, _ = drop(datetime(2025, 4, 7, tzinfo=UTC), "msis")
    storm, sim = drop(datetime(2025, 4, 11, tzinfo=UTC), "msis")
    table_run, _ = drop(datetime(2025, 4, 11, tzinfo=UTC), "table")
    assert sim.forces.label() == "2B+J2+DRAG(MSIS)"
    # the orbit's mean radius (sampled at the same phase) sinks faster in the storm
    assert storm != table_run
    assert abs(storm) > abs(quiet)


def test_the_run_gives_msis_its_clock_and_the_earths_turn(table):
    sim = _sim(datetime(2025, 4, 11, 7, 30, tzinfo=UTC), "msis")
    t = sim.t + 1234.0
    got = sim._derivative(np.arange(sim.n), [])(t, sim.y.copy())[:, 3:]
    want = sim.forces.acceleration(sim.y[:, :3], sim.y[:, 3:], sim._props(np.arange(sim.n)),
                                   sim.gmst(t), sim.jd(t))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_a_run_outside_the_measured_table_stops_with_the_reason(table):
    sim = _sim(datetime(2026, 1, 1, tzinfo=UTC), "msis")
    with pytest.raises(sw.SpaceWeatherError, match="outside the measured"):
        sim.advance(600)


def test_the_matlab_export_refuses_msis_drag():
    assert unported_forces(ForceModel(drag=True, atmosphere="msis")) == [
        "the NRLMSIS atmosphere"]
    assert unported_forces(ForceModel(drag=False, atmosphere="msis")) == []
    assert unported_forces(ForceModel(drag=True)) == []
