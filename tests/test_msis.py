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
from satflight.matlab_export import export_live_script, unported_forces
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


def test_the_matlab_export_flies_msis_drag(table, tmp_path):
    """MATLAB runs NRLMSIS through pymsis (matlab/tests/ExportTest nrlmsisDrag):
    the script names the atmosphere and reads the space weather from dataFolder."""
    assert unported_forces(ForceModel(drag=True, atmosphere="msis")) == []
    texts = {}
    for atmosphere in ("msis", "table"):
        sim = _sim(datetime(2025, 4, 11, tzinfo=UTC), atmosphere)
        path = export_live_script(sim, tmp_path / f"run_{atmosphere}.m", duration=600.0)
        texts[atmosphere] = path.read_text(encoding="utf-8")
    msis, plain = texts["msis"], texts["table"]
    assert '"atmosphere", "msis"' in msis and '"atmosphere", "table"' in plain
    assert 'dataFolder = "";' in msis and "sim.dataFolder = dataFolder;" in msis
    assert 'dataFolder = "";' not in plain and "sim.dataFolder = dataFolder;" not in plain
    assert "pymsis" in msis.split("function ")[0]          # the script says what it needs


def test_msis_state_gives_the_density_species_and_temperature(table):
    from satflight import atmosphere
    from satflight.drag_coefficient import MASS_KG, SPECIES
    jd = _jd(np.datetime64("2025-03-01T05:17", "m"))
    h = np.array([95.0, 300.0, 693.0, 1000.0, 2600.0])
    lat, lon = np.radians([10.0, -40.0, 70.0, 0.0, 5.0]), np.radians([0.0, 90.0, -150, 20, 0])
    st = atmosphere.msis_state(h, lat, lon, jd)
    assert st.number_density.shape == (5, len(SPECIES))
    assert np.array_equal(st.rho, atmosphere.msis_density(h, lat, lon, jd))
    assert np.all(np.isfinite(st.number_density)) and np.all(st.number_density >= 0)
    air = h <= atmosphere.TOP_KM
    # the species add up to the total (pymsis works in single precision)
    np.testing.assert_allclose((st.number_density[air] * MASS_KG).sum(1), st.rho[air],
                               rtol=2e-3)
    assert np.all(st.temperature[air] > 150.0)
    assert st.rho[~air] == 0.0 and np.all(st.number_density[~air] == 0.0)


def test_msis_state_refuses_a_species_pymsis_leaves_unknown_in_the_air(table, monkeypatch):
    """A NaN species reads as none only where that leaves MSIS's own total
    intact (a species it does not model at that height); one that hides real
    air is an error, not a zero."""
    from satflight import atmosphere
    real = pymsis.calculate

    def broken(*a, **k):
        out = real(*a, **k)
        out[:, 3] = np.nan                     # atomic oxygen
        return out
    monkeypatch.setattr(pymsis, "calculate", broken)
    with pytest.raises(ValueError, match="species add up"):
        atmosphere.msis_state(np.array([500.0]), np.zeros(1), np.zeros(1),
                              _jd(np.datetime64("2025-03-01T05:17", "m")))


# --- the physical drag coefficient (forces "cd_model": "dria_sphere") ---------------

def test_the_cd_model_is_checked_and_kept_in_scenario_files():
    assert ForceModel().cd_model == "fixed"
    with pytest.raises(ValueError, match="cd_model 'newton'"):
        ForceModel(drag=True, atmosphere="msis", cd_model="newton")
    with pytest.raises(ValueError, match="needs the NRLMSIS atmosphere"):
        ForceModel(drag=True, cd_model="dria_sphere")
    m = ForceModel(drag=True, atmosphere="msis", cd_model="dria_sphere")
    assert ForceModel.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    assert m.label() == "2B+J2+DRAG(MSIS,Cd)"
    assert ForceModel(drag=False, atmosphere="msis", cd_model="dria_sphere").label() == "2B+J2"


@pytest.mark.parametrize("when, lat, lon, alt", [
    ("2025-03-01T05:17", 37.0, -120.0, 693.0),
    ("2025-04-11T13:40", -71.5, 155.0, 702.0),      # the synthetic storm day
])
def test_physical_cd_scales_drag_by_the_sphere_coefficient_over_2_2(table, when, lat, lon,
                                                                     alt):
    """The satellite's Cd*A/m is taken at Cd 2.2; the drag uses the sphere's Cd
    for the air at that place and time instead."""
    from satflight import atmosphere
    from satflight.drag_coefficient import sphere_cd
    from satflight.frames import eci_to_geodetic
    t = np.datetime64(when, "m")
    gmst = 0.7
    r, v = _place(lat, lon, alt, gmst)
    fixed = accel_drag(r, v, np.array([0.02]), 1.3, "msis", gmst, _jd(t))
    got = accel_drag(r, v, np.array([0.02]), 1.3, "msis", gmst, _jd(t), cd_model="dria_sphere")
    la, lo, h = eci_to_geodetic(r, gmst)
    st = atmosphere.msis_state(h, la, lo, _jd(t))
    speed = np.linalg.norm(v - np.cross(OMEGA_VEC, r), axis=-1) * 1e3
    cd = sphere_cd(speed, st.number_density, st.temperature)
    assert 2.3 < cd[0] < 3.4
    np.testing.assert_allclose(got, fixed * cd[0] / 2.2, rtol=1e-12)


def test_physical_cd_flows_through_the_force_model(table):
    r, v = _place(10.0, 20.0, 693.0, 0.5)
    jd = _jd(np.datetime64("2025-04-11T12:00", "m"))
    base = ForceModel(drag=False).acceleration(r, v, np.array([0.02]))
    m = ForceModel(drag=True, atmosphere="msis", cd_model="dria_sphere")
    want = accel_drag(r, v, np.array([0.02]), 1.0, "msis", 0.5, jd, cd_model="dria_sphere")
    np.testing.assert_allclose(m.acceleration(r, v, np.array([0.02]), 0.5, jd) - base, want,
                               rtol=1e-9, atol=1e-18)
    assert m.breakdown(r, v, 0.02, 0.5, jd)["drag"][0] == pytest.approx(
        np.linalg.norm(want), rel=1e-12)


def test_physical_cd_is_not_exported_to_matlab_yet(table, tmp_path):
    assert unported_forces(ForceModel(drag=True, atmosphere="msis",
                                      cd_model="dria_sphere")) == ["physical Cd"]
    assert unported_forces(ForceModel(drag=False, atmosphere="msis",
                                      cd_model="dria_sphere")) == []
