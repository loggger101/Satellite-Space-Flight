"""NRLMSIS 2.1 in the drag force (forces "atmosphere": "msis"): the density at
the satellite's latitude, longitude, height and time, with the measured space
weather, held to pymsis called directly with its own reader of the same table."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

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


def test_the_matlab_export_carries_the_drag_coefficient_model(table, tmp_path):
    """MATLAB flies the physical Cd too (matlab/tests/ExportTest physicalCd): the
    script's forces name the model, fixed or the sphere."""
    assert unported_forces(ForceModel(drag=True, atmosphere="msis",
                                      cd_model="dria_sphere")) == []
    for model in ("dria_sphere", "fixed"):
        sim = Simulation(Scenario(
            name="x", epoch=datetime(2025, 4, 11, tzinfo=UTC),
            forces=ForceModel(drag=True, atmosphere="msis", cd_model=model),
            satellites=[SatSpec("S1", {"type": "elements", "altitude": 400, "i": 51.6},
                                mass=500, area=10)]))
        text = export_live_script(sim, tmp_path / f"cd_{model}.m",
                                  duration=600.0).read_text(encoding="utf-8")
        assert f'"cdModel", "{model}"' in text


def test_the_orbit_tab_reads_the_perigee_air_from_msis(table):
    """Under NRLMSIS the perigee density is MSIS's at the perigee point, at the
    run's time, where the Earth has turned by ``gmst``; the drag there follows."""
    from satflight.atmosphere import msis_density
    from satflight.elements import coe2rv
    from satflight.frames import eci_to_geodetic
    from satflight.orbitinfo import orbit_info

    jd, gmst = _jd(np.datetime64("2025-04-11T13:40", "m")), 2.0     # the storm day
    sat = type("S", (), {"mass": 100.0, "area": 1.0, "cd": 2.2})()
    rp, ra = 6378.137 + 400.0, 6378.137 + 900.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    r, v = coe2rv(a, e, math.radians(97), 0.4, math.radians(60), 2.0)
    table_info = orbit_info(r, v, jd, sat)
    info = orbit_info(r, v, jd, sat, 1.5, atmosphere="msis", gmst=gmst)
    rpv, _ = coe2rv(a, e, math.radians(97), 0.4, math.radians(60), 0.0)
    lat, lon, h = eci_to_geodetic(rpv, gmst)
    want = float(msis_density(np.atleast_1d(h), np.atleast_1d(lat), np.atleast_1d(lon), jd, 1.5)[0])
    assert info.rho_perigee == pytest.approx(want, rel=1e-12)
    assert info.rho_perigee != pytest.approx(table_info.rho_perigee * 1.5, rel=0.01)
    assert info.drag_perigee / table_info.drag_perigee == pytest.approx(
        info.rho_perigee / table_info.rho_perigee, rel=1e-9)
    with pytest.raises(ValueError, match="gmst"):
        orbit_info(r, v, jd, sat, atmosphere="msis")


def _msis_estimate_and_flight(epoch, orbit, mass, area):
    """(estimated, flown) seconds to re-entry under NRLMSIS with J2, both driven
    by the same measured table (the fixture's)."""
    from satflight import lifetime as lt
    from satflight import spaceweather
    from satflight.analysis import osculating_to_mean_a
    from satflight.elements import rv2coe
    from satflight.simulation import ACTIVE
    sim = Simulation(Scenario(name="t", epoch=epoch,
                              satellites=[SatSpec("x", orbit, mass=mass, area=area)],
                              forces=ForceModel(j2=True, drag=True, atmosphere="msis")))
    s = sim.sats[0]
    el = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    a_mean = float(osculating_to_mean_a(el.a, el.e, el.i, el.argp, el.nu))
    est = lt.lifetime(a_mean, el.e, el.i, s.cd * s.area / s.mass,
                      sun=spaceweather.load(), jd=sim.jd(), raan=el.raan)
    while s.status == ACTIVE and sim.t < 2 * est:
        sim.advance(6 * 3600)
    down = [e.t for e in sim.events if "re-entered" in e.text]
    assert down, f"still up after {sim.t / 86400:.1f} days (estimate {est / 86400:.2f})"
    return est, down[0]


@pytest.mark.parametrize("epoch, orbit, mass, area", [
    # down in about six days, through the fixture's storm on day 3: a date held at
    # the start would make it last 17 % longer
    (datetime(2025, 4, 8, tzinfo=UTC), {"type": "elements", "altitude": 350, "i": 97}, 10, 1),
    (datetime(2025, 2, 20, tzinfo=UTC),
     {"type": "elements", "perigee_alt": 200, "apogee_alt": 500, "i": 51.6}, 30, 6),
], ids=["polar-through-the-storm", "eccentric"])
def test_the_msis_lifetime_matches_a_run_flown_to_reentry(table, epoch, orbit, mass, area):
    est, flown = _msis_estimate_and_flight(epoch, orbit, mass, area)
    assert est == pytest.approx(flown, rel=0.08), (est / 86400, flown / 86400)


def test_the_msis_lifetime_follows_the_density_scale_and_needs_the_date(table):
    """Not exactly 1 / scale as under the table: denser air brings it down
    sooner, through other days' weather."""
    from satflight import lifetime as lt
    jd = _jd(np.datetime64("2025-02-10T00:00", "m"))
    args = (6378.137 + 280.0, 0.001, math.radians(97), 0.01)
    one = lt.lifetime(*args, sun=sw.load(), jd=jd, raan=1.0)
    two = lt.lifetime(*args, density_scale=2.0, sun=sw.load(), jd=jd, raan=1.0)
    assert 0 < two < one < math.inf
    assert two == pytest.approx(one / 2, rel=0.1)
    assert one != pytest.approx(lt.lifetime(*args), rel=1e-3)      # not the table's
    with pytest.raises(ValueError, match="date"):
        lt.lifetime(*args, sun=sw.load())


def test_the_msis_lifetime_turns_the_node_with_j2(table, monkeypatch):
    """The local solar time follows the node: the RAAN each step samples
    advances at J2's rate (sun-synchronous orbits keep their local time)."""
    from satflight import lifetime as lt
    from satflight.analysis import j2_secular_rates
    seen = []
    real = lt.decay_rates

    def spy(*args, **kw):
        seen.append((kw["jd"], kw["raan"]))
        return real(*args, **kw)
    monkeypatch.setattr(lt, "decay_rates", spy)
    jd = _jd(np.datetime64("2025-02-10T00:00", "m"))
    a, i = 6378.137 + 320.0, math.radians(97)
    lt.lifetime(a, 0.0005, i, 0.05, sun=sw.load(), jd=jd, raan=0.5)
    days = np.array([s[0] for s in seen]) - jd
    raan = np.array([s[1] for s in seen])
    rate = float(j2_secular_rates(a, 0.0005, i)[0]) * 86400.0        # rad/day at the start
    assert days[-1] > 5 and np.ptp(raan) > 0.05
    early = days < 3                       # before the sinking orbit speeds the node up
    np.testing.assert_allclose(raan[early] - 0.5, rate * days[early], rtol=0.01, atol=1e-4)
    assert np.all(np.diff(raan[np.argsort(days)]) >= 0)


def test_the_msis_column_matches_msis_at_and_between_its_heights(table):
    from satflight.atmosphere import msis_density
    from satflight.launch import COLUMN_KM, msis_column
    jd = _jd(np.datetime64("2025-03-01T05:17", "m"))
    lat, lon = 28.5, -80.6
    air = msis_column(lat, lon, jd)

    def want(h):
        return float(msis_density(np.array([h]), math.radians(lat), math.radians(lon), jd)[0])
    for h in (0.0, 10.0, 50.0, 86.0, 120.0, 300.0):
        assert h in COLUMN_KM
        assert air(h) == pytest.approx(want(h), rel=1e-6)
    for h in (0.4, 13.7, 85.5, 150.5, 710.0):          # between grid heights: log-linear
        assert air(h) == pytest.approx(want(h), rel=0.02)
    assert air(-0.3) == air(0.0) and air(3000.0) == 0.0


def _launch_run(atmosphere):
    from satflight.launch import LAUNCH_SITES, LaunchSpec, vehicle_preset
    lat, lon, alt = LAUNCH_SITES["Cape Canaveral SLC-40 (USA)"]
    launch = LaunchSpec(name="Up", lat=lat, lon=lon, alt=alt, payload_mass=5000.0,
                        vehicle=vehicle_preset("Falcon 9 (approx.)"), perigee_alt=400.0,
                        apogee_alt=400.0, inclination=51.6, direction="north",
                        timing="delay", delay=900.0)
    sim = Simulation(Scenario(name="x", epoch=datetime(2025, 3, 1, tzinfo=UTC),
                              forces=ForceModel(drag=True, atmosphere=atmosphere),
                              launches=[launch]))
    return sim, sim.ascent_of(sim.index_of("Up")), (lat, lon)


def test_an_msis_run_flies_its_ascents_through_the_column_over_the_pad(table):
    from satflight.atmosphere import msis_density
    from satflight.launch import us76_density
    sim, asc, (lat, lon) = _launch_run("msis")
    jd0 = sim.jd(asc.t0)                                    # liftoff, not the run's start
    for h in (10.0, 120.0):
        want = float(msis_density(np.array([h]), math.radians(lat), math.radians(lon), jd0)[0])
        assert asc.env.air(h) == pytest.approx(want, rel=1e-6)
    sim.advance(1500.0)
    assert sim.sats[sim.index_of("Up")].launch_report["outcome"] == "orbit"
    _, asc_table, _ = _launch_run("table")
    assert asc_table.env.air is us76_density


def test_an_msis_run_with_a_launch_exports_to_matlab(table, tmp_path):
    """MATLAB's ascents fly the NRLMSIS column over the pad too (its runtime's
    msisColumn; matlab/tests/ExportTest nrlmsisAscent holds it to the simulator),
    from the pad and in powered flight."""
    sim, _, _ = _launch_run("msis")
    text = export_live_script(sim, tmp_path / "up.m").read_text(encoding="utf-8")
    assert '"atmosphere", "msis"' in text and '"Up"' in text
    sim.advance(1000.0)                     # 100 s into the ascent
    export_live_script(sim, tmp_path / "flying.m", start="now")


def test_the_matlab_column_heights_are_the_simulators():
    """matlab_runtime.m's C.COLUMN_KM is launch.COLUMN_KM, so the two interpolate
    the same NRLMSIS points."""
    from satflight.launch import COLUMN_KM
    src = (Path(__file__).parents[1] / "satflight" / "matlab_runtime.m").read_text(
        encoding="utf-8")
    m = re.search(r"^C\.COLUMN_KM = \[0:149, 150:10:1000\];", src, re.M)
    assert m, "C.COLUMN_KM not found in matlab_runtime.m"
    assert tuple(float(h) for h in (*range(0, 150), *range(150, 1001, 10))) == COLUMN_KM
