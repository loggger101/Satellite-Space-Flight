"""One meaning of altitude per use (audit A3, A9, D50): orbit shape quantities
are r - R_eq, everything physical is the height above the WGS-84 ellipsoid."""

import io
import math

import numpy as np
import pytest

from satflight.analysis import classify, lowest_height
from satflight.atmosphere import density
from satflight.batch import _rows, summary
from satflight.constants import OMEGA_EARTH, R_EARTH, R_EARTH_POLAR
from satflight.elements import coe2rv, rv2coe
from satflight.forces import ForceModel
from satflight.frames import ecef_to_geodetic, eci_to_ecef, geodetic_height, geodetic_to_ecef
from satflight.orbitinfo import orbit_info
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation


def test_closed_form_geodetic_is_exact_everywhere():
    rng = np.random.default_rng(7)
    lat = rng.uniform(-math.pi / 2, math.pi / 2, 5000)
    lon = rng.uniform(-math.pi, math.pi, 5000)
    h = rng.uniform(-5.0, 400000.0, 5000)
    lat[:3], h[:3] = [math.pi / 2, -math.pi / 2, 0.0], [0.0, 300.0, 35786.0]
    la, lo, hh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert np.max(np.abs(hh - h)) < 1e-8                         # km
    assert np.max(np.abs(la - lat)) < 1e-12
    # height is the same from ECI: the ellipsoid is symmetric about the axis
    r = geodetic_to_ecef(lat, lon, h)
    assert np.allclose(geodetic_height(eci_to_ecef(r, 1.234)), h, atol=1e-8)


def test_polar_perigee_15_km_above_the_ground_is_not_suborbital():
    rp, ra = R_EARTH_POLAR + 15.0, R_EARTH + 700.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    r, v = coe2rv(a, e, math.radians(90), 0.0, math.radians(90), 0.3)
    el = rv2coe(r, v)
    assert el.rp - R_EARTH < -6                          # r_p - R_eq says "underground"
    assert lowest_height(el) == pytest.approx(15.0, abs=0.01)
    assert not classify(el).startswith("suborbital")
    r, v = coe2rv(a, e, math.radians(90), 0.0, 0.0, 0.3)  # same ellipse, perigee on the equator
    assert classify(rv2coe(r, v)).startswith("suborbital")


def test_perigee_drag_uses_the_real_height_and_the_air_speed():
    rp, ra = R_EARTH + 300.0, R_EARTH + 900.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    sat = type("S", (), {"mass": 100.0, "area": 1.0, "cd": 2.2})()
    r, v = coe2rv(a, e, math.radians(90), 0.0, math.radians(90), 1.0)   # perigee over the pole
    info = orbit_info(r, v, 2461000.0, sat)
    assert info.rp_alt == pytest.approx(300.0, abs=1e-6)
    assert info.perigee_height == pytest.approx(300.0 + R_EARTH - R_EARTH_POLAR, abs=0.05)
    assert info.rho_perigee == pytest.approx(density(info.perigee_height))
    r, v = coe2rv(a, e, 0.0, 0.0, 0.0, 1.0)                              # equatorial, prograde
    info = orbit_info(r, v, 2461000.0, sat)
    v_air = info.v_peri - OMEGA_EARTH * rp
    assert info.drag_perigee == pytest.approx(
        0.5 * info.rho_perigee * (v_air * 1e3) ** 2 * 2.2 / 100.0, rel=1e-9)


def test_batch_summary_and_csv_give_the_same_altitude():
    sc = Scenario(name="t", forces=ForceModel(j2=True), satellites=[
        SatSpec("Polar", {"type": "elements", "altitude": 500, "i": 90, "nu": 89.9,
                          "osculating": True})])
    sim = Simulation(sc)
    csv_alt = float(next(_rows(sim, sim.t, sim.y))[10])
    out = io.StringIO()
    summary(sim, out)
    row = next(line for line in out.getvalue().splitlines() if line.startswith("Polar"))
    assert float(row.split()[2]) == pytest.approx(csv_alt, abs=0.05)
    assert csv_alt == pytest.approx(500.0 + R_EARTH - R_EARTH_POLAR, abs=0.5)


def test_the_satellite_list_shows_the_height_above_the_ground():
    # over the pole the height above the ellipsoid is ~21 km more than r - R_eq;
    # the list must say what the tooltip says
    from satflight.ui.panels import _list_value
    sim = Simulation(Scenario(name="t", satellites=[SatSpec("Polar", {
        "type": "elements", "a": R_EARTH + 400.0, "e": 0.0, "i": 90.0, "raan": 0.0,
        "argp": 0.0, "nu": 90.0, "osculating": True})]))
    r = sim.y[0, :3]
    assert abs(r[2]) > 0.99 * np.linalg.norm(r)            # over the pole
    text = _list_value(sim, 0)[0]
    assert text == f"{float(geodetic_height(r)):,.0f} km"
    assert text != f"{np.linalg.norm(r) - R_EARTH:,.0f} km"
