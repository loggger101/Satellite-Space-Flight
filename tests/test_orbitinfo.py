"""The orbit inspector's derived quantities (orbitinfo.orbit_info)."""

import math

import numpy as np
import pytest

from satflight.constants import MU_EARTH, R_EARTH
from satflight.elements import coe2rv, kepler_propagate, rv2coe, vis_viva
from satflight.ephemeris import sun_position
from satflight.orbitinfo import local_time_of_node, orbit_info
from satflight.simulation import Satellite

D = np.radians
JD = 2461313.0          # 2026-09-29 12:00 UTC


def molniya(nu=10.0):
    return coe2rv(26554.0, 0.72, D(63.4), D(250), D(270), D(nu))


def test_apsis_speeds_and_shape():
    info = orbit_info(*molniya(), JD)
    el = info.el
    assert info.v_peri == pytest.approx(vis_viva(el.rp, el.a), rel=1e-12)
    assert info.v_apo == pytest.approx(vis_viva(el.ra, el.a), rel=1e-12)
    assert info.v_peri * el.rp == pytest.approx(info.v_apo * el.ra, rel=1e-12)
    assert info.b == pytest.approx(26554.0 * math.sqrt(1 - 0.72 ** 2), rel=1e-9)
    assert info.c == pytest.approx(26554.0 * 0.72, rel=1e-9)
    assert info.rp_alt + info.ra_alt + 2 * R_EARTH == pytest.approx(2 * 26554.0, rel=1e-9)
    assert info.v_esc == pytest.approx(math.sqrt(2) * info.v_circ)
    assert info.regime == "Molniya (HEO)"
    assert not info.retrograde and info.closed and not info.circular


@pytest.mark.parametrize("nu", [10.0, 170.0, 300.0])
def test_next_pass_times_land_on_the_events(nu):
    r, v = molniya(nu)
    info = orbit_info(r, v, JD)
    for t in (info.t_peri, info.t_apo, info.t_an, info.t_dn):
        assert 0 < t <= info.period
    nu_p = rv2coe(*kepler_propagate(r, v, info.t_peri)).nu
    nu_a = rv2coe(*kepler_propagate(r, v, info.t_apo)).nu
    assert min(nu_p, 2 * math.pi - nu_p) < 1e-7
    assert abs(nu_a - math.pi) < 1e-7
    r_an, v_an = kepler_propagate(r, v, info.t_an)
    r_dn, v_dn = kepler_propagate(r, v, info.t_dn)
    assert abs(r_an[2]) < 1e-3 and v_an[2] > 0
    assert abs(r_dn[2]) < 1e-3 and v_dn[2] < 0


def test_circular_equatorial_has_no_apsides_or_nodes():
    r, v = coe2rv(42164.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    info = orbit_info(r, v, JD)
    assert info.circular and info.equatorial
    assert math.isinf(info.t_peri) and math.isinf(info.t_an)
    assert math.isnan(info.ltan)
    assert info.period == pytest.approx(86164.1, abs=1.0)
    assert abs(info.ground_shift) < 0.05          # geostationary: the track stands still


def test_eclipse_fraction_matches_cylindrical_shadow_at_zero_beta():
    """Orbit plane containing the Sun: shadow arc is 2*asin(R/r) of 2*pi
    (conical shadow and penumbra shift it by well under 1 %)."""
    s = sun_position(JD)
    s_hat = s / np.linalg.norm(s)
    w = np.cross(s_hat, [0.0, 0.0, 1.0])
    w /= np.linalg.norm(w)
    rr = R_EARTH + 400.0
    r = rr * s_hat
    v = math.sqrt(MU_EARTH / rr) * w
    info = orbit_info(r, v, JD)
    assert abs(info.beta) < 1e-9
    assert info.eclipse_fraction == pytest.approx(math.asin(R_EARTH / rr) / math.pi, abs=0.01)
    assert info.timeline_lit[0] == pytest.approx(1.0)          # starts at the sub-solar point
    assert info.eclipse_duration == pytest.approx(info.eclipse_fraction * info.period)


def test_high_beta_orbit_never_eclipses():
    s = sun_position(JD)
    s_hat = s / np.linalg.norm(s)
    # orbit normal along the Sun line: beta = 90 deg
    a = np.cross(s_hat, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    rr = R_EARTH + 800.0
    info = orbit_info(rr * a, math.sqrt(MU_EARTH / rr) * np.cross(s_hat, a), JD)
    assert math.degrees(abs(info.beta)) == pytest.approx(90.0, abs=1e-6)
    assert info.eclipse_fraction == 0.0


def test_ltan_and_sun_synchronous_inclination():
    s = sun_position(JD)
    ra_sun = math.atan2(s[1], s[0])
    assert local_time_of_node(ra_sun, s) == pytest.approx(12.0)
    assert local_time_of_node(ra_sun + D(90), s) == pytest.approx(18.0)
    assert local_time_of_node(ra_sun - D(90), s) == pytest.approx(6.0)
    r, v = coe2rv(R_EARTH + 700, 0.0, D(98.19), ra_sun + D(-67.5), 0, 0.3)
    info = orbit_info(r, v, JD)
    assert info.ltan == pytest.approx(7.5, abs=1e-6)
    assert math.degrees(info.sso_inclination) == pytest.approx(98.19, abs=0.02)
    assert math.degrees(info.raan_dot) * 86400 == pytest.approx(0.9856, abs=0.002)


def test_physical_properties():
    sat = Satellite("x", (255, 255, 255), mass=400.0, area=4.0, cd=2.2)
    r, v = coe2rv(R_EARTH + 400, 0.0, D(51.6), 0, 0, 0)
    info = orbit_info(r, v, JD, sat)
    assert info.area_to_mass == pytest.approx(0.01)
    assert info.ballistic == pytest.approx(400 / (2.2 * 4))
    # 0.5 rho v^2 Cd A/m with rho(400 km) = 3.725e-12
    expected = 0.5 * 3.725e-12 * (info.v_peri * 1e3) ** 2 * 2.2 * 0.01
    assert info.drag_perigee == pytest.approx(expected, rel=1e-2)
    assert 1000 < info.footprint_km < 2500
    # ground track moves west by ~ (earth rate) * period, ~23 deg for the ISS
    assert info.ground_shift == pytest.approx(23.2, abs=0.5)


def test_hyperbolic_orbit():
    r = np.array([R_EARTH + 300.0, 0, 0])
    v = np.array([0, 11.5, 0.5])
    info = orbit_info(r, v, JD)
    assert not info.closed and info.c3 > 0
    assert math.isinf(info.period) and math.isinf(info.t_apo)
    assert info.timeline_t.size == 0 and math.isnan(info.eclipse_duration)
