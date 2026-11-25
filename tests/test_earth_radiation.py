"""Earth albedo and infrared pressure (satflight/earth_radiation.py).

The quadrature over the visible cap is held to closed forms: a sphere that
glows evenly (exitance M) lights a point at distance r with M (R/r)^2 at any
height, and a fully lit Lambertian sphere seen from far away reflects with a
geometric albedo of 2/3 its Bond albedo. Knocke's albedo and emissivity are
held to their published coefficients (Knocke, Ries & Tapley 1988).
"""

from __future__ import annotations

import numpy as np
import pytest

from satflight import earth_radiation as er
from satflight.constants import AU, P_SRP, R_EARTH

JD_DEC = 2444960.5             # 1981-12-22, Knocke's t0: the seasonal term at its peak
JD_JUN = JD_DEC + 365.25 / 2


def test_knocke_albedo_and_emissivity_are_the_published_coefficients():
    lat = np.radians([90.0, 0.0, -90.0])
    np.testing.assert_allclose(er.knocke_albedo(lat, JD_DEC), [0.34 + 0.10 + 0.29,
                                                               0.34 - 0.29 / 2,
                                                               0.34 - 0.10 + 0.29])
    np.testing.assert_allclose(er.knocke_emissivity(lat, JD_DEC), [0.68 - 0.07 - 0.18,
                                                                   0.68 + 0.18 / 2,
                                                                   0.68 + 0.07 - 0.18])
    # half a year on the seasonal term has turned round
    np.testing.assert_allclose(er.knocke_albedo(lat[:1], JD_JUN), [0.34 - 0.10 + 0.29])


@pytest.mark.parametrize("height", [300.0, 700.0, 5000.0, 36000.0])
def test_an_evenly_glowing_earth_pushes_out_as_its_closed_form(height):
    r = np.array([[0.3, -0.5, 0.8]])
    r = (R_EARTH + height) * r / np.linalg.norm(r)
    sun = np.array([AU, 0.0, 0.0])
    a = er.accel_earth_radiation(r, sun, np.array([1.0]), JD_DEC, albedo=False, infrared=True,
                                 emissivity=lambda lat, jd: np.full_like(lat, 1.0))
    # exitance M = e S / 4 with S at 1 AU; irradiance M (R/r)^2, pressure over c
    want = P_SRP / 4 * (R_EARTH / np.linalg.norm(r)) ** 2 * 1e-3
    rh = r[0] / np.linalg.norm(r)
    assert a[0] @ rh == pytest.approx(want, rel=1e-12)         # exact: linear in cos(eta)
    assert np.linalg.norm(a[0] - (a[0] @ rh) * rh) < 1e-6 * want     # straight out


def test_a_fully_lit_far_earth_reflects_two_thirds_of_its_bond_albedo():
    rh = np.array([0.0, 0.6, 0.8])
    r = (1000 * R_EARTH) * rh[None]
    sun = AU * rh                                   # the Sun behind the satellite
    a = er.accel_earth_radiation(r, sun, np.array([1.0]), JD_DEC, albedo=True, infrared=False,
                                 albedo_of=lambda lat, jd: np.full_like(lat, 0.3))
    want = 2 / 3 * 0.3 * P_SRP * (R_EARTH / 1000 / R_EARTH) ** 2 * 1e-3
    assert a[0] @ rh == pytest.approx(want, rel=2e-3)


def test_the_night_side_reflects_nothing():
    r = np.array([[7078.0, 0.0, 0.0]])
    sun = np.array([-AU, 0.0, 0.0])                 # the satellite over midnight
    a = er.accel_earth_radiation(r, sun, np.array([1.0]), JD_DEC, albedo=True, infrared=False)
    assert np.all(a == 0.0)


def test_sentinel_size_and_direction():
    # Sentinel-1 (Cr A/m about 1.3 * 10 / 2164) over the subsolar point at 693 km
    r = np.array([[R_EARTH + 693.0, 0.0, 0.0]])
    sun = np.array([AU, 0.0, 0.0])
    cram = np.array([1.3 * 10 / 2164])
    alb = er.accel_earth_radiation(r, sun, cram, JD_DEC, albedo=True, infrared=False)[0]
    ir = er.accel_earth_radiation(r, sun, cram, JD_DEC, albedo=False, infrared=True)[0]
    direct = P_SRP * cram[0] * 1e-3
    assert alb[0] > 0 and ir[0] > 0                 # both push away from the Earth
    assert alb[0] / direct == pytest.approx(0.160, abs=0.002)   # measured 2026-11-25
    assert ir[0] / direct == pytest.approx(0.156, abs=0.002)


def test_many_satellites_at_once_match_one_by_one():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(5, 3))
    r = (R_EARTH + 700) * u / np.linalg.norm(u, axis=1, keepdims=True)
    sun = AU * np.array([0.2, 0.9, 0.39]) / np.linalg.norm([0.2, 0.9, 0.39])
    cram = rng.uniform(0.001, 0.02, 5)
    allat = er.accel_earth_radiation(r, sun, cram, JD_JUN)
    one = np.vstack([er.accel_earth_radiation(r[k:k + 1], sun, cram[k:k + 1], JD_JUN)
                     for k in range(5)])
    np.testing.assert_allclose(allat, one, rtol=1e-12)


def test_on_the_ground_the_cap_closes_and_the_push_is_zero():
    r = np.array([[R_EARTH, 0.0, 0.0], [R_EARTH - 1.0, 0.0, 0.0]])
    a = er.accel_earth_radiation(r, np.array([AU, 0, 0]), np.array([1.0, 1.0]), JD_DEC)
    assert np.all(a == 0.0)


def test_the_force_model_switch_adds_the_term_and_needs_cr():
    from satflight.ephemeris import sun_position
    from satflight.forces import ForceModel
    r = np.array([[R_EARTH + 700.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.5, 0.0]])
    jd = JD_DEC + 0.3
    on, off = ForceModel(j2=False, albedo=True), ForceModel(j2=False)
    cram = np.array([0.006])
    got = (on.acceleration(r, v, np.zeros(1), jd=jd, cr_a_over_m=cram)
           - off.acceleration(r, v, np.zeros(1), jd=jd, cr_a_over_m=cram))
    want = er.accel_earth_radiation(r, sun_position(jd), cram, jd)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-18)
    assert on.label() == "2B+ALBEDO" and on.needs_time
    with pytest.raises(ValueError, match="Cr"):
        on.acceleration(r, v, np.zeros(1), jd=jd)
    assert ForceModel.from_dict(on.to_dict()) == on


def test_the_matlab_export_refuses_albedo_until_ported():
    from satflight.forces import ForceModel
    from satflight.matlab_export import unported_forces
    assert unported_forces(ForceModel(albedo=True)) == ["Earth albedo and infrared"]
