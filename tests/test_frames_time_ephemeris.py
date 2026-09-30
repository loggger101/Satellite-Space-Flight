"""Time scales, frame conversions, look angles, the Sun ephemeris and shadows."""

import math
from datetime import datetime

import numpy as np
import pytest

from satflight.constants import AU, R_EARTH, R_EARTH_POLAR
from satflight.eclipse import shadow_fraction
from satflight.ephemeris import sun_position
from satflight.frames import (
    ecef_to_eci_state,
    ecef_to_geodetic,
    eci_to_ecef_state,
    geodetic_to_ecef,
    look_angles,
)
from satflight.timeutil import UTC, gmst, julian_date


def test_julian_date_j2000():
    assert julian_date(datetime(2000, 1, 1, 12, tzinfo=UTC)) == pytest.approx(2451545.0)


def test_gmst_vallado_example_3_5():
    # Vallado Example 3-5: 1992 Aug 20, 12:14 UT1 -> GMST 152.578787810 deg
    jd = julian_date(datetime(1992, 8, 20, 12, 14, tzinfo=UTC))
    assert math.degrees(gmst(jd)) == pytest.approx(152.578787810, abs=1e-6)


def test_geodetic_round_trip_and_anchors():
    assert np.allclose(geodetic_to_ecef(0, 0, 0), [R_EARTH, 0, 0])
    assert np.allclose(geodetic_to_ecef(math.pi / 2, 0, 0), [0, 0, R_EARTH_POLAR], atol=1e-9)
    lats = np.radians([-89.9, -45, 0, 30, 60, 89.99])
    lons = np.radians([-179, -90, 0, 45, 120, 179])
    alts = np.array([0.0, 400.0, 35786.0, 10.0, 1000.0, 5.0])
    lat, lon, h = ecef_to_geodetic(geodetic_to_ecef(lats, lons, alts))
    assert np.allclose(lat, lats, atol=1e-11)
    assert np.allclose(lon, lons, atol=1e-11)
    assert np.allclose(h, alts, atol=1e-6)


def test_eci_ecef_state_round_trip_and_geo_is_stationary():
    r = np.array([42164.0, 0, 0])
    v = np.array([0, 42164.0 * 7.292115146706979e-5, 0])
    re, ve = eci_to_ecef_state(r, v, 0.3)
    assert np.allclose(ve, 0.0, atol=1e-12)
    r2, v2 = ecef_to_eci_state(re, ve, 0.3)
    assert np.allclose(r2, r) and np.allclose(v2, v)


def test_look_angles_overhead():
    site = geodetic_to_ecef(0.5, 1.0, 0.0)
    target = geodetic_to_ecef(0.5, 1.0, 500.0)
    az, el, rng = look_angles(site, 0.5, 1.0, target)
    assert math.degrees(el) == pytest.approx(90.0, abs=1e-6)
    assert rng == pytest.approx(500.0, abs=1e-6)


def test_sun_declination_at_solstice_and_equinox():
    for when, dec in ((datetime(2026, 6, 21, 8, 24, tzinfo=UTC), 23.44),
                      (datetime(2026, 3, 20, 14, 46, tzinfo=UTC), 0.0)):
        s = sun_position(julian_date(when))
        d = math.degrees(math.asin(s[2] / np.linalg.norm(s)))
        assert d == pytest.approx(dec, abs=0.02)
    s = sun_position(julian_date(datetime(2026, 1, 3, tzinfo=UTC)))
    assert np.linalg.norm(s) == pytest.approx(0.9833 * AU, rel=5e-4)   # perihelion


def test_shadow_fraction():
    sun = np.array([AU, 0, 0])
    lit = shadow_fraction([[7000, 0, 0]], sun)
    dark = shadow_fraction([[-7000, 0, 0]], sun)
    side = shadow_fraction([[0, 7000, 0]], sun)
    assert lit[0] == 1.0 and dark[0] == 0.0 and side[0] == 1.0
    # sweeping across the terminator passes through penumbra monotonically
    ys = np.linspace(6300, 6500, 200)
    frac = shadow_fraction(np.stack([-7000 * np.ones_like(ys), ys, 0 * ys], axis=1), sun)
    assert frac[0] == 0.0 and frac[-1] == 1.0
    assert np.all(np.diff(frac) >= -1e-12)
    assert np.any((frac > 0) & (frac < 1))
