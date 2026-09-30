"""Orbital elements, anomalies and universal-variable Kepler propagation,
checked against Vallado's worked examples and round trips."""

import math

import numpy as np
import pytest

from satflight.constants import MU_EARTH
from satflight.elements import (
    coe2rv,
    conic_points,
    kepler_propagate,
    mean_to_true,
    rv2coe,
    time_to_true_anomaly,
    true_to_mean,
)

D = np.radians


def test_rv2coe_vallado_example_2_5():
    # Vallado, Fundamentals of Astrodynamics, Example 2-5
    r = [6524.834, 6862.875, 6448.296]
    v = [4.901327, 5.533756, -1.976341]
    el = rv2coe(r, v)
    assert el.p == pytest.approx(11067.790, abs=0.01)
    assert el.a == pytest.approx(36127.343, abs=0.05)
    assert el.e == pytest.approx(0.832853, abs=1e-5)
    assert math.degrees(el.i) == pytest.approx(87.870, abs=1e-3)
    assert math.degrees(el.raan) == pytest.approx(227.89, abs=1e-2)
    assert math.degrees(el.argp) == pytest.approx(53.38, abs=1e-2)
    assert math.degrees(el.nu) == pytest.approx(92.335, abs=1e-3)


@pytest.mark.parametrize("a,e,i,raan,argp,nu", [
    (7000, 0.01, 51.6, 30, 40, 50),
    (26554, 0.72, 63.4, 250, 270, 10),
    (42164, 0.0, 0.0, 0, 0, 123),          # circular equatorial
    (8000, 0.0, 98.0, 200, 0, 300),        # circular inclined
    (9000, 0.2, 0.0, 0, 77, 20),           # elliptic equatorial
    (9000, 0.2, 180.0, 0, 77, 20),         # retrograde equatorial
    (-20000, 1.5, 30, 10, 20, 30),         # hyperbola
])
def test_coe_rv_round_trip(a, e, i, raan, argp, nu):
    r, v = coe2rv(a, e, D(i), D(raan), D(argp), D(nu))
    el = rv2coe(r, v)
    r2, v2 = coe2rv(el.a, el.e, el.i, el.raan, el.argp, el.nu)
    assert np.allclose(r, r2, atol=1e-6)
    assert np.allclose(v, v2, atol=1e-9)
    assert el.e == pytest.approx(e, abs=1e-12)


def test_rv2coe_vectorised_matches_scalar():
    rs, vs = coe2rv(np.array([7000, 12000.0]), np.array([0.1, 0.3]), D(np.array([10, 70.0])),
                    D(np.array([5, 50.0])), D(np.array([15, 150.0])), D(np.array([25, 250.0])))
    el = rv2coe(rs, vs)
    for k in range(2):
        e1 = rv2coe(rs[k], vs[k])
        assert el.raan[k] == pytest.approx(e1.raan)
        assert el.nu[k] == pytest.approx(e1.nu)


def test_kepler_vallado_example_2_4():
    # Vallado Example 2-4: propagate 40 minutes
    r0 = np.array([1131.340, -2282.343, 6672.423])
    v0 = np.array([-5.64305, 4.30333, 2.42879])
    r, v = kepler_propagate(r0, v0, 40 * 60.0)
    assert np.allclose(r, [-4219.7527, 4363.0292, -3958.7666], atol=2e-3)
    assert np.allclose(v, [3.689866, -1.916735, -6.112511], atol=2e-6)


@pytest.mark.parametrize("a,e", [(7000, 0.001), (26554, 0.72), (-30000, 1.3), (-8000, 3.0)])
def test_kepler_round_trip_and_invariants(a, e):
    r0, v0 = coe2rv(a, e, D(40), D(20), D(60), D(5))
    for dt in (-5000.0, 1234.5, 86400.0):
        r, v = kepler_propagate(r0, v0, dt)
        rb, vb = kepler_propagate(r, v, -dt)
        assert np.allclose(rb, r0, atol=1e-5)
        energy0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
        energy = 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)
        assert energy == pytest.approx(energy0, rel=1e-10)
        assert np.allclose(np.cross(r, v), np.cross(r0, v0), rtol=1e-10)


def test_anomaly_round_trips():
    for e in (0.0, 0.1, 0.7, 0.99):
        for nu in np.linspace(0, 2 * np.pi, 13)[:-1]:
            assert mean_to_true(true_to_mean(nu, e), e) == pytest.approx(nu % (2 * np.pi), abs=1e-9)
    for e in (1.2, 3.0):
        for nu in (-1.0, 0.0, 0.5, 1.5):
            back = mean_to_true(true_to_mean(nu, e), e)
            assert math.remainder(back - nu, 2 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_time_to_apoapsis_is_half_period():
    r, v = coe2rv(10000, 0.3, D(10), 0, 0, 0)
    el = rv2coe(r, v)
    assert time_to_true_anomaly(r, v, math.pi) == pytest.approx(el.period / 2, rel=1e-10)


def test_conic_points_lie_on_orbit():
    r, v = coe2rv(np.array([9000.0, -20000.0]), np.array([0.4, 1.4]), D(np.array([30, 60.0])),
                  0.2, 0.3, np.array([0.1, 0.2]))
    pts = conic_points(r, v, n=50)
    el = rv2coe(r, v)
    for k in range(2):
        rr = np.linalg.norm(pts[k], axis=1)
        h = np.cross(r[k], v[k])
        # every point is in the orbit plane and between periapsis and the cap
        assert np.allclose(pts[k] @ h / np.linalg.norm(h), 0.0, atol=1e-6)
        assert rr.min() >= el.rp[k] - 1e-6
