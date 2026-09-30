"""Force terms (zonal gradients, drag direction) and the integrators: tableau
consistency, accuracy against Kepler, energy conservation, early stops."""

import math

import numpy as np
import pytest

from satflight import integrators
from satflight.constants import MU_EARTH
from satflight.elements import coe2rv, kepler_propagate
from satflight.forces import (
    ForceModel,
    accel_drag,
    accel_j2,
    accel_j3,
    accel_j4,
    accel_point_mass,
    conservative_energy,
    zonal_potential,
)
from satflight.integrators import Propagator


def _num_grad(fun, r, h=1e-3):
    g = np.zeros(3)
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        g[k] = (fun(r + d) - fun(r - d)) / (2 * h)
    return g


@pytest.mark.parametrize("r", [[7000.0, 1200.0, 3000.0], [-5000.0, 4000.0, -6000.0],
                               [100.0, -200.0, 7200.0]])
def test_zonal_accelerations_are_potential_gradients(r):
    r = np.array(r)
    assert np.allclose(_num_grad(lambda x: zonal_potential(x, False, False, False), r, 0.5),
                       accel_point_mass(r), rtol=1e-6)

    # perturbing potential of each zonal term alone (no large central term to cancel)
    def pert(flags):
        return lambda x: zonal_potential(x, *flags) - MU_EARTH / np.linalg.norm(x)
    for flags, acc in (((True, False, False), accel_j2), ((False, True, False), accel_j3),
                       ((False, False, True), accel_j4)):
        assert np.allclose(acc(r), _num_grad(pert(flags), r, 0.5), rtol=1e-5, atol=1e-16)


def test_drag_opposes_relative_velocity():
    r = np.array([[6378.137 + 300, 0, 0]])
    v = np.array([[0, 7.7, 0]])
    a = accel_drag(r, v, np.array([0.02]))
    v_rel = v - np.cross([0, 0, 7.292115e-5], r)
    assert np.dot(a[0], v_rel[0]) < 0
    assert np.allclose(np.cross(a[0], v_rel[0]), 0, atol=1e-18)
    # magnitude sanity: ~1e-9..1e-7 km/s^2 at 300 km for Cd*A/m=0.02
    assert 1e-10 < np.linalg.norm(a) < 1e-7


def test_dopri5_tableau_consistency():
    for i, row in enumerate(integrators._A[1:], start=1):
        assert sum(row) == pytest.approx(integrators._C[i], abs=1e-14)
    assert integrators._B5.sum() == pytest.approx(1.0)
    assert integrators._B4.sum() == pytest.approx(1.0)


def _two_body(t, y):
    return np.concatenate([y[:, 3:], accel_point_mass(y[:, :3])], axis=1)


def test_dopri5_matches_analytic_kepler():
    r0, v0 = coe2rv(np.array([7000.0, 26554.0]), np.array([0.001, 0.72]),
                    np.radians([51.6, 63.4]), np.radians([30, 250]),
                    np.radians([10, 270]), np.radians([0, 0]))
    y0 = np.concatenate([r0, v0], axis=1)
    prop = Propagator("dopri5", rtol=1e-12, atol=1e-9, h_max=300.0)
    t, y = prop.integrate(_two_body, 0.0, y0, 86400.0)
    assert t == 86400.0
    r_ref, v_ref = kepler_propagate(r0, v0, 86400.0)
    assert np.max(np.linalg.norm(y[:, :3] - r_ref, axis=1)) < 1e-3   # < 1 m after a day
    assert np.max(np.linalg.norm(y[:, 3:] - v_ref, axis=1)) < 1e-6


@pytest.mark.parametrize("method,tol", [("dopri5", 1e-9), ("rk4", 1e-7), ("leapfrog", 1e-4)])
def test_energy_conservation_ten_orbits(method, tol):
    r0, v0 = coe2rv(7000.0, 0.05, 0.5, 0.1, 0.2, 0.0)
    y0 = np.concatenate([r0, v0])[None, :]
    period = 2 * math.pi * math.sqrt(7000.0 ** 3 / MU_EARTH)
    prop = Propagator(method, rtol=1e-11, atol=1e-9, h_max=60.0, h_fixed=5.0)
    _, y = prop.integrate(_two_body, 0.0, y0, 10 * period)
    e0 = 0.5 * v0 @ v0 - MU_EARTH / np.linalg.norm(r0)
    e1 = 0.5 * y[0, 3:] @ y[0, 3:] - MU_EARTH / np.linalg.norm(y[0, :3])
    assert abs((e1 - e0) / e0) < tol


def test_energy_with_zonals_is_conserved():
    model = ForceModel(j2=True, j3=True, j4=True)
    r0, v0 = coe2rv(7200.0, 0.02, 1.0, 0.3, 0.4, 0.5)
    y0 = np.concatenate([r0, v0])[None, :]

    def f(t, y):
        a = model.acceleration(y[:, :3], y[:, 3:], np.zeros(1))
        return np.concatenate([y[:, 3:], a], axis=1)
    prop = Propagator("dopri5", rtol=1e-12, atol=1e-10, h_max=60.0)
    _, y = prop.integrate(f, 0.0, y0, 86400.0)
    e0 = conservative_energy(y0[:, :3], y0[:, 3:], model)[0]
    e1 = conservative_energy(y[:, :3], y[:, 3:], model)[0]
    assert abs((e1 - e0) / e0) < 1e-10


def test_callback_can_stop_integration():
    y0 = np.concatenate(coe2rv(7000.0, 0.0, 0.0, 0.0, 0.0, 0.0))[None, :]
    prop = Propagator("dopri5", h_max=10.0)
    t, _ = prop.integrate(_two_body, 0.0, y0, 1000.0, lambda t, y: t >= 50.0)
    assert 50.0 <= t < 100.0
