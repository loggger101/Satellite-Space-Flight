"""Mean versus osculating semi-major axis (J2 short-period term, Kozai 1959):
the term itself, the j2mean propagator that relies on it, and formations
(Walker shells, copies along one orbit) that must share one mean value."""

import math

import numpy as np

from satflight.analysis import j2_mean_propagate, mean_to_osculating_a, osculating_to_mean_a
from satflight.constants import MU_EARTH, R_EARTH
from satflight.elements import coe2rv, kepler_propagate, rv2coe
from satflight.forces import accel_j2, accel_point_mass
from satflight.integrators import Propagator
from satflight.scenario import ConstellationSpec, spread_along_orbit, walker_states

ISS = (R_EARTH + 420.0, 0.0005, math.radians(51.6), math.radians(30), math.radians(270),
       math.radians(10))


def _j2(t, y):
    r = y[:, :3]
    return np.concatenate([y[:, 3:], accel_point_mass(r) + accel_j2(r)], axis=1)


def _cowell_j2(y0, t1, h_max=30.0, callback=None):
    return Propagator("dopri5", rtol=1e-12, atol=1e-9, h_max=h_max).integrate(
        _j2, 0.0, y0, t1, callback)[1]


def test_short_period_term_matches_the_orbit_average():
    a, e, i, raan, argp, nu = ISS
    y0 = np.concatenate(coe2rv(a, e, i, raan, argp, nu))[None]
    period = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    ts, aa = [0.0], [a]

    def record(t, y):
        ts.append(t)
        aa.append(float(rv2coe(y[0, :3], y[0, 3:]).a))
    _cowell_j2(y0, period, h_max=10.0, callback=record)
    average = np.trapezoid(aa, ts) / ts[-1]
    assert abs(osculating_to_mean_a(a, e, i, argp, nu) - average) < 0.1      # km; the term is ~6


def test_mean_and_osculating_invert_each_other():
    a = np.linspace(6700.0, 42000.0, 7)
    e = np.array([0.0, 0.001, 0.05, 0.2, 0.5, 0.72, 0.9])
    i, argp, nu = np.radians(63.4), np.radians(270.0), np.radians([0, 40, 90, 150, 200, 280, 350])
    back = osculating_to_mean_a(mean_to_osculating_a(a, e, i, argp, nu), e, i, argp, nu)
    assert np.allclose(back, a, rtol=0, atol=1e-9)


def test_j2mean_follows_the_j2_orbit():
    """Treating the osculating a as mean put the ISS ~800 km off after a day,
    farther than ignoring J2 altogether."""
    y0 = np.concatenate(coe2rv(*ISS))[None]
    ref = _cowell_j2(y0, 86400.0)
    r_mean, _ = j2_mean_propagate(y0[:, :3], y0[:, 3:], 86400.0)
    r_kep, _ = kepler_propagate(y0[:, :3], y0[:, 3:], 86400.0)
    assert np.linalg.norm(r_mean[0] - ref[0, :3]) < 30.0
    assert np.linalg.norm(r_kep[0] - ref[0, :3]) > 300.0


def test_j2mean_does_not_depend_on_chunking():
    y0 = np.concatenate(coe2rv(26554.0, 0.72, math.radians(63.4), 1.0, math.radians(270), 0.3))
    r, v = y0[None, :3], y0[None, 3:]
    for _ in range(720):
        r, v = j2_mean_propagate(r, v, 120.0)
    r1, _ = j2_mean_propagate(y0[None, :3], y0[None, 3:], 86400.0)
    assert np.linalg.norm(r[0] - r1[0]) < 1e-5


def _in_plane_gaps(y, per_plane):
    el = rv2coe(y[:, :3], y[:, 3:])
    u = np.degrees(np.mod(np.asarray(el.argp) + np.asarray(el.nu), 2 * np.pi))
    gaps = []
    for p in range(len(u) // per_plane):
        uu = np.sort(u[p * per_plane:(p + 1) * per_plane])
        gaps.append(np.diff(np.concatenate([uu, uu[:1] + 360.0])))
    return np.concatenate(gaps)


def test_walker_shell_keeps_its_spacing():
    c = ConstellationSpec("S", 550.0, 53.0, total=24, planes=3, phasing=1)
    y0 = np.array([np.concatenate([r, v]) for _, r, v in walker_states(c)])
    gaps = _in_plane_gaps(_cowell_j2(y0, 86400.0, h_max=60.0), 8)
    assert np.ptp(gaps) < 0.2                                       # degrees, from 45 each
    # the check is sensitive: equal osculating radii shear the shell by degrees a day
    el = rv2coe(y0[:, :3], y0[:, 3:])
    naive = np.array([np.concatenate(coe2rv(R_EARTH + 550.0, 0.0, el.i[k], el.raan[k], 0.0,
                                             el.argp[k] + el.nu[k])) for k in range(len(y0))])
    assert np.ptp(_in_plane_gaps(_cowell_j2(naive, 86400.0, h_max=60.0), 8)) > 2.0


def test_copies_along_an_orbit_share_the_mean_semi_major_axis():
    r, v = coe2rv(*ISS)
    states = spread_along_orbit(np.asarray(r), np.asarray(v), 5)
    assert np.allclose(states[0][0], r, atol=1e-8) and np.allclose(states[0][1], v, atol=1e-11)
    means = []
    for rr, vv in states:
        el = rv2coe(rr, vv)
        means.append(float(osculating_to_mean_a(el.a, el.e, el.i, el.argp, el.nu)))
    assert np.ptp(means) < 1e-6
