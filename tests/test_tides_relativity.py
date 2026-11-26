"""Solid-Earth tides and relativity (forces "tides", "relativity").

Each term is held to an independent reference: the tide to the IERS 2010
route through normalized coefficients (Conventions eq. 6.6) and to the
gradient of its own potential; relativity to Einstein's perigee advance,
6 pi GM / (c^2 a (1 - e^2)) per revolution, measured on an integrated orbit.
"""

from __future__ import annotations

import numpy as np
import pytest

from satflight import forces
from satflight.constants import C_LIGHT, MU_EARTH, MU_MOON, MU_SUN, R_EARTH
from satflight.ephemeris import moon_position, sun_position
from satflight.forces import ForceModel
from satflight.integrators import Propagator

JD = 2460991.5 + 0.37          # 2025-11-12 20:52 UTC, the storm day of the arcs
RNG = np.random.default_rng(7)


def _points(n=6, r=7078.0):
    u = RNG.normal(size=(n, 3))
    return r * u / np.linalg.norm(u, axis=1, keepdims=True)


def _closed_potential(r, r_body, mu_body, k2):
    """k2 GM_b R^5 / (r_b^3 r^3) P2(cos psi): the textbook form."""
    rm = np.linalg.norm(r, axis=-1)
    bm = np.linalg.norm(r_body)
    c = r @ r_body / (rm * bm)
    return k2 * mu_body * R_EARTH ** 5 / (bm ** 3 * rm ** 3) * 0.5 * (3 * c * c - 1)


def _iers_potential(r, bodies, k2):
    """IERS 2010 eq. 6.6 (degree 2, one k2 for every order): the normalized
    coefficient changes from each body's place, then the degree-2 potential
    they make at r. Latitude/longitude in the same frame on both sides."""
    def p2(s):                                  # fully normalized P2m(sin phi)
        cph = np.sqrt(1 - s * s)
        return (np.sqrt(5) * 0.5 * (3 * s * s - 1), np.sqrt(15) * s * cph,
                np.sqrt(15) / 2 * cph * cph)

    dc = np.zeros(3)
    ds = np.zeros(3)
    for rb, mu in bodies:
        bm = np.linalg.norm(rb)
        pb = p2(rb[2] / bm)
        lam = np.arctan2(rb[1], rb[0])
        for m in range(3):
            f = k2 / 5 * mu / MU_EARTH * (R_EARTH / bm) ** 3 * pb[m]
            dc[m] += f * np.cos(m * lam)
            ds[m] += f * np.sin(m * lam)        # dC - i dS = f e^{-i m lam}
    rm = np.linalg.norm(r, axis=-1)
    pr = p2(r[..., 2] / rm)
    lam = np.arctan2(r[..., 1], r[..., 0])
    return MU_EARTH / rm * (R_EARTH / rm) ** 2 * sum(
        pr[m] * (dc[m] * np.cos(m * lam) + ds[m] * np.sin(m * lam)) for m in range(3))


def test_the_tide_potential_is_the_iers_coefficient_route():
    r = _points()
    rs, rmn = sun_position(JD), moon_position(JD)
    mine = forces.tide_potential(r, rs, rmn)
    iers = _iers_potential(r, [(rs, MU_SUN), (rmn, MU_MOON)], forces.K2_EARTH)
    np.testing.assert_allclose(mine, iers, rtol=1e-12)
    closed = (_closed_potential(r, rs, MU_SUN, forces.K2_EARTH)
              + _closed_potential(r, rmn, MU_MOON, forces.K2_EARTH))
    np.testing.assert_allclose(mine, closed, rtol=1e-12)


def test_the_tide_acceleration_is_its_potentials_gradient():
    r = _points()
    rs, rmn = sun_position(JD), moon_position(JD)
    a = forces.accel_tides(r, rs, rmn)
    h = 1e-3
    grad = np.zeros_like(r)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        grad[:, k] = (forces.tide_potential(r + e, rs, rmn)
                      - forces.tide_potential(r - e, rs, rmn)) / (2 * h)
    np.testing.assert_allclose(a, grad, rtol=1e-7, atol=1e-22)


def test_the_tide_under_the_moon_pulls_down_and_is_the_size_iers_gives():
    rmn = moon_position(JD)
    r = 7078.0 * rmn / np.linalg.norm(rmn)       # the Moon overhead
    no_sun = np.array([1e15, 0.0, 0.0])           # so far its tide is nothing
    a = forces.accel_tides(r[None], no_sun, rmn)[0]
    # -3 k2 GM_m R^5 / (r_m^3 r^4) along r: the bulge below pulls down
    want = -3 * forces.K2_EARTH * MU_MOON * R_EARTH ** 5 / (
        np.linalg.norm(rmn) ** 3 * 7078.0 ** 4)
    assert a @ r / 7078.0 == pytest.approx(want, rel=1e-6)
    assert 5e-11 < -want < 5e-10                 # km/s^2: a few 1e-7 m/s^2 at 700 km


def _lrl_angle(y):
    r, v = y[:3], y[3:]
    e = np.cross(v, np.cross(r, v)) / MU_EARTH - r / np.linalg.norm(r)
    return np.arctan2(e[1], e[0])


def test_relativity_advances_the_perigee_at_einsteins_rate():
    a_km, e = 7500.0, 0.2
    rp = a_km * (1 - e)
    y0 = np.array([rp, 0, 0, 0, np.sqrt(MU_EARTH * (1 + e) / rp), 0])
    period = 2 * np.pi * np.sqrt(a_km ** 3 / MU_EARTH)
    revs = 20

    def run(gr):
        # the simulator's own Dormand-Prince, held far tighter than a run's
        def f(_t, y):
            acc = forces.accel_point_mass(y[:, :3])
            if gr:
                acc = acc + forces.accel_relativity(y[:, :3], y[:, 3:])
            return np.concatenate([y[:, 3:], acc], axis=1)
        prop = Propagator("dopri5", rtol=1e-13, atol=1e-13, h_max=600.0)
        return prop.integrate(f, 0.0, y0[None].copy(), revs * period)[1][0]

    advance = _lrl_angle(run(True)) - _lrl_angle(run(False))
    einstein = 6 * np.pi * MU_EARTH / (C_LIGHT ** 2 * a_km * (1 - e * e)) * revs
    assert advance == pytest.approx(einstein, rel=1e-3)


def test_the_switches_add_their_terms_and_name_them():
    r = _points(2)
    v = RNG.normal(size=(2, 3))
    cdam = np.zeros(2)
    base = ForceModel(j2=False)
    both = ForceModel(j2=False, tides=True, relativity=True)
    got = both.acceleration(r, v, cdam, jd=JD) - base.acceleration(r, v, cdam, jd=JD)
    want = (forces.accel_tides(r, sun_position(JD), moon_position(JD))
            + forces.accel_relativity(r, v))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-17)   # minus the 8e-3 central pull
    assert both.label() == "2B+TIDES+GR"
    assert both.needs_time and not ForceModel(relativity=True).needs_time
    with pytest.raises(ValueError, match="date"):
        ForceModel(tides=True).acceleration(r, v, cdam)
    shown = both.breakdown(r[:1], v[:1], cdam[:1], jd=JD)
    assert shown["tides"][1] and shown["relativity"][1]


def test_the_switches_round_trip_through_a_scenario_dict():
    fm = ForceModel(tides=True, relativity=True)
    assert ForceModel.from_dict(fm.to_dict()) == fm
    assert ForceModel.from_dict({}).tides is False


def test_the_matlab_export_refuses_the_new_terms_until_ported():
    from satflight.matlab_export import unported_forces
    assert unported_forces(ForceModel(tides=True, relativity=True)) == [
        "solid-Earth tides", "relativity"]
    assert unported_forces(ForceModel()) == []
