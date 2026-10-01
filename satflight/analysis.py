"""Derived orbital quantities: J2 secular rates, beta angle, coverage
footprints, orbit classification and an analytic J2-secular propagator."""

from __future__ import annotations

import math

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH, R_GEO
from .elements import coe2rv, kepler_propagate, mean_to_true, rv2coe


def j2_secular_rates(a, e, i, mu: float = MU_EARTH):
    """Secular drift (rad/s) of RAAN, argument of perigee and mean anomaly
    caused by J2 (Vallado eq. 9-41)."""
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    n = np.sqrt(mu / np.abs(a) ** 3)
    p = a * (1.0 - e * e)
    k = n * J2 * (R_EARTH / p) ** 2
    si2 = np.sin(i) ** 2
    raan_dot = -1.5 * k * np.cos(i)
    argp_dot = 0.75 * k * (4.0 - 5.0 * si2)
    m_dot = 0.75 * k * np.sqrt(np.maximum(1.0 - e * e, 0.0)) * (2.0 - 3.0 * si2)
    return raan_dot, argp_dot, m_dot


def _j2_sp_shape(e, i, argp, nu):
    """``a * da`` / (3/2 J2 Re^2): the shape of the J2 short-period term in the
    semi-major axis, which depends on a only through its 1/a prefactor."""
    e = np.asarray(e, dtype=float)
    q = 1.0 - e * e
    ar3 = ((1.0 + e * np.cos(nu)) / q) ** 3          # (a / r)^3
    s2 = np.sin(i) ** 2
    return ((2.0 / 3.0) * (1.0 - 1.5 * s2) * (ar3 - q ** -1.5)
            + s2 * ar3 * np.cos(2.0 * (argp + nu)))


def mean_to_osculating_a(a_mean, e, i, argp, nu):
    """Osculating semi-major axis at true anomaly ``nu`` of an orbit whose mean
    (orbit-averaged) semi-major axis is ``a_mean``: the first-order J2
    short-period term of Kozai (1959),

        a_osc - a_mean = 3/2 J2 Re^2 / a [2/3 (1 - 3/2 sin^2 i)((a/r)^3 - (1-e^2)^-3/2)
                                          + sin^2 i (a/r)^3 cos 2(argp + nu)]

    It swings by about +-6 km over one revolution in LEO. The other elements
    are left as given (their short-period terms do not change the period)."""
    a_mean = np.asarray(a_mean, dtype=float)
    return a_mean + 1.5 * J2 * R_EARTH ** 2 * _j2_sp_shape(e, i, argp, nu) / a_mean


def osculating_to_mean_a(a_osc, e, i, argp, nu):
    """Inverse of :func:`mean_to_osculating_a`, exactly: with the term written
    as K / a_mean, a_mean solves a_mean^2 - a_osc a_mean + K = 0."""
    a_osc = np.asarray(a_osc, dtype=float)
    k = 1.5 * J2 * R_EARTH ** 2 * _j2_sp_shape(e, i, argp, nu)
    return 0.5 * (a_osc + np.sqrt(np.maximum(a_osc * a_osc - 4.0 * k, 0.0)))


def j2_mean_propagate(r, v, dt, mu: float = MU_EARTH):
    """Analytic propagation with the J2 secular drifts. The period comes from
    the mean semi-major axis (the osculating one minus its short-period term):
    taking the osculating value as mean puts a LEO satellite hundreds of km
    off after a day. The output state carries the short-period term again, so
    re-reading it gives back the same mean value however the run is chunked.
    Open orbits fall back to Kepler."""
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)
    el = rv2coe(r, v, mu)
    e = np.atleast_1d(el.e)
    closed = e < 1.0
    r_out, v_out = kepler_propagate(r, v, dt, mu)
    if np.any(closed):
        ec = e[closed]
        inc = np.atleast_1d(el.i)[closed]
        argp0 = np.atleast_1d(el.argp)[closed]
        a = osculating_to_mean_a(np.atleast_1d(el.a)[closed], ec, inc, argp0,
                                 np.atleast_1d(el.nu)[closed])
        rd, wd, md = j2_secular_rates(a, ec, inc, mu)
        n = np.sqrt(mu / a ** 3)
        M = np.atleast_1d(el.M)[closed] + (n + md) * dt
        raan = np.atleast_1d(el.raan)[closed] + rd * dt
        argp = argp0 + wd * dt
        nu = np.atleast_1d(mean_to_true(M, ec))
        a_osc = mean_to_osculating_a(a, ec, inc, argp, nu)
        rc, vc = coe2rv(a_osc, ec, inc, raan, argp, nu, mu)
        r_out[closed] = rc
        v_out[closed] = vc
    return r_out, v_out


def beta_angle(r, v, r_sun) -> float:
    """Angle (rad) between the orbit plane and the Sun direction."""
    h = np.cross(r, v)
    h = h / np.linalg.norm(h)
    s = np.asarray(r_sun) / np.linalg.norm(r_sun)
    return float(np.arcsin(np.clip(np.dot(h, s), -1, 1)))


def coverage_half_angle(alt_km: float, min_elev: float = 0.0) -> float:
    """Earth-central angle (rad) from the sub-satellite point to the edge of
    the region that sees the satellite above ``min_elev`` (rad)."""
    rs = R_EARTH + max(alt_km, 1e-3)
    eta = math.asin(min(1.0, R_EARTH * math.cos(min_elev) / rs))   # nadir angle
    return max(0.0, math.pi / 2 - min_elev - eta)


def footprint(lat0: float, lon0: float, half_angle: float, n: int = 90):
    """Latitude/longitude (rad) of the small circle of the given Earth-central
    ``half_angle`` around a point (spherical Earth)."""
    az = np.linspace(0, 2 * np.pi, n)
    lat = np.arcsin(np.sin(lat0) * np.cos(half_angle)
                    + np.cos(lat0) * np.sin(half_angle) * np.cos(az))
    lon = lon0 + np.arctan2(np.sin(az) * np.sin(half_angle) * np.cos(lat0),
                            np.cos(half_angle) - np.sin(lat0) * np.sin(lat))
    lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi
    return lat, lon


def classify(el) -> str:
    """Human-readable orbit regime from an :class:`Elements` (single orbit)."""
    if el.e >= 1.0:
        return "escape (hyperbolic)" if el.e > 1.0 + 1e-6 else "escape (parabolic)"
    hp = el.rp - R_EARTH
    ha = el.ra - R_EARTH
    inc = math.degrees(el.i)
    if hp < 0:
        return "suborbital (intersects Earth)"
    if abs(el.a - R_GEO) < 200 and el.e < 0.01:
        return "geostationary" if inc < 1.0 else "geosynchronous"
    if el.e > 0.25:
        if 62.0 < inc < 64.9 and 2.5e4 < el.a < 2.8e4:
            return "Molniya (HEO)"
        if abs(el.a - R_GEO) < 500:
            return "Tundra (HEO)"
        return "highly elliptical (HEO)"
    if ha < 2000:
        polar = " polar" if 80 < inc < 100 else ""
        return f"low Earth orbit{polar}"
    if ha < 35000:
        return "medium Earth orbit"
    return "high Earth orbit"


def line_of_sight(r1, r2, radius: float = R_EARTH) -> bool:
    """True when the straight segment r1-r2 clears a sphere of ``radius``."""
    r1 = np.asarray(r1, dtype=float)
    d = np.asarray(r2, dtype=float) - r1
    dd = float(np.dot(d, d))
    if dd == 0:
        return True
    t = float(np.clip(-np.dot(r1, d) / dd, 0.0, 1.0))
    return float(np.linalg.norm(r1 + t * d)) > radius


__all__ = ["j2_secular_rates", "j2_mean_propagate", "mean_to_osculating_a",
           "osculating_to_mean_a", "beta_angle", "coverage_half_angle", "footprint",
           "classify", "line_of_sight"]
