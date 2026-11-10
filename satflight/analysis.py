"""Derived orbital quantities: J2 secular rates, beta angle, coverage
footprints, orbit classification and an analytic J2-secular propagator."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH, R_GEO
from .elements import coe2rv, kepler_propagate, mean_to_true, rv2coe
from .forces import ForceModel
from .frames import geodetic_height
from .integrators import Propagator
from .maneuvers import sun_synchronous_inclination


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


def osculating_to_mean_i(i_osc, a, e, argp, nu):
    """Mean inclination from the osculating one: Kozai's (1959) first-order J2
    short-period term, removed,

        i_osc - i_mean = 3/8 J2 (Re/p)^2 sin 2i [cos 2(argp + nu) + e cos(2 argp + nu)
                                                + e/3 cos(2 argp + 3 nu)]

    It swings by ~0.02 deg around each revolution in LEO, which is ~1e-3 of the
    J2 node drift at 51.6 deg if left in."""
    i_osc = np.asarray(i_osc, dtype=float)
    e = np.asarray(e, dtype=float)
    p = np.abs(a) * (1.0 - e * e)
    k = 0.375 * J2 * (R_EARTH / p) ** 2 * np.sin(2.0 * i_osc)
    return i_osc - k * (np.cos(2.0 * (argp + nu)) + e * np.cos(2.0 * argp + nu)
                        + e / 3.0 * np.cos(2.0 * argp + 3.0 * nu))


def mean_to_osculating_i(i_mean, a, e, argp, nu):
    """Inverse of :func:`osculating_to_mean_i` (three fixed-point passes, exact to
    rounding: the term is ~3e-4 rad and changes with i by less than that)."""
    i = np.asarray(i_mean, dtype=float)
    for _ in range(3):
        i = i + (i_mean - osculating_to_mean_i(i, a, e, argp, nu))
    return i


SSO_RATE = 2 * math.pi / (365.2421897 * 86400.0)   # rad/s: the mean Sun's eastward drift


@lru_cache(maxsize=256)
def tuned_sso_inclination(a: float, e: float, argp: float, j3: bool, j4: bool) -> float:
    """Mean inclination (rad) whose node drift, flown with J2 (and J3, J4 when
    given) in full, matches the mean Sun. The first-order J2 formula leaves an
    orbit drifting 0.0008 deg/day from it in a J2 Cowell run (J2's second-order
    term) and 0.003 deg/day with J3 and J4 on, 1 to 5 minutes of LTAN a year.

    The drift is measured from one ascending node to the next: the short-period
    terms repeat there, so the node's change over that revolution is all
    secular. The inclination is then found by the secant method."""
    model = ForceModel(j2=True, j3=j3, j4=j4)

    def f(t, y):
        return np.concatenate([y[:, 3:], model.acceleration(y[:, :3], y[:, 3:],
                                                            np.zeros(len(y)))], axis=1)

    def drift(i_mean):
        nu = (-argp) % (2 * math.pi)                 # start at the ascending node
        i = float(mean_to_osculating_i(i_mean, a, e, argp, nu))
        a_osc = float(mean_to_osculating_a(a, e, i, argp, nu))
        r, v = coe2rv(a_osc, e, i, 0.0, argp, nu)
        y0 = np.concatenate([r, v])[None]
        rd, wd, md = (float(x) for x in j2_secular_rates(a, e, i_mean))
        t_a = 2 * math.pi / (math.sqrt(MU_EARTH / a ** 3) + md + wd) - 120.0
        t_a, y_a = Propagator("dopri5", rtol=1e-11, atol=1e-9, h_max=60.0).integrate(
            f, 0.0, y0, t_a)

        def at(t):                                   # flown on from two minutes short
            return Propagator("dopri5", rtol=1e-11, atol=1e-9, h_max=60.0).integrate(
                f, t_a, y_a, t)[1]

        if not (y_a[0, 2] < 0 < y_a[0, 5]):
            raise ValueError("the next ascending node is not where J2 puts it")
        t_now, y = t_a, y_a
        for _ in range(8):                           # Newton on z: the node itself
            dt = -y[0, 2] / y[0, 5]
            t_now += dt
            y = at(t_now)
            if abs(dt) < 1e-7:
                break
        raan = float(rv2coe(y[0, :3], y[0, 3:]).raan)
        return math.remainder(raan, 2 * math.pi) / t_now

    i0 = float(sun_synchronous_inclination(a, e))
    i1 = i0 + 1e-4
    d0, d1 = drift(i0) - SSO_RATE, drift(i1) - SSO_RATE
    for _ in range(6):
        if d1 == d0:
            break
        i0, i1 = i1, i1 - d1 * (i1 - i0) / (d1 - d0)
        d0, d1 = d1, drift(i1) - SSO_RATE
        if abs(d1) < 1e-14:                          # rad/s: 7e-8 deg/day
            return i1
    if abs(d1) > 1e-12:
        raise ValueError("no sun-synchronous inclination found for this orbit")
    return i1


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


def lowest_height(el, samples: int = 721) -> float:
    """Lowest height (km) above the WGS-84 ellipsoid anywhere on a closed orbit.
    This, not r_p - R_eq, says whether the orbit clears the surface: the
    ellipsoid is 21 km lower at the poles than at the equator."""
    if el.e >= 1.0:
        raise ValueError("lowest_height needs a closed orbit")
    nu = np.linspace(0.0, 2.0 * np.pi, samples)
    pos, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, nu)
    rp, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, 0.0)
    return float(min(np.min(geodetic_height(pos)), geodetic_height(rp)))


def classify(el) -> str:
    """Human-readable orbit regime from an :class:`Elements` (single orbit).
    The altitude bands use r - R_eq (the orbit's shape); whether it hits the
    Earth uses the real surface (:func:`lowest_height`)."""
    if el.e >= 1.0:
        return "escape (hyperbolic)" if el.e > 1.0 + 1e-6 else "escape (parabolic)"
    ha = el.ra - R_EARTH
    inc = math.degrees(el.i)
    if el.rp < R_EARTH and lowest_height(el) < 0:       # above R_eq it clears everywhere
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
           "osculating_to_mean_a", "osculating_to_mean_i", "beta_angle", "coverage_half_angle",
           "footprint",
           "lowest_height", "classify", "line_of_sight", "mean_to_osculating_i",
           "tuned_sso_inclination", "SSO_RATE"]
