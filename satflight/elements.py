"""Classical orbital elements, anomalies and analytic two-body propagation.

Algorithms follow Vallado, *Fundamentals of Astrodynamics and Applications*
(4th ed.): RV2COE (Alg. 9), COE2RV (Alg. 10) and the universal-variable
KEPLER solution (Alg. 8), here with a Laguerre-Conway root finder for
robustness on every conic. Everything is vectorised over leading axes.

Angle conventions: all angles are radians in [0, 2pi). For orbits where an
element is undefined the conventional substitutes are used so that
``coe2rv(rv2coe(r, v))`` round-trips exactly:

* equatorial orbits: RAAN = 0 and the node line is taken as +X, so the
  argument of perigee becomes the longitude of perigee;
* circular orbits: argument of perigee = 0, so the true anomaly becomes
  the argument of latitude (or the true longitude if also equatorial).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH

TWO_PI = 2.0 * np.pi
ECC_TOL = 1e-10       # below this an orbit is treated as circular
INC_TOL = 1e-10       # sin(i) below this -> equatorial


def _norm(v):
    return np.linalg.norm(v, axis=-1)


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _angle_about(axis, frm, to):
    """Signed angle from unit vector ``frm`` to vector ``to`` about ``axis``."""
    return np.mod(np.arctan2(_dot(axis, np.cross(frm, to)), _dot(frm, to)), TWO_PI)


def _scalarise(x):
    return float(x) if np.ndim(x) == 0 else x


# --- Stumpff functions ------------------------------------------------------

def stumpff(psi):
    """Stumpff functions c2(psi), c3(psi), with series near psi = 0."""
    psi = np.asarray(psi, dtype=float)
    c2 = np.empty_like(psi)
    c3 = np.empty_like(psi)
    small = np.abs(psi) < 1e-3
    pos = (psi >= 1e-3)
    neg = (psi <= -1e-3)
    ps = psi[small]
    c2[small] = 0.5 - ps / 24.0 + ps * ps / 720.0 - ps ** 3 / 40320.0
    c3[small] = 1.0 / 6.0 - ps / 120.0 + ps * ps / 5040.0 - ps ** 3 / 362880.0
    sp = np.sqrt(psi[pos])
    c2[pos] = (1.0 - np.cos(sp)) / psi[pos]
    c3[pos] = (sp - np.sin(sp)) / sp ** 3
    sn = np.sqrt(np.minimum(-psi[neg], 700.0 ** 2))
    c2[neg] = (np.cosh(sn) - 1.0) / (-psi[neg])
    c3[neg] = (np.sinh(sn) - sn) / sn ** 3
    return c2, c3


# --- Elements -----------------------------------------------------------------

@dataclass
class Elements:
    """Osculating elements. Fields are floats for a single orbit or arrays."""
    a: np.ndarray          # semi-major axis, km (negative for hyperbolae, inf parabola)
    e: np.ndarray          # eccentricity
    i: np.ndarray          # inclination, rad
    raan: np.ndarray       # right ascension of the ascending node, rad
    argp: np.ndarray       # argument of perigee, rad
    nu: np.ndarray         # true anomaly, rad
    p: np.ndarray          # semi-latus rectum, km
    h: np.ndarray          # specific angular momentum magnitude, km^2/s
    energy: np.ndarray     # specific orbital energy, km^2/s^2
    M: np.ndarray          # mean anomaly, rad (hyperbolic/parabolic mean anomaly if open)
    u: np.ndarray          # argument of latitude (argp + nu), rad
    rp: np.ndarray         # periapsis radius, km
    ra: np.ndarray         # apoapsis radius, km (inf if open)
    period: np.ndarray     # s (inf if open)
    n: np.ndarray          # mean motion, rad/s

    def as_dict(self):
        return {k: _scalarise(v) for k, v in self.__dict__.items()}


def rv2coe(r, v, mu: float = MU_EARTH) -> Elements:
    """Position/velocity (km, km/s) -> osculating classical elements."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rm = _norm(r)
    vm = _norm(v)
    hv = np.cross(r, v)
    hm = _norm(hv)
    hm_safe = np.where(hm == 0.0, 1.0, hm)
    what = hv / hm_safe[..., None]

    nv = np.stack([-hv[..., 1], hv[..., 0], np.zeros_like(hm)], axis=-1)
    nm = _norm(nv)
    equatorial = nm < INC_TOL * hm_safe
    xhat = np.broadcast_to(np.array([1.0, 0.0, 0.0]), nv.shape)
    nhat = np.where(equatorial[..., None], xhat, nv / np.where(nm == 0, 1, nm)[..., None])

    rv = _dot(r, v)
    evec = ((vm * vm - mu / rm)[..., None] * r - rv[..., None] * v) / mu
    e = _norm(evec)
    circular = e < ECC_TOL
    ehat = np.where(circular[..., None], nhat, evec / np.where(e == 0, 1, e)[..., None])

    energy = 0.5 * vm * vm - mu / rm
    p = hm * hm / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(np.abs(1.0 - e) > 1e-12, p / (1.0 - e * e), np.inf)
    i = np.arccos(np.clip(hv[..., 2] / hm_safe, -1.0, 1.0))
    raan = np.where(equatorial, 0.0, np.mod(np.arctan2(nv[..., 1], nv[..., 0]), TWO_PI))
    argp = np.where(circular, 0.0, _angle_about(what, nhat, ehat))
    nu = _angle_about(what, ehat, r)
    u = np.mod(argp + nu, TWO_PI)

    M = true_to_mean(nu, e)
    rp = p / (1.0 + e)
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = np.where(e < 1.0, p / (1.0 - e), np.inf)
        ellipse = (e < 1.0) & (a > 0)
        n = np.where(ellipse, np.sqrt(mu / np.abs(a) ** 3), np.sqrt(mu / np.abs(a) ** 3))
        period = np.where(ellipse, TWO_PI / n, np.inf)

    return Elements(*(_scalarise(x) for x in
                      (a, e, i, raan, argp, nu, p, hm, energy, M, u, rp, ra, period, n)))


def perifocal_axes(i, raan, argp):
    """Unit vectors P (periapsis) and Q (90 deg ahead in-plane) in ECI."""
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    P = np.stack([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si], axis=-1)
    Q = np.stack([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si], axis=-1)
    return P, Q


def coe2rv(a, e, i, raan, argp, nu, mu: float = MU_EARTH, p=None):
    """Classical elements -> ECI position and velocity.

    ``p`` (semi-latus rectum) may be given instead of ``a``; it is required for
    an exactly parabolic orbit.
    """
    e = np.asarray(e, dtype=float)
    if p is None:
        p = np.asarray(a, dtype=float) * (1.0 - e * e)
    p = np.asarray(p, dtype=float)
    cn, sn = np.cos(nu), np.sin(nu)
    rr = p / (1.0 + e * cn)
    P, Q = perifocal_axes(i, raan, argp)
    sq = np.sqrt(mu / p)
    r = (rr * cn)[..., None] * P + (rr * sn)[..., None] * Q
    v = (-sq * sn)[..., None] * P + (sq * (e + cn))[..., None] * Q
    return r, v


# --- Anomalies -----------------------------------------------------------------

def true_to_mean(nu, e):
    """True anomaly -> mean anomaly for any conic (elliptic M, hyperbolic Mh,
    parabolic Barker M = D + D^3/3)."""
    nu = np.asarray(nu, dtype=float)
    e = np.asarray(e, dtype=float)
    nu, e = np.broadcast_arrays(nu, e)
    out = np.empty(nu.shape)
    ell = e < 1.0 - 1e-9
    hyp = e > 1.0 + 1e-9
    par = ~(ell | hyp)
    if np.any(ell):
        ee = e[ell]
        E = np.arctan2(np.sqrt(1.0 - ee * ee) * np.sin(nu[ell]), ee + np.cos(nu[ell]))
        out[ell] = np.mod(E - ee * np.sin(E), TWO_PI)
    if np.any(hyp):
        ee = e[hyp]
        x = np.sqrt((ee - 1.0) / (ee + 1.0)) * np.tan(nu[hyp] / 2.0)
        F = 2.0 * np.arctanh(np.clip(x, -1 + 1e-15, 1 - 1e-15))
        out[hyp] = ee * np.sinh(F) - F
    if np.any(par):
        D = np.tan(nu[par] / 2.0)
        out[par] = D + D ** 3 / 3.0
    return _scalarise(out)


def mean_to_eccentric(M, e, tol: float = 1e-14, max_iter: int = 50):
    """Solve Kepler's equation M = E - e sin E (elliptic) by Newton iteration."""
    M = np.mod(np.asarray(M, dtype=float), TWO_PI)
    e = np.asarray(e, dtype=float)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        dE = f / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return _scalarise(E)


def mean_to_hyperbolic(M, e, tol: float = 1e-14, max_iter: int = 80):
    """Solve M = e sinh F - F for the hyperbolic anomaly F."""
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    F = np.arcsinh(M / e)
    for _ in range(max_iter):
        f = e * np.sinh(F) - F - M
        dF = f / (e * np.cosh(F) - 1.0)
        F = F - dF
        if np.all(np.abs(dF) < tol * np.maximum(1.0, np.abs(F))):
            break
    return _scalarise(F)


def mean_to_true(M, e):
    """Mean anomaly -> true anomaly (elliptic or hyperbolic)."""
    e = np.asarray(e, dtype=float)
    if np.all(e < 1.0):
        E = np.asarray(mean_to_eccentric(M, e))
        nu = np.arctan2(np.sqrt(1.0 - e * e) * np.sin(E), np.cos(E) - e)
        return _scalarise(np.mod(nu, TWO_PI))
    F = np.asarray(mean_to_hyperbolic(M, e))
    nu = 2.0 * np.arctan(np.sqrt((e + 1.0) / (e - 1.0)) * np.tanh(F / 2.0))
    return _scalarise(np.mod(nu, TWO_PI))


# --- Universal-variable Kepler propagation -----------------------------------

def kepler_propagate(r0, v0, dt, mu: float = MU_EARTH, tol: float = 1e-12,
                     max_iter: int = 60):
    """Exact two-body propagation of ``(N, 3)`` (or ``(3,)``) states by ``dt`` s.

    Works for elliptic, parabolic and hyperbolic orbits. ``dt`` may be a
    scalar or an ``(N,)`` array and may be negative.
    """
    single = np.ndim(r0) == 1
    r0 = np.atleast_2d(np.asarray(r0, dtype=float))
    v0 = np.atleast_2d(np.asarray(v0, dtype=float))
    dt = np.broadcast_to(np.asarray(dt, dtype=float), r0.shape[:1]).copy()

    sqmu = np.sqrt(mu)
    r0m = _norm(r0)
    v0m = _norm(v0)
    rv = _dot(r0, v0)
    alpha = 2.0 / r0m - v0m * v0m / mu            # 1/a

    ell = alpha > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(ell, TWO_PI / np.sqrt(mu * np.abs(alpha) ** 3), np.inf)
        dt = np.where(ell, np.fmod(dt, period), dt)

        chi = np.where(ell, sqmu * dt * alpha, sqmu * dt / r0m)
        hyp = alpha < -1e-12
        a_h = np.where(hyp, 1.0 / np.where(alpha == 0, 1, alpha), -1.0)
        sgn = np.sign(dt)
        arg = (-2.0 * mu * alpha * dt) / (rv + sgn * np.sqrt(-mu * a_h) * (1.0 - r0m * alpha))
        chi_h = sgn * np.sqrt(-a_h) * np.log(np.where(arg > 0, arg, 1.0))
        chi = np.where(hyp & (arg > 0), chi_h, chi)

    k = 5.0   # Laguerre-Conway order
    for _ in range(max_iter):
        psi = chi * chi * alpha
        c2, c3 = stumpff(psi)
        F = (rv / sqmu * chi * chi * c2 + (1.0 - r0m * alpha) * chi ** 3 * c3
             + r0m * chi - sqmu * dt)
        dF = chi * chi * c2 + rv / sqmu * chi * (1.0 - psi * c3) + r0m * (1.0 - psi * c2)
        ddF = rv / sqmu * (1.0 - psi * c2) + (1.0 - r0m * alpha) * chi * (1.0 - psi * c3)
        disc = np.sqrt(np.abs((k - 1.0) ** 2 * dF * dF - k * (k - 1.0) * F * ddF))
        denom = dF + np.sign(dF) * disc
        delta = k * F / np.where(denom == 0.0, 1.0, denom)
        chi = chi - delta
        if np.all(np.abs(delta) <= tol * np.maximum(1.0, np.abs(chi))):
            break

    psi = chi * chi * alpha
    c2, c3 = stumpff(psi)
    rm = chi * chi * c2 + rv / sqmu * chi * (1.0 - psi * c3) + r0m * (1.0 - psi * c2)
    f = 1.0 - chi * chi / r0m * c2
    g = dt - chi ** 3 / sqmu * c3
    gdot = 1.0 - chi * chi / rm * c2
    fdot = sqmu / (rm * r0m) * chi * (psi * c3 - 1.0)
    r = f[:, None] * r0 + g[:, None] * v0
    v = fdot[:, None] * r0 + gdot[:, None] * v0
    if single:
        return r[0], v[0]
    return r, v


# --- Geometry helpers for display ------------------------------------------------

def conic_points(r, v, n: int = 160, mu: float = MU_EARTH, r_max: float = 3.0e5):
    """Sample the osculating conic of each state: returns ``(N, n, 3)`` km.

    Ellipses are sampled uniformly in eccentric anomaly (even spacing along
    the curve); open orbits in true anomaly, truncated at ``r_max``.
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    hv = np.cross(r, v)
    hm = _norm(hv)
    rm = _norm(r)
    degenerate = hm < 1e-9 * np.maximum(rm, 1.0)
    hm_s = np.where(degenerate, 1.0, hm)
    W = hv / hm_s[:, None]
    evec = ((_dot(v, v) - mu / rm)[:, None] * r - _dot(r, v)[:, None] * v) / mu
    e = _norm(evec)
    P = np.where((e > 1e-9)[:, None], evec / np.where(e == 0, 1, e)[:, None], r / rm[:, None])
    Q = np.cross(W, P)
    p = hm * hm / mu
    closed = e < 1.0 - 1e-6

    # closed: eccentric anomaly parameter
    E = np.linspace(0.0, TWO_PI, n)
    a = np.where(closed, p / np.maximum(1.0 - e * e, 1e-12), 1.0)
    b = a * np.sqrt(np.maximum(1.0 - e * e, 0.0))
    xc = a[:, None] * (np.cos(E)[None, :] - e[:, None])
    yc = b[:, None] * np.sin(E)[None, :]

    # open: true anomaly limited by r_max
    rcap = np.maximum(r_max, 1.5 * rm)
    cos_lim = np.clip((p / rcap - 1.0) / np.maximum(e, 1e-12), -1.0, 1.0)
    nu_lim = np.arccos(cos_lim)
    nu_lim = np.minimum(nu_lim, np.arccos(np.clip(-1.0 / np.maximum(e, 1e-12), -1, 1)) - 1e-6)
    t = np.linspace(-1.0, 1.0, n)
    nus = nu_lim[:, None] * t[None, :]
    ro = p[:, None] / (1.0 + e[:, None] * np.cos(nus))
    xo = ro * np.cos(nus)
    yo = ro * np.sin(nus)

    x = np.where(closed[:, None], xc, xo)
    y = np.where(closed[:, None], yc, yo)
    pts = x[..., None] * P[:, None, :] + y[..., None] * Q[:, None, :]
    pts[degenerate] = r[degenerate][:, None, :]
    return pts


def circular_velocity(r, mu: float = MU_EARTH):
    return np.sqrt(mu / np.asarray(r, dtype=float))


def escape_velocity(r, mu: float = MU_EARTH):
    return np.sqrt(2.0 * mu / np.asarray(r, dtype=float))


def vis_viva(r, a, mu: float = MU_EARTH):
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def period_of(a, mu: float = MU_EARTH):
    return TWO_PI * np.sqrt(np.asarray(a, dtype=float) ** 3 / mu)


def semi_major_axis_for_period(T, mu: float = MU_EARTH):
    return (mu * (np.asarray(T, dtype=float) / TWO_PI) ** 2) ** (1.0 / 3.0)


def time_to_true_anomaly(r, v, nu_target, mu: float = MU_EARTH) -> float:
    """Two-body time (s, >= 0) until a single orbit next reaches ``nu_target``.

    Returns ``inf`` for an open orbit that never gets there.
    """
    el = rv2coe(r, v, mu)
    e = el.e
    if e < 1.0:
        M0 = el.M
        M1 = true_to_mean(nu_target, e)
        dM = np.mod(M1 - M0, TWO_PI)
        if dM < 1e-9:
            dM += TWO_PI
        return float(dM / el.n)
    # open orbit: only reachable if ahead of us on the branch
    nu_max = np.arccos(-1.0 / e)
    nu0 = el.nu if el.nu <= np.pi else el.nu - TWO_PI
    nut = nu_target if nu_target <= np.pi else nu_target - TWO_PI
    if not (-nu_max < nut < nu_max) or nut <= nu0:
        return float("inf")
    return float((true_to_mean(nut, e) - true_to_mean(nu0, e)) / el.n)
