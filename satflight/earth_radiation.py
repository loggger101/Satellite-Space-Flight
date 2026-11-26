"""Sunlight the Earth reflects (albedo) and the heat it radiates (infrared),
pushing on a satellite.

Knocke's model (Knocke, Ries & Tapley 1988, AIAA 88-4292), as precise orbit
work for low satellites uses it: the albedo and emissivity of the ground vary
with latitude (a second-degree Legendre series) and the season (the first-
degree term turns once a year from 1981-12-22). Every patch of ground the
satellite can see is a Lambertian surface:

* it reflects ``a S cos(theta_sun)`` where the Sun is up (``S`` the solar
  flux at the Earth's distance);
* it radiates ``e S / 4`` day and night (the absorbed sunlight spread over the
  whole sphere).

A Lambertian surface of exitance ``M`` looks equally bright, ``M / pi``, from
every direction, so the sum runs over the directions the satellite sees the
Earth in: each ray down to the ground brings ``M / pi`` times its solid angle,
pushing along the ray back up. Over c that is a pressure on a cannonball of
``Cr A / m`` (as ``forces.accel_srp``). The rays are Gauss-Legendre nodes in
the cosine of the angle from straight down, out to the horizon, and even
steps round it; for an evenly glowing Earth the sum is then exact at any
height. The satellite's own shadow from the Sun does not dim the Earth's
light.
"""

from __future__ import annotations

import numpy as np

from .constants import AU, P_SRP, R_EARTH

KNOCKE_T0 = 2444960.5            # Julian date of 1981-12-22 0h, the seasonal term's peak
YEAR_DAYS = 365.25
# Knocke 1988: albedo a0 + a1 P1 + a2 P2 with a1 = c0 + c1 cos + c2 sin (season)
ALBEDO = dict(a0=0.34, c0=0.0, c1=0.10, c2=0.0, a2=0.29)
EMISSIVITY = dict(e0=0.68, k0=0.0, k1=-0.07, k2=0.0, e2=-0.18)

N_RINGS = 10      # Gauss-Legendre nodes from straight down to the horizon
N_SECTORS = 16    # even steps round it
_U, _UW = np.polynomial.legendre.leggauss(N_RINGS)
_PSI = (np.arange(N_SECTORS) + 0.5) * (2 * np.pi / N_SECTORS)


def _season(jd):
    w = 2 * np.pi * (np.asarray(jd, float) - KNOCKE_T0) / YEAR_DAYS
    return np.cos(w), np.sin(w)


def knocke_albedo(lat, jd):
    """Knocke's albedo at geocentric latitude(s) ``lat`` (rad) on Julian date ``jd``."""
    s = np.sin(lat)
    cw, sw = _season(jd)
    a1 = ALBEDO["c0"] + ALBEDO["c1"] * cw + ALBEDO["c2"] * sw
    return ALBEDO["a0"] + a1 * s + ALBEDO["a2"] * 0.5 * (3 * s * s - 1)


def knocke_emissivity(lat, jd):
    """Knocke's emissivity at geocentric latitude(s) ``lat`` (rad) on Julian date ``jd``."""
    s = np.sin(lat)
    cw, sw = _season(jd)
    e1 = EMISSIVITY["k0"] + EMISSIVITY["k1"] * cw + EMISSIVITY["k2"] * sw
    return EMISSIVITY["e0"] + e1 * s + EMISSIVITY["e2"] * 0.5 * (3 * s * s - 1)


def accel_earth_radiation(r, r_sun, cr_a_over_m, jd, albedo: bool = True,
                          infrared: bool = True, albedo_of=knocke_albedo,
                          emissivity=knocke_emissivity):
    """Acceleration (km/s^2, (N, 3)) of the Earth's reflected sunlight and its
    infrared on satellites at ``r`` ((N, 3) km, ECI of date) with ``Cr A / m``
    ``cr_a_over_m`` ((N,) m^2/kg); ``r_sun`` the geocentric Sun (km, (3,)).
    ``albedo_of`` / ``emissivity`` (latitude rad, jd) -> value replace
    Knocke's (the tests use even ones). On or below the ground (a rocket on
    its pad) there is no sky full of Earth to sum: those rows are zero."""
    r = np.atleast_2d(np.asarray(r, float))
    cram = np.asarray(cr_a_over_m, float).reshape(-1)
    rm = np.linalg.norm(r, axis=1)
    aloft = rm > R_EARTH
    if not np.all(aloft):
        out = np.zeros_like(r)
        if np.any(aloft):
            out[aloft] = accel_earth_radiation(r[aloft], r_sun, cram[aloft], jd, albedo,
                                               infrared, albedo_of, emissivity)
        return out
    z = r / rm[:, None]                                            # straight up
    pick = np.where(np.abs(z[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(z, pick)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(z, e1)
    # rays: cos(eta) from the horizon's, sqrt(1 - (R/r)^2), to 1 (straight down)
    q = R_EARTH / rm
    cmin = np.sqrt(1 - q * q)
    c = 0.5 * (1 - cmin)[:, None] * _U[None] + 0.5 * (1 + cmin)[:, None]    # (N, K)
    w = 0.5 * (1 - cmin)[:, None] * _UW[None] * (2 * np.pi / N_SECTORS)     # solid angle
    s = np.sqrt(1 - c * c)
    cp, sp = np.cos(_PSI), np.sin(_PSI)
    # the ray down (N, K, J, 3) and where it meets the ground
    down = (-c[..., None, None] * z[:, None, None, :]
            + s[..., None, None] * (cp[None, None, :, None] * e1[:, None, None, :]
                                    + sp[None, None, :, None] * e2[:, None, None, :]))
    rk = rm[:, None]
    t = rk * c - np.sqrt(np.clip(R_EARTH ** 2 - (rk * s) ** 2, 0.0, None))   # (N, K)
    ground = r[:, None, None, :] + t[..., None, None] * down
    n = ground / R_EARTH
    lat = np.arcsin(np.clip(n[..., 2], -1, 1))
    exitance = np.zeros(n.shape[:-1])
    if albedo:
        sun_hat = np.asarray(r_sun, float) / np.linalg.norm(r_sun)
        exitance = exitance + albedo_of(lat, jd) * np.clip(n @ sun_hat, 0.0, None)
    if infrared:
        exitance = exitance + 0.25 * emissivity(lat, jd)
    flux = P_SRP * (AU / np.linalg.norm(r_sun)) ** 2               # N/m^2 at the Earth
    # each ray: radiance M / pi over its solid angle, pushing back up the ray
    p = flux * exitance / np.pi * w[..., None]                     # (N, K, J), N/m^2
    a = -np.sum(p[..., None] * down, axis=(1, 2))
    return a * cram[:, None] * 1e-3                                # m/s^2 -> km/s^2
