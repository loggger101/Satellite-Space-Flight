"""Low-precision analytic Sun and Moon ephemerides.

Sun:  Astronomical Almanac series (Vallado Alg. 29), ~0.01 deg accuracy.
Moon: truncated Brown series (Vallado Alg. 31), ~0.3 deg / ~1000 km.

These are the same series skyfield/astropy users reach for when a JPL DE
kernel would be overkill; they are ample for third-body perturbations,
eclipse geometry and lighting. Results are geocentric ECI (mean equator of
date, treated as J2000) in km.
"""

from __future__ import annotations

import numpy as np

from .constants import AU, R_EARTH
from .timeutil import centuries_since_j2000

D2R = np.pi / 180.0


def obliquity(jd):
    t = centuries_since_j2000(jd)
    return (23.439291 - 0.0130042 * t) * D2R


def sun_position(jd) -> np.ndarray:
    """Geocentric Sun vector, km. ``jd`` scalar -> (3,), array -> (N, 3)."""
    t = centuries_since_j2000(jd)
    lam_m = 280.460 + 36000.771 * t
    m = (357.5291092 + 35999.05034 * t) * D2R
    lam = (lam_m + 1.914666471 * np.sin(m) + 0.019994643 * np.sin(2 * m)) * D2R
    r = (1.000140612 - 0.016708617 * np.cos(m) - 0.000139589 * np.cos(2 * m)) * AU
    eps = obliquity(jd)
    return np.stack([r * np.cos(lam),
                     r * np.cos(eps) * np.sin(lam),
                     r * np.sin(eps) * np.sin(lam)], axis=-1)


def moon_position(jd) -> np.ndarray:
    """Geocentric Moon vector, km."""
    t = centuries_since_j2000(jd)

    def s(deg):
        return np.sin(deg * D2R)

    def c(deg):
        return np.cos(deg * D2R)

    lam = (218.32 + 481267.8813 * t
           + 6.29 * s(134.9 + 477198.85 * t)
           - 1.27 * s(259.2 - 413335.38 * t)
           + 0.66 * s(235.7 + 890534.23 * t)
           + 0.21 * s(269.9 + 954397.70 * t)
           - 0.19 * s(357.5 + 35999.05 * t)
           - 0.11 * s(186.6 + 966404.05 * t)) * D2R
    beta = (5.13 * s(93.3 + 483202.03 * t)
            + 0.28 * s(228.2 + 960400.87 * t)
            - 0.28 * s(318.3 + 6003.18 * t)
            - 0.17 * s(217.6 - 407332.20 * t)) * D2R
    parallax = (0.9508
                + 0.0518 * c(134.9 + 477198.85 * t)
                + 0.0095 * c(259.2 - 413335.38 * t)
                + 0.0078 * c(235.7 + 890534.23 * t)
                + 0.0028 * c(269.9 + 954397.70 * t)) * D2R
    r = R_EARTH / np.sin(parallax)
    eps = obliquity(jd)
    cb, sb = np.cos(beta), np.sin(beta)
    cl, sl = np.cos(lam), np.sin(lam)
    ce, se = np.cos(eps), np.sin(eps)
    return np.stack([r * cb * cl,
                     r * (ce * cb * sl - se * sb),
                     r * (se * cb * sl + ce * sb)], axis=-1)


def subsolar_point(jd, gmst_rad):
    """Geocentric latitude/longitude (rad) of the point with the Sun overhead."""
    s = sun_position(jd)
    lat = np.arctan2(s[..., 2], np.hypot(s[..., 0], s[..., 1]))
    lon = np.mod(np.arctan2(s[..., 1], s[..., 0]) - gmst_rad + np.pi, 2 * np.pi) - np.pi
    return lat, lon
