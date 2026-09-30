"""Low-precision analytic Sun ephemeris.

Astronomical Almanac series (Vallado Alg. 29), ~0.01 deg accuracy - the
series skyfield/astropy users reach for when a JPL DE kernel would be
overkill. It is used only for lighting, eclipses, beta angle and local solar
time; the Sun exerts no force on the satellites (docs/SCOPE.md). Results are
geocentric ECI (mean equator of date, treated as J2000) in km.
"""

from __future__ import annotations

import numpy as np

from .constants import AU
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


def subsolar_point(jd, gmst_rad):
    """Geocentric latitude/longitude (rad) of the point with the Sun overhead."""
    s = sun_position(jd)
    lat = np.arctan2(s[..., 2], np.hypot(s[..., 0], s[..., 1]))
    lon = np.mod(np.arctan2(s[..., 1], s[..., 0]) - gmst_rad + np.pi, 2 * np.pi) - np.pi
    return lat, lon
