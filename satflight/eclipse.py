"""Earth shadow geometry (conical umbra/penumbra model).

Implements the apparent-disc overlap model of Montenbruck & Gill,
*Satellite Orbits* sec. 3.4.2: the fraction of the solar disc visible from
the satellite, 1 in sunlight, 0 in umbra, between in penumbra.
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH, R_SUN

SUNLIT, PENUMBRA, UMBRA = "sunlit", "penumbra", "umbra"


def shadow_fraction(r_sat: np.ndarray, r_sun: np.ndarray) -> np.ndarray:
    """Illuminated fraction (0..1) for satellites ``(N, 3)`` given the
    geocentric Sun vector ``(3,)``; all km."""
    r = np.atleast_2d(np.asarray(r_sat, dtype=float))
    d = np.asarray(r_sun, dtype=float) - r              # satellite -> Sun
    rm = np.linalg.norm(r, axis=-1)
    dm = np.linalg.norm(d, axis=-1)
    a = np.arcsin(np.clip(R_SUN / dm, -1, 1))            # apparent solar radius
    b = np.arcsin(np.clip(R_EARTH / np.maximum(rm, R_EARTH), -1, 1))  # apparent Earth radius
    c = np.arccos(np.clip(np.sum(-r * d, axis=-1) / (rm * dm), -1, 1))  # separation

    nu = np.ones_like(rm)
    umbra = c < b - a
    annular = (c < a - b) & ~umbra
    partial = (c < a + b) & ~umbra & ~annular
    nu[umbra] = 0.0
    nu[annular] = 1.0 - (b[annular] / a[annular]) ** 2
    if np.any(partial):
        ap, bp, cp = a[partial], b[partial], c[partial]
        x = (cp * cp + ap * ap - bp * bp) / (2.0 * cp)
        y = np.sqrt(np.maximum(ap * ap - x * x, 0.0))
        area = (ap * ap * np.arccos(np.clip(x / ap, -1, 1))
                + bp * bp * np.arccos(np.clip((cp - x) / bp, -1, 1)) - cp * y)
        nu[partial] = 1.0 - area / (np.pi * ap * ap)
    return nu


def shadow_state(fraction: float) -> str:
    """Classify an illuminated fraction as SUNLIT, PENUMBRA or UMBRA."""
    if fraction >= 0.999999:
        return SUNLIT
    if fraction <= 1e-6:
        return UMBRA
    return PENUMBRA
