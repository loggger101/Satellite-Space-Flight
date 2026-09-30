"""Piecewise-exponential atmosphere (Vallado Table 8-4, CIRA-72 based).

rho(h) = rho0 * exp(-(h - h0) / H) within each altitude band. Values are
for mean solar activity; real densities at 400 km swing by ~10x over the
solar cycle, so decay times are indicative rather than predictive. The
``scale`` argument lets a scenario model solar max (~3) or min (~0.3).
"""

from __future__ import annotations

import numpy as np

# base altitude km, nominal density kg/m^3, scale height km
_TABLE = np.array([
    [0, 1.225, 7.249],
    [25, 3.899e-2, 6.349],
    [30, 1.774e-2, 6.682],
    [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382],
    [60, 3.206e-4, 7.714],
    [70, 8.770e-5, 6.549],
    [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382],
    [100, 5.297e-7, 5.877],
    [110, 9.661e-8, 7.263],
    [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636],
    [140, 3.845e-9, 16.149],
    [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740],
    [200, 2.789e-10, 37.105],
    [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628],
    [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828],
    [500, 6.967e-13, 63.822],
    [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667],
    [800, 1.170e-14, 124.64],
    [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
])
H0, RHO0, SCALE_H = _TABLE[:, 0], _TABLE[:, 1], _TABLE[:, 2]
TOP_KM = 2500.0   # above this the density is treated as zero


def density(alt_km, scale: float = 1.0) -> np.ndarray:
    """Mass density in kg/m^3 at geodetic altitude(s) in km."""
    h = np.asarray(alt_km, dtype=float)
    hc = np.clip(h, 0.0, None)
    idx = np.clip(np.searchsorted(H0, hc, side="right") - 1, 0, len(H0) - 1)
    rho = RHO0[idx] * np.exp(-(hc - H0[idx]) / SCALE_H[idx]) * scale
    return np.where(h > TOP_KM, 0.0, rho)
