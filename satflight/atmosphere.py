"""Piecewise-exponential atmosphere (Vallado Table 8-4, CIRA-72 based).

rho(h) = rho0 * exp(-(h - h0) / H) within each altitude band. Values are
for mean solar activity; real densities at 400 km swing by ~10x over the
solar cycle, so decay times are indicative rather than predictive. The
``scale`` argument lets a scenario model solar max (~3) or min (~0.3).

:func:`msis_density` is the alternative (forces "atmosphere": "msis"): NRLMSIS
2.1 by place and time, driven by the measured space weather.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .drag_coefficient import MASS_KG, SPECIES

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


# --- NRLMSIS 2.1 ----------------------------------------------------------------

ATMOSPHERES = ("table", "msis")     # forces "atmosphere": the table above, or NRLMSIS
_UNIX_JD = 2440587.5


SPECIES_SUM_TOL = 0.01   # species vs total density (pymsis works in float32)
N_SPECIES = len(SPECIES)   # pymsis columns 1-9; 10 is the temperature


@dataclass(frozen=True)
class MsisState:
    """The air NRLMSIS gives at each point: ``rho`` (kg/m^3, (N,)),
    ``number_density`` (m^-3, (N, 9), the species of
    :data:`satflight.drag_coefficient.SPECIES`) and ``temperature`` (K, (N,)).
    All zero above ``TOP_KM``."""
    rho: np.ndarray
    number_density: np.ndarray
    temperature: np.ndarray


def msis_state(alt_km, lat_rad, lon_rad, jd: float, table=None) -> MsisState:
    """NRLMSIS 2.1 at geodetic heights (km), latitudes and longitudes (rad), all
    at one UTC Julian date: the total density, each species and the temperature.

    The model (pymsis) is driven by the measured space weather of the date
    (:mod:`satflight.spaceweather`; ``table`` defaults to the downloaded one)
    with the 3-hourly ap history (storm-time mode). It follows the solar cycle,
    storms, the day/night bulge, season and latitude, which the table above
    cannot. Zero above ``TOP_KM`` like the table; SpaceWeatherError for a date
    without measured space weather. A species MSIS does not model at a height
    (NaN there) counts as none, but the species must add up to the total density
    within ``SPECIES_SUM_TOL``, and a non-finite density or temperature is a
    ValueError."""
    try:
        import pymsis
    except ImportError as exc:
        raise ImportError("the NRLMSIS atmosphere needs pymsis: pip install pymsis") from exc
    from . import spaceweather
    h = np.atleast_1d(np.asarray(alt_km, dtype=float))
    lat = np.broadcast_to(np.asarray(lat_rad, dtype=float), h.shape)
    lon = np.broadcast_to(np.asarray(lon_rad, dtype=float), h.shape)
    rho = np.zeros(h.shape)
    species = np.zeros(h.shape + (N_SPECIES,))
    temperature = np.zeros(h.shape)
    air = h <= TOP_KM
    if air.any():
        # to the millisecond first, so 03:00 that comes out as 02:59:59.99996 is 03:00
        ms = np.round((jd - _UNIX_JD) * 86400e3)
        utc = np.datetime64(int(ms), "ms")
        n = int(air.sum())
        f107, f107a, ap = (table or spaceweather.load()).indices(np.array([utc]))
        out = pymsis.calculate(
            np.full(n, utc), np.degrees(lon[air]), np.degrees(lat[air]),
            np.clip(h[air], 0.0, None), np.repeat(f107, n), np.repeat(f107a, n),
            np.repeat(ap, n, axis=0), geomagnetic_activity=-1)
        rho[air] = out[:, 0]
        if not (np.all(np.isfinite(out[:, 0])) and np.all(np.isfinite(out[:, 10]))):
            raise ValueError(f"NRLMSIS returned a non-finite density or temperature at {utc} "
                             f"UTC (heights {h[air][:3]} km)")
        # NaN = a species MSIS does not model at that height (O below ~55 km, H
        # below 80, N below 100, ...): none of it. The species must still make
        # up MSIS's own total, so a NaN that hides real air is an error.
        nd = np.nan_to_num(out[:, 1:1 + N_SPECIES].astype(float), nan=0.0)
        total = nd @ MASS_KG
        off = np.abs(total / out[:, 0] - 1.0) > SPECIES_SUM_TOL
        if off.any():
            raise ValueError(f"NRLMSIS species add up to {total[off][0]:.4g} kg/m^3 but its "
                             f"density is {out[off, 0][0]:.4g} at {h[air][off][0]:.1f} km, "
                             f"{utc} UTC")
        species[air] = nd
        temperature[air] = out[:, 10]
    return MsisState(rho, species, temperature)


def msis_density(alt_km, lat_rad, lon_rad, jd: float, scale: float = 1.0,
                 table=None) -> np.ndarray:
    """NRLMSIS 2.1 total mass density (kg/m^3) times ``scale``; see
    :func:`msis_state`."""
    return msis_state(alt_km, lat_rad, lon_rad, jd, table).rho * scale
