"""Reference frames and coordinate conversions.

ECI   Earth-centered pseudo-inertial frame *of date* (TEME-like): its Z axis
      is the Earth's spin axis and X the equinox of the epoch's date, because
      the Earth turns into it by GMST alone, the Sun series is mean-of-date
      and SGP4 returns TEME. It is not J2000/GCRF: precession has moved the
      axes ~0.37 deg since 2000 (about 45 km in LEO, 270 km at GEO when a
      state is compared with a J2000 tool). An of-date frame is the usual
      choice for this level of model: the spin axis is also J2's symmetry
      axis. Typed RAANs are of-date RAANs.
ECEF  Earth-centered, Earth-fixed. Obtained from ECI by a rotation of GMST
      about +Z (polar motion neglected).
ENU   local East-North-Up frame at a ground site.
RSW   radial / along-track (in-plane) / cross-track (orbit normal).
VNB   velocity / orbit normal / binormal - the natural "prograde, normal,
      radial-ish" frame for burns.

Every function is vectorized: positions are ``(..., 3)`` arrays.
"""

from __future__ import annotations

import numpy as np

from .constants import E2_EARTH, F_EARTH, OMEGA_EARTH, R_EARTH

OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH])


def rot1(a: float) -> np.ndarray:
    """Passive rotation about X by angle ``a`` (Vallado ROT1)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def rot3(a: float) -> np.ndarray:
    """Passive rotation about Z by angle ``a`` (Vallado ROT3)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis (zero vectors stay zero)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


# --- ECI <-> ECEF ----------------------------------------------------------

def eci_to_ecef(r: np.ndarray, theta) -> np.ndarray:
    """Rotate ECI positions into ECEF. ``theta`` (GMST, rad) may be an array
    broadcasting against the leading dimensions of ``r``."""
    r = np.asarray(r, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z * np.ones_like(c * x)], axis=-1)


def ecef_to_eci(r: np.ndarray, theta) -> np.ndarray:
    """Inverse of :func:`eci_to_ecef`."""
    return eci_to_ecef(r, -np.asarray(theta))


def eci_to_ecef_state(r: np.ndarray, v: np.ndarray, theta):
    """Position and velocity ECI -> ECEF (velocity relative to rotating Earth)."""
    v_rel = v - np.cross(OMEGA_VEC, r)
    return eci_to_ecef(r, theta), eci_to_ecef(v_rel, theta)


def ecef_to_eci_state(r_ecef: np.ndarray, v_ecef: np.ndarray, theta):
    """Inverse of :func:`eci_to_ecef_state` (adds the Earth-rotation velocity back)."""
    r = ecef_to_eci(r_ecef, theta)
    v = ecef_to_eci(v_ecef, theta) + np.cross(OMEGA_VEC, r)
    return r, v


# --- Geodetic ---------------------------------------------------------------

def geodetic_to_ecef(lat, lon, alt) -> np.ndarray:
    """WGS-84 geodetic latitude/longitude (rad) and height (km) -> ECEF km."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    sl = np.sin(lat)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2_EARTH) + alt) * sl
    return np.stack([x, y, z], axis=-1)


def _heikkinen(p, z):
    """Geodetic latitude (rad) and height (km) of points at distance ``p`` from
    the polar axis and ``z`` along it: Heikkinen's exact closed form (1982;
    Zhu 1993), no iteration, millimetre-exact from the surface out past GEO.
    Its square roots fail within ~43 km of the Earth's centre, so points
    deeper than half a radius (an orbit's "perigee" far underground) get the
    geocentric-latitude estimate instead, which is all such a depth needs."""
    p = np.asarray(p, dtype=float)
    z = np.asarray(z, dtype=float)
    deep = p * p + z * z < (0.5 * R_EARTH) ** 2
    p_in, z_in = p, z
    if np.any(deep):
        p, z = np.where(deep, R_EARTH, p), np.where(deep, 0.0, z)
    a, e2 = R_EARTH, E2_EARTH
    b2 = a * a * (1.0 - e2)
    ep2 = e2 / (1.0 - e2)
    z2, p2 = z * z, p * p
    f = 54.0 * b2 * z2
    g = p2 + (1.0 - e2) * z2 - e2 * (a * a - b2)
    c = e2 * e2 * f * p2 / (g * g * g)
    s = np.cbrt(1.0 + c + np.sqrt(c * c + 2.0 * c))
    k = s + 1.0 + 1.0 / s
    pp = f / (3.0 * k * k * g * g)
    q = np.sqrt(1.0 + 2.0 * e2 * e2 * pp)
    r0 = (-pp * e2 * p / (1.0 + q)
          + np.sqrt(np.maximum(0.5 * a * a * (1.0 + 1.0 / q)
                               - pp * (1.0 - e2) * z2 / (q * (1.0 + q)) - 0.5 * pp * p2, 0.0)))
    d2 = (p - e2 * r0) ** 2
    u = np.sqrt(d2 + z2)
    v = np.sqrt(d2 + (1.0 - e2) * z2)
    z0 = b2 * z / (a * v)
    lat, h = np.arctan2(z + ep2 * z0, p), u * (1.0 - b2 / (a * v))
    if np.any(deep):
        rm = np.hypot(p_in, z_in)
        s2 = np.divide(z_in * z_in, rm * rm, out=np.zeros_like(rm), where=rm > 0)
        lat = np.where(deep, np.arctan2(z_in, p_in * (1.0 - e2)), lat)
        h = np.where(deep, rm - R_EARTH * (1.0 - F_EARTH * s2), h)
    return lat, h


def ecef_to_geodetic(r: np.ndarray):
    """ECEF km -> (geodetic latitude rad, longitude rad, height km), exact
    (see :func:`_heikkinen`)."""
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lat, h = _heikkinen(np.hypot(x, y), z)
    return lat, np.arctan2(y, x), h


def geodetic_height(r: np.ndarray):
    """Height (km) above the WGS-84 ellipsoid of positions ``r`` (..., 3), exact.
    The ellipsoid is symmetric about the polar axis, so ECI positions give the
    same height as ECEF ones and need no Earth-rotation angle."""
    r = np.asarray(r, dtype=float)
    return _heikkinen(np.hypot(r[..., 0], r[..., 1]), r[..., 2])[1]


def eci_to_geodetic(r: np.ndarray, theta):
    """ECI km -> (geodetic latitude rad, longitude rad, height km) at GMST ``theta``."""
    return ecef_to_geodetic(eci_to_ecef(r, theta))


# --- Topocentric ------------------------------------------------------------

def enu_matrix(lat: float, lon: float) -> np.ndarray:
    """Rows are the East, North and Up unit vectors expressed in ECEF."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def look_angles(site_ecef: np.ndarray, site_lat: float, site_lon: float,
                target_ecef: np.ndarray):
    """Azimuth (rad, from north through east), elevation (rad) and slant
    range (km) of ECEF targets ``(..., 3)`` seen from a ground site."""
    rho = np.asarray(target_ecef) - site_ecef
    enu = rho @ enu_matrix(site_lat, site_lon).T
    rng = np.linalg.norm(enu, axis=-1)
    el = np.arcsin(np.clip(enu[..., 2] / np.where(rng == 0, 1, rng), -1, 1))
    az = np.mod(np.arctan2(enu[..., 0], enu[..., 1]), 2 * np.pi)
    return az, el, rng


# --- Orbit-local frames --------------------------------------------------------

def rsw_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` matrices whose rows are R, S, W in ECI."""
    rh = unit(r)
    wh = unit(np.cross(r, v))
    sh = np.cross(wh, rh)
    return np.stack([rh, sh, wh], axis=-2)


def vnb_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` matrices whose rows are V (prograde), N (orbit normal)
    and B = V x N, which is radially outward on a circular orbit."""
    vh = unit(v)
    nh = unit(np.cross(r, v))
    bh = np.cross(vh, nh)
    return np.stack([vh, nh, bh], axis=-2)


def local_to_eci(vec_local: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Express a vector given in an orbit-local frame (rows of ``basis``) in ECI."""
    return np.einsum("...i,...ij->...j", vec_local, basis)
