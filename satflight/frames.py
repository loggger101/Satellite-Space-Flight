"""Reference frames and coordinate conversions.

ECI   Earth-centered inertial. Treated as J2000/GCRF; precession and nutation
      (< 0.4 deg over the 2000-2030 span) are neglected, as in most
      teaching-grade simulators.
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

from .constants import E2_EARTH, OMEGA_EARTH, R_EARTH

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


def ecef_to_geodetic(r: np.ndarray, iterations: int = 6):
    """ECEF km -> (geodetic latitude rad, longitude rad, height km).

    Fixed-point iteration on latitude; the height expression used is
    well-conditioned at the poles as well as the equator.
    """
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))
    for _ in range(iterations):
        sl = np.sin(lat)
        n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
        lat = np.arctan2(z + E2_EARTH * n * sl, p)
    sl = np.sin(lat)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
    h = p * np.cos(lat) + (z + E2_EARTH * n * sl) * sl - n
    return lat, lon, h


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
