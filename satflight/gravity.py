"""The Earth's gravity field as spherical harmonics: EGM2008 to degree 70.

The coefficients (fully normalized, tide-free, from ICGEM; see
``tools/make_gravity.py``) ship in ``satflight/data/egm2008_70.gfc``. A scenario
picks the degree and order N (2..70) through ``ForceModel.gravity_degree``;
the field to N then stands in for the separate J2/J3/J4/C22 terms.

Only the field's *perturbation* is returned (degrees 2..N): the point mass
stays ``forces.accel_point_mass`` with the simulator's GM. Degree 1 is zero
(the origin is the centre of mass).

Two independent routes, so each checks the other:

* :func:`acceleration` - Cunningham's V/W recursion (Montenbruck & Gill,
  *Satellite Orbits*, sec. 3.2.4-3.2.5), unnormalized, in Earth-fixed
  Cartesian coordinates. Run degree by degree with every order and every
  satellite at once, so a 70x70 field costs about 2 ms for a few satellites.
* :func:`potential` - fully normalized associated Legendre functions by the
  standard forward column recursion, summed in spherical coordinates.

The Earth-fixed frame is the simulator's ECI turned by ``theta`` (GMST at
UT1, see ``frames``); polar motion is neglected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "egm2008_70.gfc"
MAX_DEGREE = 70


@dataclass(frozen=True)
class Field:
    """Coefficients of a field truncated to ``degree`` (index [n, m])."""
    name: str
    degree: int
    gm: float               # km^3/s^2, the field's own GM
    re: float               # km, its reference radius
    cbar: np.ndarray        # fully normalized C (n >= 2 only; n < 2 zeroed)
    sbar: np.ndarray
    c: np.ndarray           # unnormalized, for the V/W recursion
    s: np.ndarray


def _parse(path: Path):
    head, rows = {}, []
    with open(path, encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "gfc":
                rows.append(parts[1:5])
            elif len(parts) >= 2 and not rows:
                head[parts[0]] = parts[1]
    return head, rows


@lru_cache(maxsize=8)
def load(degree: int = MAX_DEGREE, path: Path = DATA) -> Field:
    """The field to ``degree`` (2..70); ValueError for any other degree or a
    file that is not a fully normalized, tide-free model of at least that degree."""
    if not (isinstance(degree, int) and 2 <= degree <= MAX_DEGREE):
        raise ValueError(f"gravity field degree {degree!r}: expected a whole number "
                         f"2..{MAX_DEGREE}")
    head, rows = _parse(path)
    for key, want in (("norm", "fully_normalized"), ("tide_system", "tide_free")):
        if head.get(key) != want:
            raise ValueError(f"{path.name}: {key} is {head.get(key)!r}, expected {want!r}")
    if int(head.get("max_degree", -1)) < degree:
        raise ValueError(f"{path.name} goes to degree {head.get('max_degree')}, "
                         f"not {degree}")
    cbar = np.zeros((degree + 1, degree + 1))
    sbar = np.zeros((degree + 1, degree + 1))
    seen = np.zeros((degree + 1, degree + 1), dtype=bool)
    for n, m, c, s in rows:
        n, m = int(n), int(m)
        if n <= degree:
            if seen[n, m]:
                raise ValueError(f"{path.name}: degree {n} order {m} appears twice")
            seen[n, m] = True
            cbar[n, m] = float(c.replace("d", "e").replace("D", "e"))
            sbar[n, m] = float(s.replace("d", "e").replace("D", "e"))
    want = np.tri(degree + 1, dtype=bool)
    want[1, :] = False                       # degree 1 is absent (geocentric)
    if not np.array_equal(seen, want):
        n, m = np.argwhere(seen != want)[0]
        raise ValueError(f"{path.name}: degree {n} order {m} is missing or misplaced")
    cbar[:2] = sbar[:2] = 0.0                 # the point mass is forces.accel_point_mass
    n = np.arange(degree + 1)[:, None]
    m = np.arange(degree + 1)[None, :]
    lg = np.vectorize(math.lgamma)
    # N_nm = sqrt((2 - delta_m0) (2n + 1) (n - m)! / (n + m)!), the normalization
    with np.errstate(invalid="ignore"):
        log_ratio = np.where(m <= n, lg(np.maximum(n - m, 0) + 1.0) - lg(n + m + 1.0), 0.0)
    norm = np.where(m <= n, np.sqrt((2.0 - (m == 0)) * (2 * n + 1) * np.exp(log_ratio)), 0.0)
    return Field(head.get("modelname", path.stem), degree,
                 float(head["earth_gravity_constant"].replace("D", "E")) * 1e-9,
                 float(head["radius"].replace("D", "E")) * 1e-3,
                 cbar, sbar, cbar * norm, sbar * norm)


def _ecef(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    return c * x + s * y, -s * x + c * y, z, c, s


@lru_cache(maxsize=8)
def _terms(degree: int):
    """Index arrays of the (n, m) pairs, n = 2..degree, split by m == 0."""
    nm = [(n, m) for n in range(2, degree + 1) for m in range(n + 1)]
    n0 = np.array([n for n, m in nm if m == 0])
    n1 = np.array([n for n, m in nm if m > 0])
    m1 = np.array([m for n, m in nm if m > 0])
    return n0, n1, m1


def acceleration(r, theta, degree: int = MAX_DEGREE) -> np.ndarray:
    """Field perturbation (km/s^2, degrees 2..``degree``) at ECI positions ``r``
    ``(N, 3)`` km, the Earth turned by ``theta`` (rad, scalar or (N,))."""
    f = load(degree)
    r = np.atleast_2d(np.asarray(r, dtype=float))
    x, y, z, c, s = _ecef(r, theta)
    r2 = x * x + y * y + z * z
    rho = f.re * f.re / r2
    x0, y0, z0 = f.re * x / r2, f.re * y / r2, f.re * z / r2
    top = degree + 2                                 # V, W to degree + 1
    V = np.zeros((len(r), top, top))
    W = np.zeros((len(r), top, top))
    V[:, 0, 0] = f.re / np.sqrt(r2)
    for n in range(1, top):
        m = np.arange(n)                              # orders below the diagonal
        prev2 = V[:, n - 2, :n] if n >= 2 else 0.0
        prev2w = W[:, n - 2, :n] if n >= 2 else 0.0
        a = (2 * n - 1) / (n - m)
        b = (n + m - 1) / (n - m)
        V[:, n, :n] = a * z0[:, None] * V[:, n - 1, :n] - b * rho[:, None] * prev2
        W[:, n, :n] = a * z0[:, None] * W[:, n - 1, :n] - b * rho[:, None] * prev2w
        k = 2 * n - 1                                 # the diagonal from the one before
        V[:, n, n] = k * (x0 * V[:, n - 1, n - 1] - y0 * W[:, n - 1, n - 1])
        W[:, n, n] = k * (x0 * W[:, n - 1, n - 1] + y0 * V[:, n - 1, n - 1])
    n0, n1, m1 = _terms(degree)
    C0 = f.c[n0, 0]
    C, S = f.c[n1, m1], f.s[n1, m1]
    fac = 0.5 * (n1 - m1 + 1) * (n1 - m1 + 2)
    vp, wp = V[:, n1 + 1, m1 + 1], W[:, n1 + 1, m1 + 1]
    vm, wm = V[:, n1 + 1, m1 - 1], W[:, n1 + 1, m1 - 1]
    ax = -(V[:, n0 + 1, 1] @ C0) + (0.5 * (-C * vp - S * wp) + fac * (C * vm + S * wm)).sum(1)
    ay = -(W[:, n0 + 1, 1] @ C0) + (0.5 * (-C * wp + S * vp) + fac * (-C * wm + S * vm)).sum(1)
    az = (-(n0 + 1) * C0 * V[:, n0 + 1, 0]).sum(1) + \
        ((n1 - m1 + 1) * (-C * V[:, n1 + 1, m1] - S * W[:, n1 + 1, m1])).sum(1)
    k = f.gm / (f.re * f.re)
    gx, gy, gz = k * ax, k * ay, k * az
    return np.stack([c * gx - s * gy, s * gx + c * gy, gz], axis=-1)


def legendre(degree: int, sinphi: np.ndarray) -> np.ndarray:
    """Fully normalized associated Legendre functions P[..., n, m] of sin(latitude)."""
    t = np.asarray(sinphi, dtype=float)
    u = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    P = np.zeros(t.shape + (degree + 1, degree + 1))
    P[..., 0, 0] = 1.0
    if degree >= 1:
        P[..., 1, 1] = math.sqrt(3.0) * u
    for m in range(2, degree + 1):
        P[..., m, m] = math.sqrt((2 * m + 1) / (2 * m)) * u * P[..., m - 1, m - 1]
    for m in range(degree + 1):
        for n in range(m + 1, degree + 1):
            a = math.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
            b = math.sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1)
                          / ((n - m) * (n + m) * (2 * n - 3))) if n >= m + 2 else 0.0
            P[..., n, m] = a * t * P[..., n - 1, m] - (b * P[..., n - 2, m] if b else 0.0)
    return P


def potential(r, theta, degree: int = MAX_DEGREE) -> np.ndarray:
    """Field perturbation potential (km^2/s^2; the acceleration is its gradient)
    at ECI ``r`` ``(N, 3)``, degrees 2..``degree``, by Legendre functions."""
    f = load(degree)
    r = np.atleast_2d(np.asarray(r, dtype=float))
    x, y, z, _, _ = _ecef(r, theta)
    rm = np.sqrt(x * x + y * y + z * z)
    lon = np.arctan2(y, x)
    P = legendre(degree, z / rm)
    m = np.arange(degree + 1)
    cm, sm = np.cos(np.outer(lon, m)), np.sin(np.outer(lon, m))
    inner = (P * (f.cbar[None] * cm[:, None, :] + f.sbar[None] * sm[:, None, :])).sum(-1)
    q = (f.re / rm)[:, None] ** np.arange(degree + 1)[None, :]
    return f.gm / rm * (q * inner).sum(-1)
