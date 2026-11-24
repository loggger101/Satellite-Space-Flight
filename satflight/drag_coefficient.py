"""Physical drag coefficient of a sphere in free-molecular flow.

Sentman's closed form for a sphere with diffuse reemission (Sentman 1961),
each species of the air on its own, mass-weighted; the energy accommodation
from SESAM, a Langmuir fit to the atomic oxygen stuck on the surface
(Pilinski et al. 2010/2013). Checked against Moe & Moe (AAS 05-258, 2005),
whose diffuse sphere values follow from the reemitted rms speed
``v_r / v_i = sqrt(1 + alpha (3 R T_w / v^2 - 1))`` (their p. 4).

Pure numpy, SI units. Inputs are arrays of one shape; a non-finite or
out-of-range input is refused, never passed on as NaN.
"""

from __future__ import annotations

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])

K_B = 1.380649e-23                  # J/K
AMU = 1.66053907e-27                # kg
SESAM_K = 7.50e-17                  # m^3/K, Pilinski et al. sphere fit
WALL_K = 300.0                      # surface temperature, K
MIN_SPEED_RATIO = 0.1               # below this the closed form divides by ~0

# pymsis species after the total density (its columns 1-9, temperature is 10)
SPECIES = ("N2", "O2", "O", "He", "H", "Ar", "N", "anomalous O", "NO")
MASS_KG = np.array([28.0134, 31.9988, 15.9994, 4.002602, 1.00794, 39.948, 14.0067,
                    15.9994, 30.0061]) * AMU


def _checked(name, x, low=None, high=None):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"drag coefficient: {name} is not finite: {x[~np.isfinite(x)][:3]}")
    if low is not None and np.any(x < low):
        raise ValueError(f"drag coefficient: {name} below {low}: {x[x < low][:3]}")
    if high is not None and np.any(x > high):
        raise ValueError(f"drag coefficient: {name} above {high}: {x[x > high][:3]}")
    return x


def reemission_ratio(alpha, v, mass_kg, wall=WALL_K):
    """rms speed of the reemitted molecules over the incoming speed ``v`` (m/s)
    for energy accommodation ``alpha`` and a molecule of ``mass_kg``."""
    alpha = _checked("accommodation", alpha, 0.0, 1.0)
    v = _checked("speed", v, 1e-9)
    gas = K_B / np.asarray(mass_kg, dtype=float)
    return np.sqrt(1.0 + alpha * (3.0 * gas * wall / (v * v) - 1.0))


def sentman_sphere(s, vr_over_vi):
    """Sentman's sphere drag coefficient for speed ratio ``s`` (speed over
    sqrt(2 k T / m)) and reemitted-to-incoming speed ``vr_over_vi``."""
    s = _checked("speed ratio", s, MIN_SPEED_RATIO)
    vr = _checked("reemission ratio", vr_over_vi, 0.0)
    s2 = s * s
    return (np.exp(-s2) * (1.0 + 2.0 * s2) / (math.sqrt(math.pi) * s2 * s)
            + (4.0 * s2 * s2 + 4.0 * s2 - 1.0) / (2.0 * s2 * s2) * _erf(s)
            + 2.0 * math.sqrt(math.pi) / 3.0 * vr)


def accommodation(n_oxygen, temperature):
    """SESAM energy accommodation from the atomic-oxygen number density
    (m^-3) and the air temperature (K): ``K P / (1 + K P)`` with ``P = n_O T``."""
    p = SESAM_K * _checked("atomic oxygen", n_oxygen, 0.0) * _checked(
        "temperature", temperature, 1e-9)
    return p / (1.0 + p)


def sphere_cd(v_rel, number_density, temperature, wall=WALL_K):
    """Drag coefficient of a sphere, (N,), for speeds through the air ``v_rel``
    (m/s, (N,)), the species number densities (m^-3, (N, 9) in the order of
    :data:`SPECIES`) and the air temperature (K, (N,)). Each species' Sentman
    coefficient, weighted by its mass density; the accommodation from the
    atomic oxygen (:func:`accommodation`) applies to every species."""
    v = _checked("speed", v_rel, 0.0)[:, None]
    n = _checked("number density", number_density, 0.0)
    t = _checked("temperature", temperature, 1e-9)
    rho_j = n * MASS_KG
    rho = rho_j.sum(axis=1)
    if np.any(rho <= 0.0):
        raise ValueError("drag coefficient: no air (every species density is zero)")
    alpha = accommodation(n[:, SPECIES.index("O")], t)[:, None]
    s = v / np.sqrt(2.0 * K_B * t[:, None] / MASS_KG)
    cd_j = sentman_sphere(s, reemission_ratio(alpha, v, MASS_KG, wall))
    return (rho_j * cd_j).sum(axis=1) / rho
