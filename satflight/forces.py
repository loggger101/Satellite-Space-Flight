"""Force models: everything that accelerates a satellite.

All accelerations are in km/s^2 and vectorised over ``(N, 3)`` arrays.

* central body   -mu r / r^3
* zonal harmonics J2, J3, J4 (Earth oblateness and pear shape)
* atmospheric drag with an atmosphere co-rotating with the Earth
* third-body Sun and Moon (direct minus indirect term)
* solar radiation pressure (cannonball model) with conical Earth shadow

The equations are the standard ones found in Vallado ch. 8 and Montenbruck
& Gill ch. 3 - and in REBOUNDx's ``gravitational_harmonics`` /
``gas_drag`` / ``radiation_forces`` effects, which play the same role there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from . import atmosphere
from .constants import (AU, F_EARTH, J2, J3, J4, MU_EARTH, MU_MOON, MU_SUN,
                        OMEGA_EARTH, P_SRP, R_EARTH)
from .eclipse import shadow_fraction
from .ephemeris import moon_position, sun_position

OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH])


# --- Individual terms --------------------------------------------------------------

def accel_point_mass(r, mu: float = MU_EARTH):
    rm = np.linalg.norm(r, axis=-1, keepdims=True)
    return -mu * r / rm ** 3


def accel_j2(r, mu: float = MU_EARTH, re: float = R_EARTH, j2: float = J2):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = -1.5 * j2 * mu * re * re / rm ** 5
    zz = 5.0 * z * z / r2
    return np.stack([k * x * (1.0 - zz), k * y * (1.0 - zz), k * z * (3.0 - zz)], axis=-1)


def accel_j3(r, mu: float = MU_EARTH, re: float = R_EARTH, j3: float = J3):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = -2.5 * j3 * mu * re ** 3 / rm ** 7
    txy = 3.0 * z - 7.0 * z ** 3 / r2
    az = 6.0 * z * z - 7.0 * z ** 4 / r2 - 0.6 * r2
    return np.stack([k * x * txy, k * y * txy, k * az], axis=-1)


def accel_j4(r, mu: float = MU_EARTH, re: float = R_EARTH, j4: float = J4):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = 1.875 * j4 * mu * re ** 4 / rm ** 7
    s = z * z / r2
    txy = 1.0 - 14.0 * s + 21.0 * s * s
    tz = 5.0 - 70.0 / 3.0 * s + 21.0 * s * s
    return np.stack([k * x * txy, k * y * txy, k * z * tz], axis=-1)


def zonal_potential(r, j2=True, j3=False, j4=False, mu: float = MU_EARTH,
                    re: float = R_EARTH):
    """Gravitational potential Phi (km^2/s^2, a = grad Phi) incl. chosen zonals."""
    rm = np.linalg.norm(r, axis=-1)
    s = r[..., 2] / rm
    q = re / rm
    phi = np.ones_like(rm)
    if j2:
        phi -= J2 * q ** 2 * 0.5 * (3 * s * s - 1)
    if j3:
        phi -= J3 * q ** 3 * 0.5 * (5 * s ** 3 - 3 * s)
    if j4:
        phi -= J4 * q ** 4 * (35 * s ** 4 - 30 * s * s + 3) / 8.0
    return mu / rm * phi


def approx_altitude(r):
    """Height above the WGS-84 ellipsoid using the local geocentric radius
    (within ~20 m of the exact geodetic height; cheap enough for drag)."""
    rm = np.linalg.norm(r, axis=-1)
    s2 = (r[..., 2] / rm) ** 2
    return rm - R_EARTH * (1.0 - F_EARTH * s2)


def accel_drag(r, v, cd_a_over_m, density_scale: float = 1.0):
    """Drag. ``cd_a_over_m`` is Cd*A/m in m^2/kg, shape (N,)."""
    v_rel = v - np.cross(OMEGA_VEC, r)
    rho = atmosphere.density(approx_altitude(r), density_scale)   # kg/m^3
    speed = np.linalg.norm(v_rel, axis=-1)
    # 0.5*rho*CdA/m*|v|v with v in m/s gives m/s^2; the km<->m factors net to 1e3
    k = -0.5e3 * rho * cd_a_over_m * speed
    return k[..., None] * v_rel


def accel_third_body(r, r_body, mu_body):
    d = r_body - r
    dm = np.linalg.norm(d, axis=-1, keepdims=True)
    bm = np.linalg.norm(r_body)
    return mu_body * (d / dm ** 3 - r_body / bm ** 3)


def accel_srp(r, r_sun, cr_a_over_m, shadow: bool = True):
    """Cannonball SRP. ``cr_a_over_m`` is Cr*A/m in m^2/kg."""
    d = r - r_sun                      # Sun -> satellite
    dm = np.linalg.norm(d, axis=-1, keepdims=True)
    nu = shadow_fraction(r, r_sun) if shadow else 1.0
    # P [N/m^2] * CrA/m [m^2/kg] = m/s^2 ; /1e3 -> km/s^2
    mag = P_SRP * cr_a_over_m * (AU / dm[..., 0]) ** 2 * 1e-3 * nu
    return mag[..., None] * d / dm


# --- The configurable model ------------------------------------------------------------

@dataclass
class ForceModel:
    j2: bool = True
    j3: bool = False
    j4: bool = False
    drag: bool = False
    sun: bool = False
    moon: bool = False
    srp: bool = False
    density_scale: float = 1.0

    TERMS = ("j2", "j3", "j4", "drag", "sun", "moon", "srp")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def label(self) -> str:
        on = [t.upper() for t in self.TERMS if getattr(self, t)]
        return "two-body" if not on else "2B+" + "+".join(on)

    def acceleration(self, jd: float, r, v, cd_a_over_m, cr_a_over_m):
        a = accel_point_mass(r)
        if self.j2:
            a += accel_j2(r)
        if self.j3:
            a += accel_j3(r)
        if self.j4:
            a += accel_j4(r)
        if self.drag:
            a += accel_drag(r, v, cd_a_over_m, self.density_scale)
        if self.sun or self.srp:
            rs = sun_position(jd)
            if self.sun:
                a += accel_third_body(r, rs, MU_SUN)
            if self.srp:
                a += accel_srp(r, rs, cr_a_over_m)
        if self.moon:
            a += accel_third_body(r, moon_position(jd), MU_MOON)
        return a

    def breakdown(self, jd: float, r, v, cd_a_over_m, cr_a_over_m):
        """Magnitude (km/s^2) of every term for display, enabled or not."""
        r = np.atleast_2d(r)
        v = np.atleast_2d(v)
        rs = sun_position(jd)
        terms = {
            "gravity": accel_point_mass(r),
            "J2": accel_j2(r),
            "J3": accel_j3(r),
            "J4": accel_j4(r),
            "drag": accel_drag(r, v, np.atleast_1d(cd_a_over_m), self.density_scale),
            "sun": accel_third_body(r, rs, MU_SUN),
            "moon": accel_third_body(r, moon_position(jd), MU_MOON),
            "srp": accel_srp(r, rs, np.atleast_1d(cr_a_over_m)),
        }
        enabled = {"gravity": True, "J2": self.j2, "J3": self.j3, "J4": self.j4,
                   "drag": self.drag, "sun": self.sun, "moon": self.moon, "srp": self.srp}
        return {k: (float(np.linalg.norm(val[0])), enabled[k]) for k, val in terms.items()}


def conservative_energy(r, v, model: ForceModel):
    """Specific energy including the enabled zonal terms; conserved exactly
    when only point mass + zonals act (the field is axisymmetric)."""
    vm2 = np.sum(v * v, axis=-1)
    return 0.5 * vm2 - zonal_potential(r, model.j2, model.j3, model.j4)
