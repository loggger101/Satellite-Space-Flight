"""Force models: everything that accelerates a satellite.

All accelerations are in km/s^2 and vectorized over ``(N, 3)`` arrays.

* central body   -mu r / r^3
* zonal harmonics J2, J3, J4 (Earth oblateness and pear shape) and the
  sectoral C22/S22 term - or, instead of those, the full EGM2008 field to a
  chosen degree and order (``gravity_degree``, see ``gravity``)
* atmospheric drag with an atmosphere co-rotating with the Earth
* third-body gravity of the Sun and the Moon
* solar radiation pressure (cannonball, Cr*A/m) dimmed in the Earth's shadow

Every term is a per-scenario switch: the demos fly J2-J4 only, Sentinel-1
the full set (docs/SCOPE.md).

The equations are the standard ones found in Vallado ch. 8 and Montenbruck
& Gill ch. 3 - and in REBOUNDx's ``gravitational_harmonics`` / ``gas_drag``
effects, which play the same role there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from . import atmosphere, gravity
from .constants import (
    AU,
    C22,
    J2,
    J3,
    J4,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    OMEGA_EARTH,
    P_SRP,
    R_EARTH,
    S22,
)
from .eclipse import shadow_fraction
from .ephemeris import moon_position, sun_position
from .frames import OMEGA_VEC, geodetic_height
from .strict import check_keys

# --- Individual terms --------------------------------------------------------------

def accel_point_mass(r, mu: float = MU_EARTH):
    """Central-body (Keplerian) gravity."""
    rm = np.linalg.norm(r, axis=-1, keepdims=True)
    return -mu * r / rm ** 3


def accel_j2(r, mu: float = MU_EARTH, re: float = R_EARTH, j2: float = J2):
    """J2 (oblateness) perturbation."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = -1.5 * j2 * mu * re * re / rm ** 5
    zz = 5.0 * z * z / r2
    return np.stack([k * x * (1.0 - zz), k * y * (1.0 - zz), k * z * (3.0 - zz)], axis=-1)


def accel_j3(r, mu: float = MU_EARTH, re: float = R_EARTH, j3: float = J3):
    """J3 (north-south asymmetry, the "pear shape") perturbation."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = -2.5 * j3 * mu * re ** 3 / rm ** 7
    txy = 3.0 * z - 7.0 * z ** 3 / r2
    az = 6.0 * z * z - 7.0 * z ** 4 / r2 - 0.6 * r2
    return np.stack([k * x * txy, k * y * txy, k * az], axis=-1)


def accel_j4(r, mu: float = MU_EARTH, re: float = R_EARTH, j4: float = J4):
    """J4 perturbation."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    k = 1.875 * j4 * mu * re ** 4 / rm ** 7
    s = z * z / r2
    txy = 1.0 - 14.0 * s + 21.0 * s * s
    tz = 5.0 - 70.0 / 3.0 * s + 21.0 * s * s
    return np.stack([k * x * txy, k * y * txy, k * z * tz], axis=-1)


def _to_ecef_xy(r, theta):
    """ECEF x, y (and z) of ECI positions ``r`` with the Earth turned by ``theta``
    (a scalar, or one angle per row)."""
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    return c * x + s * y, -s * x + c * y, z, c, s


def accel_c22(r, theta, mu: float = MU_EARTH, re: float = R_EARTH, c22: float = C22,
              s22: float = S22):
    """Sectoral C22/S22 term (the equator is slightly elliptical) at ECI ``r``
    when the Earth has turned by GMST ``theta``: the gradient of

        U22 = 3 mu Re^2 (C22 (x^2 - y^2) + 2 S22 x y) / r^5   (x, y Earth-fixed).

    It fixes the Earth's field to the rotating Earth, so it depends on time.
    Geostationary satellites drift toward 75 E or 105 W under it."""
    x, y, z, c, s = _to_ecef_xy(r, theta)
    r2 = x * x + y * y + z * z
    k = 3.0 * mu * re * re / (r2 * r2 * np.sqrt(r2))
    w = c22 * (x * x - y * y) + 2.0 * s22 * x * y
    q = 5.0 * w / r2
    gx = k * (2.0 * (c22 * x + s22 * y) - q * x)
    gy = k * (2.0 * (s22 * x - c22 * y) - q * y)
    gz = -k * q * z
    return np.stack([c * gx - s * gy, s * gx + c * gy, gz], axis=-1)


def c22_potential(r, theta, mu: float = MU_EARTH, re: float = R_EARTH):
    """U22 of :func:`accel_c22` (km^2/s^2, the acceleration is its gradient)."""
    x, y, z, _, _ = _to_ecef_xy(r, theta)
    r2 = x * x + y * y + z * z
    w = C22 * (x * x - y * y) + 2.0 * S22 * x * y
    return 3.0 * mu * re * re * w / (r2 * r2 * np.sqrt(r2))


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


def accel_drag(r, v, cd_a_over_m, density_scale: float = 1.0):
    """Drag. ``cd_a_over_m`` is Cd*A/m in m^2/kg, shape (N,)."""
    v_rel = v - np.cross(OMEGA_VEC, r)
    rho = atmosphere.density(geodetic_height(r), density_scale)   # kg/m^3
    speed = np.linalg.norm(v_rel, axis=-1)
    # 0.5*rho*CdA/m*|v|v with v in m/s gives m/s^2; the km<->m factors net to 1e3
    k = -0.5e3 * rho * cd_a_over_m * speed
    return k[..., None] * v_rel


def accel_third_body(r, r_body, mu_body):
    """Tidal pull of a distant body at geocentric ``r_body`` (km, (3,)): its pull
    on the satellite minus its pull on the Earth (the frame's origin)."""
    d = r_body - r
    dm = np.linalg.norm(d, axis=-1, keepdims=True)
    bm = np.linalg.norm(r_body)
    return mu_body * (d / dm ** 3 - r_body / bm ** 3)


def accel_srp(r, r_sun, cr_a_over_m, shadow: bool = True):
    """Cannonball solar radiation pressure, away from the Sun, scaled by the
    sunlit fraction (Earth's penumbra and umbra). ``cr_a_over_m`` is Cr*A/m in
    m^2/kg, shape (N,)."""
    d = r - r_sun                      # Sun -> satellite
    dm = np.linalg.norm(d, axis=-1, keepdims=True)
    lit = shadow_fraction(r, r_sun) if shadow else 1.0
    # P [N/m^2] * CrA/m [m^2/kg] = m/s^2; * 1e-3 -> km/s^2
    mag = P_SRP * np.asarray(cr_a_over_m, float) * (AU / dm[..., 0]) ** 2 * 1e-3 * lit
    return mag[..., None] * d / dm


# --- The configurable model ------------------------------------------------------------

@dataclass
class ForceModel:
    """Which perturbations act on the ensemble; serialized into scenario files.

    ``gravity_degree`` N (2..70) switches the Earth's field to EGM2008 to degree
    and order N, which then stands in for the J2/J3/J4/C22 terms in the
    acceleration. The ``j*`` switches keep their meaning for the analytic
    helpers (the sun-synchronous inclination, launch guidance, mean elements),
    which use J2-J4 whatever the integrated field is. 0 = the separate terms."""

    j2: bool = True
    j3: bool = False
    j4: bool = False
    c22: bool = False
    drag: bool = False
    density_scale: float = 1.0
    gravity_degree: int = 0
    sun: bool = False
    moon: bool = False
    srp: bool = False

    TERMS = ("j2", "j3", "j4", "c22", "drag", "sun", "moon", "srp")
    NEEDS_TIME = ("sun", "moon", "srp")          # where the Sun or Moon is

    def __post_init__(self):
        g = self.gravity_degree
        if isinstance(g, bool) or not isinstance(g, int) or not (
                g == 0 or 2 <= g <= gravity.MAX_DEGREE):
            raise ValueError(f"forces gravity_degree {g!r}: expected 0 (J2/J3/J4/C22 terms) "
                             f"or a whole number 2..{gravity.MAX_DEGREE} (EGM2008)")

    def to_dict(self):
        """Plain dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build from a scenario dict; an unknown key raises ValueError."""
        check_keys(d or {}, {f.name for f in fields(cls)}, "forces")
        return cls(**(d or {}))

    @property
    def field(self) -> bool:
        """True when the EGM2008 field replaces the J2/J3/J4/C22 terms."""
        return self.gravity_degree > 0

    @property
    def earth_fixed(self) -> bool:
        """True when a term turns with the Earth (needs the rotation angle)."""
        return self.field or self.c22

    @property
    def needs_time(self) -> bool:
        """True when a term needs the date (the Sun's or Moon's position)."""
        return self.sun or self.moon or self.srp

    def active(self) -> list[str]:
        """The terms the acceleration actually includes, as labels."""
        out = [f"EGM{self.gravity_degree}"] if self.field else [
            t.upper() for t in ("j2", "j3", "j4", "c22") if getattr(self, t)]
        return out + [t.upper() for t in ("drag", "sun", "moon", "srp") if getattr(self, t)]

    def label(self) -> str:
        """Short summary such as ``2B+J2+DRAG`` or ``2B+EGM70+DRAG+SUN+MOON+SRP``."""
        on = self.active()
        return "two-body" if not on else "2B+" + "+".join(on)

    def acceleration(self, r, v, cd_a_over_m, gmst=0.0, jd=None, cr_a_over_m=None):
        """Total acceleration (km/s^2) of the enabled terms for ``(N, 3)`` states.
        ``gmst`` (rad) turns the Earth-fixed terms (C22, the EGM2008 field);
        ``jd`` (Julian date, UTC) places the Sun and Moon; ``cr_a_over_m``
        (Cr*A/m, m^2/kg, (N,)) scales sunlight pressure."""
        a = accel_point_mass(r)
        if self.field:
            a += gravity.acceleration(r, gmst, self.gravity_degree)
        else:
            if self.j2:
                a += accel_j2(r)
            if self.j3:
                a += accel_j3(r)
            if self.j4:
                a += accel_j4(r)
            if self.c22:
                a += accel_c22(r, gmst)
        if self.drag:
            a += accel_drag(r, v, cd_a_over_m, self.density_scale)
        if self.needs_time:
            if jd is None:
                raise ValueError("the Sun, Moon and sunlight-pressure terms need the date (jd)")
            if self.sun or self.srp:
                rs = sun_position(jd)
                if self.sun:
                    a += accel_third_body(r, rs, MU_SUN)
                if self.srp:
                    if cr_a_over_m is None:
                        raise ValueError("sunlight pressure needs Cr*A/m for each satellite")
                    a += accel_srp(r, rs, cr_a_over_m)
            if self.moon:
                a += accel_third_body(r, moon_position(jd), MU_MOON)
        return a

    def breakdown(self, r, v, cd_a_over_m, gmst=0.0, jd=None, cr_a_over_m=None):
        """Magnitude (km/s^2) of every term for display, enabled or not:
        {label: (magnitude, included)}. Terms the EGM2008 field stands in for
        show as not included while it is on."""
        r = np.atleast_2d(r)
        v = np.atleast_2d(v)
        sep = not self.field
        terms = {
            "gravity": (accel_point_mass(r), True),
            "J2": (accel_j2(r), sep and self.j2),
            "J3": (accel_j3(r), sep and self.j3),
            "J4": (accel_j4(r), sep and self.j4),
            "C22": (accel_c22(r, gmst), sep and self.c22),
            "drag": (accel_drag(r, v, np.atleast_1d(cd_a_over_m), self.density_scale),
                     self.drag),
        }
        if self.field:
            # the field beyond the J2 it shares with the separate terms, to compare
            terms[f"EGM{self.gravity_degree} - J2"] = (
                gravity.acceleration(r, gmst, self.gravity_degree) - accel_j2(r), True)
        if jd is not None:
            rs = sun_position(jd)
            cram = np.atleast_1d(cr_a_over_m if cr_a_over_m is not None else 0.0)
            terms["Sun"] = (accel_third_body(r, rs, MU_SUN), self.sun)
            terms["Moon"] = (accel_third_body(r, moon_position(jd), MU_MOON), self.moon)
            terms["sunlight"] = (accel_srp(r, rs, cram), self.srp)
        return {k: (float(np.linalg.norm(val[0])), bool(on)) for k, (val, on) in terms.items()}


def conservative_energy(r, v, model: ForceModel, gmst=0.0):
    """The quantity the enabled gravity terms conserve. With zonals only (an
    axisymmetric field) that is the specific energy. The C22/S22 term and the
    EGM2008 field turn with the Earth, so energy is no longer conserved; the
    Jacobi integral E - omega h_z (energy seen from the rotating Earth) is, and
    is returned instead. ``gmst`` (rad, scalar or one per row) places them.
    Drag, the Sun, the Moon and sunlight pressure change it; they are left out."""
    vm2 = np.sum(v * v, axis=-1)
    if model.field:
        e = 0.5 * vm2 - MU_EARTH / np.linalg.norm(r, axis=-1) - gravity.potential(
            r, gmst, model.gravity_degree)
    else:
        e = 0.5 * vm2 - zonal_potential(r, model.j2, model.j3, model.j4)
        if model.c22:
            e = e - c22_potential(r, gmst)
    if model.earth_fixed:
        e = e - OMEGA_EARTH * (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0])
    return e
