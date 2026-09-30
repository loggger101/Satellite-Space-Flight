"""Orbital manoeuvres: analytic transfer planners, a Lambert solver and the
schedulable manoeuvre objects the simulation executes.

Impulsive burns change velocity instantly and spend propellant via the
rocket equation; finite burns apply thrust T/m(t) continuously (with the
mass flow T / (Isp g0)) and are integrated like any other force.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .constants import MU_EARTH
from .elements import rv2coe, stumpff, time_to_true_anomaly
from .frames import local_to_eci, rsw_basis, unit, vnb_basis

G0 = 9.80665e-3   # km/s^2

KINDS = ("impulse", "circularize", "plane_change", "match_velocity", "finite")
TIMINGS = ("now", "delay", "periapsis", "apoapsis", "ascending_node", "descending_node",
           "absolute")
FRAMES = ("VNB", "RSW", "ECI")


# --- Analytic planners -----------------------------------------------------------

def hohmann(r1: float, r2: float, mu: float = MU_EARTH):
    """Coplanar circular -> circular Hohmann transfer.
    Returns (dv1, dv2, time_of_flight); dv signed (negative = retrograde)."""
    at = 0.5 * (r1 + r2)
    dv1 = math.sqrt(mu * (2 / r1 - 1 / at)) - math.sqrt(mu / r1)
    dv2 = math.sqrt(mu / r2) - math.sqrt(mu * (2 / r2 - 1 / at))
    tof = math.pi * math.sqrt(at ** 3 / mu)
    return dv1, dv2, tof


def bielliptic(r1: float, rb: float, r2: float, mu: float = MU_EARTH):
    """Bi-elliptic transfer via intermediate apoapsis ``rb``.
    Returns (dv1, dv2, dv3, total_time)."""
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (r2 + rb)
    dv1 = math.sqrt(mu * (2 / r1 - 1 / a1)) - math.sqrt(mu / r1)
    dv2 = math.sqrt(mu * (2 / rb - 1 / a2)) - math.sqrt(mu * (2 / rb - 1 / a1))
    dv3 = math.sqrt(mu / r2) - math.sqrt(mu * (2 / r2 - 1 / a2))
    tof = math.pi * (math.sqrt(a1 ** 3 / mu) + math.sqrt(a2 ** 3 / mu))
    return dv1, dv2, dv3, tof


def plane_change_dv(speed: float, delta_i: float) -> float:
    return 2.0 * speed * math.sin(abs(delta_i) / 2.0)


def lambert(r1, r2, tof: float, mu: float = MU_EARTH, prograde: bool = True,
            tol: float = 1e-10, max_iter: int = 1000):
    """Zero-revolution Lambert problem by universal variables with bisection
    (Vallado Alg. 58 / Curtis Alg. 5.2). Returns (v1, v2) in km/s."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1m, r2m = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (r1m * r2m), -1.0, 1.0)
    cz = np.cross(r1, r2)[2]
    dnu = math.acos(cos_dnu)
    if (prograde and cz < 0) or (not prograde and cz >= 0):
        dnu = 2 * math.pi - dnu
    if abs(1.0 - cos_dnu) < 1e-14:
        raise ValueError("Lambert: transfer angle of 0 deg is degenerate")
    A = math.sin(dnu) * math.sqrt(r1m * r2m / (1.0 - cos_dnu))
    if abs(A) < 1e-12:
        raise ValueError("Lambert: 180 deg transfer plane is undefined")
    sqmu = math.sqrt(mu)
    psi, low, up = 0.0, -4.0 * math.pi ** 2, 4.0 * math.pi ** 2
    y = 0.0
    for _ in range(max_iter):
        c2, c3 = (float(x) for x in stumpff(psi))
        y = r1m + r2m + A * (psi * c3 - 1.0) / math.sqrt(c2)
        if A > 0.0 and y < 0.0:
            low = psi
            psi = 0.5 * (low + up)
            continue
        chi = math.sqrt(y / c2)
        t = (chi ** 3 * c3 + A * math.sqrt(y)) / sqmu
        if abs(t - tof) < tol * max(tof, 1.0):
            break
        if t <= tof:
            low = psi
        else:
            up = psi
        psi = 0.5 * (low + up)
    else:
        raise ValueError("Lambert: no convergence (time of flight too short for a "
                         "zero-revolution transfer?)")
    f = 1.0 - y / r1m
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2m
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g


def sun_synchronous_inclination(a: float, e: float = 0.0) -> float:
    """Inclination (rad) whose J2 nodal regression matches the mean Sun
    (0.9856 deg/day eastward)."""
    from .constants import J2, R_EARTH
    target = 2 * math.pi / (365.2421897 * 86400.0)
    n = math.sqrt(MU_EARTH / a ** 3)
    p = a * (1 - e * e)
    cos_i = -target / (1.5 * n * J2 * (R_EARTH / p) ** 2)
    if abs(cos_i) > 1:
        raise ValueError("no sun-synchronous inclination at this altitude")
    return math.acos(cos_i)


# --- Schedulable manoeuvres ------------------------------------------------------

@dataclass
class Maneuver:
    sat: str
    kind: str = "impulse"
    timing: str = "now"          # how ``t`` was / should be chosen
    t: float | None = None       # absolute execution time (s since epoch)
    delay: float = 0.0           # for timing == "delay"
    dv: tuple = (0.0, 0.0, 0.0)  # km/s in ``frame`` (impulse); direction for finite
    frame: str = "VNB"
    delta_i: float = 0.0         # deg, plane change (positive = increase inclination)
    target: str = ""             # satellite to match velocity with
    thrust: float = 0.0          # N  (finite burn)
    isp: float = 300.0           # s
    duration: float = 0.0        # s  (finite burn)
    label: str = ""
    done: bool = False
    result: str = field(default="", compare=False)

    def to_dict(self):
        d = asdict(self)
        d["dv"] = list(self.dv)
        return d

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        m = cls(**{k: v for k, v in d.items() if k in names})
        m.dv = tuple(float(x) for x in m.dv)
        return m

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.kind == "impulse":
            return f"dV {np.linalg.norm(self.dv) * 1000:.1f} m/s ({self.frame})"
        if self.kind == "plane_change":
            return f"plane change {self.delta_i:+.2f} deg"
        if self.kind == "finite":
            return f"finite burn {self.thrust:.0f} N x {self.duration:.0f} s"
        if self.kind == "match_velocity":
            return f"match velocity with {self.target}"
        return self.kind


def resolve_time(m: Maneuver, t_now: float, r, v, mu: float = MU_EARTH) -> float:
    """Absolute execution time for a manoeuvre scheduled from state (r, v)."""
    if m.timing == "absolute" and m.t is not None:
        return m.t
    if m.timing == "now":
        return t_now
    if m.timing == "delay":
        return t_now + max(0.0, m.delay)
    el = rv2coe(r, v, mu)
    if m.timing == "periapsis":
        nu = 0.0
    elif m.timing == "apoapsis":
        if el.e >= 1.0:
            raise ValueError("open orbit has no apoapsis")
        nu = math.pi
    elif m.timing == "ascending_node":
        nu = (-el.argp) % (2 * math.pi)
    elif m.timing == "descending_node":
        nu = (math.pi - el.argp) % (2 * math.pi)
    else:
        raise ValueError(f"unknown timing {m.timing!r}")
    dt = time_to_true_anomaly(r, v, nu, mu)
    if not math.isfinite(dt):
        raise ValueError(f"orbit never reaches {m.timing.replace('_', ' ')}")
    return t_now + dt


def impulse_eci(m: Maneuver, r, v, target_v=None) -> np.ndarray:
    """ECI delta-v (km/s) an impulsive manoeuvre applies at state (r, v)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    if m.kind == "impulse":
        dv = np.asarray(m.dv, dtype=float)
        if m.frame == "VNB":
            return local_to_eci(dv, vnb_basis(r, v))
        if m.frame == "RSW":
            return local_to_eci(dv, rsw_basis(r, v))
        return dv
    if m.kind == "circularize":
        rh = unit(r)
        h = unit(np.cross(r, v))
        v_circ = math.sqrt(MU_EARTH / np.linalg.norm(r)) * np.cross(h, rh)
        return v_circ - v
    if m.kind == "plane_change":
        # Rotating v about r rotates h = r x v by the same angle. The orbit
        # normal is +Z turned about the ascending-node vector n by +i, so a
        # positive turn about r raises i on the ascending side (r . n > 0)
        # and lowers it on the descending side, where r points along -n.
        k = unit(r)
        node = np.cross([0.0, 0.0, 1.0], np.cross(r, v))
        ang = math.radians(m.delta_i)
        if np.dot(r, node) < 0:
            ang = -ang
        c, s = math.cos(ang), math.sin(ang)
        v_new = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
        return v_new - v
    if m.kind == "match_velocity":
        if target_v is None:
            raise ValueError("target velocity required")
        return np.asarray(target_v, dtype=float) - v
    raise ValueError(f"{m.kind} is not an impulsive manoeuvre")


def finite_direction(m: Maneuver, r, v) -> np.ndarray:
    """Unit thrust directions ``(N, 3)`` in ECI for a finite burn at states r, v."""
    d = np.asarray(m.dv, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    if m.frame == "VNB":
        return local_to_eci(d, vnb_basis(r, v))
    if m.frame == "RSW":
        return local_to_eci(d, rsw_basis(r, v))
    return np.broadcast_to(d, np.shape(r))


def propellant_after(mass: float, dv: float, isp: float) -> float:
    """Mass remaining after a burn of ``dv`` km/s (Tsiolkovsky)."""
    if isp <= 0:
        return mass
    return mass * math.exp(-dv / (isp * G0))
