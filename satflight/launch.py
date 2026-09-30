"""Launch vehicles: a powered ascent from any point on the Earth's surface.

A launch is a multi-stage rocket standing on a pad at geodetic latitude,
longitude and height. Until lift-off it turns with the Earth. From lift-off it
is integrated on its own (RK4, 1 s steps) under

* gravity: point mass plus the zonal terms the scenario enables,
* thrust ``T(h) = mdot g0 Isp(h)``: the mass flow is fixed by the vacuum
  rating, and Isp rises from its sea-level to its vacuum value as the ambient
  pressure ``p/p0 = exp(-h / 7 km)`` falls,
* aerodynamic drag in the co-rotating atmosphere (always on during ascent,
  whatever the scenario's drag switch says: no rocket climbs without it),
* a falling mass: propellant flows out, spent stages and the fairing drop off,
* an optional acceleration limit that throttles the engines (40-100 %).

Steering, as flown by real launchers:

1. vertical rise for ``vertical_time`` seconds;
2. pitch kick: the thrust leans ``kick`` degrees toward the launch azimuth
   until the air-relative velocity has leaned as far;
3. gravity turn: thrust along the air-relative velocity (zero angle of
   attack) through the dense atmosphere;
4. closed-loop guidance (``guidance == "orbit"``) from the first staging or
   50 km, whichever comes first. The radial and out-of-plane accelerations
   are shaped as ``A + B t`` so that the radius reaches the insertion radius
   with zero vertical speed, and the position and velocity out of the target
   plane both reach zero, at the predicted cut-off. The time to go comes from
   the rocket equation over the remaining stages. The engines cut off when
   the orbital energy reaches the target orbit's, which puts the payload at
   perigee of the requested orbit.

With ``guidance == "open"`` there is no step 4: the vehicle flies the gravity
turn along a fixed azimuth and burns every stage (sounding rockets, ballistic
tests). The kick angle may be given or left to :func:`plan_launch`, which
flies candidate ascents and keeps the one that reaches the target orbit with
the most propellant to spare.

The orbit plane comes from the inclination and the launch direction
(northbound or southbound) at the moment of lift-off, or the launch waits for
a *window*: the moment the pad rotates into a plane of given RAAN, or into
the plane of another satellite (its J2 regression included).
"""

from __future__ import annotations

import bisect
import copy
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .atmosphere import H0, RHO0, SCALE_H, TOP_KM
from .constants import F_EARTH, J2, J3, J4, MU_EARTH, OMEGA_EARTH, R_EARTH
from .elements import rv2coe
from .frames import ecef_to_eci_state, geodetic_to_ecef

G0_M = 9.80665            # m/s^2, standard gravity (defines Isp)
P_SCALE_H = 7.0           # km, scale height of ambient pressure (thrust vs altitude)
GUIDANCE_ALT = 50.0       # km: closed-loop guidance starts here or at the first staging
H_POWERED = 1.0           # s, integration step of the ascent
H_COAST = 5.0             # s, step for objects coasting until they join the ensemble
SAMPLE_DT = 2.0           # s between recorded profile samples
FREEZE_TGO = 4.0          # s: steering is held constant this close to cut-off
MIN_PERIGEE = 100.0       # km: lowest target perigee accepted
MAX_BURN = 3600.0         # s: longest stage burn accepted
TIMINGS = ("now", "delay", "absolute", "raan", "plane")
GUIDANCES = ("orbit", "open")

_H0, _RHO0, _SH = H0.tolist(), RHO0.tolist(), SCALE_H.tolist()


# --- Vehicle description ---------------------------------------------------------

def _from_dict(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Stage:
    name: str = "Stage"
    thrust: float = 1000.0      # kN, in vacuum
    isp_vac: float = 320.0      # s
    isp_sl: float = 0.0         # s at sea level; 0 = vacuum engine (Isp stays isp_vac)
    propellant: float = 10000.0 # kg
    dry: float = 1000.0         # kg, everything left when the tanks are empty

    @property
    def mdot(self) -> float:
        """Mass flow at full throttle (kg/s)."""
        return self.thrust * 1e3 / (self.isp_vac * G0_M)

    @property
    def burn_time(self) -> float:
        return self.propellant / self.mdot

    def isp(self, p_ratio: float) -> float:
        sl = self.isp_sl or self.isp_vac
        return self.isp_vac - (self.isp_vac - sl) * p_ratio

    def to_dict(self):
        return asdict(self)


@dataclass
class Vehicle:
    name: str = "Custom"
    stages: list = field(default_factory=list)   # list[Stage], first to burn first
    fairing: float = 0.0        # kg, jettisoned at fairing_alt
    fairing_alt: float = 110.0  # km
    diameter: float = 3.7       # m, drag reference area pi d^2 / 4
    cd: float = 0.5
    max_g: float = 0.0          # acceleration limit (g); 0 = none
    stage_coast: float = 5.0    # s from burnout to the next stage's ignition

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    def liftoff_mass(self, payload: float) -> float:
        return payload + self.fairing + sum(s.propellant + s.dry for s in self.stages)

    def stage_summary(self, payload: float):
        """Per stage: (mass at ignition kg, ideal vacuum dV km/s, burn time s,
        thrust/weight at ignition). The fairing is counted on the first stage."""
        out = []
        m = self.liftoff_mass(payload)
        for k, s in enumerate(self.stages):
            m0 = m
            m1 = m0 - s.propellant
            dv = s.isp_vac * G0_M * math.log(m0 / m1) / 1000.0 if m1 > 0 else math.inf
            thrust = s.thrust * (s.isp_sl or s.isp_vac) / s.isp_vac if k == 0 else s.thrust
            out.append((m0, dv, s.burn_time, thrust * 1e3 / (m0 * G0_M)))
            m = m1 - s.dry - (self.fairing if k == 0 else 0.0)
        return out

    def to_dict(self):
        d = asdict(self)
        d["stages"] = [asdict(s) if isinstance(s, Stage) else dict(s) for s in self.stages]
        return d

    @classmethod
    def from_dict(cls, d):
        v = _from_dict(cls, d)
        v.stages = [s if isinstance(s, Stage) else _from_dict(Stage, s) for s in v.stages]
        return v

    def copy(self) -> Vehicle:
        return Vehicle.from_dict(copy.deepcopy(self.to_dict()))


# Approximate public figures, for play and comparison rather than engineering.
VEHICLES = {
    "Falcon 9 (approx.)": Vehicle(
        "Falcon 9", [Stage("Stage 1", 8227.0, 311.0, 282.0, 395700.0, 25600.0),
                     Stage("Stage 2", 981.0, 348.0, 0.0, 92670.0, 3900.0)],
        fairing=1900.0, fairing_alt=110.0, diameter=3.7, cd=0.45, stage_coast=8.0),
    "Electron (approx.)": Vehicle(
        "Electron", [Stage("Stage 1", 224.0, 311.0, 282.0, 9250.0, 950.0),
                     Stage("Stage 2", 25.8, 343.0, 0.0, 2050.0, 250.0)],
        fairing=50.0, fairing_alt=105.0, diameter=1.2, cd=0.45, stage_coast=3.0),
    "Saturn V (approx.)": Vehicle(
        "Saturn V", [Stage("S-IC", 38700.0, 304.0, 263.0, 2077000.0, 131000.0),
                     Stage("S-II", 5141.0, 421.0, 0.0, 444000.0, 40000.0),
                     Stage("S-IVB", 1033.0, 421.0, 0.0, 109000.0, 13500.0)],
        fairing=4000.0, fairing_alt=90.0, diameter=10.1, cd=0.5, max_g=4.0, stage_coast=4.0),
    "Sounding rocket": Vehicle(
        "Sounding rocket", [Stage("Motor", 60.0, 255.0, 230.0, 1100.0, 280.0)],
        fairing=0.0, diameter=0.56, cd=0.35, stage_coast=0.0),
}


def vehicle_preset(name: str) -> Vehicle:
    return VEHICLES[name].copy()


# Public coordinates of launch complexes (geodetic deg, km).
LAUNCH_SITES = {
    "Cape Canaveral SLC-40 (USA)": (28.562, -80.577, 0.0),
    "Kennedy LC-39A (USA)": (28.608, -80.604, 0.0),
    "Vandenberg SLC-4E (USA)": (34.632, -120.611, 0.1),
    "Wallops (USA)": (37.833, -75.488, 0.0),
    "Starbase (USA)": (25.997, -97.157, 0.0),
    "Kourou ELA-3 (French Guiana)": (5.239, -52.768, 0.0),
    "Baikonur Site 1 (Kazakhstan)": (45.920, 63.342, 0.1),
    "Plesetsk (Russia)": (62.927, 40.577, 0.1),
    "Vostochny (Russia)": (51.884, 128.334, 0.2),
    "Jiuquan (China)": (40.958, 100.291, 1.0),
    "Wenchang (China)": (19.614, 110.951, 0.0),
    "Tanegashima (Japan)": (30.400, 130.975, 0.0),
    "Sriharikota (India)": (13.720, 80.230, 0.0),
    "Mahia LC-1 (New Zealand)": (-39.262, 177.865, 0.1),
    "Andoya (Norway)": (69.294, 16.021, 0.0),
}


@dataclass
class LaunchSpec:
    """Everything that defines one launch; JSON-serialisable."""
    name: str = "Payload"
    vehicle: Vehicle = field(default_factory=lambda: vehicle_preset("Falcon 9 (approx.)"))
    site: str = ""
    lat: float = 28.562          # geodetic deg
    lon: float = -80.577         # deg east
    alt: float = 0.0             # km
    timing: str = "now"          # now | delay | absolute | raan | plane
    delay: float = 0.0           # s, timing == "delay"
    t0: float | None = None      # lift-off, s since the scenario epoch (timing == "absolute")
    guidance: str = "orbit"      # orbit (closed loop to a target orbit) | open (gravity turn)
    perigee_alt: float = 400.0   # km
    apogee_alt: float = 400.0    # km
    inclination: float = 51.6    # deg
    direction: str = "north"     # north | south (which way the vehicle crosses the pad's latitude)
    raan: float = 0.0            # deg, timing == "raan"
    target: str = ""             # satellite whose plane to launch into, timing == "plane"
    azimuth: float = 90.0        # deg from north, guidance == "open"
    vertical_time: float = 10.0  # s
    kick: float | None = None    # deg; None = optimise
    payload_mass: float = 1000.0 # kg
    payload_area: float = 5.0    # m^2 once separated
    payload_cd: float = 2.2
    circularize: bool = False    # the payload circularises at its first apogee
    track_stages: bool = True    # spent stages become objects of their own
    color: list | None = None

    def to_dict(self):
        d = asdict(self)
        d["vehicle"] = self.vehicle.to_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        s = _from_dict(cls, d)
        v = d.get("vehicle")
        if isinstance(v, str):
            s.vehicle = vehicle_preset(v)
        elif isinstance(v, dict):
            s.vehicle = Vehicle.from_dict(v)
        return s

    def copy(self) -> LaunchSpec:
        return LaunchSpec.from_dict(copy.deepcopy(self.to_dict()))


def validate(spec: LaunchSpec):
    """Raise ValueError with a readable reason if the spec cannot fly."""
    v = spec.vehicle
    if not v.stages:
        raise ValueError("the vehicle has no stages")
    for s in v.stages:
        if min(s.thrust, s.isp_vac, s.propellant) <= 0 or s.dry < 0 or s.isp_sl < 0:
            raise ValueError(f"{s.name}: thrust, Isp and propellant must be positive")
        if s.burn_time > MAX_BURN:
            raise ValueError(f"{s.name} would burn {s.burn_time / 60:.0f} min; the ascent model "
                             f"allows {MAX_BURN / 60:.0f} (use a finite-burn manoeuvre instead)")
    if spec.payload_mass < 0 or v.fairing < 0 or v.diameter < 0 or v.cd < 0:
        raise ValueError("masses, diameter and Cd cannot be negative")
    if not -90.0 <= spec.lat <= 90.0:
        raise ValueError("latitude must be within +/-90 deg")
    if spec.timing not in TIMINGS:
        raise ValueError(f"unknown launch timing {spec.timing!r}")
    if spec.guidance not in GUIDANCES:
        raise ValueError(f"unknown guidance {spec.guidance!r}")
    if spec.guidance == "orbit" and min(spec.perigee_alt, spec.apogee_alt) < MIN_PERIGEE:
        raise ValueError(f"target perigee must be at least {MIN_PERIGEE:.0f} km")
    if spec.vertical_time < 0:
        raise ValueError("vertical rise time cannot be negative")


# --- Environment and pure-Python dynamics ------------------------------------------------
# The ascent integrates one vehicle for thousands of small steps, and the
# planner flies a dozen candidates, so the hot path uses floats, not numpy.

@dataclass
class AscentEnv:
    j2: bool = True
    j3: bool = False
    j4: bool = False
    density_scale: float = 1.0
    gmst: Callable[[float], float] = lambda t: OMEGA_EARTH * t

    @classmethod
    def from_sim(cls, forces, clock):
        return cls(forces.j2, forces.j3, forces.j4, forces.density_scale,
                   lambda t: float(clock.gmst(t)))


def _density(h: float) -> float:
    if h > TOP_KM:
        return 0.0
    hc = h if h > 0.0 else 0.0
    k = bisect.bisect_right(_H0, hc) - 1
    return _RHO0[k] * math.exp(-(hc - _H0[k]) / _SH[k])


def _altitude(x, y, z) -> float:
    rm = math.sqrt(x * x + y * y + z * z)
    s = z / rm
    return rm - R_EARTH * (1.0 - F_EARTH * s * s)


def _gravity(x, y, z, env: AscentEnv):
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    k = -MU_EARTH / (r2 * r)
    ax, ay, az = k * x, k * y, k * z
    if env.j2:
        f = -1.5 * J2 * MU_EARTH * R_EARTH ** 2 / (r2 * r2 * r)
        zz = 5.0 * z * z / r2
        ax += f * x * (1.0 - zz)
        ay += f * y * (1.0 - zz)
        az += f * z * (3.0 - zz)
    if env.j3 or env.j4:
        r7 = r2 * r2 * r2 * r
        if env.j3:
            f = -2.5 * J3 * MU_EARTH * R_EARTH ** 3 / r7
            txy = 3.0 * z - 7.0 * z ** 3 / r2
            ax += f * x * txy
            ay += f * y * txy
            az += f * (6.0 * z * z - 7.0 * z ** 4 / r2 - 0.6 * r2)
        if env.j4:
            f = 1.875 * J4 * MU_EARTH * R_EARTH ** 4 / r7
            s = z * z / r2
            txy = 1.0 - 14.0 * s + 21.0 * s * s
            ax += f * x * txy
            ay += f * y * txy
            az += f * z * (5.0 - 70.0 / 3.0 * s + 21.0 * s * s)
    return ax, ay, az


def _deriv(s, tau, m0, mdot, stage, mode, fixed, cd_a, env):
    """State derivative. ``mode`` "prograde" thrusts along the air-relative
    velocity, "fixed" along the unit vector ``fixed``; mdot 0 means no thrust."""
    x, y, z, vx, vy, vz = s
    ax, ay, az = _gravity(x, y, z, env)
    wx, wy, wz = vx + OMEGA_EARTH * y, vy - OMEGA_EARTH * x, vz     # v - omega x r
    alt = _altitude(x, y, z)
    m = m0 - mdot * tau
    sp = math.sqrt(wx * wx + wy * wy + wz * wz)
    if cd_a > 0.0 and alt < TOP_KM:
        rho = _density(alt) * env.density_scale
        k = -0.5e3 * rho * cd_a / m * sp
        ax += k * wx
        ay += k * wy
        az += k * wz
    if mdot > 0.0:
        p = math.exp(-(alt if alt > 0.0 else 0.0) / P_SCALE_H)
        a = mdot * G0_M * stage.isp(p) / m * 1e-3
        if mode == "prograde" and sp > 1e-6:
            ax += a * wx / sp
            ay += a * wy / sp
            az += a * wz / sp
        else:
            ax += a * fixed[0]
            ay += a * fixed[1]
            az += a * fixed[2]
    return (vx, vy, vz, ax, ay, az)


def _rk4(s, h, m0, mdot, stage, mode, fixed, cd_a, env):
    k1 = _deriv(s, 0.0, m0, mdot, stage, mode, fixed, cd_a, env)
    s2 = tuple(a + 0.5 * h * b for a, b in zip(s, k1))
    k2 = _deriv(s2, 0.5 * h, m0, mdot, stage, mode, fixed, cd_a, env)
    s3 = tuple(a + 0.5 * h * b for a, b in zip(s, k2))
    k3 = _deriv(s3, 0.5 * h, m0, mdot, stage, mode, fixed, cd_a, env)
    s4 = tuple(a + h * b for a, b in zip(s, k3))
    k4 = _deriv(s4, h, m0, mdot, stage, mode, fixed, cd_a, env)
    return tuple(a + h / 6.0 * (b + 2.0 * c + 2.0 * d + e)
                 for a, b, c, d, e in zip(s, k1, k2, k3, k4))


def coast(state, t0: float, t1: float, mass: float, cd_a: float, env: AscentEnv):
    """Unpowered flight from t0 to t1 (used to bring separated objects up to
    the ensemble's time before they join it)."""
    s = tuple(float(x) for x in state)
    t = t0
    while t < t1 - 1e-9:
        h = min(H_COAST, t1 - t)
        s = _rk4(s, h, mass, 0.0, None, None, None, cd_a, env)
        t += h
    return s


def _energy(s) -> float:
    x, y, z, vx, vy, vz = s
    return 0.5 * (vx * vx + vy * vy + vz * vz) - MU_EARTH / math.sqrt(x * x + y * y + z * z)


def _unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 0 else (0.0, 0.0, 0.0)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _plumb_up(r, env):
    """Local vertical of the rotating Earth: opposite to gravity plus the
    centrifugal term (the geodetic normal, J2 included). Thrusting along the
    geocentric radius instead leans a slow rocket over within seconds."""
    gx, gy, gz = _gravity(r[0], r[1], r[2], env)
    w2 = OMEGA_EARTH * OMEGA_EARTH
    return _unit((-(gx + w2 * r[0]), -(gy + w2 * r[1]), -gz))


def _east_north(r):
    """Local east and north unit vectors (geocentric) at ECI position r."""
    rho = math.hypot(r[0], r[1])
    east = (-r[1] / rho, r[0] / rho, 0.0) if rho > 1e-9 else (0.0, 1.0, 0.0)
    north = _cross(_unit(r), east)
    return east, north


# --- Launch geometry --------------------------------------------------------------------

def pad_state(lat: float, lon: float, alt: float, theta: float):
    """ECI (r, v) of a point fixed on the Earth (lat/lon in degrees)."""
    r_ecef = geodetic_to_ecef(math.radians(lat), math.radians(lon), alt)
    r, v = ecef_to_eci_state(r_ecef, np.zeros(3), theta)
    return tuple(float(x) for x in r), tuple(float(x) for x in v)


def plane_normal(inc: float, raan: float):
    """Unit orbit normal for inclination and RAAN (radians)."""
    si = math.sin(inc)
    return (si * math.sin(raan), -si * math.cos(raan), math.cos(inc))


def _plane_through(r_hat, inc: float, direction: str):
    """Normal of the plane of inclination ``inc`` (rad) containing the unit
    position ``r_hat``, crossing it northbound or southbound."""
    a, b, c = r_hat
    rho = math.hypot(a, b)
    si, ci = math.sin(inc), math.cos(inc)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, c))))
    if si < 1e-9 or rho < 1e-9:
        if abs(c) < 1e-6 and si < 1e-9:
            return (0.0, 0.0, 1.0 if ci > 0 else -1.0)
        raise ValueError(_unreachable(math.degrees(inc), lat))
    q = -c * ci / (rho * si)
    if abs(q) > 1.0:
        raise ValueError(_unreachable(math.degrees(inc), lat))
    lam = math.atan2(b, a)
    best = None
    for psi in (lam + math.acos(q), lam - math.acos(q)):
        n = (si * math.cos(psi), si * math.sin(psi), ci)
        north = _cross(n, r_hat)[2] >= 0.0
        if north == (direction == "north"):
            best = n
    if best is None:     # launching exactly at the plane's northern/southern extreme
        best = (si * math.cos(lam + math.acos(q)), si * math.sin(lam + math.acos(q)), ci)
    return best


def _unreachable(inc_deg: float, lat_deg: float) -> str:
    lo = abs(lat_deg)
    return (f"inclination {inc_deg:.1f} deg cannot be reached directly from latitude "
            f"{lat_deg:+.1f} deg: choose {lo:.1f} to {180 - lo:.1f} deg "
            "(dog-leg ascents are not modelled)")


def next_window(lat: float, lon: float, alt: float, normal_at, t_start: float, gmst,
                direction: str = "north", span: float = 2 * 86400.0, step: float = 120.0) -> float:
    """First time >= t_start at which the pad lies in the plane ``normal_at(t)``
    and the plane crosses it in ``direction``."""
    r_ecef = geodetic_to_ecef(math.radians(lat), math.radians(lon), alt)
    X, Y, Z = (float(x) for x in r_ecef / np.linalg.norm(r_ecef))

    def pad(t):
        th = gmst(t)
        c, s = math.cos(th), math.sin(th)
        return (c * X - s * Y, s * X + c * Y, Z)

    def f(t):
        return _dot(pad(t), normal_at(t))

    n0 = normal_at(t_start)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, n0[2]))))
    lat_gc = math.degrees(math.asin(Z))
    if not abs(lat_gc) - 1e-6 <= inc <= 180.0 - abs(lat_gc) + 1e-6:
        raise ValueError(_unreachable(inc, lat_gc))
    ta, fa = t_start, f(t_start)
    while ta < t_start + span:
        tb = ta + step
        fb = f(tb)
        if fa == 0.0 or fa * fb < 0.0:
            lo, hi, flo = ta, tb, fa
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            t = 0.5 * (lo + hi)
            north = _cross(normal_at(t), pad(t))[2] >= 0.0
            if north == (direction == "north") or inc < 1e-3 or inc > 180 - 1e-3:
                return t
        ta, fa = tb, fb
    raise ValueError("no launch window in the next two days")


@dataclass
class Resolved:
    """A launch spec turned into numbers: lift-off time, plane and targets."""
    t0: float
    azimuth: float                  # rad from north, relative to the rotating Earth
    n: tuple | None = None          # target plane normal (orbit guidance)
    inclination: float = math.nan   # deg
    r_target: float = math.nan      # km, insertion (perigee) radius
    v_target: float = math.nan      # km/s, speed there
    e_target: float = math.nan      # km^2/s^2, specific energy of the target orbit


def resolve(spec: LaunchSpec, env: AscentEnv, t_now: float, plane_of=None) -> Resolved:
    """Lift-off time, target plane and launch azimuth. ``plane_of(name)``
    returns ``(inclination deg, normal_at(t))`` for another satellite."""
    validate(spec)
    normal_at = None
    inc = spec.inclination
    if spec.guidance == "orbit" and spec.timing == "raan":
        n_fix = plane_normal(math.radians(inc), math.radians(spec.raan))
        normal_at = lambda t: n_fix        # noqa: E731
    elif spec.guidance == "orbit" and spec.timing == "plane":
        if plane_of is None or not spec.target:
            raise ValueError("choose the satellite whose plane to launch into")
        inc, normal_at = plane_of(spec.target)
    if spec.timing == "delay":
        t0 = t_now + max(0.0, spec.delay)
    elif spec.timing == "absolute":
        t0 = max(t_now, spec.t0 if spec.t0 is not None else t_now)
    elif normal_at is not None:
        t0 = next_window(spec.lat, spec.lon, spec.alt, normal_at, t_now + 1.0, env.gmst,
                         spec.direction)
    else:
        t0 = t_now
    if spec.guidance == "open":
        return Resolved(t0, math.radians(spec.azimuth))

    hp, ha = sorted((spec.perigee_alt, spec.apogee_alt))
    r_t = R_EARTH + hp
    a_t = R_EARTH + 0.5 * (hp + ha)
    v_t = math.sqrt(MU_EARTH * (2.0 / r_t - 1.0 / a_t))
    r0, _ = pad_state(spec.lat, spec.lon, spec.alt, env.gmst(t0))
    r_hat = _unit(r0)
    n = normal_at(t0) if normal_at else _plane_through(r_hat, math.radians(inc), spec.direction)
    # inertial azimuth of the in-plane direction, then the one to fly relative
    # to the rotating Earth so that the inertial velocity ends up in the plane
    d = _unit(_cross(n, r_hat))
    east, north = _east_north(r0)
    beta = math.atan2(_dot(d, east), _dot(d, north))
    v_rot = OMEGA_EARTH * math.hypot(r0[0], r0[1])
    az = math.atan2(v_t * math.sin(beta) - v_rot, v_t * math.cos(beta))
    return Resolved(t0, az, n, math.degrees(math.acos(max(-1.0, min(1.0, n[2])))),
                    r_t, v_t, -MU_EARTH / (2.0 * a_t))


# --- The ascent -----------------------------------------------------------------------------

PHASE_LABELS = {"pad": "On the pad", "vertical": "Vertical rise", "kick": "Pitch kick",
                "turn": "Gravity turn", "guided": "Closed-loop guidance",
                "released": "Separated"}


class Ascent:
    """One vehicle from the pad to payload separation.

    ``advance_to(t)`` integrates up to simulation time ``t``; events and spent
    stages accumulate in ``events`` / ``spawns`` for the owner to collect.
    After ``released`` the object is the bare payload in free flight."""

    def __init__(self, spec: LaunchSpec, res: Resolved, env: AscentEnv, kick: float | None = None,
                 record: bool = True):
        self.spec = spec
        self.veh = spec.vehicle
        self.res = res
        self.env = env
        k = kick if kick is not None else (spec.kick if spec.kick is not None else 0.0)
        self.kick_deg = float(k)
        self.kick = math.radians(k)
        self.record = record
        self.t0 = res.t0
        self.t = res.t0
        self.y = self.pad_state(res.t0)
        self.phase = "pad"
        self.k = 0
        self.prop = self.veh.stages[0].propellant
        self.mass = self.veh.liftoff_mass(spec.payload_mass)
        self.fairing_on = self.veh.fairing > 0
        self.burning = False
        self.coast_until = -math.inf
        self.throttle = 1.0
        self.thrust_dir = _unit(self.y[:3])
        self.hold = None
        self.tgo = math.nan
        self.kick_start = math.nan
        self.sat = None                 # the Satellite this ascent drives (owner's business)
        self.released = False
        self.outcome = ""               # orbit | suborbital | short | crash | no_liftoff
        self.events: list = []          # (t, text, kind)
        self.spawns: list = []          # (t, name, state, mass, cd*area)
        self.q = 0.0
        self.max_q = (0.0, math.nan)
        self._q_logged = False
        self.accel = 0.0
        self.dv_ideal = 0.0
        self.losses = {"gravity": 0.0, "drag": 0.0, "steering": 0.0}
        self.v_start = math.sqrt(sum(v * v for v in self.y[3:]))
        self.samples: list = []         # (t since lift-off, alt, downrange, speed, q kPa, accel g)
        self.track: list = []           # (lat deg, lon deg)
        self.stage_marks: list = []     # (t since lift-off, alt, downrange, label)
        self._next_sample = -math.inf
        self.insertion = None           # dict of the orbit at release
        self.planned = None             # (samples, stage marks, outcome text) of the plan
        r_ecef = geodetic_to_ecef(math.radians(spec.lat), math.radians(spec.lon), spec.alt)
        self._pad_hat = tuple(float(x) for x in r_ecef / np.linalg.norm(r_ecef))

    # --- helpers ----------------------------------------------------------------------
    def pad_state(self, t: float):
        r, v = pad_state(self.spec.lat, self.spec.lon, self.spec.alt, self.env.gmst(t))
        return r + v

    @property
    def stage(self) -> Stage:
        return self.veh.stages[min(self.k, len(self.veh.stages) - 1)]

    @property
    def cd_a(self) -> float:
        return self.veh.cd * self.veh.area

    @property
    def flying(self) -> bool:
        return self.phase != "pad" and not self.released

    @property
    def met(self) -> float:
        """Mission elapsed time: seconds since lift-off (negative before)."""
        return self.t - self.t0

    def state(self) -> np.ndarray:
        return np.array(self.y, dtype=float)

    def _event(self, text: str, kind: str = "launch"):
        self.events.append((self.t, f"{self.spec.name}: {text}", kind))

    def _geo(self, t, s):
        """(geocentric lat deg, lon deg, downrange km) of state s at time t."""
        th = self.env.gmst(t)
        c, sn = math.cos(th), math.sin(th)
        x, y, z = s[0], s[1], s[2]
        xe, ye = c * x + sn * y, -sn * x + c * y
        u = _unit((xe, ye, z))
        ang = math.acos(max(-1.0, min(1.0, _dot(u, self._pad_hat))))
        return math.degrees(math.asin(u[2])), math.degrees(math.atan2(u[1], u[0])), ang * R_EARTH

    def _sample(self, force: bool = False):
        if not self.record or (not force and self.t < self._next_sample):
            return
        self._next_sample = self.t + SAMPLE_DT
        lat, lon, dr = self._geo(self.t, self.y)
        alt = _altitude(*self.y[:3])
        speed = math.sqrt(sum(v * v for v in self.y[3:]))
        self.samples.append((self.met, alt, dr, speed, self.q / 1000.0, self.accel / G0_M))
        self.track.append((lat, lon))

    def _mark(self, label: str):
        if self.record:
            lat, lon, dr = self._geo(self.t, self.y)
            self.stage_marks.append((self.met, _altitude(*self.y[:3]), dr, label))

    # --- flight -------------------------------------------------------------------------
    def advance_to(self, t_end: float):
        """Fly (or wait on the pad) up to time ``t_end``."""
        if self.phase == "pad":
            if t_end < self.t0 - 1e-9:
                self.y = self.pad_state(t_end)
                self.t = t_end
                return
            self.t = self.t0
            self.y = self.pad_state(self.t0)
            self._liftoff()
        guard = 0
        while not self.released and self.t < t_end - 1e-9 and guard < 200000:
            guard += 1
            self._step(min(H_POWERED, t_end - self.t))
        if self.released and math.isfinite(t_end) and self.t < t_end - 1e-9:
            self.y = coast(self.y, self.t, t_end, self.mass, self.spec.payload_cd * self.spec.payload_area,
                           self.env)
            self.t = t_end

    def _liftoff(self):
        st = self.veh.stages[0]
        p = math.exp(-max(self.spec.alt, 0.0) / P_SCALE_H)
        thrust = st.mdot * G0_M * st.isp(p)
        r = math.sqrt(sum(c * c for c in self.y[:3]))
        weight = self.mass * MU_EARTH / (r * r) * 1e3
        self.phase = "vertical"
        if thrust <= weight:
            self._release("no_liftoff", f"scrubbed - thrust/weight {thrust / weight:.2f} is below 1, "
                                        "the vehicle cannot lift off", "alert")
            return
        self.burning = True
        self._event(f"lift-off on {self.veh.name} from {self.spec.site or 'the pad'} "
                    f"(T/W {thrust / weight:.2f})")
        self._mark("Lift-off")
        self._sample(force=True)

    def _throttle(self, st: Stage, alt: float) -> float:
        if self.veh.max_g <= 0:
            return 1.0
        p = math.exp(-max(alt, 0.0) / P_SCALE_H)
        a = st.mdot * G0_M * st.isp(p) / self.mass
        return max(0.4, min(1.0, self.veh.max_g * G0_M / a))

    def _step(self, h: float):
        s0, t, m0 = self.y, self.t, self.mass
        alt = _altitude(*s0[:3])
        if not self.burning and self.k < len(self.veh.stages) and self.coast_until <= t + 1e-9:
            self.burning = True
            self._event(f"{self.stage.name} ignition at {alt:.0f} km", "launch")
            self._mark(f"{self.stage.name} ign.")
        st = self.stage
        mode, fixed, mdot = None, None, 0.0
        burnout = cutoff = False
        if self.burning:
            mode, fixed = self._steer(s0, alt)
            self.throttle = self._throttle(st, alt)
            mdot = st.mdot * self.throttle
            if mdot * h >= self.prop:
                h = self.prop / mdot
                burnout = True
        elif self.coast_until > t:
            h = min(h, self.coast_until - t)
        env, cd_a = self.env, self.cd_a
        s1 = _rk4(s0, h, m0, mdot, st, mode, fixed, cd_a, env)
        if self.burning and self.phase == "guided":
            e1 = _energy(s1)
            if e1 >= self.res.e_target:
                e0 = _energy(s0)
                f = min(1.0, max(0.0, (self.res.e_target - e0) / (e1 - e0))) if e1 > e0 else 1.0
                h *= f
                s1 = _rk4(s0, h, m0, mdot, st, mode, fixed, cd_a, env)
                cutoff, burnout = True, False
        self._account(s0, m0, mdot, st, mode, fixed, alt, h)
        self.t = t + h
        self.y = s1
        self.mass = m0 - mdot * h
        self.prop = max(0.0, self.prop - mdot * h)
        self._after_step(cutoff, burnout)

    def _account(self, s, m, mdot, st, mode, fixed, alt, h):
        """Telemetry and the delta-v budget, from the state at the start of a step."""
        x, y, z, vx, vy, vz = s
        v = math.sqrt(vx * vx + vy * vy + vz * vz)
        vh = (vx / v, vy / v, vz / v) if v > 0 else (0.0, 0.0, 0.0)
        wx, wy, wz = vx + OMEGA_EARTH * y, vy - OMEGA_EARTH * x, vz
        sp = math.sqrt(wx * wx + wy * wy + wz * wz)
        rho = _density(alt) * self.env.density_scale if alt < TOP_KM else 0.0
        self.q = 0.5 * rho * (sp * 1e3) ** 2
        g = _gravity(x, y, z, self.env)
        self.losses["gravity"] -= _dot(g, vh) * h
        if m > 0 and sp > 0:
            drag = 0.5e3 * rho * self.cd_a / m * sp
            self.losses["drag"] += drag * _dot((wx, wy, wz), vh) * h
        if mdot > 0:
            p = math.exp(-max(alt, 0.0) / P_SCALE_H)
            a = mdot * G0_M * st.isp(p) / m
            self.accel = a
            d = _unit((wx, wy, wz)) if mode == "prograde" and sp > 1e-6 else fixed
            self.thrust_dir = d
            self.dv_ideal += a * 1e-3 * h
            self.losses["steering"] += a * 1e-3 * (1.0 - _dot(d, vh)) * h
        else:
            self.accel = 0.0
        if self.q > self.max_q[0]:
            self.max_q = (self.q, self.t - self.t0)
        elif not self._q_logged and self.max_q[0] > 1000.0 and self.q < 0.9 * self.max_q[0]:
            self._q_logged = True
            self._event(f"max Q {self.max_q[0] / 1000:.1f} kPa at T+{self.max_q[1]:.0f} s", "info")

    def _after_step(self, cutoff: bool, burnout: bool):
        alt = _altitude(*self.y[:3])
        if self.fairing_on and alt >= self.veh.fairing_alt:
            self.fairing_on = False
            self.mass -= self.veh.fairing
            self._event(f"fairing jettison at {alt:.0f} km", "info")
            self._mark("Fairing")
        vr = _dot(self.y[:3], self.y[3:])
        if alt < min(0.0, self.spec.alt) - 0.1 and vr < 0:
            self._release("crash", f"vehicle impacted the ground at T+{self.met:.0f} s", "alert")
            return
        if cutoff:
            self._insertion("orbit")
        elif burnout:
            self._burnout()
        self._sample()

    def _burnout(self):
        st = self.stage
        last = self.k == len(self.veh.stages) - 1
        alt = _altitude(*self.y[:3])
        self.burning = False
        if last:
            if self.spec.guidance == "orbit":
                self._insertion("short")
            else:
                self._insertion("orbit")
            return
        label = "MECO" if self.k == 0 else f"{st.name} burnout"
        self._event(f"{label} at {alt:.0f} km, {math.sqrt(sum(v * v for v in self.y[3:])):.2f} km/s; "
                    f"{st.name} separation", "launch")
        self._mark(f"{st.name} sep.")
        self._drop(st.name, st.dry)
        self.k += 1
        self.prop = self.veh.stages[self.k].propellant
        self.coast_until = self.t + self.veh.stage_coast

    def _drop(self, name: str, mass: float):
        self.mass -= mass
        if self.spec.track_stages and mass > 0:
            self.spawns.append((self.t, f"{self.spec.name} {name}", self.y, mass, self.cd_a))

    def _insertion(self, outcome: str):
        """Cut-off (or last burnout): the payload separates from the last stage."""
        st = self.stage
        el = rv2coe(np.array(self.y[:3]), np.array(self.y[3:]))
        hp = el.rp - R_EARTH
        ha = el.ra - R_EARTH if el.e < 1 else math.inf
        self.insertion = dict(t=self.met, hp=hp, ha=ha, i=math.degrees(el.i), raan=math.degrees(el.raan),
                              e=el.e, a=el.a, prop_left=self.prop, alt=_altitude(*self.y[:3]))
        closed = el.e < 1 and hp > MIN_PERIGEE
        if outcome == "orbit" and not closed:
            outcome = "escape" if el.e >= 1 else "suborbital"
        orbit = (f"{hp:,.0f} x {ha:,.0f} km, i {math.degrees(el.i):.2f} deg" if el.e < 1
                 else f"escape, e {el.e:.2f}")
        if outcome == "orbit" and self.spec.guidance == "orbit":
            text = f"SECO at T+{self.met:.0f} s - orbit {orbit}; payload separation"
            kind = "launch"
        elif outcome == "orbit" or outcome == "escape":
            text = f"final burnout at T+{self.met:.0f} s - {orbit}; payload separation"
            kind = "launch"
        elif outcome == "short":
            text = (f"{st.name} ran out of propellant before orbit (T+{self.met:.0f} s, "
                    f"perigee {hp:,.0f} km) - payload separation")
            kind = "alert"
        else:
            text = f"burnout at T+{self.met:.0f} s on a sub-orbital path (apogee {ha:,.0f} km)"
            kind = "launch"
        self._mark("Cut-off" if outcome == "orbit" else "Burnout")
        self._sample(force=True)
        remaining = self.mass - self.spec.payload_mass
        self._drop(st.name, remaining)
        self.mass = self.spec.payload_mass
        self._release(outcome, text, kind)

    def _release(self, outcome: str, text: str, kind: str):
        self.released = True
        self.burning = False
        self.outcome = outcome
        self.phase = "released"
        self._event(text, kind)

    # --- steering ----------------------------------------------------------------------------
    def _steer(self, s, alt):
        r = s[:3]
        up = _plumb_up(r, self.env)
        if self.phase == "vertical":
            if self.met < self.spec.vertical_time - 1e-9:
                return "fixed", up
            if self.kick > 1e-6:
                self.phase = "kick"
                self.kick_start = self.t
                self._event(f"pitch kick {self.kick_deg:.2f} deg toward azimuth "
                            f"{math.degrees(self.res.azimuth) % 360:.1f} deg", "info")
            else:
                self.phase = "turn"
        if self.phase == "kick":
            w = (s[3] + OMEGA_EARTH * s[1], s[4] - OMEGA_EARTH * s[0], s[5])
            sp = math.sqrt(_dot(w, w))
            lean = math.acos(max(-1.0, min(1.0, _dot(w, up) / sp))) if sp > 1e-6 else 0.0
            if lean >= self.kick or self.t - self.kick_start > 40.0:
                self.phase = "turn"
            else:
                east, north = _east_north(r)
                az = self.res.azimuth
                hz = tuple(math.sin(az) * e + math.cos(az) * n for e, n in zip(east, north))
                ck, sk = math.cos(self.kick), math.sin(self.kick)
                return "fixed", _unit(tuple(ck * u + sk * q for u, q in zip(up, hz)))
        if self.phase == "turn":
            if self.spec.guidance == "orbit" and (self.k >= 1 or alt >= GUIDANCE_ALT):
                self.phase = "guided"
                self._event(f"closed-loop guidance at {alt:.0f} km", "info")
            else:
                return "prograde", None
        return "fixed", self._guided(s)

    def _burn_time(self, dv: float) -> float:
        """Time (s) the remaining stages need to deliver ``dv`` (km/s),
        coasts between stages included; all of it if they cannot."""
        stages = self.veh.stages
        t = max(0.0, self.coast_until - self.t) if not self.burning else 0.0
        k, prop, mass = self.k, self.prop, self.mass
        while k < len(stages):
            st = stages[k]
            ve = st.isp_vac * G0_M / 1000.0
            mdot = st.mdot
            if prop > 0:
                dv_k = ve * math.log(mass / (mass - prop))
                if dv <= dv_k:
                    return t + mass * (1.0 - math.exp(-dv / ve)) / mdot
                t += prop / mdot
                dv -= dv_k
            mass -= prop + st.dry
            k += 1
            if k < len(stages):
                t += self.veh.stage_coast
                prop = stages[k].propellant
        return t

    def _guided(self, s):
        r, v = s[:3], s[3:]
        rm = math.sqrt(_dot(r, r))
        up = (r[0] / rm, r[1] / rm, r[2] / rm)
        n = self.res.n
        eh = _unit(_cross(n, up))
        vr, vh, vn, rn = _dot(v, up), _dot(v, eh), _dot(v, n), _dot(r, n)
        r_t, v_t = self.res.r_target, self.res.v_target
        g_eff = MU_EARTH / rm ** 2 - vh * vh / rm
        g_t = MU_EARTH / r_t ** 2 - v_t * v_t / r_t
        dvh = v_t - vh
        tgo = self._burn_time(math.sqrt(dvh * dvh + vn * vn + vr * vr))
        for _ in range(2):
            dv_r = -vr + 0.5 * (g_eff + g_t) * tgo
            tgo = self._burn_time(math.sqrt(dvh * dvh + vn * vn + dv_r * dv_r))
        tgo = max(tgo, 1.0)
        self.tgo = tgo
        if tgo < FREEZE_TGO and self.hold is not None:
            s_r, s_n = self.hold
        else:
            st = self.stage
            p = math.exp(-max(_altitude(*r), 0.0) / P_SCALE_H)
            a_t = st.mdot * self.throttle * G0_M * st.isp(p) / self.mass * 1e-3
            b = -12.0 * (r_t - rm - 0.5 * vr * tgo) / tgo ** 3
            a = (-vr - 0.5 * b * tgo * tgo) / tgo
            bn = -12.0 * (-rn - 0.5 * vn * tgo) / tgo ** 3
            an = (-vn - 0.5 * bn * tgo * tgo) / tgo
            s_r = max(-0.7, min(0.95, (a + g_eff) / a_t))
            s_n = max(-0.5, min(0.5, an / a_t))
            if s_r * s_r + s_n * s_n > 0.995:
                s_n = math.copysign(math.sqrt(max(0.0, 0.995 - s_r * s_r)), s_n)
            self.hold = (s_r, s_n)
        s_h = math.sqrt(max(0.0, 1.0 - s_r * s_r - s_n * s_n))
        return _unit(tuple(s_r * u + s_n * q + s_h * e for u, q, e in zip(up, n, eh)))

    # --- reporting --------------------------------------------------------------------------
    def summary(self) -> dict:
        """Figures for panels and the planner (valid once released)."""
        ins = self.insertion or {}
        return dict(outcome=self.outcome, kick=self.kick_deg, met=self.met,
                    dv_ideal=self.dv_ideal, losses=dict(self.losses), v_start=self.v_start,
                    max_q=self.max_q[0] / 1000.0, t_max_q=self.max_q[1], **ins)


# --- Planning -------------------------------------------------------------------------------

@dataclass
class LaunchPlan:
    res: Resolved
    kick: float                      # deg
    flight: Ascent                   # the simulated flight with that kick
    candidates: list                 # (kick deg, score)

    @property
    def ok(self) -> bool:
        return self.flight.outcome == "orbit"


def fly(spec: LaunchSpec, res: Resolved, env: AscentEnv, kick: float, record: bool = True) -> Ascent:
    """Simulate a whole ascent to payload separation (no side effects)."""
    a = Ascent(spec, res, env, kick, record=record)
    a.advance_to(math.inf)
    return a


def _score(a: Ascent, spec: LaunchSpec) -> float:
    ins = a.insertion
    if ins is None:
        return -1e12
    if spec.guidance == "open":
        # reach orbit with the highest perigee if possible, otherwise go highest
        if a.outcome == "orbit":
            return 1e7 + ins["hp"]
        return ins["ha"] if math.isfinite(ins["ha"]) else 1e7
    if a.outcome == "orbit":
        miss = abs(ins["hp"] - min(spec.perigee_alt, spec.apogee_alt)) + \
            abs(ins["ha"] - max(spec.perigee_alt, spec.apogee_alt))
        return 1e7 + ins["prop_left"] - 20.0 * miss
    # short of orbit: rank by the energy reached
    e = -MU_EARTH / (2.0 * ins["a"]) if ins["e"] < 1 else 0.0
    return -1e7 + e * 1e3


def plan_launch(spec: LaunchSpec, env: AscentEnv, t_now: float, plane_of=None,
                res: Resolved | None = None) -> LaunchPlan:
    """Resolve the launch and fly it; with ``spec.kick`` None, search the kick
    angle that reaches the target with the most propellant left."""
    res = res or resolve(spec, env, t_now, plane_of)
    if spec.kick is not None:
        a = fly(spec, res, env, spec.kick)
        return LaunchPlan(res, spec.kick, a, [(spec.kick, _score(a, spec))])
    tried: dict = {}

    def score(k):
        k = round(k, 3)
        if k not in tried:
            tried[k] = _score(fly(spec, res, env, k, record=False), spec)
        return tried[k]

    # low thrust-to-weight vehicles need tiny kicks, sprightly ones large
    grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0, 16.0]
    best = max(grid, key=score)
    j = grid.index(best)
    lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(7):
        if score(a_) >= score(b_):
            hi, b_ = b_, a_
            a_ = hi - g * (hi - lo)
        else:
            lo, a_ = a_, b_
            b_ = lo + g * (hi - lo)
    kick = max(tried, key=tried.get)
    flight = fly(spec, res, env, kick)
    return LaunchPlan(res, kick, flight, sorted(tried.items()))


__all__ = ["Stage", "Vehicle", "LaunchSpec", "VEHICLES", "LAUNCH_SITES", "AscentEnv", "Ascent",
           "Resolved", "LaunchPlan", "resolve", "plan_launch", "fly", "vehicle_preset", "validate",
           "next_window", "pad_state", "plane_normal", "coast"]
