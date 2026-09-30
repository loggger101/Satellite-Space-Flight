"""Scenarios: the declarative, JSON-serialisable description of a run.

A scenario lists satellites (each with an *orbit spec*), Walker
constellations, ground stations, scheduled manoeuvres, the force model and
the propagator settings. Orbit specs are dicts with a ``type``:

``elements``  classical elements. Size by ``a`` (km), ``altitude`` (circular,
              km), or ``perigee_alt`` + ``apogee_alt`` (km); angles in degrees:
              ``i``, ``raan``, ``argp``, ``nu`` (or ``M``). ``"i": "sso"`` picks
              the sun-synchronous inclination.
``state``     ECI ``r`` [km] and ``v`` [km/s].
``surface``   launch from the ground: ``lat``, ``lon`` (deg), ``alt`` (km),
              ``speed`` (km/s relative to the rotating Earth), ``azimuth``
              (deg from north), ``fpa`` (flight-path angle above horizon).
``geo``       geostationary slot at east ``lon`` (deg).
``tle``       ``line1`` / ``line2``.

``count`` > 1 on a satellite spreads copies evenly in mean anomaly.

``launches`` lists rockets that lift off from a pad during the run (see
``satflight.launch.LaunchSpec``): vehicle stages, site, target orbit, timing.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH, R_GEO
from .elements import coe2rv, mean_to_true, rv2coe
from .forces import ForceModel
from .launch import LaunchSpec
from .frames import ecef_to_eci_state, enu_matrix, geodetic_to_ecef
from .maneuvers import Maneuver, sun_synchronous_inclination
from .timeutil import Clock, format_epoch, parse_epoch, UTC

PALETTE = [
    (255, 196, 64), (90, 200, 255), (255, 110, 110), (140, 240, 140), (220, 140, 255),
    (255, 160, 60), (80, 240, 220), (255, 120, 200), (200, 220, 90), (160, 170, 255),
    (255, 230, 150), (120, 200, 160),
]


def palette_color(i: int):
    return PALETTE[i % len(PALETTE)]


# --- Specs -----------------------------------------------------------------------

def _from_dict(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class SatSpec:
    name: str
    orbit: dict
    color: list | None = None
    mass: float = 500.0      # kg
    area: float = 5.0        # m^2 (drag cross-section)
    cd: float = 2.2
    count: int = 1

    def to_dict(self):
        d = asdict(self)
        if d["color"] is not None:
            d["color"] = list(d["color"])
        return d


@dataclass
class ConstellationSpec:
    """Walker constellation i:t/p/f (delta: nodes over 360 deg, star: 180 deg)."""
    name: str = "Walker"
    altitude: float = 550.0
    inclination: float = 53.0
    total: int = 24
    planes: int = 6
    phasing: int = 1
    pattern: str = "delta"
    color: list | None = None
    mass: float = 260.0
    area: float = 4.0
    cd: float = 2.2
    raan0: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class GroundStation:
    name: str
    lat: float            # deg
    lon: float            # deg
    alt: float = 0.0      # km
    min_el: float = 10.0  # deg
    color: list | None = None

    def ecef(self) -> np.ndarray:
        return geodetic_to_ecef(math.radians(self.lat), math.radians(self.lon), self.alt)

    def to_dict(self):
        return asdict(self)


@dataclass
class IntegratorSettings:
    method: str = "dopri5"
    rtol: float = 1e-9
    atol: float = 1e-6
    h_max: float = 60.0
    h_fixed: float = 10.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Scenario:
    name: str = "Untitled"
    description: str = ""
    epoch: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    satellites: list = field(default_factory=list)
    constellations: list = field(default_factory=list)
    stations: list = field(default_factory=list)
    maneuvers: list = field(default_factory=list)
    launches: list = field(default_factory=list)      # LaunchSpec
    forces: ForceModel = field(default_factory=ForceModel)
    propagator: str = "cowell"           # cowell | kepler | j2mean
    integrator: IntegratorSettings = field(default_factory=IntegratorSettings)
    record_dt: float = 20.0              # s between trail / telemetry samples
    warp: float = 60.0                   # initial time acceleration
    # out-of-scope forces the source file asked for (not saved back)
    out_of_scope: list = field(default_factory=list, repr=False)

    # --- serialisation ---
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "epoch": format_epoch(self.epoch),
            "propagator": self.propagator,
            "integrator": self.integrator.to_dict(),
            "forces": self.forces.to_dict(),
            "record_dt": self.record_dt,
            "warp": self.warp,
            "satellites": [s.to_dict() for s in self.satellites],
            "constellations": [c.to_dict() for c in self.constellations],
            "stations": [g.to_dict() for g in self.stations],
            "maneuvers": [m.to_dict() for m in self.maneuvers],
            "launches": [s.to_dict() for s in self.launches],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            name=d.get("name", "Untitled"),
            description=d.get("description", ""),
            epoch=parse_epoch(d["epoch"]) if "epoch" in d else datetime(2026, 1, 1, tzinfo=UTC),
            propagator=d.get("propagator", "cowell"),
            integrator=_from_dict(IntegratorSettings, d.get("integrator", {})),
            forces=ForceModel.from_dict(d.get("forces", {})),
            record_dt=float(d.get("record_dt", 20.0)),
            warp=float(d.get("warp", 60.0)),
            satellites=[_from_dict(SatSpec, s) for s in d.get("satellites", [])],
            constellations=[_from_dict(ConstellationSpec, c) for c in d.get("constellations", [])],
            stations=[_from_dict(GroundStation, g) for g in d.get("stations", [])],
            maneuvers=[Maneuver.from_dict(m) for m in d.get("maneuvers", [])],
            launches=[LaunchSpec.from_dict(s) for s in d.get("launches", [])],
            out_of_scope=ForceModel.ignored_terms(d.get("forces", {})),
        )

    @classmethod
    def load(cls, path) -> "Scenario":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def copy(self) -> "Scenario":
        return Scenario.from_dict(copy.deepcopy(self.to_dict()))


# --- Building initial states ---------------------------------------------------------

def _deg(d, key, default=0.0):
    return math.radians(float(d.get(key, default)))


def orbit_state(orbit: dict, clock: Clock, t: float = 0.0):
    """ECI (r, v) at simulation time ``t`` for an orbit spec."""
    kind = orbit.get("type", "elements")
    if kind == "elements":
        if "perigee_alt" in orbit or "apogee_alt" in orbit:
            rp = R_EARTH + float(orbit.get("perigee_alt", orbit.get("apogee_alt")))
            ra = R_EARTH + float(orbit.get("apogee_alt", orbit.get("perigee_alt")))
            rp, ra = min(rp, ra), max(rp, ra)
            a = 0.5 * (rp + ra)
            e = (ra - rp) / (ra + rp)
        elif "altitude" in orbit:
            a = R_EARTH + float(orbit["altitude"])
            e = float(orbit.get("e", 0.0))
        else:
            a = float(orbit["a"])
            e = float(orbit.get("e", 0.0))
        if e >= 1.0 and a > 0:
            a = -abs(a)          # hyperbola: a < 0 by convention
        inc = orbit.get("i", 0.0)
        if isinstance(inc, str) and inc.lower() == "sso":
            inc = math.degrees(sun_synchronous_inclination(a, e))
        i = math.radians(float(inc))
        raan, argp = _deg(orbit, "raan"), _deg(orbit, "argp")
        if "M" in orbit and "nu" not in orbit:
            nu = mean_to_true(math.radians(float(orbit["M"])), e)
        else:
            nu = _deg(orbit, "nu")
        r, v = coe2rv(a, e, i, raan, argp, nu)
        return np.asarray(r, float), np.asarray(v, float)
    if kind == "state":
        return np.asarray(orbit["r"], float), np.asarray(orbit["v"], float)
    if kind == "surface":
        lat, lon = _deg(orbit, "lat"), _deg(orbit, "lon")
        r_ecef = geodetic_to_ecef(lat, lon, float(orbit.get("alt", 0.0)))
        az, fpa = _deg(orbit, "azimuth", 90.0), _deg(orbit, "fpa", 0.0)
        speed = float(orbit.get("speed", 7.8))
        enu = np.array([speed * math.cos(fpa) * math.sin(az),
                        speed * math.cos(fpa) * math.cos(az),
                        speed * math.sin(fpa)])
        v_ecef = enu @ enu_matrix(lat, lon)
        return ecef_to_eci_state(r_ecef, v_ecef, clock.gmst(t))
    if kind == "geo":
        ang = clock.gmst(t) + math.radians(float(orbit.get("lon", 0.0)))
        r = R_GEO * np.array([math.cos(ang), math.sin(ang), 0.0])
        vmag = OMEGA_EARTH * R_GEO
        v = vmag * np.array([-math.sin(ang), math.cos(ang), 0.0])
        return r, v
    if kind == "tle":
        from .tle import parse_tle
        tle = parse_tle(orbit["line1"] + "\n" + orbit["line2"])
        return tle.state_at(clock.datetime(t))
    raise ValueError(f"unknown orbit type {kind!r}")


def spread_along_orbit(r, v, count: int):
    """``count`` states sharing an orbit, evenly spaced in mean anomaly."""
    if count <= 1:
        return [(r, v)]
    el = rv2coe(r, v)
    if el.e >= 1.0:
        raise ValueError("cannot spread satellites along an open orbit")
    out = []
    for k in range(count):
        M = el.M + 2 * math.pi * k / count
        nu = mean_to_true(M, el.e)
        rr, vv = coe2rv(el.a, el.e, el.i, el.raan, el.argp, nu)
        out.append((np.asarray(rr), np.asarray(vv)))
    return out


def walker_states(c: ConstellationSpec):
    """ECI states of a Walker i:t/p/f constellation (circular orbits)."""
    if c.total % c.planes:
        raise ValueError("total satellites must be a multiple of the number of planes")
    per = c.total // c.planes
    spread = 2 * math.pi if c.pattern == "delta" else math.pi
    a = R_EARTH + c.altitude
    i = math.radians(c.inclination)
    out = []
    for p in range(c.planes):
        raan = math.radians(c.raan0) + spread * p / c.planes
        for s in range(per):
            u = 2 * math.pi * s / per + 2 * math.pi * c.phasing * p / c.total
            r, v = coe2rv(a, 0.0, i, raan, 0.0, u)
            out.append((f"{c.name}-{p + 1:02d}{s + 1:02d}", np.asarray(r), np.asarray(v)))
    return out


def expand(scenario: Scenario, clock: Clock, t: float = 0.0):
    """Every satellite as ``(name, color, props, r, v)`` - the simulation's input."""
    out = []
    idx = 0
    for spec in scenario.satellites:
        r, v = orbit_state(spec.orbit, clock, t)
        states = spread_along_orbit(r, v, int(spec.count))
        for k, (rr, vv) in enumerate(states):
            name = spec.name if len(states) == 1 else f"{spec.name}-{k + 1}"
            color = tuple(spec.color) if spec.color else palette_color(idx)
            props = dict(mass=spec.mass, area=spec.area, cd=spec.cd)
            out.append((name, color, props, rr, vv))
            idx += 1
    for c in scenario.constellations:
        color = tuple(c.color) if c.color else palette_color(idx)
        props = dict(mass=c.mass, area=c.area, cd=c.cd)
        for name, rr, vv in walker_states(c):
            out.append((name, color, props, rr, vv))
        idx += 1
    return out


# --- Presets used by the UI and examples ------------------------------------------------

PRESETS = {
    "ISS (LEO 420 km, 51.6 deg)": dict(orbit={"type": "elements", "altitude": 420, "i": 51.64,
                                             "raan": 30, "argp": 0, "nu": 0},
                                      mass=420000, area=1600, cd=2.2),
    "Hubble (540 km, 28.5 deg)": dict(orbit={"type": "elements", "altitude": 540, "i": 28.47,
                                            "raan": 120, "nu": 90},
                                     mass=11110, area=40, cd=2.2),
    "Sun-synchronous 700 km": dict(orbit={"type": "elements", "altitude": 700, "i": "sso",
                                          "raan": 90, "nu": 0},
                                   mass=1000, area=6, cd=2.2),
    "Polar LEO 800 km": dict(orbit={"type": "elements", "altitude": 800, "i": 90,
                                    "raan": 0, "nu": 45}, mass=800, area=5),
    "VLEO 250 km (decays)": dict(orbit={"type": "elements", "altitude": 250, "i": 45,
                                        "raan": 200, "nu": 0}, mass=150, area=3),
    "GPS MEO (20200 km, 55 deg)": dict(orbit={"type": "elements", "altitude": 20180, "i": 55,
                                              "raan": 60, "nu": 0}, mass=2000, area=20),
    "Geostationary (GEO)": dict(orbit={"type": "geo", "lon": -75}, mass=3500, area=30),
    "GTO (200 x 35786 km, 27 deg)": dict(orbit={"type": "elements", "perigee_alt": 200,
                                                "apogee_alt": 35786, "i": 27, "raan": 0,
                                                "argp": 178, "nu": 0}, mass=4000, area=15),
    "Molniya (12 h, 63.4 deg)": dict(orbit={"type": "elements", "a": 26554, "e": 0.72,
                                            "i": 63.4, "raan": 250, "argp": 270, "nu": 0},
                                     mass=1600, area=10),
    "Tundra (24 h, 63.4 deg)": dict(orbit={"type": "elements", "a": 42164, "e": 0.25,
                                           "i": 63.4, "raan": 120, "argp": 270, "nu": 0},
                                    mass=2000, area=15),
    "Escape (hyperbolic, e=1.2)": dict(orbit={"type": "elements", "a": -35000, "e": 1.2,
                                              "i": 10, "raan": 0, "argp": 0, "nu": 0},
                                       mass=1000, area=5),
    "Suborbital hop (Cape, 3 km/s)": dict(orbit={"type": "surface", "lat": 28.5, "lon": -80.6,
                                                 "alt": 0, "speed": 3.0, "azimuth": 90,
                                                 "fpa": 45}, mass=500, area=2),
    "Launch to orbit (7.9 km/s)": dict(orbit={"type": "surface", "lat": 5.2, "lon": -52.8,
                                              "alt": 200, "speed": 7.4, "azimuth": 90,
                                              "fpa": 0}, mass=500, area=3),
}


def preset_spec(preset: str, name: str | None = None) -> SatSpec:
    p = copy.deepcopy(PRESETS[preset])
    return SatSpec(name=name or preset.split(" (")[0], orbit=p.pop("orbit"), **p)


DEFAULT_STATIONS = [
    GroundStation("Cape Canaveral", 28.39, -80.61, 0.0, 10.0),
    GroundStation("Svalbard", 78.23, 15.39, 0.5, 5.0),
    GroundStation("Canberra", -35.40, 148.98, 0.7, 10.0),
    GroundStation("Madrid", 40.43, -4.25, 0.8, 10.0),
    GroundStation("Kourou", 5.25, -52.80, 0.0, 10.0),
]
