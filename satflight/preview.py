"""Fly a list of planned burns ahead of time, for showing them before they are
scheduled: the transfer arcs between burns, each burn's place and delta-v,
and the orbit the satellite ends in.

Two ways: ``aim=False`` coasts two-body (exact Kepler, instant; what the 3-D
preview shows while a form is being typed in) and ``aim=True`` coasts with
the run's own propagator and forces (what will really be flown). Nothing in
the simulation is changed; :func:`snapshot` makes a copy that a background
thread may fly while the run goes on.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .elements import kepler_propagate, rv2coe
from .maneuvers import G0, Maneuver, finite_direction, impulse_eci, resolve_time

ARC_SAMPLES = 160          # points along each drawn arc
FINAL_SPAN = 6 * 3600.0    # s drawn of a final orbit that never closes
NODE_DEG = 0.5             # deg of latitude: a burn this near the equator is "at a node"
APSIS_SIN = 2e-3           # sine of the flight-path angle: flatter is "at an apsis"


@dataclass
class BurnPoint:
    """One burn of a flown plan (ECI km, km/s; ``t`` s since epoch)."""

    t: float
    r: np.ndarray
    v_before: np.ndarray
    dv: np.ndarray
    label: str
    kind: str
    note: str = ""            # e.g. how a finite burn is drawn

    @property
    def size(self) -> float:
        """|dv| in km/s."""
        return float(np.linalg.norm(self.dv))

    @property
    def tilt(self) -> float:
        """Degrees the burn turns the orbit plane."""
        h0 = np.cross(self.r, self.v_before)
        h1 = np.cross(self.r, self.v_before + self.dv)
        c = float(np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1)))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    @property
    def speed_change(self) -> float:
        """Change of speed, km/s (the rest of |dv| turns the velocity)."""
        return float(np.linalg.norm(self.v_before + self.dv) - np.linalg.norm(self.v_before))

    @property
    def place(self) -> str:
        """Where on its orbit the burn fires, in words."""
        r, v = self.r, self.v_before
        rm = float(np.linalg.norm(r))
        bits = []
        sin_fpa = float(np.dot(r, v)) / (rm * float(np.linalg.norm(v)))
        if abs(sin_fpa) < APSIS_SIN:
            el = rv2coe(r, v)
            bits.append("apogee" if el.e > 1e-3 and rm > el.a else
                        "perigee" if el.e > 1e-3 else "circular orbit")
        if abs(math.degrees(math.asin(max(-1.0, min(1.0, r[2] / rm))))) < NODE_DEG:
            bits.append("node")
        return ", ".join([f"{rm - R_EARTH:,.0f} km up", *bits])


@dataclass
class FlownPlan:
    """What :func:`fly` returns."""

    sat: str
    aimed: bool
    t0: float                               # when the plan was flown from
    start: np.ndarray                       # (6,) state then
    burns: list = field(default_factory=list)       # BurnPoint, in time order
    arcs: list = field(default_factory=list)        # (M, 3) ECI: burn k to burn k + 1
    final: np.ndarray | None = None                 # (6,) just after the last burn
    final_track: np.ndarray | None = None           # (M, 3) one revolution of it
    mass0: float = 0.0                              # kg before the first burn
    mass1: float = 0.0                              # kg after the last
    maneuvers: list = field(default_factory=list)   # the Maneuver objects flown
    note: str = ""                                  # the planner's summary, if any

    @property
    def total_dv(self) -> float:
        """Sum of |dv|, km/s."""
        return sum(b.size for b in self.burns)


def snapshot(sim):
    """A copy of ``sim`` that can be flown ahead (planners, :func:`fly`) on
    another thread while the run goes on: it shares the force model and the
    satellites' drag properties, but has its own time and states."""
    snap = copy.copy(sim)
    snap.y = sim.y.copy()
    snap.maneuvers = [Maneuver.from_dict(m.to_dict()) for m in sim.maneuvers]
    return snap


def _coast(sim, i: int, y: np.ndarray, t0: float, times, aim: bool) -> np.ndarray:
    """States (len(times), 6) of satellite ``i`` flown from (t0, y)."""
    times = np.asarray(times, dtype=float)
    if aim:
        return sim._coast_samples([i], y[None], t0, times)[0]
    r, v = kepler_propagate(np.repeat(y[None, :3], len(times), 0),
                            np.repeat(y[None, 3:], len(times), 0), times - t0)
    return np.concatenate([np.atleast_2d(r), np.atleast_2d(v)], axis=1)


def _when(sim, m: Maneuver, i: int, y: np.ndarray, t: float, aim: bool) -> float:
    """Absolute time ``m`` fires, judged from state ``y`` at ``t``."""
    if m.timing == "absolute" and m.t is not None:
        return float(m.t)
    if aim and t == sim.t:
        return float(sim.resolve_time(m))
    return float(resolve_time(m, t, y[:3], y[3:]))


def _kick(sim, m: Maneuver, i: int, y: np.ndarray, t: float, mass: float, aim: bool):
    """(ECI dv km/s, note) of burn ``m`` fired at state ``y`` by a satellite of
    ``mass`` kg (a finite burn stops at the dry mass, as in the run)."""
    r, v = y[:3], y[3:]
    if m.kind == "finite":
        mdot = m.thrust / (m.isp * G0 * 1000.0)            # kg/s (G0 is km/s^2)
        burnt = min(mdot * m.duration, max(mass - sim.sats[i].dry_mass, 0.0))
        size = m.isp * G0 * math.log(mass / (mass - burnt)) if mass > burnt else 0.0
        d = finite_direction(m, r[None], v[None])[0]
        return size * d, f"a {m.duration:,.0f} s firing, drawn as one kick"
    tv = None
    if m.kind == "match_velocity":
        j = sim.require(m.target)
        tv = _coast(sim, j, sim.y[j].copy(), sim.t, [t], aim)[0, 3:]
    return impulse_eci(m, r, v, tv), ""


def fly(sim, sat: str, burns, aim: bool = False) -> FlownPlan:
    """Fly ``sat`` from now through ``burns`` (Maneuver objects, not scheduled)
    and one revolution of the orbit it ends in. ValueError if a burn cannot
    be timed or flown from where the satellite will be."""
    i = sim.require(sat)
    t, y = float(sim.t), sim.y[i].copy()
    plan = FlownPlan(sat, aim, t, y.copy())
    mass = plan.mass0 = float(sim.sats[i].mass)
    plan.maneuvers = [Maneuver.from_dict(m.to_dict()) for m in burns]
    order = sorted(burns, key=lambda m: _when(sim, m, i, y, t, aim))
    for m in order:
        t_b = _when(sim, m, i, y, t, aim)
        if t_b < t - 1e-6:
            raise ValueError(f"{m.describe()} would fire before the burn ahead of it")
        if plan.burns:                                   # the transfer arc up to this burn
            track = _coast(sim, i, y, t, np.linspace(t, t_b, ARC_SAMPLES), aim)
            plan.arcs.append(track[:, :3])
            y = track[-1].copy()
        elif t_b > t:
            y = _coast(sim, i, y, t, [t_b], aim)[0].copy()
        t = t_b
        dv, note = _kick(sim, m, i, y, t, mass, aim)
        plan.burns.append(BurnPoint(t, y[:3].copy(), y[3:].copy(), dv, m.describe(), m.kind,
                                    note))
        y[3:] += dv
        mass *= math.exp(-float(np.linalg.norm(dv)) / (m.isp * G0)) if m.isp > 0 else 1.0
    plan.final = y.copy()
    plan.mass1 = mass
    if not plan.burns:
        return plan
    el = rv2coe(y[:3], y[3:])
    span = 2 * math.pi * math.sqrt(el.a ** 3 / MU_EARTH) if el.e < 1 else FINAL_SPAN
    track = _coast(sim, i, y, t, np.linspace(t, t + span, 2 * ARC_SAMPLES), aim)
    rad = np.linalg.norm(track[:, :3], axis=1)
    down = np.flatnonzero(rad < R_EARTH)                 # it re-enters: stop at the ground
    plan.final_track = track[: down[0] + 1, :3] if down.size else track[:, :3]
    return plan


def orbit_summary(y) -> tuple[float, float, float, float]:
    """(perigee alt km, apogee alt km or inf, inclination deg, period s or inf)
    of the two-body orbit through state ``y``."""
    el = rv2coe(np.asarray(y[:3]), np.asarray(y[3:]))
    rp = el.a * (1 - el.e) if el.e < 1 else el.p / (1 + el.e)
    if el.e < 1:
        return (rp - R_EARTH, el.a * (1 + el.e) - R_EARTH, math.degrees(el.i),
                2 * math.pi * math.sqrt(el.a ** 3 / MU_EARTH))
    return rp - R_EARTH, math.inf, math.degrees(el.i), math.inf
