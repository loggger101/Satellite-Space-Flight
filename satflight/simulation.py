"""The simulation engine: an Earth-centered ensemble of satellites advanced
in time with a selectable propagator, executing scheduled maneuvers and
detecting events.

Propagators
-----------
``cowell``  numerical integration of every enabled force (default).
``kepler``  exact analytic two-body motion (universal variables).
``j2mean``  analytic two-body plus J2 secular drift of RAAN, perigee and
            mean anomaly - fast enough for thousands of satellites.

Events logged: maneuver execution, eclipse entry/exit, ground-station
AOS/LOS, close approaches between satellites, re-entry, surface impact and
escape from Earth's sphere of influence, and every step of a launch.

Launches (``satflight.launch``): a vehicle waits on its pad, turning with the
Earth, until liftoff; during the ascent it is integrated by its own ascent
model and kept out of the ensemble. At payload separation it joins the
ensemble like any other satellite, and spent stages join as objects of their
own.

Time steps belong to the physics, not to the caller: ``advance`` only ever
lands on physical boundaries (maneuvers, burnouts, liftoffs, ascent syncs)
and otherwise takes whole integrator steps, stopping at the last one that
fits. The state it reports at the requested time is integrated forward from
there and thrown away on the next call. Advancing an hour in one call or in
thousands of frame-sized calls follows the same trajectory and logs the same
events, whatever the frame rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis import j2_mean_propagate, j2_secular_rates
from .constants import SOI_EARTH
from .eclipse import shadow_fraction
from .elements import kepler_propagate, rv2coe
from .ephemeris import sun_position
from .forces import ForceModel, approx_altitude, conservative_energy
from .frames import eci_to_ecef
from .integrators import Propagator
from .launch import Ascent, AscentEnv, LaunchSpec, coast, plan_launch, plane_normal
from .maneuvers import G0, Maneuver, finite_direction, impulse_eci, propellant_after, resolve_time
from .scenario import (
    GroundStation,
    SatSpec,
    Scenario,
    expand,
    j2_acts,
    orbit_state,
    palette_color,
    spread_along_orbit,
)
from .timeutil import Clock, format_duration

PROPAGATORS = ("cowell", "kepler", "j2mean")
ACTIVE, DECAYED, IMPACTED, ESCAPED = "active", "re-entered", "impacted", "escaped"
SCRUBBED = "scrubbed"     # a launch that never left the pad

REENTRY_ALT = 80.0        # km: below this the vehicle is considered lost
ASCENT_SYNC = 1.0         # s: longest ensemble segment while a vehicle is in powered flight
EVENT_DETAIL_LIMIT = 40   # log per-satellite eclipse/AOS events only below this N
ALL_PAIRS_LIMIT = 300     # up to this N every pair is measured; above, a spatial grid


_HALF_SHELL = [(dx << 42) + (dy << 21) + dz
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) >= (0, 0, 0)]


def _near_pairs(r: np.ndarray, radius: float):
    """Index pairs (a < b) of the rows of ``r`` closer than ``radius``, and
    their distances, without measuring all N^2 pairs: points are binned in a
    grid of cubes ``radius`` wide, so a close pair sits in the same or a
    neighboring cube. Cost grows with N plus the number of close pairs."""
    cell = np.clip(np.floor(r / radius), -(1 << 20), (1 << 20) - 2).astype(np.int64) + (1 << 20)
    key = (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]
    order = np.argsort(key, kind="stable")
    skey = key[order]
    ps, qs = [], []
    # a cell and the 13 neighbors on one side of it: every neighboring pair of
    # cells is visited once, so each pair of points is found once
    for off in _HALF_SHELL:
        want = key + off
        lo = np.searchsorted(skey, want, "left")
        n = np.searchsorted(skey, want, "right") - lo
        if not n.any():
            continue
        a = np.repeat(np.arange(len(r)), n)
        start = np.repeat(lo - np.cumsum(n) + n, n)
        b = order[start + np.arange(n.sum())]
        if off == 0:
            keep = a < b
            a, b = a[keep], b[keep]
        ps.append(np.minimum(a, b))
        qs.append(np.maximum(a, b))
    if not ps:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    p, q = np.concatenate(ps), np.concatenate(qs)
    d = np.linalg.norm(r[p] - r[q], axis=1)
    near = d < radius
    p, q, d = p[near], q[near], d[near]
    sort = np.lexsort((q, p))                 # row-major order, as the all-pairs path
    return p[sort], q[sort], d[sort]


def _hermite_min(d0, c, m0, m1, samples: int = 17, iterations: int = 8):
    """Minimum distance from the origin of the cubic Hermite curves
    ``p(t) = d0 + c t + t (1 - t)^2 a - t^2 (1 - t) b`` (a = m0 - c, b = m1 - c),
    t in [0, 1], one row per curve: a coarse scan, then Newton's method on
    p . p' = 0 kept between the best sample's neighbors (over one step the
    distance has a single dip). Returns (t, distance)."""
    a, b = m0 - c, m1 - c

    def curve(t):
        t = t[..., None]
        p = d0 + c * t + (t - 2 * t * t + t ** 3) * a - (t * t - t ** 3) * b
        dp = c + (1 - 4 * t + 3 * t * t) * a - (2 * t - 3 * t * t) * b
        ddp = (6 * t - 4) * a - (2 - 6 * t) * b
        return p, dp, ddp

    grid = np.linspace(0.0, 1.0, samples)
    p = curve(np.broadcast_to(grid[:, None], (samples, len(d0))))[0]
    best = np.argmin((p * p).sum(axis=-1), axis=0)
    lo = grid[np.maximum(best - 1, 0)]
    hi = grid[np.minimum(best + 1, samples - 1)]
    t = grid[best]
    for _ in range(iterations):
        p, dp, ddp = curve(t)
        g = (p * dp).sum(axis=-1)                    # half the slope of |p|^2
        gg = (dp * dp).sum(axis=-1) + (p * ddp).sum(axis=-1)
        t = np.clip(t - g / np.where(gg > 0, gg, np.inf), lo, hi)
    p = curve(t)[0]
    return t, np.sqrt((p * p).sum(axis=-1))


@dataclass
class Satellite:
    """Per-satellite bookkeeping; the state vector itself lives in ``Simulation.y``."""

    name: str
    color: tuple
    mass: float = 500.0
    area: float = 5.0
    cd: float = 2.2
    status: str = ACTIVE
    dv_used: float = 0.0          # km/s
    shadow: float = 1.0           # illuminated fraction
    energy_ref: tuple | None = None   # (t, energy) baseline for the drift readout
    family: str = ""              # launch it came from (payload and spent stages share it)
    launch_report: dict | None = None   # ascent summary once a launched payload separates


@dataclass
class Event:
    """A log line; ``kind`` (info, maneuver, eclipse, station, alert, warn, launch)
    picks its color in the UI."""

    t: float
    text: str
    kind: str = "info"


@dataclass
class FiniteBurn:
    """A finite burn in progress: thrust acts from ``t0`` to ``t1``."""

    man: Maneuver
    t0: float
    t1: float
    m0: float
    mdot: float                    # kg/s


class History:
    """Ring buffer of past states (N, L, 6) with sample times (L,)."""

    BUDGET = 3_000_000   # doubles

    def __init__(self, n: int, length: int | None = None):
        self.length = length or self.auto_length(n)
        self.data = np.zeros((n, self.length, 6))
        self.times = np.zeros(self.length)
        self.head = 0
        self.count = 0

    @classmethod
    def auto_length(cls, n: int) -> int:
        """Samples per satellite that keep the buffer within ``BUDGET``."""
        return int(np.clip(cls.BUDGET // (6 * max(n, 1)), 120, 2000))

    def record(self, t: float, y: np.ndarray):
        """Store the ensemble state ``y`` sampled at time ``t``."""
        self.data[:, self.head] = y
        self.times[self.head] = t
        self.head = (self.head + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def _order(self):
        """Buffer indices from oldest to newest sample."""
        if self.count < self.length:
            return np.arange(self.count)
        return (np.arange(self.length) + self.head) % self.length

    def series(self, i: int | None = None):
        """(times, states) oldest -> newest; states (count, 6) or (N, count, 6)."""
        idx = self._order()
        if i is None:
            return self.times[idx], self.data[:, idx]
        return self.times[idx], self.data[i, idx]

    def remove_row(self, i: int):
        """Forget satellite ``i``'s history (its row shifts the later ones up)."""
        self.data = np.delete(self.data, i, axis=0)

    def clear(self):
        """Drop every sample; the buffer keeps its size."""
        self.head = 0
        self.count = 0


class Simulation:
    """A running scenario: the ensemble state ``y`` (N, 6) in ECI km and km/s at time
    ``t`` (s since the scenario epoch), plus the satellites, stations, pending
    maneuvers, burns, ascents and the event log."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.clock = Clock(scenario.epoch)
        self.t = 0.0
        self.forces: ForceModel = scenario.forces
        self.propagator = scenario.propagator
        s = scenario.integrator
        self.integrator = Propagator(s.method, s.rtol, s.atol, s.h_max, s.h_fixed)
        self.record_dt = scenario.record_dt
        self.sats: list[Satellite] = []
        self.y = np.zeros((0, 6))
        self.stations: list[GroundStation] = list(scenario.stations)
        self.maneuvers: list[Maneuver] = []
        self.burns: list[FiniteBurn] = []
        self.events: list[Event] = []
        self.ascents: list[Ascent] = []
        self.conjunction_km = 10.0
        self.watch: set[int] = set()       # indices the UI wants detailed events for
        self._last_record = -math.inf
        self._visible = np.zeros((0, 0), dtype=bool)
        self._close_pairs: set = set()
        self.closest = (math.inf, -1, -1)  # (km, i, j)
        self._last_sample = None           # (t, y, orbiting mask) of the last approach check
        self._grid = None                  # (t, y) the physics reached; t <= self.t
        self._shown = None                 # (t, y) reported from it, to spot outside edits

        for name, color, props, r, v in expand(scenario, self.clock, 0.0):
            self._append(name, color, props, r, v)
        self.history = History(len(self.sats))
        for m in scenario.maneuvers:
            self.schedule(Maneuver.from_dict(m.to_dict()))
        for spec in scenario.launches:
            try:
                self.launch(LaunchSpec.from_dict(spec.to_dict()))
            except ValueError as exc:
                self.log(f"Launch of {spec.name} scrubbed: {exc}", "alert")
        self._post_step(self.t, force_record=True)
        self.log(f"Scenario '{scenario.name}' loaded: {len(self.sats)} satellites, "
                 f"{len(self.stations)} stations, propagator {self.propagator}")
        if scenario.out_of_scope:
            self.log("Ignored (out of scope): " + ", ".join(scenario.out_of_scope), "warn")

    # --- bookkeeping -----------------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of satellites (including re-entered ones)."""
        return len(self.sats)

    @property
    def active(self) -> np.ndarray:
        """Boolean mask of satellites still in flight."""
        return np.array([s.status == ACTIVE for s in self.sats], dtype=bool)

    def index_of(self, name: str) -> int:
        """Index of the satellite called ``name`` (KeyError if none)."""
        for i, s in enumerate(self.sats):
            if s.name == name:
                return i
        raise KeyError(name)

    def unique_name(self, name: str) -> str:
        """``name``, or ``name 2``, ``name 3`` ... if it is taken."""
        names = {s.name for s in self.sats}
        if name not in names:
            return name
        k = 2
        while f"{name} {k}" in names:
            k += 1
        return f"{name} {k}"

    def _append(self, name, color, props, r, v):
        sat = Satellite(self.unique_name(name), tuple(color), **props)
        self.sats.append(sat)
        self.y = np.vstack([self.y, np.concatenate([r, v])[None, :]])
        return sat

    def log(self, text: str, kind: str = "info", t: float | None = None):
        """Append an event (the log keeps the latest ~500)."""
        self.events.append(Event(self.t if t is None else t, text, kind))
        if len(self.events) > 600:
            del self.events[:100]

    def datetime(self):
        """Calendar time (UTC) now."""
        return self.clock.datetime(self.t)

    def jd(self, t: float | None = None) -> float:
        """Julian date at ``t`` (default: now)."""
        return self.clock.jd(self.t if t is None else t)

    def gmst(self, t: float | None = None) -> float:
        """Greenwich mean sidereal time (rad) at ``t`` (default: now)."""
        return self.clock.gmst(self.t if t is None else t)

    # --- adding / removing ---------------------------------------------------------------
    def add_satellites(self, orbit: dict, name: str, color=None, count: int = 1, **props):
        """Add satellites defined by an orbit spec evaluated at the current time."""
        r, v = orbit_state(orbit, self.clock, self.t, j2_acts(self.propagator, self.forces))
        rows = []
        for k, (rr, vv) in enumerate(spread_along_orbit(r, v, count)):
            nm = name if count == 1 else f"{name}-{k + 1}"
            rows.append((nm, color or palette_color(self.n + k), props, rr, vv))
        return self.add_many(rows)

    def add_state(self, name: str, r, v, color=None, **props):
        """Add one satellite from an ECI state; returns it."""
        return self.add_many([(name, color or palette_color(self.n), props,
                               np.asarray(r, float), np.asarray(v, float))])[0]

    def add_many(self, rows):
        """Bulk add ``(name, color, props, r, v)`` rows (one history rebuild)."""
        n_old = self.n
        added = [self._append(*row) for row in rows]
        self._rebuild_history(n_old)
        self._resize_visible()
        names = ", ".join(s.name for s in added[:3])
        self.log(f"Added {len(added)} satellite{'s' if len(added) != 1 else ''}: {names}"
                 f"{' ...' if len(added) > 3 else ''}")
        return added

    def _resize_visible(self):
        """Fit the station-visibility table to the current stations and
        satellites, keeping what is known (a reset would log a fresh AOS
        for everything in view)."""
        old = self._visible
        new = np.zeros((len(self.stations), self.n), dtype=bool)
        s, n = min(old.shape[0], new.shape[0]), min(old.shape[1], new.shape[1])
        new[:s, :n] = old[:s, :n]
        self._visible = new

    def _rebuild_history(self, n_old: int):
        """Resize the history for the new satellite count, keeping the most
        recent samples of existing satellites; new rows start at their
        current state."""
        old = self.history
        times, data = old.series()
        new = History(self.n)
        keep = min(new.length, old.count)
        if keep:
            new.times[:keep] = times[-keep:]
            new.data[:n_old, :keep] = data[:, -keep:]
            new.data[n_old:, :keep] = self.y[n_old:, None, :]
        new.count = keep
        new.head = keep % new.length
        self.history = new

    def remove(self, i: int):
        """Delete satellite ``i`` with its history, maneuvers and burns."""
        sat = self.sats.pop(i)
        self.ascents = [a for a in self.ascents if a.sat is not sat]
        self.y = np.delete(self.y, i, axis=0)
        if self._visible.shape[1] > i:
            self._visible = np.delete(self._visible, i, axis=1)
        self.history.remove_row(i)
        self.maneuvers = [m for m in self.maneuvers if m.sat != sat.name]
        self.burns = [b for b in self.burns if b.man.sat != sat.name]
        self._resize_visible()
        self._close_pairs.clear()
        self.closest = (math.inf, -1, -1)      # its indices may be stale until the next step
        self._last_sample = None
        self.watch = {w - (w > i) for w in self.watch if w != i}
        self.log(f"Removed {sat.name}")

    def add_station(self, st: GroundStation):
        """Add a ground station mid-run."""
        self.stations.append(st)
        self._resize_visible()
        self.log(f"Ground station {st.name} ({st.lat:.2f}, {st.lon:.2f})")

    def set_propagator(self, name: str):
        """Switch between ``PROPAGATORS`` mid-run."""
        if name not in PROPAGATORS:
            raise ValueError(name)
        self.propagator = name
        self.log(f"Propagator -> {name}")

    # --- launches -----------------------------------------------------------------------
    def ascent_env(self) -> AscentEnv:
        """The Earth model ascents fly in, matching this run's forces and clock."""
        return AscentEnv.from_sim(self.forces, self.clock)

    def plane_of(self, name: str):
        """(inclination deg, normal_at(t)) of a satellite's orbit plane, with
        its J2 nodal regression - the target of a launch window."""
        i = self.index_of(name)
        el = rv2coe(self.y[i, :3], self.y[i, 3:])
        if el.e >= 1.0:
            raise ValueError(f"{name} is not on a closed orbit")
        regresses = j2_acts(self.propagator, self.forces)
        rate = float(j2_secular_rates(el.a, el.e, el.i)[0]) if regresses else 0.0
        t_ref, inc, raan = self.t, float(el.i), float(el.raan)
        return math.degrees(inc), lambda t: plane_normal(inc, raan + rate * (t - t_ref))

    def plan(self, spec: LaunchSpec):
        """Fly a launch on paper from now (see ``launch.plan_launch``)."""
        return plan_launch(spec, self.ascent_env(), self.t, self.plane_of)

    def launch(self, spec: LaunchSpec, plan=None) -> Satellite:
        """Put a vehicle on its pad; it lifts off at the planned time."""
        spec.name = self.unique_name(spec.name or "Payload")
        if plan is None:
            plan = self.plan(spec)
        asc = Ascent(spec, plan.res, self.ascent_env(), plan.kick)
        n_old = self.n
        props = dict(mass=asc.mass, area=spec.vehicle.area, cd=spec.vehicle.cd)
        y = asc.pad_state(self.t) if asc.t0 > self.t else asc.state()
        sat = self._append(spec.name, spec.color or palette_color(self.n), props,
                           np.array(y[:3]), np.array(y[3:]))
        sat.family = sat.name
        asc.sat = sat
        self._rebuild_history(n_old)
        self._resize_visible()
        self.ascents.append(asc)
        f = plan.flight
        when = "now" if asc.t0 <= self.t + 1e-6 else f"in {format_duration(asc.t0 - self.t)}"
        if f.outcome == "orbit" and spec.guidance == "orbit":
            outcome = f"orbit {f.insertion['hp']:,.0f} x {f.insertion['ha']:,.0f} km"
        else:
            outcome = f.outcome
        asc.planned = (f.samples, f.stage_marks, outcome)
        self.log(f"{sat.name}: {spec.vehicle.name} on the pad at {spec.site or 'the launch site'} "
                 f"({spec.lat:.2f}, {spec.lon:.2f}), liftoff {when}; planned {outcome}",
                 "launch" if f.outcome == "orbit" or spec.guidance == "open" else "warn")
        self._advance_ascents(self.t)
        return sat

    def ascent_of(self, i: int) -> Ascent | None:
        """The ascent driving satellite i, if it is on a pad or in powered flight."""
        if not 0 <= i < self.n:
            return None
        sat = self.sats[i]
        return next((a for a in self.ascents if a.sat is sat), None)

    def _free_idx(self) -> np.ndarray:
        """Active satellites the ensemble propagates (not on a pad or ascending)."""
        act = self.active
        for a in self.ascents:
            act[self.sats.index(a.sat)] = False
        return np.flatnonzero(act)

    def _advance_ascents(self, t: float):
        """Fly every pad/ascending vehicle to ``t``; spent stages join the ensemble."""
        if not self.ascents:
            return
        spawned = []
        for asc in list(self.ascents):
            i = self.sats.index(asc.sat)
            asc.advance_to(t)
            self.y[i] = asc.state()
            asc.sat.mass = asc.mass
            for te, text, kind in asc.events:
                self.log(text, kind, te)
            asc.events.clear()
            for ts, name, state, mass, cd_a in asc.spawns:
                s = coast(state, ts, t, mass, cd_a, asc.env)
                col = tuple(int(c * 0.6) for c in asc.sat.color)
                props = dict(mass=mass, area=asc.veh.area, cd=asc.veh.cd)
                spawned.append((name, col, props, np.array(s[:3]), np.array(s[3:]), asc.sat.family))
            asc.spawns.clear()
            if asc.released:
                self._finish_ascent(asc)
        if spawned:
            n_old = self.n
            for name, col, props, r, v, fam in spawned:
                self._append(name, col, props, r, v).family = fam
            self._rebuild_history(n_old)
            self._resize_visible()

    def _finish_ascent(self, asc: Ascent):
        """Hand a separated payload over to the ensemble."""
        self.ascents.remove(asc)
        sat, spec = asc.sat, asc.spec
        sat.mass, sat.area, sat.cd = asc.mass, spec.payload_area, spec.payload_cd
        sat.energy_ref = None
        sat.launch_report = dict(asc.summary(), vehicle=spec.vehicle.name, site=spec.site,
                                 lat=spec.lat, lon=spec.lon, t0=asc.t0)
        if asc.outcome == "crash":
            sat.status = IMPACTED
        elif asc.outcome == "no_liftoff":
            sat.status = SCRUBBED
        if sat.status != ACTIVE:
            self.maneuvers = [m for m in self.maneuvers if m.sat != sat.name]
        elif spec.circularize and asc.outcome == "orbit":
            try:
                self.schedule(Maneuver(sat.name, "circularize", timing="apoapsis",
                                       label="circularize at apogee"))
            except ValueError as exc:
                self.log(f"{sat.name}: circularization skipped ({exc})", "warn")

    # --- maneuvers ---------------------------------------------------------------------
    def schedule(self, m: Maneuver) -> Maneuver:
        """Resolve ``m``'s execution time and queue it."""
        i = self.index_of(m.sat)
        if self.ascent_of(i) is not None:
            raise ValueError(f"{m.sat} is still on its launch vehicle; "
                             "maneuver it after separation")
        r, v = self.y[i, :3], self.y[i, 3:]
        if m.timing != "absolute" or m.t is None:
            m.t = resolve_time(m, self.t, r, v)
        m.timing = "absolute"
        self.maneuvers.append(m)
        self.maneuvers.sort(key=lambda x: x.t)
        when = "now" if m.t <= self.t + 1e-6 else f"in {format_duration(m.t - self.t)}"
        self.log(f"Scheduled {m.describe()} for {m.sat} {when}", "maneuver")
        return m

    def _execute_due(self):
        """Execute every queued maneuver whose time has come."""
        due = [m for m in self.maneuvers if not m.done and m.t <= self.t + 1e-6]
        for m in due:
            m.done = True
            self.maneuvers.remove(m)
            try:
                i = self.index_of(m.sat)
            except KeyError:
                continue
            sat = self.sats[i]
            if sat.status != ACTIVE:
                continue
            r, v = self.y[i, :3].copy(), self.y[i, 3:].copy()
            if m.kind == "finite":
                self._start_finite(m, i)
                continue
            try:
                tv = None
                if m.kind == "match_velocity":
                    tv = self.y[self.index_of(m.target), 3:]
                dv = impulse_eci(m, r, v, tv)
            except (ValueError, KeyError) as exc:
                self.log(f"{m.sat}: maneuver skipped ({exc})", "warn")
                continue
            dvm = float(np.linalg.norm(dv))
            self.y[i, 3:] = v + dv
            sat.dv_used += dvm
            sat.mass = propellant_after(sat.mass, dvm, m.isp)
            sat.energy_ref = None
            m.result = f"{dvm * 1000:.2f} m/s"
            self.log(f"{m.sat}: executed {m.describe()} - dV {dvm * 1000:.2f} m/s", "maneuver")
        if due:
            self.history.record(self.t, self.y)
            if self._last_sample is not None:  # the next approach check starts after the burns
                self._last_sample = (self.t, self.y.copy(), self._last_sample[2])

    def _start_finite(self, m: Maneuver, i: int):
        """Ignite a finite burn (applied impulsively by the analytic propagators)."""
        sat = self.sats[i]
        mdot = m.thrust / (m.isp * G0 * 1000.0) if m.isp > 0 else 0.0   # kg/s
        duration = m.duration
        if mdot > 0:
            duration = min(duration, 0.95 * sat.mass / mdot)
        if self.propagator != "cowell":
            # analytic propagators cannot integrate thrust: apply the same
            # rocket-equation delta-v impulsively along the burn direction
            mf = sat.mass - mdot * duration
            dvm = m.isp * G0 * math.log(sat.mass / mf) if mf > 0 else 0.0
            d = finite_direction(m, self.y[i:i + 1, :3], self.y[i:i + 1, 3:])[0]
            self.y[i, 3:] += dvm * d
            sat.dv_used += dvm
            sat.mass = mf
            sat.energy_ref = None
            self.log(f"{m.sat}: finite burn applied impulsively ({dvm * 1000:.1f} m/s) - "
                     "switch to Cowell to integrate thrust", "maneuver")
            return
        self.burns.append(FiniteBurn(m, self.t, self.t + duration, sat.mass, mdot))
        self.log(f"{m.sat}: ignition - {m.thrust:.0f} N for {duration:.0f} s", "maneuver")

    def _finish_burns(self):
        """Book the delta-v and mass of every burn that has ended."""
        for b in [b for b in self.burns if b.t1 <= self.t + 1e-6]:
            self.burns.remove(b)
            try:
                sat = self.sats[self.index_of(b.man.sat)]
            except KeyError:
                continue
            mf = b.m0 - b.mdot * (b.t1 - b.t0)
            dvm = b.man.isp * G0 * math.log(b.m0 / mf) if mf > 0 else 0.0
            sat.mass = mf
            sat.energy_ref = None
            sat.dv_used += dvm
            self.log(f"{b.man.sat}: burnout - dV {dvm * 1000:.1f} m/s, "
                     f"mass {mf:.1f} kg", "maneuver")

    # --- dynamics ------------------------------------------------------------------------
    def _props(self, idx):
        """Cd*A/m (m^2/kg) of satellites ``idx``."""
        return np.array([self.sats[i].cd * self.sats[i].area / self.sats[i].mass for i in idx])

    def _derivative(self, idx: np.ndarray, burns: list):
        """State derivative f(t, y) for satellites ``idx`` including active finite burns."""
        cd_am = self._props(idx)
        rows = {int(i): k for k, i in enumerate(idx)}
        burn_rows = [(rows[j], b) for b in burns if (j := self.index_of(b.man.sat)) in rows]
        forces, clock = self.forces, self.clock

        def f(t, y):
            r, v = y[:, :3], y[:, 3:]
            a = forces.acceleration(r, v, cd_am, float(clock.gmst(t)) if forces.c22 else 0.0)
            for k, b in burn_rows:
                m = b.m0 - b.mdot * (t - b.t0)
                d = finite_direction(b.man, r[k:k + 1], v[k:k + 1])[0]
                a[k] += d * (b.man.thrust / m) * 1e-3
            return np.concatenate([v, a], axis=1)
        return f

    def _next_boundary(self) -> float:
        """Time of the next liftoff, maneuver, burnout or ascent sync (inf if none).
        Ascent syncs count from the ascent's own clock, so they fall at the same
        times however the simulation is advanced."""
        tb = math.inf
        for a in self.ascents:
            if a.phase == "pad":
                if self.t < a.t0 < tb:
                    tb = a.t0
            else:
                tb = min(tb, max(a.t + ASCENT_SYNC, self.t))
        for m in self.maneuvers:
            if self.t < m.t < tb:
                tb = m.t
        for b in self.burns:
            if self.t < b.t1 < tb:
                tb = b.t1
        return tb

    def advance(self, dt: float):
        """Advance the simulation by ``dt`` seconds of simulated time (see the
        module notes: the physics keeps its own steps, whatever ``dt`` is)."""
        t_end = self.t + dt
        self._resume()
        guard = 0
        self._execute_due()
        while guard < 10000:
            guard += 1
            tb = self._next_boundary()
            land = tb <= t_end
            idx = self._free_idx()
            if idx.size:
                if self._propagate(idx, tb, None if land else t_end):
                    continue            # a satellite left the ensemble: go on without it
            elif land:
                self.t = tb
            elif not any(a.phase != "pad" for a in self.ascents):
                self.t = t_end          # nothing is integrated, so there are no steps to keep
                self._post_step(self.t, idx)
            if not land:
                break
            self._advance_ascents(self.t)
            if not idx.size:
                self._post_step(self.t, idx)
            self._finish_burns()
            self._execute_due()
        self._present(t_end)

    def _resume(self):
        """Go back from the reported state to the physics state it was integrated
        from, unless someone changed it in between (a satellite added, removed or
        edited): then the physics starts again from the state as it now is."""
        grid, shown, self._grid, self._shown = self._grid, self._shown, None, None
        if (grid is not None and self.t == shown[0] and self.y.shape == shown[1].shape
                and np.array_equal(self.y, shown[1])):
            self.t, self.y = grid[0], grid[1]
        else:
            self._last_sample = None       # nothing joins the last step's states to these

    def _present(self, t: float):
        """Report the state at ``t``, which may lie up to a step past the physics:
        integrate there on the side, keeping the physics state for the next call."""
        self._grid = (self.t, self.y.copy())
        if t > self.t:
            idx = self._free_idx()
            if idx.size:
                self.y[idx] = self._peek(idx, t)
            for a in self.ascents:
                self.y[self.sats.index(a.sat)] = a.peek(t)
            self.t = t
        self._shown = (self.t, self.y.copy())

    def _peek(self, idx: np.ndarray, t1: float) -> np.ndarray:
        """States of satellites ``idx`` at ``t1``, less than a step ahead, integrated
        with a scratch integrator so the physics state and step size are untouched."""
        y0 = self.y[idx]
        if self.propagator == "kepler":
            r, v = kepler_propagate(y0[:, :3], y0[:, 3:], t1 - self.t)
            return np.concatenate([r, v], axis=1)
        if self.propagator == "j2mean":
            r, v = j2_mean_propagate(y0[:, :3], y0[:, 3:], t1 - self.t)
            return np.concatenate([r, v], axis=1)
        s = self.integrator
        scratch = Propagator(s.method, s.rtol, s.atol, s.h_max, s.h_fixed)
        scratch.h = s.h
        burns = [b for b in self.burns if b.t0 <= self.t + 1e-9 and t1 <= b.t1 + 1e-9]
        return scratch.integrate(self._derivative(idx, burns), self.t, y0.copy(), t1)[1]

    def _propagate(self, idx: np.ndarray, t_seg: float, t_stop: float | None = None) -> bool:
        """Advance satellites ``idx`` toward ``t_seg``, landing on it, or with
        ``t_stop`` ending at the last whole step before that. Returns True if it
        stopped early because a satellite left the active set."""
        stop = t_seg if t_stop is None else min(t_seg, t_stop)
        left = False

        def callback(t, ya):
            nonlocal left
            self.t = t
            self.y[idx] = ya
            left = self._post_step(t, idx=idx)
            return left

        if self.propagator == "cowell":
            burns = [b for b in self.burns if b.t0 <= self.t + 1e-9 and t_seg <= b.t1 + 1e-9]
            f = self._derivative(idx, burns)
            t, ya = self.integrator.integrate(f, self.t, self.y[idx].copy(), t_seg, callback,
                                              t_stop)
            self.t = t
            self.y[idx] = ya
        else:
            h = self.integrator.h_max
            while self.t < t_seg - 1e-9:
                step = min(h, t_seg - self.t)
                if stop < t_seg and self.t + step > stop:
                    break
                y0 = self.y[idx]
                if self.propagator == "kepler":
                    r, v = kepler_propagate(y0[:, :3], y0[:, 3:], step)
                else:
                    r, v = j2_mean_propagate(y0[:, :3], y0[:, 3:], step)
                self.integrator.stats.accepted += 1
                self.integrator.stats.last_h = step
                if callback(self.t + step, np.concatenate([r, v], axis=1)):
                    break
        return left

    # --- per-step checks ------------------------------------------------------------------
    def _post_step(self, t: float, idx=None, force_record: bool = False) -> bool:
        """Record history and detect events. Returns True if a satellite left
        the active set (the integrator must restart with the new set)."""
        if idx is None:
            idx = self._free_idx()
        for a in self.ascents:
            if a.phase == "pad":
                self.y[self.sats.index(a.sat)] = a.pad_state(t)
        changed = False
        if idx.size:
            r = self.y[idx, :3]
            v = self.y[idx, 3:]
            rm = np.linalg.norm(r, axis=1)
            alt = approx_altitude(r)
            # only a *descending* vehicle can re-enter: launches start low and climb
            descending = np.sum(r * v, axis=1) < 0
            for k in np.flatnonzero(((alt < REENTRY_ALT) & descending) | (rm > SOI_EARTH)):
                i = idx[k]
                sat = self.sats[i]
                speed = float(np.linalg.norm(v[k]))
                if rm[k] > SOI_EARTH:
                    sat.status = ESCAPED
                    self.log(f"{sat.name} left Earth's sphere of influence", "alert")
                elif alt[k] <= 0:
                    sat.status = IMPACTED
                    self.log(f"{sat.name} impacted the surface at {speed:.2f} km/s", "alert")
                else:
                    sat.status = DECAYED
                    self.log(f"{sat.name} re-entered (below {REENTRY_ALT:.0f} km, "
                             f"{speed:.2f} km/s)", "alert")
                self.maneuvers = [m for m in self.maneuvers if m.sat != sat.name]
                self.burns = [b for b in self.burns if b.man.sat != sat.name]
                changed = True
            self._events(t, idx)
        if force_record or t - self._last_record >= self.record_dt - 1e-9 or changed:
            self.history.record(t, self.y)
            self._last_record = t
        return changed

    def _events(self, t: float, idx: np.ndarray):
        """Log eclipse, station AOS/LOS and close-approach events."""
        detail = self.n <= EVENT_DETAIL_LIMIT
        free = {int(i) for i in idx}
        watch = free if detail else self.watch & free
        jd = self.clock.jd(t)
        r_all = self.y[:, :3]

        # eclipses
        if watch:
            w = np.array(sorted(watch))
            frac = shadow_fraction(r_all[w], sun_position(jd))
            for i, fr in zip(w, frac, strict=True):
                sat = self.sats[i]
                if sat.shadow >= 0.5 > fr:
                    self.log(f"{sat.name} entered Earth's shadow", "eclipse")
                elif sat.shadow < 0.5 <= fr:
                    self.log(f"{sat.name} back in sunlight", "eclipse")
                sat.shadow = float(fr)

        # ground stations
        if self.stations and watch:
            if self._visible.shape != (len(self.stations), self.n):
                self._resize_visible()
            w = np.array(sorted(watch))
            r_ecef = eci_to_ecef(r_all[w], self.clock.gmst(t))
            for s, st in enumerate(self.stations):
                for i, now in zip(w, st.sees(r_ecef), strict=True):
                    if now and not self._visible[s, i]:
                        self.log(f"AOS {self.sats[i].name} @ {st.name}", "station")
                    elif not now and self._visible[s, i]:
                        self.log(f"LOS {self.sats[i].name} @ {st.name}", "station")
                    self._visible[s, i] = now

        # close approaches: every pair in small ensembles, pairs found through a
        # spatial grid in large ones (vehicles still inside the atmosphere, e.g.
        # on a launch pad, are ignored)
        orbiting = idx[approx_altitude(r_all[idx]) > 100.0]
        if orbiting.size > 1:
            r = r_all[orbiting]
            sweep = self._sweep(t, orbiting)
            if orbiting.size <= ALL_PAIRS_LIMIT:
                p, q = np.triu_indices(orbiting.size, 1)
                d = np.linalg.norm(r[p] - r[q], axis=1)
            else:
                # during the step each satellite stayed within `spread` of the
                # middle of its chord, so a pair that met had middles within the
                # alert distance plus both spreads
                if sweep is None:
                    mid, spread = r, np.zeros(len(r))
                else:
                    mid, spread = sweep[4], sweep[5]
                radius = self.conjunction_km + 2 * float(np.max(spread, initial=0.0))
                p, q, _ = _near_pairs(mid, max(radius, 1.0))
                d = np.linalg.norm(r[p] - r[q], axis=1)
            if any(self.sats[i].family for i in orbiting):
                # a payload and the stages that carried it drift apart slowly
                fam = np.array([self.sats[i].family or self.sats[i].name for i in orbiting], object)
                d = np.where(fam[p] == fam[q], np.inf, d)
            k = int(np.argmin(d)) if d.size else -1
            if k >= 0 and math.isfinite(d[k]):
                self.closest = (float(d[k]), int(orbiting[p[k]]), int(orbiting[q[k]]))
            else:
                self.closest = (math.inf, -1, -1)
            now = {}
            for a, b, dist, tca in self._approaches(t, orbiting, p, q, d, sweep):
                now[(a, b)] = (dist, tca)
            for p, q in sorted(set(now) - self._close_pairs, key=lambda pq: now[pq][1]):
                dist, tca = now[(p, q)]
                self.log(f"Close approach: {self.sats[p].name} - {self.sats[q].name} "
                         f"{dist:.2f} km", "alert", tca)
            self._close_pairs = set(now)
            mask = np.zeros(self.n, dtype=bool)
            mask[orbiting] = True
            self._last_sample = (t, self.y.copy(), mask)
        else:
            self.closest = (math.inf, -1, -1)
            self._last_sample = None

    def _sweep(self, t: float, orbiting: np.ndarray):
        """``(t0, h, (y0, y1), reach, mid, spread)`` for the step that just
        ended, or None without a usable previous sample. ``reach`` bounds how
        far each satellite can have been during the step from where it is now
        (-inf for satellites that were not orbiting then: no curve to follow);
        ``spread`` bounds how far it was from ``mid``, the middle of its chord
        (its current position and 0 for those not orbiting then)."""
        prev = self._last_sample
        if prev is None or not prev[0] < t or prev[1].shape != self.y.shape:
            return None
        h = t - prev[0]
        y0, y1 = prev[1][orbiting], self.y[orbiting]
        moved = y1[:, :3] - y0[:, :3]
        step = np.linalg.norm(moved, axis=1)
        bow = 4 / 27 * (np.linalg.norm(h * y0[:, 3:] - moved, axis=1)
                        + np.linalg.norm(h * y1[:, 3:] - moved, axis=1))
        reach = step + bow
        mid = y1[:, :3] - 0.5 * moved
        spread = 0.5 * step + bow
        new = ~prev[2][orbiting]
        reach[new] = -math.inf
        mid[new] = y1[new, :3]
        spread[new] = 0.0
        return prev[0], h, (y0, y1), reach, mid, spread

    def _approaches(self, t: float, orbiting: np.ndarray, p: np.ndarray, q: np.ndarray,
                    d: np.ndarray, sweep):
        """Pairs ``(p, q)`` of ``orbiting`` (``d`` their distances now, inf for
        pairs to ignore) that came within ``conjunction_km`` since the last
        step (``sweep`` from :meth:`_sweep`), as (i, j, closest km, time of
        closest approach).

        A fast pass can fall between two steps, so each pair's relative motion
        over the step is modeled as the cubic Hermite curve through both ends'
        positions and velocities; it stays within 4/27 (|m0 - c| + |m1 - c|) of
        its chord c (m = velocity x step). The same bound for each satellite's
        own curve gives how far it can have been from where it is now, which
        rules out nearly every pair from the distances alone; the chord rules
        out most of the rest, and what is left is searched for its minimum."""
        lim = self.conjunction_km
        dist, tca = d.copy(), {}
        if sweep is not None:
            t0, h, (y0, y1), reach = sweep[:4]
            sel = np.flatnonzero(d - reach[p] - reach[q] < lim)
            pp, qq = p[sel], q[sel]
            rel0, rel1 = y0[pp] - y0[qq], y1[pp] - y1[qq]
            d0, m0 = rel0[:, :3], h * rel0[:, 3:]
            c, m1 = rel1[:, :3] - d0, h * rel1[:, 3:]
            cc = np.einsum("ij,ij->i", c, c)
            s = np.clip(-np.einsum("ij,ij->i", d0, c) / np.where(cc > 0, cc, 1.0), 0.0, 1.0)
            chord = np.linalg.norm(d0 + s[:, None] * c, axis=1)
            bow = 4 / 27 * (np.linalg.norm(m0 - c, axis=1) + np.linalg.norm(m1 - c, axis=1))
            near = chord - bow < lim
            if near.any():
                tau, dm = _hermite_min(d0[near], c[near], m0[near], m1[near])
                for k, x, u in zip(sel[near], dm, tau, strict=True):
                    if x < dist[k]:
                        dist[k] = x
                        tca[k] = t0 + u * h
        for k in np.flatnonzero(dist < lim):
            yield (int(orbiting[p[k]]), int(orbiting[q[k]]), float(dist[k]),
                   tca.get(int(k), t))

    # --- diagnostics ------------------------------------------------------------------------
    def energy(self, i: int) -> float:
        """Specific energy of satellite ``i`` including the enabled zonal terms."""
        r, v = self.y[i:i + 1, :3], self.y[i:i + 1, 3:]
        return float(conservative_energy(r, v, self.forces, self.gmst())[0])

    def station_visibility(self, i: int):
        """[(station, az deg, el deg, range km)] for stations that see satellite i."""
        out = []
        r_ecef = eci_to_ecef(self.y[i, :3], self.gmst())
        for st in self.stations:
            az, el, rng = st.look_angles(r_ecef)
            if el >= math.radians(st.min_el):
                out.append((st, math.degrees(float(az)), math.degrees(float(el)), float(rng)))
        return out

    def snapshot_scenario(self, name: str | None = None) -> Scenario:
        """The current state as a new scenario whose epoch is 'now'."""
        sc = self.scenario.copy()
        sc.name = name or f"{self.scenario.name} @ {format_duration(self.t)}"
        sc.epoch = self.datetime()
        sc.satellites = []
        sc.constellations = []
        pads = {id(a.sat): a for a in self.ascents if a.phase == "pad"}
        sc.launches = []
        for a in pads.values():
            spec = a.spec.copy()
            spec.timing, spec.t0, spec.kick = "absolute", a.t0 - self.t, a.kick_deg
            sc.launches.append(spec)
        for i, s in enumerate(self.sats):
            if s.status != ACTIVE or id(s) in pads:
                continue
            if self.ascent_of(i) is not None:
                self.log(f"Snapshot: {s.name} is in powered flight; "
                         "saved as a free-flying state", "warn")
            sc.satellites.append(SatSpec(
                s.name, {"type": "state", "r": self.y[i, :3].tolist(), "v": self.y[i, 3:].tolist()},
                list(s.color), s.mass, s.area, s.cd))
        sc.stations = list(self.stations)
        sc.forces = ForceModel.from_dict(self.forces.to_dict())
        sc.propagator = self.propagator
        sc.maneuvers = []
        for m in self.maneuvers:
            mm = Maneuver.from_dict(m.to_dict())
            mm.t = m.t - self.t
            sc.maneuvers.append(mm)
        return sc


__all__ = ["Simulation", "Satellite", "History", "Event", "PROPAGATORS", "ASCENT_SYNC"]
