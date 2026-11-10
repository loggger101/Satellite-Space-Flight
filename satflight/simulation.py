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

from .analysis import (
    j2_mean_propagate,
    j2_secular_rates,
    osculating_to_mean_a,
    osculating_to_mean_i,
)
from .constants import MU_EARTH, SOI_EARTH
from .eclipse import shadow_fraction
from .elements import kepler_propagate, rv2coe
from .ephemeris import sun_position
from .forces import ForceModel, conservative_energy
from .frames import eci_to_ecef, geodetic_height, unit
from .integrators import Propagator
from .launch import Ascent, AscentEnv, LaunchSpec, coast, fly, plan_launch, plane_normal, validate
from .maneuvers import G0, Maneuver, finite_direction, impulse_eci, propellant_after, resolve_time
from .scenario import (
    PROPAGATORS,
    GroundStation,
    SatSpec,
    Scenario,
    expand,
    j2_acts,
    orbit_state,
    palette_color,
    spread_along_orbit,
    zonal_terms,
)
from .strict import check_dry_mass, drag_props, one_of
from .timeutil import Clock, format_duration

ACTIVE, DECAYED, IMPACTED, ESCAPED = "active", "re-entered", "impacted", "escaped"
SCRUBBED = "scrubbed"     # a launch that never left the pad

REENTRY_ALT = 80.0        # km: below this the vehicle is considered lost
ASCENT_SYNC = 1.0         # s: longest ensemble segment while a vehicle is in powered flight
EVENT_DETAIL_LIMIT = 40   # log per-satellite eclipse/AOS events only below this N
ALL_PAIRS_LIMIT = 300     # up to this N every pair is measured; above, a spatial grid
DIP_MARGIN = 10.0         # km: satellites this close above re-entry height get a mid-step check
DIP_CHECK_LIMIT = 2000    # above this N the mid-step dip check is skipped (cost)
DEFAULT_ISP = 300.0       # s: the Isp a maneuver gets unless it names one
APSIS_TIME_TOL = 1e-6      # s: how closely an apsis burn time is pinned down
CIRC_SAMPLES = 72         # radius samples over the revolution a circularization looks at
CIRC_TOL = 1e-6           # km: once-per-rev radius term a circularized orbit may keep
CIRC_ITER = 8             # Newton passes before circularizing is given up
CIRC_STEP = 1e-5          # km/s: finite-difference step of those passes


def dv_left(sat, isp: float = DEFAULT_ISP) -> float:
    """Delta-v (km/s) the propellant still on board gives at ``isp`` seconds
    (the rocket equation down to the dry mass); inf with no dry mass set."""
    if sat.dry_mass <= 0:
        return math.inf
    return isp * G0 * math.log(sat.mass / sat.dry_mass) if sat.mass > sat.dry_mass else 0.0


def _once_per_rev(track: np.ndarray, rh: np.ndarray, th: np.ndarray, t: np.ndarray):
    """Cos u and sin u amplitudes (km) of the radius along ``track``, u being the
    angle from ``rh`` toward ``th`` in the starting plane. The fit also takes a
    constant, a drift (drag) and the J2 ripple at 2u, so only free eccentricity
    is left in the two returned numbers."""
    pos = track[:, :3]
    u = np.arctan2(pos @ th, pos @ rh)
    a = np.stack([np.ones_like(u), t / t[-1], np.cos(u), np.sin(u),
                  np.cos(2 * u), np.sin(2 * u)], axis=1)
    coef = np.linalg.lstsq(a, np.linalg.norm(pos, axis=1), rcond=None)[0]
    return coef[2:4]


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


def _hermite_at(y0: np.ndarray, y1: np.ndarray, h: float, s) -> np.ndarray:
    """Positions at fractions ``s`` of a step of ``h`` s on the cubic Hermite curve
    through states ``y0`` and ``y1`` (6,): sub-metre over a 60 s LEO step."""
    s = np.asarray(s, dtype=float)[..., None]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * y0[:3] + (s3 - 2 * s2 + s) * h * y0[3:]
            + (3 * s2 - 2 * s3) * y1[:3] + (s3 - s2) * h * y1[3:])


def _hermite_speed(y0: np.ndarray, y1: np.ndarray, h: float, s: float) -> float:
    """Speed (km/s) at fraction ``s`` of the step along the same Hermite curve."""
    s2 = s * s
    vel = ((6 * s2 - 6 * s) * y0[:3] / h + (3 * s2 - 4 * s + 1) * y0[3:]
           + (6 * s - 6 * s2) * y1[:3] / h + (3 * s2 - 2 * s) * y1[3:])
    return float(np.linalg.norm(vel))


def _crossing(f, y0: np.ndarray, y1: np.ndarray, h: float, lo: float = 0.0,
              hi: float = 1.0) -> float:
    """Fraction of the step where ``f(position, fraction)`` changes sign between
    ``lo`` and ``hi`` (it must), by bisection to ~1e-7 of the step."""
    f_lo = f(_hermite_at(y0, y1, h, lo), lo)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        f_mid = f(_hermite_at(y0, y1, h, mid), mid)
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class Satellite:
    """Per-satellite bookkeeping; the state vector itself lives in ``Simulation.y``."""

    name: str
    color: tuple
    mass: float = 500.0
    area: float = 5.0
    cd: float = 2.2
    dry_mass: float = 0.0         # kg left with the propellant gone; 0 = no limit
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
        self.propagator = one_of(scenario.propagator, PROPAGATORS, "propagator")
        s = scenario.integrator
        self.integrator = Propagator(s.method, s.rtol, s.atol, s.h_max, s.h_fixed)
        self.record_dt = scenario.record_dt
        self.sats: list[Satellite] = []
        self.y = np.zeros((0, 6))
        self.stations: list[GroundStation] = list(scenario.stations)
        self.maneuvers: list[Maneuver] = []
        self.burns: list[FiniteBurn] = []
        self.events: list[Event] = []
        self.event_count = 0               # events ever logged (the log keeps the latest)
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
        self.warn_ignored()

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

    def require(self, name: str) -> int:
        """Index of the satellite called ``name``; ValueError (a message for the
        user) if there is none."""
        for i, s in enumerate(self.sats):
            if s.name == name:
                return i
        raise ValueError(f"no satellite named {name!r}")

    def unique_name(self, name: str) -> str:
        """``name``, or ``name 2``, ``name 3`` ... if it is taken."""
        names = {s.name for s in self.sats}
        if name not in names:
            return name
        k = 2
        while f"{name} {k}" in names:
            k += 1
        return f"{name} {k}"

    def rename(self, i: int, name: str) -> str:
        """Rename satellite ``i`` (made unique); its queued maneuvers, burns
        under way and maneuvers aimed at it follow. Returns the name given."""
        old = self.sats[i].name
        if name == old:
            return old
        new = self.unique_name(name)
        for m in [*self.maneuvers, *(b.man for b in self.burns)]:
            if m.sat == old:
                m.sat = new
            if m.target == old:
                m.target = new
        self.sats[i].name = new
        self.log(f"{old} renamed {new}")
        return new

    def _append(self, name, color, props, r, v):
        sat = Satellite(self.unique_name(name), tuple(color), **props)
        drag_props(sat.mass, sat.area, sat.cd, sat.name)
        check_dry_mass(sat.dry_mass, sat.mass, sat.name)
        self.sats.append(sat)
        self.y = np.vstack([self.y, np.concatenate([r, v])[None, :]])
        return sat

    def log(self, text: str, kind: str = "info", t: float | None = None):
        """Append an event (the log keeps the latest ~500)."""
        self.events.append(Event(self.t if t is None else t, text, kind))
        self.event_count += 1
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
        r, v = orbit_state(orbit, self.clock, self.t, j2_acts(self.propagator, self.forces),
                           zonal_terms(self.propagator, self.forces))
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

    def ignored_terms(self) -> list:
        """Enabled force terms the propagator cannot apply: the analytic ones
        carry only two-body motion (kepler) or two-body plus J2 (j2mean)."""
        if self.propagator == "cowell":
            return []
        used = ("j2",) if self.propagator == "j2mean" else ()
        return [t.upper() for t in ForceModel.TERMS if getattr(self.forces, t) and t not in used]

    def warn_ignored(self):
        """Log a warning naming the enabled terms this propagator leaves out."""
        left = self.ignored_terms()
        if left:
            self.log(f"Propagator {self.propagator} ignores {', '.join(left)}: switch to "
                     "cowell to integrate them", "warn")

    # --- launches -----------------------------------------------------------------------
    def ascent_env(self) -> AscentEnv:
        """The Earth model ascents fly in, matching this run's forces and clock."""
        return AscentEnv.from_sim(self.forces, self.clock)

    def plane_of(self, name: str):
        """(inclination deg, normal_at(t)) of a satellite's orbit plane, with
        its J2 nodal regression - the target of a launch window."""
        i = self.require(name)
        el = rv2coe(self.y[i, :3], self.y[i, 3:])
        if el.e >= 1.0:
            raise ValueError(f"{name} is not on a closed orbit")
        rate = 0.0
        if j2_acts(self.propagator, self.forces):     # the mean orbit sets the drift
            a = float(osculating_to_mean_a(el.a, el.e, el.i, el.argp, el.nu))
            i_mean = float(osculating_to_mean_i(el.i, a, el.e, el.argp, el.nu))
            rate = float(j2_secular_rates(a, el.e, i_mean)[0])
        t_ref, inc, raan = self.t, float(el.i), float(el.raan)
        return math.degrees(inc), lambda t: plane_normal(inc, raan + rate * (t - t_ref))

    def plan(self, spec: LaunchSpec):
        """Fly a launch on paper from now (see ``launch.plan_launch``)."""
        return plan_launch(spec, self.ascent_env(), self.t, self.plane_of)

    def launch(self, spec: LaunchSpec, plan=None) -> Satellite:
        """Put a vehicle on its pad; it lifts off at the planned time. A spec
        with ``resume`` (from a snapshot) keeps the plan it holds and, if it
        was in flight, carries on from where it was."""
        spec.name = self.unique_name(spec.name or "Payload")
        if spec.resume is not None:
            validate(spec)
            asc = Ascent.resumed(spec, self.ascent_env(), spec.resume)
            flight = fly(spec, asc.res, asc.env, asc.kick_deg)       # the plan, for the log
        else:
            if plan is None:
                plan = self.plan(spec)
            asc = Ascent(spec, plan.res, self.ascent_env(), plan.kick)
            flight = plan.flight
        n_old = self.n
        props = dict(mass=asc.mass, area=spec.vehicle.area, cd=spec.vehicle.cd)
        if asc.phase != "pad":
            y = asc.peek(self.t)
        else:
            y = asc.pad_state(self.t) if asc.t0 > self.t else asc.state()
        sat = self._append(spec.name, spec.color or palette_color(self.n), props,
                           np.array(y[:3]), np.array(y[3:]))
        sat.family = sat.name
        asc.sat = sat
        self._rebuild_history(n_old)
        self._resize_visible()
        self.ascents.append(asc)
        f = flight
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
            if asc.resumed and t < asc.t + ASCENT_SYNC - 1e-9:
                continue        # restored from a snapshot: it steps again at its own next sync
            asc.resumed = False
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
        """Resolve ``m``'s execution time and queue it (ValueError if it cannot fly)."""
        m.check()
        i = self.require(m.sat)
        if m.kind == "match_velocity":
            self.require(m.target)
        if self.ascent_of(i) is not None:
            raise ValueError(f"{m.sat} is still on its launch vehicle; "
                             "maneuver it after separation")
        if m.timing != "absolute" or m.t is None:
            m.t = self.resolve_time(m)
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
            i = self.index_of(m.sat)
            sat = self.sats[i]
            if sat.status != ACTIVE:
                self.log(f"{m.sat}: {m.describe()} skipped ({sat.status})", "warn")
                continue
            r, v = self.y[i, :3].copy(), self.y[i, 3:].copy()
            if m.kind == "finite":
                self._start_finite(m, i)
                continue
            tv = None
            if m.kind == "match_velocity":
                if m.target not in {s.name for s in self.sats}:
                    self.log(f"{m.sat}: {m.describe()} skipped ({m.target} no longer exists)",
                             "warn")
                    continue
                tv = self.y[self.index_of(m.target), 3:]
            if m.kind == "circularize":
                try:
                    dv = self.circular_velocity(i) - v
                except ValueError as exc:
                    self.log(f"{m.sat}: {m.describe()} skipped ({exc})", "warn")
                    continue
            else:
                dv = impulse_eci(m, r, v, tv)
            dvm = float(np.linalg.norm(dv))
            room = dv_left(sat, m.isp)
            if dvm > room:
                self.log(f"{m.sat}: propellant ran out - {room * 1000:.2f} of "
                         f"{dvm * 1000:.2f} m/s delivered (dry mass {sat.dry_mass:g} kg)", "warn")
                dv, dvm = dv * (room / dvm), room
            self.y[i, 3:] = v + dv
            sat.dv_used += dvm
            sat.mass = max(propellant_after(sat.mass, dvm, m.isp), sat.dry_mass)
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
            floor = sat.dry_mass if sat.dry_mass > 0 else 0.05 * sat.mass
            burnable = max(sat.mass - floor, 0.0) / mdot
            if burnable < duration:
                why = (f"dry mass {sat.dry_mass:g} kg" if sat.dry_mass > 0
                       else "95 % of the mass, with no dry mass set")
                self.log(f"{m.sat}: propellant for only {burnable:,.0f} of {duration:,.0f} s "
                         f"of burn ({why})", "warn")
                duration = burnable
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
            sat = self.sats[self.index_of(b.man.sat)]       # remove() drops its burns
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
        burn_rows = [(rows[j], b, self.sats[j].cd * self.sats[j].area)
                     for b in burns if (j := self.index_of(b.man.sat)) in rows]
        forces, clock = self.forces, self.clock

        def f(t, y):
            r, v = y[:, :3], y[:, 3:]
            ka = cd_am
            if burn_rows and forces.drag:     # a burning craft gets lighter: A/m grows
                ka = cd_am.copy()
                for k, b, cda in burn_rows:
                    ka[k] = cda / (b.m0 - b.mdot * (t - b.t0))
            a = forces.acceleration(r, v, ka, float(clock.gmst(t)) if forces.c22 else 0.0)
            for k, b, _ in burn_rows:
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
        if not math.isfinite(dt) or dt < 0:
            raise ValueError(f"cannot advance by {dt!r} s")
        t_end = self.t + dt
        self._resume()
        self._check_finite(self.t, self._free_idx())     # before an integrator trips on it
        stuck, t_seen = 0, self.t
        self._execute_due()
        while True:
            stuck = stuck + 1 if self.t == t_seen else 0
            t_seen = self.t
            if stuck > 1000:
                raise RuntimeError(f"advance made no progress at t = {self.t:.3f} s")
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

    def coast_states(self, idx, y0: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """States ``y0`` (one row per entry of ``idx``) carried from ``t0`` to ``t1``
        under this run's own propagator and forces, coasting: no maneuvers, no
        burns, no events, and the run itself is untouched. Row k flies with the
        drag properties of satellite ``idx[k]``; an index may repeat, so several
        trial states of one satellite can be flown together. Planners aim with
        this, so their predictions carry the same J2, J3, J4, C22 and drag the
        run will apply."""
        idx = np.asarray(idx, dtype=int)
        y0 = np.array(y0, dtype=float).reshape(len(idx), 6)
        if t1 < t0:
            raise ValueError("coast_states only predicts forward in time")
        if t1 == t0 or not len(idx):
            return y0
        if self.propagator in ("kepler", "j2mean"):
            prop = kepler_propagate if self.propagator == "kepler" else j2_mean_propagate
            r, v = prop(y0[:, :3], y0[:, 3:], t1 - t0)
            return np.concatenate([np.atleast_2d(r), np.atleast_2d(v)], axis=1)
        s = self.integrator
        scratch = Propagator(s.method, s.rtol, s.atol, s.h_max, s.h_fixed)
        return scratch.integrate(self._derivative(idx, []), t0, y0, t1)[1]

    def predict(self, name: str, t1: float) -> np.ndarray:
        """State (6,) satellite ``name`` will have at time ``t1`` if it coasts
        from now (see :meth:`coast_states`)."""
        i = self.require(name)
        return self.coast_states([i], self.y[i:i + 1], self.t, t1)[0]

    def coast_track(self, i: int, times) -> np.ndarray:
        """States ``(len(times), 6)`` of satellite ``i`` at the increasing absolute
        ``times`` (all at or after now), coasting as in :meth:`coast_states`."""
        return self._coast_samples([i], self.y[i:i + 1], self.t, times)[0]

    def _coast_samples(self, idx, y0, t0: float, times, h_max: float | None = None):
        """States ``(len(idx), len(times), 6)`` of the rows ``y0`` (satellites
        ``idx``, as in :meth:`coast_states`) at the increasing absolute ``times``
        >= ``t0``. ``h_max`` may lengthen the adaptive step limit for long looks
        ahead (the tolerance still holds)."""
        times = np.asarray(times, dtype=float)
        if times.size and (times[0] < t0 or np.any(np.diff(times) < 0)):
            raise ValueError("coasting needs increasing times from the start on")
        idx = np.asarray(idx, dtype=int)
        y0 = np.array(y0, dtype=float).reshape(len(idx), 6)
        n, k = len(idx), len(times)
        if self.propagator in ("kepler", "j2mean"):
            prop = kepler_propagate if self.propagator == "kepler" else j2_mean_propagate
            r, v = prop(np.repeat(y0[:, :3], k, 0), np.repeat(y0[:, 3:], k, 0),
                        np.tile(times - t0, n))
            return np.concatenate([np.atleast_2d(r), np.atleast_2d(v)], axis=1).reshape(n, k, 6)
        s = self.integrator
        scratch = Propagator(s.method, s.rtol, s.atol, max(s.h_max, h_max or 0.0), s.h_fixed)
        f = self._derivative(idx, [])
        out = np.empty((n, k, 6))
        t, y = t0, y0
        for j, tj in enumerate(times):
            t, y = scratch.integrate(f, t, y, float(tj))   # the step size carries over
            out[:, j] = y
        return out

    def _perturbed(self) -> bool:
        """Whether anything but central gravity moves the satellites."""
        f = self.forces
        return self.propagator == "j2mean" or (
            self.propagator == "cowell" and (f.j2 or f.j3 or f.j4 or f.c22 or f.drag))

    def circular_velocity(self, i: int) -> np.ndarray:
        """Velocity that puts satellite ``i`` on a circular orbit through where it is,
        in its present plane, for this run's forces. Two-body that is sqrt(mu/r)
        level; under J2 it is ~5 m/s faster at the equator and the orbit's radius
        keeps a forced ripple at twice the orbital frequency. So "circular" means
        what it does in practice: no free eccentricity, i.e. no once-per-rev term
        in the radius over the coming revolution. Found by Newton on the radial
        and along-track speed, the next revolution flown with the run's forces."""
        r = self.y[i, :3].copy()
        rm = float(np.linalg.norm(r))
        rh = r / rm
        th = unit(np.cross(np.cross(r, self.y[i, 3:]), rh))     # level, along the motion
        vc = math.sqrt(MU_EARTH / rm)
        if not self._perturbed():
            return vc * th
        period = 2.0 * math.pi * rm / vc
        times = self.t + np.linspace(0.0, period, CIRC_SAMPLES + 1)[1:]
        p = np.array([0.0, vc])                                  # radial, along-track km/s
        step = np.array([[0.0, 0.0], [CIRC_STEP, 0.0], [0.0, CIRC_STEP]])
        for _ in range(CIRC_ITER):
            trial = p + step
            y0 = np.concatenate([np.repeat(r[None], 3, 0),
                                 trial[:, :1] * rh + trial[:, 1:] * th], axis=1)
            track = self._coast_samples([i] * 3, y0, self.t, times, h_max=period / 360)
            c = np.array([_once_per_rev(track[k], rh, th, times - self.t) for k in range(3)])
            if np.linalg.norm(c[0]) < CIRC_TOL:
                return p[0] * rh + p[1] * th
            p = p - np.linalg.solve((c[1:] - c[0]).T / CIRC_STEP, c[0])
        raise ValueError(f"no circular orbit found within {CIRC_TOL * 1e6:.0f} mm "
                         f"(still {np.linalg.norm(c[0]) * 1000:,.3f} m of eccentricity)")

    def resolve_time(self, m: Maneuver) -> float:
        """Absolute time ``m`` should fire, from now. Apsis and node timings are
        found on the trajectory this run will actually fly (J2, drag and all),
        starting from the two-body estimate: under J2 a "circular" orbit has an
        osculating eccentricity of ~1e-3 whose perigee can sit anywhere, and the
        osculating node is seconds off the real equator crossing."""
        i = self.require(m.sat)
        r, v = self.y[i, :3], self.y[i, 3:]
        t_est = resolve_time(m, self.t, r, v)
        if m.timing in ("periapsis", "apoapsis") and self.propagator != "kepler":
            return self._next_apsis(i, m.timing == "apoapsis", t_est)
        if m.timing in ("ascending_node", "descending_node") and self.propagator != "kepler":
            return self._next_node(i, t_est)
        return t_est

    def _next_apsis(self, i: int, highest: bool, t_est: float) -> float:
        """Time of the lowest (or highest) point of satellite ``i``'s coming
        revolution: the extreme of 180 samples, then the root of r.v (radial
        speed) between its neighbours by bisection, to ``APSIS_TIME_TOL``.
        Near-circular orbits make d(r.v)/dt ~ 0, which is why this is not
        Newton."""
        el = rv2coe(self.y[i, :3], self.y[i, 3:])
        if el.e >= 1.0:
            return t_est                    # an open path's periapsis is its only apsis
        times = self.t + np.linspace(0.0, el.period, 181)[1:]
        track = self.coast_track(i, times)
        rad = np.linalg.norm(track[:, :3], axis=1)
        k = int(np.argmax(rad) if highest else np.argmin(rad))
        k = min(max(k, 1), len(times) - 2)
        a, b = float(times[k - 1]), float(times[k + 1])
        # integration is a flow, so it may restart from the sample before; the
        # analytic j2mean is not (it re-reads mean elements), so it starts from now
        t0, y0 = (a, track[k - 1]) if self.propagator == "cowell" else (self.t, self.y[i])

        def g(t):
            y = self.coast_states([i], y0[None], t0, t)[0]
            return float(np.dot(y[:3], y[3:]))

        ga, gb = g(a), float(np.dot(track[k + 1, :3], track[k + 1, 3:]))
        if ga * gb > 0:
            return float(times[k])          # no sign change: the sample is the extreme
        while b - a > APSIS_TIME_TOL:       # bisection: ~26 halvings, each a short coast
            t = 0.5 * (a + b)
            gt = g(t)
            if (gt < 0) == (ga < 0):
                a, ga = t, gt
            else:
                b = t
        return 0.5 * (a + b)

    def _next_node(self, i: int, t_est: float) -> float:
        """The real equator crossing nearest the two-body node estimate, by
        Newton on z = 0 (forward from now, so never before it)."""
        t = t_est
        for _ in range(8):
            y = self.coast_states([i], self.y[i:i + 1], self.t, t)[0]
            if abs(y[5]) < 1e-9:
                raise ValueError("the orbit runs along the equator: it has no node to burn at")
            t_new = max(t - y[2] / y[5], self.t)
            if abs(t_new - t) < 1e-3:
                return t_new
            t = t_new
        return t

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
    def _check_finite(self, t: float, idx: np.ndarray):
        """Raise FloatingPointError naming the satellites ``idx`` whose state is
        not finite: a NaN is never a result to draw or carry on from."""
        bad = ~np.isfinite(self.y[idx]).all(axis=1)
        if bad.any():
            names = ", ".join(self.sats[i].name for i in idx[bad])
            raise FloatingPointError(f"non-finite state at t = {t:.3f} s for {names}")

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
            self._check_finite(t, idx)
            r = self.y[idx, :3]
            v = self.y[idx, 3:]
            rm = np.linalg.norm(r, axis=1)
            alt = geodetic_height(r)
            # only a *descending* vehicle can re-enter: launches start low and climb
            descending = np.sum(r * v, axis=1) < 0
            gone = ((alt < REENTRY_ALT) & descending) | (rm > SOI_EARTH)
            dips = self._dips(t, idx, alt) if idx.size <= DIP_CHECK_LIMIT else {}
            for k in np.flatnonzero(gone | np.isin(idx, list(dips))):
                i = idx[k]
                sat = self.sats[i]
                speed = float(np.linalg.norm(v[k]))
                if rm[k] > SOI_EARTH:
                    sat.status = ESCAPED
                    when = self._when(t, i, lambda pos, _t: np.linalg.norm(pos) - SOI_EARTH)
                    self.log(f"{sat.name} left Earth's sphere of influence", "alert", when)
                elif alt[k] <= 0:
                    sat.status = IMPACTED
                    when = self._when(t, i, lambda pos, _t: geodetic_height(pos))
                    speed = self._speed_at(t, i, when, speed)
                    self.log(f"{sat.name} impacted the surface at {speed:.2f} km/s", "alert",
                             when)
                else:
                    sat.status = DECAYED
                    when = dips.get(int(i)) or self._when(
                        t, i, lambda pos, _t: geodetic_height(pos) - REENTRY_ALT)
                    speed = self._speed_at(t, i, when, speed)
                    self.log(f"{sat.name} re-entered (below {REENTRY_ALT:.0f} km, "
                             f"{speed:.2f} km/s)", "alert", when)
                dropped = [m for m in self.maneuvers if m.sat == sat.name]
                if dropped:
                    self.log(f"{sat.name}: {len(dropped)} pending maneuver"
                             f"{'s' if len(dropped) != 1 else ''} cancelled", "warn")
                self.maneuvers = [m for m in self.maneuvers if m.sat != sat.name]
                self.burns = [b for b in self.burns if b.man.sat != sat.name]
                changed = True
            self._events(t, idx)
        if force_record or t - self._last_record >= self.record_dt - 1e-9 or changed:
            self.history.record(t, self.y)
            self._last_record = t
        return changed

    def _previous(self, t: float):
        """``(t0, y0)`` of the step that just ended at ``t``, or None when nothing
        continuous joins it to now (first step, a burn, a satellite added)."""
        prev = self._last_sample
        if prev is None or not prev[0] < t or prev[1].shape != self.y.shape:
            return None
        return prev[0], prev[1]

    def _when(self, t: float, i: int, f) -> float:
        """Time within the step just ended at ``t`` where ``f(position, time)``
        changed sign for satellite ``i``, found on the step's Hermite curve; ``t``
        itself when there is no previous state or no sign change on the curve."""
        prev = self._previous(t)
        if prev is None:
            return t
        t0, y0 = prev
        h = t - t0
        y_a, y_b = y0[i], self.y[i]

        def g(pos, s):
            return f(pos, t0 + s * h)

        if (g(y_a[:3], 0.0) < 0) == (g(y_b[:3], 1.0) < 0):
            return t
        return t0 + h * _crossing(g, y_a, y_b, h)

    def _speed_at(self, t: float, i: int, when: float, now: float) -> float:
        """Speed of satellite ``i`` at ``when`` within the step just ended at ``t``
        (``now``, its speed at the end, when there is no step to go back along)."""
        prev = self._previous(t)
        if prev is None or not when < t:
            return now
        t0, y0 = prev
        return _hermite_speed(y0[i], self.y[i], t - t0, (when - t0) / (t - t0))

    def _dips(self, t: float, idx: np.ndarray, alt: np.ndarray) -> dict:
        """Satellites ``idx`` (heights ``alt`` now) that went below the re-entry
        height during the step and came back up before it ended: {i: time}."""
        prev = self._previous(t)
        if prev is None:
            return {}
        t0, y0 = prev
        h = t - t0
        out = {}
        s = np.linspace(0.0, 1.0, 17)

        def low(pos, _s):
            return geodetic_height(pos) - REENTRY_ALT

        for k in np.flatnonzero(alt < REENTRY_ALT + DIP_MARGIN):
            i = int(idx[k])
            if not geodetic_height(y0[i, :3]) >= REENTRY_ALT:
                continue                      # already below (climbing out): not a dip
            heights = geodetic_height(_hermite_at(y0[i], self.y[i], h, s))
            below = np.flatnonzero(heights < REENTRY_ALT)
            if below.size:
                out[i] = t0 + h * _crossing(low, y0[i], self.y[i], h, 0.0, s[below[0]])
        return out

    def _events(self, t: float, idx: np.ndarray):
        """Log eclipse, station AOS/LOS and close-approach events."""
        free = {int(i) for i in idx}
        watch = free if self.n <= EVENT_DETAIL_LIMIT else self.watch & free
        if watch:
            w = np.array(sorted(watch))
            self._eclipse_events(t, w)
            if self.stations:
                self._station_events(t, w)
        self._approach_events(t, idx)

    def _eclipse_events(self, t: float, w: np.ndarray):
        """Log satellites ``w`` entering and leaving the Earth's shadow."""
        sun = sun_position(self.clock.jd(t))
        frac = shadow_fraction(self.y[:, :3][w], sun)

        def lit(pos, _t):                   # half the Sun's disc showing: the crossing
            return float(np.ravel(shadow_fraction(pos, sun))[0]) - 0.5

        for i, fr in zip(w, frac, strict=True):
            sat = self.sats[i]
            if sat.shadow >= 0.5 > fr:
                self.log(f"{sat.name} entered Earth's shadow", "eclipse", self._when(t, i, lit))
            elif sat.shadow < 0.5 <= fr:
                self.log(f"{sat.name} back in sunlight", "eclipse", self._when(t, i, lit))
            sat.shadow = float(fr)

    def _station_events(self, t: float, w: np.ndarray):
        """Log ground stations gaining (AOS) and losing (LOS) sight of satellites ``w``."""
        if self._visible.shape != (len(self.stations), self.n):
            self._resize_visible()
        r_ecef = eci_to_ecef(self.y[:, :3][w], self.clock.gmst(t))
        for s, st in enumerate(self.stations):
            mask = math.radians(st.min_el)

            def above(pos, tt, st=st, mask=mask):     # elevation over the mask, rad
                elev = st.look_angles(eci_to_ecef(pos, self.clock.gmst(tt)))[1]
                return float(np.ravel(elev)[0]) - mask

            for i, now in zip(w, st.sees(r_ecef), strict=True):
                if now and not self._visible[s, i]:
                    self.log(f"AOS {self.sats[i].name} @ {st.name}", "station",
                             self._when(t, i, above))
                elif not now and self._visible[s, i]:
                    self.log(f"LOS {self.sats[i].name} @ {st.name}", "station",
                             self._when(t, i, above))
                self._visible[s, i] = now

    def _approach_events(self, t: float, idx: np.ndarray):
        """Update :attr:`closest` and log new close approaches among the satellites
        ``idx`` (vehicles still inside the atmosphere, e.g. on a launch pad, are ignored)."""
        r_all = self.y[:, :3]
        orbiting = idx[geodetic_height(r_all[idx]) > 100.0]
        if orbiting.size < 2:
            self.closest = (math.inf, -1, -1)
            mask = np.zeros(self.n, dtype=bool)
            mask[orbiting] = True
            self._last_sample = (t, self.y.copy(), mask)   # event times still use it
            return
        r = r_all[orbiting]
        sweep = self._sweep(t, orbiting)
        p, q = self._candidate_pairs(r, sweep)
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
        now = {(a, b): (dist, tca)
               for a, b, dist, tca in self._approaches(t, orbiting, p, q, d, sweep)}
        for a, b in sorted(set(now) - self._close_pairs, key=lambda ab: now[ab][1]):
            dist, tca = now[(a, b)]
            self.log(f"Close approach: {self.sats[a].name} - {self.sats[b].name} "
                     f"{dist:.2f} km", "alert", tca)
        self._close_pairs = set(now)
        mask = np.zeros(self.n, dtype=bool)
        mask[orbiting] = True
        self._last_sample = (t, self.y.copy(), mask)

    def _candidate_pairs(self, r: np.ndarray, sweep):
        """Index pairs ``(p, q)`` of rows of ``r`` that may have come close: every
        pair in small ensembles, pairs found through a spatial grid in large ones."""
        if len(r) <= ALL_PAIRS_LIMIT:
            return np.triu_indices(len(r), 1)
        # during the step each satellite stayed within `spread` of the middle of
        # its chord, so a pair that met had middles within the alert distance plus
        # both spreads
        mid, spread = (r, np.zeros(len(r))) if sweep is None else (sweep[4], sweep[5])
        radius = self.conjunction_km + 2 * float(np.max(spread, initial=0.0))
        p, q, _ = _near_pairs(mid, max(radius, 1.0))
        return p, q

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
        # a vehicle on its pad or in powered flight stays a launch: it keeps its plan
        # (liftoff time, plane, pitch kick) and, in flight, its whole flight state
        flying = {id(a.sat) for a in self.ascents}
        # a finite burn under way goes on in the new scenario: what is left of it
        # starts at once, from the mass the satellite has now
        burning = {b.man.sat: b for b in self.burns if b.t1 > self.t + 1e-6}
        sc.launches = []
        for a in self.ascents:
            spec = a.spec.copy()
            spec.name = a.sat.name
            spec.timing, spec.t0, spec.kick = "absolute", a.t0 - self.t, a.kick_deg
            if spec.guidance == "orbit":
                spec.inclination = a.res.inclination      # a plane-timed launch takes its target's
            spec.resume = a.save(self.t)
            sc.launches.append(spec)
        for i, s in enumerate(self.sats):
            if s.status != ACTIVE or id(s) in flying:
                continue
            b = burning.get(s.name)
            mass = s.mass if b is None else b.m0 - b.mdot * (self.t - b.t0)
            sc.satellites.append(SatSpec(
                s.name, {"type": "state", "r": self.y[i, :3].tolist(), "v": self.y[i, 3:].tolist()},
                list(s.color), mass, s.area, s.cd, dry_mass=s.dry_mass))
        sc.stations = list(self.stations)
        sc.forces = ForceModel.from_dict(self.forces.to_dict())
        sc.propagator = self.propagator
        sc.maneuvers = []
        for b in burning.values():
            rest = Maneuver.from_dict(b.man.to_dict())
            rest.timing, rest.t, rest.duration = "absolute", 0.0, b.t1 - self.t
            rest.done, rest.result = False, ""
            sc.maneuvers.append(rest)
        for m in self.maneuvers:
            mm = Maneuver.from_dict(m.to_dict())
            mm.t = m.t - self.t
            sc.maneuvers.append(mm)
        return sc


__all__ = ["Simulation", "Satellite", "History", "Event", "PROPAGATORS", "ASCENT_SYNC"]
