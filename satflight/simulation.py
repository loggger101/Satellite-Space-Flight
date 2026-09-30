"""The simulation engine: an Earth-centred ensemble of satellites advanced
in time with a selectable propagator, executing scheduled manoeuvres and
detecting events.

Propagators
-----------
``cowell``  numerical integration of every enabled force (default).
``kepler``  exact analytic two-body motion (universal variables).
``j2mean``  analytic two-body plus J2 secular drift of RAAN, perigee and
            mean anomaly - fast enough for thousands of satellites.

Events logged: manoeuvre execution, eclipse entry/exit, ground-station
AOS/LOS, close approaches between satellites, re-entry, surface impact and
escape from Earth's sphere of influence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis import j2_mean_propagate
from .constants import SOI_EARTH
from .eclipse import shadow_fraction
from .elements import kepler_propagate
from .ephemeris import sun_position
from .forces import ForceModel, approx_altitude, conservative_energy
from .frames import eci_to_ecef, look_angles
from .integrators import Propagator
from .maneuvers import (G0, Maneuver, finite_direction, impulse_eci, propellant_after,
                        resolve_time)
from .scenario import GroundStation, Scenario, expand, orbit_state, spread_along_orbit
from .timeutil import Clock, format_duration

PROPAGATORS = ("cowell", "kepler", "j2mean")
ACTIVE, DECAYED, IMPACTED, ESCAPED = "active", "re-entered", "impacted", "escaped"

REENTRY_ALT = 80.0        # km: below this the vehicle is considered lost
EVENT_DETAIL_LIMIT = 40   # log per-satellite eclipse/AOS events only below this N


@dataclass
class Satellite:
    name: str
    color: tuple
    mass: float = 500.0
    area: float = 5.0
    cd: float = 2.2
    cr: float = 1.5
    status: str = ACTIVE
    dv_used: float = 0.0          # km/s
    shadow: float = 1.0           # illuminated fraction
    energy_ref: tuple | None = None   # (t, energy) baseline for the drift readout


@dataclass
class Event:
    t: float
    text: str
    kind: str = "info"


@dataclass
class FiniteBurn:
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
        return int(np.clip(cls.BUDGET // (6 * max(n, 1)), 120, 2000))

    def record(self, t: float, y: np.ndarray):
        self.data[:, self.head] = y
        self.times[self.head] = t
        self.head = (self.head + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def _order(self):
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
        self.data = np.delete(self.data, i, axis=0)

    def clear(self):
        self.head = 0
        self.count = 0


class Simulation:
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
        self.conjunction_km = 10.0
        self.watch: set[int] = set()       # indices the UI wants detailed events for
        self._last_record = -math.inf
        self._visible = np.zeros((0, 0), dtype=bool)
        self._close_pairs: set = set()
        self.closest = (math.inf, -1, -1)  # (km, i, j)

        for name, color, props, r, v in expand(scenario, self.clock, 0.0):
            self._append(name, color, props, r, v)
        self.history = History(len(self.sats))
        for m in scenario.maneuvers:
            self.schedule(Maneuver.from_dict(m.to_dict()))
        self._post_step(self.t, force_record=True)
        self.log(f"Scenario '{scenario.name}' loaded: {len(self.sats)} satellites, "
                 f"{len(self.stations)} stations, propagator {self.propagator}")

    # --- bookkeeping -----------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.sats)

    @property
    def active(self) -> np.ndarray:
        return np.array([s.status == ACTIVE for s in self.sats], dtype=bool)

    def index_of(self, name: str) -> int:
        for i, s in enumerate(self.sats):
            if s.name == name:
                return i
        raise KeyError(name)

    def unique_name(self, name: str) -> str:
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

    def log(self, text: str, kind: str = "info"):
        self.events.append(Event(self.t, text, kind))
        if len(self.events) > 600:
            del self.events[:100]

    def datetime(self):
        return self.clock.datetime(self.t)

    def jd(self, t: float | None = None) -> float:
        return self.clock.jd(self.t if t is None else t)

    def gmst(self, t: float | None = None) -> float:
        return self.clock.gmst(self.t if t is None else t)

    # --- adding / removing ---------------------------------------------------------------
    def add_satellites(self, orbit: dict, name: str, color=None, count: int = 1, **props):
        """Add satellites defined by an orbit spec evaluated at the current time."""
        r, v = orbit_state(orbit, self.clock, self.t)
        rows = []
        for k, (rr, vv) in enumerate(spread_along_orbit(r, v, count)):
            nm = name if count == 1 else f"{name}-{k + 1}"
            rows.append((nm, color or _auto_color(self.n + k), props, rr, vv))
        return self.add_many(rows)

    def add_state(self, name: str, r, v, color=None, **props):
        return self.add_many([(name, color or _auto_color(self.n), props,
                               np.asarray(r, float), np.asarray(v, float))])[0]

    def add_many(self, rows):
        """Bulk add ``(name, color, props, r, v)`` rows (one history rebuild)."""
        n_old = self.n
        added = [self._append(*row) for row in rows]
        self._rebuild_history(n_old)
        self._visible = np.zeros((len(self.stations), self.n), dtype=bool)
        names = ", ".join(s.name for s in added[:3])
        self.log(f"Added {len(added)} satellite{'s' if len(added) != 1 else ''}: {names}"
                 f"{' ...' if len(added) > 3 else ''}")
        return added

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
        sat = self.sats.pop(i)
        self.y = np.delete(self.y, i, axis=0)
        self.history.remove_row(i)
        self.maneuvers = [m for m in self.maneuvers if m.sat != sat.name]
        self.burns = [b for b in self.burns if b.man.sat != sat.name]
        self._visible = np.zeros((len(self.stations), self.n), dtype=bool)
        self._close_pairs.clear()
        self.watch = {w - (w > i) for w in self.watch if w != i}
        self.log(f"Removed {sat.name}")

    def add_station(self, st: GroundStation):
        self.stations.append(st)
        self._visible = np.zeros((len(self.stations), self.n), dtype=bool)
        self.log(f"Ground station {st.name} ({st.lat:.2f}, {st.lon:.2f})")

    def set_propagator(self, name: str):
        if name not in PROPAGATORS:
            raise ValueError(name)
        self.propagator = name
        self.log(f"Propagator -> {name}")

    # --- manoeuvres --------------------------------------------------------------------
    def schedule(self, m: Maneuver) -> Maneuver:
        i = self.index_of(m.sat)
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
                self.log(f"{m.sat}: manoeuvre skipped ({exc})", "warn")
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

    def _start_finite(self, m: Maneuver, i: int):
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
        cd_am = np.array([self.sats[i].cd * self.sats[i].area / self.sats[i].mass for i in idx])
        cr_am = np.array([self.sats[i].cr * self.sats[i].area / self.sats[i].mass for i in idx])
        return cd_am, cr_am

    def _derivative(self, idx: np.ndarray, burns: list):
        cd_am, cr_am = self._props(idx)
        rows = {int(i): k for k, i in enumerate(idx)}
        burn_rows = [(rows[self.index_of(b.man.sat)], b) for b in burns
                     if self.index_of(b.man.sat) in rows]
        clock, forces = self.clock, self.forces

        def f(t, y):
            r, v = y[:, :3], y[:, 3:]
            a = forces.acceleration(clock.jd(t), r, v, cd_am, cr_am)
            for k, b in burn_rows:
                m = b.m0 - b.mdot * (t - b.t0)
                d = finite_direction(b.man, r[k:k + 1], v[k:k + 1])[0]
                a[k] += d * (b.man.thrust / m) * 1e-3
            return np.concatenate([v, a], axis=1)
        return f

    def _next_boundary(self, t_end: float) -> float:
        tb = t_end
        for m in self.maneuvers:
            if self.t < m.t < tb:
                tb = m.t
        for b in self.burns:
            if self.t < b.t1 < tb:
                tb = b.t1
        return tb

    def advance(self, dt: float):
        """Advance the simulation by ``dt`` seconds of simulated time."""
        t_end = self.t + dt
        guard = 0
        self._execute_due()
        while self.t < t_end - 1e-9 and guard < 10000:
            guard += 1
            t_seg = self._next_boundary(t_end)
            idx = np.flatnonzero(self.active)
            if idx.size == 0:
                self.t = t_seg
            else:
                self._propagate(idx, t_seg)
            self._finish_burns()
            self._execute_due()

    def _propagate(self, idx: np.ndarray, t_seg: float):
        stop = {"flag": False}

        def callback(t, ya):
            self.t = t
            self.y[idx] = ya
            stop["flag"] = self._post_step(t, idx=idx)
            return stop["flag"]

        if self.propagator == "cowell":
            burns = [b for b in self.burns if b.t0 <= self.t + 1e-9 and t_seg <= b.t1 + 1e-9]
            f = self._derivative(idx, burns)
            t, ya = self.integrator.integrate(f, self.t, self.y[idx].copy(), t_seg, callback)
            self.t = t
            self.y[idx] = ya
        else:
            h = self.integrator.h_max
            while self.t < t_seg - 1e-9:
                step = min(h, t_seg - self.t)
                y0 = self.y[idx]
                if self.propagator == "kepler":
                    r, v = kepler_propagate(y0[:, :3], y0[:, 3:], step)
                else:
                    r, v = j2_mean_propagate(y0[:, :3], y0[:, 3:], step)
                self.integrator.stats.accepted += 1
                self.integrator.stats.last_h = step
                if callback(self.t + step, np.concatenate([r, v], axis=1)):
                    break

    # --- per-step checks ------------------------------------------------------------------
    def _post_step(self, t: float, idx=None, force_record: bool = False) -> bool:
        """Record history and detect events. Returns True if a satellite left
        the active set (the integrator must restart with the new set)."""
        if idx is None:
            idx = np.flatnonzero(self.active)
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
        detail = self.n <= EVENT_DETAIL_LIMIT
        watch = set(int(i) for i in idx) if detail else (self.watch & set(int(i) for i in idx))
        jd = self.clock.jd(t)
        r_all = self.y[:, :3]

        # eclipses
        if watch:
            w = np.array(sorted(watch))
            frac = shadow_fraction(r_all[w], sun_position(jd))
            for i, fr in zip(w, frac):
                sat = self.sats[i]
                if sat.shadow >= 0.5 > fr:
                    self.log(f"{sat.name} entered Earth's shadow", "eclipse")
                elif sat.shadow < 0.5 <= fr:
                    self.log(f"{sat.name} back in sunlight", "eclipse")
                sat.shadow = float(fr)

        # ground stations
        if self.stations and watch:
            if self._visible.shape != (len(self.stations), self.n):
                self._visible = np.zeros((len(self.stations), self.n), dtype=bool)
            w = np.array(sorted(watch))
            r_ecef = eci_to_ecef(r_all[w], self.clock.gmst(t))
            for s, st in enumerate(self.stations):
                _, el, _ = look_angles(st.ecef(), math.radians(st.lat), math.radians(st.lon), r_ecef)
                vis = el >= math.radians(st.min_el)
                for i, now in zip(w, vis):
                    if now and not self._visible[s, i]:
                        self.log(f"AOS {self.sats[i].name} @ {st.name}", "station")
                    elif not now and self._visible[s, i]:
                        self.log(f"LOS {self.sats[i].name} @ {st.name}", "station")
                    self._visible[s, i] = now

        # close approaches (pairwise, small ensembles only)
        # (vehicles still inside the atmosphere, e.g. on a launch pad, are ignored)
        orbiting = idx[approx_altitude(r_all[idx]) > 100.0]
        if 1 < orbiting.size <= 300:
            r = r_all[orbiting]
            d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            k = int(np.argmin(d))
            a, b = divmod(k, orbiting.size)
            self.closest = (float(d[a, b]), int(orbiting[a]), int(orbiting[b]))
            close = np.argwhere(np.triu(d < self.conjunction_km))
            now = {(int(orbiting[p]), int(orbiting[q])): float(d[p, q]) for p, q in close}
            for p, q in set(now) - self._close_pairs:
                self.log(f"Close approach: {self.sats[p].name} - {self.sats[q].name} "
                         f"{now[(p, q)]:.2f} km", "alert")
            self._close_pairs = set(now)
        else:
            self.closest = (math.inf, -1, -1)

    # --- diagnostics ------------------------------------------------------------------------
    def energy(self, i: int) -> float:
        r, v = self.y[i:i + 1, :3], self.y[i:i + 1, 3:]
        return float(conservative_energy(r, v, self.forces)[0])

    def station_visibility(self, i: int):
        """[(station, az deg, el deg, range km)] for stations that see satellite i."""
        out = []
        r_ecef = eci_to_ecef(self.y[i, :3], self.gmst())
        for st in self.stations:
            az, el, rng = look_angles(st.ecef(), math.radians(st.lat), math.radians(st.lon), r_ecef)
            if el >= math.radians(st.min_el):
                out.append((st, math.degrees(float(az)), math.degrees(float(el)), float(rng)))
        return out

    def snapshot_scenario(self, name: str | None = None) -> Scenario:
        """The current state as a new scenario whose epoch is 'now'."""
        from .scenario import SatSpec
        sc = self.scenario.copy()
        sc.name = name or f"{self.scenario.name} @ {format_duration(self.t)}"
        sc.epoch = self.datetime()
        sc.satellites = []
        sc.constellations = []
        for i, s in enumerate(self.sats):
            if s.status != ACTIVE:
                continue
            sc.satellites.append(SatSpec(
                s.name, {"type": "state", "r": self.y[i, :3].tolist(), "v": self.y[i, 3:].tolist()},
                list(s.color), s.mass, s.area, s.cd, s.cr))
        sc.stations = list(self.stations)
        sc.forces = ForceModel.from_dict(self.forces.to_dict())
        sc.propagator = self.propagator
        sc.maneuvers = []
        for m in self.maneuvers:
            mm = Maneuver.from_dict(m.to_dict())
            mm.t = m.t - self.t
            sc.maneuvers.append(mm)
        return sc


def _auto_color(i: int):
    from .scenario import palette_color
    return palette_color(i)


__all__ = ["Simulation", "Satellite", "History", "Event", "PROPAGATORS"]
