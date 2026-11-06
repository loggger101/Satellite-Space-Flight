"""Headless batch runs: propagate a scenario and write ephemerides to CSV.

    python -m satflight.batch scenarios/hohmann_to_geo.json --duration 12h --step 60 --out run.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import sys
from pathlib import Path

import numpy as np

from .constants import R_EARTH
from .elements import rv2coe
from .frames import ecef_to_geodetic, eci_to_ecef
from .scenario import Scenario
from .simulation import Simulation
from .timeutil import format_duration, format_epoch

EVENT_HEADER = ["t_s", "utc", "kind", "text"]
HEADER = ["t_s", "utc", "name", "status", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms",
          "alt_km", "lat_deg", "lon_deg", "a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]


def _rows(sim: Simulation, t: float, y: np.ndarray):
    """One CSV row (see ``HEADER``) per satellite for ensemble state ``y`` at time ``t``."""
    theta = sim.clock.gmst(t)
    lat, lon, alt = ecef_to_geodetic(eci_to_ecef(y[:, :3], theta))
    el = rv2coe(y[:, :3], y[:, 3:])
    a, e = np.atleast_1d(el.a), np.atleast_1d(el.e)
    angles = np.degrees(np.atleast_2d([el.i, el.raan, el.argp, el.nu]))    # (4, N)
    utc = format_epoch(sim.clock.datetime(t))
    for k, s in enumerate(sim.sats):
        yield [f"{t:.3f}", utc, s.name, s.status, *(f"{x:.6f}" for x in y[k]),
               f"{alt[k]:.4f}", f"{math.degrees(lat[k]):.6f}", f"{math.degrees(lon[k]):.6f}",
               f"{a[k]:.4f}", f"{e[k]:.8f}", *(f"{x:.6f}" for x in angles[:, k])]


def export_history_csv(sim: Simulation, path) -> int:
    """Write the simulation's recorded history (trail buffer) to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times, data = sim.history.series()
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for k, t in enumerate(times):
            for row in _rows(sim, float(t), data[:, k]):
                w.writerow(row)
                n += 1
    return n


def new_events(sim: Simulation, seen: int):
    """Events logged since ``seen`` had been (``sim.event_count`` then), oldest
    first. RuntimeError if the log, which keeps only the latest, lost some."""
    new = sim.event_count - seen
    if new > len(sim.events):
        raise RuntimeError(f"{new} events in one step overflowed the event log "
                           f"(it keeps {len(sim.events)}); take shorter steps")
    return sim.events[len(sim.events) - new:] if new else []


def parse_duration(text: str) -> float:
    """Seconds from ``90``, ``90s``, ``15m``, ``12h`` or ``3d`` (ValueError otherwise)."""
    text = str(text).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        x = float(text[:-1]) * units[text[-1]] if text and text[-1] in units else float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration such as 90, 90s, 15m, 12h or 3d") from None
    if not math.isfinite(x):
        raise ValueError(f"{text!r} is not a finite duration")
    return x


def run(scenario: Scenario, duration: float, step: float, out=None, quiet: bool = False,
        events=None):
    """Propagate ``scenario`` for ``duration`` s, writing every ``step`` s to CSV ``out``
    (if given) and every event logged to CSV ``events`` (if given), then print a
    summary unless ``quiet``. Returns the finished simulation."""
    if not duration >= 0:
        raise ValueError(f"duration must be zero or more seconds, got {duration}")
    if not step > 0:
        raise ValueError(f"step must be a positive number of seconds, got {step}")
    sim = Simulation(scenario)
    for path in (out, events):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        writer = log = None
        if out:
            writer = csv.writer(stack.enter_context(open(out, "w", newline="", encoding="utf-8")))
            writer.writerow(HEADER)
        if events:
            log = csv.writer(stack.enter_context(open(events, "w", newline="", encoding="utf-8")))
            log.writerow(EVENT_HEADER)
        t_end = sim.t + duration
        seen = 0
        while True:
            if writer:
                writer.writerows(_rows(sim, sim.t, sim.y))
            if log:
                log.writerows([f"{e.t:.3f}", format_epoch(sim.clock.datetime(e.t)), e.kind, e.text]
                              for e in new_events(sim, seen))
                seen = sim.event_count
            if sim.t >= t_end - 1e-9:
                break
            sim.advance(min(step, t_end - sim.t))
    if not quiet:
        summary(sim)
    return sim


def summary(sim: Simulation, out=None):
    """Print a per-satellite table and the key events (to ``out``, default the
    standard output at call time)."""
    out = out or sys.stdout
    print(f"Scenario: {sim.scenario.name}   elapsed {format_duration(sim.t)}   "
          f"({sim.forces.label()}, {sim.propagator})", file=out)
    print(f"{'name':<28}{'status':<12}{'alt km':>11}{'a km':>12}{'e':>10}{'i deg':>9}"
          f"{'dV m/s':>9}", file=out)
    for i, s in enumerate(sim.sats):
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        alt = np.linalg.norm(r) - R_EARTH
        print(f"{s.name[:27]:<28}{s.status:<12}{alt:>11.1f}{el.a:>12.1f}{el.e:>10.5f}"
              f"{math.degrees(el.i):>9.3f}{s.dv_used * 1000:>9.1f}", file=out)
    events = [e for e in sim.events if e.kind in ("maneuver", "alert", "warn", "launch")]
    if events:
        print("\nKey events:", file=out)
        for e in events[-40:]:
            print(f"  T+{format_duration(e.t):>13}  {e.text}", file=out)


def main(argv=None):
    """Command-line entry point (``python -m satflight.batch``)."""
    ap = argparse.ArgumentParser(description="Propagate a scenario without graphics.")
    ap.add_argument("scenario", help="scenario JSON file")
    ap.add_argument("--duration", default="1d", help="e.g. 5400, 90m, 12h, 3d (default 1d)")
    ap.add_argument("--step", default="60", help="output interval (default 60 s)")
    ap.add_argument("--out", help="CSV file for the ephemeris")
    ap.add_argument("--events", help="CSV file for the event log (t_s, utc, kind, text)")
    ap.add_argument("--propagator", choices=("cowell", "kepler", "j2mean"))
    args = ap.parse_args(argv)
    try:                                 # input the run cannot use: exit 2 saying which
        sc = Scenario.load(args.scenario)
    except (OSError, ValueError) as exc:
        ap.error(f"{args.scenario}: {exc}")
    try:
        duration, step = parse_duration(args.duration), parse_duration(args.step)
    except ValueError as exc:
        ap.error(str(exc))
    if duration < 0:
        ap.error(f"--duration cannot be negative ({args.duration})")
    if step <= 0:
        ap.error(f"--step must be positive ({args.step})")
    if args.propagator:
        sc.propagator = args.propagator
    run(sc, duration, step, args.out, events=args.events)
    for path in (args.out, args.events):
        if path:
            print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
