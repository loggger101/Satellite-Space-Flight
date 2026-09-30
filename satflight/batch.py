"""Headless batch runs: propagate a scenario and write ephemerides to CSV.

    python -m satflight.batch scenarios/hohmann_to_geo.json --duration 12h --step 60 --out run.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from .elements import rv2coe
from .frames import ecef_to_geodetic, eci_to_ecef
from .scenario import Scenario
from .simulation import Simulation
from .timeutil import format_duration, format_epoch

HEADER = ["t_s", "utc", "name", "status", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms",
          "alt_km", "lat_deg", "lon_deg", "a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]


def _rows(sim: Simulation, t: float, y: np.ndarray, names=None):
    theta = sim.clock.gmst(t)
    lat, lon, alt = ecef_to_geodetic(eci_to_ecef(y[:, :3], theta))
    el = rv2coe(y[:, :3], y[:, 3:])
    utc = format_epoch(sim.clock.datetime(t))
    for k, s in enumerate(sim.sats):
        yield [f"{t:.3f}", utc, s.name, s.status, *(f"{x:.6f}" for x in y[k]),
               f"{alt[k]:.4f}", f"{math.degrees(lat[k]):.6f}", f"{math.degrees(lon[k]):.6f}",
               f"{np.atleast_1d(el.a)[k]:.4f}", f"{np.atleast_1d(el.e)[k]:.8f}",
               *(f"{math.degrees(np.atleast_1d(x)[k]):.6f}" for x in (el.i, el.raan, el.argp, el.nu))]


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


def parse_duration(text: str) -> float:
    text = str(text).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def run(scenario: Scenario, duration: float, step: float, out=None, quiet: bool = False):
    sim = Simulation(scenario)
    writer = None
    fh = None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fh = open(out, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(HEADER)
    try:
        t_end = sim.t + duration
        while True:
            if writer:
                writer.writerows(_rows(sim, sim.t, sim.y))
            if sim.t >= t_end - 1e-9:
                break
            sim.advance(min(step, t_end - sim.t))
    finally:
        if fh:
            fh.close()
    if not quiet:
        summary(sim)
    return sim


def summary(sim: Simulation, out=sys.stdout):
    print(f"Scenario: {sim.scenario.name}   elapsed {format_duration(sim.t)}   "
          f"({sim.forces.label()}, {sim.propagator})", file=out)
    print(f"{'name':<28}{'status':<12}{'alt km':>11}{'a km':>12}{'e':>10}{'i deg':>9}"
          f"{'dV m/s':>9}", file=out)
    for i, s in enumerate(sim.sats):
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        alt = np.linalg.norm(r) - 6378.137
        print(f"{s.name[:27]:<28}{s.status:<12}{alt:>11.1f}{el.a:>12.1f}{el.e:>10.5f}"
              f"{math.degrees(el.i):>9.3f}{s.dv_used * 1000:>9.1f}", file=out)
    events = [e for e in sim.events if e.kind in ("maneuver", "alert", "warn", "launch")]
    if events:
        print("\nKey events:", file=out)
        for e in events[-40:]:
            print(f"  T+{format_duration(e.t):>13}  {e.text}", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Propagate a scenario without graphics.")
    ap.add_argument("scenario", help="scenario JSON file")
    ap.add_argument("--duration", default="1d", help="e.g. 5400, 90m, 12h, 3d (default 1d)")
    ap.add_argument("--step", default="60", help="output interval (default 60 s)")
    ap.add_argument("--out", help="CSV file for the ephemeris")
    ap.add_argument("--propagator", choices=("cowell", "kepler", "j2mean"))
    args = ap.parse_args(argv)
    sc = Scenario.load(args.scenario)
    if args.propagator:
        sc.propagator = args.propagator
    run(sc, parse_duration(args.duration), parse_duration(args.step), args.out)
    if args.out:
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
