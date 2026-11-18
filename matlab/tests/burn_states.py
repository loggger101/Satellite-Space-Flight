"""The simulator's state just before and just after each burn it applies at once.

    python burn_states.py scenario.json duration_s out.csv

Runs the scenario in the Python simulator for duration_s seconds and writes one row
per object whose velocity a due maneuver changed at once (impulses, circularize,
match velocity, and finite burns under the analytic propagators): name, t_s, then
the state before (x0..vz0, km and km/s) and after (x1..vz1). Finite burns integrated
under Cowell change the velocity gradually and are not listed. Used by the MATLAB
export tests to hold the live script's burns table to the simulator.
"""

import csv
import sys

from satflight.scenario import Scenario
from satflight.simulation import Simulation


def burn_states(path: str, duration: float) -> list[list]:
    sim = Simulation(Scenario.load(path))
    rows = []
    execute = sim._execute_due

    def watched():
        before, t = sim.y.copy(), sim.t
        names = [s.name for s in sim.sats]
        execute()
        for i, name in enumerate(names):
            if i < len(sim.y) and (sim.y[i, 3:] != before[i, 3:]).any():
                rows.append([name, t, *before[i], *sim.y[i]])

    sim._execute_due = watched    # advance() runs it first, for the burns due at once
    sim.advance(duration)
    return rows


def main(argv):
    path, duration, out = argv[1], float(argv[2]), argv[3]
    rows = burn_states(path, duration)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "t_s"] + [f"{c}0" for c in ("x", "y", "z", "vx", "vy", "vz")]
                   + [f"{c}1" for c in ("x", "y", "z", "vx", "vy", "vz")])
        w.writerows(rows)
    print(f"{len(rows)} burns")


if __name__ == "__main__":
    main(sys.argv)
