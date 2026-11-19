"""Fit the simulator's force model to a Sentinel-1 precise-orbit arc.

    python tools/fit_arc.py S1A 2025-07-05                 # one day, EGM2008 70x70, everything on
    python tools/fit_arc.py S1C 2026-01-10 --hours 6 --degree 30 --check 48

Fits the start state, Cd*A/m and Cr*A/m (``--no-cr`` holds Cr) and an impulse
at each burn the files flag (DEGRADED-MANOEUVRE), then prints the residuals
in radial / along-track / cross-track, the fitted drag and reflectivity and
each burn's delta-v. Polar motion comes out of the truth with the IERS
table (tools/fetch_eop.py); ``--no-pole`` leaves it in. With ``--check H``
it then flies the solution H more hours, estimating any burns there from the
truth with everything else held, and prints how far it strays: the force
model's own error.

Needs the files from tools/fetch_poeorb.py and tools/fetch_eop.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satflight import eop, fit, poeorb
from satflight.forces import ForceModel

# Sentinel-1's published mass (~2,164 kg for 1A at launch) and a nominal 10 m^2
# for both drag and sunlight: only the products Cd*A/m and Cr*A/m are fitted.
MASS, AREA, CD, CR = 2164.0, 10.0, 2.2, 1.3


def _rms(x):
    return np.sqrt((x ** 2).mean(0)) if len(x) else np.full(3, np.nan)


def run(mission: str, day: date, hours: float, degree: int, step: float, check: float,
        fit_cr: bool, out=print, pole: bool = True) -> dict:
    """Fit ``hours`` of ``mission`` from 00:00 UTC on ``day``; check ``check``
    hours past it. Returns the numbers it printed."""
    model = ForceModel(gravity_degree=degree, drag=True, sun=True, moon=True, srp=True)
    last = day + timedelta(days=max(0, int(np.ceil((hours + check) / 24.0)) - 1))
    arc = poeorb.load_arc(mission, day, last)
    table = eop.read() if pole else None
    obs = fit.observations(arc, step=step, hours=hours, eop=table)
    burns = [w[2] for w in fit.burn_windows(arc, hours=hours)]
    t0 = time.time()
    res = fit.fit_arc(obs, model, fit.first_state(arc, eop=table), cd_am=CD * AREA / MASS,
                      cr_am=CR * AREA / MASS, fit_cr=fit_cr, burns=burns)
    used = res.residuals_rsw[obs.use]
    out(f"{mission} {day} +{hours:g} h, {model.label()}{'' if pole else ', no pole'}, "
        f"every {step:g} s: "
        f"{res.iterations} iterations in {time.time() - t0:.0f} s")
    out(f"  fit RMS {res.rms_m:.3f} m  (R/S/W {np.round(_rms(used), 3)} m) over "
        f"{res.used} samples")
    out(f"  Cd*A/m {res.cd_am:.5f} m^2/kg (Cd {res.cd_am * MASS / AREA:.2f} at {AREA:g} m^2, "
        f"{MASS:g} kg);  Cr*A/m {res.cr_am:.5f} (Cr {res.cr_am * MASS / AREA:.2f})")
    for b in res.burns:
        out(f"  burn at +{b.t / 3600:.3f} h: R/S/W {np.round(b.rsw, 3)} mm/s")
    report = {"fit_rms_m": res.rms_m, "fit_rsw_m": _rms(used), "cd_am": res.cd_am,
              "cr_am": res.cr_am, "burns": res.burns}
    if check > 0:
        full = fit.observations(arc, step=step, hours=hours + check, eop=table)
        later = [w[2] for w in fit.burn_windows(arc, hours=hours + check)
                 if w[2] > hours * 3600.0]
        held = fit.fit_arc(full, model, res.y0, res.cd_am, res.cr_am, fit_state=False,
                           fit_cd=False, fit_cr=False, burns=[*burns, *later])
        after = (full.t > hours * 3600.0) & full.use
        e = held.residuals_rsw
        out(f"  check +{check:g} h ({len(later)} later burn(s) fitted, all else held): "
            f"RMS R/S/W {np.round(_rms(e[after]), 3)} m, "
            f"worst {np.linalg.norm(e[after], axis=1).max():.3f} m")
        for hh in (6, 24, 48, 72):
            k = np.flatnonzero(full.t >= (hours + hh) * 3600.0)
            if hh <= check and k.size:
                out(f"    at +{hh} h: R/S/W {np.round(e[k[0]], 3)} m")
        report["check_rsw_m"] = _rms(e[after])
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mission", choices=["S1A", "S1C"])
    p.add_argument("day", type=date.fromisoformat)
    p.add_argument("--hours", type=float, default=24.0, help="hours to fit (default 24)")
    p.add_argument("--degree", type=int, default=70, help="EGM2008 degree (default 70)")
    p.add_argument("--step", type=float, default=60.0, help="s between samples (default 60)")
    p.add_argument("--check", type=float, default=0.0, help="hours to fly on and compare")
    p.add_argument("--no-cr", dest="fit_cr", action="store_false", help="hold Cr at 1.3")
    p.add_argument("--no-pole", dest="pole", action="store_false",
                   help="leave polar motion in the truth (to see its size)")
    a = p.parse_args(argv)
    try:
        run(a.mission, a.day, a.hours, a.degree, a.step, a.check, a.fit_cr, pole=a.pole)
    except (poeorb.PoeorbError, eop.EopError, fit.FitError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
