"""Check this installation end to end and say plainly what is wrong.

    python tools/selfcheck.py                     # everything, 6 simulated hours a scenario
    python tools/selfcheck.py --quick             # 1 hour a scenario, no rendering
    python tools/selfcheck.py default geo_drift --hours 48 --traceback

Every check prints PASS or FAIL with what it saw, and the warnings and alerts
each run wrote to its event log. A check that raises is a FAIL showing the
exception (all of it with --traceback); the others still run. The exit status
is 1 if anything failed, else 0.

What it checks:
  environment  Python, numpy, pygame-ce, the fonts in use
  assets       Earth texture and coastlines load; bundled fonts present
  files        every scenario in scenarios/ and scenarios/user/ reads strictly
  generator    scenarios/*.json are what tools/make_scenarios.py makes now
  physics      per scenario: the run finishes, its energy holds where nothing
               adds or removes any, the integrator rejects few steps, and one
               long advance equals many short ones (bit for bit)
  render       per scenario: a headless frame that is not blank
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # before pygame is imported
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from satflight.forces import ForceModel, conservative_energy  # noqa: E402
from satflight.scenario import Scenario  # noqa: E402
from satflight.simulation import Simulation  # noqa: E402

# largest relative energy change allowed per day of a run where only gravity
# acts, by propagator (measured: Cowell/dopri5 ~3e-8, j2mean ~2e-6 per day)
ENERGY_TOL = {"cowell": 1e-6, "kepler": 1e-10, "j2mean": 1e-4}
FIXED_STEP_TOL = 1e-3          # rk4 / leapfrog / wh at their configured steps
MAX_REJECTED = 0.5             # rejected / accepted adaptive steps (steep re-entries: ~0.2)


class Report:
    """Collects PASS / FAIL lines and prints them as they come."""

    def __init__(self, show_traceback: bool):
        self.failed: list[str] = []
        self.show_traceback = show_traceback

    def result(self, name: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""),
              flush=True)
        if not ok:
            self.failed.append(name)

    def check(self, name: str, fn, *args):
        """Run ``fn(*args) -> (ok, detail)``; an exception is a FAIL naming it."""
        try:
            ok, detail = fn(*args)
        except Exception as exc:         # reported as a FAIL, with where it came from
            tb = traceback.extract_tb(exc.__traceback__)[-1]
            where = f"{Path(tb.filename).name}:{tb.lineno}"
            self.result(name, False, f"{type(exc).__name__} at {where}: {exc}")
            if self.show_traceback:
                traceback.print_exception(exc)
            return None
        self.result(name, ok, detail)
        return ok


# --- checks -------------------------------------------------------------------------------

def environment():
    import pygame

    from satflight.ui import theme
    pygame.font.init()
    ce = getattr(pygame, "IS_CE", False)
    fonts = ("bundled DejaVu" if os.environ.get("SATFLIGHT_FONTS") == "bundled"
             or not pygame.font.match_font(theme.SYSTEM_FONTS["sans"]) else "Segoe UI / Consolas")
    ok = sys.version_info >= (3, 10) and bool(ce)
    return ok, (f"Python {sys.version.split()[0]}, numpy {np.__version__}, "
                f"pygame{'-ce' if ce else ' (NOT pygame-ce)'} {pygame.version.ver}, fonts {fonts}")


def assets():
    import pygame

    from satflight.ui import theme
    from satflight.ui.earth import EarthRenderer
    pygame.init()
    earth = EarthRenderer(ROOT / "assets")
    missing_fonts = [f for f in set(theme.BUNDLED_FONTS.values())
                     if not (theme.FONT_DIR / f).exists()]
    problems = [f"no {m}" for m in earth.missing] + [f"font {f} missing" for f in missing_fonts]
    return not problems, ("; ".join(problems) if problems else
                          f"texture {earth.texture.shape[0]}x{earth.texture.shape[1]}, "
                          f"{len(earth.coastlines)} coastlines")


def scenario_file(path: Path):
    sc = Scenario.load(path)
    sim = Simulation(sc)
    return True, f"{sc.name!r}: {sim.n} objects, {len(sc.launches)} launches"


def generator():
    import make_scenarios
    diffs = {name: make_scenarios.stale(name) for name in make_scenarios.ALL}
    stale = [f"{name} ({d[0]})" for name, d in diffs.items() if d]
    return not stale, (f"stale: {', '.join(stale)} (run python tools/make_scenarios.py)"
                       if stale else f"{len(make_scenarios.ALL)} scenarios match")


def _energy_model(sim) -> ForceModel:
    """The forces whose energy the run's propagator conserves."""
    return ForceModel(j2=False) if sim.propagator == "kepler" else sim.forces


def physics(path: Path, hours: float):
    """Run ``hours``; check energy, step rejections and chunked == one call."""
    sc = Scenario.load(path)
    sim = Simulation(sc)
    model = _energy_model(sim)
    n0 = sim.n
    e0 = conservative_energy(sim.y[:, :3], sim.y[:, 3:], model, sim.gmst())
    t = time.perf_counter()
    sim.advance(hours * 3600.0)
    cost = time.perf_counter() - t
    notes = [f"{hours:g} h in {cost:.1f} s"]
    ok = True
    # energy: only where nothing adds or removes any
    if not (sc.forces.drag or sc.maneuvers or sc.launches) and sim.n == n0:
        e1 = conservative_energy(sim.y[:, :3], sim.y[:, 3:], model, sim.gmst())
        drift = float(np.max(np.abs((e1 - e0) / e0)))
        fixed = sim.integrator.method != "dopri5" and sim.propagator == "cowell"
        tol = (FIXED_STEP_TOL if fixed else ENERGY_TOL[sim.propagator]) * max(1.0, hours / 24)
        ok &= drift <= tol
        notes.append(f"energy drift {drift:.1e} (limit {tol:.0e})")
    st = sim.integrator.stats
    if sim.integrator.method == "dopri5" and st.accepted:
        frac = st.rejected / st.accepted
        ok &= frac <= MAX_REJECTED
        notes.append(f"{st.rejected}/{st.accepted} steps rejected")
    # determinism: the physics keeps its own steps whatever the caller asks for
    again, total = Simulation(Scenario.load(path)), hours * 3600.0
    while again.t < total:                  # uneven chunks that end exactly on the total
        again.advance(min(total / 37, total - again.t))
    same = again.n == sim.n and np.array_equal(again.y, sim.y)
    ok &= same
    notes.append("chunked run identical" if same else "CHUNKED RUN DIFFERS")
    kinds = {}
    for e in sim.events:
        if e.kind in ("warn", "alert"):
            kinds.setdefault(e.kind, []).append(e.text)
    for kind, texts in kinds.items():
        notes.append(f"{len(texts)} {kind}: " + "; ".join(texts[:3])
                     + ("; ..." if len(texts) > 3 else ""))
    return ok, ", ".join(notes)


def render(path: Path):
    import pygame

    from satflight.ui.app import App
    app = App(path, size=(1000, 700))
    try:
        for _ in range(3):
            app.update(0.0)
            app.draw()
        spread = float(pygame.surfarray.array3d(app.screen).std())
    finally:
        pygame.quit()
    return spread > 10, f"pixel spread {spread:.0f}"


# --- driver -------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check this installation end to end.")
    ap.add_argument("scenarios", nargs="*", help="scenario names (default: all bundled)")
    ap.add_argument("--hours", type=float, default=6.0, help="simulated hours a scenario")
    ap.add_argument("--quick", action="store_true", help="1 hour a scenario, no rendering")
    ap.add_argument("--traceback", action="store_true", help="print full tracebacks")
    args = ap.parse_args(argv)
    hours = 1.0 if args.quick else args.hours
    if not hours > 0:
        ap.error("--hours must be positive")
    bundled = sorted((ROOT / "scenarios").glob("*.json"))
    if args.scenarios:
        by_name = {p.stem: p for p in bundled}
        unknown = [s for s in args.scenarios if s not in by_name]
        if unknown:
            ap.error(f"unknown scenario {', '.join(unknown)}; choose from {', '.join(by_name)}")
        chosen = [by_name[s] for s in args.scenarios]
    else:
        chosen = bundled
    user = sorted((ROOT / "scenarios" / "user").glob("*.json"))

    rep = Report(args.traceback)
    print("Environment")
    rep.check("environment", environment)
    rep.check("assets", assets)
    print("Scenario files")
    for p in bundled + user:
        rep.check(str(p.relative_to(ROOT)), scenario_file, p)
    rep.check("generator", generator)
    print(f"Physics ({hours:g} simulated hours each)")
    for p in chosen:
        rep.check(p.stem, physics, p, hours)
    if not args.quick:
        print("Rendering")
        for p in chosen:
            rep.check(p.stem, render, p)
    print()
    if rep.failed:
        print(f"{len(rep.failed)} FAILED: {', '.join(rep.failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
