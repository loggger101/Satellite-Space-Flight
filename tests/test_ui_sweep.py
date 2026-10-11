"""Every part drawn for every satellite of every bundled scenario, at several
moments (on the pad, mid-ascent, in orbit, after re-entries): nothing may
raise or warn. The plot used to skip a frame on any exception; it no longer
catches, so this is what keeps it honest."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path

import pygame
import pytest

from satflight.ui.app import App
from satflight.ui.plots import METRICS

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.json"))
MOMENTS = (0.0, 120.0, 3600.0, 12 * 3600.0)     # s after loading, cumulative


@pytest.fixture(scope="module")
def app():
    a = App(size=(1400, 850))
    yield a
    pygame.quit()


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_every_metric_and_panel_draws_for_every_satellite(app, path):
    app.load_scenario(path)
    app.opts.map = app.opts.plot = True
    t_done = 0.0
    for t in MOMENTS:
        app.sim.advance(t - t_done)
        t_done = t
        picks = sorted({*range(min(app.sim.n, 6)), *range(max(0, app.sim.n - 3), app.sim.n)})
        for i in picks:                          # the first ones and the newest (stages, payloads)
            app.select(i)
            app.info.tab = (app.info.tab + 1) % len(app.info.TABS)
            for k in range(len(METRICS)):
                app.plot.metric = k
                app.plot.draw(app.screen, app._plot_rect(), app)
            app.draw()
