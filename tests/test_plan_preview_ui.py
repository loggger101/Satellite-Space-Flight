"""The maneuver preview: the plan drawn in the view while Plan maneuver is open."""

import os
import time
from datetime import datetime, timezone

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame
import pytest

from satflight.forces import ForceModel
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.ui import planview, tips
from satflight.ui.app import App

PARKING = {"type": "elements", "altitude": 185.0, "i": 28.5, "raan": 40.0, "nu": 300.0}


@pytest.fixture
def app():
    a = App(size=(1400, 850))
    sc = Scenario(name="t", epoch=datetime(2026, 9, 29, 12, tzinfo=timezone.utc),
                  forces=ForceModel(j2=True),
                  satellites=[SatSpec("Sat", PARKING, mass=4000, area=15)])
    a.sim = Simulation(sc)
    a.paused = True
    a.select(0)
    a.camera.distance = 1.6e5                      # GEO and the parking orbit both in view
    yield a
    pygame.quit()


def frame(app, n=1):
    for _ in range(n):
        for ev in pygame.event.get():
            app.handle(ev)
        app.update(1 / 30)
        app.draw()


def to_geo(app):
    app.open("maneuver")
    dlg = app.dialogs[-1]
    dlg.set("kind", "Change orbit to...")
    dlg.set("timing", "Ascending node")
    for key in ("tgt_perigee", "tgt_apogee"):
        dlg.widgets[key].value = "35786"
    dlg.widgets["tgt_inc"].value = "0"
    dlg.changed("tgt_inc")
    return dlg


def wait_aimed(app, limit=60.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        frame(app)
        plan = app.plan_preview.plan()
        if plan is not None and plan.aimed:
            return plan
        app.plan_preview._rested_at = 0.0          # no need to wait out the debounce
    raise AssertionError("the aimed plan never came")


def test_the_estimate_shows_at_once_and_the_aimed_plan_follows(app):
    dlg = to_geo(app)
    frame(app)
    est = app.plan_preview.plan()
    assert est is not None and not est.aimed and len(est.burns) == 2
    assert len(app.renderer.plan_markers) == 2
    assert "2 burns, ~4,27" in dlg.info["preview"]
    lines = [t for t, _, _ in planview.card_lines(app, app.plan_preview)]
    assert "two-body estimate" in lines[0]
    plan = wait_aimed(app)
    lines = [t for t, _, _ in planview.card_lines(app, app.plan_preview)]
    assert "aimed with the run's forces" in lines[0]
    assert any(t.startswith("To    35,78") for t in lines), lines
    assert any("92% of the tilt" in t for t in lines), lines
    assert plan.total_dv == pytest.approx(4.273, abs=0.005)


def test_schedule_uses_the_aimed_plan_already_flown(app):
    dlg = to_geo(app)
    plan = wait_aimed(app)
    dlg.submit()
    assert not app.dialogs
    assert [m.t for m in app.sim.maneuvers] == [b.t for b in plan.burns]
    assert [m.dv for m in app.sim.maneuvers] == [m.dv for m in plan.maneuvers]
    frame(app, 2)
    assert app.plan_preview is None and len(app.renderer.plan_markers) == 0


def test_an_impossible_plan_says_why_in_the_card(app):
    dlg = to_geo(app)
    dlg.set("timing", "Now")                       # at 25 deg south: no 0 deg plane there
    dlg.changed("timing")
    frame(app)
    lines = planview.card_lines(app, app.plan_preview)
    assert lines[0][0] == "CANNOT PLAN THIS MANEUVER"
    assert "cannot reach" in lines[1][0] and dlg.info["preview"].startswith("(")
    assert app.plan_preview.plan() is None


def test_the_view_still_turns_and_zooms_beside_the_docked_dialog(app):
    dlg = to_geo(app)
    frame(app)
    view = app.view_rect()
    start = (view.x + 40, view.y + 40)
    assert not dlg.rect.collidepoint(start)
    before = (app.camera.yaw, app.camera.distance)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=start, button=1))
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(start[0] + 60, start[1]),
                                  rel=(60, 0), buttons=(1, 0, 0)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(start[0] + 60, start[1]),
                                  button=1))
    app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    frame(app, 5)
    assert app.camera.yaw != before[0] and app.camera.distance != before[1]
    assert app.dialogs == [dlg]
    # inside the dialog the mouse is the dialog's, and keys stay with it
    inside = dlg.rect.center
    yaw = app.camera.yaw
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=inside, button=1))
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(inside[0] + 60, inside[1]),
                                  rel=(60, 0), buttons=(1, 0, 0)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=inside, button=1))
    frame(app, 3)
    assert app.camera.yaw == yaw


def test_burns_and_card_lines_have_tips(app):
    to_geo(app)
    wait_aimed(app)
    frame(app, 2)
    sx, sy, vis = app.renderer._project_markers(app.camera, [
        (b.r, (0, 0, 0), -4, "", False) for b in app.plan_preview.plan().burns])
    shown = [k for k in range(len(sx)) if vis[k] and app.view_rect().collidepoint(sx[k], sy[k])]
    assert shown
    for k in shown:
        reg = tips.at((int(sx[k]), int(sy[k])))
        assert reg is not None and "m/s" in reg[1] and "run's forces" in reg[1], reg
    rect, lines = planview.card(app, app.plan_preview)
    for n in range(len(lines)):
        y = rect.y + int((n + 1.5) * app.fonts.small.get_linesize())
        if y < rect.bottom:
            reg = tips.at((rect.x + 20, y))
            assert reg is not None and reg[1], n


def test_a_running_simulation_keeps_the_preview_current(app):
    to_geo(app)
    plan = wait_aimed(app)
    app.paused = False
    app.warp = 600.0
    t0 = app.sim.t
    for _ in range(12):
        frame(app)
        app.plan_preview._estimated_at = (0.0, app.plan_preview._estimated_at[1])
    assert app.sim.t > t0 + planview.PlanPreview.STALE
    now = app.plan_preview.plan()
    assert now is not plan and now.t0 > plan.t0          # flown again from the new state
    assert np.array_equal(now.start, app.sim.y[0]) or not now.aimed
