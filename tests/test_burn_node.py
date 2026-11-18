"""Burn node (Shift+B): shaping a burn with handles on the orbit in the view."""

import math
import os
from datetime import datetime, timezone

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame
import pytest

from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.ui import tips
from satflight.ui.app import App
from satflight.ui.burnnode import AXES, BurnNodeDialog

LEO = {"type": "elements", "altitude": 500.0, "i": 51.6, "raan": 40.0, "nu": 10.0}


@pytest.fixture
def app():
    a = App(size=(1400, 850))
    sc = Scenario(name="t", epoch=datetime(2026, 9, 29, 12, tzinfo=timezone.utc),
                  forces=ForceModel(j2=False),
                  satellites=[SatSpec("Sat", LEO, mass=1000, area=5),
                              SatSpec("Other", {**LEO, "nu": 200.0}, mass=1000, area=5)])
    a.sim = Simulation(sc)
    a.paused = True
    a.select(0)
    a.camera.distance = 4.5e4
    a.opts.geometry = "off"
    yield a
    pygame.quit()


def frame(app, n=1):
    for _ in range(n):
        for ev in pygame.event.get():
            app.handle(ev)
        app.update(1 / 30)
        app.draw()


def mouse(app, kind, pos, **kw):
    if kind == "down":
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    elif kind == "move":
        app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0),
                                      buttons=kw.get("buttons", (1, 0, 0))))
    else:
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def drag(app, start, end):
    start, end = tuple(map(int, start)), tuple(map(int, end))
    mouse(app, "down", start)
    mouse(app, "move", end)
    mouse(app, "up", end)
    frame(app)


def node(app) -> BurnNodeDialog:
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=pygame.KMOD_SHIFT,
                                  unicode="B", scancode=0))
    dlg = app.dialogs[-1]
    assert isinstance(dlg, BurnNodeDialog)
    frame(app, 2)
    return dlg


def knob(dlg, name):
    k = next(k for k, a in enumerate(AXES) if a[0] == name)
    return next((xy, u) for kk, xy, u in dlg.knobs if kk == k)


@pytest.mark.parametrize(("name", "key", "sign"), [("+V", "dv_v", 1), ("-V", "dv_v", -1),
                                                   ("+N", "dv_n", 1), ("-B", "dv_b", -1)])
def test_dragging_a_handle_out_changes_that_part_of_the_burn(app, name, key, sign):
    dlg = node(app)
    assert len(dlg.knobs) == 6 and dlg.node_xy is not None
    (x, y), u = knob(dlg, name)
    drag(app, (x, y), (x + u[0] * 100, y + u[1] * 100))
    assert float(dlg.widgets[key].value) == pytest.approx(sign * 100.0, abs=1.5)
    plan = app.plan_preview.plan()
    assert plan.burns[0].size * 1000 == pytest.approx(100.0, abs=1.5)
    others = [k for k in ("dv_v", "dv_n", "dv_b") if k != key]
    assert all(float(dlg.widgets[k].value) == 0 for k in others)


def test_prograde_raises_the_far_side_and_schedule_queues_it(app):
    dlg = node(app)
    (x, y), u = knob(dlg, "+V")
    drag(app, (x, y), (x + u[0] * 200, y + u[1] * 200))
    plan = app.plan_preview.plan()
    t = dlg.node_t
    dlg.submit()
    assert not app.dialogs and len(app.sim.maneuvers) == 1
    m = app.sim.maneuvers[0]
    assert m.t == t and m.dv[0] == pytest.approx(0.2, abs=0.002) and m.frame == "VNB"
    app.sim.advance(m.t - app.sim.t + 60)
    el = rv2coe(app.sim.y[0, :3], app.sim.y[0, 3:])
    planned = rv2coe(plan.final[:3], plan.final[3:])
    assert el.a * (1 + el.e) == pytest.approx(planned.a * (1 + planned.e), abs=0.01)
    assert el.a * (1 + el.e) - 6378.137 > 1200            # the far side went up ~740 km


def test_a_zero_burn_is_refused_and_esc_throws_the_node_away(app):
    dlg = node(app)
    dlg.submit()
    assert app.dialogs == [dlg] and "0 m/s" in dlg.error
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="",
                                  scancode=0))
    frame(app)
    assert not app.dialogs and not app.sim.maneuvers and app.plan_preview is None


def test_click_on_the_orbit_moves_the_node_and_dragging_the_node_slides_it(app):
    dlg = node(app)
    t0 = dlg.node_t
    times, sx, sy = dlg._orbit_track()
    view = app.view_rect()
    ok = [k for k in range(len(times)) if np.isfinite(sx[k]) and view.collidepoint(sx[k], sy[k])
          and not dlg.rect.collidepoint(sx[k], sy[k])
          and math.dist((sx[k], sy[k]), dlg.node_xy) > 60]
    k = ok[len(ok) // 2]
    mouse(app, "down", (int(sx[k]), int(sy[k])))
    mouse(app, "up", (int(sx[k]), int(sy[k])))
    frame(app)
    assert dlg.node_t == pytest.approx(times[k], abs=times[1] - times[0])
    assert dlg.node_t != t0
    yaw = app.camera.yaw
    j = ok[len(ok) // 4]
    drag(app, dlg.node_xy, (sx[j], sy[j]))
    assert dlg.node_t == pytest.approx(times[j], abs=2 * (times[1] - times[0]))
    assert app.camera.yaw == yaw                      # the drag moved the node, not the view
    drag(app, (view.x + 30, view.y + 30), (view.x + 90, view.y + 30))
    assert app.camera.yaw != yaw                      # beside it, the view still turns


def test_typed_numbers_drive_the_preview_and_a_passed_node_moves_a_turn_on(app):
    dlg = node(app)
    dlg.set("dv_n", "-150")
    dlg.changed("dv_n")
    frame(app)
    assert app.plan_preview.plan().burns[0].tilt == pytest.approx(math.degrees(
        math.atan2(0.15, 7.6126)), abs=0.05)
    t = dlg.node_t
    app.paused = False
    app.warp = 1000.0
    while app.sim.t < t + 5:
        frame(app)
    period = 2 * math.pi * math.sqrt(rv2coe(app.sim.y[0, :3], app.sim.y[0, 3:]).a ** 3
                                     / 398600.4418)
    assert dlg.node_t > app.sim.t and dlg.node_t - t == pytest.approx(period, rel=1e-6)


def test_every_handle_and_the_node_has_a_tip(app):
    dlg = node(app)
    for _, (x, y), _ in dlg.knobs:
        reg = tips.at((int(x), int(y)))
        assert reg is not None and "handle" in reg[1]
    reg = tips.at(tuple(map(int, dlg.node_xy)))
    assert reg is not None and "burn node" in reg[1].lower()


def test_burn_node_opens_from_the_menu_and_from_plan_maneuver(app):
    app.satellite_menu(1, (300, 300))
    item = next(i for i in app.menu.items if i.label == "Burn node...")
    item.action()
    app.menu = None
    assert isinstance(app.dialogs[-1], BurnNodeDialog) and app.dialogs[-1].sat == "Other"
    app.dialogs[-1].close()
    app.select(0)
    app.open("maneuver")
    plan = app.dialogs[-1]
    next(b for b in plan.buttons if b.text == "Burn node").callback()
    assert len(app.dialogs) == 1 and isinstance(app.dialogs[0], BurnNodeDialog)
    assert app.plan_preview.owner is app.dialogs[0]


def test_no_node_without_a_satellite_in_free_flight(app):
    app.select(-1)
    node_key = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=pygame.KMOD_SHIFT,
                                  unicode="B", scancode=0)
    app.handle(node_key)
    assert not app.dialogs and any("Select a satellite" in m for _, m in app.toasts)
