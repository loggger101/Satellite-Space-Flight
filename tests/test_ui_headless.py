"""Drive the pygame front end headlessly (SDL dummy driver) through its
dialogs, keyboard shortcuts and panels."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from satflight.constants import R_EARTH  # noqa: E402
from satflight.elements import rv2coe  # noqa: E402
from satflight.ui.app import App  # noqa: E402
from satflight.ui.widgets import TextField  # noqa: E402


@pytest.fixture
def app(tmp_path):
    a = App(size=(1400, 850))
    a.user_scenario_dir = tmp_path / "user"
    yield a
    pygame.quit()


def frame(app, n=1, dt=1 / 30):
    for _ in range(n):
        for ev in pygame.event.get():
            app.handle(ev)
        app.update(dt)
        app.draw()


def key(app, k, mod=0, text=None):
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=text or ""))


def fill(dlg, key_, value):
    w = dlg.widgets[key_]
    assert isinstance(w, TextField)
    w.value = str(value)


def test_default_scenario_renders(app):
    frame(app, 5)
    assert app.sim.n == 7
    assert app.renderer.sat_screen is not None


def test_toggles_and_views(app):
    for k in (pygame.K_e, pygame.K_m, pygame.K_g, pygame.K_v, pygame.K_x, pygame.K_r,
              pygame.K_c, pygame.K_o, pygame.K_t, pygame.K_l, pygame.K_h, pygame.K_TAB,
              pygame.K_f, pygame.K_PERIOD, pygame.K_COMMA, pygame.K_i):
        key(app, k)
        frame(app)
    assert app.opts.frame == "ECEF" and app.opts.map and app.opts.plot and app.follow
    key(app, pygame.K_i)
    frame(app, 3)


def test_add_satellite_dialog_with_preset(app):
    app.open("add")
    dlg = app.dialogs[-1]
    dlg.widgets["preset"].set("Molniya (12 h, 63.4 deg)")
    dlg._changed("preset")
    assert dlg.widgets["mode"].value == "Elements (a, e)"
    fill(dlg, "count", 3)
    dlg.submit()
    assert not app.dialogs
    assert app.sim.n == 10
    el = rv2coe(app.sim.y[-1, :3], app.sim.y[-1, 3:])
    assert el.a == pytest.approx(26554, rel=1e-9) and el.e == pytest.approx(0.72)
    frame(app, 2)


def test_add_satellite_surface_launch_and_bad_input(app):
    app.open("add")
    dlg = app.dialogs[-1]
    dlg.set("mode", "Surface launch")
    dlg.layout()
    fill(dlg, "speed", "abc")
    dlg.submit()
    assert app.dialogs and "number" in dlg.error         # stays open with a message
    fill(dlg, "speed", "3")
    dlg.submit()
    assert not app.dialogs
    r = app.sim.y[-1, :3]
    assert np.linalg.norm(r) == pytest.approx(R_EARTH, abs=25)


def test_typing_into_fields_with_events(app):
    app.open("station")
    dlg = app.dialogs[-1]
    w = dlg.widgets["name"]
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=w.rect.center))
    assert w.focused
    key(app, pygame.K_BACKSPACE, pygame.KMOD_CTRL)
    for ch in "Goonhilly":
        app.handle(pygame.event.Event(pygame.TEXTINPUT, text=ch))
    key(app, pygame.K_RETURN)
    assert not app.dialogs
    assert app.sim.stations[-1].name == "Goonhilly"


def test_walker_dialog(app):
    app.open("walker")
    dlg = app.dialogs[-1]
    fill(dlg, "total", 12)
    fill(dlg, "planes", 3)
    dlg.submit()
    assert app.sim.n == 19
    frame(app, 2)


def test_maneuver_dialog_every_kind(app):
    kinds = ["Impulse (V/N/B components)", "Hohmann transfer", "Bi-elliptic transfer",
             "Circularize", "Plane change", "Rendezvous (Lambert)", "Finite burn (thrust)"]
    for kind in kinds:
        app.select(0)
        app.open("maneuver")
        dlg = app.dialogs[-1]
        dlg.set("kind", kind)
        dlg._changed("kind")
        if kind == "Circularize":
            dlg.set("timing", "Next apoapsis")
        dlg.submit()
        assert not app.dialogs, (kind, dlg.error)
    assert len(app.sim.maneuvers) >= 8
    app.sim.advance(3 * 3600)
    assert app.sim.sats[0].dv_used > 0
    frame(app, 2)


def test_physics_dialog_switches_models(app):
    app.open("physics")
    dlg = app.dialogs[-1]
    dlg.set("drag", True)
    dlg.set("moon", True)
    dlg.set("propagator", "kepler")
    dlg.submit()
    assert app.sim.forces.drag and app.sim.forces.moon and app.sim.propagator == "kepler"
    frame(app, 3)


def test_scenario_dialog_load_save_export(app, tmp_path):
    app.open("scenario")
    dlg = app.dialogs[-1]
    dlg.set("file", "scenarios\\drag_decay.json" if os.name == "nt" else "scenarios/drag_decay.json")
    dlg.submit()
    assert app.sim.scenario.name == "Atmospheric drag decay"
    app.sim.advance(600)
    app.open("scenario")
    dlg = app.dialogs[-1]
    fill(dlg, "save_name", "snap")
    [b for b in dlg.buttons if b.text == "Save snapshot"][0].callback()
    assert (tmp_path / "user" / "snap.json").exists()
    app.open("scenario")
    [b for b in app.dialogs[-1].buttons if b.text == "Export CSV"][0].callback()
    assert list((tmp_path / "user" / "exports").glob("*.csv"))


def test_delete_edit_and_pick(app):
    frame(app, 2)
    sx, sy, vis = app.renderer.sat_screen
    k = int(np.flatnonzero(vis & np.isfinite(sx))[0])
    pos = (int(sx[k]), int(sy[k]))
    app.select(-1)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))
    assert app.selected == k
    app.open("edit")
    dlg = app.dialogs[-1]
    fill(dlg, "name", "Renamed")
    dlg.submit()
    assert app.sim.sats[k].name == "Renamed"
    n = app.sim.n
    key(app, pygame.K_DELETE)
    assert app.sim.n == n - 1
    frame(app, 2)


def test_every_bundled_scenario_draws(app):
    for p in app.scenario_files():
        app.load_scenario(p)
        app.opts.map = app.opts.plot = True
        frame(app, 2, dt=0.1)
