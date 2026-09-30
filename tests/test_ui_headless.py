"""Drive the pygame front end headlessly (SDL dummy driver) through its
dialogs, keyboard shortcuts and panels."""

import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame
import pytest

from satflight.constants import R_EARTH
from satflight.elements import rv2coe
from satflight.ui.app import App
from satflight.ui.panels import HELP, draw_help
from satflight.ui.widgets import TextField


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


def test_help_fits_the_smallest_window(app):
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=100, h=100))   # clamped to the minimum
    assert app.screen.get_size() == (900, 600)
    rect = draw_help(app.screen, app)
    assert app.screen.get_rect().contains(rect)
    # key labels start at x + 30 and must end before the descriptions at x + 200
    assert all(app.fonts.mono.size(k)[0] < 170 for k, desc in HELP if desc)


def test_add_satellite_dialog_with_preset(app):
    app.open("add")
    dlg = app.dialogs[-1]
    dlg.widgets["preset"].set("Molniya (12 h, 63.4 deg)")
    dlg.changed("preset")
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
        dlg.changed("kind")
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
    dlg.set("j3", True)
    dlg.set("propagator", "kepler")
    dlg.submit()
    assert app.sim.forces.drag and app.sim.forces.j3 and app.sim.propagator == "kepler"
    frame(app, 3)


def test_scenario_dialog_load_save_export(app, tmp_path):
    app.open("scenario")
    dlg = app.dialogs[-1]
    dlg.set("file", str(Path("scenarios/drag_decay.json")))      # labels use the OS separator
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


def test_separable_ray_geometry_matches_brute_force(app):
    """The optimised Earth tracer must hit exactly the pixels a direct
    per-pixel ray-sphere intersection hits, with the same normals."""
    from satflight.constants import R_EARTH
    cam = app.camera
    cam.distance, cam.yaw, cam.pitch = 15000.0, 0.7, 0.4
    cam.update()
    e = app.renderer.earth
    bbox = e._bbox(cam)
    geo = e._geometry(cam, bbox, 20000)
    nw, nh = geo["size"]
    x0, y0, bw, bh = bbox
    gx, gy = np.meshgrid(x0 + (np.arange(nw) + 0.5) * (bw / nw),
                         y0 + (np.arange(nh) + 0.5) * (bh / nh), indexing="ij")
    d = cam.ray_dirs(gx, gy)
    c = cam.position
    b = d @ c
    disc = b * b - (c @ c - R_EARTH ** 2)
    t = -b - np.sqrt(np.maximum(disc, 0))
    hit = (disc > 0) & (t > 0)
    assert np.array_equal(np.flatnonzero(hit), geo["hit_idx"])
    n_ref = (c + t.ravel()[geo["hit_idx"]][:, None] * d.reshape(-1, 3)[geo["hit_idx"]]) / R_EARTH
    assert np.allclose(geo["n"], n_ref, atol=1e-5)


def test_ocean_detection_on_synthetic_texture():
    from satflight.ui.earth import EarthRenderer
    tex = np.zeros((128, 64, 3), np.uint8)
    tex[...] = (11, 10, 50)                       # Blue Marble ocean navy
    tex[40:80, 20:44] = (70, 100, 30)             # a green continent
    out, water = EarthRenderer._prepare(tex)
    assert water[5, 5] > 0.99 and water[60, 32] < 0.01
    assert out[60, 32, 1] > out[60, 32, 2]        # land stays green
    assert out[5, 5, 2] > out[5, 5, 0]            # sea stays blue


def test_orbit_inspector_tabs_scroll_and_geometry_modes(app):
    frame(app, 2)
    assert app.info.tab == 0                       # the Orbit tab is the default
    info = app.orbit_info()
    assert info is not None and info is app.orbit_info()   # cached per state
    body = app.info.body
    assert app.info.content_h > body.h             # the property sheet scrolls
    pygame.mouse.set_pos(body.center)
    app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-5))
    frame(app)
    assert app.info.scroll[0] > 0
    tab = app.info.tab_rects[1]
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=tab.center))
    assert app.info.tab == 1
    key(app, pygame.K_q)
    assert app.info.tab == 0
    for mode in ("basic", "off", "full"):
        key(app, pygame.K_d)
        assert app.opts.geometry == mode
        frame(app)


def test_orbit_inspector_draws_every_bundled_orbit(app):
    """Every satellite of every scenario (circular, equatorial, HEO,
    suborbital, hyperbolic, re-entered) in every geometry mode."""
    for p in app.scenario_files():
        app.load_scenario(p)
        app.sim.advance(300)
        for i in range(min(app.sim.n, 8)):
            app.select(i)
            for mode in ("full", "basic"):
                app.opts.geometry = mode
                frame(app, 1, dt=0.0)


# --- launches ------------------------------------------------------------------------------

def test_launch_dialog_map_pick_preview_and_flight(app):
    app.paused = True
    key(app, pygame.K_u)
    dlg = app.dialogs[-1]
    prev = dlg.preview
    assert prev.plan is not None and prev.plan.ok          # default: Falcon 9 from the Cape
    frame(app)
    # click the map somewhere in the tropical Pacific: a custom site
    r = prev.map_rect
    pos = (r.x + int((-150 + 180) / 360 * (r.w - 1)), r.y + int((90 - 5) / 180 * (r.h - 1)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert dlg.widgets["site"].value == "Custom location"
    v = dlg.values()
    assert v["lat"] == pytest.approx(5, abs=1) and v["lon"] == pytest.approx(-150, abs=1)
    fill(dlg, "inc", 10)
    prev.refresh(dlg)
    assert prev.plan.ok and prev.plan.flight.insertion["i"] == pytest.approx(10, abs=0.05)
    frame(app)
    dlg.submit()
    assert not app.dialogs, dlg.error
    i = app.selected
    asc = app.sim.ascent_of(i)
    assert asc is not None and app.sim.sats[i].name == "Launch 1"
    assert asc.kick_deg == pytest.approx(prev.plan.kick)   # the optimised kick is flown
    app.paused = False
    for t, tab in ((30, 0), (200, 1), (200, 0)):
        app.sim.advance(t)
        app.info.tab = tab
        app.opts.map = True
        frame(app, 2, dt=0.0)
    app.sim.advance(600)
    assert app.sim.ascent_of(i) is None and app.sim.sats[i].launch_report["outcome"] == "orbit"
    app.info.tab = 1
    frame(app, 2, dt=0.0)


def test_launch_dialog_reports_impossible_launches(app):
    app.open("launch")
    dlg = app.dialogs[-1]
    dlg.set("site", "Plesetsk (Russia)")
    dlg.changed("site")
    fill(dlg, "inc", 28.5)
    dlg.preview.refresh(dlg)
    assert dlg.preview.plan is None and "cannot be reached" in dlg.preview.error
    frame(app)
    dlg.submit()
    assert app.dialogs and "cannot be reached" in dlg.error
    fill(dlg, "inc", 63)
    fill(dlg, "payload_mass", 60000)
    dlg.preview.refresh(dlg)
    assert dlg.preview.plan.flight.outcome == "short"
    frame(app)
    lines = " ".join(t for t, _ in dlg.preview._lines())
    assert "FAILS" in lines


def test_vehicle_dialog_builds_a_custom_three_stage_rocket(app):
    app.open("launch")
    dlg = app.dialogs[-1]
    [b for b in dlg.buttons if b.text == "Vehicle..."][0].callback()
    vd = app.dialogs[-1]
    assert vd is not dlg
    vd.set("count", "3")
    vd.set("edit", "3")
    vd.changed("count")
    fill(vd, "s3_thrust", 60)
    fill(vd, "s3_propellant", 4)
    fill(vd, "s3_dry", 0.5)
    vd.changed("s3_thrust")
    assert "Stage 3" in " ".join(vd.info.values())
    frame(app)
    vd.submit()
    assert app.dialogs[-1] is dlg
    assert len(dlg.preview.vehicle.stages) == 3 and dlg.widgets["vehicle"].value == "Custom"
    dlg.preview.refresh(dlg)
    assert dlg.preview.plan.ok


def test_following_a_rocket_on_the_pad_keeps_the_camera_above_ground(app):
    from satflight.launch import LaunchSpec
    sat = app.sim.launch(LaunchSpec(name="Pad", timing="delay", delay=3600, lat=62.9, lon=40.6,
                                    inclination=63, kick=2.0))
    app.select(app.sim.sats.index(sat))
    app.toggle_follow()
    cam = app.camera
    for yaw in range(0, 360, 30):
        for pitch in (-80, -30, 0, 30, 80):
            cam.yaw, cam.pitch = np.radians(yaw), np.radians(pitch)
            for d in (300, 3000, 30000):
                cam.distance = d
                app.update(0.0)
                assert np.linalg.norm(cam.position) >= cam.FLOOR - 1e-6
    frame(app, 2, dt=0.0)
    k = app.selected
    sx, sy, vis = app.renderer.sat_screen
    assert vis[k]                               # the pad vehicle is in sight, not behind the globe
