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
from satflight.ui import glossary, theme, tips
from satflight.ui.app import App
from satflight.ui.panels import HELP, draw_help, help_layout
from satflight.ui.theme import px
from satflight.ui.widgets import TextField


@pytest.fixture
def app(tmp_path):
    a = App(size=(1400, 850))
    a.user_scenario_dir = tmp_path / "user"
    yield a
    pygame.quit()


@pytest.fixture(params=[1.0, 1.25], ids=["100%", "125%"])
def scaled_app(request, tmp_path):
    """An App at 100 % and at 125 % display scaling (window and UI both scaled)."""
    s = request.param
    a = App(size=(round(1400 * s), round(850 * s)), ui_scale=s)
    a.user_scenario_dir = tmp_path / "user"
    yield a
    pygame.quit()
    theme.set_scale(1.0)


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


def test_help_fits_the_smallest_window(scaled_app):
    app = scaled_app
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=100, h=100))   # clamped to the minimum
    assert app.screen.get_size() == (px(900), px(600))
    rect = draw_help(app.screen, app)
    assert app.screen.get_rect().contains(rect)
    # key labels end before the descriptions start, and those end inside the box
    _, key_x, desc_x = help_layout(app.fonts, *app.screen.get_size())
    assert all(key_x + app.fonts.mono.size(k)[0] < desc_x - px(8) for k, d in HELP if d)
    assert all(desc_x + app.fonts.ui.size(d)[0] < rect.w - px(8) for k, d in HELP if d)


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
    """The optimized Earth tracer must hit exactly the pixels a direct
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


def test_right_panel_scrolls_by_scrollbar_and_keys(app):
    frame(app, 2)
    panel = app.info
    track, thumb = panel.scrollbar()
    assert thumb is not None and panel.rect.contains(track)
    end = panel.content_h - panel.body.h
    # drag the thumb to the bottom of its track: the end of the tab comes into view
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=thumb.center))
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(thumb.centerx, track.bottom + 50),
                                  rel=(0, 0), buttons=(1, 0, 0)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=thumb.center))
    frame(app)
    assert panel.scroll[0] == end
    assert panel.scrollbar()[1].bottom == track.bottom
    key(app, pygame.K_HOME)
    assert panel.scroll[0] == 0
    key(app, pygame.K_PAGEDOWN)
    assert 0 < panel.scroll[0] < end
    key(app, pygame.K_END)
    frame(app)
    assert panel.scroll[0] == end
    # a click on the track above the thumb jumps up there
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(track.centerx, track.y)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(track.centerx, track.y)))
    assert panel.scroll[0] == 0


def test_geometry_overlay_names_everything_and_has_a_foldable_key(app):
    from satflight.ui import orbitviz
    app.select(app.sim.index_of("Molniya"))
    frame(app, 2)
    sel = app.selected
    ol, om, fills, legend = orbitviz.build(app.orbit_info(), app.sim.y[sel, :3],
                                           app.sim.y[sel, 3:], app.sim.sats[sel].color,
                                           np.eye(3), app.camera, "full", app.view_rect())
    names = [m[3] for m in om]
    for part in ("perigee", "apogee", "AN ascending node", "DN descending node", "RAAN",
                 "arg. of perigee", "true anomaly", "inclination", "h orbit normal",
                 "velocity", "vernal equinox", "equatorial plane"):
        assert any(part in n for n in names), part
    keyed = [row[2] for row in legend]
    for part in ("orbital plane", "line of nodes", "RAAN", "true anomaly", "inclination"):
        assert any(part in n for n in keyed), part
    # every angle is shaded as a wedge besides the two planes
    assert len(fills) >= 2 + 4
    # the key sits in the view, labels keep clear of it, and a click folds it
    box = app.renderer.legend_rect
    assert box is not None and app.view_rect().contains(box)
    assert all(not box.colliderect(r) for r in app.renderer.label_rects if r is not box)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=box.center))
    assert app.opts.legend is False
    frame(app)
    assert app.renderer.legend_rect.h < box.h
    key(app, pygame.K_d)                           # basic mode: no key
    frame(app)
    assert app.renderer.legend_rect is None


def test_window_fits_the_screen():
    from satflight.__main__ import window_size
    assert window_size((2560, 1440)) == (1600, 900)
    w, h = window_size((1536, 960))                 # 1920 x 1200 at 125 % scaling
    assert w <= 1536 - 16 and h + 34 + 48 <= 960
    assert window_size((800, 500)) == (900, 600)    # never below the panels' minimum
    # the same screen seen by a DPI-aware process: real pixels, UI at 125 %
    w, h = window_size((1920, 1200), scale=1.25)
    assert (w, h) == (1900, 1062) and h + (34 + 48) * 1.25 <= 1200
    assert window_size((800, 500), scale=1.25) == (1125, 750)


def test_fullscreen_toggles_by_key_and_button_and_comes_back(app):
    desktop = pygame.display.get_desktop_sizes()[0]
    key(app, pygame.K_F11)
    assert app.fullscreen and pygame.display.is_fullscreen()
    assert app.screen.get_size() == desktop == (app.camera.width, app.camera.height)
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=1000, h=700))
    assert app.fullscreen and app.screen.get_size() == desktop    # a resize keeps fullscreen
    frame(app, 2)
    key(app, pygame.K_h)
    key(app, pygame.K_ESCAPE)                   # Esc first closes the help ...
    assert not app.opts.help and app.fullscreen
    key(app, pygame.K_ESCAPE)                   # ... then leaves fullscreen
    assert not app.fullscreen and not pygame.display.is_fullscreen()
    assert app.screen.get_size() == (1400, 850) == (app.camera.width, app.camera.height)
    key(app, pygame.K_RETURN, pygame.KMOD_ALT)
    assert app.fullscreen
    frame(app)
    button = next(b for b in app.topbar.buttons if b.tooltip == "F11")
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=button.rect.center))
    assert not app.fullscreen and app.screen.get_size() == (1400, 850)
    key(app, pygame.K_t, pygame.KMOD_ALT)       # other Alt keys still reach their shortcut
    assert not app.opts.trails


def test_top_bar_keeps_the_clock_in_the_smallest_window(scaled_app):
    app = scaled_app
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=100, h=100))
    frame(app)
    clock = app.topbar.clock_rect               # the title gives way first if it must
    assert clock is not None and clock.right <= app.topbar.left_of_buttons - px(8)
    assert clock.w >= app.fonts.mono.size("00:00:00 UTC")[0]


def test_ui_scale_sizes_everything_in_proportion(scaled_app):
    """At 125 % the panels, rows and fonts are 1.25 times their 100 % size, and
    every button's text still fits inside it."""
    app, s = scaled_app, theme.S
    frame(app)
    assert app.satlist.rect.w == round(262 * s) and app.info.rect.w == round(340 * s)
    assert app.log.rect.h == round(132 * s)
    theme.set_scale(1.0)
    ref = theme.Fonts()                                       # the 100 % fonts
    theme.set_scale(s)
    assert app.fonts.ui.get_height() >= ref.ui.get_height() * s - 1
    app.open("launch")
    frame(app)
    for b in app._buttons() + app.dialogs[-1].buttons:
        if b.text:
            assert app.fonts.ui.size(b.text)[0] <= b.rect.w - px(4), b.text
    assert app.dialogs[-1].row_h == round(32 * s)


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
    assert asc.kick_deg == pytest.approx(prev.plan.kick)   # the optimized kick is flown
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


def test_start_screen_opens_on_request_and_covers_every_bundled_scenario(tmp_path):
    from satflight.ui.welcome import CARDS, StartScreen
    a = App(size=(1400, 850), welcome=True)
    try:
        assert isinstance(a.dialogs[-1], StartScreen)
        assert not a.toasts                              # nothing drawn over it
        frame(a, 2)
        bundled = {p.stem for p in a.scenario_dir.glob("*.json")}
        assert bundled == {stem for stem, _ in CARDS}    # a new scenario needs a blurb
        assert len(a.dialogs[-1].cards) == len(bundled)
    finally:
        pygame.quit()


def test_start_screen_fits_the_smallest_window_without_cutting_text(scaled_app):
    app = scaled_app
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=100, h=100))
    app.open("start")
    scr = app.dialogs[-1]
    assert app.screen.get_rect().contains(scr.rect)
    assert all(scr.rect.contains(r) for r in scr.card_rects + [b.rect for b in scr.buttons])
    for k in range(len(scr.cards)):
        assert not scr.blurb_lines(k)[-1].endswith("..."), scr.cards[k][1]
    frame(app)


def test_start_screen_keys_clicks_and_buttons(app):
    key(app, pygame.K_n, pygame.KMOD_CTRL)
    scr = app.dialogs[-1]
    assert scr.cards[scr.focus][0] == app.scenario_path          # starts on the loaded one
    key(app, pygame.K_RIGHT)
    key(app, pygame.K_RETURN)
    assert not app.dialogs and app.sim.scenario.name == "Launch day"
    app.open("start")
    scr = app.dialogs[-1]
    pos = scr.card_rects[2].center
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert not app.dialogs and app.scenario_path.stem == scr.cards[2][0].stem
    app.open("start")
    key(app, pygame.K_ESCAPE)                                     # keeps the scenario
    assert not app.dialogs and app.scenario_path.stem == scr.cards[2][0].stem
    app.open("start")
    key(app, pygame.K_a)                                          # build your own
    assert app.sim.n == 0 and app.dialogs[-1].title == "Add satellite"
    app.dialogs[-1].close()
    app.open("start")
    [b for b in app.dialogs[-1].buttons if b.text == "Launch a rocket"][0].callback()
    assert len(app.dialogs) == 1 and app.dialogs[0].title == "Launch"
    frame(app)


def test_start_menu_is_its_own_screen_with_nothing_running(monkeypatch):
    a = App(size=(1400, 850), welcome=True)
    try:
        assert a.in_menu and not a.started and a.sim.n == 0     # nothing loaded yet
        drawn = []
        monkeypatch.setattr(a.renderer, "draw", lambda *args: drawn.append(1))
        t = a.sim.t
        frame(a, 3)
        assert a.sim.t == t and not drawn                   # not stepped, scene not drawn
        corner = a.screen.get_at((2, 2))[:3]
        assert a.dialogs[-1]._backdrop.get_at((2, 2))[:3] == corner   # the menu covers the window
        key(a, pygame.K_ESCAPE)
        assert a.in_menu                                    # nothing to go back to
        assert "Resume" not in [b.text for b in a.dialogs[-1].buttons]
        key(a, pygame.K_h)                                  # the controls open over the menu
        assert a.opts.help and a.in_menu
        frame(a)
        key(a, pygame.K_ESCAPE)
        assert not a.opts.help and a.in_menu
        key(a, pygame.K_o, pygame.KMOD_CTRL)                # so does the scenario picker
        assert a.in_menu and len(a.dialogs) == 2
        a.dialogs[-1].close()
        assert a.in_menu
        key(a, pygame.K_RETURN)
        assert not a.in_menu and a.started and a.sim.n > 0
        frame(a, 2)
        assert a.sim.t > t and drawn
    finally:
        pygame.quit()


def test_start_menu_holds_a_running_simulation_until_resumed(app):
    frame(app, 2)
    t = app.sim.t
    key(app, pygame.K_n, pygame.KMOD_CTRL)
    assert app.in_menu and not app.toasts
    frame(app, 3)
    assert app.sim.t == t                                   # held while on the menu
    resume = [b for b in app.dialogs[-1].buttons if b.text == "Resume"][0]
    assert app.dialogs[-1].rect.contains(resume.rect)
    resume.callback()
    assert not app.in_menu
    frame(app)
    assert app.sim.t > t


# --- readability ---------------------------------------------------------------------------

def hover(app, pos):
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))


def rest(app, pos):
    """Move the mouse to ``pos`` and let it rest there; returns the tip shown
    (rect, text, key) and the tip box's rect."""
    hover(app, pos)
    frame(app, dt=0.0)
    assert app.tip_rect is None                               # only after a short rest
    tips._rest = (tips._rest[0], tips._rest[1] - 1.0)
    frame(app, dt=0.0)
    return tips.at(pos), app.tip_rect


def test_every_button_explains_itself_in_a_tooltip_inside_the_window(scaled_app):
    app = scaled_app
    for size in ((1400, 850), (900, 600)):
        app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
        frame(app)
        for b in app._buttons():
            assert b.hint and b.tooltip, b.text
            tip, box = rest(app, b.rect.center)
            assert tip[1] == b.hint and tip[2] == b.tooltip, b.text
            assert box is not None and app.screen.get_rect().contains(box), b.text
    app.open("walker")
    assert not any(b.hover for b in app._buttons())           # none left over afterward
    frame(app)
    for b in app.dialogs[-1].buttons:
        assert b.hint, b.text


def tip_states(app):
    """Draw the app in turn with every panel, tab, overlay and dialog showing."""
    app.load_scenario(app.scenario_dir / "launch_day.json")
    app.sim.advance(600)
    app.opts.map = app.opts.plot = True
    for tab in (0, 1):
        app.info.tab = tab
        for sel in range(app.sim.n):
            app.select(sel)
            frame(app, dt=0.0)
            yield f"tab {tab}, satellite {sel}"
    app.info.tab = 0
    app.load_scenario(app.scenario_dir / "default.json")
    for sel in range(app.sim.n):
        app.select(sel)
        frame(app, dt=0.0)
        yield f"orbit tab, {app.sim.sats[sel].name}"
    for name in ("add", "launch", "walker", "maneuver", "station", "physics", "scenario",
                 "edit", "start"):
        app.open(name)
        frame(app, dt=0.0)
        yield name
        if name == "launch":
            next(b for b in app.dialogs[-1].buttons if b.text == "Vehicle...").callback()
            frame(app, dt=0.0)
            yield "vehicle"
        app.dialogs.clear()


def test_every_part_of_the_window_has_a_tip_that_fits(scaled_app):
    """Points all over every panel and tab have a tip, as do every row and button
    of every dialog; every tip's box stays inside the window."""
    app = scaled_app
    w, h = app.screen.get_size()
    screen = app.screen.get_rect()
    step = px(24)
    for state in tip_states(app):
        regions = tips.regions()
        assert regions, state
        dlg = app.dialogs[-1] if app.dialogs else None
        if dlg is None:
            for area in (app.satlist.rect, app.info.rect, app.log.rect, app._map_rect(),
                         app._plot_rect(), pygame.Rect(0, 0, w, px(38))):
                pts = [(x, y) for x in range(area.x + px(6), area.right - px(6), step)
                       for y in range(area.y + px(6), area.bottom - px(6), step)]
                covered = sum(tips.at(p) is not None for p in pts)
                # only the gaps between the top bar's readouts may go without a tip
                assert covered >= 0.7 * len(pts), (state, area, covered, len(pts))
        elif hasattr(dlg, "_rows"):
            raw = dlg.raw()
            for s, r in dlg._rows:
                tip = tips.at((r.right - px(2), r.centery))
                assert tip is not None and s.tip in tip[1], (state, s.key)
                if s.kind != "info":
                    assert tips.at((dlg.rect.x + px(22), r.centery))[1] == dlg.field_tip(s, raw)
        else:                                               # the start screen
            for r in dlg.card_rects:
                assert "Click" in tips.at(r.center)[1]
        for b in getattr(dlg, "buttons", []):
            assert tips.at(b.rect.center) is not None, (state, b.text)
        for rect, text, key in regions:
            if text or key:
                box = tips.draw_tip(app.screen, app.fonts, rect, text, key, rect.center)
                assert screen.contains(box), (state, text)
                for line in tips.wrap(text, app.fonts.ui, px(tips.MAX_W)):
                    assert app.fonts.ui.size(line)[0] <= px(tips.MAX_W) or " " not in line, \
                        (state, line)


def test_every_property_row_and_form_field_is_explained(app):
    """The right panel's rows (both tabs, every bundled orbit) and every dialog
    field have their own explanation."""
    from satflight.ui.orbitpanel import property_sections
    missing = set()
    for path in sorted(app.scenario_dir.glob("*.json")):
        app.load_scenario(path)
        app.sim.advance(900)
        for i in range(app.sim.n):
            app.select(i)
            rows = [r for title, rs in app.info.report(i) if title != "GROUND CONTACT"
                    for r in rs]
            info = app.orbit_info()
            if info is not None:
                rows += [r for _, rs in property_sections(info, 0.0) for r in rs]
            missing |= {label for label, _ in rows if not glossary.row(label)}
            titles = [t for t, _ in app.info.report(i)]
            assert all(glossary.section(t) for t in titles), titles
    assert not missing, missing
    app.load_scenario(app.scenario_dir / "default.json")
    for name in ("add", "launch", "walker", "maneuver", "station", "physics", "scenario",
                 "edit"):
        app.open(name)
        dlg = app.dialogs[-1]
        assert all(s.tip for s in dlg.specs), [s.key for s in dlg.specs if not s.tip]
        for s in dlg.specs:                           # every option of a choice says what it is
            if s.option_tips:
                assert set(s.option_tips) == set(s.options), s.key
        app.dialogs.clear()


def test_tips_follow_what_is_on_top(app):
    frame(app)
    # a list row names its satellite; the 3-D view names the satellite under the mouse
    row = pygame.Rect(app.satlist.list_rect.x, app.satlist.list_rect.y, 50, px(20))
    tip, box = rest(app, row.center)
    assert app.sim.sats[0].name in tip[1] and box is not None
    sx, sy, vis = app.renderer.sat_screen
    k = next(k for k in range(app.sim.n)
             if vis[k] and app.view_rect().collidepoint(sx[k], sy[k])
             and app.renderer.legend_rect is not None
             and not app.renderer.legend_rect.collidepoint(sx[k], sy[k]))
    tip, _ = rest(app, (int(sx[k]), int(sy[k])))
    assert tip is not None and tip[1].startswith(app.sim.sats[k].name), tip
    # the Earth says where the mouse points on it
    cx, cy, _ = app.camera.project(np.zeros((1, 3)))
    r = app.camera.screen_radius(np.zeros(3), R_EARTH)
    spots = [(int(cx[0] + r * a), int(cy[0] + r * b)) for a in (-0.5, 0.0, 0.5)
             for b in (-0.5, 0.0, 0.5)]
    earth = []
    for p in spots:                     # the Earth's tip is made for where the mouse is
        hover(app, p)
        frame(app, dt=0.0)
        earth.append(tips.at(p))
    assert any(t is not None and "The Earth here" in t[1] for t in earth), earth
    hover(app, (int(cx[0]), int(cy[0]) - int(0.3 * r)))
    frame(app, dt=0.0)
    lat, lon = app.renderer.ground_at(app.camera, app.mouse)
    assert -90 <= lat <= 90 and -180 <= lon <= 180
    assert app.renderer.ground_at(app.camera, (2, app.screen.get_height() - 2)) is None
    # a known point of the globe (lat 20 N, lon 35 E) reads back where it is drawn
    la, lo = np.radians(20.0), np.radians(35.0)
    ecef = R_EARTH * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    world = app.renderer.world_to_ecef.T @ ecef
    app.camera.target, app.camera.distance = np.zeros(3), 4 * R_EARTH
    app.camera.yaw, app.camera.pitch = 0.0, 0.0
    app.camera.update()
    for yaw in np.linspace(0, 2 * np.pi, 13):
        app.camera.yaw = yaw
        app.camera.update()
        if not app.camera.hidden_by_sphere(world * 1.0001):
            break
    px_, py_, _ = app.camera.project(world[None, :])
    lat, lon = app.renderer.ground_at(app.camera, (px_[0], py_[0]))
    assert abs(lat - 20.0) < 0.5 and abs(lon - 35.0) < 0.5, (lat, lon)
    # a dialog hides every tip beneath it: nothing from the top bar shows through
    button = app.topbar.buttons[0].rect.center
    app.open("walker")
    frame(app)
    assert tips.at(button) is None or tips.at(button)[1] != app.topbar.buttons[0].hint
    dlg = app.dialogs[-1]
    field = dlg.widgets["planes"].rect
    tip, box = rest(app, field.center)
    assert "planes" in tip[1] and box is not None
    # a click hides the tip until the mouse moves again
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=field.center))
    frame(app)
    assert app.tip_rect is None


def test_a_tip_line_is_never_wider_than_the_limit(app):
    text = ("A long explanation that goes on and on to check that the tip wraps its "
            "words onto several lines rather than running off the window. " * 3)
    lines = tips.wrap(text, app.fonts.ui, px(tips.MAX_W))
    assert len(lines) > 2
    assert all(app.fonts.ui.size(line)[0] <= px(tips.MAX_W) for line in lines)
    assert tips.wrap("one\ntwo", app.fonts.ui, 500) == ["one", "two"]


def test_scene_labels_never_overlap(scaled_app):
    """Crowded scenes (a launch cluster, a constellation) keep every label legible."""
    app = scaled_app
    for name in ("launch_day", "launches_and_arcs", "gps_constellation", "default"):
        app.load_scenario(app.scenario_dir / f"{name}.json")
        app.sim.advance(1200)
        frame(app, 2, dt=0.0)
        rects = app.renderer.label_rects
        assert rects, name
        for k, r in enumerate(rects):
            assert r.collidelist(rects[k + 1:]) < 0, (name, r)


def test_long_names_are_shortened_to_fit(app):
    f = app.fonts
    assert f.fit("ISS", 200) == "ISS"
    s = f.fit("Sun-synchronous 700 km with a much longer name", 120, f.small)
    assert s.endswith("...") and f.small.size(s)[0] <= 120
    app.sim.sats[0].name = "A satellite with a name far too long for the list"
    frame(app)                                                # the list row must not overflow


def test_frame_rate_does_not_change_the_physics(app):
    """A second of real time at 64 fps or at 16 fps reaches the same state."""
    app.update(1 / 64)                                        # settle the first frame
    app.reset()
    frame(app, 64, 1 / 64)
    t, y = app.sim.t, app.sim.y.copy()
    app.reset()
    frame(app, 16, 1 / 16)
    assert app.sim.t == t and np.array_equal(app.sim.y, y)
