"""The interactive application: window, main loop, input and time control."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from ..constants import R_EARTH
from ..orbitinfo import orbit_info
from ..scenario import Scenario, orbit_state, palette_color
from ..simulation import ACTIVE, Simulation
from ..tle import split_tles
from . import dialogs, launchui, theme, tips, welcome
from .camera import Camera
from .groundtrack import GroundTrackView
from .orbitviz import MODES as GEOMETRY_MODES
from .panels import (
    GAP,
    LEFT_W,
    LOG_H,
    RIGHT_W,
    TOP_H,
    EventLog,
    InfoPanel,
    SatList,
    TopBar,
    draw_help,
)
from .plots import PlotView
from .render3d import SceneRenderer
from .theme import px

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Options:
    """View toggles; most have a keyboard shortcut (see ``App._key``)."""

    frame: str = "ECI"          # ECI | ECEF
    orbits: str = "auto"        # auto | all | selected | none
    geometry: str = "full"      # selected orbit's geometry overlay: full | basic | off
    legend: bool = True         # the overlay's key open (clicking it folds it)
    trails: bool = True
    labels: bool = True
    axes: bool = True
    vectors: bool = False
    stations: bool = True
    geo_ring: bool = False
    equator: bool = False
    footprint: bool = True
    coastlines: bool = True
    map: bool = False
    plot: bool = False
    help: bool = False
    panels: bool = True


ORBIT_MODES = ("auto", "all", "selected", "none")
WARPS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600, 7200, 21600, 86400]
# Real seconds a single frame may count for. The physics takes the same steps
# whatever the frame rate; this only stops a stall (a dragged window, a slow
# dialog) from turning into one huge jump. Below 1 / MAX_FRAME_DT fps the
# simulation runs slower than the chosen warp.
MAX_FRAME_DT = 0.25
MIN_SIZE = (900, 600)      # smallest window the panels are laid out for (design px)
IDLE_FPS = 10              # frame rate while the window is minimized (nothing is drawn)
RESTING_FPS = 20           # paused (or on the start menu) and untouched for REST_AFTER s
REST_AFTER = 1.0           # s without input before a still picture is redrawn less often
MAX_DROPPED = 3000         # most satellites a dropped TLE file adds
CURSORS = {"hand": pygame.SYSTEM_CURSOR_HAND, "text": pygame.SYSTEM_CURSOR_IBEAM,
           "move": pygame.SYSTEM_CURSOR_SIZEALL, None: pygame.SYSTEM_CURSOR_ARROW}


class App:
    """The window: owns the simulation, camera, panels and dialogs, and runs the
    event / update / draw loop. ``root`` is the repository folder holding
    ``scenarios/`` and ``assets/``. ``welcome`` opens on the start menu, which
    covers the window and holds the simulation still; with no ``scenario`` nothing
    is loaded until one is picked there. ``size`` is in screen pixels; ``ui_scale``
    is how many of them a design pixel of the UI takes (Windows' display scaling,
    see :mod:`.theme`)."""

    def __init__(self, scenario=None, size=(1600, 900), root: Path = ROOT,
                 welcome: bool = False, fullscreen: bool = False, ui_scale: float = 1.0):
        theme.set_scale(ui_scale)
        pygame.init()
        self.root = Path(root)
        self.scenario_dir = self.root / "scenarios"
        self.user_scenario_dir = self.scenario_dir / "user"
        pygame.display.set_icon(theme.app_icon())
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Satellite Space Flight")
        pygame.key.set_repeat(350, 35)
        self.fonts = theme.Fonts()
        self.clock = pygame.time.Clock()
        self.camera = Camera(*size)
        self.opts = Options()
        self.renderer = SceneRenderer(self.root / "assets")
        # vector coastlines are redundant over a real texture; K toggles them
        self.opts.coastlines = not self.renderer.earth.has_image
        self.topbar = TopBar(self)
        self.satlist = SatList(self)
        self.info = InfoPanel(self)
        self.log = EventLog(self)
        self.map = GroundTrackView(self.renderer.earth)
        self.plot = PlotView()
        self.dialogs: list = []
        self.toasts: list[tuple[float, str]] = []
        self.selected = -1
        self.follow = False
        self.paused = False
        self.warp = 60.0
        self.running = True
        self.sim_cost = 0.0
        self._drag = None
        self.scenario_path: Path | None = None
        self._info_key = None
        self._info = None
        self.mouse = None            # where the mouse is (None: outside the window)
        self._tip_hidden = False     # a click hides the tip until the mouse moves on
        self.tip_rect = None         # where this frame's hover tip went (tests read it)
        self.fullscreen = False
        self._windowed = (size, None)   # window size and position to return to from fullscreen
        self.visible = True          # False while minimized: the loop idles and draws nothing
        self._last_input = time.monotonic()   # when the last event arrived (see frame_rate)
        self.cursor = None           # mouse cursor kind shown (see tips.hot), None the arrow
        self._cursor_ok = True       # False where SDL has no system cursors (dummy driver)
        self.started = False    # a scenario was chosen: the start menu can go back to it
        if welcome and scenario is None:
            self.load_scenario(Scenario(name="Empty"))    # a placeholder the menu replaces
            self.started = False
        else:
            self.load_scenario(scenario if scenario is not None
                               else self.scenario_dir / "default.json")
        if welcome:
            self.open("start")
        if fullscreen:
            self.toggle_fullscreen()

    # --- scenario management -----------------------------------------------------------
    def scenario_files(self):
        """Bundled scenarios, then the user's saved ones."""
        return (sorted(self.scenario_dir.glob("*.json"))
                + sorted(self.user_scenario_dir.glob("*.json")))

    def load_scenario(self, src):
        """Start a new simulation from a :class:`Scenario` or a JSON path (a missing
        file gives an empty scenario), leaving the start menu if it is open."""
        self.dialogs = [d for d in self.dialogs if not isinstance(d, welcome.StartScreen)]
        if isinstance(src, Scenario):
            sc, path = src, None
        else:
            path = Path(src)
            sc = Scenario.load(path) if path.exists() else Scenario(name="Empty")
        self.sim = Simulation(sc)
        self.scenario_path = path
        self.warp = sc.warp
        self.follow = False
        self.select(0 if self.sim.n else -1)
        if self.sim.n:
            rmed = float(np.median(np.linalg.norm(self.sim.y[:, :3], axis=1)))
            self.camera.distance = float(np.clip(rmed * 2.4, 4.0 * R_EARTH, 1.2e6))
        self.camera.target = np.zeros(3)
        self.started = True
        self.toast(f"Loaded '{sc.name}'")

    def open_file(self, path):
        """Open a file dropped on the window: a scenario (``.json``) replaces the
        run; any other file is read for TLEs, whose satellites are added."""
        path = Path(path)
        try:
            if path.suffix.lower() == ".json":
                self.load_scenario(path)
                return
            tles = split_tles(path.read_text(encoding="utf-8", errors="replace"))
            if not tles:
                raise ValueError("no TLEs found")
            if self.in_menu:
                self.load_scenario(Scenario(name=path.stem))
            rows = []
            for k, t in enumerate(tles[:MAX_DROPPED]):
                r, v = orbit_state({"type": "tle", "line1": t.line1, "line2": t.line2},
                                   self.sim.clock, self.sim.t)
                rows.append((t.name, palette_color(self.sim.n + k), {}, r, v))
            self.sim.add_many(rows)
            self.toast(f"Added {len(rows)} satellite{'s' if len(rows) != 1 else ''} "
                       f"from {path.name}")
        except Exception as exc:          # a bad file must not end the session
            self.toast(f"Could not open {path.name}: {exc}")

    def reset(self):
        """Reload the current scenario from its file (or its in-memory original)."""
        if self.scenario_path is not None:
            self.load_scenario(self.scenario_path)
        else:
            self.load_scenario(self.sim.scenario)

    def quick_save(self):
        """Save the current state to ``scenarios/user/snapshot_<time>.json``."""
        name = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")
        path = self.user_scenario_dir / f"{name}.json"
        self.sim.snapshot_scenario(name).save(path)
        self.toast(f"Saved scenarios/user/{path.name}")

    # --- selection / commands -------------------------------------------------------------
    def select(self, i: int):
        """Select satellite ``i`` (-1 for none) and ask the simulation for its events."""
        self.selected = i if 0 <= i < self.sim.n else -1
        self.sim.watch = {self.selected} if self.selected >= 0 else set()
        if self.selected >= 0:
            self.satlist.ensure_visible(self.selected)

    def cycle_selection(self, step: int):
        """Select the satellite ``step`` places along the list (wrapping; key Tab)."""
        if self.sim.n:
            self.select((self.selected + step) % self.sim.n)

    def delete_selected(self):
        """Remove the selected satellite and select its neighbor (key Delete)."""
        if 0 <= self.selected < self.sim.n:
            self.sim.remove(self.selected)
            self.select(min(self.selected, self.sim.n - 1))

    def orbit_info(self):
        """Derived orbit properties of the selected satellite, computed once
        per simulation state (None when nothing active is selected)."""
        i = self.selected
        if not 0 <= i < self.sim.n or self.sim.sats[i].status != ACTIVE:
            return None
        asc = self.sim.ascent_of(i)
        if asc is not None and asc.phase == "pad":
            return None                 # a rocket on its pad has no orbit to inspect
        s = self.sim.sats[i]
        key = (i, self.sim.t, self.sim.n, self.sim.y[i].tobytes(), id(self.sim),
               s.mass, s.area, s.cd, self.sim.forces.density_scale)
        if key != self._info_key:
            self._info = orbit_info(self.sim.y[i, :3], self.sim.y[i, 3:], self.sim.jd(), s,
                                    self.sim.forces.density_scale)
            self._info_key = key
        return self._info

    def cycle_orbits(self):
        """Step through which orbits are drawn (``ORBIT_MODES``)."""
        k = ORBIT_MODES.index(self.opts.orbits)
        self.opts.orbits = ORBIT_MODES[(k + 1) % len(ORBIT_MODES)]

    def cycle_geometry(self):
        """Step through the orbit-geometry overlay modes (key D)."""
        k = GEOMETRY_MODES.index(self.opts.geometry)
        self.opts.geometry = GEOMETRY_MODES[(k + 1) % len(GEOMETRY_MODES)]
        self.toast(f"Orbit geometry: {self.opts.geometry}")

    def toggle(self, name: str):
        """Flip the boolean view option ``name`` of :class:`Options`."""
        setattr(self.opts, name, not getattr(self.opts, name))

    def toggle_fullscreen(self):
        """Switch between the window and fullscreen (key F11). Fullscreen covers the
        whole desktop at its own resolution, so the display mode never changes."""
        if self.fullscreen:
            self.fullscreen = False
            size, pos = self._windowed
            self._set_mode(size)
            if pos is not None:
                pygame.display.set_window_position(pos)
        else:
            self._windowed = (self.screen.get_size(), pygame.display.get_window_position())
            self.fullscreen = True
            self._set_mode((0, 0), pygame.FULLSCREEN)

    def _set_mode(self, size, flags=pygame.RESIZABLE):
        self.screen = pygame.display.set_mode(size, flags)
        self._resized()

    def _resized(self):
        """Fit the camera and any open dialogs to the window's new size."""
        self.camera.resize(*self.screen.get_size())
        for d in self.dialogs:
            d.layout()

    def toggle_pause(self):
        """Pause or resume the simulation (key Space)."""
        self.paused = not self.paused

    def change_warp(self, step: int):
        """Move ``step`` notches along the ``WARPS`` ladder from the nearest one."""
        now = math.log(max(self.warp, 1))
        k = min(range(len(WARPS)), key=lambda j: abs(math.log(WARPS[j]) - now))
        self.warp = float(WARPS[max(0, min(len(WARPS) - 1, k + step))])

    def real_time(self):
        """Run at 1x real time (key 1)."""
        self.warp = 1.0

    def toggle_frame(self):
        """Switch the view between inertial (ECI) and Earth-fixed (ECEF) axes (key E)."""
        self.opts.frame = "ECEF" if self.opts.frame == "ECI" else "ECI"

    def toggle_follow(self):
        """Toggle the camera following the selected satellite."""
        if not self.follow and not 0 <= self.selected < self.sim.n:
            self.toast("Select a satellite to follow")
            return
        self.follow = not self.follow
        if self.follow:
            self.camera.distance = min(self.camera.distance, 4000.0)
        else:
            self.camera.target = np.zeros(3)
            self.camera.distance = max(self.camera.distance, 3.2 * R_EARTH)

    def open(self, name: str):
        """Open the dialog called ``name`` (see the factory table)."""
        factory = {"add": dialogs.add_satellite_dialog, "walker": dialogs.walker_dialog,
                   "maneuver": dialogs.maneuver_dialog, "station": dialogs.station_dialog,
                   "physics": dialogs.physics_dialog, "scenario": dialogs.scenario_dialog,
                   "edit": dialogs.edit_satellite_dialog, "launch": launchui.launch_dialog,
                   "start": welcome.StartScreen}[name]
        dlg = factory(self)
        if dlg is not None:
            self._clear_hover()                 # no stale tooltip once the dialog closes
            self.dialogs.append(dlg)
            if name == "start":
                self.toasts.clear()             # "Loaded ..." would sit on top of the menu
            else:                               # the start menu has no text fields
                pygame.key.start_text_input()

    @property
    def in_menu(self) -> bool:
        """The start menu is open: it fills the window and the simulation stands still."""
        return any(isinstance(d, welcome.StartScreen) for d in self.dialogs)

    def close_dialog(self, dlg):
        """Remove ``dlg`` from the dialog stack (dialogs call this themselves)."""
        if dlg in self.dialogs:
            self.dialogs.remove(dlg)

    def toast(self, msg: str):
        """Show ``msg`` briefly at the top of the view."""
        self.toasts.append((time.monotonic() + 3.5, msg))

    def screenshot(self, path: Path | None = None):
        """Save the window to ``path`` (default ``screenshots/shot_<time>.png``)."""
        if path is None:
            path = self.root / "screenshots" / datetime.now().strftime("shot_%Y%m%d_%H%M%S.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        self.toast(f"Screenshot {Path(path).name}")
        return path

    # --- input -------------------------------------------------------------------------------
    def handle(self, ev):
        """Route an event: dialogs first, then panels, then the 3-D view."""
        self._last_input = time.monotonic()
        if ev.type == pygame.QUIT:
            self.running = False
            return
        if ev.type == pygame.WINDOWLEAVE:
            self._clear_hover()
            self.mouse = None
            return
        if ev.type == pygame.DROPFILE:
            self.open_file(ev.file)
            return
        if ev.type in (pygame.WINDOWMINIMIZED, pygame.WINDOWHIDDEN):
            self.visible = False
            return
        if ev.type in (pygame.WINDOWRESTORED, pygame.WINDOWMAXIMIZED, pygame.WINDOWSHOWN):
            self.visible = True
            return
        if ev.type == pygame.MOUSEMOTION:
            self.mouse, self._tip_hidden = ev.pos, any(ev.buttons)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            self.mouse, self._tip_hidden = ev.pos, True
        if ev.type == pygame.VIDEORESIZE:
            if self.fullscreen:                 # the screen's size, not the user's: keep it
                self._resized()
            else:
                self._set_mode((max(px(MIN_SIZE[0]), ev.w), max(px(MIN_SIZE[1]), ev.h)))
            return
        if self.dialogs:
            self.dialogs[-1].handle(ev)
            return
        key = self.renderer.legend_rect
        if (key is not None and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and key.collidepoint(ev.pos)):
            self.opts.legend = not self.opts.legend
            return
        if self.opts.panels:
            if self.topbar.handle(ev):
                return
            if (self.opts.map and ev.type == pygame.MOUSEBUTTONDOWN
                    and self._map_rect().collidepoint(ev.pos)):
                return
            if self.opts.plot and self.plot.handle(ev):
                return
            if self.satlist.handle(ev) or self.info.handle(ev):
                return
        if ev.type == pygame.KEYDOWN:
            self._key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
            self._drag = (ev.pos, ev.pos, ev.button)
        elif ev.type == pygame.MOUSEMOTION and self._drag and any(ev.buttons):
            start, last, btn = self._drag
            dx, dy = ev.pos[0] - last[0], ev.pos[1] - last[1]
            turn = 0.006 / theme.S                  # radians per screen px
            self.camera.rotate(-dx * turn, dy * turn)
            self._drag = (start, ev.pos, btn)
        elif ev.type == pygame.MOUSEBUTTONUP and self._drag:
            start, _, btn = self._drag
            self._drag = None
            if btn == 1 and abs(ev.pos[0] - start[0]) + abs(ev.pos[1] - start[1]) < px(5):
                self._pick(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            self._zoom(0.87 ** ev.y)

    def _buttons(self):
        """Every button of the top bar and the left panel."""
        return self.topbar.buttons + self.satlist.buttons

    def _clear_hover(self):
        """Forget which button the mouse is over (it moved away without a motion event)."""
        for b in self._buttons():
            b.hover = False
        tips.reset()

    def _scene_tips(self):
        """Hover tips for the 3-D view: the satellite nearest the mouse, else the
        Earth under it (with the latitude and longitude there)."""
        m, view = self.mouse, self.view_rect()
        if m is None or not view.collidepoint(m) or tips.at(m) is not None:
            return                                  # over the key, the callout or the Sun
        k, d = -1, np.zeros(0)
        if self.renderer.sat_screen is not None:
            sx, sy, vis = self.renderer.sat_screen
            d = np.hypot(sx - m[0], sy - m[1])
            d = np.where(np.isfinite(d) & vis, d, np.inf)
            k = int(np.argmin(d)) if d.size else -1
        if k >= 0 and d[k] < px(10):
            s = self.sim.sats[k]
            rect = pygame.Rect(0, 0, px(20), px(20))
            rect.center = (int(sx[k]), int(sy[k]))
            if s.status == ACTIVE:
                alt = float(np.linalg.norm(self.sim.y[k, :3])) - R_EARTH
                state = f"{alt:,.0f} km up, {'in sunlight' if s.shadow > 0.5 else 'in shadow'}"
            else:
                state = s.status
            what = "Selected - F makes the camera follow it." if k == self.selected else \
                "Click to select it."
            tips.add(rect, f"{s.name}: {state}.\n{what}")
            return
        ground = self.renderer.ground_at(self.camera, m)
        if ground is not None:
            lat, lon = ground
            r = self.camera.screen_radius(np.zeros(3), R_EARTH)
            cx, cy, _ = self.camera.project(np.zeros((1, 3)))
            disc = pygame.Rect(0, 0, int(2 * r), int(2 * r))
            disc.center = (int(cx[0]), int(cy[0]))
            ns, ew = "N" if lat >= 0 else "S", "E" if lon >= 0 else "W"
            tips.add(pygame.Rect(m[0] - px(3), m[1] - px(3), px(6), px(6)).clip(disc),
                     f"The Earth here: {abs(lat):.1f}\N{DEGREE SIGN} {ns}, "
                     f"{abs(lon):.1f}\N{DEGREE SIGN} {ew}. The night side is dark.\n"
                     "Drag to turn the view, scroll to zoom, click a satellite to select it.")

    def _zoom(self, factor):
        self.camera.zoom(factor, 5.0 if self.follow else None)

    def _pick(self, pos):
        """Select the satellite drawn nearest the click, if within 14 design px."""
        data = self.renderer.sat_screen
        if data is None:
            return
        sx, sy, vis = data
        d = np.hypot(sx - pos[0], sy - pos[1])
        d = np.where(np.isfinite(d), d, np.inf) + np.where(vis, 0.0, px(6))
        k = int(np.argmin(d)) if d.size else -1
        if k >= 0 and d[k] < px(14):
            self.select(k)

    def _key(self, ev):
        """Keyboard shortcuts of the main view."""
        k, ctrl, shift = ev.key, ev.mod & pygame.KMOD_CTRL, ev.mod & pygame.KMOD_SHIFT
        if ctrl:
            actions = {pygame.K_o: lambda: self.open("scenario"), pygame.K_s: self.quick_save,
                       pygame.K_n: lambda: self.open("start"),
                       pygame.K_e: lambda: self.open("edit"), pygame.K_r: self.reset,
                       pygame.K_q: lambda: setattr(self, "running", False)}
            if k in actions:
                actions[k]()
            return
        if ev.mod & pygame.KMOD_ALT and k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.toggle_fullscreen()
            return
        simple = {
            pygame.K_SPACE: self.toggle_pause,
            pygame.K_COMMA: lambda: self.change_warp(-1),
            pygame.K_PERIOD: lambda: self.change_warp(1),
            pygame.K_1: self.real_time,
            pygame.K_f: self.toggle_follow,
            pygame.K_e: self.toggle_frame,
            pygame.K_o: self.cycle_orbits,
            pygame.K_t: lambda: self.toggle("trails"),
            pygame.K_l: lambda: self.toggle("labels"),
            pygame.K_v: lambda: self.toggle("vectors"),
            pygame.K_x: lambda: self.toggle("axes"),
            pygame.K_c: lambda: self.toggle("footprint"),
            pygame.K_r: lambda: self.toggle("geo_ring"),
            pygame.K_m: lambda: self.toggle("map"),
            pygame.K_g: lambda: self.toggle("plot"),
            pygame.K_i: lambda: self.toggle("panels"),
            pygame.K_k: lambda: self.toggle("coastlines"),
            pygame.K_d: self.cycle_geometry,
            pygame.K_q: self.info.cycle_tab,
            pygame.K_h: lambda: self.toggle("help"),
            pygame.K_F1: lambda: self.toggle("help"),
            pygame.K_a: lambda: self.open("add"),
            pygame.K_u: lambda: self.open("launch"),
            pygame.K_w: lambda: self.open("walker"),
            pygame.K_b: lambda: self.open("maneuver"),
            pygame.K_n: lambda: self.open("station"),
            pygame.K_p: lambda: self.open("physics"),
            pygame.K_DELETE: self.delete_selected,
            pygame.K_F11: self.toggle_fullscreen,
            pygame.K_F12: self.screenshot,
            pygame.K_ESCAPE: self._escape,
            pygame.K_TAB: lambda: self.cycle_selection(-1 if shift else 1),
            pygame.K_EQUALS: lambda: self._zoom(0.8),
            pygame.K_KP_PLUS: lambda: self._zoom(0.8),
            pygame.K_MINUS: lambda: self._zoom(1.25),
            pygame.K_KP_MINUS: lambda: self._zoom(1.25),
            pygame.K_LEFT: lambda: self.camera.rotate(0.08, 0),
            pygame.K_RIGHT: lambda: self.camera.rotate(-0.08, 0),
            pygame.K_UP: lambda: self.camera.rotate(0, 0.06),
            pygame.K_DOWN: lambda: self.camera.rotate(0, -0.06),
        }
        if k in simple:
            simple[k]()

    def _escape(self):
        """Esc closes the help; with no help open it leaves fullscreen."""
        if self.opts.help:
            self.opts.help = False
        elif self.fullscreen:
            self.toggle_fullscreen()

    # --- frame ---------------------------------------------------------------------------------
    def update(self, dt_real: float):
        """Advance the simulation by ``warp`` times the real frame time and move the camera.
        The frame only sets how far; the physics keeps its own time steps. Nothing
        moves while the start menu is open."""
        if self.in_menu:
            return
        if not self.paused:
            t0 = time.perf_counter()
            self.sim.advance(self.warp * min(dt_real, MAX_FRAME_DT))
            self.sim_cost = time.perf_counter() - t0
        if self.follow and 0 <= self.selected < self.sim.n:
            W = self.renderer.world_rotation(self.sim, self.opts.frame)
            self.camera.target = W @ self.sim.y[self.selected, :3]
        elif self.follow:
            self.follow = False
            self.camera.target = np.zeros(3)
        self.camera.update()

    def center_span(self):
        """(left, right, window height) of the area between the side panels."""
        w, h = self.screen.get_size()
        gap = px(GAP)
        x0 = 2 * gap + px(LEFT_W) if self.opts.panels else gap
        x1 = w - px(RIGHT_W) - 2 * gap if self.opts.panels else w - gap
        return x0, x1, h

    def view_rect(self):
        """The part of the window not covered by panels."""
        x0, x1, _ = self.center_span()
        top = px(TOP_H) + px(GAP) if self.opts.panels else px(GAP)
        return pygame.Rect(x0, top, x1 - x0, self._bottom() - top)

    def _bottom(self) -> int:
        """Lowest y of the view: the top of the event log, or the window edge."""
        h = self.screen.get_height()
        return h - px(LOG_H) - 2 * px(GAP) if self.opts.panels else h - px(GAP)

    def _map_rect(self):
        """Where the ground-track map sits: along the bottom of the view."""
        x0, x1, _ = self.center_span()
        mh = int(min((x1 - x0) / 2 + px(26), px(330)))
        return pygame.Rect(x0, self._bottom() - mh, x1 - x0, mh)

    def _plot_rect(self):
        """Where the plot sits: above the map, or along the bottom of the view."""
        x0, x1, _ = self.center_span()
        bottom = self._map_rect().y - px(GAP) if self.opts.map else self._bottom()
        return pygame.Rect(x0, bottom - px(190), x1 - x0, px(190))

    def draw(self):
        """Draw one frame: 3-D scene, panels, map/plot, help, dialogs and toasts;
        or, on the start menu, only the menu and what is open over it."""
        s = self.screen
        tips.begin()
        if self.in_menu:
            self._draw_menu(s)
            return
        self.renderer.draw(s, self)
        if self.renderer.legend_rect is not None:
            tips.hot(self.renderer.legend_rect)          # a click folds the key
        self._scene_tips()
        if self.opts.panels:
            self.topbar.draw(s)
            self.satlist.draw(s)
            self.info.draw(s)
            self.log.draw(s)
        if self.opts.map:
            self.map.draw(s, self._map_rect(), self)
        if self.opts.plot:
            self.plot.draw(s, self._plot_rect(), self)
        if self.opts.help:
            draw_help(s, self)
        if self.follow and 0 <= self.selected < self.sim.n:
            r = self.fonts.draw(s, f"following {self.sim.sats[self.selected].name}  "
                                   "(F to release)", (s.get_width() // 2, px(TOP_H) + px(14)),
                                theme.ACCENT, self.fonts.small, "midtop")
            tips.add(r, "The camera moves with the selected satellite.", "F")
        for d in self.dialogs:
            d.draw(s)
        self._draw_overlays(s)

    def _draw_menu(self, s):
        """The start menu paints the whole window; dialogs it opened and the
        controls sit on top. The scene and panels are not drawn at all."""
        for d in self.dialogs:
            d.draw(s)
        if self.opts.help:
            draw_help(s, self)
        self._draw_overlays(s)

    def _draw_overlays(self, s):
        """Toasts, then the hover tip."""
        now = time.monotonic()
        self.toasts = [(t, m) for t, m in self.toasts if t > now][-4:]
        for k, (_, msg) in enumerate(self.toasts):
            txt = self.fonts.render(msg, theme.TEXT, self.fonts.ui)
            r = txt.get_rect(midtop=(s.get_width() // 2, px(TOP_H) + px(40) + k * px(30)))
            theme.panel(s, r.inflate(px(24), px(10)), (20, 30, 55, 230), theme.ACCENT, 6)
            s.blit(txt, r)
        busy = self._tip_hidden or self._drag is not None
        self.tip_rect = None if busy else tips.draw(s, self.fonts, self.mouse)
        self._update_cursor()

    def _update_cursor(self):
        """A hand over what can be clicked, an I-beam over text fields, the move
        cursor while the view is dragged, else the arrow."""
        kind = "move" if self._drag is not None else tips.cursor_at(self.mouse)
        if kind == self.cursor:
            return
        self.cursor = kind
        if self._cursor_ok:
            try:
                pygame.mouse.set_cursor(CURSORS[kind])
            except pygame.error:          # no system cursors here (the dummy video driver)
                self._cursor_ok = False

    def frame_rate(self, now: float | None = None) -> int:
        """Frames per second to aim for: 60 while anything can change; fewer when
        nothing moves (paused or on the start menu, no input for ``REST_AFTER``
        s), so a still picture does not keep a CPU core busy; ``IDLE_FPS`` while
        minimized. Any event brings 60 back on the next frame."""
        if not self.visible:
            return IDLE_FPS
        now = time.monotonic() if now is None else now
        still = (self.paused or self.in_menu) and self._drag is None
        if still and now - self._last_input > REST_AFTER:
            return RESTING_FPS
        return 60

    def run(self, max_frames: int | None = None, screenshot: Path | None = None):
        """Main loop at up to 60 fps; optionally stop after ``max_frames`` and save a screenshot."""
        frames = 0
        while self.running:
            dt = self.clock.tick(self.frame_rate()) / 1000.0
            for ev in pygame.event.get():
                self.handle(ev)
            self.update(dt)
            if self.visible:                # minimized: time runs on, nothing is drawn
                self.draw()
                pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if screenshot:
            self.screenshot(Path(screenshot))
        pygame.quit()
