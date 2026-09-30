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
from ..scenario import Scenario
from ..simulation import ACTIVE, Simulation
from . import dialogs, theme
from .camera import Camera
from .groundtrack import GroundTrackView
from .panels import GAP, LEFT_W, LOG_H, RIGHT_W, TOP_H, EventLog, InfoPanel, SatList, TopBar, draw_help
from .orbitviz import MODES as GEOMETRY_MODES
from .plots import PlotView
from .render3d import SceneRenderer

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Options:
    frame: str = "ECI"          # ECI | ECEF
    orbits: str = "auto"        # auto | all | selected | none
    geometry: str = "full"      # selected orbit's geometry overlay: full | basic | off
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


class App:
    def __init__(self, scenario=None, size=(1600, 900), root: Path = ROOT):
        pygame.init()
        self.root = Path(root)
        self.scenario_dir = self.root / "scenarios"
        self.user_scenario_dir = self.scenario_dir / "user"
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
        self.load_scenario(scenario if scenario is not None else self.scenario_dir / "default.json")

    # --- scenario management -----------------------------------------------------------
    def scenario_files(self):
        files = sorted(self.scenario_dir.glob("*.json")) + sorted(self.user_scenario_dir.glob("*.json"))
        return files

    def load_scenario(self, src):
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
        self.toast(f"Loaded '{sc.name}'")

    def reset(self):
        if self.scenario_path is not None:
            self.load_scenario(self.scenario_path)
        else:
            self.load_scenario(self.sim.scenario)

    def quick_save(self):
        name = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")
        path = self.user_scenario_dir / f"{name}.json"
        self.sim.snapshot_scenario(name).save(path)
        self.toast(f"Saved scenarios/user/{path.name}")

    # --- selection / commands -------------------------------------------------------------
    def select(self, i: int):
        self.selected = i if 0 <= i < self.sim.n else -1
        self.sim.watch = {self.selected} if self.selected >= 0 else set()
        if self.selected >= 0:
            self.satlist.ensure_visible(self.selected)

    def cycle_selection(self, step: int):
        if self.sim.n:
            self.select((self.selected + step) % self.sim.n)

    def delete_selected(self):
        if 0 <= self.selected < self.sim.n:
            self.sim.remove(self.selected)
            self.select(min(self.selected, self.sim.n - 1))

    def orbit_info(self):
        """Derived orbit properties of the selected satellite, computed once
        per simulation state (None when nothing active is selected)."""
        i = self.selected
        if not 0 <= i < self.sim.n or self.sim.sats[i].status != ACTIVE:
            return None
        s = self.sim.sats[i]
        key = (i, self.sim.t, self.sim.n, self.sim.y[i].tobytes(), id(self.sim),
               s.mass, s.area, s.cd, self.sim.forces.density_scale)
        if key != self._info_key:
            self._info = orbit_info(self.sim.y[i, :3], self.sim.y[i, 3:], self.sim.jd(), s,
                                    self.sim.forces.density_scale)
            self._info_key = key
        return self._info

    def cycle_geometry(self):
        k = GEOMETRY_MODES.index(self.opts.geometry)
        self.opts.geometry = GEOMETRY_MODES[(k + 1) % len(GEOMETRY_MODES)]
        self.toast(f"Orbit geometry: {self.opts.geometry}")

    def toggle(self, name: str):
        setattr(self.opts, name, not getattr(self.opts, name))

    def toggle_pause(self):
        self.paused = not self.paused

    def change_warp(self, step: int):
        k = min(range(len(WARPS)), key=lambda j: abs(math.log(WARPS[j]) - math.log(max(self.warp, 1))))
        self.warp = float(WARPS[max(0, min(len(WARPS) - 1, k + step))])

    def real_time(self):
        self.warp = 1.0

    def toggle_frame(self):
        self.opts.frame = "ECEF" if self.opts.frame == "ECI" else "ECI"

    def toggle_follow(self):
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
        factory = {"add": dialogs.add_satellite_dialog, "walker": dialogs.walker_dialog,
                   "maneuver": dialogs.maneuver_dialog, "station": dialogs.station_dialog,
                   "physics": dialogs.physics_dialog, "scenario": dialogs.scenario_dialog,
                   "edit": dialogs.edit_satellite_dialog}[name]
        dlg = factory(self)
        if dlg is not None:
            self.dialogs.append(dlg)
            pygame.key.start_text_input()

    def close_dialog(self, dlg):
        if dlg in self.dialogs:
            self.dialogs.remove(dlg)

    def toast(self, msg: str):
        self.toasts.append((time.monotonic() + 3.5, msg))

    def screenshot(self, path: Path | None = None):
        if path is None:
            path = self.root / "screenshots" / datetime.now().strftime("shot_%Y%m%d_%H%M%S.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        self.toast(f"Screenshot {Path(path).name}")
        return path

    # --- input -------------------------------------------------------------------------------
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
            return
        if ev.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(900, ev.w), max(600, ev.h)), pygame.RESIZABLE)
            self.camera.resize(*self.screen.get_size())
            for d in self.dialogs:
                d.layout()
            return
        if self.dialogs:
            self.dialogs[-1].handle(ev)
            return
        if self.opts.panels:
            if self.topbar.handle(ev):
                return
            if self.opts.map and self._map_rect().collidepoint(getattr(ev, "pos", (-1, -1))) \
                    and ev.type == pygame.MOUSEBUTTONDOWN:
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
            self.camera.rotate(-dx * 0.006, dy * 0.006)
            self._drag = (start, ev.pos, btn)
        elif ev.type == pygame.MOUSEBUTTONUP and self._drag:
            start, _, btn = self._drag
            self._drag = None
            if btn == 1 and abs(ev.pos[0] - start[0]) + abs(ev.pos[1] - start[1]) < 5:
                self._pick(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            self._zoom(0.87 ** ev.y)

    def _zoom(self, factor):
        self.camera.zoom(factor, 5.0 if self.follow else None)

    def _pick(self, pos):
        data = self.renderer.sat_screen
        if data is None:
            return
        sx, sy, vis = data
        d = np.hypot(sx - pos[0], sy - pos[1])
        d = np.where(np.isfinite(d), d, np.inf) + np.where(vis, 0.0, 6.0)
        k = int(np.argmin(d)) if d.size else -1
        if k >= 0 and d[k] < 14:
            self.select(k)

    def _key(self, ev):
        k, ctrl, shift = ev.key, ev.mod & pygame.KMOD_CTRL, ev.mod & pygame.KMOD_SHIFT
        if ctrl:
            actions = {pygame.K_o: lambda: self.open("scenario"), pygame.K_s: self.quick_save,
                       pygame.K_e: lambda: self.open("edit"), pygame.K_r: self.reset,
                       pygame.K_q: lambda: setattr(self, "running", False)}
            if k in actions:
                actions[k]()
            return
        simple = {
            pygame.K_SPACE: self.toggle_pause,
            pygame.K_COMMA: lambda: self.change_warp(-1),
            pygame.K_PERIOD: lambda: self.change_warp(1),
            pygame.K_1: self.real_time,
            pygame.K_f: self.toggle_follow,
            pygame.K_e: self.toggle_frame,
            pygame.K_o: lambda: setattr(self.opts, "orbits",
                                        ORBIT_MODES[(ORBIT_MODES.index(self.opts.orbits) + 1) % 4]),
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
            pygame.K_w: lambda: self.open("walker"),
            pygame.K_b: lambda: self.open("maneuver"),
            pygame.K_n: lambda: self.open("station"),
            pygame.K_p: lambda: self.open("physics"),
            pygame.K_DELETE: self.delete_selected,
            pygame.K_F12: self.screenshot,
            pygame.K_ESCAPE: lambda: setattr(self.opts, "help", False),
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

    # --- frame ---------------------------------------------------------------------------------
    def update(self, dt_real: float):
        if not self.paused:
            t0 = time.perf_counter()
            self.sim.advance(self.warp * min(dt_real, 0.1))
            self.sim_cost = time.perf_counter() - t0
        if self.follow and 0 <= self.selected < self.sim.n:
            W = self.renderer.world_rotation(self.sim, self.opts.frame)
            self.camera.target = W @ self.sim.y[self.selected, :3]
        elif self.follow:
            self.follow = False
            self.camera.target = np.zeros(3)
        self.camera.update()

    def _center_rect(self):
        w, h = self.screen.get_size()
        x0 = 2 * GAP + LEFT_W if self.opts.panels else GAP
        x1 = w - RIGHT_W - 2 * GAP if self.opts.panels else w - GAP
        return x0, x1, h

    def view_rect(self):
        """The part of the window not covered by panels."""
        x0, x1, h = self._center_rect()
        top = TOP_H + GAP if self.opts.panels else GAP
        bottom = h - LOG_H - 2 * GAP if self.opts.panels else h - GAP
        return pygame.Rect(x0, top, x1 - x0, bottom - top)

    def _map_rect(self):
        x0, x1, h = self._center_rect()
        wc = x1 - x0
        mh = int(min(wc / 2 + 26, 330))
        bottom = h - LOG_H - 2 * GAP if self.opts.panels else h - GAP
        return pygame.Rect(x0, bottom - mh, wc, mh)

    def _plot_rect(self):
        x0, x1, h = self._center_rect()
        bottom = (self._map_rect().y - GAP) if self.opts.map else \
            (h - LOG_H - 2 * GAP if self.opts.panels else h - GAP)
        return pygame.Rect(x0, bottom - 190, x1 - x0, 190)

    def draw(self):
        s = self.screen
        self.renderer.draw(s, self)
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
            self.fonts.draw(s, f"following {self.sim.sats[self.selected].name}  (F to release)",
                            (s.get_width() // 2, TOP_H + 14), theme.ACCENT, self.fonts.small, "midtop")
        for d in self.dialogs:
            d.draw(s)
        now = time.monotonic()
        self.toasts = [(t, m) for t, m in self.toasts if t > now][-4:]
        for k, (_, msg) in enumerate(self.toasts):
            r = self.fonts.render(msg, theme.TEXT, self.fonts.ui).get_rect(midtop=(s.get_width() // 2, TOP_H + 40 + k * 30))
            theme.panel(s, r.inflate(24, 10), (20, 30, 55, 230), theme.ACCENT, 6)
            s.blit(self.fonts.render(msg, theme.TEXT, self.fonts.ui), r)

    def run(self, max_frames: int | None = None, screenshot: Path | None = None):
        frames = 0
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                self.handle(ev)
            self.update(dt)
            self.draw()
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if screenshot:
            self.screenshot(Path(screenshot))
        pygame.quit()
