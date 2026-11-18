"""Burn node: shape a burn by dragging handles on the orbit in the 3-D view
(key Shift+B).

A node sits on the selected satellite's orbit, where the burn will fire.
Six handles stick out of it along the burn axes (prograde / retrograde,
normal / anti-normal, radial out / in, the VNB frame); dragging one along
its line changes that part of the delta-v, and the orbit the burn makes is
redrawn at once (the maneuver preview, ui.planview). Dragging the node
slides it along the orbit, and a click on the orbit puts it there. The
docked form beside the view shows the same numbers and takes typed ones;
Enter (Schedule) queues the burn, Esc throws it away.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..constants import MU_EARTH
from ..elements import kepler_propagate, rv2coe
from ..frames import vnb_basis
from ..maneuvers import Maneuver, resolve_time
from ..simulation import ACTIVE
from ..timeutil import format_duration
from . import theme, tips
from .planview import PlanPreview
from .theme import px
from .widgets import FieldSpec as F
from .widgets import FormDialog

HANDLE_LEN = 70            # design px from the node to a handle's knob
GRAB = 11                  # design px: how near a knob or the node a press must be
ORBIT_PICK = 9             # design px: how near the orbit a click must be
ORBIT_SAMPLES = 360
OPEN_SPAN = 6 * 3600.0     # s of an open orbit offered to the node
RATE = 1.0                 # m/s per design px dragged (Shift 0.1, Ctrl 10)
# (name, VNB axis, sign, colour, hover text)
AXES = [
    ("+V", 0, 1, (255, 214, 90), "Prograde: faster along the motion. Raises the far side of "
                                 "the orbit."),
    ("-V", 0, -1, (200, 160, 60), "Retrograde: slower. Lowers the far side of the orbit."),
    ("+N", 1, 1, (200, 130, 255), "Normal: along the orbit's axis. Tilts the plane; at a node "
                                  "it changes the inclination."),
    ("-N", 1, -1, (150, 100, 200), "Anti-normal: tilts the plane the other way."),
    ("+B", 2, 1, (100, 220, 240), "Radial out (B = V x N): turns the velocity away from the "
                                  "Earth. Moves where the apsides are rather than their size."),
    ("-B", 2, -1, (70, 160, 180), "Radial in: turns the velocity toward the Earth."),
]
KEYS = ("dv_v", "dv_n", "dv_b")
DV_TIP = ("Velocity change along this axis in m/s, negative for the opposite way. Dragging "
          "the matching handle in the view changes it (Shift: fine, Ctrl: coarse).")


class BurnNodeDialog(FormDialog):
    """The docked form plus the node and handles drawn in the view."""

    def __init__(self, app, i: int):
        sim = app.sim
        self.sat = sim.sats[i].name
        specs = [
            F("dv_v", "Prograde V (m/s)", "float", "0", tip=DV_TIP),
            F("dv_n", "Normal N (m/s)", "float", "0", tip=DV_TIP),
            F("dv_b", "Radial out B (m/s)", "float", "0", tip=DV_TIP),
            F("isp", "Specific impulse Isp (s)", "float", "320",
              tip="Engine efficiency: the higher it is, the less propellant the burn uses."),
            F("where", "", "info", "",
              tip="When and where the node fires. Drag the node along the orbit, or click "
                  "the orbit, to move it."),
        ]
        super().__init__(app, f"Burn node - {self.sat}", specs, self._schedule, "Schedule",
                         width=420, on_change=self._changed, dock="right",
                         subtitle="Drag the handles in the view; click the orbit to move it",
                         ok_hint="Queue this burn; the event log shows it when it fires")
        self.preview = app.plan_preview = PlanPreview(app, self)
        self.node_t = self._first_time()
        self.drag = None               # ("axis", k, start pos, start value) or ("slide",)
        self.knobs: list = []          # (k, (x, y), (ux, uy)) of the handles drawn last
        self.node_xy = None
        self._press = None             # where a press beside the dialog went down
        self._changed(self, None)

    # --- the plan -----------------------------------------------------------------------
    def _state(self):
        sim = self.app.sim
        i = sim.require(self.sat)
        return sim, i, sim.y[i]

    def _first_time(self) -> float:
        """The next apogee of an eccentric orbit, else an eighth of a turn ahead."""
        sim, _, y = self._state()
        el = rv2coe(y[:3], y[3:])
        if 0.01 < el.e < 1:
            return resolve_time(Maneuver(self.sat, timing="apoapsis"), sim.t, y[:3], y[3:])
        return sim.t + self._period() / 8

    def _period(self) -> float:
        _, _, y = self._state()
        el = rv2coe(y[:3], y[3:])
        return 2 * math.pi * math.sqrt(el.a ** 3 / MU_EARTH) if el.e < 1 else OPEN_SPAN

    def burn(self, v=None) -> Maneuver:
        """The burn the node stands for, from form values ``v`` (ValueError if bad)."""
        v = v if v is not None else self.values()
        m = Maneuver(self.sat, "impulse", timing="absolute", t=self.node_t,
                     dv=tuple(v[k] / 1000.0 for k in KEYS), frame="VNB", isp=v["isp"])
        m.label = f"Burn node ({math.hypot(*m.dv) * 1000:,.1f} m/s)"
        m.check()
        return m

    def _changed(self, dlg, key):
        try:
            m = self.burn()
        except ValueError as exc:
            self.preview.clear(str(exc))
            return
        sat = self.sat
        self.preview.show((m.t, m.dv, m.isp), lambda run, aim: (sat, [m], ""),
                          self.app.sim.sats[self.app.sim.require(sat)].color)

    def _schedule(self, v):
        m = self.burn(v)
        if not any(m.dv):
            return "Drag a handle (or type a number) first: the burn is 0 m/s"
        self.app.sim.schedule(m)
        return None

    def _keep_ahead(self):
        """A node the run has passed moves on a whole turn, to the same place."""
        sim = self.app.sim
        if self.node_t > sim.t + 1.0:
            return
        period = self._period()
        self.node_t += period * math.ceil((sim.t + 1.0 - self.node_t) / period)
        self._changed(self, None)

    # --- the view --------------------------------------------------------------------------
    def _world(self):
        return self.app.renderer.world_rotation(self.app.sim, self.app.opts.frame)

    def _orbit_track(self):
        """(times, screen x, screen y) of the coming turn of the orbit."""
        sim, _, y = self._state()
        times = sim.t + np.linspace(0.0, self._period(), ORBIT_SAMPLES + 1)[1:]
        r, _ = kepler_propagate(np.repeat(y[None, :3], len(times), 0),
                                np.repeat(y[None, 3:], len(times), 0), times - sim.t)
        sx, sy, _ = self.app.camera.project(r @ self._world().T)
        return times, sx, sy

    def _orbit_time_at(self, pos):
        """Time of the orbit point drawn nearest ``pos``, if within ORBIT_PICK."""
        times, sx, sy = self._orbit_track()
        d = np.hypot(sx - pos[0], sy - pos[1])
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        return float(times[k]) if d[k] <= px(ORBIT_PICK) else None

    def _slide_to(self, pos):
        times, sx, sy = self._orbit_track()
        d = np.where(np.isfinite(sx), np.hypot(sx - pos[0], sy - pos[1]), np.inf)
        self.node_t = float(times[int(np.argmin(d))])
        self._changed(self, None)

    def _knob_at(self, pos):
        for k, (x, y), _ in self.knobs:
            if math.hypot(x - pos[0], y - pos[1]) <= px(GRAB):
                return k
        return None

    def handle(self, ev) -> bool:
        """Presses on a knob or the node drag it; a click on the orbit moves the
        node there; everything else is the form's (or, beside it, the view's)."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self._beside(ev):
            k = self._knob_at(ev.pos)
            if k is not None:
                axis = AXES[k][1]
                self.drag = ("axis", k, ev.pos, self._value(KEYS[axis]))
                return True
            if self.node_xy and math.dist(self.node_xy, ev.pos) <= px(GRAB):
                self.drag = ("slide",)
                return True
            self._press = ev.pos
            return False
        if ev.type == pygame.MOUSEMOTION and self.drag is not None:
            self._drag_to(ev.pos, pygame.key.get_mods())
            return True
        if ev.type == pygame.MOUSEBUTTONUP and self.drag is not None:
            self.drag = None
            return True
        if ev.type == pygame.MOUSEBUTTONUP and self._press is not None:
            press, self._press = self._press, None
            if math.dist(press, ev.pos) < px(5):
                t = self._orbit_time_at(ev.pos)
                if t is not None:
                    self.node_t = t
                    self._changed(self, None)
            return False
        return super().handle(ev)

    def _value(self, key) -> float:
        try:
            return float(self.widgets[key].value)
        except ValueError:
            return 0.0

    def _drag_to(self, pos, mods):
        if self.drag[0] == "slide":
            self._slide_to(pos)
            return
        _, k, start, value = self.drag
        _, axis, sign, _, _ = AXES[k]
        ux, uy = next(u for kk, _, u in self.knobs if kk == k)
        outward = ((pos[0] - start[0]) * ux + (pos[1] - start[1]) * uy) / theme.S
        rate = RATE * (0.1 if mods & pygame.KMOD_SHIFT else 10.0 if mods & pygame.KMOD_CTRL
                       else 1.0)
        self.set(KEYS[axis], f"{value + sign * outward * rate:.1f}")
        self.changed(KEYS[axis])

    def draw(self, surf):
        """The node and its handles in the view, then the form."""
        self._keep_ahead()
        plan = self.preview.plan()
        self.knobs, self.node_xy = [], None
        if plan is not None and plan.burns:
            self._draw_node(surf, plan.burns[0])
            b = plan.burns[0]
            when = b.t - self.app.sim.t
            self.info["where"] = (f"Fires in {format_duration(when)}, {b.place}; "
                                  f"{b.size * 1000:,.1f} m/s")
        super().draw(surf)

    def _draw_node(self, surf, b):
        cam, view = self.app.camera, self.app.view_rect()
        W = self._world()
        p = W @ b.r
        sx, sy, _ = cam.project(p[None])
        if not (np.isfinite(sx[0]) and view.collidepoint(sx[0], sy[0])):
            return
        cx, cy = float(sx[0]), float(sy[0])
        per_km = cam.screen_radius(p, 1.0)
        if per_km <= 0:
            return
        length = px(HANDLE_LEN) / per_km                    # km for a fixed screen length
        basis = vnb_basis(b.r, b.v_before)                  # rows V, N, B in ECI
        clip = surf.get_clip()
        surf.set_clip(view)
        for k, (name, axis, sign, col, text) in enumerate(AXES):
            tip_w = p + W @ (sign * length * basis[axis])
            tx, ty, _ = cam.project(tip_w[None])
            if not np.isfinite(tx[0]):
                continue
            x, y = float(tx[0]), float(ty[0])
            n = math.hypot(x - cx, y - cy)
            if n < px(6):                                   # end-on: nothing to grab
                continue
            u = ((x - cx) / n, (y - cy) / n)
            pygame.draw.line(surf, col, (cx, cy), (x, y), px(2))
            pygame.draw.circle(surf, col, (int(x), int(y)), px(6))
            hot = self.drag is not None and self.drag[0] == "axis" and self.drag[1] == k
            if hot:
                pygame.draw.circle(surf, (255, 255, 255), (int(x), int(y)), px(9), 1)
            label = self.app.fonts.draw(surf, name, (int(x + u[0] * px(14)),
                                                     int(y + u[1] * px(14))),
                                        col, self.app.fonts.small, "center")
            box = pygame.Rect(0, 0, px(2 * GRAB), px(2 * GRAB))
            box.center = (int(x), int(y))
            tips.add(box.union(label), f"{name} handle - {text}\nDrag along its line to "
                                       "change that part of the burn.")
            tips.hot(box)
            self.knobs.append((k, (x, y), u))
        pygame.draw.circle(surf, (255, 255, 255), (int(cx), int(cy)), px(7), 2)
        surf.set_clip(clip)
        self.node_xy = (cx, cy)
        node = pygame.Rect(0, 0, px(2 * GRAB), px(2 * GRAB))
        node.center = (int(cx), int(cy))
        tips.add(node, "The burn node: where the burn fires. Drag it along the orbit, or "
                       "click anywhere on the orbit to move it there.")
        tips.hot(node)


def node_dialog(app):
    """Open a burn node on the selected satellite (key Shift+B)."""
    sim, i = app.sim, app.selected
    if not 0 <= i < sim.n:
        app.toast("Select a satellite first, then Shift+B")
        return None
    if sim.sats[i].status != ACTIVE or sim.ascent_of(i) is not None:
        app.toast(f"{sim.sats[i].name} cannot burn now (not in free flight)")
        return None
    return BurnNodeDialog(app, i)
