"""Heads-up panels: top bar, satellite list, telemetry, event log, help."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..analysis import beta_angle, classify, j2_secular_rates
from ..constants import OMEGA_EARTH, R_EARTH
from ..eclipse import shadow_state
from ..elements import rv2coe
from ..ephemeris import sun_position
from ..frames import eci_to_ecef, ecef_to_geodetic
from ..simulation import ACTIVE
from ..timeutil import format_duration, format_period
from . import theme
from .orbitpanel import draw_orbit_tab
from .widgets import Button

LEFT_W = 262
RIGHT_W = 340
TOP_H = 38
LOG_H = 132
GAP = 8


def fmt(x, unit="", prec=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:,.{prec}f}{(' ' + unit) if unit else ''}"


# --- Top bar -------------------------------------------------------------------------------

class TopBar:
    def __init__(self, app):
        self.app = app
        a = app
        self.buttons = [
            Button("Play", a.toggle_pause, active=lambda: not a.paused, tooltip="Space"),
            Button("<<", lambda: a.change_warp(-1), tooltip=","),
            Button(">>", lambda: a.change_warp(1), tooltip="."),
            Button("1x", a.real_time, tooltip="1"),
            Button("ECI", a.toggle_frame, tooltip="E"),
            Button("Map", lambda: a.toggle("map"), active=lambda: a.opts.map, tooltip="M"),
            Button("Plot", lambda: a.toggle("plot"), active=lambda: a.opts.plot, tooltip="G"),
            Button("Follow", a.toggle_follow, active=lambda: a.follow, tooltip="F"),
            Button("Help", lambda: a.toggle("help"), active=lambda: a.opts.help, tooltip="H"),
        ]

    def layout(self, w):
        x = w - GAP
        for b in reversed(self.buttons):
            bw = 64 if b.text not in ("<<", ">>", "1x") else 38
            b.rect = pygame.Rect(x - bw, 5, bw, TOP_H - 10)
            x -= bw + 5
        self.left_of_buttons = x

    def handle(self, ev):
        return any(b.handle(ev) for b in self.buttons)

    def draw(self, surf):
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        w = surf.get_width()
        self.layout(w)
        self.buttons[0].text = "Pause" if not app.paused else "Play"
        self.buttons[4].text = app.opts.frame
        theme.panel(surf, pygame.Rect(0, 0, w, TOP_H), (8, 12, 22, 235), None, 0)
        pygame.draw.line(surf, theme.PANEL_EDGE, (0, TOP_H), (w, TOP_H))
        x = 12
        x = fonts.draw(surf, "SATELLITE SPACE FLIGHT", (x, TOP_H // 2), theme.ACCENT, fonts.bold, "midleft").right + 18
        dt = sim.datetime()
        parts = [
            (dt.strftime("%Y-%m-%d %H:%M:%S UTC"), theme.TEXT),
            (f"T+{format_duration(sim.t)}", theme.DIM),
            (("PAUSED" if app.paused else f"x{app.warp:g}"), theme.WARN if app.paused else theme.GOOD),
            (f"{sim.propagator}" + (f"/{sim.integrator.method} h={sim.integrator.stats.last_h:.1f}s"
                                    if sim.propagator == "cowell" else ""), theme.DIM),
            (sim.forces.label(), theme.DIM),
            (f"{int(sim.active.sum())}/{sim.n} sats", theme.DIM),
            (f"{app.clock.get_fps():.0f} fps", theme.FAINT),
        ]
        for text, col in parts:
            r = fonts.draw(surf, text, (x, TOP_H // 2), col, fonts.mono, "midleft")
            x = r.right + 16
            if x > self.left_of_buttons - 60:
                break
        for b in self.buttons:
            b.draw(surf, fonts)


# --- Left: satellite list -------------------------------------------------------------------------

class SatList:
    ROW = 21

    def __init__(self, app):
        self.app = app
        a = app
        self.buttons = [
            Button("+ Satellite", lambda: a.open("add"), tooltip="A"),
            Button("Walker", lambda: a.open("walker"), tooltip="W"),
            Button("Manoeuvre", lambda: a.open("maneuver"), tooltip="B"),
            Button("Station", lambda: a.open("station"), tooltip="N"),
            Button("Physics", lambda: a.open("physics"), tooltip="P"),
            Button("Scenarios", lambda: a.open("scenario"), tooltip="Ctrl+O"),
            Button("Edit", lambda: a.open("edit"), tooltip="Ctrl+E"),
            Button("Delete", a.delete_selected, tooltip="Del"),
        ]
        self.scroll = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.list_rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, h):
        self.rect = pygame.Rect(GAP, TOP_H + GAP, LEFT_W, h - TOP_H - 2 * GAP)
        bw = (LEFT_W - 3 * 8) // 2
        for k, b in enumerate(self.buttons):
            col, row = k % 2, k // 2
            b.rect = pygame.Rect(self.rect.x + 8 + col * (bw + 8), self.rect.y + 34 + row * 34, bw, 28)
        top = self.rect.y + 34 + 4 * 34 + 8
        self.list_rect = pygame.Rect(self.rect.x + 6, top, LEFT_W - 12, self.rect.bottom - top - 8)

    def handle(self, ev):
        if any(b.handle(ev) for b in self.buttons):
            return True
        if ev.type == pygame.MOUSEWHEEL and self.list_rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, self.scroll - ev.y * 3)
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.list_rect.collidepoint(ev.pos):
            k = (ev.pos[1] - self.list_rect.y) // self.ROW + self.scroll
            if 0 <= k < self.app.sim.n:
                self.app.select(k)
            return True
        return self.rect.collidepoint(getattr(ev, "pos", (-1, -1)))

    def ensure_visible(self, i):
        rows = max(1, self.list_rect.h // self.ROW)
        if i < self.scroll:
            self.scroll = i
        elif i >= self.scroll + rows:
            self.scroll = i - rows + 1

    def draw(self, surf):
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        self.layout(surf.get_height())
        theme.panel(surf, self.rect)
        fonts.draw(surf, "SCENARIO", (self.rect.x + 10, self.rect.y + 9), theme.FAINT, fonts.small)
        name = sim.scenario.name
        if fonts.bold.size(name)[0] > LEFT_W - 92:
            while len(name) > 4 and fonts.bold.size(name + '...')[0] > LEFT_W - 92:
                name = name[:-1]
            name = name.rstrip() + '...'
        fonts.draw(surf, name, (self.rect.x + 80, self.rect.y + 7), theme.TEXT, fonts.bold)
        for b in self.buttons:
            b.draw(surf, fonts)
        lr = self.list_rect
        rows = max(1, lr.h // self.ROW)
        self.scroll = max(0, min(self.scroll, max(0, sim.n - rows)))
        clip = surf.get_clip()
        surf.set_clip(lr)
        for k in range(self.scroll, min(sim.n, self.scroll + rows)):
            s = sim.sats[k]
            y = lr.y + (k - self.scroll) * self.ROW
            row = pygame.Rect(lr.x, y, lr.w, self.ROW - 1)
            if k == app.selected:
                pygame.draw.rect(surf, theme.ACCENT_DARK, row, border_radius=4)
            pygame.draw.circle(surf, s.color if s.status == ACTIVE else (100, 100, 100),
                               (row.x + 10, row.centery), 5)
            fonts.draw(surf, s.name[:20], (row.x + 22, row.centery), theme.TEXT if s.status == ACTIVE else theme.FAINT,
                       fonts.ui, "midleft")
            if s.status == ACTIVE:
                alt = np.linalg.norm(sim.y[k, :3]) - R_EARTH
                txt = f"{alt:,.0f} km" if alt < 1e6 else f"{alt / 1e6:.2f} Gm"
                col = theme.DIM if s.shadow > 0.5 else (150, 140, 230)
            else:
                txt, col = s.status, theme.BAD
            fonts.draw(surf, txt, (row.right - 6, row.centery), col, fonts.small, "midright")
        surf.set_clip(clip)
        if sim.n > rows:
            frac = rows / sim.n
            bar_h = max(20, int(lr.h * frac))
            y = lr.y + int((lr.h - bar_h) * self.scroll / max(1, sim.n - rows))
            pygame.draw.rect(surf, theme.PANEL_EDGE, (lr.right - 3, y, 3, bar_h), border_radius=2)


# --- Right: telemetry --------------------------------------------------------------------------

class InfoPanel:
    TABS = ("Orbit", "Telemetry")

    def __init__(self, app):
        self.app = app
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.tab = 0
        self.tab_rects: list[pygame.Rect] = []
        self.scroll = [0, 0]           # per tab, pixels
        self.content_h = 0
        self.body = pygame.Rect(0, 0, 0, 0)

    def layout(self, w, h):
        self.rect = pygame.Rect(w - RIGHT_W - GAP, TOP_H + GAP, RIGHT_W, h - TOP_H - 2 * GAP)
        x, y = self.rect.x + 12, self.rect.y + 36
        self.tab_rects = []
        for name in self.TABS:
            tw = self.app.fonts.small.size(name)[0] + 22
            self.tab_rects.append(pygame.Rect(x, y, tw, 22))
            x += tw + 4
        self.body = pygame.Rect(self.rect.x + 4, y + 28, RIGHT_W - 8, self.rect.bottom - y - 28 - 30)

    def cycle_tab(self):
        self.tab = (self.tab + 1) % len(self.TABS)

    def handle(self, ev):
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll[self.tab] = max(0, self.scroll[self.tab] - ev.y * 48)
            return True
        if ev.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(ev.pos):
            return False
        for k, r in enumerate(self.tab_rects):
            if r.collidepoint(ev.pos) and ev.button == 1:
                self.tab = k
        return True

    def report(self, i):
        sim = self.app.sim
        s = sim.sats[i]
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        theta = sim.gmst()
        r_ecef = eci_to_ecef(r, theta)
        lat, lon, alt = (float(x) for x in ecef_to_geodetic(r_ecef))
        rm, vm = float(np.linalg.norm(r)), float(np.linalg.norm(v))
        v_ground = float(np.linalg.norm(v - np.cross([0, 0, OMEGA_EARTH], r)))
        fpa = math.degrees(math.asin(np.clip(np.dot(r, v) / (rm * vm), -1, 1)))
        D = math.degrees
        sections = []
        sections.append(("STATE", [
            ("Altitude", fmt(alt, "km", 2)),
            ("Latitude / Longitude", f"{D(lat):+.3f} / {D(lon):+.3f} deg"),
            ("Radius", fmt(rm, "km", 1)),
            ("Speed (inertial)", fmt(vm, "km/s", 4)),
            ("Speed (ground-rel.)", fmt(v_ground, "km/s", 4)),
            ("Flight-path angle", fmt(fpa, "deg", 3)),
        ]))
        orbit = [
            ("Semi-major axis a", fmt(el.a, "km", 2)),
            ("Eccentricity e", f"{el.e:.6f}"),
            ("Inclination i", fmt(D(el.i), "deg", 4)),
            ("RAAN", fmt(D(el.raan), "deg", 4)),
            ("Arg. of perigee", fmt(D(el.argp), "deg", 3)),
            ("True anomaly", fmt(D(el.nu), "deg", 3)),
            ("Arg. of latitude", fmt(D(el.u), "deg", 3)),
        ]
        if el.e < 1:
            orbit += [
                ("Perigee / apogee alt", f"{el.rp - R_EARTH:,.1f} / {el.ra - R_EARTH:,.1f} km"),
                ("Period", format_period(el.period)),
                ("Revs per day", f"{86400 / el.period:.4f}"),
            ]
            rd, wd, _ = j2_secular_rates(el.a, el.e, el.i)
            orbit.append(("J2 dRAAN / dargp", f"{D(rd) * 86400:+.4f} / {D(wd) * 86400:+.4f} deg/d"))
        else:
            orbit += [("Perigee alt", fmt(el.rp - R_EARTH, "km", 1)),
                      ("Hyperbolic excess v", fmt(math.sqrt(max(0, 2 * el.energy)), "km/s", 4))]
        orbit += [("Specific energy", fmt(el.energy, "km^2/s^2", 4)),
                  ("Angular momentum", fmt(el.h, "km^2/s", 1))]
        sections.append(("ORBIT  (" + classify(el) + ")", orbit))

        jd = sim.jd()
        sun = sun_position(jd)
        env = [("Illumination", f"{shadow_state(s.shadow)} ({s.shadow * 100:.0f}%)"),
               ("Beta angle", fmt(D(beta_angle(r, v, sun)), "deg", 2))]
        cd_am = s.cd * s.area / s.mass
        br = sim.forces.breakdown(r, v, cd_am)
        for name, (mag, on) in br.items():
            env.append((f"  a_{name}" + ("" if on else " (off)"), f"{mag * 1e3:.3e} m/s^2"))
        e_now = sim.energy(i)
        if s.energy_ref is None or any(b.man.sat == s.name for b in sim.burns):
            s.energy_ref = (sim.t, e_now)
        t_ref, e_ref = s.energy_ref
        env.append((f"Energy drift (last {format_duration(sim.t - t_ref).split('.')[0]})",
                    f"{(e_now - e_ref) / abs(e_ref):+.2e}"))
        sections.append(("ENVIRONMENT", env))

        craft = [("Mass", fmt(s.mass, "kg", 2)), ("dV spent", fmt(s.dv_used * 1000, "m/s", 2)),
                 ("Cd*A/m", f"{cd_am:.4f} m^2/kg")]
        pending = [m for m in sim.maneuvers if m.sat == s.name]
        for m in pending[:3]:
            craft.append((f"  in {format_duration(m.t - sim.t)}", m.describe()[:26]))
        for b in sim.burns:
            if b.man.sat == s.name:
                craft.append(("  BURNING", f"{b.man.thrust:.0f} N, {b.t1 - sim.t:.0f} s left"))
        sections.append(("SPACECRAFT", craft))
        vis = sim.station_visibility(i)
        if sim.stations:
            sections.append(("GROUND CONTACT", [(st.name[:16], f"az {az:5.1f} el {el_:4.1f} {rng:,.0f} km")
                                                for st, az, el_, rng in vis] or [("(none in view)", "")]))
        return sections

    def draw(self, surf):
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        self.layout(*surf.get_size())
        theme.panel(surf, self.rect)
        x, y = self.rect.x + 12, self.rect.y + 10
        i = app.selected
        if not 0 <= i < sim.n:
            fonts.draw(surf, "No satellite selected", (x, y), theme.DIM, fonts.ui)
            fonts.draw(surf, "Click one in the view or list, or press A to add.", (x, y + 22),
                       theme.FAINT, fonts.small)
            self._closest(surf, x, self.rect.bottom - 30)
            return
        s = sim.sats[i]
        pygame.draw.circle(surf, s.color, (x + 6, y + 11), 6)
        fonts.draw(surf, s.name, (x + 20, y), theme.TEXT, fonts.title)
        status_col = theme.GOOD if s.status == ACTIVE else theme.BAD
        fonts.draw(surf, s.status.upper(), (self.rect.right - 12, y + 4), status_col, fonts.small, "topright")
        for k, (name, r) in enumerate(zip(self.TABS, self.tab_rects)):
            on = k == self.tab
            pygame.draw.rect(surf, theme.ACCENT_DARK if on else theme.FIELD, r, border_radius=5)
            if on:
                pygame.draw.rect(surf, theme.ACCENT, r, 1, border_radius=5)
            fonts.draw(surf, name, r.center, theme.TEXT if on else theme.DIM, fonts.small, "center")
        fonts.draw(surf, "Q", (self.rect.right - 12, self.tab_rects[0].centery), theme.FAINT, fonts.small,
                   "midright")
        body = self.body
        top = self.scroll[self.tab] = max(0, min(self.scroll[self.tab], self.content_h - body.h))
        clip = surf.get_clip()
        surf.set_clip(body)
        y0 = body.y + 2 - top
        if self.tab == 0:
            y = draw_orbit_tab(surf, x, y0, RIGHT_W - 24, app, i)
        else:
            y = y0
            for title, rows in self.report(i):
                fonts.draw(surf, title, (x, y), theme.ACCENT, fonts.small)
                y += 18
                for label, value in rows:
                    fonts.draw(surf, label, (x + 4, y), theme.DIM, fonts.small)
                    fonts.draw(surf, value, (self.rect.right - 12, y), theme.TEXT, fonts.small, "topright")
                    y += 16
                y += 6
        self.content_h = y - y0
        surf.set_clip(clip)
        if self.content_h > body.h:
            frac = body.h / self.content_h
            bar_h = max(24, int(body.h * frac))
            yb = body.y + int((body.h - bar_h) * top / max(1, self.content_h - body.h))
            pygame.draw.rect(surf, theme.PANEL_EDGE, (self.rect.right - 5, yb, 3, bar_h), border_radius=2)
        self._closest(surf, x, self.rect.bottom - 22)

    def _closest(self, surf, x, y):
        sim, fonts = self.app.sim, self.app.fonts
        d, a, b = sim.closest
        if a >= 0 and math.isfinite(d):
            col = theme.BAD if d < sim.conjunction_km else theme.FAINT
            clip = surf.get_clip()
            surf.set_clip(self.rect.inflate(-8, 0))
            fonts.draw(surf, f"Closest: {sim.sats[a].name} - {sim.sats[b].name}  {d:,.1f} km",
                       (x, y), col, fonts.small)
            surf.set_clip(clip)


# --- Bottom: event log ----------------------------------------------------------------------------

class EventLog:
    def __init__(self, app):
        self.app = app
        self.rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, w, h):
        x0 = GAP * 2 + LEFT_W if self.app.opts.panels else GAP
        x1 = w - RIGHT_W - 2 * GAP if self.app.opts.panels else w - GAP
        self.rect = pygame.Rect(x0, h - LOG_H - GAP, x1 - x0, LOG_H)

    def draw(self, surf):
        sim, fonts = self.app.sim, self.app.fonts
        self.layout(*surf.get_size())
        theme.panel(surf, self.rect)
        fonts.draw(surf, "EVENTS", (self.rect.x + 10, self.rect.y + 6), theme.FAINT, fonts.small)
        y = self.rect.bottom - 20
        for ev in reversed(sim.events[-7:]):
            col = theme.EVENT_COLORS.get(ev.kind, theme.DIM)
            fonts.draw(surf, f"T+{format_duration(ev.t):>13}", (self.rect.x + 10, y), theme.FAINT, fonts.small)
            fonts.draw(surf, ev.text[:110], (self.rect.x + 130, y), col, fonts.small)
            y -= 16
            if y < self.rect.y + 22:
                break


HELP = [
    ("Mouse", ""),
    ("Left-drag", "orbit the camera"),
    ("Wheel", "zoom"),
    ("Click satellite", "select it (list or 3-D view)"),
    ("Keyboard", ""),
    ("Space", "pause / resume"),
    (", .", "slower / faster time warp"),
    ("1", "real time"),
    ("Tab / Shift+Tab", "next / previous satellite"),
    ("F", "follow selected satellite"),
    ("E", "ECI (inertial) / ECEF (Earth-fixed) view"),
    ("O", "orbits: auto / all / selected / none"),
    ("T  L  V  X", "trails, labels, velocity vectors, axes"),
    ("C  R  K", "coverage footprint, GEO ring, coastlines"),
    ("D", "orbit geometry: full / basic / off"),
    ("Q", "right panel: orbit / telemetry tab"),
    ("M  G", "ground-track map, telemetry plot"),
    ("A  W  B  N", "add satellite, Walker, manoeuvre, station"),
    ("P", "physics & integrator settings"),
    ("Ctrl+O / Ctrl+S", "scenarios / save snapshot"),
    ("Ctrl+E  Del", "edit / delete selected satellite"),
    ("Ctrl+R", "reset scenario"),
    ("I", "hide / show panels"),
    ("F12", "screenshot to screenshots/"),
    ("H or F1", "this help"),
]


def draw_help(surf, app):
    fonts = app.fonts
    w, h = surf.get_size()
    rect = pygame.Rect(0, 0, 520, 34 + 20 * len(HELP) + 20)
    rect.center = (w // 2, h // 2)
    theme.panel(surf, rect, (14, 20, 36, 245), theme.ACCENT)
    y = rect.y + 14
    fonts.draw(surf, "Controls", (rect.x + 18, y), theme.TEXT, fonts.title)
    y += 30
    for key, desc in HELP:
        if not desc:
            fonts.draw(surf, key.upper(), (rect.x + 18, y + 2), theme.ACCENT, fonts.small)
        else:
            fonts.draw(surf, key, (rect.x + 30, y + 10), theme.TEXT, fonts.mono, "midleft")
            fonts.draw(surf, desc, (rect.x + 200, y + 10), theme.DIM, fonts.ui, "midleft")
        y += 20
