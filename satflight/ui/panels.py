"""Heads-up panels: top bar, satellite list, telemetry, event log, help."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..analysis import beta_angle, classify, j2_secular_rates
from ..constants import R_EARTH
from ..eclipse import shadow_state
from ..elements import rv2coe
from ..ephemeris import sun_position
from ..frames import OMEGA_VEC, ecef_to_geodetic, eci_to_ecef
from ..simulation import ACTIVE, DECAYED, ESCAPED, IMPACTED, REENTRY_ALT, SCRUBBED
from ..timeutil import format_duration, format_period
from . import glossary, theme, tips
from .launchui import draw_ascent_tab, launch_rows
from .orbitpanel import draw_orbit_tab, draw_section, num
from .theme import px
from .widgets import Button

# design px (see theme.px)
LEFT_W = 262
RIGHT_W = 340
TOP_H = 38
LOG_H = 132
GAP = 8          # between panels and window edges
PAGE_KEYS = (pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END)
TAB_TIPS = ("Diagrams and properties of the orbit (the ascent while a rocket climbs).",
            "Live readings: position, orbital elements, forces, spacecraft and ground "
            "contact.")
STATUS_TIPS = {
    ACTIVE: "Active: the satellite is flying and being simulated.",
    DECAYED: f"Re-entered: it sank below {REENTRY_ALT:.0f} km and burned up in the "
             "atmosphere.",
    IMPACTED: "Impacted: its path met the ground.",
    ESCAPED: "Escaped: it left the Earth's sphere of influence.",
    SCRUBBED: "Scrubbed: the launch never left the pad.",
}


# --- Top bar -------------------------------------------------------------------------------

class FullscreenButton(Button):
    """A square button showing four corner marks, pointing out to enter fullscreen
    and in to leave it."""

    def __init__(self, callback, **kw):
        super().__init__("", callback, **kw)

    def draw(self, surf, fonts):
        super().draw(surf, fonts)
        r = self.rect.inflate(-px(14), -px(14))
        k = max(px(3), r.width // 3)
        into = self.active is not None and self.active()
        for cx, cy, dx, dy in ((r.left, r.top, 1, 1), (r.right - 1, r.top, -1, 1),
                               (r.left, r.bottom - 1, 1, -1), (r.right - 1, r.bottom - 1, -1, -1)):
            if into:                             # the corner sits inside, arms point outwards
                cx, cy, dx, dy = cx + dx * k, cy + dy * k, -dx, -dy
            pygame.draw.lines(surf, theme.TEXT, False,
                              [(cx + dx * k, cy), (cx, cy), (cx, cy + dy * k)], px(2))


class TopBar:
    """Title, clock, warp and run statistics, with the view-toggle buttons on the right."""

    def __init__(self, app):
        self.app = app
        a = app
        self.buttons = [
            Button("Play", a.toggle_pause, active=lambda: not a.paused, tooltip="Space",
                   hint="Pause or resume the simulation"),
            Button("<<", lambda: a.change_warp(-1), tooltip=",", hint="Slower time warp"),
            Button(">>", lambda: a.change_warp(1), tooltip=".", hint="Faster time warp"),
            Button("1x", a.real_time, tooltip="1", hint="Real time"),
            Button("ECI", a.toggle_frame, tooltip="E",
                   hint="View axes: inertial (ECI) or Earth-fixed (ECEF)"),
            Button("Map", lambda: a.toggle("map"), active=lambda: a.opts.map, tooltip="M",
                   hint="Ground-track map"),
            Button("Plot", lambda: a.toggle("plot"), active=lambda: a.opts.plot, tooltip="G",
                   hint="Telemetry plot (click the plot to change what it shows)"),
            Button("Follow", a.toggle_follow, active=lambda: a.follow, tooltip="F",
                   hint="Camera follows the selected satellite"),
            FullscreenButton(a.toggle_fullscreen, active=lambda: a.fullscreen,
                             tooltip="F11", hint="Fullscreen (F11 or Alt+Enter; Esc leaves it)"),
            Button("Help", lambda: a.toggle("help"), active=lambda: a.opts.help, tooltip="H",
                   hint="Mouse and keyboard controls"),
        ]

    def layout(self, w):
        """Right-align the buttons in a window ``w`` px wide."""
        x, bh = w - px(GAP), px(TOP_H - 10)
        for b in reversed(self.buttons):
            bw = px(64 if b.text not in ("<<", ">>", "1x") else 38)
            if isinstance(b, FullscreenButton):
                bw = bh                          # square: the clock needs the room at 900 px
            b.rect = pygame.Rect(x - bw, px(5), bw, bh)
            x -= bw + px(5)
        self.left_of_buttons = x

    def handle(self, ev):
        """Pass the event to the buttons; True if one used it."""
        return any(b.handle(ev) for b in self.buttons)

    def draw(self, surf):
        """Title, UTC and mission time, warp, propagator, forces, counts, fps and the buttons."""
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        w = surf.get_width()
        self.layout(w)
        self.buttons[0].text = "Pause" if not app.paused else "Play"   # labels show the state
        self.buttons[4].text = app.opts.frame
        top_h, room = px(TOP_H), self.left_of_buttons - px(8)
        theme.panel(surf, pygame.Rect(0, 0, w, top_h), (8, 12, 22, 235), None, 0)
        tips.block(pygame.Rect(0, 0, w, top_h))
        pygame.draw.line(surf, theme.PANEL_EDGE, (0, top_h), (w, top_h))
        title = fonts.draw(surf, "SATELLITE SPACE FLIGHT", (px(12), top_h // 2), theme.ACCENT,
                           fonts.bold, "midleft")
        tips.add(title, "An Earth-orbit simulator. Rest the mouse on any part of the window "
                        "to see what it is; H lists every control.", "H")
        x = title.right + px(18)
        dt = sim.datetime()
        stamp = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        if x + fonts.mono.size(stamp)[0] > room:
            stamp = dt.strftime("%H:%M:%S UTC")         # narrow window: the time matters most
        cowell = sim.propagator == "cowell"
        parts = [
            (stamp, theme.TEXT, f"Simulation date and time (UTC): "
                                f"{dt.strftime('%A %d %B %Y, %H:%M:%S')}.", ""),
            (f"T+{format_duration(sim.t)}", theme.DIM,
             "Time simulated since the scenario started; Ctrl+R starts it again.", "Ctrl+R"),
            ("PAUSED", theme.WARN, "The simulation is paused.", "Space") if app.paused else
            (f"x{app.warp:g}", theme.GOOD, f"Time warp: {app.warp:g} simulated seconds pass "
                                           "every real second.", ", ."),
            (sim.propagator + (f"/{sim.integrator.method} h={sim.integrator.stats.last_h:.1f}s"
                               if cowell else ""), theme.DIM,
             "Propagator" + (f": Cowell integrates every force numerically with "
                             f"{sim.integrator.method}; h is its last time step." if cowell
                             else f" {sim.propagator}: an analytic orbit formula, no "
                                  "numerical integration.") + " Change it in Physics.", "P"),
            (sim.forces.label(), theme.DIM, "Forces acting: 2B is the Earth's central gravity, "
                                            "J2-J4 its uneven shape, DRAG the upper atmosphere.",
             "P"),
            (f"{int(sim.active.sum())}/{sim.n} sats", theme.DIM,
             "Satellites still flying / all objects in the scenario (re-entered, crashed "
             "and escaped ones included).", ""),
            (f"{app.clock.get_fps():.0f} fps", theme.FAINT,
             "Frames drawn per second. The physics keeps the same steps however slow the "
             "drawing is.", ""),
        ]
        for text, col, tip, keys in parts:
            if x + fonts.mono.size(text)[0] > room:
                break                   # only whole items, never one cut by the buttons
            r = fonts.draw(surf, text, (x, top_h // 2), col, fonts.mono, "midleft")
            tips.add(r.inflate(px(8), px(12)), tip, keys)
            x = r.right + px(16)
        for b in self.buttons:
            b.draw(surf, fonts)


# --- Left: satellite list -------------------------------------------------------------------------

class SatList:
    """Left panel: scenario name, action buttons and the scrollable satellite list."""

    ROW = 21                      # design px per list row

    def __init__(self, app):
        self.app = app
        a = app
        self.buttons = [
            Button("+ Satellite", lambda: a.open("add"), tooltip="A",
                   hint="Add a satellite: preset, orbit elements, state vector or TLE"),
            Button("Launch", lambda: a.open("launch"), tooltip="U", accent=True,
                   hint="Launch a rocket from anywhere on Earth"),
            Button("Walker", lambda: a.open("walker"), tooltip="W",
                   hint="Add a Walker constellation"),
            Button("Manoeuvre", lambda: a.open("maneuver"), tooltip="B",
                   hint="Plan a burn for the selected satellite"),
            Button("Station", lambda: a.open("station"), tooltip="N", hint="Add a ground station"),
            Button("Physics", lambda: a.open("physics"), tooltip="P",
                   hint="Force models and integrator"),
            Button("Scenarios", lambda: a.open("scenario"), tooltip="Ctrl+O",
                   hint="Load, save or export a scenario"),
            Button("Start screen", lambda: a.open("start"), tooltip="Ctrl+N",
                   hint="Choose a bundled scenario"),
            Button("Edit", lambda: a.open("edit"), tooltip="Ctrl+E",
                   hint="Edit the selected satellite"),
            Button("Delete", a.delete_selected, tooltip="Del",
                   hint="Remove the selected satellite"),
        ]
        self.scroll = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.list_rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, h):
        """Place the panel, the two-column button grid and the list for window height ``h``."""
        gap, top_h, w = px(GAP), px(TOP_H), px(LEFT_W)
        self.rect = pygame.Rect(gap, top_h + gap, w, h - top_h - 2 * gap)
        pad, pitch = px(8), px(34)
        bw = (w - 3 * pad) // 2
        for k, b in enumerate(self.buttons):
            col, row = k % 2, k // 2
            b.rect = pygame.Rect(self.rect.x + pad + col * (bw + pad),
                                 self.rect.y + pitch + row * pitch, bw, px(28))
        top = self.rect.y + pitch + (len(self.buttons) + 1) // 2 * pitch + pad
        self.list_rect = pygame.Rect(self.rect.x + px(6), top, w - px(12),
                                     self.rect.bottom - top - pad)

    def handle(self, ev):
        """Buttons, then list scrolling and selection; swallows clicks on the panel."""
        if any(b.handle(ev) for b in self.buttons):
            return True
        if ev.type == pygame.MOUSEWHEEL and self.list_rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, self.scroll - ev.y * 3)
            return True
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.list_rect.collidepoint(ev.pos)):
            k = (ev.pos[1] - self.list_rect.y) // px(self.ROW) + self.scroll
            if 0 <= k < self.app.sim.n:
                self.app.select(k)
            return True
        return self.rect.collidepoint(getattr(ev, "pos", (-1, -1)))

    def ensure_visible(self, i):
        """Scroll so that row ``i`` is in view."""
        rows = max(1, self.list_rect.h // px(self.ROW))
        if i < self.scroll:
            self.scroll = i
        elif i >= self.scroll + rows:
            self.scroll = i - rows + 1

    def draw(self, surf):
        """Scenario name, buttons and the scrolling satellite list."""
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        self.layout(surf.get_height())
        theme.panel(surf, self.rect)
        tips.block(self.rect)                           # the panel hides the 3-D view
        head = fonts.draw(surf, "SCENARIO", (self.rect.x + px(10), self.rect.y + px(9)),
                          theme.FAINT, fonts.small)
        name = fonts.fit(sim.scenario.name, px(LEFT_W) - px(92), fonts.bold)
        r = fonts.draw(surf, name, (self.rect.x + px(80), self.rect.y + px(7)), theme.TEXT,
                       fonts.bold)
        about = sim.scenario.description or "The scenario now running."
        tips.add(head.union(r).inflate(px(4), px(6)),
                 f"{sim.scenario.name}: {about}\nScenarios (Ctrl+O) loads another one.", "Ctrl+O")
        for b in self.buttons:
            b.draw(surf, fonts)
        lr, row_h = self.list_rect, px(self.ROW)
        rows = max(1, lr.h // row_h)
        self.scroll = max(0, min(self.scroll, max(0, sim.n - rows)))
        tips.add(lr, f"Every object in the scenario ({sim.n}): satellites, rockets and spent "
                     "stages, with their altitude. Click one to select it; the wheel scrolls.",
                 "Tab")
        clip = surf.get_clip()
        surf.set_clip(lr)
        for k in range(self.scroll, min(sim.n, self.scroll + rows)):
            s = sim.sats[k]
            y = lr.y + (k - self.scroll) * row_h
            row = pygame.Rect(lr.x, y, lr.w, row_h - 1)
            if k == app.selected:
                pygame.draw.rect(surf, theme.ACCENT_DARK, row, border_radius=px(4))
            pygame.draw.circle(surf, s.color if s.status == ACTIVE else (100, 100, 100),
                               (row.x + px(10), row.centery), px(5))
            asc = sim.ascent_of(k) if sim.ascents else None
            if asc is not None and asc.phase == "pad":
                txt, col = f"T-{format_duration(asc.t0 - sim.t).split('.')[0]}", theme.WARN
                what = "a rocket on its pad; the countdown to lift-off"
            elif asc is not None:
                alt = np.linalg.norm(sim.y[k, :3]) - R_EARTH
                txt, col = f"{alt:,.0f} km ^", theme.WARN
                what = "a rocket climbing (^); its altitude"
            elif s.status == ACTIVE:
                alt = np.linalg.norm(sim.y[k, :3]) - R_EARTH
                txt = f"{alt:,.0f} km" if alt < 1e6 else f"{alt / 1e6:.2f} Gm"
                col = theme.DIM if s.shadow > 0.5 else (150, 140, 230)
                what = ("altitude, grey in sunlight" if s.shadow > 0.5
                        else "altitude, violet while in the Earth's shadow")
            else:
                txt, col = s.status, theme.BAD
                what = f"no longer flying ({s.status})"
            r = fonts.draw(surf, txt, (row.right - px(6), row.centery), col, fonts.small,
                           "midright")
            # the name gets whatever the value leaves, so long names never run into it
            name = fonts.fit(s.name, r.x - px(10) - (row.x + px(22)))
            fonts.draw(surf, name, (row.x + px(22), row.centery),
                       theme.TEXT if s.status == ACTIVE else theme.FAINT, fonts.ui, "midleft")
            act = ("Selected: its details are in the right panel." if k == app.selected
                   else "Click to select it.")
            tips.add(row, f"{s.name} - {what}.\n{act} Tab / Shift+Tab steps through the list, "
                          "the wheel scrolls it.", clip=lr)
        surf.set_clip(clip)
        if sim.n > rows:
            theme.scrollbar(surf, lr.right - px(3), lr, self.scroll, rows, sim.n, 20)


# --- Right: telemetry --------------------------------------------------------------------------

class InfoPanel:
    """Right panel for the selected satellite: the Orbit tab (or the ascent while a
    rocket flies) and the Telemetry tab."""

    TABS = ("Orbit", "Telemetry")

    def __init__(self, app):
        self.app = app
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.tab = 0
        self.tab_rects: list[pygame.Rect] = []
        self.scroll = [0, 0]           # per tab, pixels
        self.content_h = 0
        self.body = pygame.Rect(0, 0, 0, 0)
        self._drag = None              # grab offset while the scrollbar thumb is dragged

    def layout(self, w, h):
        """Place the panel, its tab buttons and the scrollable body."""
        gap, top_h, pw = px(GAP), px(TOP_H), px(RIGHT_W)
        self.rect = pygame.Rect(w - pw - gap, top_h + gap, pw, h - top_h - 2 * gap)
        x, y = self.rect.x + px(12), self.rect.y + px(36)
        self.tab_rects = []
        for name in self.TABS:
            tw = self.app.fonts.small.size(name)[0] + px(22)
            self.tab_rects.append(pygame.Rect(x, y, tw, px(22)))
            x += tw + px(4)
        self.body = pygame.Rect(self.rect.x + px(4), y + px(28), pw - px(8),
                                self.rect.bottom - y - px(58))

    def cycle_tab(self):
        """Switch to the next tab (key Q)."""
        self.tab = (self.tab + 1) % len(self.TABS)

    def scrollbar(self):
        """The scrollbar's track and thumb rects; the thumb is None when the tab fits."""
        body = self.body
        track = pygame.Rect(self.rect.right - px(9), body.y, px(6), body.h)
        if self.content_h <= body.h:
            return track, None
        thumb_h = max(px(24), int(body.h * body.h / self.content_h))
        f = self.scroll[self.tab] / (self.content_h - body.h)
        return track, pygame.Rect(track.x, body.y + int((body.h - thumb_h) * min(1.0, f)),
                                  track.w, thumb_h)

    def _thumb_to(self, y):
        """Scroll so that the thumb's top is at screen ``y``."""
        track, thumb = self.scrollbar()
        if thumb is None:
            return
        f = (y - track.y) / max(1, track.h - thumb.h)
        self.scroll[self.tab] = int(round(min(1.0, max(0.0, f)) * (self.content_h - track.h)))

    def handle(self, ev):
        """Scroll the current tab (wheel, scrollbar, PgUp/PgDn/Home/End), or switch
        tabs on a click; swallows clicks on the panel."""
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll[self.tab] = max(0, self.scroll[self.tab] - ev.y * px(48))
            return True
        if ev.type == pygame.KEYDOWN and ev.key in PAGE_KEYS and not ev.mod & pygame.KMOD_CTRL:
            page, s = max(px(48), self.body.h - px(48)), self.scroll[self.tab]
            s = {pygame.K_PAGEUP: s - page, pygame.K_PAGEDOWN: s + page, pygame.K_HOME: 0,
                 pygame.K_END: self.content_h}[ev.key]
            self.scroll[self.tab] = max(0, s)       # draw() clamps the far end
            return True
        if self._drag is not None:
            if ev.type == pygame.MOUSEMOTION:
                self._thumb_to(ev.pos[1] - self._drag)
                return True
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self._drag = None
                return True
        if ev.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(ev.pos):
            return False
        track, thumb = self.scrollbar()
        if ev.button == 1 and thumb is not None and track.inflate(px(10), 0).collidepoint(ev.pos):
            if not thumb.collidepoint(ev.pos):      # a click on the track jumps there
                self._thumb_to(ev.pos[1] - thumb.h // 2)
                track, thumb = self.scrollbar()
            self._drag = ev.pos[1] - thumb.y
            return True
        for k, r in enumerate(self.tab_rects):
            if r.collidepoint(ev.pos) and ev.button == 1:
                self.tab = k
        return True

    def report(self, i):
        """Telemetry tab content for satellite ``i``: [(section title, [(label, value)])]."""
        sim = self.app.sim
        s = sim.sats[i]
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        theta = sim.gmst()
        r_ecef = eci_to_ecef(r, theta)
        lat, lon, alt = (float(x) for x in ecef_to_geodetic(r_ecef))
        rm, vm = float(np.linalg.norm(r)), float(np.linalg.norm(v))
        v_ground = float(np.linalg.norm(v - np.cross(OMEGA_VEC, r)))
        fpa = math.degrees(math.asin(np.clip(np.dot(r, v) / (rm * vm), -1, 1)))
        D = math.degrees
        sections = [("STATE", [
            ("Altitude", num(alt, "km", 2)),
            ("Latitude / Longitude", f"{D(lat):+.3f} / {D(lon):+.3f} deg"),
            ("Radius", num(rm, "km", 1)),
            ("Speed (inertial)", num(vm, "km/s", 4)),
            ("Speed (ground-rel.)", num(v_ground, "km/s", 4)),
            ("Flight-path angle", num(fpa, "deg", 3)),
        ])]
        orbit = [
            ("Semi-major axis a", num(el.a, "km", 2)),
            ("Eccentricity e", f"{el.e:.6f}"),
            ("Inclination i", num(D(el.i), "deg", 4)),
            ("RAAN", num(D(el.raan), "deg", 4)),
            ("Arg. of perigee", num(D(el.argp), "deg", 3)),
            ("True anomaly", num(D(el.nu), "deg", 3)),
            ("Arg. of latitude", num(D(el.u), "deg", 3)),
        ]
        if el.e < 1:
            orbit += [
                ("Perigee / apogee alt", f"{el.rp - R_EARTH:,.1f} / {el.ra - R_EARTH:,.1f} km"),
                ("Period", format_period(el.period)),
                ("Revs per day", f"{86400 / el.period:.4f}"),
            ]
            rd, wd, _ = j2_secular_rates(el.a, el.e, el.i)
            orbit.append(("J2 dRAAN / dargp",
                          f"{D(rd) * 86400:+.4f} / {D(wd) * 86400:+.4f} deg/d"))
        else:
            orbit += [("Perigee alt", num(el.rp - R_EARTH, "km", 1)),
                      ("Hyperbolic excess v", num(math.sqrt(max(0, 2 * el.energy)), "km/s", 4))]
        orbit += [("Specific energy", num(el.energy, "km^2/s^2", 4)),
                  ("Angular momentum", num(el.h, "km^2/s", 1))]
        sections.append(("ORBIT  (" + classify(el) + ")", orbit))

        jd = sim.jd()
        sun = sun_position(jd)
        env = [("Illumination", f"{shadow_state(s.shadow)} ({s.shadow * 100:.0f}%)"),
               ("Beta angle", num(D(beta_angle(r, v, sun)), "deg", 2))]
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

        launch = launch_rows(sim, i)
        if launch:
            sections.insert(0, ("LAUNCH", launch))
        craft = [("Mass", num(s.mass, "kg", 2)), ("dV spent", num(s.dv_used * 1000, "m/s", 2)),
                 ("Cd*A/m", f"{cd_am:.4f} m^2/kg")]
        pending = [m for m in sim.maneuvers if m.sat == s.name]
        for m in pending[:3]:
            craft.append((f"  in {format_duration(m.t - sim.t)}", m.describe()[:26]))
        for b in sim.burns:
            if b.man.sat == s.name:
                craft.append(("  BURNING", f"{b.man.thrust:.0f} N, {b.t1 - sim.t:.0f} s left"))
        sections.append(("SPACECRAFT", craft))
        if sim.stations:
            contact = [(st.name[:16], f"az {az:5.1f} el {el_:4.1f} {rng:,.0f} km")
                       for st, az, el_, rng in sim.station_visibility(i)]
            sections.append(("GROUND CONTACT", contact or [("(none in view)", "")]))
        return sections

    def draw(self, surf):
        """Tab buttons and the scrolled content of the current tab."""
        app, sim, fonts = self.app, self.app.sim, self.app.fonts
        self.layout(*surf.get_size())
        theme.panel(surf, self.rect)
        tips.block(self.rect)
        x, y = self.rect.x + px(12), self.rect.y + px(10)
        i = app.selected
        if not 0 <= i < sim.n:
            fonts.draw(surf, "No satellite selected", (x, y), theme.DIM, fonts.ui)
            fonts.draw(surf, "Click one in the view or list, or press A to add.", (x, y + px(22)),
                       theme.FAINT, fonts.small)
            self._closest(surf, x, self.rect.bottom - px(30))
            return
        s = sim.sats[i]
        pygame.draw.circle(surf, s.color, (x + px(6), y + px(11)), px(6))
        r = fonts.draw(surf, s.name, (x + px(20), y), theme.TEXT, fonts.title)
        tips.add(r.union(pygame.Rect(x, y, px(20), r.h)),
                 "The selected satellite, drawn in its colour. Ctrl+E renames it or changes "
                 "its mass and drag area.", "Ctrl+E")
        status_col = theme.GOOD if s.status == ACTIVE else theme.BAD
        r = fonts.draw(surf, s.status.upper(), (self.rect.right - px(12), y + px(4)), status_col,
                       fonts.small, "topright")
        tips.add(r, STATUS_TIPS.get(s.status, ""))
        for k, (name, r) in enumerate(zip(self.TABS, self.tab_rects, strict=True)):
            on = k == self.tab
            pygame.draw.rect(surf, theme.ACCENT_DARK if on else theme.FIELD, r,
                             border_radius=px(5))
            if on:
                pygame.draw.rect(surf, theme.ACCENT, r, 1, border_radius=px(5))
            fonts.draw(surf, name, r.center, theme.TEXT if on else theme.DIM, fonts.small, "center")
            tips.add(r, TAB_TIPS[k], "Q")
        q = fonts.draw(surf, "Q", (self.rect.right - px(12), self.tab_rects[0].centery),
                       theme.FAINT, fonts.small, "midright")
        tips.add(q.inflate(px(8), px(8)), "Q switches tab; PgUp / PgDn / Home / End or the wheel "
                                          "scroll it.", "Q")
        body = self.body
        tips.add(body, f"Details of {s.name}. {TAB_TIPS[self.tab]}\nQ switches tab; the wheel "
                       "or PgUp / PgDn scroll it.", "Q")
        top = self.scroll[self.tab] = max(0, min(self.scroll[self.tab], self.content_h - body.h))
        clip = surf.get_clip()
        surf.set_clip(body)
        y0, width = body.y + px(2) - top, px(RIGHT_W) - px(24)
        asc = sim.ascent_of(i) if sim.ascents else None
        if self.tab == 0 and asc is not None:
            y = draw_ascent_tab(surf, x, y0, width, app, i, asc)
        elif self.tab == 0:
            y = draw_orbit_tab(surf, x, y0, width, app, i)
        else:
            y = y0
            for title, rows in self.report(i):
                y = draw_section(surf, fonts, x, y, width, title, rows)
        self.content_h = y - y0
        surf.set_clip(clip)
        track, thumb = self.scrollbar()
        if thumb is not None:
            pygame.draw.rect(surf, theme.FIELD, track, border_radius=px(3))
            pygame.draw.rect(surf, theme.ACCENT if self._drag is not None else theme.SCROLL_THUMB,
                             thumb, border_radius=px(3))
            tips.add(track.inflate(px(10), 0), "Drag the bar or click the track to scroll; "
                                               "the wheel and PgUp / PgDn work too.")
        self._closest(surf, x, self.rect.bottom - px(22))

    def _closest(self, surf, x, y):
        """Footer line naming the closest pair of satellites."""
        sim, fonts = self.app.sim, self.app.fonts
        d, a, b = sim.closest
        if a >= 0 and math.isfinite(d):
            col = theme.BAD if d < sim.conjunction_km else theme.FAINT
            # the distance stays whole on the right; the pair's names shorten to fit
            r = fonts.draw(surf, f"{d:,.1f} km", (self.rect.right - px(12), y), col, fonts.small,
                           "topright")
            pair = fonts.fit(f"Closest: {sim.sats[a].name} - {sim.sats[b].name}",
                             r.x - px(10) - x, fonts.small)
            left = fonts.draw(surf, pair, (x, y), col, fonts.small)
            tips.add(left.union(r), f"The two satellites nearest each other now: "
                                    f"{sim.sats[a].name} and {sim.sats[b].name}, {d:,.1f} km "
                                    f"apart. It turns red below the close-approach alert "
                                    f"distance ({sim.conjunction_km:g} km, set in Physics).")


# --- Bottom: event log ----------------------------------------------------------------------------

class EventLog:
    """Bottom panel: the latest simulation events, newest at the bottom."""

    def __init__(self, app):
        self.app = app
        self.rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, w, h):
        """Span the view between the side panels, at the bottom of the window."""
        x0, x1, _ = self.app.center_span()
        self.rect = pygame.Rect(x0, h - px(LOG_H) - px(GAP), x1 - x0, px(LOG_H))

    def draw(self, surf):
        """The latest events that fit, newest at the bottom."""
        sim, fonts = self.app.sim, self.app.fonts
        self.layout(*surf.get_size())
        theme.panel(surf, self.rect)
        tips.add(self.rect, "The simulation's log: launches, burns, eclipses, station passes "
                            "and alerts, newest at the bottom.")
        x = self.rect.x + px(10)
        fonts.draw(surf, "EVENTS", (x, self.rect.y + px(6)), theme.FAINT, fonts.small)
        y = self.rect.bottom - px(20)
        tx = self.rect.x + px(130)
        text_w = self.rect.right - px(10) - tx
        for ev in reversed(sim.events[-7:]):
            col = theme.EVENT_COLORS.get(ev.kind, theme.DIM)
            fonts.draw(surf, f"{'T+' + format_duration(ev.t):>15}", (x, y), theme.FAINT,
                       fonts.small)
            fonts.draw(surf, fonts.fit(ev.text, text_w, fonts.small), (tx, y), col, fonts.small)
            tips.add(pygame.Rect(x, y, self.rect.right - px(10) - x, px(16)),
                     f"T+{format_duration(ev.t)}: {ev.text}\n{glossary.EVENTS.get(ev.kind, '')}")
            y -= px(16)
            if y < self.rect.y + px(22):
                break


HELP = [
    ("Mouse", ""),
    ("Drag  Wheel", "orbit the camera / zoom"),
    ("Rest on a part", "a tip says what it is and does"),
    ("Click satellite", "select it (list or 3-D view)"),
    ("Keyboard", ""),
    ("Arrows  + -", "rotate / zoom the camera"),
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
    ("Q  PgUp PgDn", "right panel: switch tab / scroll it"),
    ("M  G", "ground-track map, telemetry plot"),
    ("A  W  B  N", "add satellite, Walker, manoeuvre, station"),
    ("U", "launch a rocket from anywhere on Earth"),
    ("P", "physics & integrator settings"),
    ("Ctrl+N  O  S", "start screen / scenarios / save snapshot"),
    ("Ctrl+E  Del", "edit / delete selected satellite"),
    ("Ctrl+R  Ctrl+Q", "reset scenario / quit"),
    ("I", "hide / show panels"),
    ("F11  F12", "fullscreen / screenshot to screenshots/"),
    ("H or F1  Esc", "this help / close it (or leave fullscreen)"),
]   # draw_help needs 54 + 20 px per row: keep it within the 600 px minimum window height


def draw_help(surf, app):
    """The controls overlay (key H), centred; returns its rect."""
    fonts = app.fonts
    w, h = surf.get_size()
    line = px(20)
    rect = pygame.Rect(0, 0, px(520), px(34) + line * len(HELP) + px(20))
    rect.center = (w // 2, h // 2)
    theme.panel(surf, rect, (14, 20, 36, 245), theme.ACCENT)
    tips.block(rect)                     # it explains itself; a tip would hide the text
    y = rect.y + px(14)
    fonts.draw(surf, "Controls", (rect.x + px(18), y), theme.TEXT, fonts.title)
    y += px(30)
    for key, desc in HELP:
        if not desc:
            fonts.draw(surf, key.upper(), (rect.x + px(18), y + px(2)), theme.ACCENT, fonts.small)
        else:
            fonts.draw(surf, key, (rect.x + px(30), y + line // 2), theme.TEXT, fonts.mono,
                       "midleft")
            fonts.draw(surf, desc, (rect.x + px(200), y + line // 2), theme.DIM, fonts.ui,
                       "midleft")
        y += line
    return rect
