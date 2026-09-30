"""Launch front end.

* the Launch dialog: site (preset, typed, or clicked on the map), vehicle,
  payload, target orbit, timing and ascent profile, with a live preview - the
  whole ascent is flown on paper while you type: ground track, altitude
  profile with staging events, the orbit reached and the delta-v budget;
* the Vehicle dialog: stages (thrust, Isp, propellant, dry mass), fairing,
  aerodynamics and acceleration limit;
* the Ascent view that replaces the Orbit tab while the selected vehicle is on
  its pad or in powered flight, and a LAUNCH section for the Telemetry tab.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pygame

from ..constants import R_EARTH
from ..elements import kepler_propagate, rv2coe
from ..frames import ecef_to_geodetic, eci_to_ecef
from ..launch import (
    G0_M,
    LAUNCH_SITES,
    PHASE_LABELS,
    VEHICLES,
    LaunchSpec,
    Stage,
    Vehicle,
    vehicle_preset,
)
from ..simulation import ACTIVE
from . import theme
from .groundtrack import GroundTrackView
from .orbitpanel import countdown, draw_section, num
from .widgets import FieldSpec as F
from .widgets import FormDialog

WHEN = {"Now": "now", "After delay": "delay", "Window: RAAN": "raan",
        "Window: plane of satellite": "plane"}
GUIDANCE = {"Closed loop to orbit": "orbit", "Open loop (gravity turn)": "open"}
DIRECTION = {"Northbound": "north", "Southbound": "south"}
CUSTOM_SITE = "Custom location"
CUSTOM_VEHICLE = "Custom"
PLANNED = (120, 140, 175)
TRACK = (255, 196, 90)
ORBIT = (120, 210, 255)


def _clock(met: float) -> str:
    """Mission clock ``T-HH:MM:SS`` / ``T+HH:MM:SS``."""
    sign = "T-" if met < 0 else "T+"
    s = int(round(abs(met)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def _km(x) -> str:
    return num(x, prec=0)


def _wrap(font, text: str, width: int) -> list[str]:
    """Split text into lines no wider than ``width`` pixels (continuations indented)."""
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}" if cur else word
        if cur and font.size(trial)[0] > width:
            lines.append(cur)
            cur = "  " + word
        else:
            cur = trial
    return lines + [cur]


# --- Live preview side panel ------------------------------------------------------------

class LaunchPreview:
    """Right-hand panel of the Launch dialog: the pad map, the planned altitude
    profile and a text summary, re-planned shortly after the form stops changing."""

    width = 470
    min_height = 600
    DEBOUNCE = 0.25

    def __init__(self, app, vehicle: Vehicle):
        self.app = app
        self.vehicle = vehicle
        self.map = GroundTrackView(app.renderer.earth)
        self.plan = None
        self.spec = None
        self.error = ""
        self.key = None
        self.dirty_at = None
        self.map_rect = pygame.Rect(0, 0, 0, 0)
        self._orbit_track = None

    def mark_dirty(self):
        """Schedule a re-plan ``DEBOUNCE`` s from now."""
        self.dirty_at = time.monotonic()

    def refresh(self, dlg):
        """Build the spec from the form and fly it (cached per spec)."""
        self.dirty_at = None
        try:
            spec = build_spec(dlg, self.vehicle)
            key = json.dumps(spec.to_dict(), sort_keys=True, default=str)
            if key != self.key:
                self.plan = self.app.sim.plan(spec)
                self.key = key
                self._orbit_track = None
            self.spec = spec
            self.error = ""
        except Exception as exc:     # keep the dialog alive on any bad input
            self.plan, self.key, self.error = None, None, str(exc)

    # --- input ---
    def handle(self, ev, dlg) -> bool:
        """A click on the map moves the pad there."""
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.map_rect.collidepoint(ev.pos)):
            r = self.map_rect
            lon = (ev.pos[0] - r.x) / (r.w - 1) * 360.0 - 180.0
            lat = 90.0 - (ev.pos[1] - r.y) / (r.h - 1) * 180.0
            dlg.set("site", CUSTOM_SITE)
            dlg.set("lat", f"{lat:.3f}")
            dlg.set("lon", f"{lon:.3f}")
            dlg.set("salt", "0")
            dlg.changed("lat")
            return True
        return False

    # --- drawing ---
    def draw(self, surf, rect, dlg):
        if self.dirty_at is not None and time.monotonic() - self.dirty_at > self.DEBOUNCE:
            self.refresh(dlg)
        fonts = self.app.fonts
        x, y, w = rect.x + 6, rect.y, rect.w - 18
        fonts.draw(surf, "LAUNCH SITE - click the map to launch from anywhere", (x, y), theme.FAINT,
                   fonts.small)
        y += 18
        mh = min(w // 2, max(120, rect.h - 330))
        mw = mh * 2
        self.map_rect = pygame.Rect(x, y, mw, mh)
        self._draw_map(surf, self.map_rect, dlg)
        y += mh + 8
        ph = max(90, min(150, rect.bottom - y - 130))
        prof = pygame.Rect(x, y, w, ph)
        flight = self.plan.flight if self.plan else None
        draw_profile(surf, prof, fonts, flight.samples if flight else [],
                     flight.stage_marks if flight else [], None, None, TRACK)
        y += ph + 8
        for text, col in self._lines():
            for line in _wrap(fonts.small, text, w):
                if y > rect.bottom - 14:
                    return
                fonts.draw(surf, line, (x, y), col, fonts.small)
                y += 16

    def _draw_map(self, surf, r, dlg):
        """Launch sites, the pad, the planned ascent track and the first orbit."""
        surf.blit(self.map.background(r.size), r.topleft)
        box = (r.x, r.y, r.w, r.h)
        clip = surf.get_clip()
        surf.set_clip(r)
        for lat, lon, _ in LAUNCH_SITES.values():
            sx, sy = self.map.xy(lat, lon, box)
            pygame.draw.circle(surf, (150, 160, 190), (int(sx), int(sy)), 2)
        if self.plan is not None:
            f = self.plan.flight
            track = self._orbit()
            if track is not None:
                self.map.polyline(surf, ORBIT, track[0], track[1], box)
            if len(f.track) > 1:
                lat, lon = np.array(f.track).T
                self.map.polyline(surf, TRACK, lat, lon, box, 2)
        try:
            v = dlg.values()
            sx, sy = self.map.xy(v["lat"], v["lon"], box)
            sx, sy = int(sx), int(sy)
            pygame.draw.polygon(surf, (255, 255, 255),
                                [(sx, sy - 7), (sx - 5, sy + 4), (sx + 5, sy + 4)])
            pygame.draw.polygon(surf, (255, 90, 60),
                                [(sx, sy - 5), (sx - 3, sy + 3), (sx + 3, sy + 3)])
        except (ValueError, KeyError):
            pass                # the pad coordinates do not parse yet
        surf.set_clip(clip)
        pygame.draw.rect(surf, theme.PANEL_EDGE, r, 1)

    def _orbit(self):
        """Ground track of the first revolution after separation."""
        if self._orbit_track is not None or self.plan is None:
            return self._orbit_track
        f = self.plan.flight
        r, v = np.array(f.y[:3]), np.array(f.y[3:])
        el = rv2coe(r, v)
        if el.e >= 1 or el.rp < R_EARTH:
            return None
        ts = np.linspace(0.0, el.period, 240)
        rr, _ = kepler_propagate(np.repeat(r[None], len(ts), 0), np.repeat(v[None], len(ts), 0),
                                 ts)
        th = self.app.sim.clock.gmst(f.t + ts)
        lat, lon, _ = ecef_to_geodetic(eci_to_ecef(rr, th))
        self._orbit_track = (np.degrees(lat), np.degrees(lon))
        return self._orbit_track

    def _lines(self):
        """(text, colour) summary lines: vehicle, lift-off, outcome and delta-v budget."""
        out = []
        veh = self.vehicle
        try:
            payload = float(self.spec.payload_mass) if self.spec else 0.0
            summary = veh.stage_summary(payload)
            n = len(veh.stages)
            out.append((f"{veh.name}: {n} stage{'s' if n != 1 else ''}, lift-off "
                        f"{veh.liftoff_mass(payload) / 1000:,.1f} t, T/W {summary[0][3]:.2f}, "
                        f"ideal dV {sum(s[1] for s in summary):.2f} km/s", theme.DIM))
        except (ValueError, ZeroDivisionError, IndexError):
            pass                # an incomplete vehicle has no summary
        if self.error:
            out.append((self.error, theme.BAD))
            return out
        if self.plan is None:
            out.append(("planning...", theme.FAINT))
            return out
        p, f, sim = self.plan, self.plan.flight, self.app.sim
        wait = p.res.t0 - sim.t
        at = sim.clock.datetime(p.res.t0).strftime("%H:%M:%S")
        when = "Lift-off now" if wait < 1 else f"Lift-off in {countdown(wait)} ({at} UTC)"
        auto = self.spec is not None and self.spec.kick is None
        out.append((f"{when}, azimuth {math.degrees(p.res.azimuth) % 360:.1f} deg, "
                    f"kick {p.kick:.2f} deg{' (optimised)' if auto else ''}", theme.TEXT))
        ins = f.insertion or {}
        hp, ha, inc = ins.get("hp", math.nan), ins.get("ha", math.nan), ins.get("i", math.nan)
        if f.outcome == "orbit":
            head = "ORBIT" if self.spec.guidance == "orbit" else "Orbit at burnout"
            out.append((f"{head}: {_km(hp)} x {_km(ha)} km, i {inc:.2f} deg, "
                        f"cut-off {_clock(f.met)}", theme.GOOD))
            if self.spec.guidance == "orbit":
                out.append((f"Propellant left {ins.get('prop_left', 0):,.0f} kg in {f.stage.name}"
                            f"  |  max Q {f.max_q[0] / 1000:.1f} kPa at T+{f.max_q[1]:.0f} s",
                            theme.DIM))
        elif f.outcome in ("suborbital", "escape"):
            txt = (f"Sub-orbital: apogee {_km(ha)} km, burnout {_clock(f.met)}"
                   if f.outcome == "suborbital" else f"Escape trajectory, burnout {_clock(f.met)}")
            out.append((txt, theme.WARN))
        elif f.outcome == "short":
            out.append((f"FAILS: out of propellant at {_clock(f.met)} at {f.speed:.2f} of "
                        f"{p.res.v_target:.2f} km/s, perigee {_km(hp)} km", theme.BAD))
            out.append(("Lighten the payload, lower the orbit or add stages/propellant.",
                        theme.DIM))
        elif f.outcome == "crash":
            out.append((f"FAILS: vehicle falls back and hits the ground at {_clock(f.met)} - "
                        "try another kick", theme.BAD))
        elif f.outcome == "no_liftoff":
            out.append(("FAILS: thrust is below the weight - the vehicle cannot lift off",
                        theme.BAD))
        lo = f.losses
        out.append((f"dV {f.dv_ideal:.2f} km/s = gravity {lo['gravity']:.2f} + "
                    f"drag {lo['drag']:.2f} + steering {lo['steering']:.2f} + speed gained "
                    f"{f.dv_ideal - sum(lo.values()):.2f}", theme.FAINT))
        return out


def draw_profile(surf, rect, fonts, samples, marks, planned, now, color):
    """Altitude against downrange distance: ``samples`` rows are (t, alt,
    downrange, ...); ``planned`` is drawn dashed underneath; ``now`` is the
    (downrange, alt) of the vehicle."""
    pygame.draw.rect(surf, (8, 12, 24), rect, border_radius=6)
    pygame.draw.rect(surf, theme.PANEL_EDGE, rect, 1, border_radius=6)
    fonts.draw(surf, "ALTITUDE vs DOWNRANGE", (rect.x + 8, rect.y + 4), theme.FAINT, fonts.small)
    rows = [*(planned or []), *samples]
    if now is not None:
        rows.append((0.0, now[1], now[0]))
    if not rows:
        fonts.draw(surf, "no trajectory", rect.center, theme.FAINT, fonts.small, "center")
        return
    xmax = max(10.0, max(s[2] for s in rows)) * 1.05
    ymax = max(10.0, max(s[1] for s in rows)) * 1.12
    plot = pygame.Rect(rect.x + 44, rect.y + 20, rect.w - 54, rect.h - 38)

    def P(dr, alt):
        return (plot.x + dr / xmax * plot.w, plot.bottom - max(0.0, alt) / ymax * plot.h)

    pygame.draw.line(surf, theme.PANEL_EDGE, plot.bottomleft, plot.bottomright)
    pygame.draw.line(surf, theme.PANEL_EDGE, plot.bottomleft, plot.topleft)
    fonts.draw(surf, f"{ymax / 1.12:,.0f} km", (plot.x - 4, P(0, ymax / 1.12)[1]), theme.FAINT,
               fonts.small, "midright")
    fonts.draw(surf, "0", (plot.x - 4, plot.bottom), theme.FAINT, fonts.small, "midright")
    fonts.draw(surf, f"{xmax / 1.05:,.0f} km downrange", (plot.right, plot.bottom + 2), theme.FAINT,
               fonts.small, "topright")
    if planned and len(planned) > 1:
        pts = [P(s[2], s[1]) for s in planned]
        for a, b in zip(pts[::2], pts[1::2], strict=False):     # dashes: every other segment
            pygame.draw.line(surf, PLANNED, a, b)
    if len(samples) > 1:
        pygame.draw.lines(surf, color, False, [P(s[2], s[1]) for s in samples], 2)
    taken: list[pygame.Rect] = []
    for _, alt, dr, label in marks:
        px, py = P(dr, alt)
        pygame.draw.circle(surf, (255, 255, 255), (int(px), int(py)), 3)
        tw, th = fonts.small.size(label)
        # label left-above the event, else right-below; skip it if both collide
        for lx, ly in ((px - tw - 4, py - th - 2), (px + 5, py + 2), (px + 5, py - th - 2)):
            box = pygame.Rect(int(lx), int(ly), tw, th)
            if plot.contains(box) and box.collidelist(taken) < 0:
                taken.append(box)
                fonts.draw(surf, label, box.topleft, theme.DIM, fonts.small)
                break
    if now is not None:
        px, py = P(now[0], now[1])
        pygame.draw.circle(surf, (255, 255, 255), (int(px), int(py)), 5)
        pygame.draw.circle(surf, color, (int(px), int(py)), 3)


# --- Dialogs ----------------------------------------------------------------------------

def _is(key, *names):
    """Visibility predicate: field ``key`` holds one of ``names``."""
    return lambda raw: raw.get(key) in names


def _orbit(raw):
    """Visibility predicate: closed-loop guidance to an orbit is selected."""
    return GUIDANCE.get(raw.get("guidance")) == "orbit"


def build_spec(dlg, vehicle: Vehicle) -> LaunchSpec:
    """The :class:`LaunchSpec` described by the Launch dialog's fields."""
    v = dlg.values()
    guidance = GUIDANCE[v["guidance"]]
    timing = WHEN[v["when"]] if guidance == "orbit" else WHEN.get(v["when"], "now")
    if guidance == "open" and timing in ("raan", "plane"):
        raise ValueError("launch windows need closed-loop guidance to an orbit")
    kick = v["kick"]
    if isinstance(kick, str):
        if kick.strip().lower() not in ("auto", ""):
            raise ValueError("pitch kick: a number of degrees or 'auto'")
        kick = None
    site = v["site"] if v["site"] != CUSTOM_SITE else ""
    return LaunchSpec(
        name=v["name"] or "Payload", vehicle=vehicle.copy(), site=site,
        lat=v["lat"], lon=v["lon"], alt=v["salt"], timing=timing,
        delay=v.get("delay", 0.0) * 60.0, guidance=guidance,
        perigee_alt=v.get("hp", 400.0), apogee_alt=v.get("ha", 400.0),
        inclination=v.get("inc", 51.6), direction=DIRECTION[v.get("direction", "Northbound")],
        raan=v.get("raan", 0.0), target=v.get("target", ""), azimuth=v.get("azimuth", 90.0),
        vertical_time=v["vertical"], kick=kick, payload_mass=v["payload_mass"],
        payload_area=v["payload_area"], circularize=bool(v.get("circ", False)),
        track_stages=bool(v["track"]))


def launch_dialog(app):
    """The Launch dialog (key U) with its live preview."""
    sim = app.sim
    targets = [s.name for i, s in enumerate(sim.sats)
               if s.status == ACTIVE and sim.ascent_of(i) is None]
    default_vehicle = "Falcon 9 (approx.)"
    site0 = "Cape Canaveral SLC-40 (USA)"
    lat0, lon0, alt0 = LAUNCH_SITES[site0]
    n = sum(1 for s in sim.sats if s.family == s.name) + 1
    specs = [
        F("name", "Payload name", "text", f"Launch {n}"),
        F("vehicle", "Launch vehicle", "choice", default_vehicle, [*VEHICLES, CUSTOM_VEHICLE]),
        F("payload_mass", "Payload mass (kg)", "float", "5000"),
        F("payload_area", "Payload area (m^2)", "float", "10"),
        F("site", "Launch site", "choice", site0, [*LAUNCH_SITES, CUSTOM_SITE]),
        F("lat", "Latitude (deg)", "float", f"{lat0}"),
        F("lon", "Longitude (deg E)", "float", f"{lon0}"),
        F("salt", "Site altitude (km)", "float", f"{alt0}"),
        F("guidance", "Guidance", "choice", "Closed loop to orbit", list(GUIDANCE)),
        F("hp", "Target perigee alt (km)", "float", "400", visible=_orbit),
        F("ha", "Target apogee alt (km)", "float", "400", visible=_orbit),
        F("inc", "Inclination (deg)", "float", "51.6",
          visible=lambda r: _orbit(r) and r.get("when") != "Window: plane of satellite"),
        F("direction", "Direction over the pad", "choice", "Northbound", list(DIRECTION),
          visible=_orbit),
        F("azimuth", "Flight azimuth (deg from N)", "float", "90",
          visible=lambda r: not _orbit(r)),
        F("when", "Lift-off", "choice", "Now", list(WHEN)),
        F("delay", "Delay (min)", "float", "10", visible=_is("when", "After delay")),
        F("raan", "Plane RAAN (deg)", "float", "0",
          visible=lambda r: _orbit(r) and r.get("when") == "Window: RAAN"),
        F("target", "Into the plane of", "choice", targets[0] if targets else "(none)",
          targets or ["(none)"],
          visible=lambda r: _orbit(r) and r.get("when") == "Window: plane of satellite"),
        F("kick", "Pitch kick (deg or 'auto')", "floatx", "auto"),
        F("vertical", "Vertical rise (s)", "float", "10"),
        F("circ", "Circularise at apogee", "bool", False, visible=_orbit),
        F("track", "Keep spent stages as objects", "bool", True),
    ]
    preview = LaunchPreview(app, vehicle_preset(default_vehicle))

    def on_change(dlg, key):
        if key == "site" and dlg.widgets["site"].value in LAUNCH_SITES:
            lat, lon, alt = LAUNCH_SITES[dlg.widgets["site"].value]
            dlg.set("lat", lat)
            dlg.set("lon", lon)
            dlg.set("salt", alt)
        elif key in ("lat", "lon", "salt"):
            dlg.set("site", CUSTOM_SITE)
        elif key == "vehicle" and dlg.widgets["vehicle"].value in VEHICLES:
            preview.vehicle = vehicle_preset(dlg.widgets["vehicle"].value)
        preview.mark_dirty()

    def on_ok(v):
        spec = build_spec(dlg, preview.vehicle)
        preview.refresh(dlg)
        if preview.plan is not None and spec.kick is None:
            spec.kick = preview.plan.kick            # the optimised pitch-over, flown for real
        sat = sim.launch(spec)
        app.select(sim.sats.index(sat))
        asc = sim.ascent_of(app.selected)
        app.toast(f"{sat.name} on the pad" if asc is not None and asc.phase == "pad"
                  else f"{sat.name}: lift-off!")
        return None

    def edit_vehicle(dlg):
        try:
            payload = float(dlg.widgets["payload_mass"].value)
        except ValueError:
            payload = 0.0

        def done(vehicle):
            preview.vehicle = vehicle
            dlg.set("vehicle", CUSTOM_VEHICLE)
            preview.mark_dirty()
        vd = vehicle_dialog(app, preview.vehicle, payload, done)
        app.dialogs.append(vd)
        return None

    dlg = FormDialog(app, "Launch", specs, on_ok, "Launch", width=500, on_change=on_change,
                     extra_buttons=[("Vehicle...", edit_vehicle)], side=preview,
                     subtitle="Fly a rocket from any point on Earth - "
                              "the preview flies it first")
    dlg.preview = preview
    preview.refresh(dlg)
    return dlg


STAGE_FIELDS = [("name", "Name", "text"), ("thrust", "Thrust, vacuum (kN)", "float"),
                ("isp_vac", "Isp vacuum (s)", "float"),
                ("isp_sl", "Isp sea level (s, 0 = vac.)", "float"),
                ("propellant", "Propellant (t)", "float"), ("dry", "Dry mass (t)", "float")]
MAX_STAGES = 4


def vehicle_dialog(app, vehicle: Vehicle, payload: float, on_done):
    """Edit a vehicle's stages; ``on_done(Vehicle)`` receives the result."""
    stages = list(vehicle.stages) + [Stage(f"Stage {k + 1}", 100.0, 330.0, 0.0, 5000.0, 600.0)
                                     for k in range(len(vehicle.stages), MAX_STAGES)]
    counts = [str(k) for k in range(1, MAX_STAGES + 1)]
    specs = [
        F("vname", "Vehicle name", "text", vehicle.name),
        F("count", "Number of stages", "choice", str(len(vehicle.stages)), counts),
        F("edit", "Edit stage", "choice", "1", counts),
    ]
    for k, st in enumerate(stages, start=1):
        def vis(r, k=k):
            return r.get("edit") == str(k) and int(r.get("count", "1")) >= k

        for key, label, kind in STAGE_FIELDS:
            val = getattr(st, key)
            if key in ("propellant", "dry"):
                val = f"{val / 1000:g}"          # tonnes in the form
            elif isinstance(val, float):
                val = f"{val:g}"
            specs.append(F(f"s{k}_{key}", label, kind, val, visible=vis))
    specs += [
        F("fairing", "Fairing mass (kg)", "float", f"{vehicle.fairing:g}"),
        F("fairing_alt", "Fairing jettison alt (km)", "float", f"{vehicle.fairing_alt:g}"),
        F("diameter", "Diameter (m)", "float", f"{vehicle.diameter:g}"),
        F("cd", "Drag coefficient Cd", "float", f"{vehicle.cd:g}"),
        F("max_g", "Acceleration limit (g, 0 = none)", "float", f"{vehicle.max_g:g}"),
        F("coast", "Coast between stages (s)", "float", f"{vehicle.stage_coast:g}"),
    ]
    shown: dict = {}         # summary rows that have text (empty ones take no space)
    specs += [F(f"sum{k}", "", "info", "", visible=lambda r, k=k: bool(shown.get(k)))
              for k in range(MAX_STAGES + 1)]

    def build(dlg) -> Vehicle:
        """The vehicle described by the form (masses entered in tonnes)."""
        raw = dlg.raw()
        n = int(raw["count"])
        out = []
        for k in range(1, n + 1):
            vals = {}
            for key, label, kind in STAGE_FIELDS:
                txt = raw[f"s{k}_{key}"].strip()
                if kind == "text":
                    vals[key] = txt or f"Stage {k}"
                    continue
                try:
                    vals[key] = float(txt)
                except ValueError:
                    raise ValueError(f"stage {k}: '{label}' needs a number") from None
            vals["propellant"] *= 1000.0
            vals["dry"] *= 1000.0
            out.append(Stage(**vals))
        v = dlg.values()
        return Vehicle(v["vname"] or "Custom", out, v["fairing"], v["fairing_alt"], v["diameter"],
                       v["cd"], v["max_g"], v["coast"])

    def on_change(dlg, key):
        for k in range(MAX_STAGES + 1):
            dlg.info[f"sum{k}"] = ""
        try:
            veh = build(dlg)
            rows = veh.stage_summary(payload)
            for k, (m0, dv, tb, tw) in enumerate(rows):
                dlg.info[f"sum{k}"] = (f"{veh.stages[k].name[:12]}: ignition {m0 / 1000:,.1f} t, "
                                       f"burn {tb:.0f} s, T/W {tw:.2f}, dV {dv:.2f} km/s")
            dlg.info[f"sum{len(rows)}"] = (f"Total ideal dV {sum(r[1] for r in rows):.2f} km/s "
                                           f"with {payload:,.0f} kg payload "
                                           "(orbit needs ~9.3-9.8)")
            dlg.error = ""
        except (ValueError, ZeroDivisionError) as exc:
            dlg.error = str(exc)
        for k in range(MAX_STAGES + 1):
            shown[k] = dlg.info[f"sum{k}"]

    def on_ok(v):
        veh = build(dlg)
        if any(s.thrust <= 0 or s.isp_vac <= 0 or s.propellant <= 0 for s in veh.stages):
            return "every stage needs thrust, Isp and propellant"
        on_done(veh)
        return None

    dlg = FormDialog(app, "Launch vehicle", specs, on_ok, "Use vehicle", width=560,
                     on_change=on_change,
                     subtitle="Stages burn in order; thrust and Isp rise from sea level to vacuum")
    on_change(dlg, None)
    dlg.layout()
    return dlg


# --- Ascent view ------------------------------------------------------------------------------

def draw_ascent_tab(surf, x, y, w, app, i, asc) -> int:
    """The Orbit tab while satellite ``i`` is on its pad or climbing; returns the y
    below the content."""
    sim, fonts = app.sim, app.fonts
    sat = sim.sats[i]
    spec = asc.spec
    met = sim.t - asc.t0
    head = "ON THE PAD" if asc.phase == "pad" else PHASE_LABELS.get(asc.phase, asc.phase).upper()
    if asc.phase != "pad" and not asc.burning and not asc.released:
        head = "STAGING COAST"
    fonts.draw(surf, _clock(met), (x, y), theme.WARN if met < 0 else theme.GOOD, fonts.title)
    fonts.draw(surf, head, (x + w, y + 4), theme.TEXT, fonts.small, "topright")
    y += 28
    site = spec.site or f"{spec.lat:.2f}, {spec.lon:.2f}"
    fonts.draw(surf, f"{spec.vehicle.name} from {site}"[:52], (x, y), theme.DIM, fonts.small)
    y += 20
    st = asc.telemetry()
    planned = asc.planned
    draw_profile(surf, pygame.Rect(x, y, w, 150), fonts, asc.samples, asc.stage_marks,
                 planned[0] if planned else None,
                 None if asc.phase == "pad" else (st["downrange"], st["alt"]), sat.color)
    y += 158
    # stages
    fonts.draw(surf, "STAGES", (x, y), theme.ACCENT, fonts.small)
    y += 18
    for k, stg in enumerate(spec.vehicle.stages):
        if k < asc.k:
            frac, note, col = 0.0, "separated", theme.FAINT
        elif k == asc.k:
            frac = asc.prop / stg.propellant
            note = ("burning" if asc.burning else ("waiting" if asc.phase == "pad" else "coasting"))
            col = theme.WARN if asc.burning else theme.DIM
        else:
            frac, note, col = 1.0, "full", theme.DIM
        fonts.draw(surf, stg.name[:10], (x + 4, y), theme.TEXT if k == asc.k else theme.DIM,
                   fonts.small)
        bar = pygame.Rect(x + 90, y + 3, w - 180, 10)
        pygame.draw.rect(surf, theme.FIELD, bar, border_radius=3)
        if frac > 0:
            fill = bar.copy()
            fill.w = max(3, int(bar.w * frac))
            pygame.draw.rect(surf, col if k != asc.k else sat.color, fill, border_radius=3)
        fonts.draw(surf, f"{frac * 100:3.0f}% {note}", (x + w, y), col, fonts.small, "topright")
        y += 17
    y += 6
    el = st["el"]
    if el.e >= 1:
        orbit_now = "escape"
    elif el.rp > R_EARTH:
        orbit_now = f"{el.rp - R_EARTH:,.0f} x {el.ra - R_EARTH:,.0f} km"
    else:
        orbit_now = f"apogee {el.ra - R_EARTH:,.0f} km (sub-orbital)"
    if spec.guidance == "orbit":
        target = (f"{spec.perigee_alt:,.0f} x {spec.apogee_alt:,.0f} km, "
                  f"i {asc.res.inclination:.2f} deg")
    else:
        target = f"open loop, azimuth {spec.azimuth:.1f} deg"
    steering = f"{math.degrees(asc.res.azimuth) % 360:.1f} / {asc.kick_deg:.2f} deg"
    rows = [
        ("Altitude", f"{st['alt']:,.2f} km"),
        ("Downrange", f"{st['downrange']:,.1f} km"),
        ("Speed inertial / air", f"{st['speed']:.3f} / {st['air']:.3f} km/s"),
        ("Vertical speed", f"{st['vr'] * 1000:,.0f} m/s"),
        ("Dynamic pressure", f"{asc.q / 1000:.1f} kPa (max {asc.max_q[0] / 1000:.1f})"),
        ("Acceleration / throttle", f"{asc.accel / G0_M:.2f} g / {asc.throttle * 100:.0f}%"),
        ("Pitch above horizon", f"{st['pitch']:.1f} deg" if asc.burning else "-"),
        ("Vehicle mass", f"{asc.mass / 1000:,.2f} t"),
        ("Time to cut-off",
         f"{asc.tgo:.0f} s" if asc.phase == "guided" and math.isfinite(asc.tgo) else "-"),
        ("Orbit now", orbit_now),
        ("Target", target),
        ("Launch azimuth / kick", steering),
        ("dV spent (ideal)", f"{asc.dv_ideal:.3f} km/s"),
        ("  gravity / drag loss", f"{asc.losses['gravity']:.3f} / {asc.losses['drag']:.3f} km/s"),
        ("  steering loss", f"{asc.losses['steering']:.3f} km/s"),
    ]
    if asc.phase == "pad":
        rows = [("Lift-off in", countdown(asc.t0 - sim.t)),
                ("Lift-off at", sim.clock.datetime(asc.t0).strftime("%Y-%m-%d %H:%M:%S UTC")),
                ("Lift-off mass", f"{asc.mass / 1000:,.1f} t"),
                ("Target", target),
                ("Launch azimuth / kick", steering)]
        if planned and planned[2]:
            rows.append(("Planned outcome", planned[2]))
    return draw_section(surf, fonts, x, y, w, "ASCENT", rows)


def launch_rows(sim, i):
    """Rows for a LAUNCH section of the Telemetry tab, or None."""
    asc = sim.ascent_of(i)
    if asc is not None:
        stage = asc.stage.name
        return [("Phase", PHASE_LABELS.get(asc.phase, asc.phase)),
                ("Mission time", _clock(sim.t - asc.t0)),
                ("Stage", f"{stage} ({'burning' if asc.burning else 'idle'})"),
                ("Propellant in stage", f"{asc.prop:,.0f} kg")]
    rep = sim.sats[i].launch_report
    if not rep:
        return None
    rows = [("Vehicle", rep["vehicle"]),
            ("Site", rep["site"] or f"{rep['lat']:.2f}, {rep['lon']:.2f}"),
            ("Outcome", rep["outcome"]),
            ("Cut-off", _clock(rep["met"]))]
    if "hp" in rep:
        rows.append(("Insertion orbit", f"{_km(rep['hp'])} x {_km(rep['ha'])} km, "
                                        f"i {rep['i']:.2f}"))
        rows.append(("Propellant left", f"{rep['prop_left']:,.0f} kg"))
    rows += [("Max Q", f"{rep['max_q']:.1f} kPa at T+{rep['t_max_q']:.0f} s"),
             ("Ascent dV / losses",
              f"{rep['dv_ideal']:.2f} / {sum(rep['losses'].values()):.2f} km/s")]
    return rows
