"""Scenario-building dialogs: add satellites, Walker constellations,
manoeuvres, ground stations, physics settings and scenario files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..constants import MU_EARTH, R_EARTH
from ..elements import rv2coe
from ..forces import ForceModel
from ..integrators import METHODS
from ..maneuvers import Maneuver
from ..planner import (plan_bielliptic, plan_hohmann, plan_rendezvous, scan_rendezvous,
                       plane_change_summary)
from ..scenario import PRESETS, ConstellationSpec, GroundStation, orbit_state, walker_states
from ..simulation import PROPAGATORS
from ..timeutil import format_period
from .widgets import FieldSpec as F
from .widgets import FormDialog

# --- Add satellite -------------------------------------------------------------------------

MODES = ["Circular altitude", "Perigee / apogee", "Elements (a, e)", "Surface launch",
         "State vector (ECI)", "Geostationary slot", "TLE"]
_ELEM_MODES = MODES[:3]


def _mode(*names):
    return lambda raw: raw.get("mode") in names


def _orbit_from(v: dict) -> dict:
    mode = v["mode"]
    if mode in _ELEM_MODES:
        o = {"type": "elements", "i": v["i"], "raan": v["raan"], "argp": v["argp"], "nu": v["nu"]}
        if mode == "Circular altitude":
            o["altitude"] = v["alt"]
            o["argp"] = 0.0
        elif mode == "Perigee / apogee":
            o["perigee_alt"], o["apogee_alt"] = v["hp"], v["ha"]
        else:
            o["a"], o["e"] = v["a"], v["e"]
        return o
    if mode == "Surface launch":
        return {"type": "surface", "lat": v["lat"], "lon": v["lon"], "alt": v["salt"],
                "speed": v["speed"], "azimuth": v["azimuth"], "fpa": v["fpa"]}
    if mode == "State vector (ECI)":
        return {"type": "state", "r": [v["x"], v["y"], v["z"]], "v": [v["vx"], v["vy"], v["vz"]]}
    if mode == "Geostationary slot":
        return {"type": "geo", "lon": v["geolon"]}
    return {"type": "tle", "line1": v["line1"], "line2": v["line2"]}


def add_satellite_dialog(app):
    sim = app.sim
    specs = [
        F("name", "Name", "text", f"Sat {sim.n + 1}"),
        F("preset", "Preset", "choice", "Custom", ["Custom"] + list(PRESETS)),
        F("mode", "Initial condition", "choice", MODES[0], MODES),
        F("alt", "Altitude (km)", "float", "550", visible=_mode("Circular altitude")),
        F("hp", "Perigee altitude (km)", "float", "300", visible=_mode("Perigee / apogee")),
        F("ha", "Apogee altitude (km)", "float", "2000", visible=_mode("Perigee / apogee")),
        F("a", "Semi-major axis a (km)", "float", "8000", visible=_mode("Elements (a, e)")),
        F("e", "Eccentricity e", "float", "0.1", visible=_mode("Elements (a, e)")),
        F("i", "Inclination (deg or 'sso')", "floatx", "51.6", visible=_mode(*_ELEM_MODES)),
        F("raan", "RAAN (deg)", "float", "0", visible=_mode(*_ELEM_MODES)),
        F("argp", "Arg. of perigee (deg)", "float", "0", visible=_mode("Perigee / apogee", "Elements (a, e)")),
        F("nu", "True anomaly (deg)", "float", "0", visible=_mode(*_ELEM_MODES)),
        F("lat", "Latitude (deg)", "float", "28.5", visible=_mode("Surface launch")),
        F("lon", "Longitude (deg E)", "float", "-80.6", visible=_mode("Surface launch")),
        F("salt", "Altitude (km)", "float", "0", visible=_mode("Surface launch")),
        F("speed", "Ground-relative speed (km/s)", "float", "7.5", visible=_mode("Surface launch")),
        F("azimuth", "Azimuth (deg from N)", "float", "90", visible=_mode("Surface launch")),
        F("fpa", "Flight-path angle (deg)", "float", "0", visible=_mode("Surface launch")),
        F("x", "x (km)", "float", "7000", visible=_mode("State vector (ECI)")),
        F("y", "y (km)", "float", "0", visible=_mode("State vector (ECI)")),
        F("z", "z (km)", "float", "0", visible=_mode("State vector (ECI)")),
        F("vx", "vx (km/s)", "float", "0", visible=_mode("State vector (ECI)")),
        F("vy", "vy (km/s)", "float", "7.546", visible=_mode("State vector (ECI)")),
        F("vz", "vz (km/s)", "float", "0", visible=_mode("State vector (ECI)")),
        F("geolon", "Slot longitude (deg E)", "float", "-75", visible=_mode("Geostationary slot")),
        F("line1", "Line 1", "text", "", visible=_mode("TLE"), wide=True),
        F("line2", "Line 2", "text", "", visible=_mode("TLE"), wide=True),
        F("count", "Copies along orbit", "int", "1"),
        F("mass", "Mass (kg)", "float", "500"),
        F("area", "Area (m^2)", "float", "5"),
        F("cd", "Drag coefficient Cd", "float", "2.2"),
        F("preview", "", "info", ""),
    ]

    def on_change(dlg, key):
        if key == "preset":
            _apply_preset(dlg)
        _preview(dlg)

    def on_ok(v):
        orbit = _orbit_from(v)
        props = dict(mass=v["mass"], area=v["area"], cd=v["cd"])
        if min(props.values()) <= 0 or v["mass"] <= 0:
            return "mass, area and coefficients must be positive"
        added = sim.add_satellites(orbit, v["name"] or "Sat", count=max(1, v["count"]), **props)
        app.select(sim.n - len(added))
        return None

    dlg = FormDialog(app, "Add satellite", specs, on_ok, "Add", width=560, on_change=on_change,
                     subtitle="Tab moves between fields - click or scroll a choice to change it")
    _preview(dlg)
    return dlg


def _apply_preset(dlg):
    name = dlg.widgets["preset"].value
    if name not in PRESETS:
        return
    p = PRESETS[name]
    o = p["orbit"]
    dlg.set("name", name.split(" (")[0])
    for k in ("mass", "area", "cd"):
        if k in p:
            dlg.set(k, p[k])
    t = o["type"]
    if t == "elements":
        if "altitude" in o:
            dlg.set("mode", "Circular altitude")
            dlg.set("alt", o["altitude"])
        elif "perigee_alt" in o:
            dlg.set("mode", "Perigee / apogee")
            dlg.set("hp", o["perigee_alt"])
            dlg.set("ha", o["apogee_alt"])
        else:
            dlg.set("mode", "Elements (a, e)")
            dlg.set("a", o["a"])
            dlg.set("e", o["e"])
        for k in ("i", "raan", "argp", "nu"):
            dlg.set(k, o.get(k, 0))
    elif t == "surface":
        dlg.set("mode", "Surface launch")
        for k, key in (("lat", "lat"), ("lon", "lon"), ("alt", "salt"), ("speed", "speed"),
                       ("azimuth", "azimuth"), ("fpa", "fpa")):
            dlg.set(key, o.get(k, 0))
    elif t == "geo":
        dlg.set("mode", "Geostationary slot")
        dlg.set("geolon", o.get("lon", 0))


def _preview(dlg):
    try:
        v = dlg.values()
        r, vv = orbit_state(_orbit_from(v), dlg.app.sim.clock, dlg.app.sim.t)
        el = rv2coe(r, vv)
        if el.e < 1:
            txt = (f"a {el.a:,.0f} km  e {el.e:.4f}  i {math.degrees(el.i):.2f} deg  "
                   f"hp {el.rp - R_EARTH:,.0f} / ha {el.ra - R_EARTH:,.0f} km  T {format_period(el.period)}")
        else:
            vinf = math.sqrt(max(0.0, 2 * el.energy))
            txt = f"open orbit: e {el.e:.3f}, v_inf {vinf:.2f} km/s, hp {el.rp - R_EARTH:,.0f} km"
        dlg.info["preview"] = txt
    except Exception:
        dlg.info["preview"] = ""


# --- Walker constellation ------------------------------------------------------------------

def walker_dialog(app):
    sim = app.sim
    specs = [
        F("name", "Name", "text", "Walker"),
        F("altitude", "Altitude (km)", "float", "1200"),
        F("inclination", "Inclination (deg)", "float", "53"),
        F("total", "Total satellites t", "int", "24"),
        F("planes", "Planes p", "int", "6"),
        F("phasing", "Phasing f (0..p-1)", "int", "1"),
        F("pattern", "Pattern", "choice", "delta", ["delta", "star"]),
        F("raan0", "First plane RAAN (deg)", "float", "0"),
        F("mass", "Mass each (kg)", "float", "260"),
        F("preview", "", "info", ""),
    ]

    def on_change(dlg, key):
        try:
            v = dlg.values()
            per = v["total"] / max(1, v["planes"])
            a = R_EARTH + v["altitude"]
            dlg.info["preview"] = (f"{v['pattern']} {v['inclination']:.1f}: {v['total']}/{v['planes']}/"
                                   f"{v['phasing']} - {per:g} per plane, T "
                                   f"{format_period(2 * math.pi * math.sqrt(a ** 3 / MU_EARTH))}")
        except Exception:
            dlg.info["preview"] = ""

    def on_ok(v):
        c = ConstellationSpec(name=v["name"] or "Walker", altitude=v["altitude"],
                              inclination=v["inclination"], total=v["total"], planes=v["planes"],
                              phasing=v["phasing"], pattern=v["pattern"], raan0=v["raan0"],
                              mass=v["mass"])
        if c.total <= 0 or c.planes <= 0:
            return "need at least one satellite and one plane"
        states = walker_states(c)
        from ..scenario import palette_color
        color = palette_color(sim.n)
        props = dict(mass=c.mass, area=c.area, cd=c.cd)
        n0 = sim.n
        sim.add_many([(name, color, props, r, v_) for name, r, v_ in states])
        app.select(n0)
        return None

    dlg = FormDialog(app, "Walker constellation", specs, on_ok, "Create", on_change=on_change,
                     subtitle="i:t/p/f - delta spreads planes over 360 deg, star over 180 deg")
    on_change(dlg, None)
    return dlg


# --- Manoeuvres ------------------------------------------------------------------------------

KINDS = {"Impulse (V/N/B components)": "impulse", "Hohmann transfer": "hohmann",
         "Bi-elliptic transfer": "bielliptic", "Circularize": "circularize",
         "Plane change": "plane_change", "Rendezvous (Lambert)": "rendezvous",
         "Finite burn (thrust)": "finite"}
TIMINGS = {"Now": "now", "After delay": "delay", "Next periapsis": "periapsis",
           "Next apoapsis": "apoapsis", "Ascending node": "ascending_node",
           "Descending node": "descending_node"}
DIRECTIONS = {"Prograde": (1, 0, 0), "Retrograde": (-1, 0, 0), "Normal": (0, 1, 0),
              "Anti-normal": (0, -1, 0), "Radial out": (0, 0, 1), "Radial in": (0, 0, -1)}


def _kind(*names):
    return lambda raw: KINDS.get(raw.get("kind")) in names


def maneuver_dialog(app):
    sim = app.sim
    # a payload still riding its rocket cannot manoeuvre until it separates
    names = [s.name for i, s in enumerate(sim.sats) if s.status == "active" and sim.ascent_of(i) is None]
    if not names:
        app.toast("No active satellites to manoeuvre")
        return None
    cur = sim.sats[app.selected].name if 0 <= app.selected < sim.n else names[0]
    other = next((n for n in names if n != cur), cur)
    specs = [
        F("sat", "Satellite", "choice", cur, names),
        F("kind", "Manoeuvre", "choice", "Hohmann transfer", list(KINDS)),
        F("timing", "Execute", "choice", "Now", list(TIMINGS)),
        F("delay", "Delay (s)", "float", "600", visible=lambda r: r.get("timing") == "After delay"),
        F("dv_v", "dV prograde V (m/s)", "float", "100", visible=_kind("impulse")),
        F("dv_n", "dV normal N (m/s)", "float", "0", visible=_kind("impulse")),
        F("dv_b", "dV binormal B (m/s)", "float", "0", visible=_kind("impulse")),
        F("frame", "Frame", "choice", "VNB", ["VNB", "RSW", "ECI"], visible=_kind("impulse")),
        F("target_alt", "Target altitude (km)", "float", "35786", visible=_kind("hohmann", "bielliptic")),
        F("rb_alt", "Intermediate apoapsis (km)", "float", "100000", visible=_kind("bielliptic")),
        F("delta_i", "Inclination change (deg)", "float", "-10", visible=_kind("plane_change")),
        F("target", "Target satellite", "choice", other, names, visible=_kind("rendezvous")),
        F("auto_tof", "Cheapest safe time of flight", "bool", True, visible=_kind("rendezvous")),
        F("tof", "Time of flight (min)", "float", "45",
          visible=lambda r: KINDS.get(r.get("kind")) == "rendezvous" and not r.get("auto_tof")),
        F("thrust", "Thrust (N)", "float", "400", visible=_kind("finite")),
        F("duration", "Burn duration (s)", "float", "300", visible=_kind("finite")),
        F("direction", "Direction", "choice", "Prograde", list(DIRECTIONS), visible=_kind("finite")),
        F("isp", "Specific impulse Isp (s)", "float", "320"),
        F("preview", "", "info", ""),
    ]

    scan_cache: dict = {}

    def build(v):
        kind = KINDS[v["kind"]]
        timing = TIMINGS[v["timing"]]
        delay = v.get("delay", 0.0)
        sat = v["sat"]
        common = dict(timing=timing, delay=delay, isp=v["isp"])
        if kind == "hohmann":
            burns, summary = plan_hohmann(sim, sat, v["target_alt"], timing, delay)
        elif kind == "bielliptic":
            burns, summary = plan_bielliptic(sim, sat, v["rb_alt"], v["target_alt"], timing, delay)
        elif kind == "rendezvous":
            if v.get("auto_tof"):
                key = (sat, v["target"], timing, delay, round(sim.t, 1))
                if key not in scan_cache:
                    scan_cache[key] = scan_rendezvous(sim, sat, v["target"], steps=60,
                                                      timing=timing, delay=delay)[0]
                tof = scan_cache[key]
            else:
                tof = v["tof"] * 60.0
            burns, summary = plan_rendezvous(sim, sat, v["target"], tof, timing, delay)
            summary = f"TOF {tof / 60:.1f} min: " + summary
        elif kind == "impulse":
            dv = (v["dv_v"] / 1000, v["dv_n"] / 1000, v["dv_b"] / 1000)
            burns = [Maneuver(sat, "impulse", dv=dv, frame=v["frame"], **common)]
            summary = f"|dV| {np.linalg.norm(dv) * 1000:.1f} m/s"
        elif kind == "circularize":
            burns = [Maneuver(sat, "circularize", **common)]
            summary = "burn computed from the state at execution"
        elif kind == "plane_change":
            burns = [Maneuver(sat, "plane_change", delta_i=v["delta_i"], **common)]
            summary = plane_change_summary(sim, sat, v["delta_i"], timing, delay)
        else:
            burns = [Maneuver(sat, "finite", thrust=v["thrust"], duration=v["duration"],
                              dv=DIRECTIONS[v["direction"]], frame="VNB", **common)]
            m = sim.sats[sim.index_of(sat)].mass
            mf = m - v["thrust"] / (v["isp"] * 9.80665) * v["duration"]
            dvm = v["isp"] * 9.80665 * math.log(m / mf) if mf > 0 else float("inf")
            summary = f"dV ~{dvm:.1f} m/s, accel {v['thrust'] / m * 1000:.2f} mm/s^2"
        for b in burns:
            b.isp = v["isp"]
        return burns, summary

    def on_change(dlg, key):
        try:
            _, summary = build(dlg.values())
            dlg.info["preview"] = summary
        except Exception as exc:
            dlg.info["preview"] = f"({exc})"

    def on_ok(v):
        burns, _ = build(v)
        for b in burns:
            sim.schedule(b)
        return None

    dlg = FormDialog(app, "Plan manoeuvre", specs, on_ok, "Schedule", width=560, on_change=on_change,
                     subtitle="VNB: V prograde, N orbit normal, B = V x N (radial out on circular orbits)")
    on_change(dlg, None)
    return dlg


# --- Ground station -------------------------------------------------------------------------

def station_dialog(app):
    specs = [
        F("name", "Name", "text", f"Station {len(app.sim.stations) + 1}"),
        F("lat", "Latitude (deg)", "float", "51.48"),
        F("lon", "Longitude (deg E)", "float", "0.0"),
        F("alt", "Altitude (km)", "float", "0.05"),
        F("min_el", "Minimum elevation (deg)", "float", "10"),
    ]

    def on_ok(v):
        if not -90 <= v["lat"] <= 90:
            return "latitude must be within +/-90 deg"
        app.sim.add_station(GroundStation(v["name"] or "Station", v["lat"], v["lon"], v["alt"], v["min_el"]))
        return None

    return FormDialog(app, "Add ground station", specs, on_ok, "Add")


# --- Physics settings -----------------------------------------------------------------------

def physics_dialog(app):
    sim = app.sim
    fm = sim.forces
    it = sim.integrator
    specs = [
        F("propagator", "Propagator", "choice", sim.propagator, list(PROPAGATORS)),
        F("method", "Integrator (Cowell)", "choice", it.method, list(METHODS)),
        F("rtol", "Relative tolerance", "float", f"{it.rtol:g}"),
        F("atol", "Absolute tolerance", "float", f"{it.atol:g}"),
        F("h_max", "Max step (s)", "float", f"{it.h_max:g}"),
        F("h_fixed", "Fixed step rk4/leapfrog (s)", "float", f"{it.h_fixed:g}"),
        F("j2", "J2 oblateness", "bool", fm.j2),
        F("j3", "J3", "bool", fm.j3),
        F("j4", "J4", "bool", fm.j4),
        F("drag", "Atmospheric drag", "bool", fm.drag),
        F("density_scale", "Density scale (solar activity)", "float", f"{fm.density_scale:g}"),
        F("record_dt", "Trail sample interval (s)", "float", f"{sim.record_dt:g}"),
        F("conj", "Close-approach alert (km)", "float", f"{sim.conjunction_km:g}"),
    ]

    def on_ok(v):
        if v["rtol"] <= 0 or v["atol"] <= 0 or v["h_max"] <= 0 or v["h_fixed"] <= 0:
            return "tolerances and steps must be positive"
        sim.forces = ForceModel(j2=v["j2"], j3=v["j3"], j4=v["j4"], drag=v["drag"],
                                density_scale=v["density_scale"])
        sim.set_propagator(v["propagator"])
        it.method = v["method"]
        it.rtol, it.atol, it.h_max, it.h_fixed = v["rtol"], v["atol"], v["h_max"], v["h_fixed"]
        it.h = min(it.h, it.h_max)
        sim.record_dt = max(0.1, v["record_dt"])
        sim.conjunction_km = v["conj"]
        for s in sim.sats:
            s.energy_ref = None       # the conserved quantity changed with the model
        sim.log(f"Physics: {sim.forces.label()}, {sim.propagator}/{it.method}")
        return None

    return FormDialog(app, "Physics & integrator", specs, on_ok, "Apply", width=520,
                      subtitle="Cowell integrates every enabled force; kepler / j2mean are analytic")


# --- Scenario files ---------------------------------------------------------------------------

def scenario_dialog(app):
    files = app.scenario_files()
    labels = [str(p.relative_to(app.root)) if p.is_relative_to(app.root) else str(p) for p in files]
    current = app.scenario_path
    cur_label = (str(current.relative_to(app.root)) if current and current.is_relative_to(app.root)
                 else (labels[0] if labels else ""))
    specs = [
        F("file", "Scenario file", "choice", cur_label, labels or ["(none found)"]),
        F("save_name", "Save snapshot as", "text", "my_scenario"),
        F("desc", "", "info", ""),
    ]
    lookup = dict(zip(labels, files))

    def describe(dlg, key=None):
        p = lookup.get(dlg.widgets["file"].value)
        if p:
            try:
                import json
                d = json.loads(Path(p).read_text(encoding="utf-8"))
                dlg.info["desc"] = (d.get("description") or d.get("name", ""))[:88]
            except Exception:
                dlg.info["desc"] = ""

    def on_ok(v):
        p = lookup.get(v["file"])
        if not p:
            return "no scenario selected"
        app.load_scenario(p)
        return None

    def save(dlg):
        v = dlg.values()
        name = "".join(ch for ch in v["save_name"] if ch.isalnum() or ch in "-_ ").strip() or "snapshot"
        path = app.user_scenario_dir / f"{name}.json"
        app.sim.snapshot_scenario(name).save(path)
        app.scenario_path = path
        app.toast(f"Saved {path.name}")
        dlg.close()

    def export(dlg):
        from ..batch import export_history_csv
        path = app.user_scenario_dir / "exports" / f"{app.sim.scenario.name.replace(' ', '_')}_history.csv"
        n = export_history_csv(app.sim, path)
        app.toast(f"Exported {n} rows to {path}")
        dlg.close()

    def reset(dlg):
        app.reset()
        dlg.close()

    dlg = FormDialog(app, "Scenarios", specs, on_ok, "Load", width=620, on_change=describe,
                     extra_buttons=[("Save snapshot", save), ("Export CSV", export),
                                    ("Reset current", reset)],
                     subtitle="Load a scenario, save the current state (epoch = now), or export history")
    describe(dlg)
    return dlg


# --- Edit satellite -------------------------------------------------------------------------------

def edit_satellite_dialog(app):
    sim = app.sim
    if not 0 <= app.selected < sim.n:
        app.toast("Select a satellite first")
        return None
    s = sim.sats[app.selected]
    specs = [
        F("name", "Name", "text", s.name),
        F("mass", "Mass (kg)", "float", f"{s.mass:g}"),
        F("area", "Area (m^2)", "float", f"{s.area:g}"),
        F("cd", "Drag coefficient Cd", "float", f"{s.cd:g}"),
    ]

    def on_ok(v):
        if min(v["mass"], v["area"], v["cd"]) <= 0:
            return "values must be positive"
        new = v["name"] or s.name
        if new != s.name:
            new = sim.unique_name(new)
            for m in sim.maneuvers:
                if m.sat == s.name:
                    m.sat = new
                if m.target == s.name:
                    m.target = new
            s.name = new
        s.mass, s.area, s.cd = v["mass"], v["area"], v["cd"]
        return None

    return FormDialog(app, "Edit satellite", specs, on_ok, "Apply")
