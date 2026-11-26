"""Scenario-building dialogs: add satellites, Walker constellations,
maneuvers, ground stations, physics settings and scenario files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import spaceweather
from ..batch import export_history_csv, parse_duration
from ..constants import CR_DEFAULT, R_EARTH
from ..elements import period_of, rv2coe
from ..forces import ForceModel
from ..integrators import METHODS
from ..launch import G0_M
from ..maneuvers import Maneuver
from ..matlab_export import ROOT, export_live_script, matlab_name
from ..planner import (
    plan_bielliptic,
    plan_hohmann,
    plan_rendezvous,
    plan_target_orbit,
    plane_change_summary,
    scan_rendezvous,
)
from ..scenario import (
    PRESETS,
    ConstellationSpec,
    GroundStation,
    Scenario,
    check_cr,
    j2_acts,
    orbit_state,
    palette_color,
    walker_states,
    zonal_terms,
)
from ..simulation import ACTIVE, PROPAGATORS
from ..timeutil import format_duration, format_period
from .planview import PlanPreview
from .widgets import FieldSpec as F
from .widgets import FormDialog

# --- Add satellite -------------------------------------------------------------------------

MODES = ["Circular altitude", "Perigee / apogee", "Elements (a, e)", "Surface launch",
         "State vector (ECI)", "Geostationary slot", "TLE"]
_ELEM_MODES = MODES[:3]
MODE_TIPS = {
    "Circular altitude": "A circular orbit at one height.",
    "Perigee / apogee": "An ellipse given by the heights of its lowest and highest points.",
    "Elements (a, e)": "An ellipse given by its semi-major axis and eccentricity.",
    "Surface launch": "Start at a point on (or above) the ground with a speed and heading, "
                      "and see where it goes: a hop, an orbit or an escape.",
    "State vector (ECI)": "Position and velocity in the Earth-centered inertial frame.",
    "Geostationary slot": "A geostationary orbit parked over one longitude.",
    "TLE": "Paste a NORAD two-line element set (e.g. from celestrak.org).",
}
INC_TIP = ("Tilt of the orbit against the equator in degrees: 0 equatorial, 90 polar, above "
           "90 retrograde. Type 'sso' for the Sun-synchronous inclination of this orbit.")
RAAN_TIP = ("Right ascension of the ascending node: where, measured east from the vernal "
            "equinox, the orbit crosses the equator going north.")
MASS_TIP = "Spacecraft mass; with the area and Cd it sets how strongly air drag slows it."
DRY_TIP = ("Mass left once the propellant is gone. Burns stop there (the log says so). "
           "0 means no limit: any delta-v can be spent.")
AREA_TIP = "Cross-section facing the airflow, for air drag."
CD_TIP = ("Drag coefficient: about 2.2 for a typical satellite. When Physics works Cd out "
          "from the air (physical drag coefficient), this is a correction on it instead: "
          "2.2 = as modelled, 2.42 = 10% more drag.")
CR_TIP = ("Reflectivity, for sunlight pressure (Forces): 1 absorbs all sunlight, 2 mirrors "
          "it straight back; 0 to 2.5.")


# (orbit-spec key, form field) pairs: the size and shape of each element mode,
# and a surface launch
_SHAPE_FIELDS = {"Circular altitude": [("altitude", "alt")],
                 "Perigee / apogee": [("perigee_alt", "hp"), ("apogee_alt", "ha")],
                 "Elements (a, e)": [("a", "a"), ("e", "e")]}
_SURFACE_FIELDS = [("lat", "lat"), ("lon", "lon"), ("alt", "salt"), ("speed", "speed"),
                   ("azimuth", "azimuth"), ("fpa", "fpa")]


def _mode(*names):
    """Visibility predicate: the initial-condition mode is one of ``names``."""
    return lambda raw: raw.get("mode") in names


def _orbit_from(v: dict) -> dict:
    """The scenario orbit spec described by the Add-satellite form."""
    mode = v["mode"]
    if mode in _SHAPE_FIELDS:
        o = {"type": "elements", "i": v["i"], "raan": v["raan"],
             "argp": v.get("argp", 0.0), "nu": v["nu"]}         # a circle has no perigee
        o.update((key, v[field]) for key, field in _SHAPE_FIELDS[mode])
        return o
    if mode == "Surface launch":
        return {"type": "surface", **{key: v[field] for key, field in _SURFACE_FIELDS}}
    if mode == "State vector (ECI)":
        return {"type": "state", "r": [v["x"], v["y"], v["z"]], "v": [v["vx"], v["vy"], v["vz"]]}
    if mode == "Geostationary slot":
        return {"type": "geo", "lon": v["geolon"]}
    return {"type": "tle", "line1": v["line1"], "line2": v["line2"]}


def add_satellite_dialog(app):
    """Add one satellite (or copies along its orbit) from any initial condition (key A)."""
    sim = app.sim
    elements, surface = _mode(*_ELEM_MODES), _mode("Surface launch")
    sv = _mode("State vector (ECI)")
    sv_r = "Position component in the Earth-centered inertial frame (x toward the vernal " \
           "equinox, z toward the north pole)."
    sv_v = "Velocity component in the Earth-centered inertial frame."
    specs = [
        F("name", "Name", "text", f"Sat {sim.n + 1}",
          tip="Name shown in the list and the view. Copies get -1, -2, ... added."),
        F("preset", "Preset", "choice", "Custom", ["Custom"] + list(PRESETS),
          tip="Fill the form with a well-known orbit (ISS, GPS, Molniya, ...). You can "
              "still change any field afterward."),
        F("mode", "Initial condition", "choice", MODES[0], MODES,
          tip="How you describe where the satellite starts. The fields below change to "
              "match.", option_tips=MODE_TIPS),
        F("alt", "Altitude (km)", "float", "550", visible=_mode("Circular altitude"),
          tip="Height of the circular orbit above the equatorial radius (6,378 km), averaged "
              "over a revolution when J2 acts. The ISS flies at about 420 km, "
              "geostationary satellites at 35,786 km."),
        F("hp", "Perigee altitude (km)", "float", "300", visible=_mode("Perigee / apogee"),
          tip="Height of the orbit's lowest point above the equatorial radius (6,378 km); "
              "a mean value when J2 acts."),
        F("ha", "Apogee altitude (km)", "float", "2000", visible=_mode("Perigee / apogee"),
          tip="Height of the orbit's highest point above the equatorial radius (6,378 km); "
              "a mean value when J2 acts."),
        F("a", "Semi-major axis a (km)", "float", "8000", visible=_mode("Elements (a, e)"),
          tip="Half the long axis of the ellipse, from the Earth's center (the Earth's "
              "radius is 6,378 km); the orbit-averaged (mean) value when J2 acts."),
        F("e", "Eccentricity e", "float", "0.1", visible=_mode("Elements (a, e)"),
          tip="0 is a circle; the nearer to 1, the longer and thinner the ellipse."),
        F("i", "Inclination (deg or 'sso')", "floatx", "51.6", visible=elements, tip=INC_TIP),
        F("raan", "RAAN (deg)", "float", "0", visible=elements, tip=RAAN_TIP),
        F("argp", "Arg. of perigee (deg)", "float", "0",
          visible=_mode("Perigee / apogee", "Elements (a, e)"),
          tip="Where the lowest point lies: the angle from the ascending node to the "
              "perigee, in the direction of motion."),
        F("nu", "True anomaly (deg)", "float", "0", visible=elements,
          tip="Where on the orbit the satellite starts: the angle from the perigee (from the "
              "ascending node on a circular orbit)."),
        F("lat", "Latitude (deg)", "float", "28.5", visible=surface,
          tip="Starting latitude, north positive."),
        F("lon", "Longitude (deg E)", "float", "-80.6", visible=surface,
          tip="Starting longitude, east positive (west is negative)."),
        F("salt", "Altitude (km)", "float", "0", visible=surface,
          tip="Starting height above the surface."),
        F("speed", "Ground-relative speed (km/s)", "float", "7.5", visible=surface,
          tip="Speed relative to the turning ground. About 7.8 km/s reaches low orbit, "
              "about 11 km/s escapes."),
        F("azimuth", "Azimuth (deg from N)", "float", "90", visible=surface,
          tip="Compass heading: 0 north, 90 east (the Earth's spin helps), 180 south."),
        F("fpa", "Flight-path angle (deg)", "float", "0", visible=surface,
          tip="Angle above the horizontal: 0 flies level, 90 straight up."),
        F("x", "x (km)", "float", "7000", visible=sv, tip=sv_r),
        F("y", "y (km)", "float", "0", visible=sv, tip=sv_r),
        F("z", "z (km)", "float", "0", visible=sv, tip=sv_r),
        F("vx", "vx (km/s)", "float", "0", visible=sv, tip=sv_v),
        F("vy", "vy (km/s)", "float", "7.546", visible=sv, tip=sv_v),
        F("vz", "vz (km/s)", "float", "0", visible=sv, tip=sv_v),
        F("geolon", "Slot longitude (deg E)", "float", "-75", visible=_mode("Geostationary slot"),
          tip="The longitude the satellite hovers over, east positive."),
        F("line1", "Line 1", "text", "", visible=_mode("TLE"), wide=True,
          tip="First line of the two-line element set (starts with '1 '). Ctrl+V pastes."),
        F("line2", "Line 2", "text", "", visible=_mode("TLE"), wide=True,
          tip="Second line of the two-line element set (starts with '2 '). Ctrl+V pastes."),
        F("count", "Copies along orbit", "int", "1",
          tip="Add this many satellites on the same orbit, spaced evenly in time (mean "
              "anomaly)."),
        F("mass", "Mass (kg)", "float", "500", tip=MASS_TIP),
        F("dry_mass", "Dry mass (kg)", "float", "0", tip=DRY_TIP),
        F("area", "Area (m^2)", "float", "5", tip=AREA_TIP),
        F("cd", "Drag coefficient Cd", "float", "2.2", tip=CD_TIP),
        F("cr", "Reflectivity Cr", "float", f"{CR_DEFAULT:g}", tip=CR_TIP),
        F("preview", "", "info", "",
          tip="The orbit these values give: semi-major axis, eccentricity, inclination, "
              "perigee / apogee heights and period."),
    ]

    def on_change(dlg, key):
        if key == "preset":
            _apply_preset(dlg)
        _preview(dlg)

    def on_ok(v):
        orbit = _orbit_from(v)
        props = dict(mass=v["mass"], area=v["area"], cd=v["cd"])
        if min(props.values()) <= 0:
            return "mass, area and coefficients must be positive"
        if not 0 <= v["dry_mass"] <= v["mass"]:
            return "dry mass must be between 0 (no limit) and the mass"
        added = sim.add_satellites(orbit, v["name"] or "Sat", count=v["count"],
                                   dry_mass=v["dry_mass"], cr=v["cr"], **props)
        app.select(sim.n - len(added))
        return None

    dlg = FormDialog(app, "Add satellite", specs, on_ok, "Add", width=560, on_change=on_change,
                     subtitle="Tab moves between fields - click or scroll a choice to change it",
                     ok_hint="Add the satellite (or copies) and select it")
    _preview(dlg)
    return dlg


def _apply_preset(dlg):
    """Fill the form from the chosen entry of ``PRESETS``."""
    name = dlg.widgets["preset"].value
    if name not in PRESETS:     # the "custom" entry: the form keeps what was typed
        return
    p = PRESETS[name]
    o = p["orbit"]
    dlg.set("name", name.split(" (")[0])
    for k in ("mass", "area", "cd"):
        if k in p:
            dlg.set(k, p[k])
    t = o["type"]
    if t == "elements":
        # the mode whose first size key the preset has (a and e otherwise)
        mode = next((m for m, fields in _SHAPE_FIELDS.items() if fields[0][0] in o),
                    "Elements (a, e)")
        dlg.set("mode", mode)
        for key, field in _SHAPE_FIELDS[mode]:
            dlg.set(field, o[key])
        for k in ("i", "raan", "argp", "nu"):
            dlg.set(k, o.get(k, 0))
    elif t == "surface":
        dlg.set("mode", "Surface launch")
        for key, field in _SURFACE_FIELDS:
            dlg.set(field, o.get(key, 0))
    elif t == "geo":
        dlg.set("mode", "Geostationary slot")
        dlg.set("geolon", o.get("lon", 0))


def _preview(dlg):
    """One-line summary of the orbit the form describes (blank while it is invalid)."""
    try:
        v = dlg.values()
        sim = dlg.app.sim
        r, vv = orbit_state(_orbit_from(v), sim.clock, sim.t, j2_acts(sim.propagator, sim.forces),
                            zonal_terms(sim.propagator, sim.forces))
        el = rv2coe(r, vv)
        if el.e < 1:
            txt = (f"a {el.a:,.0f} km  e {el.e:.4f}  i {math.degrees(el.i):.2f} deg  "
                   f"hp {el.rp - R_EARTH:,.0f} / ha {el.ra - R_EARTH:,.0f} km  "
                   f"T {format_period(el.period)}")
        else:
            vinf = math.sqrt(max(0.0, 2 * el.energy))
            txt = f"open orbit: e {el.e:.3f}, v_inf {vinf:.2f} km/s, hp {el.rp - R_EARTH:,.0f} km"
        dlg.info["preview"] = txt
    except ValueError as exc:   # incomplete or invalid input: say what
        dlg.info["preview"] = f"({exc})"


# --- Walker constellation ------------------------------------------------------------------

def walker_dialog(app):
    """Create a Walker constellation (key W)."""
    sim = app.sim
    specs = [
        F("name", "Name", "text", "Walker",
          tip="Base name; each satellite gets its plane and slot number added (-0102 is "
              "plane 1, slot 2)."),
        F("altitude", "Altitude (km)", "float", "1200",
          tip="Mean height of every satellite's circular orbit above the equatorial "
              "radius (6,378 km)."),
        F("inclination", "Inclination (deg)", "float", "53",
          tip="Tilt of every plane against the equator."),
        F("total", "Total satellites t", "int", "24",
          tip="How many satellites in all, shared evenly between the planes."),
        F("planes", "Planes p", "int", "6",
          tip="How many orbital planes, spread evenly in RAAN."),
        F("phasing", "Phasing f (0..p-1)", "int", "1",
          tip="How far each plane's satellites are shifted along the orbit relative to the "
              "previous plane, in units of 360 / t degrees. It keeps planes from bunching."),
        F("pattern", "Pattern", "choice", "delta", ["delta", "star"],
          tip="How the planes are spread around the Earth.",
          option_tips={"delta": "Delta: planes spread over the full 360 degrees of RAAN "
                                "(Galileo, GPS-like).",
                       "star": "Star: planes spread over 180 degrees, so they meet near the "
                               "poles (Iridium-like polar orbits)."}),
        F("raan0", "First plane RAAN (deg)", "float", "0",
          tip="RAAN of the first plane; the others follow from it."),
        F("mass", "Mass each (kg)", "float", "260", tip=MASS_TIP),
        F("dry_mass", "Dry mass each (kg)", "float", "0", tip=DRY_TIP),
        F("area", "Area each (m^2)", "float", "4", tip=AREA_TIP),
        F("cd", "Drag coefficient Cd", "float", "2.2", tip=CD_TIP),
        F("cr", "Reflectivity Cr each", "float", f"{CR_DEFAULT:g}", tip=CR_TIP),
        F("preview", "", "info", "",
          tip="The constellation in Walker notation i:t/p/f, satellites per plane and "
              "the orbital period."),
    ]

    def spec(v) -> ConstellationSpec:
        """The constellation the form describes (ValueError saying what is wrong)."""
        c = ConstellationSpec(name=v["name"] or "Walker", altitude=v["altitude"],
                              inclination=v["inclination"], total=v["total"], planes=v["planes"],
                              phasing=v["phasing"], pattern=v["pattern"], raan0=v["raan0"],
                              mass=v["mass"], dry_mass=v["dry_mass"], area=v["area"],
                              cd=v["cd"], cr=v["cr"])
        if c.total % c.planes:
            raise ValueError("total satellites must be a multiple of the number of planes")
        return c

    def on_change(dlg, key):
        try:
            c = spec(dlg.values())
            period = format_period(float(period_of(R_EARTH + c.altitude)))
            dlg.info["preview"] = (f"{c.pattern} {c.inclination:.1f}: "
                                   f"{c.total}/{c.planes}/{c.phasing} - "
                                   f"{c.total // c.planes} per plane, T {period}")
        except ValueError as exc:   # incomplete or invalid input: say what
            dlg.info["preview"] = f"({exc})"

    def on_ok(v):
        c = spec(v)
        states = walker_states(c, j2_acts(sim.propagator, sim.forces))
        color = palette_color(sim.n)
        props = dict(mass=c.mass, area=c.area, cd=c.cd, dry_mass=c.dry_mass, cr=c.cr)
        n0 = sim.n
        sim.add_many([(name, color, props, r, v_) for name, r, v_ in states])
        app.select(n0)
        return None

    dlg = FormDialog(app, "Walker constellation", specs, on_ok, "Create", on_change=on_change,
                     subtitle="i:t/p/f - delta spreads planes over 360 deg, star over 180 deg",
                     ok_hint="Add every satellite of the constellation")
    on_change(dlg, None)
    return dlg


# --- Maneuvers -------------------------------------------------------------------------------

KINDS = {"Change orbit to...": "target_orbit",
         "Impulse (V/N/B components)": "impulse", "Hohmann transfer": "hohmann",
         "Bi-elliptic transfer": "bielliptic", "Circularize": "circularize",
         "Plane change": "plane_change", "Rendezvous (Lambert)": "rendezvous",
         "Finite burn (thrust)": "finite"}
TIMINGS = {"Now": "now", "After delay": "delay", "Next periapsis": "periapsis",
           "Next apoapsis": "apoapsis", "Ascending node": "ascending_node",
           "Descending node": "descending_node"}
DIRECTIONS = {"Prograde": (1, 0, 0), "Retrograde": (-1, 0, 0), "Normal": (0, 1, 0),
              "Anti-normal": (0, -1, 0), "Radial out": (0, 0, 1), "Radial in": (0, 0, -1)}
KIND_TIPS = {
    "Change orbit to...": "Give the orbit you want (perigee, apogee, tilt) and get the two "
                          "burns that reach it, with the plane change split between them "
                          "the cheapest way. GTO to GEO is the classic case.",
    "Impulse (V/N/B components)": "An instant kick of the velocity, given by its three "
                                  "components.",
    "Hohmann transfer": "Two burns to a new circular altitude: one to enter a transfer "
                        "ellipse, one half an orbit later to circularize. The cheapest way "
                        "between two circular orbits of similar size.",
    "Bi-elliptic transfer": "Three burns via a high intermediate apoapsis. Cheaper than "
                            "Hohmann only when the target is more than about 12 times the "
                            "starting radius.",
    "Circularize": "One burn that makes the orbit circular at the height where it fires. "
                   "Best at perigee or apogee.",
    "Plane change": "Tilt the orbit by a number of degrees. Cheapest at a node, and costly: "
                    "10 degrees in low orbit takes about 1.3 km/s.",
    "Rendezvous (Lambert)": "Fly to another satellite: a Lambert arc that meets it after the "
                            "time of flight, then a burn to match its velocity.",
    "Finite burn (thrust)": "A real engine firing for a while along a fixed direction, "
                            "instead of an instant kick.",
}
TIMING_TIPS = {
    "Now": "Burn at once.",
    "After delay": "Burn after the delay below.",
    "Next periapsis": "Burn at the orbit's next lowest point.",
    "Next apoapsis": "Burn at the orbit's next highest point.",
    "Ascending node": "Burn where the orbit next crosses the equator going north.",
    "Descending node": "Burn where the orbit next crosses the equator going south.",
}
FRAME_TIPS = {
    "VNB": "VNB: V along the velocity, N along the orbit normal, B = V x N.",
    "RSW": "RSW: the three numbers are radial (out from the Earth), along-track and "
           "cross-track (orbit normal).",
    "ECI": "ECI: the three numbers are x, y, z in the inertial frame.",
}
DV_TIP = "Velocity change, in m/s, along this axis of the chosen frame (see Frame)."


def _kind(*names):
    """Visibility predicate: the maneuver kind is one of ``names``."""
    return lambda raw: KINDS.get(raw.get("kind")) in names


def maneuver_dialog(app):
    """Plan and schedule a maneuver, previewing its delta-v (key B)."""
    sim = app.sim
    # a payload still riding its rocket cannot maneuver until it separates
    names = [s.name for i, s in enumerate(sim.sats)
             if s.status == ACTIVE and sim.ascent_of(i) is None]
    if not names:
        app.toast("No active satellites to maneuver")
        return None
    cur = sim.sats[app.selected].name if 0 <= app.selected < sim.n else names[0]
    other = next((n for n in names if n != cur), cur)
    now = rv2coe(*np.split(sim.y[sim.require(cur)], 2))      # the target starts as "as is"
    now_p = max(now.a * (1 - now.e) - R_EARTH, 0.0)
    now_a = now.a * (1 + now.e) - R_EARTH if now.e < 1 else now_p
    specs = [
        F("sat", "Satellite", "choice", cur, names, tip="The satellite that will burn."),
        F("kind", "Maneuver", "choice", "Hohmann transfer", list(KINDS),
          tip="What the burn should achieve.", option_tips=KIND_TIPS),
        F("timing", "Execute", "choice", "Now", list(TIMINGS),
          tip="When the (first) burn fires.", option_tips=TIMING_TIPS),
        F("delay", "Delay (s)", "float", "600", visible=lambda r: r.get("timing") == "After delay",
          tip="Seconds of simulated time until the burn."),
        F("dv_v", "dV prograde V (m/s)", "float", "100", visible=_kind("impulse"), tip=DV_TIP),
        F("dv_n", "dV normal N (m/s)", "float", "0", visible=_kind("impulse"), tip=DV_TIP),
        F("dv_b", "dV binormal B (m/s)", "float", "0", visible=_kind("impulse"), tip=DV_TIP),
        F("frame", "Frame", "choice", "VNB", ["VNB", "RSW", "ECI"], visible=_kind("impulse"),
          tip="The axes the three velocity changes are measured along.",
          option_tips=FRAME_TIPS),
        F("target_alt", "Target altitude (km)", "float", "35786",
          visible=_kind("hohmann", "bielliptic"),
          tip="Height of the circular orbit to end in (35,786 km is geostationary)."),
        F("tgt_perigee", "Target perigee (km)", "float", f"{now_p:.0f}",
          visible=_kind("target_orbit"),
          tip="Height of the new orbit's lowest point above the equator's sea level. "
              "It starts at the satellite's present value."),
        F("tgt_apogee", "Target apogee (km)", "float", f"{now_a:.0f}",
          visible=_kind("target_orbit"),
          tip="Height of the new orbit's highest point (equal to the perigee for a circle; "
              "35,786 km for both is geostationary)."),
        F("tgt_inc", "Target inclination (deg)", "float", f"{math.degrees(now.i):.2f}",
          visible=_kind("target_orbit"),
          tip="Tilt of the new orbit against the equator (0 is equatorial). Tilting is "
              "cheapest where the satellite is slow, so most of it goes to the higher burn."),
        F("rb_alt", "Intermediate apoapsis (km)", "float", "100000", visible=_kind("bielliptic"),
          tip="How high the first transfer ellipse reaches before the second burn."),
        F("delta_i", "Inclination change (deg)", "float", "-10", visible=_kind("plane_change"),
          tip="Degrees to add to the inclination (negative lowers it)."),
        F("target", "Target satellite", "choice", other, names, visible=_kind("rendezvous"),
          tip="The satellite to fly to."),
        F("auto_tof", "Cheapest safe time of flight", "bool", True, visible=_kind("rendezvous"),
          tip="Search transfer times for the one needing the least delta-v whose path stays "
              "clear of the atmosphere. Untick to type your own."),
        F("tof", "Time of flight (min)", "float", "45",
          visible=lambda r: KINDS.get(r.get("kind")) == "rendezvous" and not r.get("auto_tof"),
          tip="Minutes from the first burn until meeting the target."),
        F("thrust", "Thrust (N)", "float", "400", visible=_kind("finite"),
          tip="Engine force in newtons."),
        F("duration", "Burn duration (s)", "float", "300", visible=_kind("finite"),
          tip="How long the engine fires."),
        F("direction", "Direction", "choice", "Prograde", list(DIRECTIONS),
          visible=_kind("finite"),
          tip="Where the thrust points, relative to the orbit: prograde speeds the satellite "
              "up (raising the far side of the orbit), retrograde slows it, normal / "
              "anti-normal tilt the plane, radial points away from / toward the Earth."),
        F("isp", "Specific impulse Isp (s)", "float", "320",
          tip="Engine efficiency: the higher it is, the less propellant a burn uses. "
              "Chemical engines 300-450 s, ion engines 1,500 s and more."),
        F("preview", "", "info", "",
          tip="The planned burns' delta-v, or why the maneuver is impossible."),
    ]

    scan_cache: dict = {}     # the rendezvous TOF scan is slow: reuse it while the form changes
    plans: dict = {}          # planners fly the run's forces: one plan per form and state

    def plan(v, aim=True):
        key = (repr(sorted(v.items())), aim, sim.t, sim.n, hash(sim.y.tobytes()))
        if key not in plans:
            try:
                plans[key] = _plan_maneuver(sim, v, scan_cache, aim)
            except ValueError as exc:
                plans[key] = exc
        if isinstance(plans[key], ValueError):
            raise plans[key]
        burns, summary = plans[key]
        return [Maneuver.from_dict(b.to_dict()) for b in burns], summary

    def on_change(dlg, key):
        try:
            v = dlg.values()
            plan(v, aim=False)                 # quick, while typing; raises if impossible
        except ValueError as exc:
            dlg.info["preview"] = f"({exc})"
            preview.clear(str(exc))
            return

        def plan_fn(run, aim):
            """(sat, burns, summary) of this form flown from ``run`` (a snapshot
            when aimed: that runs on the preview's thread, with a cache of its own)."""
            if run is sim and not aim:
                return (v["sat"], *plan(v, aim=False))
            return (v["sat"], *_plan_maneuver(run, v, {}, aim))

        preview.show(repr(sorted(v.items())), plan_fn, sim.sats[sim.require(v["sat"])].color)
        flown = preview.estimate
        if flown is None:
            dlg.info["preview"] = f"({preview.error})"
        else:
            n = len(flown.burns)
            dlg.info["preview"] = (f"{n} burn{'s' if n > 1 else ''}, ~{flown.total_dv * 1000:,.0f}"
                                   " m/s - the card in the view explains")

    def on_ok(v):
        flown, i = preview.plan(), sim.require(v["sat"])
        if (flown is not None and flown.aimed and preview.key == repr(sorted(v.items()))
                and flown.t0 == sim.t and np.array_equal(flown.start, sim.y[i])):
            burns = flown.maneuvers               # flown already, from this very state
        else:
            burns, _ = plan(v)
        for b in burns:
            sim.schedule(Maneuver.from_dict(b.to_dict()))
        return None

    def to_node(dlg):
        """Close this and open a burn node on the chosen satellite instead."""
        app.select(sim.require(dlg.values()["sat"]))
        app.close_dialog(dlg)
        app.open("node")

    dlg = FormDialog(app, "Plan maneuver", specs, on_ok, "Schedule", width=560,
                     on_change=on_change, ok_hint="Schedule the burns; the event log shows "
                                                  "them when they fire",
                     extra_buttons=[("Burn node", to_node,
                                     "Shape the burn yourself by dragging handles on the "
                                     "orbit in the view (Shift+B)")],
                     subtitle="VNB: V prograde, N orbit normal, B = V x N "
                              "(radial out on circular orbits)", dock="right")
    preview = app.plan_preview = PlanPreview(app, dlg)
    on_change(dlg, None)
    return dlg


def _plan_maneuver(sim, v: dict, scan_cache: dict, aim: bool = True):
    """(burns, summary) for the maneuver form values ``v``; raises ValueError
    if the maneuver is impossible. ``scan_cache`` keeps the automatic
    rendezvous time of flight between calls; without ``aim`` transfers are
    the two-body estimate (the preview)."""
    kind = KINDS[v["kind"]]
    timing = TIMINGS[v["timing"]]
    delay = v.get("delay", 0.0)
    sat = v["sat"]
    common = dict(timing=timing, delay=delay, isp=v["isp"])
    if kind == "target_orbit":
        burns, summary = plan_target_orbit(sim, sat, v["tgt_perigee"], v["tgt_apogee"],
                                           v["tgt_inc"], timing, delay, aim)
    elif kind == "hohmann":
        burns, summary = plan_hohmann(sim, sat, v["target_alt"], timing, delay, aim)
    elif kind == "bielliptic":
        burns, summary = plan_bielliptic(sim, sat, v["rb_alt"], v["target_alt"], timing, delay,
                                         aim)
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
        dv = (v["dv_v"] / 1000, v["dv_n"] / 1000, v["dv_b"] / 1000)      # m/s -> km/s
        burns = [Maneuver(sat, "impulse", dv=dv, frame=v["frame"], **common)]
        burns[0].check()
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
        burns[0].check()
        # rocket equation for the preview: m_final = m - mdot * t, mdot = F / (Isp g0)
        m = sim.sats[sim.require(sat)].mass
        mf = m - v["thrust"] / (v["isp"] * G0_M) * v["duration"]
        dvm = v["isp"] * G0_M * math.log(m / mf) if mf > 0 else float("inf")
        summary = f"dV ~{dvm:.1f} m/s, accel {v['thrust'] / m * 1000:.2f} mm/s^2"
    for b in burns:
        b.isp = v["isp"]
    return burns, summary


# --- Ground station -------------------------------------------------------------------------

def station_dialog(app):
    """Add a ground station (key N)."""
    specs = [
        F("name", "Name", "text", f"Station {len(app.sim.stations) + 1}",
          tip="Name shown on the map and in the event log."),
        F("lat", "Latitude (deg)", "float", "51.48", tip="North positive, -90 to 90."),
        F("lon", "Longitude (deg E)", "float", "0.0", tip="East positive (west is negative)."),
        F("alt", "Altitude (km)", "float", "0.05", tip="Height of the antenna above sea level."),
        F("min_el", "Minimum elevation (deg)", "float", "10",
          tip="A satellite counts as in contact (AOS) once it rises this many degrees above "
              "the station's horizon; hills and buildings block lower ones."),
    ]

    def on_ok(v):
        if not -90 <= v["lat"] <= 90:
            return "latitude must be within +/-90 deg"
        app.sim.add_station(GroundStation(v["name"] or "Station", v["lat"], v["lon"], v["alt"],
                                          v["min_el"]))
        return None

    return FormDialog(app, "Add ground station", specs, on_ok, "Add",
                      ok_hint="Add the station; the log reports each satellite it sees")


# --- Physics settings -----------------------------------------------------------------------

# what each propagator and Cowell integrator does (the Physics dialog and the start menu)
PROPAGATOR_TIPS = {"cowell": "cowell: integrate every enabled force numerically. The "
                             "most complete, and the only one with drag and burns.",
                   "kepler": "kepler: the exact two-body ellipse. Fast, but ignores the "
                             "Earth's shape and air.",
                   "j2mean": "j2mean: the ellipse plus the average drift caused by the "
                             "Earth's bulge (J2), timed by the mean orbit. Fast and "
                             "good for long runs; no drag or burns."}
METHOD_TIPS = {"dopri5": "dopri5: Dormand-Prince 5(4), adapts its step to meet the "
                         "tolerances. The accurate default.",
               "rk4": "rk4: classic Runge-Kutta with a fixed step.",
               "wh": "wh: Wisdom-Holman. Follows the exact ellipse and adds the "
                     "small forces as kicks, so long fixed steps (60 s or more) "
                     "stay accurate and the energy does not drift.",
               "leapfrog": "leapfrog: a simple symplectic fixed-step method. Its "
                           "energy stays bounded, but its timing drifts quickly: "
                           "it needs steps of a few seconds."}
# what each force-model switch does
FORCE_TIPS = {
    "j2": "The Earth's equatorial bulge. It turns orbital planes (node drift) and "
          "ellipses (perigee drift): the largest effect after central gravity.",
    "j3": "The Earth's slight pear shape (north-south asymmetry). It slowly changes "
          "eccentricity.",
    "j4": "The next, smaller term of the Earth's shape.",
    "c22": "The equator is slightly oval, and the oval turns with the Earth. Small, but "
           "it is what pulls geostationary satellites off their slots (toward 75 E or "
           "105 W) over weeks. Cowell only.",
    "drag": "Air resistance from the upper atmosphere. It makes low orbits decay until the "
            "satellite re-enters.",
    "sun": "The Sun's pull, less its pull on the Earth: a tide that slowly tilts and "
           "stretches orbits. Hundreds of metres a day for a low satellite.",
    "moon": "The Moon's tide, about twice the Sun's. It matters most for high orbits.",
    "srp": "Sunlight pushing on the satellite (Cr x area / mass), off in the Earth's "
           "shadow. Tens to hundreds of metres a day for a low satellite.",
    "tides": "The Sun and Moon stretch the solid Earth by about 30 cm; the bulge pulls on "
             "the satellite: at 700 km about 10 m along-track after a day and 50 m after "
             "three. IERS 2010, degree 2; ocean tides are not modelled.",
    "relativity": "General relativity's correction to the Earth's pull (the Schwarzschild "
                  "term): at 700 km about 2.5 m along-track a day.",
}
# the Earth's gravity field: the separate J2/J3/J4/C22 terms, or EGM2008 to N x N
FIELD_CHOICES = {"J2-J4 terms": 0, "EGM2008 8x8": 8, "EGM2008 20x20": 20,
                 "EGM2008 40x40": 40, "EGM2008 70x70": 70}
FIELD_TIPS = {
    "J2-J4 terms": "The Earth's shape as the switches below say: J2, J3, J4 and C22.",
    "EGM2008 8x8": "The full measured field to degree and order 8: the big lumps.",
    "EGM2008 20x20": "To degree 20: most of the effect on a low orbit.",
    "EGM2008 40x40": "To degree 40.",
    "EGM2008 70x70": "To degree 70, as precise orbit work uses for a 700 km satellite. "
                     "About 2 ms per step.",
}


def field_choice(degree: int) -> str:
    """The Physics dialog's label for ``gravity_degree`` (one is added if needed)."""
    for label, n in FIELD_CHOICES.items():
        if n == degree:
            return label
    label = f"EGM2008 {degree}x{degree}"
    FIELD_CHOICES[label] = degree
    FIELD_TIPS[label] = f"The full measured field to degree and order {degree}."
    return label
DENSITY_TIP = ("Multiplies the air density: about 0.3 at solar minimum, 1 on average, 3 at "
               "solar maximum (when the Sun heats and puffs up the upper atmosphere). With "
               "NRLMSIS, which already follows the Sun, it is a correction: 1 as modelled.")
# what drag looks the air up in
ATMOSPHERE_CHOICES = {"Mean-Sun table": "table",
                      "NRLMSIS 2.1 (measured space weather)": "msis"}
ATMOSPHERE_TIPS = {
    "Mean-Sun table": "Density by height alone, for average solar activity (scaled by the "
                      "density scale below). Fast; needs no data.",
    "NRLMSIS 2.1 (measured space weather)": (
        "The US Naval Research Lab's model: density by place and time, following the "
        "solar cycle, storms (7 times denser at 700 km in a big one), day and night, "
        "season. Needs the measured space weather: python tools/fetch_space_weather.py. "
        "MATLAB exports run it through pymsis in MATLAB's Python."),
}
# where drag's Cd comes from
CD_MODEL_CHOICES = {"Fixed (each satellite's Cd)": "fixed",
                    "Physical, sphere (needs NRLMSIS)": "dria_sphere"}
CD_MODEL_TIPS = {
    "Fixed (each satellite's Cd)": "Each satellite's own Cd, the same in any air.",
    "Physical, sphere (needs NRLMSIS)": (
        "Cd worked out from the NRLMSIS air at the satellite: its temperature and the "
        "atomic oxygen stuck on the surface (Sentman's sphere, SESAM accommodation). About "
        "2.8 at 700 km; it varies a few % around the orbit and dips in storms. Each "
        "satellite's Cd becomes a correction on it (2.2 = as modelled). MATLAB exports "
        "fly it too."),
}


def physics_dialog(app):
    """Force model, propagator and integrator settings (key P)."""
    sim = app.sim
    fm = sim.forces
    it = sim.integrator
    specs = [
        F("propagator", "Propagator", "choice", sim.propagator, list(PROPAGATORS),
          tip="How orbits are moved forward in time.",
          option_tips=PROPAGATOR_TIPS),
        F("method", "Integrator (Cowell)", "choice", it.method, list(METHODS),
          tip="The numerical method Cowell uses.",
          option_tips=METHOD_TIPS),
        F("rtol", "Relative tolerance", "float", f"{it.rtol:g}",
          tip="dopri5's allowed error per step, relative to the state. Smaller is more "
              "accurate and slower."),
        F("atol", "Absolute tolerance", "float", f"{it.atol:g}",
          tip="dopri5's allowed error per step in km and km/s, for values near zero."),
        F("h_max", "Max step (s)", "float", f"{it.h_max:g}",
          tip="Longest step dopri5 may take, however smooth the motion."),
        F("h_fixed", "Fixed step (s)", "float", f"{it.h_fixed:g}",
          tip="Step length of the fixed-step methods rk4, wh and leapfrog. Low orbits need "
              "about 30 s or less with rk4, 60 s with wh, a few seconds with leapfrog."),
        F("field", "Earth's gravity field", "choice", field_choice(fm.gravity_degree),
          list(FIELD_CHOICES), option_tips=FIELD_TIPS,
          tip="EGM2008 replaces J2/J3/J4/C22 in the motion when chosen; the switches "
              "still set the analytic helpers (sun-synchronous tilt, launches)."),
        F("j2", "J2 oblateness", "bool", fm.j2, tip=FORCE_TIPS["j2"]),
        F("j3", "J3", "bool", fm.j3, tip=FORCE_TIPS["j3"]),
        F("j4", "J4", "bool", fm.j4, tip=FORCE_TIPS["j4"]),
        F("c22", "C22/S22 (elliptical equator)", "bool", fm.c22, tip=FORCE_TIPS["c22"]),
        F("drag", "Atmospheric drag", "bool", fm.drag, tip=FORCE_TIPS["drag"]),
        F("atmosphere", "Atmosphere for drag", "choice",
          next(k for k, a in ATMOSPHERE_CHOICES.items() if a == fm.atmosphere),
          list(ATMOSPHERE_CHOICES), option_tips=ATMOSPHERE_TIPS,
          tip="What drag looks the air density up in."),
        F("cd_model", "Drag coefficient", "choice",
          next(k for k, c in CD_MODEL_CHOICES.items() if c == fm.cd_model),
          list(CD_MODEL_CHOICES), option_tips=CD_MODEL_TIPS,
          tip="Where drag's Cd comes from: each satellite's, or the air itself."),
        F("density_scale", "Density scale (solar activity)", "float", f"{fm.density_scale:g}",
          tip=DENSITY_TIP),
        F("sun", "Sun's gravity", "bool", fm.sun, tip=FORCE_TIPS["sun"]),
        F("moon", "Moon's gravity", "bool", fm.moon, tip=FORCE_TIPS["moon"]),
        F("srp", "Sunlight pressure", "bool", fm.srp, tip=FORCE_TIPS["srp"]),
        F("tides", "Solid-Earth tides", "bool", fm.tides,
          tip=FORCE_TIPS["tides"]),
        F("relativity", "Relativity", "bool", fm.relativity,
          tip=FORCE_TIPS["relativity"]),
        F("record_dt", "Trail sample interval (s)", "float", f"{sim.record_dt:g}",
          tip="Seconds of simulated time between the points stored for trails, the map "
              "tracks and the plot."),
        F("conj", "Close-approach alert (km)", "float", f"{sim.conjunction_km:g}",
          tip="Log an alert when two satellites pass closer than this."),
    ]

    def on_ok(v):
        if v["rtol"] <= 0 or v["atol"] <= 0 or v["h_max"] <= 0 or v["h_fixed"] <= 0:
            return "tolerances and steps must be positive"
        try:
            forces = ForceModel(j2=v["j2"], j3=v["j3"], j4=v["j4"], c22=v["c22"],
                                drag=v["drag"], density_scale=v["density_scale"],
                                gravity_degree=FIELD_CHOICES[v["field"]], sun=v["sun"],
                                moon=v["moon"], srp=v["srp"], tides=v["tides"],
                                relativity=v["relativity"],
                                atmosphere=ATMOSPHERE_CHOICES[v["atmosphere"]],
                                cd_model=CD_MODEL_CHOICES[v["cd_model"]])
        except ValueError as exc:     # physical Cd with the table
            return str(exc).removeprefix("forces ")
        if forces.msis_drag:          # the space weather must cover now, or say why not
            now = np.datetime64(sim.datetime().replace(tzinfo=None), "m")
            try:
                spaceweather.load().indices(np.array([now]))
            except spaceweather.SpaceWeatherError as exc:
                return f"NRLMSIS: {exc}"
        sim.forces = forces
        sim.set_propagator(v["propagator"])
        it.method = v["method"]
        it.rtol, it.atol, it.h_max, it.h_fixed = v["rtol"], v["atol"], v["h_max"], v["h_fixed"]
        it.h = min(it.h, it.h_max)
        sim.record_dt = max(0.1, v["record_dt"])
        sim.conjunction_km = v["conj"]
        for s in sim.sats:
            s.energy_ref = None       # the conserved quantity changed with the model
        sim.log(f"Physics: {sim.forces.label()}, {sim.propagator}/{it.method}")
        sim.warn_ignored()
        return None

    return FormDialog(app, "Physics & integrator", specs, on_ok, "Apply", width=520,
                      ok_hint="Use these settings from now on",
                      subtitle="Cowell integrates every enabled force; "
                               "kepler / j2mean are analytic")


# --- Scenario files ---------------------------------------------------------------------------

def scenario_dialog(app):
    """Load, save, export or reset scenarios (Ctrl+O)."""
    def label(p: Path) -> str:
        return str(p.relative_to(app.root)) if p.is_relative_to(app.root) else str(p)

    files = app.scenario_files()
    labels = [label(p) for p in files]
    current = app.scenario_path
    cur_label = label(current) if current else (labels[0] if labels else "")
    specs = [
        F("file", "Scenario file", "choice", cur_label, labels or ["(none found)"],
          tip="The scenario Load opens: the bundled ones, then those you saved "
              "(scenarios/user)."),
        F("save_name", "Save snapshot as", "text", "my_scenario",
          tip="File name for Save snapshot, stored in scenarios/user."),
        F("desc", "", "info", "", tip="What the chosen scenario contains."),
    ]
    lookup = dict(zip(labels, files, strict=True))

    def describe(dlg, key=None):
        p = lookup.get(dlg.widgets["file"].value)
        if p:
            try:
                sc = Scenario.load(p)
                dlg.info["desc"] = (sc.description or sc.name)[:88]
            except (OSError, ValueError) as exc:     # unusable: say why
                dlg.info["desc"] = f"cannot be opened: {exc}"[:88]

    def on_ok(v):
        p = lookup.get(v["file"])
        if not p:
            return "no scenario selected"
        app.load_scenario(p)
        return None

    def save(dlg):
        v = dlg.values()
        name = "".join(ch for ch in v["save_name"] if ch.isalnum() or ch in "-_ ").strip()
        name = name or "snapshot"
        path = app.user_scenario_dir / f"{name}.json"
        app.sim.snapshot_scenario(name).save(path)
        app.scenario_path = path
        app.toast(f"Saved {path.name}")
        dlg.close()

    def export(dlg):
        stem = app.sim.scenario.name.replace(" ", "_")
        path = app.user_scenario_dir / "exports" / f"{stem}_history.csv"
        n = export_history_csv(app.sim, path)
        app.toast(f"Exported {n} rows to {path}")
        dlg.close()

    def reset(dlg):
        app.reset()
        dlg.close()

    def to_matlab(dlg):
        dlg.close()
        app.open("matlab")

    dlg = FormDialog(app, "Scenarios", specs, on_ok, "Load", width=620, on_change=describe,
                     extra_buttons=[
                         ("Save snapshot", save, "Save every satellite as it is now, as a new "
                                                 "scenario starting at this moment"),
                         ("Export CSV", export, "Write the recorded history (time, position, "
                                                "velocity of each satellite) to a CSV file"),
                         ("Export MATLAB", to_matlab, "Write this simulation as a MATLAB live "
                                                      "script that runs it again in MATLAB"),
                         ("Reset current", reset, "Restart the current scenario from its "
                                                  "beginning (Ctrl+R)")],
                     ok_hint="Replace what is running with the chosen scenario",
                     subtitle="Load a scenario, save the current state (epoch = now), "
                              "or export history")
    describe(dlg)
    return dlg


def _short_path(path: Path, limit: int) -> str:
    """``path`` for a one-line label: relative to the project folder when inside
    it, and at most ``limit`` characters, the middle elided (the end names the
    folder)."""
    try:
        text = str(path.relative_to(ROOT))
    except ValueError:
        text = str(path)
    if len(text) <= limit:
        return text
    head = (limit - 3) // 3
    return text[:head] + "..." + text[len(text) - (limit - 3 - head):]


def matlab_export_dialog(app):
    """Write the simulation as a MATLAB live script (Scenarios > Export MATLAB)."""
    sim = app.sim
    # time runs on behind the dialog: "now" is the moment Export is pressed
    starts = {"Current moment": "now", "Scenario start": "start"}
    labels = list(starts)
    contents = {"With a cross-check": True, "MATLAB only": False}
    kinds = list(contents)
    folder = app.user_scenario_dir / "exports"
    specs = [
        F("start", "Start from", "choice", labels[0], labels,
          tip="Where the MATLAB run begins: every satellite as it is now, with the burns "
              "still to come, or the scenario as it was loaded.",
          option_tips={labels[0]: "Like Save snapshot: positions and velocities at the moment "
                                  "you press Export.",
                       labels[1]: "The scenario from its epoch, as Reset would restart it."}),
        F("compare", "Script", "choice", kinds[0], kinds,
          tip="Whether the script also runs this simulator on the same scenario and compares "
              "the two runs, or is just the MATLAB simulation.",
          option_tips={kinds[0]: "A last section runs this simulator (Python, from this "
                                 "folder) on the same scenario and plots the differences. "
                                 "It can be switched off in the script.",
                       kinds[1]: "Only the MATLAB simulation and its plots: no Python, no "
                                 "comparison. Runs anywhere MATLAB does."}),
        F("duration", "Run length", "text", "1d",
          tip="How far the script propagates: seconds, or 90m, 12h, 3d. It can be changed "
              "in the script."),
        F("step", "Sample every", "text", "60",
          tip="Seconds between output samples (or 5m, 1h)."),
        F("name", "Script name", "text", matlab_name(sim.scenario.name),
          tip="Letters, digits and underscores, starting with a letter: MATLAB runs a "
              "script by its name."),
        F("where", "", "info", f"Written to {_short_path(folder, 50)}",
          tip=f"{folder}. Open it in MATLAB R2025a or later and press Run. The simulation "
              "needs no Python; only the cross-check, if included, runs this simulator."),
    ]

    def on_ok(v):
        name = v["name"].strip()
        if matlab_name(name) != name:
            return f"'{name}' cannot name a MATLAB script; try {matlab_name(name)}"
        start = starts[v["start"]]
        compare = contents[v["compare"]]
        path = export_live_script(sim, folder / f"{name}.m", start,
                                  parse_duration(v["duration"]), parse_duration(v["step"]),
                                  compare)
        when = f"from T+{format_duration(sim.t)}" if start == "now" else "from the start"
        what = "with a cross-check" if compare else "MATLAB only"
        app.toast(f"Wrote {path.name} ({when}, {what}) to {path.parent}")
        return None

    return FormDialog(app, "Export to MATLAB", specs, on_ok, "Export", width=560,
                      ok_hint="Write the live script",
                      subtitle="A live script that runs this simulation again in MATLAB")


# --- Edit satellite -------------------------------------------------------------------------------

def edit_satellite_dialog(app):
    """Rename the selected satellite or change its drag properties (Ctrl+E)."""
    sim = app.sim
    if not 0 <= app.selected < sim.n:
        app.toast("Select a satellite first")
        return None
    s = sim.sats[app.selected]
    specs = [
        F("name", "Name", "text", s.name,
          tip="A new name; planned maneuvers follow the satellite."),
        F("mass", "Mass (kg)", "float", f"{s.mass:g}", tip=MASS_TIP),
        F("dry_mass", "Dry mass (kg)", "float", f"{s.dry_mass:g}", tip=DRY_TIP),
        F("area", "Area (m^2)", "float", f"{s.area:g}", tip=AREA_TIP),
        F("cd", "Drag coefficient Cd", "float", f"{s.cd:g}", tip=CD_TIP),
        F("cr", "Reflectivity Cr", "float", f"{s.cr:g}", tip=CR_TIP),
    ]

    def on_ok(v):
        if min(v["mass"], v["area"], v["cd"]) <= 0:
            return "values must be positive"
        if not 0 <= v["dry_mass"] <= v["mass"]:
            return "dry mass must be between 0 (no limit) and the mass"
        check_cr(v["cr"], f"satellite {s.name!r}")     # ValueError: shown, nothing applied
        sim.rename(sim.sats.index(s), v["name"] or s.name)
        s.mass, s.area, s.cd, s.dry_mass = v["mass"], v["area"], v["cd"], v["dry_mass"]
        s.cr = v["cr"]
        return None

    return FormDialog(app, "Edit satellite", specs, on_ok, "Apply",
                      ok_hint="Apply the changes to the selected satellite")
