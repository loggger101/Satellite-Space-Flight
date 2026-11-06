"""Export a simulation as a MATLAB live script that runs it again in MATLAB.

    python -m satflight.matlab_export scenarios/hohmann_to_geo.json --out hohmann.m
    python -m satflight.matlab_export scenarios/default.json --at 3h --out iss_later.m

The script is a plain-text Live Script (the ``.m`` Live Code format of MATLAB
R2025a and later): it opens in the Live Editor with headings and explanations,
and runs as an ordinary script in any MATLAB that reads the format. It holds

* the satellites' exact ECI states at its start, their mass and drag
  properties, every scheduled maneuver (resolved to seconds after the epoch),
  the force model, propagator and integrator tolerances, ground stations;
* every launch: the vehicle and its stages, the site, the target orbit and
  timing, the simulator's plan for it (liftoff time, plane, pitch kick) and,
  for a vehicle exported in powered flight, its whole flight state;
* the simulator's physics ported to MATLAB (``satflight/matlab_runtime.m``,
  appended as local functions): the propagators and forces, the rocket ascent
  (pad, vertical rise, pitch kick, gravity turn, closed-loop guidance, staging,
  fairing, separation, spent stages as objects, circularization), the launch
  planner, and the event log (burns, eclipses, station passes, close
  approaches, re-entries), so the run needs MATLAB alone;
* a cross-check section that runs the Python simulator on the same scenario
  through the ``matlab/`` toolkit and compares the trajectories and the event
  logs line by line.

It starts either at the moment shown in the app (``start="now"``, every
satellite as it is, like Save snapshot, with launches on their pads or in
flight carried on) or at the start of the scenario (``start="start"``, the
scenario as loaded).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

from .batch import parse_duration
from .launch import PHASE_LABELS
from .scenario import Scenario
from .simulation import Simulation
from .strict import one_of
from .timeutil import UTC, ensure_utc, format_duration

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = Path(__file__).with_name("matlab_runtime.m")
COASTLINES = ROOT / "assets" / "coastlines.json"
STARTS = ("now", "start")
NAME_MAX = 63          # MATLAB's namelengthmax: a longer script name cannot be run


def matlab_name(text: str) -> str:
    """``text`` made into a name MATLAB can run a script by: letters, digits and
    underscores, starting with a letter, at most 63 characters."""
    name = re.sub(r"\W+", "_", text, flags=re.ASCII).strip("_")
    if not name or not name[0].isalpha():
        name = "sim_" + name
    return name[:NAME_MAX].rstrip("_")


def _str(text) -> str:
    """A MATLAB string literal holding ``text`` (line breaks become spaces)."""
    text = " ".join(str(text).splitlines())
    return '"' + text.replace('"', '""') + '"'


def _num(x) -> str:
    """A MATLAB number literal that reads back as exactly the same double."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot export the non-finite number {x}")
    text = repr(x)
    return text[:-2] if text.endswith(".0") else text


def _vec(xs) -> str:
    return "[" + " ".join(_num(x) for x in xs) + "]"


def _bool(b) -> str:
    return "true" if b else "false"


def _text(markdown: str) -> list[str]:
    """Live Script text lines (one paragraph per non-empty line of ``markdown``)."""
    return [f"%[text] {line}" if line else "%[text]" for line in markdown.split("\n")]


def _chunks(text: str, size: int = 150) -> list[str]:
    return [text[k:k + size] for k in range(0, len(text), size)] or [""]


def _coastline_rows(min_step: float = 0.5) -> list[str]:
    """``assets/coastlines.json`` thinned to points at least ``min_step`` degrees
    apart, as rows of a MATLAB matrix with NaN between polylines."""
    rows = []
    for line in json.loads(COASTLINES.read_text(encoding="utf-8")):
        kept = [line[0]]
        for p in line[1:-1]:
            if math.dist(p, kept[-1]) >= min_step:
                kept.append(p)
        kept.append(line[-1])
        rows += [f"{p[0]:.1f} {p[1]:.1f}" for p in kept]
        rows.append("NaN NaN")
    return rows


def _epoch_text(epoch: datetime) -> str:
    """ISO-8601 UTC with the microseconds kept: a snapshot's epoch falls between
    whole seconds, and dropping the fraction would turn the Earth under it."""
    return ensure_utc(epoch).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def export_live_script(sim: Simulation, path, start: str = "now", duration: float = 86400.0,
                       step: float = 60.0) -> Path:
    """Write ``sim`` as a MATLAB live script at ``path`` (see the module notes).
    ``start`` is "now" or "start"; ``duration`` and ``step`` (s) are the script's
    initial settings. Returns the path. ValueError if the file name is not one
    MATLAB can run a script by, or a setting is unusable."""
    one_of(start, STARTS, "export start")
    if not (math.isfinite(duration) and duration >= 0):
        raise ValueError(f"duration must be zero or more seconds, got {duration}")
    if not (math.isfinite(step) and step > 0):
        raise ValueError(f"step must be a positive number of seconds, got {step}")
    path = Path(path)
    if path.suffix != ".m" or matlab_name(path.stem) != path.stem:
        raise ValueError(f"{path.name}: a MATLAB script needs a .m name made of letters, "
                         "digits and underscores, starting with a letter "
                         f"(e.g. {matlab_name(path.stem)}.m)")

    if start == "now":
        scenario = sim.snapshot_scenario(sim.scenario.name)
        origin = (f"the moment it was exported, T+{format_duration(sim.t)} into "
                  f"*{sim.scenario.name}*")
    else:
        scenario = sim.scenario.copy()
        origin = "the start of the scenario"
    data = scenario.to_dict()
    data["epoch"] = _epoch_text(scenario.epoch)
    Scenario.from_dict(json.loads(json.dumps(data, allow_nan=False)))  # the cross-check reads it
    src = Simulation(scenario)

    vehicles = {id(a.sat) for a in src.ascents}
    rows = [(i, s) for i, s in enumerate(src.sats) if id(s) not in vehicles]
    maneuvers = list(src.maneuvers)
    scrubbed = [e.text for e in src.events if e.kind == "alert" and e.text.startswith("Launch of ")]
    if not rows and not src.ascents:
        raise ValueError("nothing to export: no satellite and no launch")
    lines = _script(src, scenario, data, rows, maneuvers, scrubbed, origin, duration, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _numx(x) -> str:
    """Like :func:`_num`, but an infinite or undefined number is written as such."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return _num(x)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' * (n != 1)}"


MODES = {None: 0, "prograde": 1, "fixed": 2}    # thrust modes as the MATLAB ascent codes them


def _launch(a) -> list[str]:
    """The MATLAB lines that put launch ``a`` (an Ascent of the exported run) in ``sim``."""
    spec, v, r = a.spec, a.spec.vehicle, a.res
    where = spec.site or f"({spec.lat:.3f}, {spec.lon:.3f})"
    if a.resumed:
        status = (f"in powered flight at T+{a.met:.1f} s ({PHASE_LABELS[a.phase].lower()}, "
                  f"stage {a.k + 1}): it carries on from the state saved below")
    elif a.t0 > 1e-6:
        status = f"lifts off at T+{format_duration(a.t0)}"
    else:
        status = "lifts off at once"
    out = _text(f"**{spec.name}**: {v.name} from {where}, {status}.")
    out += ["% stages: thrust (kN, vacuum), Isp vacuum (s), Isp sea level (s, 0 for a vacuum "
            "engine), propellant (kg), dry mass (kg)",
            "stages = ["]
    out += [f"    {_num(s.thrust)} {_num(s.isp_vac)} {_num(s.isp_sl)} {_num(s.propellant)} "
            f"{_num(s.dry)}   % {s.name}" for s in v.stages] + ["];"]
    names = ", ".join(_str(s.name) for s in v.stages)
    out += [f"L = satflightLaunch({_str(spec.name)}, satflightVehicle({_str(v.name)}, [{names}], "
            f"stages, {_num(v.fairing)}, {_num(v.fairing_alt)}, {_num(v.diameter)}, {_num(v.cd)}, "
            f"{_num(v.max_g)}, {_num(v.stage_coast)}));   % fairing kg, jettison km, "
            "diameter m, Cd, g limit (0 none), s between stages",
            f"L.site = {_str(spec.site)}; L.lat = {_num(spec.lat)}; L.lon = {_num(spec.lon)}; "
            f"L.alt = {_num(spec.alt)};   % geodetic deg, km",
            f"L.timing = {_str(spec.timing)}; L.delay = {_num(spec.delay)}; "
            f"L.t0 = {_numx(math.nan if spec.t0 is None else spec.t0)};   "
            "% now | delay | absolute | raan | plane",
            f"L.guidance = {_str(spec.guidance)}; L.perigeeAlt = {_num(spec.perigee_alt)}; "
            f"L.apogeeAlt = {_num(spec.apogee_alt)}; L.inclination = {_num(spec.inclination)}; "
            f"L.direction = {_str(spec.direction)};",
            f"L.raan = {_num(spec.raan)}; L.target = {_str(spec.target)}; "
            f"L.azimuth = {_num(spec.azimuth)}; L.verticalTime = {_num(spec.vertical_time)}; "
            f"L.kick = {_numx(math.nan if spec.kick is None else spec.kick)};   "
            "% kick NaN: searched by the planner",
            f"L.payloadMass = {_num(spec.payload_mass)}; "
            f"L.payloadArea = {_num(spec.payload_area)}; L.payloadCd = {_num(spec.payload_cd)}; "
            f"L.circularize = {_bool(spec.circularize)}; "
            f"L.trackStages = {_bool(spec.track_stages)};"]
    n = "[]" if r.n is None else _vec(r.n)
    out += ["% the simulator's plan: liftoff (s), azimuth (rad), plane normal, inclination (deg), "
            "insertion radius (km), speed (km/s), orbital energy, pitch kick (deg)",
            f"L.plan = satflightPlan({_num(r.t0)}, {_num(r.azimuth)}, {n}, "
            f"{_numx(r.inclination)}, {_numx(r.r_target)}, {_numx(r.v_target)}, "
            f"{_numx(r.e_target)}, {_num(a.kick_deg)});"]
    if a.resumed:
        s = a.save(0.0)["state"]
        c = s["ctrl"]
        ctrl = "[]" if c is None else (
            f"struct(\"mdot\", {_num(c['mdot'])}, \"stage\", {c['stage'] + 1}, "
            f"\"mode\", {MODES[c['mode']]}, \"fixed\", "
            f"{_vec(c['fixed'] if c['fixed'] is not None else (0.0, 0.0, 0.0))})")
        hold = "[]" if s["hold"] is None else _vec(s["hold"])
        lo = s["losses"]
        out += ["% its flight state when the run was exported",
                f"L.state = struct(\"t\", {_num(s['t'])}, \"y\", {_vec(s['y'])}, "
                f"\"phase\", {_str(s['phase'])}, \"stage\", {s['stage'] + 1}, ...",
                f"    \"prop\", {_num(s['prop'])}, \"mass\", {_num(s['mass'])}, "
                f"\"fairingOn\", {_bool(s['fairing_on'])}, \"burning\", {_bool(s['burning'])}, ...",
                f"    \"coastUntil\", {_numx(_none(s['coast_until'], -math.inf))}, "
                f"\"throttle\", {_num(s['throttle'])}, \"thrustDir\", {_vec(s['thrust_dir'])}, ...",
                f"    \"ctrl\", {ctrl}, \"hold\", {hold}, ...",
                f"    \"tgo\", {_numx(_none(s['tgo'], math.nan))}, "
                f"\"kickStart\", {_numx(_none(s['kick_start'], math.nan))}, "
                f"\"q\", {_num(s['q'])}, \"maxQ\", {_num(s['max_q'])}, ...",
                f"    \"tMaxQ\", {_numx(_none(s['t_max_q'], math.nan))}, "
                f"\"qLogged\", {_bool(s['q_logged'])}, \"accel\", {_num(s['accel'])}, "
                f"\"dvIdeal\", {_num(s['dv_ideal'])}, ...",
                f"    \"losses\", {_vec([lo['gravity'], lo['drag'], lo['steering']])}, "
                f"\"vStart\", {_num(s['v_start'])}, "
                f"\"nextSample\", {_numx(_none(s['next_sample'], -math.inf))});"]
    out += ["sim.launches(end + 1) = L;"]
    return out


def _none(x, default: float) -> float:
    return default if x is None else x


def _script(src, scenario, data, rows, maneuvers, scrubbed, origin, duration, step) -> list[str]:
    F, it = src.forces, scenario.integrator
    epoch = ensure_utc(scenario.epoch)
    exported = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    launches = list(src.ascents)
    what = [_plural(len(rows), "satellite")]
    if launches:
        what.append(_plural(len(launches), "launch").replace("launchs", "launches"))
    what.append(_plural(len(maneuvers), "scheduled maneuver"))
    out = _text(
        f"# {scenario.name}\n"
        f"Exported from **Satellite Space Flight** on {exported}. The run starts at {origin}: "
        f"{epoch.strftime('%Y-%m-%d %H:%M:%S')} UTC, with {', '.join(what[:-1])} and {what[-1]}.\n"
        "Press **Run** to run it here in MATLAB with the simulator's own physics"
        + (" and rocket ascents" if launches else "")
        + ", plot the results and, when the simulator is installed, run it there too and "
        "compare.")
    out += ["%%"]
    out += _text("## Settings\nHow far to propagate and how often to sample (seconds). "
                 "Change them freely.")
    out += [f"duration = {_num(duration)};   % s ({format_duration(duration)})",
            f"step = {_num(step)};   % s between samples",
            "crossCheck = true;   % also run the Python simulator and compare (last section)"]
    if launches:
        out += ["planLaunches = false;   % true: plan the launches on their pads here (window, "
                "pitch kick), e.g. after editing one"]
    out += ["%%"]

    method = it.method
    if src.propagator == "cowell":
        how = (f"numerical integration of every enabled force. The simulator used {method} "
               f"(rtol {it.rtol:g}, atol {it.atol:g}, steps up to {it.h_max:g} s); this script "
               "uses ode45, the same Dormand-Prince 5(4) family, with those tolerances.")
    elif src.propagator == "kepler":
        how = (f"exact two-body motion (universal variables), in steps of {it.h_max:g} s as the "
               "simulator takes them. Finite burns are applied at once.")
    else:
        how = (f"analytic two-body motion plus the J2 secular drift of the node, perigee and "
               f"mean anomaly, in steps of {it.h_max:g} s. Finite burns are applied at once.")
    out += _text(f"## Scenario\n**Forces:** {F.label()}"
                 + (f" (atmosphere density x{F.density_scale:g})" if F.drag else "")
                 + f". **Propagator:** {src.propagator}, {how}")
    out += ["sim = struct();",
            f"sim.name = {_str(scenario.name)};",
            f"sim.epoch = datetime({epoch.year}, {epoch.month}, {epoch.day}, {epoch.hour}, "
            f"{epoch.minute}, {_num(epoch.second + epoch.microsecond / 1e6)}, "
            'TimeZone="UTC");',
            f"sim.jd0 = {_num(src.clock.jd0)};   % Julian date of the epoch",
            f"sim.propagator = {_str(src.propagator)};",
            f"sim.forces = struct(\"j2\", {_bool(F.j2)}, \"j3\", {_bool(F.j3)}, "
            f"\"j4\", {_bool(F.j4)}, \"c22\", {_bool(F.c22)}, \"drag\", {_bool(F.drag)}, "
            f"\"densityScale\", {_num(F.density_scale)});",
            f"sim.integrator = struct(\"method\", {_str(method)}, \"rtol\", {_num(it.rtol)}, "
            f"\"atol\", {_num(it.atol)}, \"hMax\", {_num(it.h_max)});"]

    if rows:
        out += _text("Satellites at the epoch: ECI position (km) and velocity (km/s), mass (kg), "
                     "drag area (m^2) and drag coefficient.")
        out += ["sim.names = ["] + [f"    {_str(s.name)}" for _, s in rows] + ["];"]
        out += ["% x, y, z (km), vx, vy, vz (km/s)", "sim.y0 = ["]
        out += ["    " + " ".join(_num(x) for x in src.y[i]) for i, _ in rows] + ["];"]
        out += ["% mass (kg), area (m^2), Cd", "props = ["]
        out += [f"    {_num(s.mass)} {_num(s.area)} {_num(s.cd)}" for _, s in rows] + ["];"]
    else:
        out += _text("No satellites in orbit at the epoch: everything starts on a launch pad.")
        out += ["sim.names = strings(0, 1);", "sim.y0 = zeros(0, 6);", "props = zeros(0, 3);"]
    out += ["sim.mass = props(:, 1);", "sim.area = props(:, 2);", "sim.cd = props(:, 3);"]

    if maneuvers:
        out += _text("Scheduled maneuvers, in the order they execute. Columns: satellite, kind, "
                     "time (s after the epoch), dV or thrust direction (km/s) in the frame, "
                     "frame, plane change (deg), target, thrust (N), Isp (s), burn time (s), "
                     "label.")
        out += ["sim.maneuvers = ["]
        for m in maneuvers:
            args = [_str(m.sat), _str(m.kind), _num(m.t), _vec(m.dv), _str(m.frame),
                    _num(m.delta_i), _str(m.target), _num(m.thrust), _num(m.isp),
                    _num(m.duration), _str(m.describe())]
            out.append(f"    satflightBurn({', '.join(args)})")
        out += ["];"]
    else:
        out += _text("No maneuvers are scheduled.")
        out += ["sim.maneuvers = satflightNoBurns();"]

    stations = src.stations
    out += _text("Ground stations (marked on the ground track; passes are logged as AOS and LOS "
                 "above each one's elevation mask):" if stations else "No ground stations.")
    out.append('sim.stations = struct("name", {{{}}}, "lat", {{{}}}, "lon", {{{}}}, '
               '"alt", {{{}}}, "minEl", {{{}}});'.format(
                   ", ".join(_str(g.name) for g in stations),
                   ", ".join(_num(g.lat) for g in stations),
                   ", ".join(_num(g.lon) for g in stations),
                   ", ".join(_num(g.alt) for g in stations),
                   ", ".join(_num(g.min_el) for g in stations)))
    out += ["sim.launches = satflightNoLaunches();"]
    if launches:
        out += ["%%"]
        out += _text(
            "## Launches\nEach vehicle waits on its pad, turning with the Earth, and lifts off at "
            "the planned time. It then flies the simulator's ascent: a vertical rise, the pitch "
            "kick, a gravity turn through the dense air, then closed-loop guidance to the target "
            "orbit, with staging, fairing jettison and an acceleration limit, until its payload "
            "separates and joins the run. Spent stages fly on as objects of their own. The plan "
            "(launch window and pitch kick) is the simulator's; set `planLaunches = true` in "
            "Settings to plan again here after editing a launch.")
        for a in launches:
            out += _launch(a)
    if scrubbed:
        out += _text("Scrubbed by the simulator when the scenario loaded (left out here too): "
                     + "; ".join(scrubbed))
    out += ["%%"]

    out += _text("## Run\nOne row per object per sample, in the columns of the simulator's "
                 "ephemeris CSV (`satflight.readRun` reads the same); the event log, line for "
                 "line as the simulator writes it (key events shown)"
                 + ("; and per launch its outcome, insertion orbit, delta-v budget and ascent "
                    "profile." if launches else "."))
    plan = ", PlanLaunches=planLaunches" if launches else ""
    out += [f"[T, events, launches] = satflightRun(sim, duration, step{plan});",
            'disp(events(ismember(events.kind, ["launch", "maneuver", "alert", "warn"]), :))',
            "head(T, max(1, numel(sim.names)))"]
    if launches:
        out += ['struct2table(rmfield(launches, ["insertion", "losses_kms", "profile", "marks", '
                '"planned", "plannedMarks"]), AsArray=true)']
    out += ["%%"]
    out += _text("## Plots\nGround track, altitude, orbital elements and the orbits in 3-D"
                 + (", and each launch's ascent" if launches else "")
                 + ". Each takes a list of names, e.g. `satflightPlotAltitude(T, \"ISS\")`.")
    out += ["satflightPlotGroundTrack(T, sim.stations);",
            "satflightPlotAltitude(T);",
            "satflightPlotElements(T);",
            "satflightPlotOrbits(T);"]
    if launches:
        out += ["satflightPlotAscent(launches);"]
    out += ["%%"]
    out += _text("## Cross-check against the simulator\nRuns the Python simulator on the same "
                 "scenario (through the `matlab/` toolkit of the folder below), plots how far "
                 "apart the two runs are and pairs up the two event logs line by line (shown: "
                 "the lines that differ). It needs Satellite Space Flight and Python with numpy; "
                 "set `crossCheck = false` in Settings to skip it.")
    out += [f"simulatorFolder = {_str(ROOT)};",
            "if crossCheck",
            "    [P, E] = satflightRunSimulator(simulatorFolder, scenarioJson(), duration, step);",
            "    differences = satflightCompare(T, P)",
            "    eventDifferences = satflightCompareEvents(events, E);",
            "    eventDifferences(~eventDifferences.same_text, :)",
            "end",
            ""]

    out += ["function text = scenarioJson()",
            "% the scenario as the simulator reads it (for the cross-check)",
            "text = strjoin(["]
    out += [f"    {_str(c)}" for c in _chunks(json.dumps(data, separators=(",", ":")))]
    out += ['], "");', "end", ""]
    out += ["function c = satflightCoastline()",
            "% coastlines (Natural Earth, public domain), thinned to 0.5 degree",
            "c = ["] + [f"    {r}" for r in _coastline_rows()] + ["];", "end", ""]
    out += RUNTIME.read_text(encoding="utf-8").rstrip("\n").split("\n")
    out += ["", '%[appendix]{"version":"1.0"}', "%---", "%[metadata:view]",
            '%   data: {"layout":"inline"}', "%---"]
    return out


def main(argv=None):
    """Command-line entry point (``python -m satflight.matlab_export``)."""
    ap = argparse.ArgumentParser(description="Write a scenario as a MATLAB live script.")
    ap.add_argument("scenario", help="scenario JSON file")
    ap.add_argument("--out", required=True, help="the .m file to write")
    ap.add_argument("--at", help="run this long first (e.g. 90m, 3h) and export that moment; "
                                 "without it the export starts at the scenario's start")
    ap.add_argument("--duration", default="1d", help="the script's run length (default 1d)")
    ap.add_argument("--step", default="60", help="the script's sample interval (default 60 s)")
    args = ap.parse_args(argv)
    try:
        sc = Scenario.load(args.scenario)
    except (OSError, ValueError) as exc:
        ap.error(f"{args.scenario}: {exc}")
    try:
        at = parse_duration(args.at) if args.at else None
        duration, step = parse_duration(args.duration), parse_duration(args.step)
    except ValueError as exc:
        ap.error(str(exc))
    if at is not None and at < 0:
        ap.error(f"--at cannot be negative ({args.at})")
    sim = Simulation(sc)
    if at:
        sim.advance(at)
    try:
        path = export_live_script(sim, args.out, "now" if at is not None else "start",
                                  duration, step)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
