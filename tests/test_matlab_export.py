"""Scenarios > Export MATLAB: the live script carries the simulation exactly, its
MATLAB port of the engine uses the simulator's constants, and unusable input is
refused. (Running the scripts in MATLAB is matlab/tests/ExportTest.m.)"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from satflight import atmosphere, constants
from satflight.maneuvers import Maneuver
from satflight.matlab_export import (
    CROSS_CHECK,
    FIXED_STEP,
    RUNTIME,
    _epoch_text,
    drop_functions,
    export_live_script,
    main,
    matlab_name,
    runtime_lines,
)
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.json"))


def block(text: str, name: str) -> list[str]:
    """Rows of the MATLAB matrix ``name = [ ... ];`` in ``text`` (comments cut off),
    or the one value of ``name = value;``."""
    m = re.search(rf"^{re.escape(name)} = \[\n(.*?)\n\];", text, re.S | re.M)
    if not m:
        m = re.search(rf"^{re.escape(name)} = ([^\[\n].*);$", text, re.M)
    assert m, f"no {name} block"
    return [r.split("   % ")[0].strip() for r in m.group(1).split("\n")]


def matlab_strings(lines: list[str]) -> list[str]:
    """The texts of MATLAB string literals, one per line."""
    out = []
    for line in lines:
        m = re.fullmatch(r'\s*"((?:[^"]|"")*)"', line)
        assert m, line
        out.append(m.group(1).replace('""', '"'))
    return out


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_exports_its_exact_starting_state(path, tmp_path):
    sim = Simulation(Scenario.load(path))
    out = export_live_script(sim, tmp_path / "run_it.m", "start", 3600, 60)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("%[text] # ")
    assert lines[-5:] == ['%[appendix]{"version":"1.0"}', "%---", "%[metadata:view]",
                          '%   data: {"layout":"inline"}', "%---"]
    assert ("function [T, events, launches, burns, objects] = satflightRun(sim, duration, "
            "step, opts)") in text
    vehicles = {id(a.sat) for a in sim.ascents}
    rows = [i for i, s in enumerate(sim.sats) if id(s) not in vehicles]
    if rows:
        names = [r.strip('"').replace('""', '"') for r in block(text, "sim.names")]
        assert names == [sim.sats[i].name for i in rows]
        y0 = np.array([[float(x) for x in r.split()] for r in block(text, "sim.y0")])
        assert np.array_equal(y0, sim.y[rows])                   # bit for bit
    burns = block(text, "sim.maneuvers") if sim.maneuvers else []
    times = [float(re.match(r'satflightBurn\("[^"]*", "[^"]*", ([^,]+),', b).group(1))
             for b in burns]
    assert times == [m.t for m in sim.maneuvers]
    plans = re.findall(r"^L.plan = satflightPlan\(([^,]+), ([^,]+), .*, ([^,]+)\);$", text, re.M)
    assert [(float(t0), float(az), float(kick)) for t0, az, kick in plans] == [
        (a.t0, a.res.azimuth, a.kick_deg) for a in sim.ascents]   # bit for bit
    for a in sim.ascents:
        assert f'L = satflightLaunch("{a.spec.name}", satflightVehicle(' in text


ISS_TLE = ("1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
           "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537")


def every_option(propagator: str, method: str, h_fixed: float = 10.0) -> dict:
    """A scenario that reaches every option the exporter must carry: each orbit type,
    every maneuver kind, timing and frame, every launch timing and guidance and each
    vehicle, a constellation, ground stations and names MATLAB must quote."""
    def el(**kw):
        return {"type": "elements", **kw}

    sats = [
        {"name": "Elements", "orbit": el(altitude=500, i=40, raan=10, nu=5), "mass": 800,
         "area": 4, "dry_mass": 600},
        {"name": "Ellipse", "orbit": el(perigee_alt=400, apogee_alt=2000, i=60, raan=50,
                                        argp=30, nu=100)},
        {"name": "By a and e", "orbit": el(a=26554, e=0.72, i=63.4, raan=0, argp=270, M=10)},
        {"name": "SSO", "orbit": el(altitude=700, i="sso", raan=30)},
        {"name": "State", "orbit": {"type": "state", "r": [7000, 0, 0], "v": [0, 7.5, 1]}},
        {"name": "Hop", "orbit": {"type": "surface", "lat": 28.5, "lon": -80.6, "alt": 100,
                                  "speed": 4.0, "azimuth": 90, "fpa": 45}},
        {"name": "GEO", "orbit": {"type": "geo", "lon": -75}},
        {"name": "ISS (TLE)", "orbit": {"type": "tle", "line1": ISS_TLE[0],
                                        "line2": ISS_TLE[1]}},
        {"name": 'Quote "Q" & é', "orbit": el(altitude=600, i=97, raan=200)},
        {"name": "Copies", "orbit": el(altitude=550, i=53), "count": 3},
    ]
    timings = ("now", "delay", "periapsis", "apoapsis", "ascending_node", "descending_node",
               "absolute")
    maneuvers = [{"sat": "Ellipse", "kind": "impulse", "timing": tm, "delay": 600.0,
                  "t": 900.0 + 100 * k, "dv": [0.002, 0.001, -0.001], "frame": frame}
                 for k, (tm, frame) in enumerate(zip(timings, ("VNB", "RSW", "ECI") * 3,
                                                     strict=False))]
    maneuvers += [
        {"sat": "Elements", "kind": "finite", "timing": "delay", "delay": 300,
         "dv": [1, 0, 0.2], "thrust": 400, "isp": 310, "duration": 900, "frame": "RSW"},
        {"sat": "SSO", "kind": "plane_change", "timing": "descending_node", "delta_i": 0.5},
        {"sat": "By a and e", "kind": "circularize", "timing": "apoapsis"},
        {"sat": 'Quote "Q" & é', "kind": "match_velocity", "timing": "delay", "delay": 2000,
         "target": "SSO"},
    ]
    launches = [
        {"name": "Now", "vehicle": "Falcon 9 (approx.)", "timing": "now"},
        {"name": "Later", "vehicle": "Electron (approx.)", "timing": "delay", "delay": 120,
         "lat": -39.262, "lon": 177.865, "inclination": 97.5, "direction": "south",
         "perigee_alt": 500, "apogee_alt": 520, "payload_mass": 150, "circularize": True},
        {"name": "At a time", "vehicle": "Saturn V (approx.)", "timing": "absolute", "t0": 200,
         "track_stages": False},
        {"name": "To a node", "vehicle": "Falcon 9 (approx.)", "timing": "raan", "raan": 200,
         "inclination": 53},
        {"name": "Into a plane", "vehicle": "Falcon 9 (approx.)", "timing": "plane",
         "target": "Elements"},
        {"name": "Sounder", "vehicle": "Sounding rocket", "timing": "delay", "delay": 60,
         "guidance": "open", "azimuth": 0, "lat": 69.29, "lon": 16.02},
    ]
    return {
        "name": f"Every option {propagator} {method}", "epoch": "2026-09-29T12:00:00Z",
        "propagator": propagator,
        "integrator": {"method": method, "h_max": 60.0, "h_fixed": h_fixed},
        "forces": {"j2": True, "j3": propagator == "cowell", "j4": propagator == "cowell",
                   "c22": propagator == "cowell", "drag": propagator == "cowell"},
        "satellites": sats,
        "constellations": [{"name": "Walker", "total": 6, "planes": 3, "phasing": 1,
                            "pattern": "star", "altitude": 1200, "inclination": 86}],
        "stations": [{"name": "Kiruna", "lat": 67.86, "lon": 20.96},
                     {"name": "Hobart", "lat": -42.8, "lon": 147.4, "alt": 0.1, "min_el": 5}],
        "maneuvers": maneuvers, "launches": launches,
    }


@pytest.mark.parametrize("propagator, method", [("cowell", "dopri5"), ("cowell", "rk4"),
                                                ("cowell", "wh"), ("cowell", "leapfrog"),
                                                ("kepler", "dopri5"), ("j2mean", "dopri5")])
def test_every_option_exports_from_the_start_and_mid_run(propagator, method, tmp_path):
    """Whatever a scenario holds, the export carries all of it: every object, burn and
    launch, the propagator and the integrator the run really uses (its fixed step
    too), from the start and from a moment with burns and ascents under way."""
    sim = Simulation(Scenario.from_dict(every_option(propagator, method)))
    for start, at in (("start", 0.0), ("now", 333.7)):
        sim.advance(at)
        out = export_live_script(sim, tmp_path / f"every_{start}.m", start, 1800, 60)
        text = out.read_text(encoding="utf-8")
        integrator = re.search(r'^sim\.integrator = struct\("method", "(\w+)", .*"hFixed", '
                               r'([^)]+)\);', text, re.M)
        assert integrator and integrator.groups() == (method, "10")
        assert f'sim.propagator = "{propagator}";' in text
        if propagator == "cowell" and method != "dopri5":
            assert f"fixed-step {FIXED_STEP[method]}, in steps of 10 s" in text
        data = json.loads(scenario_json(text))      # what the script starts from
        names = [r.strip('"').replace('""', '"') for r in block(text, "sim.names")]
        vehicles = {id(a.sat) for a in sim.ascents}
        launches = len(re.findall(r"^sim\.launches\(end \+ 1\) = L;$", text, re.M))
        flying = len(re.findall(r"^L\.state = struct\(", text, re.M))
        assert launches == len(data["launches"])
        assert flying == sum(bool(L.get("resume", {}).get("state")) for L in data["launches"])
        assert len(re.findall(r"^    satflightBurn\(", text, re.M)) == len(data["maneuvers"])
        if start == "start":
            assert names == [s.name for s in sim.sats if id(s) not in vehicles]
            assert launches == 6 and len(data["maneuvers"]) == 11
        else:
            free = [s.name for s in sim.sats if id(s) not in vehicles and s.status == "active"]
            assert names == free
            assert flying >= 1, "nothing exported in flight"


def scenario_json(text: str) -> str:
    """The scenario JSON a script's cross-check hands the simulator."""
    body = re.search(r'^function text = scenarioJson\(\)\n.*?\ntext = strjoin\(\[\n(.*?)\n'
                     r'\], ""\);', text, re.S | re.M).group(1)
    return "".join(matlab_strings(body.split("\n")))


def test_every_integrator_has_a_description():
    from satflight.integrators import METHODS
    assert sorted(FIXED_STEP) == sorted(set(METHODS) - {"dopri5"})


def test_the_scenario_json_breaks_between_items_and_joins_back_exactly(tmp_path):
    """The cross-check's copy of the scenario is cut into lines after commas (not
    inside a name or a number) and reads back as exactly what the simulator loads."""
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "launch_day.json"))
    sim.advance(333.7)
    text = export_live_script(sim, tmp_path / "j.m", "now", 60, 60).read_text(encoding="utf-8")
    m = re.search(r"^text = strjoin\(\[\n(.*?)\n\], \"\"\);", text, re.S | re.M)
    pieces = matlab_strings(m.group(1).split("\n"))
    snap = sim.snapshot_scenario(sim.scenario.name)
    data = snap.to_dict() | {"epoch": _epoch_text(snap.epoch)}
    assert "".join(pieces) == json.dumps(data, separators=(",", ":"))
    assert all(p.endswith(",") for p in pieces[:-1])
    assert max(len(p) for p in pieces) < 400 and len(pieces) > 20


def test_coastlines_are_packed_and_unchanged(tmp_path):
    """Ten points to a line instead of one: the same points, in the same order."""
    from satflight.matlab_export import COASTLINES
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "rendezvous.json"))
    text = export_live_script(sim, tmp_path / "c.m", "start", 60, 60).read_text(encoding="utf-8")
    m = re.search(r"^c = \[ \.\.\.\n(.*?)\n    \];\nc = reshape\(c, 2, \[\]\)\.';$", text,
                  re.S | re.M)
    rows = m.group(1).split("\n")
    assert all(r.endswith(" ...") for r in rows)
    got = np.array([float(x) for r in rows for x in r[:-4].split()]).reshape(-1, 2)
    want = []
    for line in json.loads(COASTLINES.read_text(encoding="utf-8")):
        kept = [line[0]]
        for p in line[1:-1]:
            if np.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= 0.5:
                kept.append(p)
        want += [[round(p[0], 1), round(p[1], 1)] for p in kept + [line[-1]]] + [[np.nan] * 2]
    np.testing.assert_array_equal(got, np.array(want))
    assert len(rows) < len(want) / 9


def test_every_satellite_row_says_what_orbit_it_is_on(tmp_path):
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "launches_and_arcs.json"))
    text = export_live_script(sim, tmp_path / "o.m", "start", 60, 60).read_text(encoding="utf-8")
    m = re.search(r"^sim\.y0 = \[\n(.*?)\n\];", text, re.S | re.M)
    notes = [r.split("   % ")[1] for r in m.group(1).split("\n")]
    assert notes[0].startswith("Hop 15 deg: sub-orbital, apogee ")
    assert re.fullmatch(r"Insertion: [\d,]+ x [\d,]+ km, i \d+\.\d deg", notes[3]), notes[3]
    assert re.fullmatch(r"Escape: escaping \(e \d\.\d\d\), i \d+\.\d deg", notes[4]), notes[4]
    one = Simulation(Scenario.load(ROOT / "scenarios" / "launch_day.json"))
    text = export_live_script(one, tmp_path / "one.m", "start", 60, 60).read_text(encoding="utf-8")
    assert 'sim.names = "ISS";' in text                     # one name: no brackets
    assert 'satflightVehicle("Sounding rocket", "Motor", stages' in text


def headings(text: str) -> list[str]:
    return re.findall(r"^%\[text\] ## (.+)$", text, re.M)


def lone_satellite() -> Scenario:
    return Scenario(name="Alone", satellites=[SatSpec("A", {"type": "elements",
                                                            "altitude": 500})])


@pytest.mark.parametrize("scenario, compare, sections", [
    ("launch_day", True, ["Settings", "Scenario", "Launches", "Run", "Plots",
                          "Passes and eclipses", "Maneuvers and close approaches",
                          "Orbit transfers", "Orbit evolution", "Physics diagnostics",
                          "Animation", "Save", "Cross-check against the simulator"]),
    ("hohmann_to_geo", False, ["Settings", "Scenario", "Run", "Plots", "Passes and eclipses",
                               "Maneuvers and close approaches", "Orbit transfers",
                               "Orbit evolution", "Physics diagnostics", "Animation",
                               "Save"]),
    ("orbit_tour", False, ["Settings", "Scenario", "Run", "Plots", "Passes and eclipses",
                        "Close approaches", "Orbit evolution", "Physics diagnostics",
                        "Animation", "Save"]),
    (None, False, ["Settings", "Scenario", "Run", "Plots", "Passes and eclipses",
                   "Orbit evolution", "Physics diagnostics", "Animation", "Save"])])
def test_sections_in_order(scenario, compare, sections, tmp_path):
    """Every section starts a new %% cell; the passes table only with stations, the
    maneuvers with maneuvers or launches (which schedule circularizations), the close
    approaches with two objects or more, the orbit evolution always, the animation always,
    behind its own setting."""
    sc = Scenario.load(ROOT / "scenarios" / f"{scenario}.json") if scenario else lone_satellite()
    sim = Simulation(sc)
    text = export_live_script(sim, tmp_path / "s.m", "start", 60, 60, compare=compare).read_text(
        encoding="utf-8")
    assert headings(text) == sections
    cells = text.split("\n%%\n")
    assert [re.match(r"%\[text\] ## (.+)", c).group(1) for c in cells[1:]] == sections
    assert ("passes = satflightPasses(T, events, sim.stations);" in text) == bool(sim.stations)
    assert ("maneuvers = satflightManeuvers(events);" in text) == (scenario != "orbit_tour"
                                                                  and scenario is not None)
    assert ("approaches = satflightApproaches(events);" in text) == (scenario is not None)
    assert "eclipses = satflightEclipses(T, events);" in text
    assert "satflightPlotTimeline(T, events, shown);" in text
    assert ("satflightPlotSky(T, events, sim.stations, shown);" in text) == bool(sim.stations)
    assert "drift = satflightDrift(T, events, sim);\ndisplay(drift)" in text
    assert re.search(r'^animate = true;   % .+\nanimationFile = "";   % .+\n'
                     r'animation3DFile = "";   % .+\nfigureFolder = "";   % ', text, re.M)
    assert ("    satflightAnimate(T, events, sim.stations, shown, File=mapFile);\n"
            "    satflightAnimate3D(T, events, burns, shown, File=globeFile);\nend") in text
    assert ('if figureFolder ~= ""\n    savedFiles = satflightSaveFigures(figureFolder);'
            in text)
    burns = scenario is not None and scenario != "orbit_tour"
    assert ("transfers = satflightTransfers(burns);" in text) == burns
    assert ("[diagnostics, conservation] = satflightDiagnostics(T, sim, burns, objects);"
            in text)
    assert ("coverage = satflightCoverage(T, passes, sim.stations);" in text) == bool(
        sim.stations)
    assert "betaAngles = satflightBeta(T);\nsatflightPlotBeta(betaAngles, shown);" in text
    assert re.search(r'^resultsFile = "";   % ', text, re.M)
    # the Save section saves exactly the variables the sections before it make, each
    # named once, and says which in its text
    save = re.search(r'if resultsFile ~= ""\n    save\(resultsFile, \.\.\.\n((?:        .+\n)+)',
                     text).group(1)
    saved = re.findall(r'"(\w+)"', save)
    assert len(saved) == len(set(saved))
    before = text[:text.index("## Save")]
    made = {name for left in re.findall(r"^(\w+|\[[\w, ]+\]) = ", before, re.M)
            for name in re.findall(r"\w+", left)}
    assert set(saved) <= made
    assert {"T", "events", "eclipses", "betaAngles", "drift"} <= set(saved)
    assert ("passes" in saved) == ("coverage" in saved) == bool(sim.stations)
    assert ("launchTable" in saved) == bool(sim.ascents)
    assert "".join(f"`{v}`" for v in saved) == "".join(re.findall(
        r"`\w+`", text[text.index("brings them back): "):text.index("if resultsFile")]))


def test_the_introduction_counts_in_words(tmp_path):
    from satflight.launch import LaunchSpec
    one = Simulation(Scenario.load(ROOT / "scenarios" / "launch_day.json"))
    text = export_live_script(one, tmp_path / "a.m", "start", 60, 60).read_text(encoding="utf-8")
    assert "with 1 satellite in orbit, 6 launches and no scheduled maneuvers." in text
    two = Simulation(Scenario.load(ROOT / "scenarios" / "hohmann_to_geo.json"))
    text = export_live_script(two, tmp_path / "b.m", "start", 60, 60).read_text(encoding="utf-8")
    assert "with 2 satellites in orbit and 2 scheduled maneuvers." in text
    pad = Simulation(Scenario(name="Pad only", launches=[LaunchSpec(
        name="Solo", timing="delay", delay=60.0, kick=2.5)]))
    text = export_live_script(pad, tmp_path / "c.m", "start").read_text(encoding="utf-8")
    assert "with no satellites in orbit, 1 launch and no scheduled maneuvers." in text


TWINLESS = {"satflightTruthGap", "satflightPlotTruthGap"}    # no precise-orbit twin


@pytest.mark.parametrize("scenario, gone", [
    ("hohmann_to_geo", {"satflightNoBurns", "satflightPlan", "satflightPlotAscent",
                        "mergeMarks", "satflightPasses", "elevationDeg", "satflightPlotSky",
                        "satflightCoverage"} | TWINLESS),
    # the maneuvers stay without burns: the orbit evolution reads them; the transfer
    # section goes with the burns
    ("orbit_tour", {"satflightPlan", "satflightPlotAscent", "mergeMarks", "satflightTransfers",
                 "satflightPlotTransfer", "satflightPlotDeltaV", "satflightPlotBurnWindow",
                 "burnMarks", "onlyFirst", "plural", "orbitText", "orbitSummary",
                 "julianUtc"} | TWINLESS),
    ("launch_day", TWINLESS)])
def test_a_script_holds_only_the_helpers_it_calls(scenario, gone, tmp_path):
    """Helpers the sections never call are left out (MATLAB marks an unused local
    function); every function is defined once, and the rest are all there."""
    sim = Simulation(Scenario.load(ROOT / "scenarios" / f"{scenario}.json"))
    text = export_live_script(sim, tmp_path / "h.m", "start", 60, 60).read_text(encoding="utf-8")
    defined = re.findall(r"^function (?:.*= *)?(\w+)\(", text, re.M)
    assert len(defined) == len(set(defined))
    everything = set(re.findall(r"^function (?:.*= *)?(\w+)\(", RUNTIME.read_text(
        encoding="utf-8"), re.M)) | {"scenarioJson", "satflightCoastline"}
    assert everything - set(defined) == gone


def test_dropping_a_function_needs_it_whole(tmp_path):
    lines = ["function a = one(x)", "a = x;", "end", "", "function b = two(x)",
             "for k = 1:2", "end", "b = x;", "end", "", "% --- tail", "function c = three()",
             "c = 1;"]
    assert drop_functions(lines, ["two"]) == lines[:4] + lines[10:]
    assert drop_functions(lines, ["one"])[0] == "function b = two(x)"
    with pytest.raises(RuntimeError, match="expected one function four"):
        drop_functions(lines, ["four"])
    with pytest.raises(RuntimeError, match="does not end in 'end'"):
        drop_functions(lines, ["three"])


def test_current_moment_export_starts_where_the_app_is(tmp_path):
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "orbit_tour.json"))
    sim.advance(3 * 3600 + 17.25)
    text = export_live_script(sim, tmp_path / "later.m", "now").read_text(encoding="utf-8")
    assert "T+03:00:17.2" in text
    assert "datetime(2026, 9, 29, 15, 0, 17.25" in text             # the fraction is kept
    y0 = np.array([[float(x) for x in r.split()] for r in block(text, "sim.y0")])
    assert np.array_equal(y0, sim.y)


def test_launches_in_flight_carry_their_flight_state(tmp_path):
    """Exported mid-ascent, a vehicle keeps its stage, propellant, steering and
    the step grid of its ascent; one on its pad keeps the simulator's plan."""
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "launch_day.json"))
    sim.advance(333.7)
    flying = [a for a in sim.ascents if a.phase != "pad"]
    assert [a.phase for a in flying] == ["guided", "turn"]
    text = export_live_script(sim, tmp_path / "mid.m", "now", 600, 60).read_text(encoding="utf-8")
    states = re.findall(r'^L.state = struct\("t", ([^,]+), "y", \[([^\]]+)\], "phase", "(\w+)", '
                        r'"stage", (\d+)', text, re.M)
    assert len(states) == 2
    for a, (t, y, phase, stage) in zip(flying, states, strict=True):
        assert float(t) == pytest.approx(a.t - sim.t, abs=1e-9)
        assert np.array_equal([float(x) for x in y.split()], a.y)
        assert phase == a.phase and int(stage) == a.k + 1
    assert text.count("sim.launches(end + 1) = L;") == len(sim.ascents) == 5
    assert "in powered flight at T+213.0 s (closed-loop guidance, stage 2)" in text


def test_matlab_only_export_leaves_out_the_cross_check(tmp_path):
    """compare=False: the same simulation, plots and engine, without the section
    that runs the Python simulator, its scenario JSON or the comparison code."""
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "launch_day.json"))
    sim.advance(333.7)                      # vehicles in flight and on their pads
    both = export_live_script(sim, tmp_path / "both.m", "now", 600, 60).read_text(
        encoding="utf-8")
    alone = export_live_script(sim, tmp_path / "alone.m", "now", 600, 60, compare=False)
    alone = alone.read_text(encoding="utf-8")
    cross = ("crossCheck", "simulatorFolder", "scenarioJson", "satflightRunSimulator",
             "satflightCompare", "satflightCompareEvents", "eventKeys", "satflight.runScenario",
             "addpath", "## Cross-check")
    for word in cross:
        assert word in both, word
        assert word not in alone, word
    assert "It needs MATLAB alone." in alone
    assert alone.splitlines()[-5:] == both.splitlines()[-5:]          # still a Live Script
    # every other line is the same: states, burns, launches, run, plots, engine
    kept = set(both.splitlines())
    differ = [line for line in alone.splitlines() if line not in kept
              and not line.startswith("%[text] Exported from")]
    assert differ == [line for line in alone.splitlines() if line.startswith("%[text] Press")]
    engine = drop_functions(runtime_lines(False), sorted(TWINLESS))
    assert engine and "\n".join(engine) in alone
    assert len(engine) < len(runtime_lines(True))


def test_a_runtime_without_its_cross_check_marker_is_an_error(tmp_path, monkeypatch):
    """The MATLAB-only export cuts the runtime at a marker line: losing it must
    stop the export, not write a script with the comparison code left in."""
    text = RUNTIME.read_text(encoding="utf-8").replace(CROSS_CHECK, "% --- renamed ")
    fake = tmp_path / "matlab_runtime.m"
    fake.write_text(text, encoding="utf-8")
    monkeypatch.setattr("satflight.matlab_export.RUNTIME", fake)
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "rendezvous.json"))
    export_live_script(sim, tmp_path / "with.m", "start", 60, 60)      # needs no cut
    with pytest.raises(RuntimeError, match="cross-check against the simulator"):
        export_live_script(sim, tmp_path / "alone.m", "start", 60, 60, compare=False)
    assert not (tmp_path / "alone.m").exists()
    with pytest.raises(TypeError, match="compare"):
        export_live_script(sim, tmp_path / "x.m", "start", 60, 60, compare="no")


def test_command_line(tmp_path, capsys):
    out = tmp_path / "hohmann.m"
    main([str(ROOT / "scenarios" / "hohmann_to_geo.json"), "--out", str(out), "--at", "15m",
          "--duration", "2h", "--step", "30"])
    text = out.read_text(encoding="utf-8")
    assert "duration = 7200;" in text and "step = 30;" in text
    assert "T+00:15:00" in text and "crossCheck = true;" in text
    alone = tmp_path / "alone.m"
    main([str(ROOT / "scenarios" / "hohmann_to_geo.json"), "--out", str(alone), "--no-compare"])
    assert "crossCheck" not in alone.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(ROOT / "scenarios" / "hohmann_to_geo.json"), "--out", str(tmp_path / "1x.m")])
    assert "MATLAB script" in capsys.readouterr().err


@pytest.mark.parametrize("name, ok", [("run_1", True), ("Run", True), ("1run", False),
                                      ("my run", False), ("a" * 64, False), ("x.y", False)])
def test_script_names_matlab_can_run(name, ok, tmp_path):
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "rendezvous.json"))
    if ok:
        export_live_script(sim, tmp_path / f"{name}.m", "start", 60, 60)
    else:
        with pytest.raises(ValueError, match="MATLAB script"):
            export_live_script(sim, tmp_path / f"{name}.m", "start", 60, 60)
    assert matlab_name("my run!") == "my_run" and matlab_name("1st") == "sim_1st"
    assert matlab_name("a" * 99) == "a" * 63


@pytest.mark.parametrize("kw, match", [(dict(start="later"), "export start"),
                                       (dict(duration=-1.0), "duration"),
                                       (dict(step=0.0), "step"),
                                       (dict(duration=float("nan")), "duration")])
def test_bad_settings_are_refused(kw, match, tmp_path):
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "rendezvous.json"))
    with pytest.raises(ValueError, match=match):
        export_live_script(sim, tmp_path / "x.m", **{"start": "start", **kw})
    assert not (tmp_path / "x.m").exists()


def test_launches_alone_can_be_exported(tmp_path):
    from satflight.launch import LaunchSpec
    sc = Scenario(name="Pad only", launches=[LaunchSpec(name="Solo", timing="delay", delay=60.0,
                                                         kick=2.5)])
    text = export_live_script(Simulation(sc), tmp_path / "solo.m", "start").read_text(
        encoding="utf-8")
    assert "sim.names = strings(0, 1);" in text and "planLaunches = false;" in text


def test_a_launched_payload_keeps_its_reflectivity_in_matlab(tmp_path):
    from satflight.forces import ForceModel
    from satflight.launch import LaunchSpec
    sc = Scenario(name="Sail", forces=ForceModel(j2=True, sun=True, srp=True),
                  launches=[LaunchSpec(name="Sail", timing="delay", delay=60.0, kick=2.5,
                                       payload_cr=1.7)])
    text = export_live_script(Simulation(sc), tmp_path / "sail.m", "start").read_text(
        encoding="utf-8")
    assert "L.payloadCr = 1.7;" in text
    src = RUNTIME.read_text(encoding="utf-8")
    assert "s.cr(i) = L.payloadCr;" in src          # set at separation, as Simulation does
    # the MATLAB launch defaults match LaunchSpec's
    assert re.search(r'"payloadCr", ([\d.]+)', src).group(1) == repr(LaunchSpec().payload_cr)


def test_nothing_in_orbit_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing to export"):
        export_live_script(Simulation(Scenario(name="Empty")), tmp_path / "x.m", "start")


def test_runtime_constants_are_the_simulators():
    src = RUNTIME.read_text(encoding="utf-8")

    def const(name):
        m = re.search(rf"^C\.{name} = ([^;]+);", src, re.M)
        assert m, name
        num, _, den = m.group(1).partition("/")
        return float(num) / float(den) if den else float(num)

    assert const("MU") == constants.MU_EARTH
    assert const("RE") == constants.R_EARTH
    assert const("F") == pytest.approx(constants.F_EARTH, rel=1e-15)
    assert const("OMEGA") == constants.OMEGA_EARTH
    for k in ("J2", "J3", "J4", "C22", "S22"):
        assert const(k) == getattr(constants, k)
    assert const("SOI") == constants.SOI_EARTH
    assert const("R_SUN") == constants.R_SUN and const("AU") == constants.AU
    assert const("MU_SUN") == constants.MU_SUN and const("MU_MOON") == constants.MU_MOON
    assert const("P_SRP") == constants.P_SRP
    from satflight.scenario import CR_DEFAULT
    assert const("CR_DEFAULT") == CR_DEFAULT
    # the precise-orbit reader's checks (a state vector every 10 s is written in)
    from satflight import poeorb
    assert const("SEAM_TOL_M") == poeorb.SEAM_TOL_M and poeorb.STEP == 10.0
    assert const("POEORB_V_TOL") == poeorb.V_TOL
    for name, want in (("POEORB_R", poeorb.R_BOUNDS), ("POEORB_V", poeorb.V_BOUNDS)):
        m = re.search(rf"^C\.{name} = \[([^\]]+)\];", src, re.M)
        assert m and tuple(float(x) for x in m.group(1).split()) == want, name
    m = re.search(r"^C\.QUALITIES = \[([^\]]+)\];", src, re.M)
    assert m and tuple(re.findall(r'"([^"]*)"', m.group(1))) == poeorb.QUALITIES
    assert "C.DEG = pi / 180;" in src
    from satflight import launch, simulation
    from satflight.maneuvers import G0
    assert const("G0") == G0 and const("REENTRY_ALT") == simulation.REENTRY_ALT
    assert const("ASCENT_SYNC") == simulation.ASCENT_SYNC
    assert const("EVENT_DETAIL_LIMIT") == simulation.EVENT_DETAIL_LIMIT
    assert const("ALL_PAIRS_LIMIT") == simulation.ALL_PAIRS_LIMIT
    assert const("DIP_MARGIN") == simulation.DIP_MARGIN
    assert const("DIP_CHECK_LIMIT") == simulation.DIP_CHECK_LIMIT
    assert const("CONJUNCTION_KM") == Simulation(Scenario()).conjunction_km
    for name in ("G0_M", "THROTTLE_MIN", "GUIDANCE_ALT", "H_POWERED", "H_COAST", "SAMPLE_DT",
                 "FREEZE_TGO", "MIN_PERIGEE", "MAX_BURN"):
        assert const(name.replace("G0_M", "G0M")) == getattr(launch, name), name
    for name in ("APSIS_TIME_TOL", "CIRC_SAMPLES", "CIRC_TOL", "CIRC_ITER", "CIRC_STEP"):
        assert const(name) == getattr(simulation, name), name
    assert const("US76_TOP") == launch._US76_TOP and const("US76_GMR") == launch._GMR
    assert const("US76_R0") == launch._R0_GEOPOT
    us76 = re.search(r"C\.US76 = \[(.*?)\];", src, re.S).group(1)
    rows = [[float(x) for x in r.split()] for r in us76.replace("...", "").split(";")]
    assert rows == [list(r) for r in launch._US76]
    assert const("TOP_KM") == atmosphere.TOP_KM
    table = re.search(r"C\.ATM = \[(.*?)\];", src, re.S).group(1)
    atm = np.array([[float(x) for x in row.split()] for row in table.replace("...", "").split(";")])
    python_table = np.column_stack([atmosphere.H0, atmosphere.RHO0, atmosphere.SCALE_H])
    assert np.array_equal(atm, python_table)


def test_the_sun_moon_and_sunlight_pressure_export_with_each_satellites_cr(tmp_path):
    """The MATLAB port flies them (ExportTest/sunMoonAndSunlight holds it to the
    simulator); the script carries the switches and every satellite's Cr."""
    from satflight.forces import ForceModel
    sc = Scenario(name="Lit", forces=ForceModel(j2=True, drag=True, sun=True, moon=True,
                                                srp=True),
                  satellites=[SatSpec("Shiny", {"type": "elements", "altitude": 700, "i": 98},
                                      cr=1.8),
                              SatSpec("Plain", {"type": "elements", "altitude": 500})])
    text = export_live_script(Simulation(sc), tmp_path / "lit.m", "start", 600, 60).read_text(
        encoding="utf-8")
    assert re.search(r'^sim\.forces = struct\(.*"sun", true, "moon", true, "srp", true, '
                     r'"tides", false, "relativity", false, '
                     r'"gravityDegree", 0, "atmosphere", "table", "cdModel", "fixed"\);',
                     text, re.M)
    props = [[float(x) for x in r.split()] for r in block(text, "props")]
    assert [p[3] for p in props] == [1.8, 1.3]
    assert "sim.cr = props(:, 4);" in text


def test_every_force_model_term_exports(tmp_path):
    from satflight.forces import ForceModel
    from satflight.matlab_export import unported_forces
    assert unported_forces(ForceModel(gravity_degree=70, drag=True, sun=True, moon=True,
                                      srp=True, tides=True, relativity=True)) == []


def _field_rows(text, name):
    """The rows the script sets for one coefficient array: {n: [values for m = 0..n]}."""
    rows = re.findall(rf"^sim\.forces\.field\.{name}\((\d+), 1:\d+\) = \[([^\]]*)\];", text,
                      re.M)
    return {int(k) - 1: [float(v) for v in vals.split()] for k, vals in rows}


def test_the_egm2008_field_exports_with_the_simulators_own_coefficients(tmp_path):
    from satflight import gravity
    from satflight.forces import ForceModel
    sc = lone_satellite()
    sc.forces = ForceModel(gravity_degree=12, sun=True)
    text = export_live_script(Simulation(sc), tmp_path / "egm.m", "start", 60, 60).read_text(
        encoding="utf-8")
    assert '"gravityDegree", 12' in text
    f = gravity.load(12)
    m = re.search(r'^sim\.forces\.field = struct\("gm", ([^,]+), "re", ([^,]+), '
                  r'"degree", (\d+), "c", zeros\(13\), "s", zeros\(13\)\);', text, re.M)
    assert m and float(m.group(1)) == f.gm and float(m.group(2)) == f.re and m.group(3) == "12"
    for name, want in (("c", f.c), ("s", f.s)):
        rows = _field_rows(text, name)
        assert sorted(rows) == list(range(2, 13))           # degrees 0 and 1 stay zero
        for n, vals in rows.items():
            assert vals == list(want[n, :n + 1]), (name, n)  # the same doubles, digit for digit


def test_without_the_field_the_script_carries_no_coefficients(tmp_path):
    text = export_live_script(Simulation(lone_satellite()), tmp_path / "plain.m", "start", 60,
                              60).read_text(encoding="utf-8")
    assert '"gravityDegree", 0' in text and "sim.forces.field = struct(" not in text
    assert "sim.forces.field = [];" in text


def test_snapshot_carries_a_burn_under_way():
    """A snapshot taken mid-burn finishes the burn: going on from it ends where
    the original run does (before, the rest of the burn was dropped)."""
    sc = Scenario(name="burn", satellites=[SatSpec("B", {"type": "elements", "altitude": 500},
                                                   mass=800.0)])
    sc.maneuvers = [Maneuver("B", "finite", timing="delay", delay=300.0, dv=(1.0, 0.0, 0.0),
                             thrust=400.0, isp=310.0, duration=900.0)]
    sim = Simulation(sc)
    sim.advance(750.0)
    snap = Simulation(sim.snapshot_scenario())
    assert [m.kind for m in snap.maneuvers] == ["finite"]
    assert snap.maneuvers[0].duration == pytest.approx(450.0)
    sim.advance(1500.0)
    snap.advance(1500.0)
    assert np.allclose(snap.y, sim.y, atol=1e-3)                 # km
    assert snap.sats[0].mass == pytest.approx(sim.sats[0].mass)
