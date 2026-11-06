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
    assert "function [T, events, launches] = satflightRun(sim, duration, step, opts)" in text
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


@pytest.mark.parametrize("scenario, gone", [
    ("hohmann_to_geo", {"satflightNoBurns", "satflightPlan", "satflightPlotAscent",
                        "mergeMarks"}),
    ("default", {"satflightPlan", "satflightPlotAscent", "mergeMarks"}),
    ("launch_day", set())])
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
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "default.json"))
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
    engine = runtime_lines(False)
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
    assert "C.DEG = pi / 180;" in src
    from satflight import launch, simulation
    from satflight.maneuvers import G0
    assert const("G0") == G0 and const("REENTRY_ALT") == simulation.REENTRY_ALT
    assert const("ASCENT_SYNC") == simulation.ASCENT_SYNC
    assert const("EVENT_DETAIL_LIMIT") == simulation.EVENT_DETAIL_LIMIT
    assert const("ALL_PAIRS_LIMIT") == simulation.ALL_PAIRS_LIMIT
    assert const("CONJUNCTION_KM") == Simulation(Scenario()).conjunction_km
    for name in ("G0_M", "P_SCALE_H", "GUIDANCE_ALT", "H_POWERED", "H_COAST", "SAMPLE_DT",
                 "FREEZE_TGO", "MIN_PERIGEE", "MAX_BURN"):
        assert const(name.replace("G0_M", "G0M")) == getattr(launch, name), name
    assert const("TOP_KM") == atmosphere.TOP_KM
    table = re.search(r"C\.ATM = \[(.*?)\];", src, re.S).group(1)
    atm = np.array([[float(x) for x in row.split()] for row in table.replace("...", "").split(";")])
    python_table = np.column_stack([atmosphere.H0, atmosphere.RHO0, atmosphere.SCALE_H])
    assert np.array_equal(atm, python_table)


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
