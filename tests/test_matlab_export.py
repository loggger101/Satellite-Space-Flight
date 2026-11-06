"""Scenarios > Export MATLAB: the live script carries the simulation exactly, its
MATLAB port of the engine uses the simulator's constants, and unusable input is
refused. (Running the scripts in MATLAB is matlab/tests/ExportTest.m.)"""

import re
from pathlib import Path

import numpy as np
import pytest

from satflight import atmosphere, constants
from satflight.maneuvers import Maneuver
from satflight.matlab_export import RUNTIME, export_live_script, main, matlab_name
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.json"))


def block(text: str, name: str) -> list[str]:
    """Rows of the MATLAB matrix ``name = [ ... ];`` in ``text``."""
    m = re.search(rf"^{re.escape(name)} = \[\n(.*?)\n\];", text, re.S | re.M)
    assert m, f"no {name} block"
    return [r.strip() for r in m.group(1).split("\n")]


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_exports_its_exact_starting_state(path, tmp_path):
    sim = Simulation(Scenario.load(path))
    out = export_live_script(sim, tmp_path / "run_it.m", "start", 3600, 60)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("%[text] # ")
    assert lines[-5:] == ['%[appendix]{"version":"1.0"}', "%---", "%[metadata:view]",
                          '%   data: {"layout":"inline"}', "%---"]
    assert "function [T, events] = satflightRun(sim, duration, step)" in text
    on_pad = {id(a.sat) for a in sim.ascents}
    rows = [i for i, s in enumerate(sim.sats) if id(s) not in on_pad]
    names = [r.strip('"').replace('""', '"') for r in block(text, "sim.names")]
    assert names == [sim.sats[i].name for i in rows]
    y0 = np.array([[float(x) for x in r.split()] for r in block(text, "sim.y0")])
    assert np.array_equal(y0, sim.y[rows])                       # bit for bit
    burns = block(text, "sim.maneuvers") if sim.maneuvers else []
    times = [float(re.match(r'satflightBurn\("[^"]*", "[^"]*", ([^,]+),', b).group(1))
             for b in burns]
    assert times == [m.t for m in sim.maneuvers]
    for a in sim.ascents:
        assert a.spec.name in text                               # listed as not reproduced


def test_current_moment_export_starts_where_the_app_is(tmp_path):
    sim = Simulation(Scenario.load(ROOT / "scenarios" / "default.json"))
    sim.advance(3 * 3600 + 17.25)
    text = export_live_script(sim, tmp_path / "later.m", "now").read_text(encoding="utf-8")
    assert "T+03:00:17.2" in text
    assert "datetime(2026, 9, 29, 15, 0, 17.25" in text             # the fraction is kept
    y0 = np.array([[float(x) for x in r.split()] for r in block(text, "sim.y0")])
    assert np.array_equal(y0, sim.y)


def test_command_line(tmp_path, capsys):
    out = tmp_path / "hohmann.m"
    main([str(ROOT / "scenarios" / "hohmann_to_geo.json"), "--out", str(out), "--at", "15m",
          "--duration", "2h", "--step", "30"])
    text = out.read_text(encoding="utf-8")
    assert "duration = 7200;" in text and "step = 30;" in text
    assert "T+00:15:00" in text
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
    from satflight.maneuvers import G0
    from satflight.simulation import REENTRY_ALT
    assert const("G0") == G0 and const("REENTRY_ALT") == REENTRY_ALT
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
