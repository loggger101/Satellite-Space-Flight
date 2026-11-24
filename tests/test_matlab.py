"""The MATLAB toolkit in matlab/: what it hard-codes matches the simulator, and
its own test suite passes when MATLAB is installed."""

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from satflight import batch, constants

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "matlab" / "+satflight"
MATLAB_SESSION_LIMIT = 900   # s per MATLAB session: one test alone took 230-290 s on
                             # this machine (2026-11-22), so not the 300 s default
MATLAB_TIMEOUT = MATLAB_SESSION_LIMIT - 20   # s; MATLAB is stopped first, then the watchdog


def _quoted(text):
    return re.findall(r'"([^"]*)"', text)


def test_readrun_columns_match_the_batch_header():
    """readRun.m refuses any CSV whose header differs from its list, so the list
    must be the batch writer's header exactly."""
    src = (PKG / "readRun.m").read_text(encoding="utf-8")
    block = re.search(r"expected = \[(.*?)\];", src, re.S)
    assert block, "readRun.m: no 'expected = [...]' header list"
    assert _quoted(block.group(1)) == batch.HEADER


def test_matlab_constants_match_the_simulator():
    """The Earth drawn by plotOrbit3D and the constants of the analysis example."""
    def value(path, name):
        m = re.search(rf"^\s*{name}\s*=\s*([0-9.e+-]+(?:\s*/\s*[0-9.e+-]+)?);", path.read_text(
            encoding="utf-8"), re.M)
        assert m, f"{path.name}: no {name}"
        num, _, den = m.group(1).partition("/")      # a number, or number / number
        return float(num) / float(den) if den else float(num)

    plot3d = PKG / "plotOrbit3D.m"
    example = ROOT / "matlab" / "examples" / "analyzeRunExample.m"
    assert value(plot3d, "R_EARTH") == constants.R_EARTH
    assert value(plot3d, "F_EARTH") == pytest.approx(constants.F_EARTH, rel=1e-15)
    assert value(example, "RE") == constants.R_EARTH
    assert value(example, "MU") == constants.MU_EARTH
    assert value(example, "J2") == constants.J2


# ExportTest.m in two sessions of five ran over the limit (2026-11-17, MATLAB
# R2026a): it runs in sessions of ~half the limit at most. The comments give each
# test's rough time in s, to balance the parts (taken on a throttled machine, so high).
EXPORT_PARTS = (("matlabOnly",),                                           # 136
                ("launchDay", "reEntry"),                                  # 90 + 30
                ("everyKindOfBurn", "fromTheCurrentMoment", "midBurnSnapshot"),  # 84+26+21
                ("launchDayMidAscent", "hohmannFromTheStart"),             # 78 + 42
                ("launchesPlannedHere",),        # these two took 809 s in one session
                ("passesAndAnimationFile",),     # on 2026-11-22
                ("transferFiguresAndFiles",),                              # ~150 (2 videos)
                ("everyIntegrator",),                                      # ~90
                ("sunMoonAndSunlight",),
                ("launchInSunlight",),
                ("customLaunches",),                                       # 79
                ("egm2008Field",),
                ("nrlmsisDrag", "nrlmsisPastTheMeasuredDays"),
                ("preciseOrbitReplay", "replayWithoutItsFiles"))
BUNDLED_STARTS = ("fromTheStart", "anHourIn")
BUNDLED_SPLIT = 2     # sessions per start: one start's scenarios had run over in
                      # one session (launch_day, starlink_shell the slowest), so
                      # every other scenario in each


def test_every_export_test_runs_in_exactly_one_part():
    src = (ROOT / "matlab" / "tests" / "ExportTest.m").read_text(encoding="utf-8")
    body = src.split("methods (Test)", 1)[1]
    names = re.findall(r"^\s*function\s+(\w+)\s*\(", body, re.M)
    parts = [n for part in EXPORT_PARTS for n in part]
    assert sorted(parts) == sorted(names), "update EXPORT_PARTS to match ExportTest.m"
    bundled = (ROOT / "matlab" / "tests" / "BundledExportTest.m").read_text(encoding="utf-8")
    starts = re.search(r"start = struct\((.*)\)", bundled).group(1)
    assert re.findall(r'"(\w+)", "[^"]*"', starts) == list(BUNDLED_STARTS), \
        "update BUNDLED_STARTS to match BundledExportTest.m's start parameter"


def test_every_bundled_case_runs_in_exactly_one_session():
    picked = [n for suite, pick, _ in _sessions() if suite == "BundledExportTest"
              for n in re.findall(r"'(BundledExportTest/[^']*)'", pick)]
    scenarios = sorted(p.stem for p in (ROOT / "scenarios").glob("*.json"))
    assert sorted(picked) == sorted(f"BundledExportTest/bundled(scenario={s},start={st})"
                                    for s in scenarios for st in BUNDLED_STARTS)


def _sessions():
    """(suite, runtests selector, how many tests it must run) per MATLAB session.
    ExportTest by EXPORT_PARTS; BundledExportTest by start, then every
    BUNDLED_SPLIT-th scenario of scenarios/ (a new one is picked up by itself)."""
    def cell(names):
        return "{" + ", ".join(f"'{n}'" for n in names) + "}"

    scenarios = sorted(p.stem for p in (ROOT / "scenarios").glob("*.json"))
    bundled = []
    for start in BUNDLED_STARTS:
        for k in range(BUNDLED_SPLIT):
            names = [f"BundledExportTest/bundled(scenario={s},start={start})"
                     for s in scenarios[k::BUNDLED_SPLIT]]
            bundled.append(("BundledExportTest", f", 'Name', {cell(names)}", len(names)))
    return [("SatflightToolkitTest", "", None),
            *(("ExportTest", f", 'ProcedureName', {cell(p)}", len(p)) for p in EXPORT_PARTS),
            *bundled]


def _ids():
    return (["toolkit"] + [f"export-{k + 1}" for k in range(len(EXPORT_PARTS))]
            + [f"bundled-{start}-{k + 1}" for start in BUNDLED_STARTS
               for k in range(BUNDLED_SPLIT)])


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is not on the PATH")
@pytest.mark.time_limit(MATLAB_SESSION_LIMIT)
@pytest.mark.parametrize("suite, pick, count", _sessions(), ids=_ids())
def test_matlab_suite(suite, pick, count):
    """matlab/tests/SatflightToolkitTest.m: scenarios round-trip through MATLAB
    unchanged, built ones pass the simulator's checks, refusals reach MATLAB,
    runs are read back and plotted. matlab/tests/ExportTest.m: exported live
    scripts run in MATLAB and agree with the simulator; BundledExportTest.m: so
    does every bundled scenario, from its start and part-way in. One MATLAB
    session per suite or part of one (``pick``, which must run ``count``
    tests), to stay inside the per-test time limit."""
    code = (f"addpath('matlab'); r = runtests('matlab/tests/{suite}.m'{pick}); "
            "disp(table(r)); assert(numel(r) " + ("> 0" if count is None else f"== {count}")
            + ", 'not every test ran'); assertSuccess(r);")
    proc = subprocess.Popen([shutil.which("matlab"), "-batch", code], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace",
                            start_new_session=sys.platform != "win32")   # its own group
    try:
        out, err = proc.communicate(timeout=MATLAB_TIMEOUT)
    except subprocess.TimeoutExpired:
        # the matlab launcher starts MATLAB.exe as a child that holds the pipes:
        # killing the launcher alone leaves both running and the read waiting forever
        if proc.poll() is None:
            what = _kill_tree(proc)
        else:   # the timeout was seen late: the machine slept or stalled meanwhile
            what = (f"MATLAB had already exited (status {proc.returncode}) when the timeout "
                    "was seen: did the machine sleep?")
        out, err = proc.communicate(timeout=60)
        pytest.fail(f"{suite} ran over {MATLAB_TIMEOUT} s ({what}); its output:\n"
                    + out[-6000:] + err[-2000:])
    assert proc.returncode == 0, out[-6000:] + err[-2000:]


def _kill_tree(proc):
    """Stop ``proc`` and every process it started; say whether that worked."""
    if sys.platform == "win32":
        r = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return f"taskkill failed, MATLAB may still be running: {r.stdout}{r.stderr}".strip()
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    return "MATLAB was stopped"


def test_readstudy_columns_match_the_study_writer():
    """readStudy.m refuses any study CSV whose header differs from its lists, so
    each list must be what tools/sentinel1_study.py writes (the arcs' from its
    ARC_COLUMNS, the rest from the files it wrote)."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sentinel1_study as tool
    finally:
        sys.path.remove(str(ROOT / "tools"))
    src = (PKG / "readStudy.m").read_text(encoding="utf-8")
    # the named lists and the lists written straight into a readOne call (second
    # argument); readOne's own definition and the text-column lists are neither
    calls = r"(?:dailyCols = |arcCols = |readOne\(fullfile\([^()]*\), )\[(.*?)\]"
    lists = [_quoted(m) for m in re.findall(calls, src, re.S)]
    study = ROOT / "docs" / "analysis" / "sentinel1"

    def header(name):
        return (study / name).read_text(encoding="utf-8").splitlines()[0].split(",")

    assert lists == [header("daily_S1A.csv"), header("phasing.csv"), header("burns.csv"),
                     tool.ARC_COLUMNS]
    assert header("daily_S1C.csv") == header("daily_S1A.csv")
