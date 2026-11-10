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
MATLAB_TIMEOUT = 280     # s; under the suite's 300 s watchdog


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


# ExportTest.m alone takes ~270 s, too close to the limit: it runs in two sessions
# (BundledExportTest.m too, split by its start parameter: see _sessions)
EXPORT_PARTS = (("hohmannFromTheStart", "everyKindOfBurn", "reEntry", "fromTheCurrentMoment",
                 "launchDay"),
                ("launchDayMidAscent", "launchesPlannedHere", "matlabOnly",
                 "passesAndAnimationFile", "midBurnSnapshot"))


def test_every_export_test_runs_in_exactly_one_part():
    src = (ROOT / "matlab" / "tests" / "ExportTest.m").read_text(encoding="utf-8")
    body = src.split("methods (Test)", 1)[1]
    names = re.findall(r"^\s*function\s+(\w+)\s*\(", body, re.M)
    parts = [n for part in EXPORT_PARTS for n in part]
    assert sorted(parts) == sorted(names), "update EXPORT_PARTS to match ExportTest.m"
    bundled = (ROOT / "matlab" / "tests" / "BundledExportTest.m").read_text(encoding="utf-8")
    starts = re.search(r"start = struct\((.*)\)", bundled).group(1)
    assert re.findall(r'"(\w+)", "[^"]*"', starts) == ["fromTheStart", "anHourIn"], \
        "update _sessions to match BundledExportTest.m's start parameter"


def _sessions():
    """(suite, runtests selector, how many tests it must run) per MATLAB session.
    ExportTest (~270 s) and BundledExportTest (~235 s) are each split in two to
    stay well inside the time limit; BundledExportTest by its start parameter,
    one test per bundled scenario each."""
    def names(part):
        return ", 'ProcedureName', {" + ", ".join(f"'{n}'" for n in part) + "}", len(part)

    n_bundled = len(list((ROOT / "scenarios").glob("*.json")))
    return [("SatflightToolkitTest", "", None),
            *(("ExportTest", *names(p)) for p in EXPORT_PARTS),
            *(("BundledExportTest", f", 'ParameterName', '{start}'", n_bundled)
              for start in ("fromTheStart", "anHourIn"))]


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is not on the PATH")
@pytest.mark.parametrize("suite, pick, count", _sessions(),
                         ids=["toolkit", "export-1", "export-2", "bundled-start",
                              "bundled-later"])
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
        _kill_tree(proc)
        out, err = proc.communicate(timeout=60)
        pytest.fail(f"{suite} ran over {MATLAB_TIMEOUT} s (MATLAB was stopped); its output:\n"
                    + out[-6000:] + err[-2000:])
    assert proc.returncode == 0, out[-6000:] + err[-2000:]


def _kill_tree(proc):
    """Stop ``proc`` and every process it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
