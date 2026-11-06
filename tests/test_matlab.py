"""The MATLAB toolkit in matlab/: what it hard-codes matches the simulator, and
its own test suite passes when MATLAB is installed."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from satflight import batch, constants

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "matlab" / "+satflight"


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


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is not on the PATH")
@pytest.mark.parametrize("suite", ["SatflightToolkitTest", "ExportTest"])
def test_matlab_suite(suite):
    """matlab/tests/SatflightToolkitTest.m: scenarios round-trip through MATLAB
    unchanged, built ones pass the simulator's checks, refusals reach MATLAB,
    runs are read back and plotted. matlab/tests/ExportTest.m: exported live
    scripts run in MATLAB and agree with the simulator. One MATLAB session each,
    to stay inside the per-test time limit."""
    code = (f"addpath('matlab'); r = runtests('matlab/tests/{suite}.m'); disp(table(r)); "
            "assertSuccess(r);")
    proc = subprocess.run([shutil.which("matlab"), "-batch", code], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=280)
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-2000:]
