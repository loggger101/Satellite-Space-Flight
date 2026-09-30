"""The double-click launcher (start.py): requirements and where it sets up."""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start", ROOT / "start.py")
start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(start)


def test_requirements_match_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps = re.search(r"^dependencies = \[(.*?)\]", text, re.M | re.S).group(1)
    assert re.findall(r'"([^"]+)"', deps) == start.REQUIREMENTS


def test_private_environment_stays_out_of_the_project_folder(monkeypatch):
    monkeypatch.delenv("SATFLIGHT_ENV", raising=False)
    env = Path(start.env_dir()).resolve()
    assert ROOT not in env.parents          # a synced folder must not carry it between machines
    monkeypatch.setenv("SATFLIGHT_ENV", str(ROOT / "x"))
    assert Path(start.env_dir()) == ROOT / "x"


def test_this_python_counts_as_ready():
    # the test suite itself needs numpy and pygame-ce, so start.py takes its fast path
    assert start.has_requirements(sys.executable)
    assert not start.has_requirements(str(ROOT / "no-such-python"))
