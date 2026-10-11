"""The test suite's own guards work: a hanging test fails with its stack and
the run carries on, and a warning fails a test."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_suite(tmp_path, body: str, timeout: str = "2") -> str:
    shutil.copy(HERE / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_x.py").write_text(body, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\nfilterwarnings = error\n", encoding="utf-8")
    env = {**os.environ, "SATFLIGHT_TEST_TIMEOUT": timeout}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(tmp_path)], capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=120, stdin=subprocess.DEVNULL)
    return done.stdout + done.stderr


def test_a_hanging_test_fails_with_its_stack_and_the_run_goes_on(tmp_path):
    out = run_suite(tmp_path, "import time\n\n"
                              "def test_hangs():\n    while True:\n        time.sleep(0.05)\n\n"
                              "def test_after():\n    pass\n")
    assert "ran over 2 s" in out and "test_x.py" in out and "time.sleep" in out
    assert "1 failed, 1 passed" in out


def test_a_warning_is_a_failure(tmp_path):
    out = run_suite(tmp_path, "import numpy as np\n\n"
                              "def test_divides():\n    np.float64(1.0) / np.float64(0.0)\n")
    assert "RuntimeWarning" in out and "1 failed" in out
