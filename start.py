"""Start Satellite Space Flight, installing what it needs on the first run.

Double-click "Start Satellite Space Flight" (.bat on Windows, .command on macOS),
or run this file from any terminal:

    python start.py                          # start screen
    python start.py scenarios/launch_day.json
    python start.py --reinstall              # rebuild the private environment

If the Python running this script already has numpy and pygame-ce, the
simulator runs on it straight away. Otherwise the first start builds a private
environment for this computer (outside the project folder, so a synced folder
never carries it between machines) and installs them there: that needs an
internet connection once, for about a minute. Arguments are passed on to
``python -m satflight``.

Only the standard library is used here, since this file runs before anything
is installed, and nothing newer than Python 3.6 so an old Python still gets
the message saying it is too old.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_PYTHON = (3, 10)
# keep in step with [project] dependencies in pyproject.toml (a test checks)
REQUIREMENTS = ["numpy>=1.24", "pygame-ce>=2.4"]
CHECK = ("import numpy, pygame, sys; "
         "sys.exit(0 if getattr(pygame, 'IS_CE', False) else 1)")


def say(*lines):
    print("\n".join(lines), flush=True)


def env_dir():
    """This computer's private environment for the simulator (``SATFLIGHT_ENV``
    overrides the location)."""
    if os.environ.get("SATFLIGHT_ENV"):
        return os.path.abspath(os.environ["SATFLIGHT_ENV"])
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "SatelliteSpaceFlight", "venv")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/SatelliteSpaceFlight/venv")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "satellite-space-flight", "venv")


def env_python(env):
    if sys.platform == "win32":
        return os.path.join(env, "Scripts", "python.exe")
    return os.path.join(env, "bin", "python")


def has_requirements(python):
    """True if ``python`` can import numpy and pygame-ce (classic pygame does not count)."""
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
    try:
        return subprocess.call([python, "-c", CHECK], env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def build_env(env):
    """Create the private environment and install the requirements; False on failure."""
    import venv

    say("", "First start on this computer: installing numpy and pygame-ce.",
        "This happens once, needs an internet connection and takes about a minute.",
        "Setting up in " + env, "")
    try:
        venv.EnvBuilder(clear=True, with_pip=True).create(env)
    except Exception as exc:
        say("", "Could not create the environment: " + str(exc))
        if sys.platform.startswith("linux"):
            say("On Debian or Ubuntu, install venv support first:  sudo apt install python3-venv")
        return False
    py = env_python(env)
    cmd = [py, "-m", "pip", "install", "--disable-pip-version-check", *REQUIREMENTS]
    if subprocess.call(cmd) != 0 or not has_requirements(py):
        say("", "Installing numpy and pygame-ce failed (see the messages above).",
            "Check the internet connection and start again; if it keeps failing,",
            "run:  python start.py --reinstall")
        return False
    say("", "Setup finished.", "")
    return True


def main(argv):
    if sys.version_info < MIN_PYTHON:
        need, have = ".".join(map(str, MIN_PYTHON)), ".".join(map(str, sys.version_info[:2]))
        say(f"Satellite Space Flight needs Python {need} or newer; this is Python {have}.",
            "Install a newer one from https://www.python.org/downloads/ and start again.")
        return 1
    reinstall = "--reinstall" in argv
    argv = [a for a in argv if a != "--reinstall"]

    if not reinstall and has_requirements(sys.executable):
        say("Starting Satellite Space Flight...")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        sys.path.insert(0, ROOT)            # this checkout, not an installed copy
        from satflight.__main__ import main as run
        run(argv)
        return 0

    env = env_dir()
    py = env_python(env)
    if (reinstall or not has_requirements(py)) and not build_env(env):
        return 1
    say("Starting Satellite Space Flight...")
    return subprocess.call([py, "-m", "satflight", *argv], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
