"""Regenerate the README pictures in ``docs/images/``.

    python tools/make_screenshots.py              # every picture
    python tools/make_screenshots.py hohmann      # just these

Each function below sets up one scene (``SCENES`` maps picture names to
them). The simulation is advanced first and then drawn with the clock
stopped, so a scene looks the same every time it is made.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

from satflight.ui.app import App  # noqa: E402

OUT = ROOT / "docs" / "images"
SIZE = (1600, 900)


def _app(scenario: str, advance: float = 0.0, select: str | None = None, size=SIZE) -> App:
    """An App on ``scenarios/<scenario>.json``, ``advance`` seconds in."""
    app = App(ROOT / "scenarios" / f"{scenario}.json", size=size)
    if advance:
        app.sim.advance(advance)
    if select:
        app.select(app.sim.index_of(select))
    return app


def overview():
    """Every classic orbit around the Earth."""
    return _app("default", 45 * 60)


def hohmann():
    """Half-way up a Hohmann transfer to GEO."""
    app = _app("hohmann_to_geo", 3 * 3600, "Transfer")
    app.camera.distance = 110000.0      # the whole transfer ellipse
    return app


def starlink():
    """1,584 satellites in one shell."""
    app = _app("starlink_shell", 10 * 60)
    app.opts.geometry = "off"           # the shell itself, not one member's orbit
    return app


def molniya_ecef_map():
    """Molniya and Tundra ground tracks in the Earth-fixed frame."""
    app = _app("molniya_tundra", 20 * 3600, "Molniya-1")
    app.opts.frame = "ECEF"
    app.opts.map = True
    return app


def j2_plot():
    """Four days of J2: the altitude plot of an eccentric orbit."""
    app = _app("j2_precession", 4 * 86400, "i = 30")
    app.opts.plot = True
    app.opts.geometry = "basic"
    return app


def follow_iss():
    """The camera riding along with the ISS."""
    app = _app("default", 20 * 60, "ISS")
    app.toggle_follow()
    return app


def orbit_inspector():
    """The orbit inspector on the Molniya orbit."""
    return _app("default", 2.5 * 3600, "Molniya")


def launch_dialog():
    """The launch dialog, planning a launch from the Australian outback."""
    app = _app("default")
    app.open("launch")
    dlg = app.dialogs[-1]
    for key, value in (("name", "Outback-1"), ("payload_mass", 8000),
                       ("site", "Custom location"), ("lat", -25.2), ("lon", 133.703),
                       ("hp", 500), ("ha", 500), ("inc", 40)):
        dlg.set(key, value)
    dlg.preview.refresh(dlg)
    dlg.layout()
    return app


def launch_ascent():
    """Following a Falcon 9 through its second-stage burn."""
    app = _app("launch_day", 350, "Starlink batch")
    app.toggle_follow()
    app.camera.distance = 9000.0        # back far enough to see the coast and the other pads
    return app


def start_screen():
    """The start menu as the simulator opens: nothing loaded yet."""
    return App(size=(1400, 860), welcome=True)


SCENES = {f.__name__: f for f in (overview, hohmann, starlink, molniya_ecef_map, j2_plot,
                                  follow_iss, orbit_inspector, launch_dialog, launch_ascent,
                                  start_screen)}


def render(name: str, frames: int = 10):
    """Build scene ``name``, draw it with the clock stopped and save the picture."""
    app = SCENES[name]()
    app.toasts.clear()                  # no "Loaded ..." banner over the scene
    for _ in range(frames):             # a fresh App's first frames draw a coarser Earth
        app.update(0.0)
        app.draw()
    path = OUT / f"{name}.png"
    pygame.image.save(app.screen, str(path))
    pygame.quit()
    return path


def main(argv=None):
    """Render the named scenes (all of them by default)."""
    names = (argv if argv is not None else sys.argv[1:]) or list(SCENES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"wrote {render(name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
