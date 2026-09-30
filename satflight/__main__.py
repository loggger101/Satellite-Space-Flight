"""Launch the interactive simulator.

    python -m satflight                                  # start menu, then pick a scenario
    python -m satflight scenarios/walker_constellation.json
    python -m satflight --headless --frames 120 --screenshot out.png   # render offscreen
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# design px, like the UI (see ui.theme): multiplied by the display scale
DEFAULT_SIZE = (1600, 900)
TASKBAR = 48        # kept free for the taskbar or dock
TITLE_BAR = 34      # window frame above the drawing area


def window_size(desktop, want=DEFAULT_SIZE, scale=1.0):
    """``want`` (design px) shrunk so the whole window, frame included, fits a
    ``desktop`` of (w, h) screen px, but never below the 900 x 600 minimum the
    panels need; in screen px, at ``scale`` screen px per design px."""
    def s(v):
        return round(v * scale)
    dw, dh = desktop
    return (max(s(900), min(s(want[0]), dw - s(16))),
            max(s(600), min(s(want[1]), dh - s(TASKBAR + TITLE_BAR + 28))))


def display_scale() -> float:
    """The main screen's scaling (1.25 at 125 %): Windows' DPI over 96. The
    process is DPI-aware there (SDL_WINDOWS_DPI_AWARENESS), so the window gets
    the screen's real pixels and the UI draws itself this much larger. 1 on
    other systems, which scale for the program. Call after the display is
    initialized."""
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes
        return max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96.0)
    except (AttributeError, OSError):     # Windows before 10
        return 1.0


def fit_window(scale=None):
    """The UI scale (``scale``, or the display's) and a window size that fits the
    main screen, with a position that centers the window above the taskbar (a
    window larger than the screen hides the panels' right and bottom edges)."""
    import pygame
    pygame.display.init()
    if scale is None:
        scale = display_scale()
    desktop = (pygame.display.get_desktop_sizes() or [DEFAULT_SIZE])[0]
    w, h = window_size(desktop, scale=scale)
    bar, frame = round(TASKBAR * scale), round(TITLE_BAR * scale)
    y = max(frame, (desktop[1] - bar - h) // 2 + frame // 2)
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{max(0, (desktop[0] - w) // 2)},{y}")
    return scale, (w, h)


def main(argv=None):
    """Parse the command line, build the :class:`App` and run it."""
    ap = argparse.ArgumentParser(prog="satflight", description="Satellite Space Flight simulator")
    ap.add_argument("scenario", nargs="?", help="scenario JSON (default: scenarios/default.json)")
    ap.add_argument("--size", help="window size WxH in screen pixels (default: 1600x900 "
                                   "times the UI scale, or less to fit the screen)")
    ap.add_argument("--ui-scale", type=float,
                    help="size of the panels and text, 1 = 100%% (default: the display's "
                         "scaling, e.g. 1.25 at 125%%; 1 with --headless)")
    ap.add_argument("--fullscreen", action="store_true",
                    help="start fullscreen (F11 or Alt+Enter switches back to a window)")
    ap.add_argument("--welcome", action=argparse.BooleanOptionalAction,
                    help="open on the start menu (default: only when no scenario is given "
                         "and not --headless)")
    ap.add_argument("--headless", action="store_true", help="render without a window (SDL dummy)")
    ap.add_argument("--frames", type=int, help="quit after this many frames")
    ap.add_argument("--screenshot", help="save the last frame to this PNG")
    ap.add_argument("--warp", type=float, help="initial time warp")
    ap.add_argument("--advance", default="0",
                    help="simulate this long before the first frame (e.g. 3h)")
    ap.add_argument("--select", help="name of the satellite to select")
    ap.add_argument("--follow", action="store_true", help="camera follows the selected satellite")
    ap.add_argument("--frame", choices=("ECI", "ECEF"), help="view frame")
    ap.add_argument("--show", default="",
                    help="comma list of panels/overlays: map,plot,help,vectors,geo_ring")
    ap.add_argument("--distance", type=float, help="camera distance (km)")
    ap.add_argument("--yaw", type=float, help="camera yaw (deg)")
    ap.add_argument("--pitch", type=float, help="camera pitch (deg)")
    args = ap.parse_args(argv)

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"     # must precede importing pygame
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    else:
        # the screen's real pixels, not a 100 % image Windows stretches (and blurs)
        os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from .batch import parse_duration
    from .ui.app import App

    if args.headless:
        scale = args.ui_scale or 1.0
        w, h = window_size((1 << 16, 1 << 16), scale=scale)
    else:
        scale, (w, h) = fit_window(args.ui_scale)
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
    welcome = args.welcome
    if welcome is None:
        welcome = not (args.scenario or args.headless)
    app = App(Path(args.scenario) if args.scenario else None, size=(w, h), welcome=welcome,
              fullscreen=args.fullscreen, ui_scale=scale)
    if args.warp:
        app.warp = args.warp
    adv = parse_duration(args.advance)
    if adv > 0:
        app.sim.advance(adv)
    if args.select:
        try:
            app.select(app.sim.index_of(args.select))
        except KeyError:
            print(f"no satellite named {args.select!r}")
    if args.frame:
        app.opts.frame = args.frame
    for name in filter(None, (s.strip() for s in args.show.split(","))):
        setattr(app.opts, name, True)
    if args.distance:
        app.camera.distance = args.distance
    if args.yaw is not None:
        app.camera.yaw = math.radians(args.yaw)
    if args.pitch is not None:
        app.camera.pitch = math.radians(args.pitch)
    if args.follow:
        app.toggle_follow()
        if args.distance:                           # following resets the distance
            app.camera.distance = args.distance
    app.run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
