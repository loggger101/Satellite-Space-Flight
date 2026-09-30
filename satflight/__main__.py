"""Launch the interactive simulator.

    python -m satflight                                  # start screen, then pick a scenario
    python -m satflight scenarios/walker_constellation.json
    python -m satflight --headless --frames 120 --screenshot out.png   # render offscreen
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def main(argv=None):
    """Parse the command line, build the :class:`App` and run it."""
    ap = argparse.ArgumentParser(prog="satflight", description="Satellite Space Flight simulator")
    ap.add_argument("scenario", nargs="?", help="scenario JSON (default: scenarios/default.json)")
    ap.add_argument("--size", default="1600x900", help="window size WxH")
    ap.add_argument("--welcome", action=argparse.BooleanOptionalAction,
                    help="show the start screen (default: only when no scenario is given "
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
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from .batch import parse_duration
    from .ui.app import App

    w, h = (int(x) for x in args.size.lower().split("x"))
    welcome = args.welcome
    if welcome is None:
        welcome = not (args.scenario or args.headless)
    app = App(Path(args.scenario) if args.scenario else None, size=(w, h), welcome=welcome)
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
