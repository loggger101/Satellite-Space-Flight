"""The start menu shown when the simulator opens (Ctrl+N brings it back). Most of
it is a form that sets up a new simulation: when it starts, what it begins with
and the physics. Beside the form, a scenario or TLE file can be opened instead;
along the bottom, a row of ready-made examples. It fills the window and the
simulation stands still behind it (see ``App.in_menu``)."""

from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pygame

from ..forces import ForceModel
from ..integrators import METHODS
from ..scenario import (
    DEFAULT_EPOCH,
    DEFAULT_STATIONS,
    PROPAGATORS,
    IntegratorSettings,
    Scenario,
)
from ..timeutil import parse_epoch
from . import theme, tips
from .background import Latest
from .dialogs import DENSITY_TIP, FORCE_TIPS, METHOD_TIPS, PROPAGATOR_TIPS
from .theme import px
from .widgets import (
    Button,
    Checkbox,
    Choice,
    TextField,
    field_tip,
    is_visible,
    make_widget,
    parse_fields,
)
from .widgets import FieldSpec as F

# Bundled scenarios in the order the examples row shows them, each with a short
# label for its button and a plain-language blurb for its hover tip (the files'
# own descriptions are written for people who know the jargon).
CARDS = [
    ("default", "Orbit tour", "One satellite in every classic orbit, from low to geostationary."),
    ("launch_day", "Launch day", "Six rockets lift off from four continents; stages fall away."),
    ("hohmann_to_geo", "Hohmann",
     "Two engine burns raise a satellite from low orbit to GEO."),
    ("rendezvous", "Rendezvous",
     "A chaser catches up with a space station and matches its speed."),
    ("starlink_shell", "Starlink shell",
     "A Starlink-like shell of 1,584 satellites around the planet."),
    ("gps_constellation", "GPS", "24 GPS-like satellites in six planes, 20,000 km up."),
    ("polar_star", "Polar star",
     "Iridium-like polar orbits, with a seam where planes pass head-on."),
    ("molniya_tundra", "Molniya", "Long, looping orbits that linger over the far north."),
    ("j2_precession", "J2 precession",
     "The Earth's bulge slowly twists orbits at six inclinations."),
    ("geo_drift", "GEO drift",
     "Geostationary satellites drift from their slots toward two resting points."),
    ("drag_decay", "Drag decay", "Air drag pulls three spacecraft down. The balloon falls first."),
    ("launches_and_arcs", "Ballistic arcs",
     "Short hops, one satellite reaching orbit, one escaping Earth."),
]
BLURBS = {stem: blurb for stem, _, blurb in CARDS}
SHORT = {stem: short for stem, short, _ in CARDS}

# --- The setup form ------------------------------------------------------------------------

# what the new simulation begins with: the dialog that opens once it has started
START_WITH = {"One satellite": "add", "Constellation": "walker", "Launch": "launch",
              "Empty Earth": None}
START_TIPS = {
    "One satellite": "The Add-satellite form opens next: pick an orbit (ISS, GPS, "
                     "Molniya, ...) or type your own.",
    "Constellation": "The Walker-constellation form opens next: many satellites in evenly "
                     "spread planes.",
    "Launch": "The Launch form opens next: fly a rocket from a pad into orbit.",
    "Empty Earth": "Nothing yet: add satellites, constellations and launches from the left "
                   "panel when you are ready.",
}
SPEEDS = {"1x": 1.0, "10x": 10.0, "60x": 60.0, "600x": 600.0, "3600x": 3600.0,
          "86400x": 86400.0}
SPEED_TIPS = {"1x": "Real time: a low orbit takes an hour and a half.",
              "10x": "Ten seconds a second.", "60x": "A minute a second: a low orbit in 90 s.",
              "600x": "Ten minutes a second.", "3600x": "An hour a second.",
              "86400x": "A day a second: for slow drifts (J2, GEO, decay)."}
SOLAR = {"Quiet x0.3": 0.3, "Average x1": 1.0, "Active x3": 3.0}


def _cowell(raw) -> bool:
    return raw.get("propagator") == "cowell"


RUN_FIELDS = [
    F("name", "Name", "text", "New simulation",
      tip="What the simulation is called: shown in the left panel and used when you save it."),
    F("date", "Date (UTC)", "text", DEFAULT_EPOCH.strftime("%Y-%m-%d"),
      tip="The calendar day the clock starts on, as YYYY-MM-DD (1900 to 2100). It sets where "
          "the Sun is (day and night, eclipses) and how the Earth is turned."),
    F("time", "Time (UTC)", "text", "00:00:00",
      tip="The time of day the clock starts at, as HH:MM or HH:MM:SS in UTC."),
    F("start_with", "Begin with", "choice", "One satellite", list(START_WITH),
      tip="What to put in orbit first. Anything else can be added once it runs.",
      option_tips=START_TIPS),
    F("speed", "Time speed", "choice", "60x", list(SPEEDS),
      tip="How many simulated seconds pass each real second. The , and . keys change it "
          "while it runs; Space pauses.", option_tips=SPEED_TIPS),
    F("stations", "Stations", "bool", True,
      tip="Ground stations: five tracking sites (Cape Canaveral, Svalbard, Canberra, Madrid, "
          "Kourou) that log each satellite they see. Key N adds more."),
]
PHYSICS_FIELDS = [
    F("propagator", "Propagator", "choice", "cowell", list(PROPAGATORS),
      tip="How orbits are moved forward in time.", option_tips=PROPAGATOR_TIPS),
    F("method", "Integrator", "choice", "dopri5", list(METHODS), visible=_cowell,
      tip="The numerical method Cowell uses. Its tolerances and steps are in Physics (key P) "
          "once the simulation runs.", option_tips=METHOD_TIPS),
    F("j2", "J2 oblateness", "bool", True, visible=_cowell, tip=FORCE_TIPS["j2"]),
    F("j3", "J3 pear shape", "bool", False, visible=_cowell, tip=FORCE_TIPS["j3"]),
    F("j4", "J4", "bool", False, visible=_cowell, tip=FORCE_TIPS["j4"]),
    F("c22", "C22 equator", "bool", False, visible=_cowell, tip=FORCE_TIPS["c22"]),
    F("drag", "Air drag", "bool", False, visible=_cowell, tip=FORCE_TIPS["drag"]),
    F("solar", "Solar activity", "choice", "Average x1", list(SOLAR),
      visible=lambda raw: _cowell(raw) and raw.get("drag"), tip=DENSITY_TIP),
]
SETUP_FIELDS = RUN_FIELDS + PHYSICS_FIELDS


def setup_scenario(raw: dict) -> Scenario:
    """The empty scenario the setup form's unparsed values ``raw`` describe;
    ValueError saying what is wrong. The force switches are kept even where
    the propagator hides them, so Physics (P) finds them as they were set."""
    date, time = raw["date"].strip(), raw["time"].strip() or "00:00:00"
    try:
        epoch = parse_epoch(f"{date}T{time}")
    except ValueError:
        raise ValueError(f"start '{date} {time}' is not a date YYYY-MM-DD and a time "
                         "HH:MM[:SS]") from None
    if not 1900 <= epoch.year < 2100:
        raise ValueError(f"start year {epoch.year} is outside 1900-2099, where the Earth's "
                         "rotation and the Sun are modelled")
    forces = ForceModel(j2=raw["j2"], j3=raw["j3"], j4=raw["j4"], c22=raw["c22"],
                        drag=raw["drag"], density_scale=SOLAR[raw["solar"]])
    return Scenario(name=raw["name"].strip() or "New simulation",
                    description="Set up on the start menu.", epoch=epoch,
                    propagator=raw["propagator"],
                    integrator=IntegratorSettings(method=raw["method"]), forces=forces,
                    warp=SPEEDS[raw["speed"]],
                    stations=copy.deepcopy(DEFAULT_STATIONS) if raw["stations"] else [])


# --- Files ----------------------------------------------------------------------------------

def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def scenario_card(path: Path) -> tuple[str, str, str, str]:
    """(title, blurb, size tag, description) for a scenario file; OSError or
    ValueError if it cannot be read or is not a valid scenario."""
    sc = Scenario.load(path)
    sats = sum(int(s.count) for s in sc.satellites) + sum(c.total for c in sc.constellations)
    launches = len(sc.launches)
    tag = " · ".join(filter(None, (_count(sats, "satellite", "satellites") if sats else "",
                                   _count(launches, "launch", "launches") if launches else "")))
    return sc.name, BLURBS.get(path.stem) or sc.description, tag, sc.description


def unreadable_card(path: Path, exc: Exception) -> tuple[str, str, str, str]:
    """The card of a scenario file that cannot be used: it says why."""
    return path.stem, f"Cannot be opened: {exc}", "unreadable", str(exc)


# The file dialog runs in a process of its own: Tk and SDL each expect the main
# thread of their process to themselves (on macOS mixing them can crash).
_PICKER = """\
import sys, tkinter
from tkinter import filedialog
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
name = filedialog.askopenfilename(
    parent=root, initialdir=sys.argv[1], title="Open a scenario or TLE file",
    filetypes=[("Scenarios and TLE files", "*.json *.tle *.txt"), ("All files", "*.*")])
sys.stdout.write(name or "")
"""


def pick_file(initial: Path) -> Path | None:
    """Ask for a file with the system's file dialog, opened in ``initial``. None
    if it was cancelled; OSError if no dialog could be shown (no tkinter, no
    display), with the reason."""
    done = subprocess.run([sys.executable, "-c", _PICKER, str(initial)], capture_output=True,
                          text=True, encoding="utf-8", check=False,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode:
        last = (done.stderr.strip().splitlines() or [f"exit code {done.returncode}"])[-1]
        raise OSError(last)
    name = done.stdout.strip()
    return Path(name) if name else None


def backdrop(size: tuple[int, int]) -> pygame.Surface:
    """The start menu's own background: deep space with stars and the glow of the
    Earth's limb along the bottom (the same picture for the same size)."""
    w, h = size
    surf = pygame.Surface(size)
    top, bottom = (3, 5, 12), (10, 18, 38)
    for y in range(h):
        pygame.draw.line(surf, theme.mix(top, bottom, y / max(1, h - 1)), (0, y), (w, y))
    rng = random.Random(7)
    for _ in range(w * h // 2600):
        x, y = rng.randrange(w), rng.randrange(h)
        v = rng.randrange(70, 230)
        c = (v, v, min(255, v + 25))
        if rng.random() < 0.08:
            pygame.draw.circle(surf, c, (x, y), max(1, px(1)))
        else:
            surf.set_at((x, y), c)
    # a planet far wider than the window: only its edge shows, a blue arc along the bottom
    r = max(w, h) * 2
    cx, cy = w // 2, h + r - px(70)
    glow = pygame.Surface(size, pygame.SRCALPHA)
    for k in range(24, 0, -1):
        pygame.draw.circle(glow, (70, 140, 255, 5), (cx, cy), r + k * px(3))
    pygame.draw.circle(glow, (8, 16, 34, 255), (cx, cy), r)
    pygame.draw.circle(glow, (90, 160, 255, 150), (cx, cy), r, max(1, px(2)))
    surf.blit(glow, (0, 0))
    return surf


SETUP_BG = (18, 27, 48)         # the setup box: a shade above the panel, below the fields
FILE_BG = (14, 20, 36)


class StartScreen:
    """The start menu, kept on the dialog stack. The setup form takes most of it;
    the file box sits beside it and the examples run along the bottom. Enter
    starts the simulation the form describes (or opens the typed path); Esc
    (or Resume) goes back to the simulation the menu was opened from."""

    GAP = 12           # design px
    HEAD = 80          # title and subtitle
    ROW = 32           # a form row, squeezed down to MIN_ROW in a low window
    MIN_ROW = 26
    BOX_TOP = 36       # a box's title line
    START_ROW = 56     # the Start button and the message beside it
    CHIP_H = 28
    MIN_CHIP_W = 124
    FILE_ROW = 24
    FOOT = 26          # the key hint line

    def __init__(self, app):
        self.app = a = app
        # the bundled scenarios are part of the program: a broken one raises here
        known = [a.scenario_dir / f"{stem}.json" for stem, _, _ in CARDS]
        cards = [(p, *scenario_card(p)) for p in known]
        self.cards = [c[:4] for c in cards]                 # (path, title, blurb, tag)
        self.descriptions = {c[0]: c[4] for c in cards}     # the fuller hover text
        self.files = self._saved_files()                    # (path, title, blurb, tag)
        self.widgets = {s.key: make_widget(s, self.changed) for s in SETUP_FIELDS}
        for k, v in (a.last_setup or {}).items():           # as it was last started
            w = self.widgets[k]
            if isinstance(w, Choice):
                w.set(v)
            else:
                w.value = v
        self.path = TextField("", "text")
        self.error = ""
        self.picker = Latest()                  # the file dialog's process, off the UI thread
        self.can_resume = a.started             # else nothing is loaded yet to go back to
        self.start_button = Button("Start simulation", self.start, accent=True, tooltip="Enter",
                                   hint="Start a new simulation with these settings")
        self.browse_button = Button("Browse...", self.browse, tooltip="Ctrl+O",
                                    hint="Pick a scenario (.json) or TLE file with your "
                                         "system's file dialog")
        self.open_button = Button("Open", self.open_typed,
                                  hint="Open the file whose path is typed above")
        self.head_buttons = [Button("Controls", lambda: setattr(a.opts, "help", True),
                                    tooltip="H", hint="Every mouse and keyboard control")]
        if self.can_resume:
            self.head_buttons.append(Button("Resume", self.close, tooltip="Esc",
                                            hint=f"Back to '{a.sim.scenario.name}' where it "
                                                 "stopped"))
        self.buttons = [self.start_button, self.browse_button, self.open_button,
                        *self.head_buttons]
        self.hot = None                         # ("card" | "file", k) under the mouse
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._backdrop: pygame.Surface | None = None
        self.layout()

    def _saved_files(self):
        """Scenario files that are not bundled, newest first: those saved in
        scenarios/user, and any other put in scenarios/. One that cannot be used
        gets a card saying why."""
        a = self.app
        paths = [p for p in a.scenario_dir.glob("*.json") if p.stem not in BLURBS]
        paths += list(a.user_scenario_dir.glob("*.json"))
        out = []
        for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                out.append((p, *scenario_card(p)[:3]))
            except (OSError, ValueError) as exc:
                out.append((p, *unreadable_card(p, exc)[:3]))
        return out

    # --- values -------------------------------------------------------------------------
    def raw(self) -> dict:
        """The setup form's unparsed values by key."""
        return {k: w.value for k, w in self.widgets.items()}

    def visible_fields(self, column):
        """The fields of ``column`` (RUN_FIELDS or PHYSICS_FIELDS) shown now."""
        raw = self.raw()
        return [s for s in column if is_visible(s, raw)]

    def changed(self, key):
        """A choice or checkbox changed: rows may appear or go."""
        self.error = ""
        self.layout()

    def text_fields(self) -> list[TextField]:
        """The text fields Tab moves between: the visible form's, then the path."""
        shown = self.visible_fields(RUN_FIELDS) + self.visible_fields(PHYSICS_FIELDS)
        return [self.widgets[s.key] for s in shown
                if isinstance(self.widgets[s.key], TextField)] + [self.path]

    # --- actions ------------------------------------------------------------------------
    def start(self):
        """Start the simulation the form describes, then open what it begins with;
        a value that does not work is shown beside the button instead."""
        raw = self.raw()
        try:
            parse_fields(SETUP_FIELDS, self.widgets)
            sc = setup_scenario(raw)
        except ValueError as exc:
            self.error = str(exc)
            return
        self.app.load_scenario(sc)              # leaves the menu
        self.app.last_setup = raw
        pygame.key.stop_text_input()
        then = START_WITH[raw["start_with"]]
        if then:
            self.app.open(then)

    def choose(self, k: int):
        """Load example ``k`` (a bundled scenario) and close."""
        self.open(self.cards[k][0])

    def open(self, path: Path):
        """Open a scenario or TLE file (see ``App.open_path``) and close; if it
        cannot be used, say why."""
        try:
            self.app.open_path(path)
        except (OSError, ValueError) as exc:
            self.error = f"Could not open {path.name}: {exc}"
            return
        pygame.key.stop_text_input()

    def open_typed(self):
        """Open the file whose path is typed in the path field."""
        text = self.path.value.strip().strip('"')     # Windows' "Copy as path" adds quotes
        if not text:
            self.error = "Type a file's path first, or use Browse..."
            return
        p = Path(text).expanduser()
        if not p.is_file():
            self.error = f"There is no file at {p}"
            return
        self.open(p)

    def browse(self):
        """Show the system's file dialog (its answer arrives in :meth:`poll`)."""
        if self.picker.busy:
            return
        a = self.app
        start = a.user_scenario_dir if a.user_scenario_dir.is_dir() else a.scenario_dir
        self.error = ""
        self.picker.submit("pick", lambda: pick_file(start))

    def poll(self):
        """Open the file picked in the file dialog, once it has closed."""
        done = self.picker.poll()
        if done is None:
            return
        _, path, exc = done
        if isinstance(exc, OSError):
            self.error = (f"The file dialog could not open ({exc}). Type the path instead, "
                          "or drop the file on the window.")
        elif exc is not None:
            raise exc                                   # a bug, not a missing dialog
        elif path is not None:
            self.path.value = str(path)
            self.open(path)

    def close(self):
        """Close and go back to the scenario that is loaded (Esc, Resume)."""
        pygame.key.stop_text_input()
        self.app.close_dialog(self)

    # --- layout -------------------------------------------------------------------------
    def _chip_grid(self, inner: int) -> tuple[int, int]:
        """(columns, rows) of the examples: as few rows as fit, evenly filled."""
        g = px(8)
        most = max(1, (inner + g) // (px(self.MIN_CHIP_W) + g))
        rows = math.ceil(len(self.cards) / most)
        return math.ceil(len(self.cards) / rows), rows

    def layout(self):
        """Size the menu to the window and place the boxes, fields and buttons."""
        sw, sh = self.app.screen.get_size()
        w = min(px(1100), sw - px(40))
        pad, g = px(18), px(self.GAP)
        inner = w - 2 * pad
        cols, rows = self._chip_grid(inner)
        chips_h = px(24) + rows * (px(self.CHIP_H) + px(6))
        n = max(len(RUN_FIELDS), len(PHYSICS_FIELDS))
        fixed = (px(self.HEAD) + px(self.BOX_TOP) + px(self.START_ROW) + g + chips_h
                 + px(self.FOOT))
        avail = min(sh - px(40), px(800)) - fixed
        row = max(px(self.MIN_ROW), min(px(self.ROW), avail // n))
        self.row_h = row
        h = fixed + n * row
        self.rect = pygame.Rect((sw - w) // 2, max(px(10), (sh - h) // 2), w, h)
        box_h = px(self.BOX_TOP) + n * row + px(self.START_ROW)
        setup_w = int((inner - g) * 0.69)
        top = self.rect.y + px(self.HEAD)
        self.setup_rect = pygame.Rect(self.rect.x + pad, top, setup_w, box_h)
        self.file_rect = pygame.Rect(self.setup_rect.right + g, top, inner - setup_w - g, box_h)
        self.examples_rect = pygame.Rect(self.rect.x + pad, self.setup_rect.bottom + g, inner,
                                         chips_h)
        self._layout_form()
        self._layout_files()
        cw = (inner - (cols - 1) * px(8)) // cols
        y0 = self.examples_rect.y + px(24)
        self.card_rects = [pygame.Rect(self.examples_rect.x + (k % cols) * (cw + px(8)),
                                       y0 + (k // cols) * (px(self.CHIP_H) + px(6)),
                                       cw, px(self.CHIP_H))
                           for k in range(len(self.cards))]
        x = self.rect.right - pad
        for b in reversed(self.head_buttons):
            bw = self.app.fonts.ui.size(b.text)[0] + px(28)
            b.rect = pygame.Rect(x - bw, self.rect.y + px(18), bw, px(32))
            x -= bw + px(10)

    def _layout_form(self):
        """The two columns of the setup form, and the Start button under them."""
        r, row = self.setup_rect, self.row_h
        cpad, gap = px(14), px(18)
        col_w = (r.w - 2 * cpad - gap) // 2
        self.label_w = int(col_w * 0.46)
        self.col_x = (r.x + cpad, r.x + cpad + col_w + gap)
        self.form_rows = []                         # (spec, row rect, widget)
        for x, column in zip(self.col_x, (RUN_FIELDS, PHYSICS_FIELDS), strict=True):
            y = r.y + px(self.BOX_TOP)
            for s in self.visible_fields(column):
                w = self.widgets[s.key]
                wx = x + self.label_w
                h = row - px(6)
                w.rect = (pygame.Rect(wx, y, px(24), h) if isinstance(w, Checkbox)
                          else pygame.Rect(wx, y, x + col_w - wx, h))
                self.form_rows.append((s, pygame.Rect(x, y, col_w, h), w))
                y += row
        bw = self.app.fonts.ui.size(self.start_button.text)[0] + px(40)
        self.start_button.rect = pygame.Rect(r.x + cpad, r.bottom - px(46), bw, px(34))

    def _layout_files(self):
        """The path field and its buttons, then the list of saved files."""
        r, fonts = self.file_rect, self.app.fonts
        cpad = px(14)
        y = r.y + px(self.BOX_TOP)
        self.path.rect = pygame.Rect(r.x + cpad, y, r.w - 2 * cpad, self.row_h - px(6))
        y = self.path.rect.bottom + px(8)
        x = r.x + cpad
        for b in (self.browse_button, self.open_button):
            bw = max(px(76), fonts.ui.size(b.text)[0] + px(24))
            b.rect = pygame.Rect(x, y, bw, px(30))
            x += bw + px(8)
        self.list_top = self.browse_button.rect.bottom + px(12)
        first = self.list_top + px(22)
        room = max(0, (r.bottom - px(28) - first) // px(self.FILE_ROW))
        self.file_rects = [pygame.Rect(r.x + cpad, first + k * px(self.FILE_ROW),
                                       r.w - 2 * cpad, px(self.FILE_ROW) - px(2))
                           for k in range(min(room, len(self.files)))]

    # --- events -------------------------------------------------------------------------
    def handle(self, ev) -> bool:
        """Keys, then buttons, the form's widgets, the files and the examples; modal.
        While the controls are shown, a key or click only closes them."""
        if self.app.opts.help:
            if ev.type == pygame.MOUSEBUTTONDOWN or (
                    ev.type == pygame.KEYDOWN
                    and ev.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_F1)):
                self.app.opts.help = False
            return True
        if ev.type == pygame.KEYDOWN and self._key(ev.key, getattr(ev, "mod", 0)):
            return True
        for b in self.buttons:
            if b.handle(ev):
                return True
        widgets = [w for _, _, w in self.form_rows] + [self.path]
        if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            for w in widgets:                   # every one sees clicks: text fields lose focus
                w.handle(ev)
            self.hot = self._spot(ev.pos)
            if self.hot and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                kind, k = self.hot
                if kind == "card":
                    self.choose(k)
                else:
                    self.open(self.files[k][0])
            return True
        for w in widgets:
            if w.handle(ev):
                if isinstance(w, TextField):
                    self.error = ""
                return True
        return True    # modal: swallow everything

    def _spot(self, pos):
        """("card", k) or ("file", k) at ``pos``, or None."""
        for kind, rects in (("card", self.card_rects), ("file", self.file_rects)):
            k = next((k for k, r in enumerate(rects) if r.collidepoint(pos)), -1)
            if k >= 0:
                return kind, k
        return None

    def _key(self, k, mods=0) -> bool:
        """The menu's keys; False leaves the key to the text field being typed in."""
        fields = self.text_fields()
        typing = next((w for w in fields if w.focused), None)
        if k == pygame.K_ESCAPE:
            if typing is not None:
                typing.focused = False
            elif self.can_resume:
                self.close()
        elif k == pygame.K_F11 or (k in (pygame.K_RETURN, pygame.K_KP_ENTER)
                                   and mods & pygame.KMOD_ALT):
            self.app.toggle_fullscreen()
        elif k == pygame.K_q and mods & pygame.KMOD_CTRL:
            self.app.running = False
        elif k == pygame.K_o and mods & pygame.KMOD_CTRL:
            self.browse()
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if typing is self.path:
                self.open_typed()
            else:
                self.start()
        elif k == pygame.K_TAB:
            cur = fields.index(typing) if typing is not None else -1
            for w in fields:
                w.focused = False
            fields[(cur + (-1 if mods & pygame.KMOD_SHIFT else 1)) % len(fields)].focused = True
        elif typing is not None:
            return False
        elif k in (pygame.K_h, pygame.K_F1):
            self.head_buttons[0].callback()
        return True

    # --- drawing ------------------------------------------------------------------------
    def draw(self, surf):
        """Cover the window with the backdrop, then the panel and its three parts."""
        fonts = self.app.fonts
        if self._backdrop is None or self._backdrop.get_size() != surf.get_size():
            self._backdrop = backdrop(surf.get_size())
        surf.blit(self._backdrop, (0, 0))
        tips.block(surf.get_rect())
        r = self.rect
        theme.panel(surf, r, (14, 20, 36, 240), theme.ACCENT, 10)
        fonts.draw(surf, "Satellite Space Flight", (r.x + px(18), r.y + px(14)), theme.TEXT,
                   fonts.big)
        fonts.draw(surf, "Set up a new simulation, or open one from a file. Everything can "
                         "be changed once it is running.", (r.x + px(20), r.y + px(52)),
                   theme.DIM, fonts.ui)
        for b in self.head_buttons:
            b.draw(surf, fonts)
        self._draw_form(surf)
        self._draw_files(surf)
        self._draw_examples(surf)
        back = "   Esc goes back to the simulation" if self.can_resume else ""
        fonts.draw(surf, f"Enter starts   Tab moves between the text fields{back}   Ctrl+N "
                         "brings this menu back", (r.x + px(20), r.bottom - px(9)),
                   theme.FAINT, fonts.small, "bottomleft")

    def _box(self, surf, rect, bg, edge, title, tip):
        pygame.draw.rect(surf, bg, rect, border_radius=px(8))
        pygame.draw.rect(surf, edge, rect, 1, border_radius=px(8))
        head = self.app.fonts.draw(surf, title, (rect.x + px(14), rect.y + px(10)), theme.TEXT,
                                   self.app.fonts.title)
        tips.add(head, tip)

    def _draw_form(self, surf):
        fonts, r = self.app.fonts, self.setup_rect
        self._box(surf, r, SETUP_BG, theme.ACCENT, "New simulation",
                  "Choose how the simulation starts, then Start (or Enter). Rest the mouse on "
                  "a field to see what it means.")
        cap = fonts.draw(surf, "Physics", (self.col_x[1], r.y + px(14)), theme.DIM, fonts.bold)
        tips.add(cap, "The forces that act and how orbits are computed. Physics (key P) "
                      "changes them while the simulation runs.")
        raw = self.raw()
        for s, row, w in self.form_rows:
            label = fonts.fit(s.label, self.label_w - px(8), fonts.ui)
            fonts.draw(surf, label, (row.x, row.centery), theme.DIM, fonts.ui, "midleft")
            w.draw(surf, fonts)
            tips.add(row, field_tip(s, raw))
        b = self.start_button
        b.draw(surf, fonts)
        if self.error:
            x = b.rect.right + px(14)
            lines = self.error_lines(r.right - px(14) - x)
            line_h = fonts.small.get_linesize()
            y = b.rect.centery - line_h * len(lines) // 2
            box = pygame.Rect(x, y, 0, 0)
            for line in lines:
                box.union_ip(fonts.draw(surf, line, (x, y), theme.BAD, fonts.small))
                y += line_h
            tips.add(box, self.error)
        else:
            fonts.draw(surf, "Enter", (b.rect.right + px(10), b.rect.centery), theme.FAINT,
                       fonts.small, "midleft")

    def error_lines(self, width: int) -> list[str]:
        """The message beside the Start button in at most two lines of ``width``
        px (the second shortened if it must be; the hover tip has all of it)."""
        fonts = self.app.fonts
        lines = tips.wrap(self.error, fonts.small, width)
        if len(lines) > 2:
            lines = [lines[0], " ".join(lines[1:])]
        return [fonts.fit(line, width, fonts.small) for line in lines]

    def _draw_files(self, surf):
        fonts, r = self.app.fonts, self.file_rect
        self._box(surf, r, FILE_BG, theme.PANEL_EDGE, "Open a file",
                  "Start from a file instead: a scenario (.json) holds a whole setup, a TLE "
                  "file adds the satellites it lists.")
        self.path.draw(surf, fonts)
        tips.add(self.path.rect, "The path of a scenario (.json) or TLE file; Enter opens it. "
                                 "Ctrl+V pastes.")
        if not self.path.value and not self.path.focused:
            hint = fonts.fit("Path of a .json or TLE file", self.path.rect.w - px(14),
                             fonts.small)
            fonts.draw(surf, hint, (self.path.rect.x + px(7), self.path.rect.centery),
                       theme.FAINT, fonts.small, "midleft")
        self.browse_button.draw(surf, fonts)
        self.open_button.draw(surf, fonts)
        if self.picker.busy:
            fonts.draw(surf, "waiting for the dialog", (self.open_button.rect.right + px(10),
                                                        self.open_button.rect.centery),
                       theme.FAINT, fonts.small, "midleft")
        x, w = r.x + px(14), r.w - px(28)
        shown = len(self.file_rects)
        if self.files:
            more = f" ({shown} of {len(self.files)})" if shown < len(self.files) else ""
            cap = "Saved simulations" + more
        else:
            cap = "Nothing saved yet"
        head = fonts.draw(surf, fonts.fit(cap, w, fonts.bold), (x, self.list_top), theme.DIM,
                          fonts.bold)
        tips.add(head, "Scenarios you saved (Ctrl+S, or Save snapshot in Scenarios), newest "
                       "first. Click one to open it." if self.files else
                       "Ctrl+S in a running simulation saves it into scenarios/user; it is "
                       "listed here.")
        if not self.files:
            y = head.bottom + px(6)
            for line in tips.wrap("Ctrl+S in a running simulation saves it, and it shows up "
                                  "here.", fonts.small, w):
                fonts.draw(surf, line, (x, y), theme.FAINT, fonts.small)
                y += fonts.small.get_linesize()
        for k, rect in enumerate(self.file_rects):
            path, title, blurb, tag = self.files[k]
            if self.hot == ("file", k):
                pygame.draw.rect(surf, theme.mix(FILE_BG, theme.ACCENT, 0.25), rect,
                                 border_radius=px(4))
            bad = tag == "unreadable"
            tag_w = fonts.small.size(tag)[0] if tag else 0
            name = fonts.fit(title, rect.w - tag_w - px(22), fonts.ui)
            fonts.draw(surf, name, (rect.x + px(6), rect.centery), theme.BAD if bad else
                       theme.TEXT, fonts.ui, "midleft")
            if tag:
                fonts.draw(surf, tag, (rect.right - px(6), rect.centery),
                           theme.BAD if bad else theme.ACCENT, fonts.small, "midright")
            tips.hot(rect)
            tips.add(rect, f"{blurb}\n{path.name}" if bad else
                     f"{title}{f' ({tag})' if tag else ''}: {blurb or 'no description'}\n"
                     f"{path.name} - click to open it.")
        fonts.draw(surf, fonts.fit("Or drop a file onto the window", w, fonts.small),
                   (x, r.bottom - px(10)), theme.FAINT, fonts.small, "bottomleft")

    def _draw_examples(self, surf):
        fonts = self.app.fonts
        cap = fonts.draw(surf, "Or watch a ready-made example:",
                         (self.examples_rect.x + px(2), self.examples_rect.y + px(2)),
                         theme.DIM, fonts.small)
        tips.add(cap, "Scenarios that come with the simulator, each showing one idea.")
        for k, rect in enumerate(self.card_rects):
            path, title, blurb, tag = self.cards[k]
            hot = self.hot == ("card", k)
            bg = theme.mix(theme.FIELD, theme.ACCENT, 0.3) if hot else theme.FIELD
            pygame.draw.rect(surf, bg, rect, border_radius=px(5))
            pygame.draw.rect(surf, theme.ACCENT if hot else theme.PANEL_EDGE, rect, 1,
                             border_radius=px(5))
            fonts.draw(surf, self.chip_label(k), rect.center, theme.TEXT, fonts.ui, "center")
            tips.hot(rect)
            tips.add(rect, "\n".join(filter(None, (f"{title}{f' ({tag})' if tag else ''}: {blurb}",
                                                   self.descriptions.get(path),
                                                   "Click to start it."))))

    def chip_label(self, k: int) -> str:
        """Example ``k``'s button text, shortened if its button is too narrow."""
        return self.app.fonts.fit(SHORT[self.cards[k][0].stem], self.card_rects[k].w - px(12))
