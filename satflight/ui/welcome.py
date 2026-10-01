"""The start menu shown when the simulator opens: pick a scenario, launch a
rocket or build your own (Ctrl+N brings it back). It fills the window and the
simulation stands still behind it (see ``App.in_menu``)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pygame

from ..scenario import Scenario
from . import theme, tips
from .theme import px
from .widgets import Button

# Bundled scenarios in the order the start menu shows them, each with a plain-language
# blurb (the files' own descriptions are written for people who know the jargon).
CARDS = [
    ("default", "One satellite in every classic orbit, from low to geostationary."),
    ("launch_day", "Six rockets lift off from four continents; stages fall away."),
    ("hohmann_to_geo", "Two engine burns raise a satellite from low orbit to GEO."),
    ("rendezvous", "A chaser catches up with a space station and matches its speed."),
    ("starlink_shell", "A Starlink-like shell of 1,584 satellites around the planet."),
    ("gps_constellation", "24 GPS-like satellites in six planes, 20,000 km up."),
    ("polar_star", "Iridium-like polar orbits, with a seam where planes pass head-on."),
    ("molniya_tundra", "Long, looping orbits that linger over the far north."),
    ("j2_precession", "The Earth's bulge slowly twists orbits at six inclinations."),
    ("geo_drift", "Geostationary satellites drift from their slots toward two resting points."),
    ("drag_decay", "Air drag pulls three spacecraft down. The balloon falls first."),
    ("launches_and_arcs", "Short hops, one satellite reaching orbit, one escaping Earth."),
]
BLURBS = dict(CARDS)


def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def scenario_card(path: Path) -> tuple[str, str, str] | None:
    """(title, blurb, size tag) for a scenario file, or None if it cannot be read."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        sats = (sum(int(s.get("count", 1)) for s in d.get("satellites", []))   # copies count
                + sum(int(c.get("total", 0)) for c in d.get("constellations", [])))
        launches = len(d.get("launches", []))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    tag = " · ".join(filter(None, (_count(sats, "satellite", "satellites") if sats else "",
                                   _count(launches, "launch", "launches") if launches else "")))
    blurb = BLURBS.get(path.stem) or d.get("description", "")
    return d.get("name", path.stem), blurb, tag


def describe(path: Path) -> str:
    """The scenario file's own description ('' if it has none or cannot be read)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return str(d.get("description", ""))
    except (OSError, ValueError, AttributeError):
        return ""


def wrap(text: str, font, width: int, max_lines: int) -> list[str]:
    """Break ``text`` into at most ``max_lines`` lines of ``width`` px; the last one
    ends in '...' when text was cut."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and font.size(trial)[0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.size(last + "...")[0] > width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last + "..."
    return lines


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


class StartScreen:
    """The start menu, kept on the dialog stack: a grid of scenario cards and a row of
    ways in, over its own background across the whole window. Enter opens the
    highlighted card; Esc (or Resume) goes back to the simulation it was opened from."""

    GAP = 10           # design px
    HEAD = 84          # title and subtitle
    FOOT = 100         # buttons with their keys, and the hint line

    def __init__(self, app):
        self.app = app
        a = app
        known = [a.scenario_dir / f"{stem}.json" for stem, _ in CARDS]
        extra = sorted(p for p in a.scenario_dir.glob("*.json") if p.stem not in BLURBS)
        self.cards = [(p, *info) for p in known + extra
                      if p.exists() and (info := scenario_card(p)) is not None]
        self.descriptions = {p: describe(p) for p, *_ in self.cards}    # the fuller hover text
        cur = a.scenario_path
        self.focus = next((k for k, c in enumerate(self.cards)
                           if cur is not None and c[0] == cur), 0)
        self.can_resume = a.started         # else nothing is loaded yet to go back to
        self.buttons = [
            Button("Launch a rocket", lambda: self._then(lambda: a.open("launch")), accent=True,
                   tooltip="U", hint="Fly a rocket from any point on Earth into orbit"),
            Button("Build your own", self.build_your_own, tooltip="A",
                   hint="Start from an empty Earth and add satellites yourself"),
            Button("Saved scenarios", lambda: a.open("scenario"), tooltip="Ctrl+O",
                   hint="Open a scenario you saved, or any scenario file"),
            Button("Controls", lambda: setattr(a.opts, "help", True), tooltip="H",
                   hint="Every mouse and keyboard control"),
        ]
        if self.can_resume:
            self.buttons.append(Button("Resume", self.close, tooltip="Esc",
                                       hint=f"Back to '{a.sim.scenario.name}' where it stopped"))
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._backdrop: pygame.Surface | None = None
        self.card_rects: list[pygame.Rect] = []
        self.cols = 3
        self.layout()

    # --- actions ----------------------------------------------------------------------
    def choose(self, k: int):
        """Load card ``k``'s scenario and close."""
        self.close()
        self.app.load_scenario(self.cards[k][0])

    def build_your_own(self):
        """Start from an empty scenario with the Add-satellite dialog open."""
        self.close()
        self.app.load_scenario(Scenario(name="Empty"))
        self.app.open("add")

    def _then(self, action):
        self.close()
        action()

    def close(self):
        """Close and go back to the scenario that is loaded (Esc, Resume)."""
        self.app.close_dialog(self)

    # --- layout -----------------------------------------------------------------------
    def layout(self):
        """Size the panel to the window and place the cards and buttons."""
        sw, sh = self.app.screen.get_size()
        w = min(px(1000), sw - px(40))
        self.cols = 3 if w >= px(700) else 2
        rows = max(1, math.ceil(len(self.cards) / self.cols))
        g, head, foot = px(self.GAP), px(self.HEAD), px(self.FOOT)
        avail = min(sh - px(40), px(760)) - head - foot
        card_h = max(px(56), min(px(100), (avail - (rows - 1) * g) // rows))
        h = head + rows * card_h + (rows - 1) * g + foot
        self.rect = pygame.Rect((sw - w) // 2, max(px(10), (sh - h) // 2), w, h)
        card_w = (w - px(36) - (self.cols - 1) * g) // self.cols
        x0, y0 = self.rect.x + px(18), self.rect.y + head
        self.card_rects = [pygame.Rect(x0 + (k % self.cols) * (card_w + g),
                                       y0 + (k // self.cols) * (card_h + g), card_w, card_h)
                           for k in range(len(self.cards))]
        x = self.rect.x + px(18)
        by = self.rect.bottom - foot + px(14)
        for b in self.buttons:
            bw = self.app.fonts.ui.size(b.text)[0] + px(32)
            b.rect = pygame.Rect(x, by, bw, px(34))
            x += bw + px(10)
        if self.can_resume:                         # Resume sits apart, at the right
            self.buttons[-1].rect.right = self.rect.right - px(18)

    # --- events -----------------------------------------------------------------------
    def handle(self, ev) -> bool:
        """Keys (Esc, Enter, arrows), buttons, then card hover and clicks; modal.
        While the controls are shown, a key or click only closes them."""
        if self.app.opts.help:
            if ev.type == pygame.MOUSEBUTTONDOWN or (
                    ev.type == pygame.KEYDOWN
                    and ev.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_F1)):
                self.app.opts.help = False
            return True
        if ev.type == pygame.KEYDOWN:
            self._key(ev.key, getattr(ev, "mod", 0))
            return True
        for b in self.buttons:
            if b.handle(ev):
                return True
        if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            k = next((k for k, r in enumerate(self.card_rects) if r.collidepoint(ev.pos)), -1)
            if k >= 0:
                self.focus = k
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.choose(k)
        return True    # modal: swallow everything

    def _key(self, k, mods=0):
        n = len(self.cards)
        moves = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: -self.cols,
                 pygame.K_DOWN: self.cols}
        if k == pygame.K_ESCAPE:
            if self.can_resume:
                self.close()
        elif k == pygame.K_F11 or (k in (pygame.K_RETURN, pygame.K_KP_ENTER)
                                   and mods & pygame.KMOD_ALT):
            self.app.toggle_fullscreen()
        elif k == pygame.K_q and mods & pygame.KMOD_CTRL:
            self.app.running = False
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and n:
            self.choose(self.focus)
        elif k in moves and n:
            f = self.focus + moves[k]
            self.focus = f if 0 <= f < n else self.focus
        elif k == pygame.K_u:
            self.buttons[0].callback()
        elif k == pygame.K_a:
            self.build_your_own()
        elif k in (pygame.K_h, pygame.K_F1):
            self.buttons[3].callback()
        elif k == pygame.K_o and mods & pygame.KMOD_CTRL:
            self.buttons[2].callback()

    # --- drawing ----------------------------------------------------------------------
    def draw(self, surf):
        """Cover the window with the backdrop, then the panel, cards, buttons and hint."""
        fonts = self.app.fonts
        if self._backdrop is None or self._backdrop.get_size() != surf.get_size():
            self._backdrop = backdrop(surf.get_size())
        surf.blit(self._backdrop, (0, 0))
        tips.block(surf.get_rect())
        r = self.rect
        theme.panel(surf, r, (14, 20, 36, 240), theme.ACCENT, 10)
        fonts.draw(surf, "Satellite Space Flight", (r.x + px(18), r.y + px(14)), theme.TEXT,
                   fonts.big)
        fonts.draw(surf, "Pick a scenario to start. Everything can be changed once it is "
                         "running.", (r.x + px(20), r.y + px(54)), theme.DIM, fonts.ui)
        for k in range(len(self.cards)):
            self._draw_card(surf, k)
        for b in self.buttons:
            b.draw(surf, fonts)
            fonts.draw(surf, b.tooltip, (b.rect.centerx, b.rect.bottom + px(3)), theme.FAINT,
                       fonts.small, "midtop")
        back = "Esc goes back to the simulation   " if self.can_resume else ""
        fonts.draw(surf, f"Enter opens the highlighted scenario   {back}Ctrl+N brings this "
                         "menu back", (r.x + px(20), r.bottom - px(10)), theme.FAINT,
                   fonts.small, "bottomleft")

    def tag_place(self, k: int) -> str:
        """Where card ``k``'s size tag goes: "bottom"; or, when the blurb needs the
        bottom line, "title" (beside the title) if both fit there, else "" (left
        to the card's hover tip)."""
        fonts, rect = self.app.fonts, self.card_rects[k]
        _, title, _, tag = self.cards[k]
        if not tag or not self._blurb(k, 0)[-1].endswith("..."):
            return "bottom" if tag else ""
        fits = fonts.bold.size(title)[0] + fonts.small.size(tag)[0] + px(34) <= rect.w
        return "title" if fits else ""

    def _blurb(self, k: int, extra: int) -> list[str]:
        """Card ``k``'s blurb wrapped into the lines above the tag, plus ``extra``.
        Rather than cut it, the text may take 4 px of the right margin: glyph widths
        are whole pixels, so a scaled font runs a little wider than the scaled card."""
        font, rect = self.app.fonts.small, self.card_rects[k]
        line_h = font.get_linesize() + 1
        rows = max(1, (rect.h - px(34) - line_h) // line_h + extra)
        lines = wrap(self.cards[k][2], font, rect.w - px(24), rows)
        if lines[-1].endswith("..."):
            lines = wrap(self.cards[k][2], font, rect.w - px(20), rows)
        return lines

    def blurb_lines(self, k: int) -> list[str]:
        """Card ``k``'s blurb as drawn (one line more when the tag sits up top)."""
        return self._blurb(k, 0 if self.tag_place(k) == "bottom" else 1)

    def _draw_card(self, surf, k):
        fonts = self.app.fonts
        path, title, blurb, tag = self.cards[k]
        rect, focused = self.card_rects[k], k == self.focus
        tips.hot(rect)
        tips.add(rect, f"{title}{f' ({tag})' if tag else ''}: "
                       f"{self.descriptions.get(path) or blurb}\nClick (or Enter) to start it.")
        bg = theme.mix(theme.FIELD, theme.ACCENT, 0.28) if focused else theme.FIELD
        pygame.draw.rect(surf, bg, rect, border_radius=px(7))
        pygame.draw.rect(surf, theme.ACCENT if focused else theme.PANEL_EDGE, rect, 1,
                         border_radius=px(7))
        clip = surf.get_clip()
        surf.set_clip(rect.inflate(-px(4), -px(4)))
        fonts.draw(surf, title, (rect.x + px(12), rect.y + px(9)), theme.TEXT, fonts.bold)
        y = rect.y + px(32)
        for line in self.blurb_lines(k):
            fonts.draw(surf, line, (rect.x + px(12), y), theme.DIM, fonts.small)
            y += fonts.small.get_linesize() + 1
        place = self.tag_place(k)
        if place == "title":
            fonts.draw(surf, tag, (rect.right - px(10), rect.y + px(9) + fonts.bold.get_ascent()),
                       theme.ACCENT, fonts.small, "bottomright")
        elif place == "bottom":
            fonts.draw(surf, tag, (rect.right - px(10), rect.bottom - px(6)), theme.ACCENT,
                       fonts.small, "bottomright")
        surf.set_clip(clip)
