"""The start screen shown when the simulator opens: pick a scenario, launch a
rocket or build your own (Ctrl+N brings it back)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pygame

from ..scenario import Scenario
from . import theme, tips
from .theme import px
from .widgets import Button

# Bundled scenarios in the order the start screen shows them, each with a plain-language
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
        sats = len(d.get("satellites", [])) + sum(int(c.get("total", 0))
                                                  for c in d.get("constellations", []))
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


class StartScreen:
    """Modal start screen on the dialog stack: a grid of scenario cards and a row of
    ways in. Enter opens the highlighted card; Esc keeps the scenario already loaded."""

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
        self.buttons = [
            Button("Launch a rocket", lambda: self._then(lambda: a.open("launch")), accent=True,
                   tooltip="U", hint="Fly a rocket from any point on Earth into orbit"),
            Button("Build your own", self.build_your_own, tooltip="A",
                   hint="Start from an empty Earth and add satellites yourself"),
            Button("Saved scenarios", lambda: self._then(lambda: a.open("scenario")),
                   tooltip="Ctrl+O", hint="Open a scenario you saved, or any scenario file"),
            Button("Controls", lambda: self._then(lambda: setattr(a.opts, "help", True)),
                   tooltip="H", hint="Every mouse and keyboard control"),
        ]
        self.rect = pygame.Rect(0, 0, 0, 0)
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
        """Close and keep whatever scenario is loaded (Esc)."""
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

    # --- events -----------------------------------------------------------------------
    def handle(self, ev) -> bool:
        """Keys (Esc, Enter, arrows), buttons, then card hover and clicks; modal."""
        if ev.type == pygame.KEYDOWN:
            self._key(ev.key)
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

    def _key(self, k):
        n = len(self.cards)
        moves = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: -self.cols,
                 pygame.K_DOWN: self.cols}
        if k == pygame.K_ESCAPE:
            self.close()
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
        elif k == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self.buttons[2].callback()

    # --- drawing ----------------------------------------------------------------------
    def draw(self, surf):
        """Shade the window, then the panel, cards, buttons and hint."""
        fonts = self.app.fonts
        shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 90))
        surf.blit(shade, (0, 0))
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
        fonts.draw(surf, "Enter opens the highlighted scenario   Esc closes this screen   "
                         "Ctrl+N brings it back", (r.x + px(20), r.bottom - px(10)), theme.FAINT,
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
