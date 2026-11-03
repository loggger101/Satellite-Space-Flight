"""A small immediate-feel widget toolkit for pygame: buttons, text fields,
choice cyclers, checkboxes and a declarative modal form dialog."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pygame

from . import theme, tips
from .theme import px


class Widget:
    """Base class: the owner sets ``rect``, forwards events to :meth:`handle` and
    calls :meth:`draw` every frame."""

    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.hover = False

    def handle(self, ev) -> bool:
        """React to a pygame event; True if it was consumed."""
        return False

    def draw(self, surf, fonts):
        """Draw the widget into ``surf`` at ``rect``."""


class Button(Widget):
    """Push button. ``active()`` highlights it as a toggle that is on; ``tooltip``
    records the matching keyboard shortcut and ``hint`` says what the button
    does (both shown in its hover tip, see :mod:`.tips`); ``accent`` marks the
    primary action."""

    def __init__(self, text: str, callback: Callable[[], Any],
                 active: Callable[[], bool] | None = None, tooltip: str = "",
                 accent: bool = False, hint: str = ""):
        super().__init__()
        self.text = text
        self.callback = callback
        self.active = active
        self.tooltip = tooltip
        self.hint = hint
        self.accent = accent

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and self.rect.collidepoint(ev.pos)):
            self.callback()
            return True
        return False

    def draw(self, surf, fonts):
        on = self.active() if self.active else False
        bg = theme.ACCENT_DARK if (on or self.accent) else theme.FIELD
        if self.hover:
            bg = theme.mix(bg, theme.ACCENT, 0.35)
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(5))
        edge = theme.ACCENT if on else theme.PANEL_EDGE
        pygame.draw.rect(surf, edge, self.rect, 1, border_radius=px(5))
        if self.text:
            fonts.draw(surf, self.text, self.rect.center, theme.TEXT, fonts.ui, "center")
        tips.add(self.rect, self.hint, self.tooltip)
        tips.hot(self.rect)


def _clipboard() -> str:
    """The clipboard's text ("" where there is no clipboard)."""
    with contextlib.suppress(pygame.error):     # no clipboard on this platform
        pygame.scrap.init()
        return (pygame.scrap.get_text() if hasattr(pygame.scrap, "get_text") else "") or ""
    return ""


def _copy(text: str):
    """Put ``text`` on the clipboard, where there is one."""
    with contextlib.suppress(pygame.error):
        pygame.scrap.init()
        if hasattr(pygame.scrap, "put_text"):
            pygame.scrap.put_text(text)


class TextField(Widget):
    """Single-line text entry; ``kind`` (float, floatx, int, text) sets how
    :meth:`parsed` reads the value. A click puts the caret where it lands; Left,
    Right, Home and End move it, Backspace and Delete remove a character.
    Gaining focus without a click (Tab) or Ctrl+A selects the whole text, so
    what is typed next replaces it. Ctrl+Backspace clears, Ctrl+C / Ctrl+X copy
    / cut the whole text, Ctrl+V pastes at the caret."""

    def __init__(self, value: str = "", kind: str = "float"):
        super().__init__()
        self.kind = kind
        self.error = False
        self._focused = False
        self.value = str(value)
        self._font = None             # as last drawn: maps a click to a caret position
        self._text_x = 0
        self._scroll = 0              # px the text is shifted left when it is too long
        self._typed = 0               # ticks of the last edit: the caret shows while typing

    @property
    def value(self) -> str:
        """The text as typed. Setting it puts the caret at its end."""
        return self._value

    @value.setter
    def value(self, text):
        self._value = str(text)
        self.caret = len(self._value)
        self.selected = False

    @property
    def focused(self) -> bool:
        """Whether typing goes here. Gaining focus selects the whole text."""
        return self._focused

    @focused.setter
    def focused(self, on):
        if on and not self._focused:
            self.caret, self.selected = len(self._value), bool(self._value)
        elif not on:
            self.selected = False
        self._focused = bool(on)

    def _edit(self, text: str, cut_from: int, cut_to: int):
        """Replace characters ``cut_from:cut_to`` with ``text`` (the whole text
        while it is selected) and leave the caret after it."""
        if self.selected:
            cut_from, cut_to = 0, len(self._value)
        self._value = self._value[:cut_from] + text + self._value[cut_to:]
        self.caret, self.selected, self.error = cut_from + len(text), False, False
        self._typed = pygame.time.get_ticks()

    def _move(self, to: int):
        self.caret, self.selected = max(0, min(len(self._value), to)), False
        self._typed = pygame.time.get_ticks()

    def _index_at(self, x: int) -> int:
        """The caret position nearest the window x coordinate ``x``."""
        if self._font is None:
            return len(self._value)
        widths = [self._font.size(self._value[:k])[0] for k in range(len(self._value) + 1)]
        return min(range(len(widths)), key=lambda k: abs(self._text_x + widths[k] - x))

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.focused = self.rect.collidepoint(ev.pos)
            if self.focused:
                self._move(self._index_at(ev.pos[0]))
                pygame.key.start_text_input()
            return self.focused
        if not self.focused:
            return False
        if ev.type == pygame.TEXTINPUT:
            self._edit(ev.text, self.caret, self.caret)
            return True
        if ev.type != pygame.KEYDOWN:
            return False
        k, ctrl, c = ev.key, ev.mod & pygame.KMOD_CTRL, self.caret
        if k == pygame.K_BACKSPACE:
            if ctrl:
                self.value = ""
            else:
                self._edit("", max(0, c - 1), c)
        elif k == pygame.K_DELETE:
            self._edit("", c, c + 1)
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            left = k == pygame.K_LEFT
            if self.selected:
                self._move(0 if left else len(self._value))
            else:
                self._move(c - 1 if left else c + 1)
        elif k == pygame.K_HOME:
            self._move(0)
        elif k == pygame.K_END:
            self._move(len(self._value))
        elif ctrl and k == pygame.K_a:
            self.selected = bool(self._value)
        elif ctrl and k in (pygame.K_c, pygame.K_x):
            _copy(self._value)
            if k == pygame.K_x:
                self.value = ""
        elif ctrl and k == pygame.K_v:
            self._edit(_clipboard().strip(), c, c)     # TLE lines keep their inner spaces
        else:
            return False
        return True

    def parsed(self):
        """The value converted per ``kind``; raises ValueError if it does not parse
        (or is not finite: "nan" and "inf" parse as floats but are no input)."""
        s = self.value.strip()
        if self.kind == "text":
            return s
        if self.kind == "floatx":        # float or keyword (e.g. "sso")
            try:
                x = float(s)
            except ValueError:
                return s
        else:
            x = float(s)
        if not math.isfinite(x):
            raise ValueError(f"{s!r} is not a finite number")
        if self.kind == "int":
            if x != int(x):
                raise ValueError(f"{s!r} is not a whole number")
            return int(x)
        return x

    def draw(self, surf, fonts):
        tips.hot(self.rect, "text")
        bg = theme.FIELD_FOCUS if self.focused else theme.FIELD
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(4))
        edge = theme.BAD if self.error else (theme.ACCENT if self.focused else theme.PANEL_EDGE)
        pygame.draw.rect(surf, edge, self.rect, 1, border_radius=px(4))
        clip = surf.get_clip()
        surf.set_clip(self.rect.inflate(-px(6), 0))
        txt = fonts.render(self.value, theme.TEXT, fonts.mono)
        room = self.rect.w - px(14)
        at = fonts.mono.size(self.value[:self.caret])[0]
        # a text too long for the field slides just enough to keep the caret in view
        if txt.get_width() <= room:
            self._scroll = 0
        else:
            self._scroll = max(at - room, min(self._scroll, at, txt.get_width() - room))
        x = self.rect.x + px(6) - self._scroll
        self._font, self._text_x = fonts.mono, x
        if self.focused and self.selected:
            pygame.draw.rect(surf, theme.ACCENT_DARK, (x, self.rect.y + px(4), txt.get_width(),
                                                      self.rect.h - px(8)))
        surf.blit(txt, (x, self.rect.centery - txt.get_height() // 2))
        now = pygame.time.get_ticks()
        if self.focused and not self.selected and ((now - self._typed) // 500) % 2 == 0:
            cx = x + at
            pygame.draw.line(surf, theme.TEXT, (cx, self.rect.y + px(5)),
                             (cx, self.rect.bottom - px(5)), px(1))
        surf.set_clip(clip)


class Choice(Widget):
    """One of several options. A click opens the list of them (see
    :meth:`handle_list`); the wheel steps through them without opening it, and
    a right-click steps back. ``tips`` explains options in the list on hover.

    While the list is open (``open``), its owner sends every event to
    :meth:`handle_list` instead of anywhere else and calls :meth:`draw_list`
    after drawing everything else (see :func:`open_list`)."""

    ITEM = 26                      # design px of a row in the list

    def __init__(self, options: list[str], index: int = 0, on_change=None, tips=None):
        super().__init__()
        self.options = list(options)
        self.index = max(0, min(index, len(self.options) - 1))
        self.on_change = on_change
        self.tips = dict(tips or {})
        self.open = False
        self.list_rect = pygame.Rect(0, 0, 0, 0)
        self._lit = self.index        # the row the mouse or the arrow keys are on
        self._first = 0               # the first row shown, when the list scrolls
        self._rows = 0                # how many rows the list shows
        self._widest = (None, 0)      # (options, px of the widest) as last measured

    @property
    def value(self):
        """The selected option (empty string if there are none)."""
        return self.options[self.index] if self.options else ""

    def set(self, value):
        """Select ``value`` if it is one of the options."""
        if value in self.options:
            self.index = self.options.index(value)

    def _step(self, d):
        """Move ``d`` options along (wrapping) and notify ``on_change``."""
        if not self.options:
            return
        self.index = (self.index + d) % len(self.options)
        if self.on_change:
            self.on_change(self.value)

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            if ev.button == 1 and self.options:
                self.open, self._lit, self._first = True, self.index, 0
                self._place()
                self._show(self._lit)
                return True
            if ev.button == 5:
                self._step(1)
                return True
            if ev.button in (3, 4):
                self._step(-1)
                return True
        elif ev.type == pygame.MOUSEWHEEL and self.hover:
            self._step(-1 if ev.y > 0 else 1)
            return True
        return False

    # --- the open list ---------------------------------------------------------------
    def _place(self):
        """Put the list under the field (above it when there is more room there),
        as many rows as fit."""
        screen = pygame.display.get_surface()
        area = (screen.get_rect() if screen is not None else self.rect.inflate(0, px(800))
                ).inflate(-px(16), -px(16))
        row, n = px(self.ITEM), len(self.options)
        below, above = area.bottom - self.rect.bottom, self.rect.top - area.top
        up = n * row > below and above > below
        self._rows = max(1, min(n, ((above if up else below) - px(6)) // row))
        w = max(self.rect.w, self._widest[1] + px(30))
        h = self._rows * row + px(4)
        y = self.rect.top - px(2) - h if up else self.rect.bottom + px(2)
        self.list_rect = pygame.Rect(max(area.x, min(self.rect.x, area.right - w)), y, w, h)
        self._first = max(0, min(self._first, n - self._rows))

    def _show(self, k):
        """Scroll the list so row ``k`` is in it."""
        self._first = max(0, min(self._first, k, len(self.options) - self._rows))
        self._first = max(self._first, k - self._rows + 1)

    def _row_at(self, pos) -> int:
        """The option under ``pos`` in the open list, or -1."""
        if not self.list_rect.collidepoint(pos):
            return -1
        k = self._first + (pos[1] - self.list_rect.y - px(2)) // px(self.ITEM)
        return k if self._first <= k < min(len(self.options), self._first + self._rows) else -1

    def choose(self, k: int):
        """Close the list with option ``k`` chosen (``on_change`` if it is new)."""
        self.open = False
        if k != self.index:
            self.index = k
            if self.on_change:
                self.on_change(self.value)

    def handle_list(self, ev) -> bool:
        """Events while the list is open: the mouse lights and picks a row, the
        wheel scrolls; Up, Down, Home, End and Enter (or Space) do the same from
        the keyboard. Esc, Tab or a click outside closes it unchanged. Takes
        every event (the list is modal)."""
        n = len(self.options)
        if ev.type == pygame.MOUSEMOTION:
            k = self._row_at(ev.pos)
            self._lit = k if k >= 0 else self._lit
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            k = self._row_at(ev.pos)
            if k >= 0:
                self.choose(k)
            else:
                self.open = False
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            self.open = False
        elif ev.type == pygame.MOUSEWHEEL:
            self._first = max(0, min(n - self._rows, self._first - ev.y))
        elif ev.type == pygame.KEYDOWN:
            moves = {pygame.K_UP: self._lit - 1, pygame.K_DOWN: self._lit + 1,
                     pygame.K_PAGEUP: self._lit - self._rows,
                     pygame.K_PAGEDOWN: self._lit + self._rows,
                     pygame.K_HOME: 0, pygame.K_END: n - 1}
            if ev.key in moves:
                self._lit = max(0, min(n - 1, moves[ev.key]))
                self._show(self._lit)
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.choose(self._lit)
            elif ev.key in (pygame.K_ESCAPE, pygame.K_TAB):
                self.open = False
        return True

    def draw_list(self, surf, fonts):
        """The open list over everything drawn before it."""
        self._place()
        r, row = self.list_rect, px(self.ITEM)
        tips.block(r)
        pygame.draw.rect(surf, (14, 20, 36), r, border_radius=px(4))
        pygame.draw.rect(surf, theme.ACCENT, r, 1, border_radius=px(4))
        for k in range(self._first, min(len(self.options), self._first + self._rows)):
            line = pygame.Rect(r.x + px(2), r.y + px(2) + (k - self._first) * row,
                               r.w - px(4), row)
            if k == self._lit:
                pygame.draw.rect(surf, theme.ACCENT_DARK, line, border_radius=px(3))
            opt = self.options[k]
            fonts.draw(surf, fonts.fit(opt, line.w - px(30)), (line.x + px(22), line.centery),
                       theme.TEXT if k == self.index else theme.DIM, fonts.ui, "midleft")
            if k == self.index:
                pygame.draw.circle(surf, theme.ACCENT, (line.x + px(10), line.centery), px(3))
            tips.hot(line)
            tips.add(line, self.tips.get(opt, ""))
        if len(self.options) > self._rows:
            theme.scrollbar(surf, r.right - px(5), r.inflate(0, -px(8)), self._first,
                            self._rows, len(self.options))

    def draw(self, surf, fonts):
        tips.hot(self.rect)
        if self._widest[0] != self.options:
            self._widest = (list(self.options),
                            max((fonts.ui.size(o)[0] for o in self.options), default=0))
        lit = self.hover or self.open
        bg = theme.mix(theme.FIELD, theme.ACCENT, 0.2) if lit else theme.FIELD
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(4))
        pygame.draw.rect(surf, theme.ACCENT if self.open else theme.PANEL_EDGE, self.rect, 1,
                         border_radius=px(4))
        clip = surf.get_clip()
        # clip on the right only (for the arrow): a symmetric inset cut the first letter
        surf.set_clip(pygame.Rect(self.rect.x + px(2), self.rect.y, self.rect.w - px(26),
                                  self.rect.h))
        fonts.draw(surf, self.value, (self.rect.x + px(8), self.rect.centery), theme.TEXT,
                   fonts.ui, "midleft")
        surf.set_clip(clip)
        cx, cy, s = self.rect.right - px(13), self.rect.centery, px(4)
        pts = ([(cx - s, cy + s // 2), (cx + s, cy + s // 2), (cx, cy - s // 2)] if self.open
               else [(cx - s, cy - s // 2), (cx + s, cy - s // 2), (cx, cy + s // 2)])
        pygame.draw.polygon(surf, theme.ACCENT if lit else theme.DIM, pts)


class Checkbox(Widget):
    """A checkbox; ``on_change(value)`` fires on every click."""

    def __init__(self, value: bool = False, on_change=None):
        super().__init__()
        self.value = bool(value)
        self.on_change = on_change

    def handle(self, ev):
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos)):
            self.value = not self.value
            if self.on_change:
                self.on_change(self.value)
            return True
        return False

    def draw(self, surf, fonts):
        tips.hot(self.rect)
        box = pygame.Rect(self.rect.x, self.rect.centery - px(9), px(18), px(18))
        pygame.draw.rect(surf, theme.FIELD, box, border_radius=px(3))
        edge = theme.ACCENT if self.value else theme.PANEL_EDGE
        pygame.draw.rect(surf, edge, box, 1, border_radius=px(3))
        if self.value:
            x, y = box.topleft
            tick = [(x + px(4), box.centery), (x + px(8), box.bottom - px(4)),
                    (box.right - px(4), y + px(4))]
            pygame.draw.lines(surf, theme.ACCENT, False, tick, px(2))


@dataclass
class MenuItem:
    """A command of a :class:`Menu`: its label, the keys that do the same, what
    it does and a line saying so (shown on hover)."""

    label: str
    keys: str
    action: Callable[[], Any]
    hint: str = ""


class Menu:
    """A pop-up list of commands under a title, opened at ``pos`` (a right-click
    menu). While ``open`` its owner sends it every event (it is modal) and draws
    it over everything else. The mouse or Up / Down light a command, a click or
    Enter runs it; Esc or a click anywhere else closes the menu."""

    ITEM = 26                      # design px per command

    def __init__(self, title: str, items: list[MenuItem], pos):
        self.title = title
        self.items = list(items)
        self.pos = pos
        self.open = True
        self.rect = pygame.Rect(pos, (0, 0))
        self.lit = -1

    def place(self, fonts, area):
        """Size the menu to its text and put it at ``pos``, inside ``area``
        (flipped left or up where it would run out)."""
        widest = max([fonts.small.size(self.title)[0] + px(8)]
                     + [fonts.ui.size(i.label)[0] + fonts.small.size(i.keys)[0] + px(34)
                        for i in self.items])
        w, h = max(px(170), widest + px(16)), px(26) + len(self.items) * px(self.ITEM) + px(4)
        x, y = self.pos
        x = x if x + w <= area.right else x - w
        y = y if y + h <= area.bottom else y - h
        self.rect = pygame.Rect(max(area.x, x), max(area.y, y), w, h)

    def _row(self, k) -> pygame.Rect:
        return pygame.Rect(self.rect.x + px(2), self.rect.y + px(24) + k * px(self.ITEM),
                           self.rect.w - px(4), px(self.ITEM))

    def _item_at(self, pos) -> int:
        return next((k for k in range(len(self.items)) if self._row(k).collidepoint(pos)), -1)

    def run(self, k):
        """Close the menu and run command ``k``."""
        self.open = False
        self.items[k].action()

    def handle(self, ev) -> bool:
        """Every event while the menu is open (it takes them all)."""
        if ev.type == pygame.MOUSEMOTION:
            k = self._item_at(ev.pos)
            self.lit = k if k >= 0 else self.lit
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
            k = self._item_at(ev.pos)
            if k >= 0 and ev.button == 1:
                self.run(k)
            elif not self.rect.collidepoint(ev.pos):
                self.open = False
        elif ev.type == pygame.KEYDOWN:
            n = len(self.items)
            if ev.key in (pygame.K_UP, pygame.K_DOWN):
                step = -1 if ev.key == pygame.K_UP else 1
                self.lit = (self.lit + step) % n if self.lit >= 0 else (0 if step > 0 else n - 1)
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and self.lit >= 0:
                self.run(self.lit)
            elif ev.key in (pygame.K_ESCAPE, pygame.K_TAB):
                self.open = False
        return True

    def draw(self, surf, fonts):
        """The menu over everything drawn before it."""
        self.place(fonts, surf.get_rect().inflate(-px(8), -px(8)))
        r = self.rect
        tips.block(r)
        pygame.draw.rect(surf, (14, 20, 36), r, border_radius=px(4))
        pygame.draw.rect(surf, theme.ACCENT, r, 1, border_radius=px(4))
        fonts.draw(surf, fonts.fit(self.title, r.w - px(20), fonts.small),
                   (r.x + px(10), r.y + px(6)), theme.FAINT, fonts.small)
        pygame.draw.line(surf, theme.PANEL_EDGE, (r.x + px(6), r.y + px(23)),
                         (r.right - px(7), r.y + px(23)))
        for k, item in enumerate(self.items):
            line = self._row(k)
            if k == self.lit:
                pygame.draw.rect(surf, theme.ACCENT_DARK, line, border_radius=px(3))
            fonts.draw(surf, item.label, (line.x + px(10), line.centery), theme.TEXT, fonts.ui,
                       "midleft")
            fonts.draw(surf, item.keys, (line.right - px(8), line.centery), theme.FAINT,
                       fonts.small, "midright")
            tips.hot(line)
            tips.add(line, item.hint, item.keys)


# --- Form dialog ------------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """One row of a :class:`FormDialog`; ``visible(raw_values)`` hides it conditionally.
    ``tip`` explains the field on hover; ``option_tips`` adds a line about the
    option a choice currently shows."""

    key: str
    label: str
    kind: str = "float"            # float | floatx | int | text | choice | bool | info
    default: Any = ""
    options: list = field(default_factory=list)
    visible: Callable[[dict], bool] | None = None
    wide: bool = False             # text spanning the full row (e.g. TLE lines)
    tip: str = ""
    option_tips: dict = field(default_factory=dict)


CHOICE_HELP = "Click for the list of options; the wheel steps through them."


def open_list(widgets) -> Choice | None:
    """The choice among ``widgets`` whose list is open, if any: it takes every
    event and is drawn last."""
    return next((w for w in widgets if isinstance(w, Choice) and w.open), None)


def make_widget(s: FieldSpec, changed: Callable[[str], Any]) -> Widget | None:
    """The widget for field ``s`` (None for an info row). Choices and checkboxes
    call ``changed(key)`` when they change; text fields leave that to their owner."""
    if s.kind == "choice":
        opts = list(s.options)
        idx = opts.index(s.default) if s.default in opts else 0
        return Choice(opts, idx, on_change=lambda v, k=s.key: changed(k), tips=s.option_tips)
    if s.kind == "bool":
        return Checkbox(bool(s.default), on_change=lambda v, k=s.key: changed(k))
    if s.kind == "info":
        return None
    return TextField(s.default, s.kind)


def is_visible(s: FieldSpec, raw: dict) -> bool:
    """Whether field ``s`` shows, given the unparsed values ``raw``."""
    return s.visible is None or bool(s.visible(raw))


def parse_fields(specs: list[FieldSpec], widgets: dict[str, Widget]) -> dict:
    """Parsed values of the *visible* fields; raises ValueError naming the first
    text field that does not parse (and marks it red)."""
    raw = {k: w.value for k, w in widgets.items()}
    out = {}
    for s in specs:
        if s.kind == "info" or not is_visible(s, raw):
            continue
        w = widgets[s.key]
        if isinstance(w, TextField):
            try:
                out[s.key] = w.parsed()
                w.error = False
            except ValueError as exc:
                w.error = True
                raise ValueError(f"'{s.label}' needs a number ({exc})") from None
        else:
            out[s.key] = w.value
    return out


def message_lines(fonts, text: str, width: int) -> list[str]:
    """``text`` in at most two lines of ``width`` px of the small font, the
    second shortened if it must be (the hover tip has all of it)."""
    lines = tips.wrap(text, fonts.small, max(px(40), width))
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return [fonts.fit(line, max(px(40), width), fonts.small) for line in lines]


def field_tip(s: FieldSpec, raw: dict) -> str:
    """The hover tip of field ``s``: its ``tip``, what the chosen option means,
    and how to change a choice."""
    parts = [s.tip]
    if s.kind == "choice":
        parts += [s.option_tips.get(raw.get(s.key), ""), CHOICE_HELP]
    return "\n".join(p for p in parts if p)


class FormDialog:
    """A modal form. ``side`` is an optional panel drawn to the right of the
    fields: any object with ``width``, ``min_height``, ``draw(surf, rect,
    dialog)`` and ``handle(ev, dialog) -> bool``. It is left out when the
    window is too narrow for it. ``extra_buttons`` are (text, callback) or
    (text, callback, hint); ``ok_hint`` explains the OK button."""

    ROW = 32                       # design px, like ``width`` and the side panel's sizes
    MIN_ROW = 22

    def __init__(self, app, title: str, specs: list[FieldSpec], on_ok, ok_text: str = "OK",
                 width: int = 520, on_change=None, extra_buttons=None, subtitle: str = "",
                 side=None, ok_hint: str = ""):
        self.app = app
        self.title = title
        self.subtitle = subtitle
        self.specs = specs
        self.on_ok = on_ok
        self.on_change = on_change
        self.width = width
        self.side = side
        self.side_rect = None
        self.row_h = px(self.ROW)
        self.error = ""
        self.info: dict[str, str] = {}
        self.widgets: dict[str, Widget] = {}
        for s in specs:
            w = make_widget(s, self.changed)
            if w is None:
                self.info[s.key] = str(s.default)
            else:
                self.widgets[s.key] = w
        self.buttons = [Button(ok_text, self.submit, accent=True, tooltip="Enter",
                               hint=ok_hint or f"{ok_text} with these settings"),
                        Button("Cancel", self.close, tooltip="Esc",
                               hint="Close without changing anything")]
        for text, cb, *hint in (extra_buttons or []):
            self.buttons.insert(-1, Button(text, lambda cb=cb: self._extra(cb),
                                           hint=hint[0] if hint else ""))
        self.rect = pygame.Rect(0, 0, px(width), px(100))
        self.layout()

    # --- values -----------------------------------------------------------------------
    def raw(self) -> dict:
        """Unparsed widget values by key (text as typed)."""
        return {k: w.value for k, w in self.widgets.items()}

    def set(self, key, value):
        """Set a field's value, whatever kind of widget it is (KeyError for a
        field the form does not have, ValueError for an option a choice lacks)."""
        w = self.widgets.get(key)
        if isinstance(w, TextField):
            w.value = str(value)
        elif isinstance(w, Choice):
            if value not in w.options:
                raise ValueError(f"{key}: {value!r} is not one of {w.options}")
            w.set(value)
        elif isinstance(w, Checkbox):
            w.value = bool(value)
        elif key in self.info:
            self.info[key] = str(value)
        else:
            raise KeyError(f"the form has no field {key!r}")

    def values(self) -> dict:
        """Parsed values of *visible* fields; raises ValueError naming the field."""
        return parse_fields(self.specs, self.widgets)

    def _visible(self, s: FieldSpec, raw) -> bool:
        return is_visible(s, raw)

    def changed(self, key):
        """A field changed: tell ``on_change`` and re-layout (visibility may change)."""
        if self.on_change:
            try:
                self.on_change(self, key)
            except ValueError as exc:  # bad intermediate input: say so, keep the dialog
                self.error = str(exc)
        self.layout()

    def _extra(self, cb):
        """Run an extra button's callback; its return value (or error) is shown."""
        try:
            msg = cb(self)
            self.error = msg or ""
        except (ValueError, OSError) as exc:    # bad input, or a file that cannot be used
            self.error = str(exc)

    # --- layout -----------------------------------------------------------------------
    def layout(self):
        """Size and center the dialog and place every visible row and button."""
        raw = self.raw()
        rows = [s for s in self.specs if self._visible(s, raw)]
        sw, sh = self.app.screen.get_size()
        width = px(self.width)
        side_w = 0
        if self.side is not None:
            side_w = min(px(self.side.width), sw - width - px(34))
            side_w = side_w if side_w >= px(260) else 0
        total_w = width + (side_w + px(6) if side_w else 0)
        top = px(52 + (22 if self.subtitle else 0))
        avail = sh - px(20) - top - px(90)
        # squeeze the rows (never below MIN_ROW) rather than run off the screen
        row = max(px(self.MIN_ROW), min(px(self.ROW), avail // max(1, len(rows))))
        body = row * len(rows) + px(12)
        if side_w:
            body = max(body, min(px(self.side.min_height), avail + px(12)))
        h = min(top + body + px(78), sh - px(20))
        self.row_h = row
        self.rect = pygame.Rect((sw - total_w) // 2, max(px(10), (sh - h) // 2), total_w, h)
        self.side_rect = (pygame.Rect(self.rect.x + width, self.rect.y + top - px(6), side_w,
                                      self.rect.bottom - px(56) - (self.rect.y + top - px(6)))
                          if side_w else None)
        y = self.rect.y + top
        lx = self.rect.x + px(18)
        right = self.rect.x + width - px(18)
        wx = self.rect.x + int(width * 0.44)
        ww = right - wx
        gap = px(6)                                     # between rows
        self._rows = []
        for s in rows:
            if s.kind == "info":
                self._rows.append((s, pygame.Rect(lx, y, width - px(36), row - gap)))
            else:
                w = self.widgets[s.key]
                if s.wide:
                    w.rect = pygame.Rect(lx + px(60), y, right - lx - px(60), row - gap)
                elif s.kind == "bool":
                    w.rect = pygame.Rect(wx, y, px(24), row - gap)
                else:
                    w.rect = pygame.Rect(wx, y, ww, row - gap)
                self._rows.append((s, w.rect))
            y += row
        bx = self.rect.right - px(18)
        for b in reversed(self.buttons):
            bw = max(px(90), self.app.fonts.ui.size(b.text)[0] + px(28))
            b.rect = pygame.Rect(bx - bw, self.rect.bottom - px(46), bw, px(32))
            bx -= bw + px(10)
        self._visible_keys = {s.key for s in rows}

    # --- events -----------------------------------------------------------------------
    def submit(self):
        """Parse and pass the values to ``on_ok``; close unless it returns a message."""
        try:
            vals = self.values()
            msg = self.on_ok(vals)
        except (ValueError, OSError) as exc:    # bad input or an unusable file; else a bug
            self.error = str(exc)
            return
        if msg:
            self.error = msg
        else:
            self.close()

    def close(self):
        """Close without submitting (Cancel, Esc)."""
        pygame.key.stop_text_input()
        self.app.close_dialog(self)

    def _text_fields(self):
        return [self.widgets[s.key] for s in self.specs
                if s.key in self._visible_keys
                and isinstance(self.widgets.get(s.key), TextField)]

    def handle(self, ev) -> bool:
        """An open option list takes everything; else keyboard shortcuts (Esc,
        Enter, Tab), then buttons, side panel and fields."""
        menu = open_list(self.widgets.values())
        if menu is not None:
            return menu.handle_list(ev)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.close()
                return True
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.submit()
                return True
            if ev.key == pygame.K_TAB:
                tf = self._text_fields()
                if tf:
                    cur = next((i for i, w in enumerate(tf) if w.focused), -1)
                    for w in tf:
                        w.focused = False
                    step = -1 if ev.mod & pygame.KMOD_SHIFT else 1
                    tf[(cur + step) % len(tf)].focused = True
                return True
        for b in self.buttons:
            if b.handle(ev):
                return True
        if self.side_rect is not None and self.side.handle(ev, self):
            return True
        visible = [s for s in self.specs if s.key in self._visible_keys and s.key in self.widgets]
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            # every widget sees clicks so text fields can lose focus
            for s in visible:
                self.widgets[s.key].handle(ev)
            return True
        for s in visible:
            w = self.widgets[s.key]
            before = w.value
            if w.handle(ev):
                if isinstance(w, TextField) and w.value != before:   # not a caret move
                    self.changed(s.key)
                return True
        return True    # modal: swallow everything

    def field_tip(self, s: FieldSpec, raw: dict) -> str:
        """The hover tip of field ``s`` (see :func:`field_tip`)."""
        return field_tip(s, raw)

    def draw(self, surf):
        """Shade the window, then draw the frame, rows, side panel, buttons and error."""
        fonts = self.app.fonts
        surf.blit(theme.veil(surf.get_size(), (0, 0, 0, 110)), (0, 0))
        tips.block(surf.get_rect())             # nothing under the shade explains itself
        theme.panel(surf, self.rect, (14, 20, 36, 245), theme.ACCENT)
        head = fonts.draw(surf, self.title, (self.rect.x + px(18), self.rect.y + px(14)),
                          theme.TEXT, fonts.title)
        tips.add(head, "Enter applies the form, Esc closes it, Tab moves between the text "
                       "fields. Rest the mouse on a field to see what it means.")
        if self.subtitle:
            sub = fonts.draw(surf, self.subtitle, (self.rect.x + px(18), self.rect.y + px(42)),
                             theme.DIM, fonts.small)
            tips.add(sub, self.subtitle)
        raw = self.raw()
        for s, r in self._rows:
            if s.kind == "info":
                text = self.info.get(s.key, "")
                fonts.draw(surf, text, (r.x, r.centery), theme.ACCENT, fonts.small, "midleft")
                tips.add(r, "\n".join(p for p in (text, s.tip) if p))
                continue
            fonts.draw(surf, s.label, (self.rect.x + px(18), r.centery), theme.DIM, fonts.ui,
                       "midleft")
            self.widgets[s.key].draw(surf, fonts)
            tips.add(pygame.Rect(self.rect.x + px(18), r.y, r.right - self.rect.x - px(18), r.h),
                     self.field_tip(s, raw))
        if self.side_rect is not None:
            self.side.draw(surf, self.side_rect, self)
        for b in self.buttons:
            b.draw(surf, fonts)
        if self.error:
            x = self.rect.x + px(18)
            left = min(b.rect.x for b in self.buttons)
            lines = message_lines(fonts, self.error, left - px(14) - x)
            line_h = fonts.small.get_linesize()
            y = self.rect.bottom - px(30) - line_h * len(lines) // 2
            box = pygame.Rect(x, y, 0, 0)
            for line in lines:
                box.union_ip(fonts.draw(surf, line, (x, y), theme.BAD, fonts.small))
                y += line_h
            tips.add(box, self.error)
        menu = open_list(self.widgets.values())
        if menu is not None:
            menu.draw_list(surf, fonts)
