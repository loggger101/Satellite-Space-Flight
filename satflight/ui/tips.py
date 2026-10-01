"""Hover tooltips for every part of the UI.

Whatever draws a part of the window also says what it is: while drawing, it
calls :func:`add` with the part's rect and a sentence or two about it (and the
matching keyboard shortcut, if any). The regions are collected afresh every
frame, in draw order, so the last one under the mouse is the thing on top.
:func:`block` marks a surface that hides what was drawn before it (a modal
dialog's shade, the help overlay) without a tip of its own. Once the mouse has
rested :data:`DELAY` seconds on one region, :func:`draw` shows its tip.

The same pass records where a click does something: :func:`hot` marks a part
as clickable (a hand cursor) or as a text field (an I-beam), and
:func:`cursor_at` gives the mouse cursor for a point. Blockers hide these too.
"""

from __future__ import annotations

import time

import pygame

from . import theme
from .theme import px

DELAY = 0.4          # s the mouse rests on a part before its tip shows
MAX_W = 380          # design px of tip text per line

_regions: list[tuple[pygame.Rect, str, str]] = []
_hot: list[tuple[pygame.Rect, str | None]] = []     # (rect, cursor kind); None: a blocker
_rest: tuple = (None, 0.0)           # (region under the mouse, since when)


def begin():
    """Forget last frame's regions (App.draw calls this first)."""
    _regions.clear()
    _hot.clear()


def add(rect, text, key: str = "", clip: pygame.Rect | None = None):
    """Give the part drawn at ``rect`` the tip ``text`` (and shortcut ``key``).
    ``text`` may be a function returning it, called only if the tip is shown.
    ``clip`` is the visible area when the part sits in a scrolled panel."""
    if not (text or key):
        return
    r = pygame.Rect(rect)
    if clip is not None:
        r = r.clip(clip)
    if r.w > 0 and r.h > 0:
        _regions.append((r, text, key))


def block(rect):
    """``rect`` now covers everything drawn before it: no tips from under it."""
    _regions.append((pygame.Rect(rect), "", ""))
    _hot.append((pygame.Rect(rect), None))


def hot(rect, kind: str = "hand", clip: pygame.Rect | None = None):
    """A click on ``rect`` does something: show the ``kind`` cursor ("hand", or
    "text" for a text field) over it."""
    r = pygame.Rect(rect)
    if clip is not None:
        r = r.clip(clip)
    if r.w > 0 and r.h > 0:
        _hot.append((r, kind))


def cursor_at(pos) -> str | None:
    """The cursor kind of the topmost clickable part at ``pos`` (None: the arrow)."""
    if pos is None:
        return None
    for rect, kind in reversed(_hot):
        if rect.collidepoint(pos):
            return kind
    return None


def regions() -> list[tuple[pygame.Rect, str, str]]:
    """This frame's (rect, text, key) regions in draw order, blockers included
    (tests read it)."""
    return [(r, t() if callable(t) else t, k) for r, t, k in _regions]


def at(pos):
    """The topmost region at ``pos`` as (rect, text, key); None over a blocker or nothing."""
    if pos is None:
        return None
    for rect, text, key in reversed(_regions):
        if rect.collidepoint(pos):
            shown = text() if callable(text) else text
            return (rect, shown, key) if (shown or key) else None
    return None


def reset():
    """Restart the rest timer (the mouse left the window, or a click happened)."""
    global _rest
    _rest = (None, 0.0)


def draw(surf, fonts, pos, now: float | None = None):
    """The tip of the region the mouse at ``pos`` has rested on for ``DELAY``;
    returns the tip's rect, or None."""
    global _rest
    now = time.monotonic() if now is None else now
    reg = at(pos)
    ident = None if reg is None else (tuple(reg[0]), reg[1], reg[2])
    if ident != _rest[0]:
        _rest = (ident, now)
    if reg is None or now - _rest[1] < DELAY:
        return None
    return draw_tip(surf, fonts, reg[0], reg[1], reg[2], pos)


def wrap(text: str, font, width: int) -> list[str]:
    """``text`` broken into lines of at most ``width`` px ("\\n" starts a new line)."""
    out = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}" if cur else word
            if cur and font.size(trial)[0] > width:
                out.append(cur)
                cur = word
            else:
                cur = trial
        out.append(cur)
    return out


def draw_tip(surf, fonts, rect: pygame.Rect, text: str, key: str = "", pos=None):
    """A box explaining the part at ``rect``: under it (or under the mouse ``pos``
    for large parts), above when there is no room below, always inside the window;
    returns the box's rect."""
    lines = [fonts.render(line, theme.TEXT, fonts.ui) for line in wrap(text, fonts.ui, px(MAX_W))
             ] if text else []
    keys = fonts.render(key, theme.ACCENT, fonts.mono) if key else None
    line_h = fonts.ui.get_linesize()
    first_w = (lines[0].get_width() if lines else 0) + (keys.get_width() + px(14) if keys else 0)
    w = max([first_w] + [s.get_width() for s in lines[1:]]) + px(20)
    h = max(1, len(lines)) * line_h + px(10)
    mx, my = pos if pos is not None else rect.center
    small = rect.h <= px(48)
    below, above = (rect.bottom, rect.top) if small else (my + px(14), my - px(8))
    box = pygame.Rect(0, 0, w, h)
    box.midtop = (rect.centerx if rect.w <= px(220) else mx, below + px(6))
    screen = surf.get_rect().inflate(-px(8), -px(8))
    if box.bottom > screen.bottom:
        box.bottom = above - px(6)
    box.clamp_ip(screen)
    theme.panel(surf, box, (20, 30, 55, 245), theme.ACCENT, 5)
    x, y = box.x + px(10), box.y + px(5)
    for k, line in enumerate(lines):
        surf.blit(line, (x, y + k * line_h))
    if keys is not None:
        kx = box.right - px(10) - keys.get_width() if lines else x
        surf.blit(keys, (kx, y + (line_h - keys.get_height()) // 2))
    return box
