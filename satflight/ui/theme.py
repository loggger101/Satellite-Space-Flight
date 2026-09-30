"""Colors, fonts, the UI scale and a text-surface cache shared by every UI module.

Sizes in the UI code are written in design pixels, the layout at 100 % display
scaling. :func:`px` turns them into screen pixels: the window has the screen's
real pixels (the process is DPI-aware), so at 125 % scaling every size is
drawn 1.25 times larger and the UI stays the size the user chose, only sharper."""

from __future__ import annotations

import os
from pathlib import Path

import pygame

BG = (4, 6, 14)
PANEL = (12, 17, 30, 238)     # nearly opaque: the 3-D scene must not show through text
PANEL_EDGE = (48, 64, 96)
TEXT = (222, 230, 244)
DIM = (140, 152, 176)
FAINT = (114, 128, 154)       # ~4.7:1 on PANEL, the WCAG minimum for small text
ACCENT = (96, 180, 255)
ACCENT_DARK = (34, 70, 120)
GOOD = (120, 230, 140)
WARN = (255, 196, 90)
BAD = (255, 110, 110)
FIELD = (22, 30, 50)
FIELD_FOCUS = (32, 46, 78)
SCROLL_THUMB = (92, 114, 152)
SUN = (255, 222, 120)
AXIS_X = (230, 90, 90)
AXIS_Y = (110, 220, 120)
AXIS_Z = (100, 150, 255)

S = 1.0                       # UI scale: screen pixels per design pixel (set_scale)


def set_scale(scale: float):
    """Use ``scale`` screen pixels per design pixel from now on."""
    global S
    S = float(scale)


def px(v: float) -> int:
    """Design pixels ``v`` in screen pixels (exactly ``v`` at scale 1)."""
    return int(round(v * S))


EVENT_COLORS = {
    "info": DIM, "maneuver": ACCENT, "eclipse": (170, 150, 255), "station": GOOD,
    "alert": BAD, "warn": WARN, "launch": (255, 170, 90),
}


FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SYSTEM_FONTS = {"sans": "segoeui", "mono": "consolas"}     # Windows' own; the layout's reference
BUNDLED_FONTS = {("sans", False): "DejaVuSans.ttf", ("sans", True): "DejaVuSans-Bold.ttf",
                 ("mono", False): "DejaVuSansMono.ttf", ("mono", True): "DejaVuSansMono.ttf"}


def load_font(kind: str, size: int, bold: bool = False) -> pygame.font.Font:
    """The ``kind`` ("sans" or "mono") font at ``size`` px: Windows' Segoe UI /
    Consolas where installed, else the DejaVu fonts in ``assets/fonts``, so every
    other system (and CI) draws the same, known text widths. Setting
    ``SATFLIGHT_FONTS=bundled`` forces DejaVu, to check the layout with it."""
    name = SYSTEM_FONTS[kind]
    if os.environ.get("SATFLIGHT_FONTS") != "bundled" and pygame.font.match_font(name):
        return pygame.font.SysFont(name, size, bold=bold)
    path = FONT_DIR / BUNDLED_FONTS[kind, bold]
    if path.exists():
        return pygame.font.Font(str(path), size)
    return pygame.font.SysFont("dejavusansmono,menlo,couriernew" if kind == "mono"
                               else "dejavusans,helvetica,arial", size, bold=bold)


class Fonts:
    """Lazily created fonts plus a cache of rendered text surfaces."""

    def __init__(self):
        self.small = load_font("mono", px(13))
        self.mono = load_font("mono", px(14))
        self.ui = load_font("sans", px(15))
        self.bold = load_font("sans", px(15), bold=True)
        self.title = load_font("sans", px(19), bold=True)
        self.big = load_font("sans", px(30), bold=True)
        self._cache: dict = {}

    def render(self, text: str, color=TEXT, font=None) -> pygame.Surface:
        """Rendered text surface, cached by (text, color, font)."""
        font = font or self.ui
        key = (text, color, id(font))
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 3000:
                self._cache.clear()
            surf = font.render(text, True, color)
            self._cache[key] = surf
        return surf

    def fit(self, text: str, max_w: int, font=None) -> str:
        """``text`` shortened with "..." so it is at most ``max_w`` px wide."""
        font = font or self.ui
        key = ("fit", text, max_w, id(font))
        out = self._cache.get(key)
        if out is None:
            out = text
            if font.size(text)[0] > max_w:
                while out and font.size(out + "...")[0] > max_w:
                    out = out[:-1]
                out = out.rstrip() + "..."
            self._cache[key] = out
        return out

    def draw(self, target, text, pos, color=TEXT, font=None, anchor="topleft"):
        """Blit ``text`` with its ``anchor`` point at ``pos``; returns the rect used."""
        surf = self.render(text, color, font)
        rect = surf.get_rect(**{anchor: pos})
        target.blit(surf, rect)
        return rect


def panel(surface: pygame.Surface, rect: pygame.Rect, alpha_color=PANEL, edge=PANEL_EDGE,
          radius: int = 8):
    """Translucent rounded panel (``radius`` in design px)."""
    radius = px(radius)
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, alpha_color, layer.get_rect(), border_radius=radius)
    surface.blit(layer, rect.topleft)
    if edge:
        pygame.draw.rect(surface, edge, rect, 1, border_radius=radius)


def scrollbar(surface: pygame.Surface, x: int, area: pygame.Rect, first: float, visible: float,
              total: float, min_len: int = 20):
    """Thin scroll indicator at ``x`` beside ``area``: ``visible`` of ``total`` units
    shown, starting at ``first`` (``min_len`` in design px)."""
    bar_h = max(px(min_len), int(area.h * visible / total))
    y = area.y + int((area.h - bar_h) * first / max(1, total - visible))
    pygame.draw.rect(surface, FIELD, (x, area.y, px(3), area.h), border_radius=px(2))
    pygame.draw.rect(surface, SCROLL_THUMB, (x, y, px(3), bar_h), border_radius=px(2))


def dim(color, k: float):
    """``color`` scaled by ``k`` (clamped to 0-255), alpha dropped."""
    return tuple(max(0, min(255, int(c * k))) for c in color[:3])


def mix(a, b, k: float):
    """Linear blend from color ``a`` (k = 0) to ``b`` (k = 1)."""
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))
