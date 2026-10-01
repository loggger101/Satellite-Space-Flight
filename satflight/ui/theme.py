"""Colors, fonts, the UI scale and a text-surface cache shared by every UI module.

Sizes in the UI code are written in design pixels, the layout at 100 % display
scaling. :func:`px` turns them into screen pixels: the window has the screen's
real pixels (the process is DPI-aware), so at 125 % scaling every size is
drawn 1.25 times larger and the UI stays the size the user chose, only sharper."""

from __future__ import annotations

import os
from collections import OrderedDict
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


class LRU(OrderedDict):
    """A dict that keeps at most ``size`` entries, dropping the least recently
    used one when full (instead of emptying itself and re-rendering everything
    in one frame)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def get(self, key, default=None):
        """The value for ``key``, now the most recently used."""
        try:
            self.move_to_end(key)
        except KeyError:
            return default
        return self[key]

    def put(self, key, value):
        """Store ``value``; returns it."""
        self[key] = value
        if len(self) > self.size:
            self.popitem(last=False)
        return value


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
        self._cache = LRU(3000)

    def render(self, text: str, color=TEXT, font=None) -> pygame.Surface:
        """Rendered text surface, cached by (text, color, font)."""
        font = font or self.ui
        key = (text, color, id(font))
        surf = self._cache.get(key)
        if surf is None:
            surf = self._cache.put(key, font.render(text, True, color))
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
            self._cache.put(key, out)
        return out

    def draw(self, target, text, pos, color=TEXT, font=None, anchor="topleft"):
        """Blit ``text`` with its ``anchor`` point at ``pos``; returns the rect used."""
        surf = self.render(text, color, font)
        rect = surf.get_rect(**{anchor: pos})
        target.blit(surf, rect)
        return rect


# panel backgrounds by (size, color, radius); a resized window brings new sizes,
# so the cache stays small (a full-height side panel is ~1.5 MB at 1920x1200)
_PANELS = LRU(32)
_VEILS = LRU(2)              # window-sized shades: ~9 MB each at 1920x1200
_SCRATCH: list = [None]


def veil(size, rgba) -> pygame.Surface:
    """A surface of ``size`` filled with the translucent ``rgba`` (a dialog's
    shade over the window), made once per size and color."""
    key = (tuple(size), tuple(rgba))
    out = _VEILS.get(key)
    if out is None:
        out = _VEILS.put(key, pygame.Surface(size, pygame.SRCALPHA))
        out.fill(rgba)
    return out


def scratch(size) -> pygame.Surface:
    """A transparent per-pixel-alpha surface of ``size`` to draw a layer on and
    blit straight away. It is a view of one shared surface that grows as
    needed, cleared on every call: no allocation per translucent shape. The
    previous scratch layer is overwritten, so blit it before asking again."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    big = _SCRATCH[0]
    if big is None or big.get_width() < w or big.get_height() < h:
        bw = w if big is None else max(w, big.get_width())
        bh = h if big is None else max(h, big.get_height())
        big = _SCRATCH[0] = pygame.Surface((bw, bh), pygame.SRCALPHA)
    layer = big.subsurface((0, 0, w, h))
    layer.fill((0, 0, 0, 0))
    return layer


def panel(surface: pygame.Surface, rect: pygame.Rect, alpha_color=PANEL, edge=PANEL_EDGE,
          radius: int = 8):
    """Translucent rounded panel (``radius`` in design px)."""
    radius = px(radius)
    key = (rect.size, tuple(alpha_color), radius)
    layer = _PANELS.get(key)
    if layer is None:
        layer = _PANELS.put(key, pygame.Surface(rect.size, pygame.SRCALPHA))
        pygame.draw.rect(layer, alpha_color, layer.get_rect(), border_radius=radius)
    surface.blit(layer, rect.topleft)
    if edge:
        pygame.draw.rect(surface, edge, rect, 1, border_radius=radius)


def app_icon(size: int = 64) -> pygame.Surface:
    """The window icon: the Earth crossed by an orbit with a satellite on it."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c, k = size // 2, size / 64.0
    pygame.draw.circle(s, (28, 84, 160), (c, c), round(19 * k))
    pygame.draw.circle(s, (70, 150, 105), (c - round(6 * k), c - round(5 * k)), round(8 * k))
    pygame.draw.circle(s, (60, 135, 95), (c + round(8 * k), c + round(7 * k)), round(5 * k))
    orbit = pygame.Rect(round(3 * k), c - round(11 * k), size - round(6 * k), round(22 * k))
    pygame.draw.ellipse(s, ACCENT, orbit, max(2, round(3 * k)))
    pygame.draw.circle(s, (255, 196, 64), (orbit.right - round(9 * k), c + round(8 * k)),
                       round(6 * k))
    return s


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
