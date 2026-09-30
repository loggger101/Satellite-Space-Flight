"""Colours, fonts and a text-surface cache shared by every UI module."""

from __future__ import annotations

import pygame

BG = (4, 6, 14)
PANEL = (12, 17, 30, 215)
PANEL_EDGE = (48, 64, 96)
TEXT = (222, 230, 244)
DIM = (140, 152, 176)
FAINT = (86, 98, 122)
ACCENT = (96, 180, 255)
ACCENT_DARK = (34, 70, 120)
GOOD = (120, 230, 140)
WARN = (255, 196, 90)
BAD = (255, 110, 110)
FIELD = (22, 30, 50)
FIELD_FOCUS = (32, 46, 78)
SUN = (255, 222, 120)
MOON = (190, 190, 200)
AXIS_X = (230, 90, 90)
AXIS_Y = (110, 220, 120)
AXIS_Z = (100, 150, 255)

EVENT_COLORS = {
    "info": DIM, "maneuver": ACCENT, "eclipse": (170, 150, 255), "station": GOOD,
    "alert": BAD, "warn": WARN,
}


class Fonts:
    """Lazily created fonts plus a cache of rendered text surfaces."""

    def __init__(self, scale: float = 1.0):
        mono = "consolas,dejavusansmono,menlo,couriernew"
        sans = "segoeui,dejavusans,helvetica,arial"
        self.small = pygame.font.SysFont(mono, int(13 * scale))
        self.mono = pygame.font.SysFont(mono, int(14 * scale))
        self.ui = pygame.font.SysFont(sans, int(15 * scale))
        self.bold = pygame.font.SysFont(sans, int(15 * scale), bold=True)
        self.title = pygame.font.SysFont(sans, int(19 * scale), bold=True)
        self._cache: dict = {}

    def render(self, text: str, color=TEXT, font=None) -> pygame.Surface:
        font = font or self.ui
        key = (text, color, id(font))
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 3000:
                self._cache.clear()
            surf = font.render(text, True, color)
            self._cache[key] = surf
        return surf

    def draw(self, target, text, pos, color=TEXT, font=None, anchor="topleft"):
        surf = self.render(text, color, font)
        rect = surf.get_rect(**{anchor: pos})
        target.blit(surf, rect)
        return rect


def panel(surface: pygame.Surface, rect: pygame.Rect, alpha_color=PANEL, edge=PANEL_EDGE,
          radius: int = 8):
    """Translucent rounded panel."""
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, alpha_color, layer.get_rect(), border_radius=radius)
    surface.blit(layer, rect.topleft)
    if edge:
        pygame.draw.rect(surface, edge, rect, 1, border_radius=radius)


def dim(color, k: float):
    return tuple(max(0, min(255, int(c * k))) for c in color[:3])


def mix(a, b, k: float):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))
