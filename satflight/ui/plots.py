"""Telemetry plot of the selected satellite's recent history."""

from __future__ import annotations

import numpy as np
import pygame

from ..constants import R_EARTH
from ..elements import rv2coe
from ..forces import conservative_energy
from ..frames import ecef_to_geodetic, eci_to_ecef
from . import theme
from .theme import px


def _el(s):
    """Elements of a (L, 6) state history."""
    return rv2coe(s[:, :3], s[:, 3:])


def _altitude(sim, t, s):
    return ecef_to_geodetic(eci_to_ecef(s[:, :3], sim.clock.gmst(t)))[2]


def _energy_drift(sim, t, s):
    e = conservative_energy(s[:, :3], s[:, 3:], sim.forces)
    return (e - e[0]) / abs(e[0])


# (name, unit, fn(sim, times, states) -> values); angles are unwrapped so drifts plot smoothly
METRICS = [
    ("Altitude", "km", _altitude),
    ("Speed", "km/s", lambda sim, t, s: np.linalg.norm(s[:, 3:], axis=1)),
    ("Semi-major axis", "km", lambda sim, t, s: _el(s).a),
    ("Eccentricity", "", lambda sim, t, s: _el(s).e),
    ("Inclination", "deg", lambda sim, t, s: np.degrees(_el(s).i)),
    ("RAAN", "deg", lambda sim, t, s: np.degrees(np.unwrap(_el(s).raan))),
    ("Arg. of perigee", "deg", lambda sim, t, s: np.degrees(np.unwrap(_el(s).argp))),
    ("Perigee altitude", "km", lambda sim, t, s: _el(s).rp - R_EARTH),
    ("Energy drift (rel.)", "", _energy_drift),
]


class PlotView:
    """One ``METRICS`` series of the selected satellite over the history buffer;
    clicking cycles the metric."""

    def __init__(self):
        self.metric = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.title_rect = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        """Left click / wheel down: next metric; right click / wheel up: previous."""
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            if ev.button in (1, 5):
                self.metric = (self.metric + 1) % len(METRICS)
            elif ev.button in (3, 4):
                self.metric = (self.metric - 1) % len(METRICS)
            return True
        return False

    def draw(self, surf, rect: pygame.Rect, app):
        """The selected satellite's metric over the history buffer up to now, with a value grid."""
        sim, fonts = app.sim, app.fonts
        self.rect = rect
        theme.panel(surf, rect)
        name, unit, fn = METRICS[self.metric]
        i = app.selected
        title = f"{name}" + (f" [{unit}]" if unit else "") + "   (click to change)"
        fonts.draw(surf, title, (rect.x + px(10), rect.y + px(5)), theme.ACCENT, fonts.small)
        if not 0 <= i < sim.n:
            fonts.draw(surf, "select a satellite", rect.center, theme.FAINT, fonts.small, "center")
            return
        t, s = sim.history.series(i)
        if len(t) < 3:
            fonts.draw(surf, "collecting samples...", rect.center, theme.FAINT, fonts.small,
                       "center")
            return
        t = np.append(t, sim.t)
        s = np.vstack([s, sim.y[i]])
        try:
            y = np.asarray(fn(sim, t, s), dtype=float)
        except Exception:        # degenerate states (e.g. at impact): skip the frame
            return
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return
        t, y = t[ok], y[ok]
        pr = pygame.Rect(rect.x + px(70), rect.y + px(24), rect.w - px(84), rect.h - px(44))
        lo, hi = float(y.min()), float(y.max())
        if hi - lo < 1e-12 * max(1.0, abs(hi)):
            lo, hi = lo - 1e-9 - abs(lo) * 1e-9, hi + 1e-9 + abs(hi) * 1e-9
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        for k in range(5):
            yy = pr.bottom - k * pr.h / 4
            pygame.draw.line(surf, (30, 40, 62), (pr.x, yy), (pr.right, yy))
            val = lo + (hi - lo) * k / 4
            txt = f"{val:.4g}" if abs(val) < 1e5 else f"{val:.3e}"
            fonts.draw(surf, txt, (pr.x - px(6), yy), theme.FAINT, fonts.small, "midright")
        t0, t1 = t[0], t[-1]
        span = max(t1 - t0, 1e-9)
        xs = pr.x + (t - t0) / span * pr.w
        ys = pr.bottom - (y - lo) / (hi - lo) * pr.h
        pygame.draw.aalines(surf, sim.sats[i].color, False, np.stack([xs, ys], 1).tolist())
        unit_t, div = ("h", 3600.0) if span > 7200 else ("min", 60.0)
        fonts.draw(surf, f"-{span / div:.1f} {unit_t}", (pr.x, pr.bottom + px(3)), theme.FAINT,
                   fonts.small)
        fonts.draw(surf, "now", (pr.right, pr.bottom + px(3)), theme.FAINT, fonts.small,
                   "topright")
        fonts.draw(surf, f"{y[-1]:.6g}" + (f" {unit}" if unit else ""), (pr.right, pr.y),
                   theme.TEXT, fonts.small, "topright")
