"""The 3-D preview of a maneuver being planned: what the burns will do,
drawn in the scene while the Plan maneuver dialog is open.

While the form changes, the planner's two-body estimate is flown at once
(Kepler, a few ms). Once the form has rested ``DEBOUNCE`` s, the aimed plan
(the run's own forces, what Schedule will really fly) is worked out on a
background thread from a snapshot of the run and replaces it. An aimed plan
older than ``STALE`` s of simulated time is asked for again, so a running
simulation keeps the preview current.
"""

from __future__ import annotations

import math
import time

import pygame

from ..planner import ESTIMATE
from ..preview import fly, orbit_summary, snapshot
from ..timeutil import format_duration, format_period
from . import theme, tips
from .background import Latest
from .theme import px


class PlanPreview:
    """Owned by one dialog; shown while that dialog is open."""

    DEBOUNCE = 0.4            # s of real time the form must rest before aiming
    STALE = 60.0              # s of simulated time an aimed plan stays on show
    REFRESH = 0.25            # s of real time between estimates while the run goes on

    def __init__(self, app, owner):
        self.app = app
        self.owner = owner
        self.key = None               # the form the preview is for
        self.estimate = None          # FlownPlan, two-body
        self.aimed = None             # FlownPlan with the run's forces, for ``aimed_key``
        self.aimed_key = None
        self.error = ""
        self.color = (255, 255, 255)
        self._plan_fn = None          # (sim, aim) -> (sat, burns)
        self._worker = Latest()
        self._pending = None
        self._rested_at = None
        self._estimated_at = None     # (real time, sim time) of the estimate

    @property
    def active(self) -> bool:
        """True while the owning dialog is open."""
        return self.owner in self.app.dialogs

    def show(self, key, plan_fn, color):
        """The form now reads ``key``; ``plan_fn(sim, aim)`` returns (sat name,
        burns, the planner's summary) for it or raises ValueError. Flies the
        estimate now."""
        self.color = color
        if key != self.key:
            self.key, self._plan_fn = key, plan_fn
            self._rested_at = time.monotonic()
        self._estimate()

    def clear(self, error: str = ""):
        """Nothing to show (the form does not make a plan): say why."""
        self.key = self._plan_fn = self.estimate = None
        self.error = error

    def _estimate(self):
        sim = self.app.sim
        try:
            sat, burns, note = self._plan_fn(sim, False)
            self.estimate = fly(sim, sat, burns, aim=False)
            self.estimate.note = note
            self.error = ""
        except ValueError as exc:
            self.estimate, self.error = None, str(exc)
        self._estimated_at = (time.monotonic(), sim.t)

    def tick(self):
        """Once a frame: collect an aimed plan, ask for one when due, and keep
        the estimate up with a running simulation."""
        done = self._worker.poll()
        if done is not None and done[0] == self._pending:
            key, result, error = done
            self._pending = None
            if error is not None and not isinstance(error, ValueError):
                raise error                   # a bug in a planner, not bad input
            if error is None and key[0] == self.key:
                self.aimed, self.aimed_key = result, key[0]
            elif error is not None and key[0] == self.key:
                self.aimed, self.aimed_key, self.error = None, None, str(error)
        if self._plan_fn is None:
            return
        sim, now = self.app.sim, time.monotonic()
        if (self._estimated_at is not None and sim.t != self._estimated_at[1]
                and now - self._estimated_at[0] > self.REFRESH):
            self._estimate()
        fresh = (self.aimed is not None and self.aimed_key == self.key
                 and abs(sim.t - self.aimed.t0) <= self.STALE)
        if (not fresh and self._pending is None and self.estimate is not None
                and self._rested_at is not None and now - self._rested_at > self.DEBOUNCE):
            snap, plan_fn = snapshot(sim), self._plan_fn
            self._pending = (self.key, snap.t)

            def job():
                sat, burns, note = plan_fn(snap, True)
                plan = fly(snap, sat, burns, aim=True)
                plan.note = note
                return plan

            self._worker.submit(self._pending, job)

    def plan(self):
        """The FlownPlan to draw: the aimed one while it is current, else the
        estimate (None when the form makes no plan)."""
        if (self.aimed is not None and self.aimed_key == self.key and self.key is not None
                and abs(self.app.sim.t - self.aimed.t0) <= self.STALE):
            return self.aimed
        return self.estimate

    @property
    def aiming(self) -> bool:
        """True while the aimed plan is being worked out."""
        return self._pending is not None


# --- the plan card: what the burns do, in the view beside the drawn plan ------------------

CARD_W = 520               # design px
CARD_PAD = 10


def _orbit_text(y) -> str:
    """'185 x 35,786 km, 28.50 deg, 10.5 h' for the orbit through state ``y``."""
    hp, ha, inc, period = orbit_summary(y)
    size = (f"{hp:,.0f} x {ha:,.0f} km" if math.isfinite(ha)
            else f"perigee {hp:,.0f} km, escaping")
    return f"{size}, {inc:.2f} deg, {format_period(period)}"


def _reasons(note: str) -> list[str]:
    """The planner summary's explanations: every clause after the first (the
    first repeats the numbers the table shows)."""
    note = note.removeprefix(ESTIMATE)
    parts = [p.strip() for p in note.split("; ") if p.strip()]
    if len(parts) == 1 and not any(c.isdigit() for c in parts[0]):
        return parts                          # a summary that is only words
    return [p[0].upper() + p[1:] for p in parts[1:]]


def card_lines(app, preview) -> list[tuple[str, tuple, str]]:
    """The card as (text, colour, hover tip) lines; empty when there is no plan
    and nothing to say."""
    plan = preview.plan()
    if plan is None:
        if not preview.error:
            return []
        return [("CANNOT PLAN THIS MANEUVER", theme.BAD,
                 "The form asks for something that cannot be flown from here."),
                (preview.error, theme.TEXT, preview.error)]
    sim = app.sim
    if plan.aimed:
        mode, mode_tip = "aimed with the run's forces", (
            "Flown ahead with the run's own propagator and forces (J2, drag...): what "
            "Schedule will really do.")
    else:
        mode = "two-body estimate" + (", aiming..." if preview.aiming else "")
        mode_tip = ("The exact two-body ellipse, worked out at once while you type. The plan "
                    "aimed with the run's forces replaces it when the form rests.")
    out = [(f"PLANNED: {plan.sat} - {mode}", theme.ACCENT, mode_tip),
           (f"{'#':<2}{'when':>12} {'dV m/s':>7} {'speed':>7} {'tilt':>7}  where", theme.FAINT,
            "Each burn: how long until it fires, its delta-v, how much of it changes the "
            "speed (+ faster, - slower) and how far it turns the orbit's plane, and where on "
            "the orbit it fires. Speed and turn share one delta-v: |dV| is not their sum.")]
    for k, b in enumerate(plan.burns, 1):
        when = b.t - sim.t
        when_txt = "now" if abs(when) < 1 else format_duration(when)
        out.append((f"{k:<2}{when_txt:>12} {b.size * 1000:>7,.0f} {b.speed_change * 1000:>+7,.0f}"
                    f" {b.tilt:>6.2f}d  {b.place}", theme.TEXT,
                    f"{b.label}." + (f" It is {b.note}." if b.note else "")))
        if k < len(plan.burns):
            coast = plan.burns[k].t - b.t
            out.append((f"{'':2}{'coast':>12} {format_duration(coast)} on the transfer orbit "
                        "(dashed)", theme.DIM,
                        "The time between the two burns, spent on the transfer orbit drawn "
                        "dashed in the view."))
    used = plan.mass0 - plan.mass1
    dry = sim.sats[sim.require(plan.sat)].dry_mass
    isp = sorted({m.isp for m in plan.maneuvers})
    isp_txt = "/".join(f"{x:g}" for x in isp)
    short = plan.mass1 < dry
    out.append((f"Total {plan.total_dv * 1000:,.0f} m/s, {used:,.0f} kg of propellant at Isp "
                f"{isp_txt} s" + (" - MORE THAN IT CARRIES" if short else ""),
                theme.BAD if short else theme.TEXT,
                "The sum of the burns' delta-v and the propellant the rocket equation says "
                "they burn (m_after = m e^(-dV / (Isp g0))). " +
                (f"It carries only {plan.mass0 - dry:,.0f} kg: the burns would stop short."
                 if short else f"It carries {plan.mass0 - dry:,.0f} kg.")))
    out.append((f"From  {_orbit_text(plan.start)}", theme.DIM,
                "The orbit now: perigee x apogee altitude, inclination, period (two-body, "
                "osculating)."))
    out.append((f"To    {_orbit_text(plan.final)}", theme.GOOD,
                "The orbit after the last burn (drawn faded in the view): perigee x apogee "
                "altitude, inclination, period (two-body, osculating; J2 makes these swing "
                "a little around the orbit)."))
    for reason in _reasons(plan.note):
        out.append((reason, theme.WARN if reason.startswith("WARNING") else theme.DIM,
                    "Why the planner chose this."))
    return out


def card(app, preview):
    """(rect, lines) of the plan card at the bottom left of the view, or None."""
    if preview is None or not preview.active:
        return None
    lines = card_lines(app, preview)
    if not lines:
        return None
    fonts, view = app.fonts, app.view_rect()
    width = min(px(CARD_W), view.w - px(16))
    wrapped = []
    for text, col, tip in lines:
        for part in tips.wrap(text, fonts.small, width - 2 * px(CARD_PAD)):
            wrapped.append((part, col, tip))
    line_h = fonts.small.get_linesize()
    h = len(wrapped) * line_h + 2 * px(CARD_PAD)
    if h > view.h - px(16) or width < px(200):
        return None                            # no room: the dialog line still says it
    rect = pygame.Rect(view.x + px(8), view.bottom - h - px(8), width, h)
    return rect, wrapped


def draw_card(surf, app, rect, lines):
    """Draw the card made by :func:`card`, with a hover tip per line."""
    fonts = app.fonts
    theme.panel(surf, rect, (10, 15, 28, 235), theme.ACCENT_DARK, 6)
    tips.block(rect)
    line_h = fonts.small.get_linesize()
    y = rect.y + px(CARD_PAD)
    for text, col, tip in lines:
        fonts.draw(surf, text, (rect.x + px(CARD_PAD), y), col, fonts.small)
        tips.add(pygame.Rect(rect.x, y, rect.w, line_h), tip)
        y += line_h
