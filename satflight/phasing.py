"""How satellites sharing an orbit sit along it and over the ground: the
Telemetry tab's PHASING section.

Everything comes from the ascending-node crossings the simulation notes
(``Satellite.nodes``), the way ``study.phasing`` and the track offsets read
ESA's precise orbits: a companion's phase from when it reaches the node, its
plane from the RAAN there (both satellites at the same point of their orbits,
so J2's short-period swing cancels), and its track from where on the ground
the node lands. A repeat orbit's nodes all land on a grid of tracks
(Sentinel-1: 175 tracks 229 km apart at the equator, 12 days to fly them all),
so a companion is on the same tracks when its nodes land on that grid too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import R_EARTH
from .elements import rv2coe
from .simulation import ACTIVE

MAX_DAYS = 30                  # longest repeat cycle looked for
# a cycle must close within this fraction of its own track spacing at the equator:
# one orbit's node spacing is all there is to go on, and for a real orbit (the
# precise states) it misses the exact repeat by a few 1e-4 deg, some 10 km over
# Sentinel-1's 175 orbits, against tracks 229 km apart. The next-best cycles miss
# by whole tracks (102 orbits in 7 days: 229 km).
CLOSURE_FRACTION = 0.1
CLOSURE_KM = 25.0              # and never by more than this: few orbits a day space
#                               their tracks thousands of km apart
MAX_COMPANIONS = 4
MIN_TILT_DEG = 1.0             # below it the equator crossings are not defined
EQUATOR_KM = 2 * math.pi * R_EARTH


@dataclass(frozen=True)
class Repeat:
    """A repeat ground track: ``orbits`` revolutions in ``days`` days."""

    orbits: int
    days: int
    spacing_km: float          # between neighbouring tracks at the equator
    closure_km: float          # how far the cycle misses closing


@dataclass(frozen=True)
class Companion:
    """Satellite ``index`` on the same orbit as the one asked about; None
    where its nodes do not tell yet."""

    index: int
    ahead_deg: float | None    # how far along the orbit it leads (- behind)
    plane_deg: float | None    # its RAAN minus this one's, at the same time (+ east)
    track_km: float | None     # east of this satellite's track grid, at the equator


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


def _period_and_shift(nodes):
    """(node-to-node period s, westward step of the node per orbit deg, RAAN
    rate deg/s) from the last two crossings; None if they do not join (one
    missed, or a burn in between that the step skipped)."""
    if len(nodes) < 2:
        return None
    (t1, lon1, raan1), (t2, lon2, raan2) = nodes[-2:]
    p = t2 - t1
    if p <= 0:
        return None
    return p, -_wrap(lon2 - lon1, 360.0), _wrap(raan2 - raan1, 360.0) / p


def repeat_cycle(sim, i) -> Repeat | None:
    """The repeat cycle satellite ``i``'s last orbit flies (the fewest days up to
    MAX_DAYS that close within CLOSURE_FRACTION of their track spacing and
    CLOSURE_KM), None if none does or it has not
    crossed the equator northward twice yet."""
    ps = _period_and_shift(sim.sats[i].nodes)
    if ps is None or not 0 < ps[1] < 360:
        return None
    return cycle_of(ps[1] / 360.0)


def cycle_of(x: float) -> Repeat | None:
    """The repeat cycle of an orbit under which the Earth turns ``x`` times
    (relative to the node) per revolution: the fewest days up to MAX_DAYS whose
    whole number of orbits closes within CLOSURE_FRACTION of its track spacing
    (and CLOSURE_KM)."""
    if not 0 < x < 1:
        return None
    for days in range(1, MAX_DAYS + 1):
        orbits = round(days / x)
        if orbits < 1:
            continue
        closure = abs(orbits * x - days) * EQUATOR_KM
        if closure < min(CLOSURE_FRACTION * EQUATOR_KM / orbits, CLOSURE_KM):
            return Repeat(orbits, days, EQUATOR_KM / orbits, closure)
    return None


def _same_orbit(el, i):
    """Which satellites fly satellite ``i``'s orbit, from the elements ``el``
    of all of them: near-circular, the same size, tilt and plane within a few
    degrees (Sentinel-1C's plane is 0.25 deg from 1A's). An orbit tilted less
    than MIN_TILT_DEG has no node to read phasing from: no companions."""
    if not el.i[i] > math.radians(MIN_TILT_DEG):
        return np.zeros(len(el.a), dtype=bool)
    return ((np.abs(el.a - el.a[i]) < 30.0) & (el.e < 0.05)
            & (np.abs(el.i - el.i[i]) < math.radians(0.5))
            & (np.abs(_wrap(el.raan - el.raan[i], 2 * math.pi)) < math.radians(5.0)))


def companions(sim, i) -> list[Companion]:
    """The other flying satellites on satellite ``i``'s orbit (near-circular,
    same size, tilt and plane within a few degrees), up to MAX_COMPANIONS."""
    sats = sim.sats
    if sats[i].status != ACTIVE:
        return []
    el = rv2coe(sim.y[:, :3], sim.y[:, 3:])
    same = _same_orbit(el, i)
    if not same[i]:
        return []
    ps = _period_and_shift(sats[i].nodes)
    rep = repeat_cycle(sim, i)
    out = []
    for j, s in enumerate(sats):
        if j == i or s.status != ACTIVE or not same[j]:
            continue
        ahead = plane = track = None
        if ps is not None and s.nodes:
            p, _, raan_rate = ps
            t_me, lon_me, raan_me = sats[i].nodes[-1]
            t_it, lon_it, raan_it = s.nodes[-1]
            ahead = _wrap(360.0 * (t_me - t_it) / p, 360.0)
            plane = _wrap(raan_it - raan_me - raan_rate * (t_it - t_me), 360.0)
            if rep is not None:
                grid = 360.0 / rep.orbits
                track = math.radians(_wrap(lon_it - lon_me, grid)) * R_EARTH
        out.append(Companion(j, ahead, plane, track))
        if len(out) == MAX_COMPANIONS:
            break
    return out


def rows(sim, i) -> list[tuple[str, str]]:
    """The PHASING rows for satellite ``i``: [] when nothing shares its orbit."""
    comps = companions(sim, i)
    if not comps:
        return []
    rep = repeat_cycle(sim, i)
    if rep is not None:
        out = [("Repeat cycle", f"{rep.orbits} orbits / {rep.days} d, tracks "
                                f"{rep.spacing_km:,.0f} km apart")]
    elif _period_and_shift(sim.sats[i].nodes) is None:
        out = [("Repeat cycle", "after two equator crossings")]
    else:
        out = [("Repeat cycle", f"none within {MAX_DAYS} d")]
    for c in comps:
        out.append(("Companion", sim.sats[c.index].name))
        if c.ahead_deg is None:
            out.append(("Phase angle", "at its next equator crossing"))
            continue
        side = "ahead" if c.ahead_deg >= 0 else "behind"
        out.append(("Phase angle", f"{abs(c.ahead_deg):.2f} deg {side}"))
        out.append(("Plane offset", f"{c.plane_deg:+.3f} deg "
                                    f"({'E' if c.plane_deg >= 0 else 'W'})"))
        if c.track_km is not None:
            side = "E" if c.track_km >= 0 else "W"
            out.append(("Track offset", f"{abs(c.track_km):,.1f} km {side} of these tracks"))
    return out
