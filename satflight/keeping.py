"""Ground-track keeping: the small along-track burns that hold a repeat-orbit
satellite on its reference tracks, as ESA holds Sentinel-1 (the precise
orbits show one every 3.5-4 days, a few mm/s each: docs/analysis/sentinel1).

Drag lowers the orbit, the period shortens, the Earth turns less under each
revolution and the equator crossings creep east, faster and faster: the
offset from the reference track follows a parabola. At every ascending node
the keeper measures that offset against the repeat cycle's grid (fixed at the
first crossing after ``start``) and fits a parabola to the crossings since
its last burn. When the fit says the offset will leave the band within half a
day, it raises (or lowers) the orbit half a revolution later, by just enough
to turn the drift around so that the parabola peaks at the band's far edge
and drag brings it back: the classic ground-track control. Everything the
keeper decides comes from the crossings and the satellite's own state, so
the MATLAB export flies the same rule (satflight/matlab_runtime.m).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .constants import J2, MU_EARTH, OMEGA_EARTH, R_EARTH

KM_PER_DEG = R_EARTH * math.pi / 180.0     # along the equator
FIT_DAYS = 1.0                 # crossings the fit needs since the last burn
LOOKAHEAD_DAYS = 0.5           # burn when the fit leaves the band this soon
SLOW_DAYS = 2.0                # without drag to turn it back: cross the band in this long


@dataclass
class TrackKeeping:
    """Keep satellite ``sat`` within ``band_km`` east or west (at the equator) of
    the tracks of an ``orbits``-in-``days`` repeat cycle, from the first
    ascending node after ``start`` (s after the scenario epoch), with burns of
    specific impulse ``isp`` (s)."""

    sat: str
    band_km: float = 0.1
    orbits: int = 175
    days: int = 12
    start: float = 0.0
    isp: float = 220.0

    def check(self):
        """ValueError naming the first setting that cannot work."""
        if not self.band_km > 0:
            raise ValueError(f"track keeping for {self.sat}: band_km must be positive")
        if not 0 < self.days < self.orbits:
            raise ValueError(f"track keeping for {self.sat}: a repeat cycle needs "
                             "0 < days < orbits")
        if not self.isp > 0:
            raise ValueError(f"track keeping for {self.sat}: isp must be positive")
        if not self.start >= 0:
            raise ValueError(f"track keeping for {self.sat}: start must not be negative")
        return self

    def to_dict(self):
        """Plain dict for JSON."""
        return asdict(self)


def drift_per_km(r, v) -> float:
    """How a change of 1 km in the semi-major axis turns the crossings' east-west
    drift (km/day; negative: a higher orbit drifts west), for the orbit (r, v):
    the period's sensitivity 3P/2a times the Earth's turn relative to the
    orbit's node, over a day of orbits."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rm, v2 = float(np.linalg.norm(r)), float(v @ v)
    a = 1.0 / (2.0 / rm - v2 / MU_EARTH)
    h = np.cross(r, v)
    p = float(h @ h) / MU_EARTH
    n = math.sqrt(MU_EARTH / a ** 3)
    raan_dot = -1.5 * n * J2 * (R_EARTH / p) ** 2 * h[2] / math.sqrt(float(h @ h))
    return -1.5 * (OMEGA_EARTH - raan_dot) * R_EARTH * 86400.0 / a


class TrackKeeper:
    """One satellite's keeper: fed its ascending-node crossings, it answers
    with the burns to make. ``offsets`` holds (t s, offset km, + east) for every
    crossing from the reference on; ``burns`` the (t s, along-track km/s) it
    asked for."""

    def __init__(self, spec: TrackKeeping):
        self.spec = spec.check()
        self.ref = None                  # longitude (deg) of the reference crossing
        self.count = 0                   # revolutions since the reference
        self.last = None                 # time of the last crossing noted
        self.since = []                  # (t, offset) since the last burn
        self.wait_until = -math.inf      # a burn is scheduled: decide nothing before it
        self.offsets = []
        self.burns = []

    def drift_per_km(self, r, v) -> float:
        return drift_per_km(r, v)

    def crossing(self, t: float, lon_deg: float, r, v):
        """Note an ascending node at ``t`` (s) and longitude ``lon_deg``, the
        satellite's state about then ``r``, ``v`` (km, km/s). Returns
        (burn time s, along-track delta-v km/s) or None."""
        s = self.spec
        if t < s.start:
            return None
        rm, v2 = float(np.linalg.norm(r)), float(np.dot(v, v))
        a = 1.0 / (2.0 / rm - v2 / MU_EARTH)
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
        if self.ref is None:
            self.ref, self.count = lon_deg, 0
            off = 0.0
        else:
            # revolutions since the last crossing noted, from the time between them:
            # a crossing the run did not see does not shift the grid a whole track
            self.count += max(1, round((t - self.last) / period))
            shift = 360.0 * s.days / s.orbits * self.count
            off = math.remainder(lon_deg - self.ref + shift, 360.0) * KM_PER_DEG
        self.last = t
        self.offsets.append((t, off))
        if t < self.wait_until:
            return None
        self.since.append((t, off))
        t0 = self.since[0][0]
        if (t - t0) / 86400.0 < FIT_DAYS:
            return None
        tau = np.array([(x - t0) / 86400.0 for x, _ in self.since])
        e = np.array([y for _, y in self.since])
        c2, c1, c0 = np.polyfit(tau, e, 2)
        now = tau[-1]
        ahead = now + LOOKAHEAD_DAYS
        e_now = c0 + c1 * now + c2 * now * now
        e_ahead = c0 + c1 * ahead + c2 * ahead * ahead
        rate = c1 + 2.0 * c2 * now
        if abs(e_ahead) < s.band_km:
            return None
        side = 1.0 if e_ahead > 0 else -1.0
        back = 2.0 * c2 * side               # drag's pull back toward this edge
        reach = max(e_now * side + s.band_km, 0.0)
        if back > 0:
            # out to the far edge and back: the parabola's apex there
            target = -side * math.sqrt(2.0 * back * reach)
        else:
            target = -side * s.band_km / SLOW_DAYS
        da = (target - rate) / drift_per_km(r, v)
        dv = math.sqrt(MU_EARTH / a) * da / (2.0 * a)
        t_burn = t + period / 2.0                              # half an orbit on
        self.wait_until = t_burn
        self.since = []
        self.burns.append((t_burn, dv))
        return t_burn, dv
