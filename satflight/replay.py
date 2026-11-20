"""Replaying Sentinel-1's precise orbits in the simulator.

Two orbit types in a scenario use the downloaded files (``satflight.poeorb``):

``{"type": "replay", "mission": "S1A", "until": "2025-07-07"}``
    The satellite follows the recorded orbit from the scenario's epoch to the
    end of ``until`` (UTC). It is not integrated: its state is interpolated
    between the 10 s samples (cubic Hermite on position and velocity, under a
    millimetre). It cannot burn, and when the recording ends it stops
    (status ``ENDED``), saying so in the log.

``{"type": "precise", "mission": "S1A"}``
    The satellite starts from the recorded state at the epoch and is then
    flown by the scenario's force model like any other: the twin. With a
    replayed satellite of the same mission in the run, the simulation reports
    the twin's gap to it in the truth's radial / along-track / cross-track
    frame (``Simulation.gap_to_truth``).

Both take polar motion out with the IERS table (``satflight.eop``). Missing
files are an error naming the first missing day; nothing is bridged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

ENDED = "end of data"     # status of a replayed satellite past its recording
KINDS = ("replay", "precise")


class ReplayError(ValueError):
    """A recording asked for a time it does not hold, or a malformed request."""


@dataclass(frozen=True, eq=False)
class Ephemeris:
    """Recorded ECI states ``r`` (N, 3) km, ``v`` (N, 3) km/s at times ``t``
    (N,) s since the scenario epoch."""

    mission: str
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    until: date | None = None

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def state(self, t: float) -> np.ndarray:
        """(6,) state at ``t`` s, cubic Hermite between the samples around it."""
        if not self.start - 1e-9 <= t <= self.end + 1e-9:
            raise ReplayError(f"{self.mission}: t = {t:.1f} s is outside the recording "
                              f"({self.start:.0f}..{self.end:.0f} s)")
        k = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        t0, h = self.t[k], self.t[k + 1] - self.t[k]
        s = (t - t0) / h
        r0, r1, m0, m1 = self.r[k], self.r[k + 1], self.v[k] * h, self.v[k + 1] * h
        s2, s3 = s * s, s * s * s
        r = ((2 * s3 - 3 * s2 + 1) * r0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * r1
             + (s3 - s2) * m1)
        v = ((6 * s2 - 6 * s) * r0 + (3 * s2 - 4 * s + 1) * m0 + (-6 * s2 + 6 * s) * r1
             + (3 * s2 - 2 * s) * m1) / h
        return np.concatenate([r, v])


def check(orbit: dict, where: str):
    """Raise ValueError unless a replay / precise orbit names a known mission
    (and, for a replay, a valid ``until`` date)."""
    from .poeorb import MISSIONS
    if orbit.get("mission") not in MISSIONS:
        raise ValueError(f"{where} mission: {orbit.get('mission')!r} is not a Sentinel-1 unit "
                         f"({', '.join(MISSIONS)})")
    if orbit.get("type") == "replay":
        until_date(orbit, where)


def until_date(orbit: dict, where: str = "orbit") -> date:
    try:
        return date.fromisoformat(str(orbit["until"]))
    except ValueError:
        raise ValueError(f"{where} until: expected a date as YYYY-MM-DD, got "
                         f"{orbit['until']!r}") from None


def for_orbit(orbit: dict, epoch: datetime) -> Ephemeris:
    """The recording a replay / precise orbit spec refers to, timed from ``epoch``."""
    epoch = epoch.astimezone(timezone.utc)
    last = until_date(orbit) if orbit.get("type") == "replay" else epoch.date()
    if last < epoch.date():
        raise ReplayError(f"replay until {last} ends before the scenario starts "
                          f"({epoch:%Y-%m-%d})")
    from .poeorb import data_root
    return _load(str(data_root()), orbit["mission"], epoch, last)


@lru_cache(maxsize=8)
def _load(root: str, mission: str, epoch: datetime, last: date) -> Ephemeris:
    from pathlib import Path

    from . import eop, poeorb
    t0 = np.datetime64(epoch.replace(tzinfo=None), "us")
    first = epoch.date()
    arc = poeorb.load_arc(mission, first, last, Path(root))
    if arc.utc[0] > t0:
        # a day's own samples start at 00:00:02: the moments before come from the day before
        arc = poeorb.load_arc(mission, first - timedelta(days=1), last, Path(root))
    r, v = arc.eci(_table(str(eop.table_path(Path(root)))))
    return Ephemeris(mission, arc.seconds(t0), r, v, last)


@lru_cache(maxsize=2)
def _table(path: str):
    from pathlib import Path

    from . import eop
    return eop.read(Path(path))


def clear_cache():
    """Forget loaded recordings (after new downloads, or between tests)."""
    _load.cache_clear()
    _table.cache_clear()
