"""Time scales: Julian dates, sidereal time and epoch helpers.

UT1 is taken as UTC (|UT1-UTC| < 0.9 s, at most ~13 arcseconds of Earth
rotation) unless a :class:`Clock` is given ``dut1``: precise orbit files
(``poeorb``) carry it, and there 0.9 s is ~0.4 km of ground track.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import JD_J2000, SECONDS_PER_DAY

UTC = timezone.utc


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime (naive values are taken as UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_epoch(text: str) -> datetime:
    """Parse an ISO-8601 string such as ``2026-09-29T12:00:00Z``."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return ensure_utc(datetime.fromisoformat(text))


def format_epoch(dt: datetime) -> str:
    """ISO-8601 UTC string with a ``Z`` suffix, whole seconds."""
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def julian_date(dt: datetime) -> float:
    """Julian date of a datetime (Vallado Algorithm 14, valid 1900-2100)."""
    dt = ensure_utc(dt)
    y, m, d = dt.year, dt.month, dt.day
    frac = (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    jd = (367 * y - int(7 * (y + int((m + 9) / 12)) / 4) + int(275 * m / 9)
          + d + 1721013.5)
    return jd + frac


def datetime_from_jd(jd: float) -> datetime:
    """Inverse of :func:`julian_date`."""
    return datetime(2000, 1, 1, 12, tzinfo=UTC) + timedelta(days=jd - JD_J2000)


def centuries_since_j2000(jd):
    """Julian centuries of 36525 days since J2000.0 (scalar or array)."""
    return (np.asarray(jd, dtype=float) - JD_J2000) / 36525.0


def gmst(jd):
    """Greenwich mean sidereal time in radians (IAU-82, Vallado eq. 3-47).

    Accepts a scalar or an array of Julian dates (UT1).
    """
    t = centuries_since_j2000(jd)
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t - 6.2e-6 * t * t * t)
    theta = np.radians(np.mod(sec, SECONDS_PER_DAY) / 240.0)   # 240 s of time per degree
    if np.ndim(theta) == 0:
        return float(theta)
    return theta


class Clock:
    """Maps simulation seconds since the scenario epoch onto calendar time."""

    def __init__(self, epoch: datetime, dut1: float = 0.0):
        """``dut1`` is UT1 - UTC in s (|dut1| < 0.9), the Earth's rotation lag:
        0 takes UT1 as UTC; precise orbit files give the real value."""
        if not abs(dut1) < 0.9:
            raise ValueError(f"UT1-UTC = {dut1} s; it is always within 0.9 s")
        self.epoch = ensure_utc(epoch)
        self.jd0 = julian_date(self.epoch)
        self.dut1 = float(dut1)

    def jd(self, t):
        """Julian date at simulation time(s) ``t`` (s)."""
        if np.ndim(t):
            return self.jd0 + np.asarray(t, dtype=float) / SECONDS_PER_DAY
        return self.jd0 + float(t) / SECONDS_PER_DAY

    def gmst(self, t):
        """Greenwich mean sidereal time (rad) at simulation time(s) ``t``: the
        Earth's rotation, so taken at UT1 (UTC + ``dut1``)."""
        if self.dut1:
            return gmst(self.jd(t) + self.dut1 / SECONDS_PER_DAY)
        return gmst(self.jd(t))

    def datetime(self, t: float) -> datetime:
        """Calendar time (UTC) at simulation time ``t``."""
        return self.epoch + timedelta(seconds=float(t))


def format_duration(seconds: float) -> str:
    """``3d 04:05:06`` style elapsed-time string."""
    sign = "-" if seconds < 0 else ""
    days, s = divmod(abs(seconds), 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    days, h, m = int(days), int(h), int(m)
    if days:
        return f"{sign}{days}d {h:02d}:{m:02d}:{int(s):02d}"
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"


def format_period(seconds: float) -> str:
    """Orbital period in the most readable unit (min, h or d)."""
    if not math.isfinite(seconds):
        return "-"
    if seconds >= 86400 * 2:
        return f"{seconds / 86400:.3f} d"
    if seconds >= 7200:
        return f"{seconds / 3600:.3f} h"
    return f"{seconds / 60:.2f} min"
