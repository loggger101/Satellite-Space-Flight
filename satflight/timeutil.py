"""Time scales: Julian dates, sidereal time and epoch helpers.

UT1 is approximated by UTC (|UT1-UTC| < 0.9 s), which shifts the Earth's
rotation angle by at most ~13 arcseconds - far below what the display or the
low-precision ephemerides resolve.
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
    return datetime(2000, 1, 1, 12, tzinfo=UTC) + timedelta(days=jd - JD_J2000)


def centuries_since_j2000(jd):
    return (np.asarray(jd, dtype=float) - JD_J2000) / 36525.0


def gmst(jd):
    """Greenwich mean sidereal time in radians (IAU-82, Vallado eq. 3-47).

    Accepts a scalar or an array of Julian dates (UT1).
    """
    t = centuries_since_j2000(jd)
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t - 6.2e-6 * t * t * t)
    theta = np.mod(sec, SECONDS_PER_DAY) / 240.0          # degrees
    theta = np.radians(theta)
    if np.ndim(theta) == 0:
        return float(theta)
    return theta


class Clock:
    """Maps simulation seconds since the scenario epoch onto calendar time."""

    def __init__(self, epoch: datetime):
        self.epoch = ensure_utc(epoch)
        self.jd0 = julian_date(self.epoch)

    def jd(self, t):
        if np.ndim(t):
            return self.jd0 + np.asarray(t, dtype=float) / SECONDS_PER_DAY
        return self.jd0 + float(t) / SECONDS_PER_DAY

    def gmst(self, t):
        return gmst(self.jd(t))

    def datetime(self, t: float) -> datetime:
        return self.epoch + timedelta(seconds=float(t))


def format_duration(seconds: float) -> str:
    """``3d 04:05:06`` style elapsed-time string."""
    sign = "-" if seconds < 0 else ""
    s = abs(seconds)
    days = int(s // 86400)
    s -= days * 86400
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    if days:
        return f"{sign}{days}d {h:02d}:{m:02d}:{int(s):02d}"
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"


def format_period(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "-"
    if seconds >= 86400 * 2:
        return f"{seconds / 86400:.3f} d"
    if seconds >= 7200:
        return f"{seconds / 3600:.3f} h"
    return f"{seconds / 60:.2f} min"
