"""Two-line element sets.

When the optional ``sgp4`` package (the propagator skyfield uses) is
installed, TLEs are initialized through SGP4 for the proper mean-to-
osculating conversion; the TEME frame it returns is used as ECI (they differ
by well under a milliradian). Without it the TLE's mean motion is read as
the rate of the mean anomaly under J2 (true to first order in J2, as in
SGP4), which gives the mean semi-major axis; the J2 short-period term turns
that into the osculating one, and the analytic J2 propagator carries the
state from the TLE epoch, so the plane regresses as it should. Drag, J3+
and the higher-order SGP4 terms are left out: tens of km over days.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .analysis import j2_mean_propagate, j2_secular_rates, mean_to_osculating_a
from .constants import MU_EARTH
from .elements import coe2rv, mean_to_true
from .timeutil import UTC, ensure_utc, julian_date

try:  # optional dependency
    from sgp4.api import Satrec  # type: ignore
    HAVE_SGP4 = True
except ImportError:  # pragma: no cover - depends on environment
    Satrec = None
    HAVE_SGP4 = False


def checksum(line: str) -> int:
    """Modulo-10 checksum of a TLE line (digits count face value, '-' counts 1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def _implied_decimal(field: str) -> float:
    """TLE exponent notation such as ``-11606-4`` -> -0.11606e-4."""
    s = field.strip()
    if not s:
        return 0.0
    sign = -1.0 if s[0] == "-" else 1.0
    s = s.lstrip("+-")
    mant, exp = s[:-2], s[-2:]
    return sign * float("0." + mant.strip()) * 10.0 ** int(exp)


@dataclass
class TLE:
    """A parsed two-line element set (mean elements at ``epoch``)."""

    name: str
    line1: str
    line2: str
    satnum: int
    epoch: datetime
    bstar: float
    inclination: float     # rad
    raan: float            # rad
    eccentricity: float
    argp: float            # rad
    mean_anomaly: float    # rad
    mean_motion: float     # rad/s
    checksum_ok: bool

    @property
    def semi_major_axis(self) -> float:
        """Mean semi-major axis (km): the one whose two-body rate plus J2's
        secular drift of the mean anomaly is the TLE's mean motion."""
        n0, e, i = self.mean_motion, self.eccentricity, self.inclination
        a = (MU_EARTH / n0 ** 2) ** (1.0 / 3.0)
        for _ in range(6):                    # contracts by ~1e-3 per pass
            md = float(j2_secular_rates(a, e, i)[2])
            a = (MU_EARTH / (n0 - md) ** 2) ** (1.0 / 3.0)
        return a

    def state_at(self, when: datetime):
        """ECI (TEME) position and velocity at ``when``."""
        when = ensure_utc(when)
        if HAVE_SGP4:
            sat = Satrec.twoline2rv(self.line1, self.line2)
            jd = julian_date(when)
            jd_i = math.floor(jd - 0.5) + 0.5
            err, r, v = sat.sgp4(jd_i, jd - jd_i)
            if err == 0:
                return np.array(r), np.array(v)
        e, i = self.eccentricity, self.inclination
        nu = mean_to_true(self.mean_anomaly, e)
        a = float(mean_to_osculating_a(self.semi_major_axis, e, i, self.argp, nu))
        r, v = coe2rv(a, e, i, self.raan, self.argp, nu)
        dt = (when - self.epoch).total_seconds()
        if dt:
            r, v = j2_mean_propagate(np.asarray(r)[None], np.asarray(v)[None], dt)
            r, v = r[0], v[0]
        return np.asarray(r), np.asarray(v)


def split_tles(text: str) -> list[TLE]:
    """Every TLE in ``text`` (a Celestrak-style file): pairs of lines starting
    with "1 " and "2 ", each named by the line just before it if that is not
    part of a pair. Pairs that do not parse are skipped."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, prev, k = [], "", 0
    while k < len(lines):
        ln = lines[k]
        if ln.startswith("1 ") and k + 1 < len(lines) and lines[k + 1].startswith("2 "):
            with contextlib.suppress(ValueError):
                out.append(parse_tle(ln + "\n" + lines[k + 1], prev.lstrip("0 ").strip()))
            prev, k = "", k + 2
            continue
        prev, k = ln.strip(), k + 1
    return out


def parse_tle(text: str, name: str = "") -> TLE:
    """Parse two or three lines (optional title line first)."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 3:
        name = name or lines[0].lstrip("0 ").strip()
        lines = lines[1:]
    if len(lines) != 2 or not lines[0].startswith("1 ") or not lines[1].startswith("2 "):
        raise ValueError("expected TLE lines starting with '1 ' and '2 '")
    l1, l2 = lines
    if len(l1) < 64 or len(l2) < 63:
        raise ValueError("TLE lines are too short")
    ok = True
    for ln in (l1, l2):
        if len(ln) >= 69 and ln[68].isdigit():
            ok &= checksum(ln) == int(ln[68])
    yy = int(l1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    day = float(l1[20:32])
    epoch = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day - 1.0)
    return TLE(
        name=name or f"SAT {int(l1[2:7])}",
        line1=l1, line2=l2,
        satnum=int(l1[2:7]),
        epoch=epoch,
        bstar=_implied_decimal(l1[53:61]),
        inclination=math.radians(float(l2[8:16])),
        raan=math.radians(float(l2[17:25])),
        eccentricity=float("0." + l2[26:33].strip()),
        argp=math.radians(float(l2[34:42])),
        mean_anomaly=math.radians(float(l2[43:51])),
        mean_motion=float(l2[52:63]) * 2.0 * math.pi / 86400.0,
        checksum_ok=ok,
    )
