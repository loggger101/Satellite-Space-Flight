"""Earth orientation from the IERS: polar motion (and UT1-UTC) by day.

The Earth's spin axis wanders over its crust by a few tenths of an arcsecond
(polar motion: about 15 m at Sentinel-1's height). The Sentinel-1 precise
orbits are given in the ITRF, fixed to the crust; the simulator's Earth turns
about the spin axis itself. Without the correction the ITRF's z axis is taken
as the spin axis, and the simulator's "inertial" frame wobbles once a day by
that tilt, which a fit sees as a daily cross-track error of ~10 m.

The table is the IERS ``finals2000A.all`` (Bulletin A, daily since 1973, then
a year of predictions), kept outside the repository under
``$SATFLIGHT_DATA/eop/`` (``tools/fetch_eop.py``). Only measured rows (flag
``I``) are used; a time needing predicted or missing rows is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)
FILE_NAME = "finals2000A.all"
URL = "https://datacenter.iers.org/data/9/finals2000A.all"
MJD_JD = 2400000.5


class EopError(ValueError):
    """An Earth-orientation table that cannot be used, or a time it does not cover."""


def table_path(root: Path | None = None) -> Path:
    """Where the IERS table lives: ``<data root>/eop/finals2000A.all``."""
    from .poeorb import data_root
    return (root or data_root()) / "eop" / FILE_NAME


@dataclass(frozen=True)
class Table:
    """Daily values at 0h UTC: MJD, pole x/y (rad), UT1-UTC (s), measured?"""
    mjd: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    dut1: np.ndarray
    measured: np.ndarray

    def pole(self, utc) -> tuple[np.ndarray, np.ndarray]:
        """Polar motion (x_p, y_p) in rad at datetime64 UTC times, linear between days."""
        from .poeorb import julian_dates
        utc = np.atleast_1d(np.asarray(utc, "datetime64[us]"))
        m = julian_dates(utc) - MJD_JD
        i = np.searchsorted(self.mjd, m, side="right") - 1
        before = i < 0
        after = (i >= len(self.mjd) - 1) & (m > self.mjd[-1])
        if before.any() or after.any():
            k = int(np.flatnonzero(before | after)[0])
            raise EopError(f"{str(utc[k])[:19]} is outside the Earth-orientation table "
                           f"(MJD {self.mjd[0]:.0f}..{self.mjd[-1]:.0f})")
        i = np.minimum(i, len(self.mjd) - 2)
        f = m - self.mjd[i]
        need_next = f > 0
        ok = self.measured[i] & (self.measured[i + 1] | ~need_next)
        if not ok.all():
            k = int(np.flatnonzero(~ok)[0])
            raise EopError(f"{str(utc[k])[:19]} needs predicted Earth-orientation values; "
                           "only measured ones are used (fetch a newer table)")
        return (self.xp[i] + f * (self.xp[i + 1] - self.xp[i]),
                self.yp[i] + f * (self.yp[i + 1] - self.yp[i]))


def _row(line: str, n: int):
    try:
        mjd = float(line[7:15])
        flag, ut1_flag = line[16], line[57]
        if flag not in "IP" or ut1_flag not in "IP":
            raise ValueError(f"flags {flag!r}/{ut1_flag!r}")
        return (mjd, float(line[18:27]), float(line[37:46]), float(line[58:68]),
                flag == "I" and ut1_flag == "I")
    except (ValueError, IndexError) as exc:
        raise EopError(f"line {n}: not a finals2000A row ({exc}): {line[:40]!r}") from None


def read(path: Path | None = None) -> Table:
    """Read a finals2000A table, strictly: every row up to the blank future
    must parse, and the days must follow one another."""
    path = Path(path) if path is not None else table_path()
    if not path.is_file():
        raise EopError(f"no Earth-orientation table at {path}: run python tools/fetch_eop.py")
    rows, blank_at = [], None
    for n, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if len(line) <= 16 or line[16] == " ":                  # the future: no values yet
            blank_at = blank_at or n
            continue
        if blank_at is not None:
            raise EopError(f"line {blank_at}: an empty row before line {n}'s values")
        rows.append(_row(line, n))
        if len(rows) > 1 and rows[-1][0] != rows[-2][0] + 1.0:
            raise EopError(f"line {n}: MJD {rows[-1][0]:.0f} does not follow "
                           f"{rows[-2][0]:.0f}")
    if len(rows) < 2:
        raise EopError(f"{path}: fewer than two days of values")
    mjd, xp, yp, dut1, measured = (np.array(c) for c in zip(*rows, strict=True))
    return Table(mjd, xp * ARCSEC, yp * ARCSEC, dut1, measured.astype(bool))


def itrf_to_tirs(xp, yp) -> np.ndarray:
    """(N, 3, 3) rotations taking ITRF vectors into the frame that turns about
    the spin axis: IERS's W = R2(x_p) R1(y_p) (GCRS = Q R W ITRS; s' ignored,
    it is under 0.1 mas)."""
    xp, yp = np.atleast_1d(np.asarray(xp, float)), np.atleast_1d(np.asarray(yp, float))
    cx, sx, cy, sy = np.cos(xp), np.sin(xp), np.cos(yp), np.sin(yp)
    z, o = np.zeros_like(xp), np.ones_like(xp)
    r2 = np.stack([np.stack([cx, z, -sx], -1), np.stack([z, o, z], -1),
                   np.stack([sx, z, cx], -1)], -2)                       # R2(x_p)
    r1 = np.stack([np.stack([o, z, z], -1), np.stack([z, cy, sy], -1),
                   np.stack([z, -sy, cy], -1)], -2)                      # R1(y_p)
    return r2 @ r1
