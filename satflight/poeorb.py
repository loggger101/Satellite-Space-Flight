"""ESA precise orbit files (POEORB) for Sentinel-1: where they live, which file
covers which day, and reading them.

The files come from <https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/>,
one per satellite per day (``tools/fetch_poeorb.py`` downloads them). They are
kept outside the repository, under ``$SATFLIGHT_DATA/poeorb/<S1A|S1C>/``
(default ``~/satflight-data``), because a year of them is about half a GB.

A file named ``S1A_OPER_AUX_POEORB_OPOD_<created>_V<start>_<stop>.EOF.zip``
covers 26 h, from 22:59:42 UTC the day before its *nominal day* to 00:59:42
the day after; ESA files it under the month of ``<start>``. Contains modified
Copernicus Sentinel data.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BASE_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
MISSIONS = ("S1A", "S1B", "S1C", "S1D")

NAME_RE = re.compile(r"^(?P<mission>S1[A-D])_OPER_AUX_POEORB_OPOD_(?P<created>\d{8}T\d{6})"
                     r"_V(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.EOF(?:\.zip)?$")


def data_root() -> Path:
    """Where downloaded data lives: ``$SATFLIGHT_DATA`` or ``~/satflight-data``."""
    env = os.environ.get("SATFLIGHT_DATA", "").strip()
    return Path(env).expanduser() if env else Path.home() / "satflight-data"


def mission_dir(mission: str, root: Path | None = None) -> Path:
    """The folder holding one satellite's POEORB files."""
    if mission not in MISSIONS:
        raise ValueError(f"unknown Sentinel-1 unit {mission!r}; expected one of {MISSIONS}")
    return (root or data_root()) / "poeorb" / mission


def _stamp(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class FileName:
    """What a POEORB file name says."""
    name: str
    mission: str
    created: datetime
    start: datetime
    stop: datetime

    @property
    def day(self) -> date:
        """The nominal day: the one in the middle of the file's 26 h."""
        return (self.start + (self.stop - self.start) / 2).date()

    @property
    def folder(self) -> str:
        """ESA's ``YYYY/MM`` folder for this file (the month of its start)."""
        return f"{self.start:%Y/%m}"


def parse_name(name: str) -> FileName:
    """The parts of a POEORB file name; ValueError if it is not one."""
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"{name!r} is not a POEORB file name")
    start, stop = _stamp(m["start"]), _stamp(m["stop"])
    if not timedelta(hours=20) <= stop - start <= timedelta(hours=30):
        raise ValueError(f"{name}: covers {stop - start}, not about 26 h")
    return FileName(name, m["mission"], _stamp(m["created"]), start, stop)


def listing_names(html: str) -> list[str]:
    """POEORB file names linked from one of ESA's folder index pages."""
    return [h for h in re.findall(r'href="([^"?/]+)"', html) if NAME_RE.match(h)]


def newest_per_day(names) -> dict[date, FileName]:
    """{nominal day: file} keeping the most recently created file of each day
    (ESA can reissue a day)."""
    out: dict[date, FileName] = {}
    for n in names:
        f = parse_name(n)
        if f.day not in out or f.created > out[f.day].created:
            out[f.day] = f
    return out


def folders_for(first: date, last: date) -> list[str]:
    """ESA ``YYYY/MM`` folders that hold the files of days ``first``..``last``:
    a day's file sits in the folder of the day before."""
    d, out = first - timedelta(days=1), []
    while d < last:                         # the day before each of first..last
        f = f"{d:%Y/%m}"
        if f not in out:
            out.append(f)
        d += timedelta(days=1)
    return out


def local_files(mission: str, root: Path | None = None) -> dict[date, Path]:
    """{nominal day: path} of the files already downloaded for ``mission``,
    the newest of each day."""
    folder = mission_dir(mission, root)
    paths = {p.name: p for p in folder.glob("*.EOF.zip")} if folder.is_dir() else {}
    return {d: paths[f.name] for d, f in newest_per_day(paths).items()}


# --- reading a file -------------------------------------------------------------------------

STEP = 10.0                     # s between state vectors
R_BOUNDS = (6900.0, 7250.0)     # km: Sentinel-1 flies ~693 km up (|r| ~7071 km)
V_BOUNDS = (6.5, 8.0)           # km/s, Earth-fixed speed
V_TOL = 0.5e-3                  # km/s: velocity vs the positions' central difference
MISSION_NAMES = {f"Sentinel-1{u[-1]}": u for u in MISSIONS}
# the quality flags in a year of S1A and S1C files (2025-07..2026-06):
# DEGRADED-MANOEUVRE marks the ~10 min around an orbit-keeping burn (every 1-3
# days), DEGRADED-GAP a stretch with no tracking data, bridged by ESA's orbit
# model (minutes to ~11 h, a few times a year). Any other value is refused.
QUALITIES = ("NOMINAL", "DEGRADED-MANOEUVRE", "DEGRADED-GAP")


class PoeorbError(ValueError):
    """A POEORB file that cannot be used, with the file (and record) named."""


@dataclass(frozen=True)
class OrbitFile:
    """One POEORB file's state vectors, Earth-fixed (ITRF), every 10 s."""
    path: Path
    mission: str                # "S1A"
    start: datetime             # validity window (UTC)
    stop: datetime
    utc: np.ndarray             # datetime64[us], (N,)
    dut1: np.ndarray            # UT1 - UTC, s, (N,)
    tai_utc: float              # TAI - UTC, s (constant within a file)
    abs_orbit: np.ndarray       # int, (N,)
    r: np.ndarray               # km, (N, 3) ECEF
    v: np.ndarray               # km/s, (N, 3) ECEF (relative to the rotating Earth)
    manoeuvre: np.ndarray       # bool, (N,): flagged DEGRADED-MANOEUVRE (a burn nearby)
    gap: np.ndarray             # bool, (N,): flagged DEGRADED-GAP (no tracking data)

    @property
    def day(self) -> date:
        return parse_name(self.path.name).day

    @property
    def degraded(self) -> np.ndarray:
        """Samples flagged anything but NOMINAL."""
        return self.manoeuvre | self.gap


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name: str, where: str):
    for c in el:
        if _local(c.tag) == name:
            return c
    raise PoeorbError(f"{where}: no <{name}>")


def _text(el, path: str, where: str) -> str:
    for name in path.split("/"):
        el = _child(el, name, where)
    return (el.text or "").strip()


def _times(values, prefix: str, where: str) -> np.ndarray:
    bad = next((k for k, s in enumerate(values) if not s.startswith(prefix)), None)
    if bad is not None:
        raise PoeorbError(f"{where}, OSV {bad}: time {values[bad]!r} lacks {prefix!r}")
    try:
        return np.array([s[len(prefix):] for s in values], dtype="datetime64[us]")
    except ValueError as exc:
        raise PoeorbError(f"{where}: unreadable time ({exc})") from None


def _seconds(d: np.ndarray) -> np.ndarray:
    """timedelta64[us] -> float seconds."""
    return d.astype("int64") / 1e6


def _load_xml(path: Path):
    import xml.etree.ElementTree as ET
    import zipfile
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                inner = z.namelist()
                if len(inner) != 1:
                    raise PoeorbError(f"{path.name}: holds {inner}, expected one .EOF")
                raw = z.read(inner[0])
        else:
            raw = path.read_bytes()
        return ET.fromstring(raw)
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise PoeorbError(f"{path.name}: cannot be read ({exc})") from None


def read_file(path) -> OrbitFile:
    """Read and check one POEORB file (``.EOF.zip`` or ``.EOF``). Anything off
    raises :class:`PoeorbError` naming the file and the state vector."""
    path = Path(path)
    where = path.name
    try:
        fn = parse_name(path.name)
    except ValueError as exc:
        raise PoeorbError(str(exc)) from None
    root = _load_xml(path)
    if _local(root.tag) != "Earth_Explorer_File":
        raise PoeorbError(f"{where}: root is <{_local(root.tag)}>, not an Earth Explorer file")
    head = _child(root, "Earth_Explorer_Header", where)
    expect = {
        "Fixed_Header/File_Name": path.name.split(".")[0],
        "Fixed_Header/File_Type": "AUX_POEORB",
        "Fixed_Header/Mission": next(k for k, u in MISSION_NAMES.items() if u == fn.mission),
        "Fixed_Header/Validity_Period/Validity_Start": f"UTC={fn.start:%Y-%m-%dT%H:%M:%S}",
        "Fixed_Header/Validity_Period/Validity_Stop": f"UTC={fn.stop:%Y-%m-%dT%H:%M:%S}",
        "Variable_Header/Ref_Frame": "EARTH_FIXED",
        "Variable_Header/Time_Reference": "UTC",
    }
    for p, want in expect.items():
        got = _text(head, p, where)
        if got != want:
            raise PoeorbError(f"{where}: {p.rsplit('/', 1)[-1]} is {got!r}, expected {want!r}")

    lst = _child(_child(root, "Data_Block", where), "List_of_OSVs", where)
    osvs = [e for e in lst if _local(e.tag) == "OSV"]
    if str(len(osvs)) != lst.get("count"):
        raise PoeorbError(f"{where}: {len(osvs)} state vectors, but count says "
                          f"{lst.get('count')}")
    if len(osvs) < 3:
        raise PoeorbError(f"{where}: only {len(osvs)} state vectors")

    units = {"X": "m", "Y": "m", "Z": "m", "VX": "m/s", "VY": "m/s", "VZ": "m/s"}
    cols = {k: [] for k in ("TAI", "UTC", "UT1", "Absolute_Orbit", *units, "Quality")}
    for k, osv in enumerate(osvs):
        fields = {_local(c.tag): c for c in osv}
        for name, col in cols.items():
            el = fields.get(name)
            if el is None:
                raise PoeorbError(f"{where}, OSV {k}: no <{name}>")
            if name in units and el.get("unit") != units[name]:
                raise PoeorbError(f"{where}, OSV {k}: {name} in {el.get('unit')!r}, "
                                  f"expected {units[name]!r}")
            col.append((el.text or "").strip())

    bad = next((k for k, q in enumerate(cols["Quality"]) if q not in QUALITIES), None)
    if bad is not None:
        raise PoeorbError(f"{where}, OSV {bad}: unknown quality {cols['Quality'][bad]!r} "
                          f"(known: {', '.join(QUALITIES)})")
    utc = _times(cols["UTC"], "UTC=", where)
    tai = _times(cols["TAI"], "TAI=", where)
    ut1 = _times(cols["UT1"], "UT1=", where)
    try:
        num = {k: np.array(cols[k], dtype=float) for k in units}
        orbit = np.array(cols["Absolute_Orbit"], dtype=int)
    except ValueError as exc:
        raise PoeorbError(f"{where}: unreadable number ({exc})") from None

    step = _seconds(np.diff(utc))
    bad = np.flatnonzero(step != STEP)
    if bad.size:
        raise PoeorbError(f"{where}, OSV {bad[0] + 1}: {step[bad[0]]:g} s after the one "
                          f"before, not {STEP:g} s")
    first = np.datetime64(fn.start.replace(tzinfo=None), "us")
    last = np.datetime64(fn.stop.replace(tzinfo=None), "us")
    if utc[0] != first or utc[-1] != last:
        raise PoeorbError(f"{where}: state vectors run {utc[0]}..{utc[-1]}, not the "
                          f"validity window {first}..{last}")
    leap = _seconds(tai - utc)
    if np.any(leap != leap[0]):
        k = int(np.flatnonzero(leap != leap[0])[0])
        raise PoeorbError(f"{where}, OSV {k}: TAI-UTC changes ({leap[0]:g} -> {leap[k]:g} s)")
    dut1 = _seconds(ut1 - utc)
    if np.any(np.abs(dut1) >= 0.9):
        k = int(np.flatnonzero(np.abs(dut1) >= 0.9)[0])
        raise PoeorbError(f"{where}, OSV {k}: UT1-UTC = {dut1[k]:g} s, beyond 0.9 s")
    if np.any(np.diff(orbit) < 0):
        k = int(np.flatnonzero(np.diff(orbit) < 0)[0]) + 1
        raise PoeorbError(f"{where}, OSV {k}: the orbit number goes back")

    r = np.column_stack([num["X"], num["Y"], num["Z"]]) / 1000.0
    v = np.column_stack([num["VX"], num["VY"], num["VZ"]]) / 1000.0
    for name, x, (lo, hi), unit in (("|r|", r, R_BOUNDS, "km"), ("|v|", v, V_BOUNDS, "km/s")):
        mag = np.linalg.norm(x, axis=1)
        out = np.flatnonzero(~((mag >= lo) & (mag <= hi)))      # NaN is outside too
        if out.size:
            k = int(out[0])
            raise PoeorbError(f"{where}, OSV {k}: {name} = {mag[k]:.3f} {unit}, outside "
                              f"{lo:g}..{hi:g}")
    # the velocities must be the positions' rate of change (catches swapped or
    # mis-scaled columns): central differences agree to ~0.1 m/s at 10 s. Not
    # across a burn, where thrust changes the velocity between samples. (Gaps
    # are bridged by an orbit model, so they are smooth and still checked.)
    burn = np.array([q == "DEGRADED-MANOEUVRE" for q in cols["Quality"]])
    gap = np.array([q == "DEGRADED-GAP" for q in cols["Quality"]])
    quiet = ~(burn[:-2] | burn[1:-1] | burn[2:])
    err = np.linalg.norm((r[2:] - r[:-2]) / (2 * STEP) - v[1:-1], axis=1)
    err[~quiet] = 0.0
    if err.max() > V_TOL:
        k = int(err.argmax())
        raise PoeorbError(f"{where}, OSV {k + 1}: velocity disagrees with the positions by "
                          f"{err[k] * 1e3:.2f} m/s")
    return OrbitFile(path, fn.mission, fn.start, fn.stop, utc, dut1, float(leap[0]), orbit,
                     r, v, burn, gap)


# --- several days as one arc ----------------------------------------------------------------

@dataclass(frozen=True)
class Seam:
    """Where two neighbouring days' files overlap: how far apart they put the
    satellite at the samples both flag NOMINAL. ``points`` 0 (and NaN
    distances) when a burn or tracking gap covers the whole overlap."""
    day: date                   # the later day
    points: int
    max_pos_m: float
    max_vel_mm_s: float


@dataclass(frozen=True)
class Arc:
    """One satellite's state vectors over whole days, each day taken from its own
    file (00:00 to 24:00 UTC), every 10 s with no gaps or repeats."""
    mission: str
    first: date
    last: date
    utc: np.ndarray
    dut1: np.ndarray
    tai_utc: np.ndarray
    abs_orbit: np.ndarray
    r: np.ndarray
    v: np.ndarray
    manoeuvre: np.ndarray
    gap: np.ndarray
    seams: tuple[Seam, ...]

    def seconds(self, t0=None) -> np.ndarray:
        """Time of each state vector in s since ``t0`` (default the arc's start)."""
        t0 = self.utc[0] if t0 is None else np.datetime64(t0, "us")
        return _seconds(self.utc - t0)

    def eci(self, eop=None):
        """(r, v) in the simulator's ECI frame (km, km/s), see :func:`to_eci`;
        with ``eop`` (an ``eop.Table``) polar motion is taken out."""
        pole = None if eop is None else eop.pole(self.utc)
        return to_eci(self.utc, self.dut1, self.r, self.v, pole=pole)


def julian_dates(utc: np.ndarray) -> np.ndarray:
    """Julian dates of datetime64 UTC times."""
    unix = np.datetime64("1970-01-01T00:00:00", "us")
    return 2440587.5 + _seconds(np.asarray(utc, "datetime64[us]") - unix) / 86400.0


def to_eci(utc, dut1, r_ecef, v_ecef, pole=None):
    """Earth-fixed states (km, km/s) into the simulator's ECI frame: the Earth
    turned back by GMST at each state's UT1 (``frames``: ECI is of-date, so no
    precession or nutation enters). ``pole`` = (x_p, y_p) rad per state (see
    ``eop``) first takes the ITRF onto the spin axis; without it polar motion
    (~15 m, a daily cross-track wobble) is neglected."""
    from .frames import ecef_to_eci_state
    from .timeutil import gmst
    r_ecef, v_ecef = np.asarray(r_ecef, float), np.asarray(v_ecef, float)
    if pole is not None:
        from .eop import itrf_to_tirs
        W = itrf_to_tirs(*pole)
        r_ecef = np.einsum("nij,nj->ni", W, r_ecef)
        v_ecef = np.einsum("nij,nj->ni", W, v_ecef)
    theta = gmst(julian_dates(utc) + np.asarray(dut1, float) / 86400.0)
    return ecef_to_eci_state(r_ecef, v_ecef, theta)


# m: neighbouring days' solutions must agree this well where they overlap (at
# samples both flag NOMINAL). Over S1A and S1C 2025-07..2026-06 (718 seams) the
# median is 2 mm and 95 % are under 23 mm, but 7 differ by 0.6-5.7 m with no
# burn: two independent daily solutions. A unit or file mix-up is km, a 10 s
# slip 75 km.
SEAM_TOL_M = 10.0


def _day_slice(f: OrbitFile, day: date) -> slice:
    t0 = np.datetime64(day, "us")
    t1 = t0 + np.timedelta64(1, "D")
    return slice(int(np.searchsorted(f.utc, t0)), int(np.searchsorted(f.utc, t1)))


def _seam(prev: OrbitFile, cur: OrbitFile) -> Seam:
    _, ip, ic = np.intersect1d(prev.utc, cur.utc, return_indices=True)
    keep = ~(prev.degraded[ip] | cur.degraded[ic])
    ip, ic = ip[keep], ic[keep]
    if ip.size == 0:
        return Seam(cur.day, 0, math.nan, math.nan)
    dr = np.linalg.norm(prev.r[ip] - cur.r[ic], axis=1).max() * 1e3
    dv = np.linalg.norm(prev.v[ip] - cur.v[ic], axis=1).max() * 1e6
    return Seam(cur.day, int(ip.size), float(dr), float(dv))


def load_arc(mission: str, first: date, last: date, root: Path | None = None,
             seam_tol_m: float = SEAM_TOL_M) -> Arc:
    """Days ``first``..``last`` of ``mission`` from the downloaded files, as one
    arc. A day with no file, or neighbouring days that disagree by more than
    ``seam_tol_m`` where they overlap, is an error (nothing is bridged)."""
    have = local_files(mission, root)
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    missing = [d for d in days if d not in have]
    if missing:
        raise PoeorbError(f"{mission}: no downloaded file for {len(missing)} day(s), "
                          f"first {missing[0]} (run tools/fetch_poeorb.py)")
    parts, seams, prev = [], [], None
    for d in days:
        f = read_file(have[d])
        if prev is not None:
            s = _seam(prev, f)
            if s.max_pos_m > seam_tol_m:
                raise PoeorbError(f"{f.path.name}: differs from the day before by "
                                  f"{s.max_pos_m:.3f} m where they overlap (limit "
                                  f"{seam_tol_m:g} m)")
            seams.append(s)
        sl = _day_slice(f, d)
        parts.append((f, sl))
        prev = f

    def cat(attr):
        return np.concatenate([getattr(f, attr)[sl] for f, sl in parts])

    utc = cat("utc")
    expect = int(len(days) * 86400 / STEP)
    if len(utc) != expect or np.any(_seconds(np.diff(utc)) != STEP):
        raise PoeorbError(f"{mission} {first}..{last}: {len(utc)} state vectors, expected "
                          f"{expect} every {STEP:g} s")
    tai_utc = np.concatenate([np.full(sl.stop - sl.start, f.tai_utc) for f, sl in parts])
    return Arc(mission, first, last, utc, cat("dut1"), tai_utc, cat("abs_orbit"), cat("r"),
               cat("v"), cat("manoeuvre"), cat("gap"), tuple(seams))
