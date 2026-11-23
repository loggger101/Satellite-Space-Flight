"""Space weather from CelesTrak: the solar flux and geomagnetic indices NRLMSIS needs.

The upper atmosphere swells with the Sun's extreme-ultraviolet output (tracked by
the 10.7 cm radio flux, F10.7) and with geomagnetic storms (the ap index). At
Sentinel-1's 700 km a storm with ap 300 makes the air about 7 times denser; the
year study's worst along-track errors are storm weeks.

The table is CelesTrak's ``SW-All.csv`` (daily since 1957, then predictions),
kept outside the repository under ``$SATFLIGHT_DATA/spaceweather/``
(``tools/fetch_space_weather.py``). Only measured days are used: rows of type
``OBS``, and ``INT`` (CelesTrak's fill for the odd day without a Penticton
reading, from the days around it). A time whose inputs reach a predicted day
is refused, as is one needing the F10.7 of a solar radio burst (over 400 sfu:
a flare's radio noise, not the quiet-Sun flux the model is fitted to).

The inputs follow the model's definition (and pymsis's own reader of the file,
which the tests hold this one to): F10.7 of the previous day, its 81-day mean
centred on the day, and the ap array: the day's Ap, the 3-hour ap now and 3, 6
and 9 hours before, and the means of the eight 3-hour values 12-33 and 36-57
hours before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FILE_NAME = "SW-All.csv"
URL = "https://celestrak.org/SpaceData/SW-All.csv"
HEADER = ("DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,"
          "AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,"
          "F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81")
MEASURED = ("OBS", "INT")
PREDICTED = ("PRD", "PRM")
RADIO_BURST = 400.0           # sfu; above it the day's F10.7 is a flare's, not the Sun's
HALF_WINDOW = 40              # days either side in the centred 81-day mean
_COLUMNS = HEADER.split(",")
_AP = [_COLUMNS.index(f"AP{j}") for j in range(1, 9)]
_AP_AVG, _F107, _KIND, _F107A = (_COLUMNS.index(c) for c in (
    "AP_AVG", "F10.7_OBS", "F10.7_DATA_TYPE", "F10.7_OBS_CENTER81"))


class SpaceWeatherError(ValueError):
    """A space-weather table that cannot be used, or a time it cannot serve."""


def table_path(root: Path | None = None) -> Path:
    """Where the table lives: ``<data root>/spaceweather/SW-All.csv``."""
    from .poeorb import data_root
    return (root or data_root()) / "spaceweather" / FILE_NAME


@dataclass(frozen=True)
class Table:
    """The measured days: date, 3-hourly ap (N, 8), daily Ap, F10.7 (sfu) and
    its centred 81-day mean."""
    day: np.ndarray
    ap3: np.ndarray
    apd: np.ndarray
    f107: np.ndarray
    f107a: np.ndarray

    def indices(self, utc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(F10.7, F10.7a, ap (N, 7)) at datetime64 UTC times, as NRLMSIS takes
        them. SpaceWeatherError naming the first time that needs a day the table
        does not measure."""
        utc = np.atleast_1d(np.asarray(utc, "datetime64[m]"))
        minutes = (utc - self.day[0].astype("datetime64[m]")).astype(np.int64)
        d = minutes // 1440                        # day index
        k = minutes // 180                         # 3-hour slot index
        last = len(self.day) - 1

        def refuse(bad, why):
            if np.any(bad):
                i = int(np.flatnonzero(bad)[0])
                raise SpaceWeatherError(f"{str(utc[i])[:16]}: {why(i)}")

        refuse((d < 1) | (d > last), lambda i: (
            f"outside the measured space-weather table ({self.day[0]}..{self.day[-1]}); "
            "fetch a newer table (python tools/fetch_space_weather.py)"))
        refuse(k < 19, lambda i: (
            f"the ap history (57 h before it) starts before the table's {self.day[0]}"))
        refuse(d + HALF_WINDOW > last, lambda i: (
            f"its 81-day mean F10.7 reaches past the last measured day {self.day[-1]} "
            "into predictions; only measured space weather is used"))
        f = self.f107[d - 1]
        refuse((f > RADIO_BURST) | (f <= 0), lambda i: (
            f"it needs the F10.7 of {self.day[d[i] - 1]}, {f[i]:.1f} sfu: a solar radio "
            f"burst (over {RADIO_BURST:.0f}), not the quiet-Sun flux the model takes"))
        s = self.ap3.reshape(-1)
        mean8 = np.convolve(s, np.ones(8) / 8.0, mode="valid")   # mean8[j] = s[j..j+7]
        ap = np.stack([self.apd[d], s[k], s[k - 1], s[k - 2], s[k - 3],
                       mean8[k - 11], mean8[k - 19]], axis=-1)
        return f, self.f107a[d], ap


def _row(fields: list[str], n: int):
    try:
        if len(fields) != len(_COLUMNS):
            raise ValueError(f"{len(fields)} fields, expected {len(_COLUMNS)}")
        aps = [int(fields[c]) for c in _AP]
        if min(aps) < 0:
            raise ValueError("a negative ap")
        return (np.datetime64(fields[0], "D"), aps, int(fields[_AP_AVG]),
                float(fields[_F107]), float(fields[_F107A]))
    except ValueError as exc:
        raise SpaceWeatherError(f"line {n}: not an SW-All row ({exc}): "
                                f"{','.join(fields)[:60]!r}") from None


def read(path: Path | None = None) -> Table:
    """Read an SW-All.csv strictly: the columns must be CelesTrak's, every
    measured row must parse, the days must follow one another, and no measured
    day may come after a predicted one."""
    path = Path(path) if path is not None else table_path()
    if not path.is_file():
        raise SpaceWeatherError(f"no space-weather table at {path}: "
                                "run python tools/fetch_space_weather.py")
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    if not lines or lines[0].strip() != HEADER:
        raise SpaceWeatherError(f"{path}: not CelesTrak's SW-All columns "
                                f"(first line {lines[0][:60] if lines else ''!r})")
    rows, predicted_at = [], None
    for n, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.strip().split(",")
        kind = fields[_KIND] if len(fields) > _KIND else ""
        if kind in PREDICTED:
            predicted_at = predicted_at or n
            continue
        if kind not in MEASURED:
            raise SpaceWeatherError(f"line {n}: F10.7 type {kind!r}, expected one of "
                                    f"{', '.join(MEASURED + PREDICTED)}")
        if predicted_at is not None:
            raise SpaceWeatherError(f"line {n}: a measured day after a prediction "
                                    f"(line {predicted_at})")
        rows.append(_row(fields, n))
        if len(rows) > 1 and rows[-1][0] != rows[-2][0] + np.timedelta64(1, "D"):
            raise SpaceWeatherError(f"line {n}: {rows[-1][0]} does not follow {rows[-2][0]}")
    if len(rows) < 2:
        raise SpaceWeatherError(f"{path}: fewer than two measured days")
    day, aps, apd, f107, f107a = zip(*rows, strict=True)
    return Table(np.array(day, "datetime64[D]"), np.array(aps, float),
                 np.array(apd, float), np.array(f107), np.array(f107a))


_CACHE: dict = {}


def load(path: Path | None = None) -> Table:
    """The table at ``path`` (default :func:`table_path`), read once and re-read
    when the file changes."""
    path = Path(path) if path is not None else table_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is None or _CACHE.get("key") != key:
        table = read(path)
        _CACHE.update(key=key, table=table)
    return _CACHE["table"]
