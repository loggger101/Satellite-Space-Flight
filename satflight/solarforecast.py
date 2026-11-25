"""The future Sun, for NRLMSIS lifetimes that look years ahead.

NRLMSIS needs F10.7, its 81-day mean and ap for every day it is asked about.
:mod:`satflight.spaceweather` serves only measured days. For the days after
them this module supplies, in order:

1. NOAA SWPC's predicted solar cycle (``predicted-solar-cycle.json``, monthly,
   to the end of the current cycle), with its high and low bands;
2. after that, the mean of the measured solar cycles, repeated. The cycles are
   the five complete ones in the measured table, between the SILSO minima in
   ``CYCLE_MINIMA``; each is put on a phase from 0 (minimum) to 1 (next
   minimum), the 81-day mean F10.7 is averaged across them by phase, and the
   repeats last their mean length (about 11 years), the first starting where
   the SWPC forecast ends. The high and low bands are the busiest and quietest
   of the measured cycles at each phase.

A forecast has no storms: ap is held at its long-term mean (the measured daily
Ap over the same cycles) in all seven of NRLMSIS's ap inputs. Measured days
are used as they are, at every level. :class:`SolarForecast` has the same
``indices(utc)`` as :class:`satflight.spaceweather.Table`, so NRLMSIS takes
either (:func:`satflight.atmosphere.msis_state`'s ``table``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .spaceweather import HALF_WINDOW, SpaceWeatherError, Table

FILE_NAME = "predicted-solar-cycle.json"
URL = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
# solar-cycle minima (13-month smoothed sunspot number), SILSO, Royal Observatory
# of Belgium: the starts of cycles 20-25
CYCLE_MINIMA = ("1964-10", "1976-03", "1986-09", "1996-08", "2008-12", "2019-12")
LEVELS = ("mean", "high", "low")
_SWPC_KEY = {"mean": "predicted_f10.7", "high": "high_f10.7", "low": "low_f10.7"}
F107_RANGE = (50.0, 400.0)        # sfu: a forecast outside this is not the Sun's
PHASES = 400                      # points on the mean cycle


def forecast_path(root: Path | None = None) -> Path:
    """Where the SWPC forecast lives: ``<data root>/spaceweather/<FILE_NAME>``."""
    from .poeorb import data_root
    return (root or data_root()) / "spaceweather" / FILE_NAME


@dataclass(frozen=True)
class Swpc:
    """SWPC's monthly predicted F10.7 (sfu) at each level, month by month."""
    month: np.ndarray                   # datetime64[M]
    f107: dict


def read_swpc(path: Path | None = None) -> Swpc:
    """Read the SWPC predicted-solar-cycle JSON strictly: consecutive months,
    every level's F10.7 present and in ``F107_RANGE``."""
    path = Path(path) if path is not None else forecast_path()
    if not path.is_file():
        raise SpaceWeatherError(f"no solar-cycle forecast at {path}: "
                                "run python tools/fetch_space_weather.py")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise SpaceWeatherError(f"{path}: not JSON ({exc})") from None
    if not isinstance(rows, list) or not rows:
        raise SpaceWeatherError(f"{path}: no months in the forecast")
    month, f107 = [], {lev: [] for lev in LEVELS}
    for n, row in enumerate(rows):
        try:
            m = np.datetime64(row["time-tag"], "M")
            vals = {lev: float(row[key]) for lev, key in _SWPC_KEY.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise SpaceWeatherError(f"{path}: month {n}: missing or bad {exc}") from None
        if month and m != month[-1] + np.timedelta64(1, "M"):
            raise SpaceWeatherError(f"{path}: {m} does not follow {month[-1]}")
        for lev, v in vals.items():
            if not F107_RANGE[0] <= v <= F107_RANGE[1]:
                raise SpaceWeatherError(f"{path}: {m} {lev} F10.7 {v} sfu is outside "
                                        f"{F107_RANGE[0]:g}-{F107_RANGE[1]:g}")
            f107[lev].append(v)
        month.append(m)
    return Swpc(np.array(month, "datetime64[M]"),
                {lev: np.array(v) for lev, v in f107.items()})


@dataclass(frozen=True)
class MeanCycle:
    """F10.7 (sfu) at each level on a phase from 0 (minimum) to 1, and the mean
    cycle length in days."""
    phase: np.ndarray
    f107: dict
    period_days: float


def _minima() -> np.ndarray:
    return np.array([np.datetime64(m, "M").astype("datetime64[D]") for m in CYCLE_MINIMA])


def mean_cycle(table: Table) -> MeanCycle:
    """The measured cycles between ``CYCLE_MINIMA``, averaged by phase (module
    docstring). SpaceWeatherError if the table does not cover them."""
    mins = _minima()
    if table.day[0] > mins[0] or table.day[-1] < mins[-1]:
        raise SpaceWeatherError(f"the measured table ({table.day[0]}..{table.day[-1]}) does "
                                f"not cover the solar cycles {mins[0]}..{mins[-1]}")
    phase = np.linspace(0.0, 1.0, PHASES)
    days = (table.day - table.day[0]).astype(float)
    curves = []
    for a, b in zip(mins[:-1], mins[1:], strict=True):
        t = (a - table.day[0]).astype(float) + phase * float((b - a).astype(int))
        curves.append(np.interp(t, days, table.f107a))
    curves = np.array(curves)
    return MeanCycle(phase, {"mean": curves.mean(0), "high": curves.max(0),
                             "low": curves.min(0)},
                     float((mins[-1] - mins[0]).astype(int)) / (len(mins) - 1))


def mean_ap(table: Table) -> float:
    """The measured daily Ap averaged over the cycles of ``CYCLE_MINIMA``."""
    mins = _minima()
    inside = (table.day >= mins[0]) & (table.day < mins[-1])
    if not inside.any():
        raise SpaceWeatherError("the measured table does not cover the solar cycles")
    return float(table.apd[inside].mean())


@dataclass(frozen=True)
class SolarForecast:
    """Measured space weather, then SWPC's forecast, then the mean cycle, at one
    ``level`` (module docstring). Has :meth:`Table.indices`'s interface."""
    table: Table
    swpc: Swpc
    cycle: MeanCycle
    ap_mean: float
    level: str = "mean"

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"solar forecast level {self.level!r}: expected one of "
                             f"{', '.join(LEVELS)}")

    def at_level(self, level: str) -> SolarForecast:
        return replace(self, level=level)

    @property
    def measured_until(self) -> np.datetime64:
        """The first day the measured table cannot serve (its 81-day mean would
        reach past the last measured day)."""
        return self.table.day[-1] - np.timedelta64(HALF_WINDOW - 1, "D")

    def f107_after(self, utc) -> np.ndarray:
        """The forecast F10.7 (sfu) at datetime64 times past the measured days."""
        t = np.atleast_1d(np.asarray(utc, "datetime64[m]"))
        first = self.swpc.month[0].astype("datetime64[m]")
        if np.any(t < first):
            raise SpaceWeatherError(
                f"{str(t[t < first][0])[:16]}: the solar-cycle forecast starts "
                f"{self.swpc.month[0]}, after the measured days end "
                f"({self.measured_until}); fetch both again "
                "(python tools/fetch_space_weather.py)")
        day = 1440.0
        mid = (self.swpc.month.astype("datetime64[D]")
               + np.timedelta64(14, "D")).astype("datetime64[m]")
        x = (t - first).astype(float) / day
        xm = (mid - first).astype(float) / day
        end = (self.swpc.month[-1] + np.timedelta64(1, "M")).astype("datetime64[m]")
        out = np.interp(x, xm, self.swpc.f107[self.level])
        past = t >= end
        if past.any():
            ph = ((t[past] - end).astype(float) / day / self.cycle.period_days) % 1.0
            out[past] = np.interp(ph, self.cycle.phase, self.cycle.f107[self.level])
        return out

    def indices(self, utc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(F10.7, F10.7a, ap (N, 7)) at datetime64 UTC times, as NRLMSIS takes
        them: measured where the table serves, the forecast after."""
        utc = np.atleast_1d(np.asarray(utc, "datetime64[m]"))
        later = utc >= self.measured_until.astype("datetime64[m]")
        f = np.empty(utc.shape)
        fa = np.empty(utc.shape)
        ap = np.empty(utc.shape + (7,))
        if (~later).any():
            f[~later], fa[~later], ap[~later] = self.table.indices(utc[~later])
        if later.any():
            v = self.f107_after(utc[later])
            f[later], fa[later], ap[later] = v, v, self.ap_mean
        return f, fa, ap


def load(level: str = "mean", root: Path | None = None) -> SolarForecast:
    """The forecast from the downloaded measured table and SWPC file."""
    from . import spaceweather
    table = spaceweather.load(spaceweather.table_path(root))
    return SolarForecast(table, read_swpc(forecast_path(root)), mean_cycle(table),
                         mean_ap(table), level)


__all__ = ["SolarForecast", "MeanCycle", "Swpc", "read_swpc", "mean_cycle", "mean_ap",
           "load", "forecast_path", "CYCLE_MINIMA", "LEVELS", "URL", "FILE_NAME"]
