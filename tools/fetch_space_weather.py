"""Download CelesTrak's space-weather table (SW-All.csv, ~3 MB) and NOAA SWPC's
predicted solar cycle (predicted-solar-cycle.json, ~20 kB).

    python tools/fetch_space_weather.py   # into $SATFLIGHT_DATA/spaceweather/
                                          # (default ~/satflight-data)

F10.7 and the 3-hourly ap index by day since 1957, measured up to yesterday,
then predictions (satflight.spaceweather uses only the measured rows). The
NRLMSIS atmosphere (forces "atmosphere": "msis") reads it. The download is read
with the same strict reader before it replaces the old table, so a broken or
truncated download is never kept. Re-run it to extend the table.

The SWPC forecast (monthly F10.7 to the end of the current solar cycle) is the
future Sun for NRLMSIS lifetime estimates (satflight.solarforecast), checked
the same way.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satflight import solarforecast as sf
from satflight import spaceweather as sw


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "satflight-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(root: Path | None = None, get=_get, out=print) -> Path:
    """Download, check and keep the table; returns its path. SpaceWeatherError
    (and nothing written) if the download does not read as a table."""
    dest = sw.table_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    part.write_bytes(get(sw.URL))
    try:
        t = sw.read(part)
    except sw.SpaceWeatherError:
        part.unlink()
        raise
    part.replace(dest)
    out(f"{dest}: measured {t.day[0]}..{t.day[-1]} ({len(t.day)} days)")
    return dest


def fetch_forecast(root: Path | None = None, get=_get, out=print) -> Path:
    """Download, check and keep SWPC's predicted solar cycle; returns its path.
    SpaceWeatherError (and nothing written) if it does not read as one."""
    dest = sf.forecast_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    part.write_bytes(get(sf.URL))
    try:
        f = sf.read_swpc(part)
    except sw.SpaceWeatherError:
        part.unlink()
        raise
    part.replace(dest)
    out(f"{dest}: forecast {f.month[0]}..{f.month[-1]} ({len(f.month)} months)")
    return dest


def main() -> int:
    try:
        fetch()
        fetch_forecast()
    except (sw.SpaceWeatherError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
