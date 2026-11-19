"""Download the IERS Earth-orientation table (finals2000A.all, ~4 MB).

    python tools/fetch_eop.py            # into $SATFLIGHT_DATA/eop/ (default ~/satflight-data)

Polar motion and UT1-UTC by day since 1973, measured up to about a week ago,
then a year of predictions (satflight.eop uses only the measured rows). The
download is read with the same strict reader before it replaces the old table,
so a broken or truncated download is never kept. Re-run it to extend the table.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satflight import eop


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "satflight-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(root: Path | None = None, get=_get, out=print) -> Path:
    """Download, check and keep the table; returns its path. EopError (and
    nothing written) if the download does not read as a table."""
    dest = eop.table_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    part.write_bytes(get(eop.URL))
    try:
        t = eop.read(part)
    except eop.EopError:
        part.unlink()
        raise
    part.replace(dest)
    last = t.mjd[t.measured][-1]
    out(f"{dest}: MJD {t.mjd[0]:.0f}..{t.mjd[-1]:.0f}, measured to MJD {last:.0f}")
    return dest


def main() -> int:
    try:
        fetch()
    except (eop.EopError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
