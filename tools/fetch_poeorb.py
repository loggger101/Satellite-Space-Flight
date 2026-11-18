"""Download Sentinel-1 precise orbit files (POEORB) from ESA.

    python tools/fetch_poeorb.py                       # S1A and S1C, 2025-07-01..2026-06-30
    python tools/fetch_poeorb.py --sats S1C --from 2026-01-01 --to 2026-01-07
    python tools/fetch_poeorb.py --dry-run             # say what would be fetched

Files go to ``$SATFLIGHT_DATA/poeorb/<unit>/`` (default ``~/satflight-data``),
outside the repository. One file per day; a day already downloaded is skipped,
so an interrupted run picks up where it stopped. Each download is checked (a
readable zip holding the one .EOF its name promises) before it is kept. A day
ESA has no file for is reported and the exit status is 1: nothing is
substituted for it.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satflight.poeorb import (
    BASE_URL,
    folders_for,
    listing_names,
    local_files,
    mission_dir,
    newest_per_day,
)

FIRST, LAST = date(2025, 7, 1), date(2026, 6, 30)
PAUSE = 0.5            # s between requests: ESA's server is shared
RETRIES = (5, 20, 60)  # s to wait before each retry of a failed request


def _get(url: str) -> bytes:
    """``url``'s body; retries a few times on network errors and 5xx, never on 4xx."""
    req = urllib.request.Request(url, headers={"User-Agent": "satflight-poeorb-fetch"})
    for wait in (*RETRIES, None):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or wait is None:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if wait is None:
                raise
        print(f"  retrying {url} in {wait} s", flush=True)
        time.sleep(wait)
    raise AssertionError("unreachable")


def check_zip(name: str, data: bytes) -> None:
    """ValueError unless ``data`` is a sound zip holding exactly the .EOF that
    ``name`` (``*.EOF.zip``) promises."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            bad = z.testzip()
            inner = z.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{name}: not a zip file ({exc})") from None
    if bad is not None:
        raise ValueError(f"{name}: damaged member {bad}")
    want = name.removesuffix(".zip")
    if [Path(n).name for n in inner] != [want]:
        raise ValueError(f"{name}: holds {inner}, expected just {want}")


def remote_files(mission: str, first: date, last: date) -> dict:
    """{day: FileName} on ESA's server for days ``first``..``last``."""
    names = []
    for folder in folders_for(first, last):
        names += listing_names(_get(f"{BASE_URL}/{mission}/{folder}/").decode("utf-8", "replace"))
        time.sleep(PAUSE)
    return {d: f for d, f in newest_per_day(names).items() if first <= d <= last}


def fetch(mission: str, first: date, last: date, dry_run: bool = False) -> list[date]:
    """Download ``mission``'s missing days; returns the days ESA has no file for."""
    folder = mission_dir(mission)
    have = local_files(mission)
    remote = remote_files(mission, first, last)
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    missing = [d for d in days if d not in remote]
    todo = [remote[d] for d in days if d in remote
            and (d not in have or have[d].name != remote[d].name)]
    print(f"{mission}: {len(days)} days, {len(days) - len(todo) - len(missing)} already "
          f"here, {len(todo)} to fetch, {len(missing)} not on the server", flush=True)
    if dry_run:
        return missing
    folder.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(todo, 1):
        data = _get(f"{BASE_URL}/{mission}/{f.folder}/{f.name}")
        check_zip(f.name, data)
        tmp = folder / (f.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(folder / f.name)
        old = have.get(f.day)
        if old is not None and old.name != f.name:     # ESA reissued the day
            old.unlink()
        if k % 25 == 0 or k == len(todo):
            print(f"  {mission} {k}/{len(todo)} (up to {f.day})", flush=True)
        time.sleep(PAUSE)
    return missing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sats", nargs="+", default=["S1A", "S1C"])
    p.add_argument("--from", dest="first", type=date.fromisoformat, default=FIRST)
    p.add_argument("--to", dest="last", type=date.fromisoformat, default=LAST)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.last < a.first:
        p.error("--to is before --from")
    print(f"into {mission_dir(a.sats[0]).parent}", flush=True)
    gaps = {s: fetch(s, a.first, a.last, a.dry_run) for s in a.sats}
    for s, days in gaps.items():
        if days:
            print(f"{s}: no file on ESA's server for {len(days)} day(s): "
                  + ", ".join(map(str, days)))
    return 1 if any(gaps.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
