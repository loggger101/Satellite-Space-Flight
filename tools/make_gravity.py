"""Cut ICGEM's EGM2008 down to the degree the simulator ships.

    python tools/make_gravity.py [path/to/EGM2008.zip] [--degree 70]

The full model (72 MB zipped, degree 2190, fully normalized, tide-free) comes
from ICGEM: https://icgem.gfz.de/tom_longtime (EGM2008 row, "zip"). It is not
kept in the repository; this writes ``satflight/data/egm2008_<N>.gfc`` with
the same header, the coefficients up to degree N, and a note of where they
came from (the source file's SHA-256), so the cut can be checked and redone.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satflight.poeorb import data_root  # noqa: E402

SOURCE_URL = ("https://icgem.gfz.de/getmodel/zip/c50128797a9cb62e936337c890e4425f03f0461d7329b"
              "09a8cc8561504465340/EGM2008.zip")


def cut(zip_path: Path, degree: int) -> str:
    """The text of the truncated .gfc file."""
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".gfc")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected one .gfc inside, found {names}")
        head, rows, in_body = [], [], False
        with z.open(names[0]) as f:
            for raw in f:
                line = raw.decode("ascii").rstrip()
                if not in_body:
                    if line.startswith("max_degree"):
                        line = f"max_degree                  {degree}"
                    head.append(line)
                    in_body = line.startswith("end_of_head")
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] != "gfc":
                    raise ValueError(f"unexpected line in the body: {line[:60]!r}")
                n = int(parts[1])
                if n > degree:
                    break                    # ordered by degree: the rest is higher
                rows.append(line)
    # every (n, m) up to the degree once, except degree 1: a geocentric field
    # has none (C10 = C11 = S11 = 0), and EGM2008 leaves those rows out
    got = sorted((int(r.split()[1]), int(r.split()[2])) for r in rows)
    want = [(n, m) for n in range(degree + 1) if n != 1 for m in range(n + 1)]
    if got != want:
        raise ValueError(f"degree/order rows up to {degree} are not complete and unique "
                         f"({len(got)} rows, expected {len(want)})")
    note = [f"comment  cut to degree {degree} by tools/make_gravity.py from ICGEM's EGM2008.zip",
            f"comment  source {SOURCE_URL}",
            f"comment  source sha256 {digest}"]
    return "\n".join(note + head + rows) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("zip", nargs="?", type=Path, default=data_root() / "gravity" / "EGM2008.zip")
    p.add_argument("--degree", type=int, default=70)
    a = p.parse_args(argv)
    out = ROOT / "satflight" / "data" / f"egm2008_{a.degree}.gfc"
    out.parent.mkdir(exist_ok=True)
    out.write_text(cut(a.zip, a.degree), encoding="ascii", newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
