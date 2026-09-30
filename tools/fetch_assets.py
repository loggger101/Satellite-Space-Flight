"""Download optional public-domain Earth assets into ``assets/``.

    python tools/fetch_assets.py

* ``assets/earth.jpg``        NASA Visible Earth "Blue Marble" (2048 x 1024,
                              public domain) - textures the 3-D globe and the map.
* ``assets/coastlines.json``  Natural Earth 1:110m coastlines (public domain),
                              converted to a list of [lon, lat] polylines.

The simulator works without them (it draws a procedural globe).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

TEXTURE_URL = ("https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
               "land_shallow_topo_2048.jpg")
COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
             "geojson/ne_110m_coastline.geojson")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "satflight-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_texture():
    data = _get(TEXTURE_URL)
    (ASSETS / "earth.jpg").write_bytes(data)
    print(f"earth.jpg        {len(data) / 1e6:.1f} MB")


def fetch_coastlines():
    gj = json.loads(_get(COAST_URL))
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for part in parts:
            lines.append([[round(x, 3), round(y, 3)] for x, y in part])
    (ASSETS / "coastlines.json").write_text(json.dumps(lines, separators=(",", ":")), encoding="utf-8")
    print(f"coastlines.json  {len(lines)} polylines, {sum(len(p) for p in lines)} points")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    ok = True
    for fn in (fetch_texture, fetch_coastlines):
        try:
            fn()
        except Exception as exc:  # network problems should not be fatal
            ok = False
            print(f"{fn.__name__} failed: {exc}", file=sys.stderr)
    sys.exit(0 if ok else 1)
