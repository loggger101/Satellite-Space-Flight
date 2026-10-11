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
    """Download ``url`` (60 s timeout)."""
    req = urllib.request.Request(url, headers={"User-Agent": "satflight-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def check_texture(data: bytes) -> tuple[int, int]:
    """Size of the image in ``data``; ValueError if it is not one (a server's
    error page, a cut-off download) - checked before anything is written."""
    import io

    import pygame
    try:
        img = pygame.image.load(io.BytesIO(data), "earth.jpg")
    except pygame.error as exc:
        raise ValueError(f"the download is not an image ({exc}); nothing was saved") from exc
    w, h = img.get_size()
    if w != 2 * h:
        raise ValueError(f"the texture is {w}x{h}, not an equirectangular 2:1 map")
    return w, h


def coastlines_from_geojson(gj: dict) -> list:
    """[lon, lat] polylines from a GeoJSON FeatureCollection of LineStrings and
    MultiLineStrings; any other geometry is an error, not skipped."""
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            raise ValueError(f"unexpected geometry {geom['type']!r} in the coastline file")
        for part in parts:
            lines.append([[round(x, 3), round(y, 3)] for x, y in part])
    if not lines:
        raise ValueError("the coastline file has no lines")
    return lines


def fetch_texture():
    """Save the Blue Marble texture as ``assets/earth.jpg``."""
    data = _get(TEXTURE_URL)
    w, h = check_texture(data)
    (ASSETS / "earth.jpg").write_bytes(data)
    print(f"earth.jpg        {len(data) / 1e6:.1f} MB, {w}x{h}")


def fetch_coastlines():
    """Convert the Natural Earth GeoJSON to ``assets/coastlines.json`` polylines."""
    lines = coastlines_from_geojson(json.loads(_get(COAST_URL)))
    text = json.dumps(lines, separators=(",", ":"))
    (ASSETS / "coastlines.json").write_text(text, encoding="utf-8")
    print(f"coastlines.json  {len(lines)} polylines, {sum(len(p) for p in lines)} points")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    ok = True
    for fn in (fetch_texture, fetch_coastlines):
        try:
            fn()
        except (OSError, ValueError) as exc:   # network or content: reported, exit 1
            ok = False
            print(f"{fn.__name__} failed: {exc}", file=sys.stderr)
    sys.exit(0 if ok else 1)
