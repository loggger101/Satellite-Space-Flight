# Optional assets

Run `python tools/fetch_assets.py` to download (both public domain):

- `earth.jpg` - NASA Visible Earth "Blue Marble" equirectangular texture
- `coastlines.json` - Natural Earth 1:110m coastlines as `[lon, lat]` polylines

Any equirectangular image named `earth.jpg` or `earth.png` works. Without these
files the simulator draws a procedural globe with a latitude/longitude grid.
