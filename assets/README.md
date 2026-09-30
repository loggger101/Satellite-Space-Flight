# Assets

The Earth texture and coastlines are public domain and bundled with the
repository; `python tools/fetch_assets.py` re-downloads them.

- `earth.jpg` - NASA Visible Earth "Blue Marble" (land_shallow_topo_2048),
  equirectangular, 2048 x 1024. Credit: NASA Goddard Space Flight Center.
- `coastlines.json` - Natural Earth 1:110m coastlines, converted to
  `[lon, lat]` polylines.

- `fonts/` - DejaVu Sans, DejaVu Sans Bold and DejaVu Sans Mono (free license
  in `fonts/LICENSE_DEJAVU`). The UI uses Windows' Segoe UI and Consolas where
  they are installed and these everywhere else (Linux, macOS, CI), so text
  widths are known on every system; `SATFLIGHT_FONTS=bundled` forces them.

Any equirectangular image named `earth.jpg` or `earth.png` can replace the
texture (oceans are detected by color for the sun-glint effect). Without
these files the simulator draws a procedural globe with a latitude/longitude
grid.
