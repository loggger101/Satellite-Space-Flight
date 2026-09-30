# Assets

Both files are public domain and bundled with the repository;
`python tools/fetch_assets.py` re-downloads them.

- `earth.jpg` - NASA Visible Earth "Blue Marble" (land_shallow_topo_2048),
  equirectangular, 2048 x 1024. Credit: NASA Goddard Space Flight Center.
- `coastlines.json` - Natural Earth 1:110m coastlines, converted to
  `[lon, lat]` polylines.

Any equirectangular image named `earth.jpg` or `earth.png` can replace the
texture (oceans are detected by colour for the sun-glint effect). Without
these files the simulator draws a procedural globe with a latitude/longitude
grid.
