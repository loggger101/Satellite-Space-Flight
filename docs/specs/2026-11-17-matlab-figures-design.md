# MATLAB export: theme-aware figures, orbit transfers, diagnostics, 3-D animation, saving

Date: 2026-11-17. Approved in chat ("Approve design").

## Goal

The exported live script (`satflight/matlab_export.py` + `satflight/matlab_runtime.m`) should
produce output that is as detailed as the run allows, with figures of high quality in
whatever theme MATLAB uses, and animations where they help: above all for orbit transfers,
the project's main subject.

User choices: figures follow MATLAB's theme; add orbit-transfer figures, a 3-D animation and
physics diagnostics; save figures as PNG + PDF and animations as MP4 (plus the existing GIF).
Not chosen: a textured Earth.

## Problems found in the current output (GTO to GEO, 2 days, rendered in -batch)

- Colours are hard-coded. In a dark theme the ground-track night patch (tinted from
  `ax.Color` towards a pale blue) is drawn bright and the day side is the black axes, so day
  and night look swapped; the animation figure is white with pale grey text.
- The transfer (first 5 h) is one vertical step in 48 h element plots; the 3-D plot has no
  burns, transfer arcs or before/after orbits.
- The RAAN y-label is clipped.
- Figures cannot be saved as files except via the GIF.

## Design

Everything new lives in `matlab_runtime.m` (appended to every script; the script still needs
MATLAB alone, no toolbox) and in new sections written by `matlab_export.py`. The Python
simulator does not change, so the engine-port rule is not triggered. `T` and `events` stay
line-for-line what the simulator writes (the cross-check depends on it).

### 1. Theme

- `satflightTheme(fig)` returns a palette from the figure's theme (`fig.Theme`, light or dark):
  text, muted text, background, axes background, grid, night shade, Earth fill, coastline,
  station marker, Sun marker.
- Every hard-coded colour in plotting code uses the palette. The night shade is darker than
  the axes background in both themes. The animation figure follows the theme.
- Fix the clipped elements label; consistent fonts and label sizes.

### 2. Orbit transfers (section shown when any burn happens)

- `satflightRun` returns a fourth output `burns`: per burn (impulse, apsis-timed,
  finite), the object, label, start and end time, the state and mass just before the start
  and just after the end, and the dV vector split into V/N/B. Recorded where the engine
  executes the burn, so states are exact, not sampled.
- `satflightTransfers(burns)`: table, one row per burn: time, object, label, dV total and
  V/N/B parts (m/s), perigee/apogee altitude, inclination and period before and after,
  propellant used (kg).
- `satflightPlotTransfer(T, burns, names)`: per object that burns, a 3-D view (the orbit
  before the first burn dashed, each coast arc between burns, the final orbit, burn points
  labelled with dV, thrust arrows) and an edge-on view of the plane so plane changes show.
- `satflightPlotDeltaV(burns)`: stacked V/N/B bars per burn, cumulative dV line, mass.
- `satflightPlotBurnWindow(T, burns, names)`: perigee, apogee, inclination and a over the
  span holding the burns (with margin), burns marked.

### 3. Physics diagnostics

- `satflightDiagnostics(T, sim)`: per sample specific energy, |h|, h_z, speed, radius,
  period; per enabled force (central gravity, J2, J3, J4, C22, drag) its acceleration
  magnitude, from the ported force terms.
- `satflightPlotDiagnostics`: energy/momentum/speed/period panels with burns marked; force
  magnitudes on a log scale.
- Conservation table: relative drift of energy and h_z over each coast (between burns, in
  orbit). Two-body coasts must hold energy to integrator tolerance; drag shows a steady loss.

### 4. 3-D animation

- `satflightAnimate3D(T, events, burns, names, opts)`: turning Earth with coastlines in theme
  colours, fading trails over the last orbit, a flash and thrust arrow while a burn fires,
  event-log ticker.
- Adaptive time: frame times dense around burns and events, sparse over long coasts.
- Plays as the existing animation does (Live Editor playback; about 20 s in a figure
  window; batch draws the last frame only unless writing a file).

### 5. Saving

- Settings: `figureFolder = ""` saves every figure as 300 dpi PNG and vector PDF named after
  the figure, and the animations as MP4 there; `animationFile` accepts `.gif` or `.mp4`.
- `satflightSaveFigures(folder)`; the Save section lists the files written.

### 6. Verification

- Every figure rendered in light and dark: each text object reaches a 4.5:1 contrast ratio
  against what it sits on; the night shade is darker than day. A planted defect must fail it.
- `transfers` against the simulator: dV within 0.1 m/s of the event log; before/after orbits
  within 1 km of the Python sim at those instants; GTO to GEO apogee kick 1837.6 m/s.
- Diagnostics: energy drift over a two-body coast below 1e-8 relative.
- Saved PNG/PDF/MP4 exist; MP4 has the expected frame count and size.
- `codeIssues` zero issues on every export (existing check); rendered PNGs inspected by eye.
- New MATLAB tests run in their own sessions, each inside the 280 s per-session limit.
