# Sentinel-1 panel model for drag - design probe

**Date:** 2026-11-24. **Branch:** `panel-model` (off `physical-cd`).
**Roadmap item 2** of the drag work (memory: drag roadmap after MSIS).
**Outcome so far:** the probe says a drag panel model will not move the
Sentinel-1 arcs, so nothing is built yet. The decision is the user's (section 5).

## 1. The published Sentinel-1 macro-model

Two sources, both read in full text:

- GMV, *Sentinel-1 properties for GPS POD*, GMV-GMESPOD-TN-0025 v1.4,
  2019-09-16 (Copernicus POD Service). Table 5-3 gives eight panels with
  areas and thermo-optical coefficients; Table 5-1 gives the projected
  along-track area: 5.57 m^2 at 0 deg yaw steering, 6.12 at -3.9 deg, 8.15 at
  +3.9 deg, 6.35 averaged over an orbit. Body axes (section 2.1): X along the
  SAR antenna and the flight direction, Y away from the Sun (normal to the
  faces the arrays stow on), Z along the radar boresight. Nominal attitude is
  the zero-Doppler frame plus a roll about X (the roll-steering law); the
  attitude code is only in figures.
- Peter, Fernandez, Femenias, *Copernicus Sentinel-1 satellites: sensitivity
  of antenna offset estimation to orbit and observation modelling*, Adv.
  Geosci. 50, 87-100, 2020, doi:10.5194/adgeo-50-87-2020. Table 4 adds the
  body-frame normals (same areas as GMV):

| # | panel | normal (body) | area m^2 |
|---|---|---|---|
| 1 | +x | (+1, 0, 0) | 5.570 |
| 2 | -x | (-1, 0, 0) | 5.570 |
| 3 | +y | (0, +1, 0) | 6.790 |
| 4 | -y | (0, -1, 0) | 6.790 |
| 5 | +z (SAR front) | (0, 0, +1) | 12.550 |
| 6 | -z (SAR back) | (0, 0, -1) | 12.550 |
| 7 | solar array front | (0, -0.866, -0.500) | 34.456 |
| 8 | solar array back | (0, +0.866, +0.500) | 34.456 |

  (Their table prints panel 2's normal as +1; it is the -x face.) The arrays
  are **fixed** at 30 deg to the body Z axis, not sun-tracking, and the bus is
  rolled about 30 deg off nadir. The old macro-model had 4.33 m^2 for +-y and
  31.0 m^2 for the arrays; the SAR antenna's 0.2 m thickness and the arrays'
  triangular mountings (about 10% of their area) were added. Area uncertainty
  is about 5%. The POD solutions use MSISE90 and estimate 15 drag
  coefficients per arc.

The arrays' normals have no X component, so with the body X along the flow
they are edge-on, as are the SAR antenna and the +-y faces. Only the +x face
(5.57 m^2) meets the flow head-on.

## 2. Probe: what a panel model would change

`panel_probe.py` (scratchpad) sums Sentman one-sided flat plates over the
eight panels, Cd per plate
`P/sqrt(pi) + g Q Z + (g/2)(vr/vi)(g sqrt(pi) Z + P)` with
`P = exp(-g^2 s^2)/s`, `Q = 1 + 1/(2 s^2)`, `Z = 1 + erf(g s)`, `g` the cosine
between the inward normal and the flow, the same species mixture, SESAM
accommodation and rms reemission as `drag_coefficient.py`. Flow along -X
(yaw 0) or yawed 3.9 deg; grid mean at 693 km, density weighted, 7.5 km/s;
no shadowing between panels.

| arc start | Cd*A, sphere (Cd x 10 m^2) | Cd*A, panels yaw 0 / 3.9 deg |
|---|---|---|
| 2025-08-05 | 31.9 | 30.9 / 32.2 |
| 2025-11-04 | 31.5 | 30.6 / 31.8 |
| 2025-11-11 | 31.2 | 30.3 / 31.5 |
| 2025-12-09 | 31.2 | 30.4 / 31.6 |
| 2026-01-20 | 30.3 | 29.4 / 30.6 |

Day-to-day change against the fit day, for the panels and the sphere:

| arc | day 1 | day 2 | day 3 |
|---|---|---|---|
| 2025-11-04 | -1.2% / -1.1% | -1.7% / -1.6% | -2.1% / -2.0% |
| 2025-11-11 | -2.8% / -3.1% | -1.6% / -1.6% | +1.2% / +1.0% |
| 2026-01-20 | +0.4% / +0.6% | +1.9% / +2.5% | +1.4% / +2.0% |
| 2025-12-09 | -0.8% / -0.6% | -1.2% / -0.8% | -0.7% / -0.2% |
| 2025-08-05 | -0.2% / -0.2% | +0.1% / -0.1% | -0.6% / -0.4% |

Two findings:

1. **The level is right.** About half the panel Cd*A is the head-on +x face
   (Cd about 2.7 on 5.57 m^2); the rest is shear on the edge-on surfaces,
   chiefly the two array faces (2 x 34.5 m^2 at about 0.15 each). That is why
   the 10 m^2 cannonball with a fitted Cd of 2.7-2.8 worked: the arrays add
   drag edge-on.
2. **The time variation is the sphere's.** The panels move with the sphere
   within 0.6 percentage points on every day of every arc. The fits absorb
   the level, so a panel model would fly the arcs like the physical Cd did:
   within 10%, slightly worse in storms (PHYSICS.md, physical drag
   coefficient). Yaw steering adds about 4% once per orbit, which averages out
   over a day.

So the drag panel model fails the proof bar before it is built. What it would
add is a physical Cd*A level (about 30 m^2) without fitting, which the arcs do
not use.

## 3. Where the remaining error is

The MSIS fits with the physical Cd want a density correction of 0.54-0.98:
the physical Cd*A of about 31 m^2 times 0.54-0.98 gives 17-30 m^2. Either
NRLMSIS 2.1 runs heavy at 693 km in quiet weather and light in the January
2026 storm, or the panel Cd*A is high (wake shadowing of the arrays by the bus,
lower accommodation). The storm arcs' error comes from the density changing
after the fit day, which no Cd model sees.

## 4. Sunlight pressure (not drag)

The same panels matter more for sunlight pressure. The arrays face the Sun
(normal 0.866 along -Y, the sun side), about 30 m^2 projected. The study
flies SRP with Cr held at 1.3 on the 10 m^2 drag area (Cr*A 13 m^2), so SRP
there is probably underestimated two to three times. (The app's default.json
has SRP off.) In the dawn-dusk orbit (Sun 58-87 deg from the
orbit plane) that force is mostly cross-track and the in-plane part is
periodic, so its along-track effect is small, except in the May-August eclipse
season. The quiet summer arc (1A 2025-08-05, 22 m at day 3, the only one MSIS
did not improve) is in that season. This has not been measured.

## 5. Options for the user

- **A. Skip the drag panel model; go to thermospheric winds (item 3).** Winds
  change the speed through the air directly, and storm-time winds are
  strongest when the storm arcs fail.
- **B. Build an SRP panel model instead** (fixed arrays, eight panels, the
  GMV optical coefficients), checked against the summer arc. It is a new
  item, outside the agreed roadmap.
- **C. Build the drag panel model anyway** as an opt-in, for the physical
  level and as the base for B. Expected result: no change to the arcs.
- **D. Storm-time density** (audit D36, calibrating density from tracking):
  aimed at the actual storm error, but a larger job.
