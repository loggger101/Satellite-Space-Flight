# Sentinel-1A and 1C from ESA precise orbits (POEORB) - plan

Drafted 2026-11-18. Decisions settled with the user the same day. Source:
<https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/>. The project's main
goal is to map Sentinel-1A and 1C accurately; these files are the ground truth.
**Nothing is downloaded until the user approves this plan.**

## 1. The data (checked against the live folder listings, 2026-11-18)

- **Layout:** `POEORB/<S1A|S1C>/YYYY/MM/`, sorted by the month the file's
  coverage starts. Year folders exist up to 2028 but are empty placeholders.
- **File names:** `S1A_OPER_AUX_POEORB_OPOD_<created>_V<start>_<stop>.EOF.zip`.
  Example: `..._20260720T070517_V20260629T225942_20260701T005942.EOF.zip`.
- **Coverage:** each file covers 26 h, from 22:59:42 UTC the day before to
  00:59:42 UTC the day after. Day D's file therefore sits in the folder of
  D-1: the file covering 1 July is in the June folder. Files are published
  about 20 days after the fact.
- **Size:** about 600 KB zipped per file.
- **Coverage by satellite:**
  - S1A runs to **2026-06-30**: its last file covers 29-30 June, and there
    are no files from July 2026 on.
  - S1C starts in March 2025.
  - S1D has files from January 2026. It is out of scope.
- **Contents:** Earth Explorer XML. Each OSV (orbit state vector, one every
  10 s, about 9,361 per file) has:
  - TAI, UTC and UT1 times;
  - the absolute orbit number;
  - X, Y, Z in m and VX, VY, VZ in m/s, in the Earth-fixed ITRF frame;
  - a quality flag.
  The exact tag names are confirmed on the first file.
- **Licence:** free and open Copernicus data. Credit it as "Contains modified
  Copernicus Sentinel data [year]". Raw files are never committed.

## 2. Settled decisions

| Topic | Decision |
|---|---|
| Satellites and span | S1A and S1C over their **common span, 2025-07-01 to 2026-06-30**: 365 days x 2 = 730 files, about 0.45 GB. S1D is left out. |
| Storage | **Outside OneDrive**, at `%SATFLIGHT_DATA%\poeorb\<S1A|S1C>\` (default `~/satflight-data`, so `C:\Users\Loggg\satflight-data\poeorb`). An environment variable moves it on the other PC. The repo keeps only small derived results. |
| Download | The **first implementation step**, once this plan is approved. One file at a time, with a timeout and a polite delay. Resumable: files already present are skipped. A missing day is reported, never substituted. |
| In-app replay | The scenario **references the data folder** (satellite and date range) and interpolates the 10 s points (Hermite). If the data is missing, it says so; there is no fallback. |
| Replay view | **Truth and a simulated twin side by side**: the twin starts from the same OSV with the fitted Cd and Cr, and the radial, along-track and cross-track gap is shown live. |
| Gravity | Add **EGM2008, normalized, tide-free, from ICGEM**, cut to degree and order 70 and committed (about 100 KB). The download happens when implementation starts. **Each scenario chooses its degree** (0 = off): Sentinel-1 uses 70, and the demos keep J2-J4 (+C22) and their speed. |
| Sun, Moon, SRP | **Brought back** (from git `c079ded`): third-body Sun and Moon, and cannonball SRP with eclipse. They are per-scenario switches, on for Sentinel-1 and off for the demos. `docs/SCOPE.md` is updated to match. |
| Cd, Cr | **Fitted from the truth** per arc, together with the initial state, by least squares. The app uses the mean over the year. |
| Atmosphere | **Keep the exponential table and fit its scale** per arc (no space-weather feed). The study plots the fitted scale over the year. |
| Manoeuvres | **Detect, list, model.** Burns are found from steps in the mean a and i between arcs. The list holds each burn's time and delta-v estimate. Accuracy arcs that cross a burn apply it as an impulse, fitted from the truth. |
| Accuracy arcs | **Weekly starts, 1-day fit, checked to 3 days**: about 52 per satellite, about 200 fits in all, a few hours in the background. |
| Reports | **A script plus Markdown and PNGs**: `tools/` regenerates `docs/analysis/sentinel1/` (an accuracy report and a long-term study). Small derived tables, such as daily mean elements and the burn list, are committed as CSV. |
| MATLAB | The **study results** go out as CSV/MAT with a plotting function, and the **Export MATLAB live script works for replay scenarios**. EGM2008, Sun/Moon and SRP are **ported to `matlab_runtime.m`**, so MATLAB keeps agreeing with Python. |

## 3. Frames and time (no user decision needed)

- **ITRF to the simulator's frame:** the simulator's ECI is an of-date frame
  (ECEF turned by GMST, `frames.py`). ITRF goes to ECI through
  `ecef_to_eci_state` at the GMST of the OSV's **UT1**. No precession or
  nutation is needed. Going through J2000/GCRF would be wrong by about 45 km.
- **UT1:** `Clock` gains `dut1` (UT1-UTC, from the files) so the Earth's
  rotation angle is exact. Today the clock treats UT1 as UTC, which is up to
  0.4 km off on the ground track.
- **Comparisons** are made in ECEF/RSW, the frame the data is in, so the
  GMST-versus-GAST convention cancels.
- **Polar motion** (about 10 m) is neglected, and this is said in the report.
  It is the next refinement if the residual after the fit gets near that level.
- **Sun and Moon positions** use the low-precision series, taken as of-date.
  That is ample for perturbations.

## 4. Work order (separate PRs)

1. **Data.**
   - `satflight/poeorb.py`, the reader: `.EOF.zip` or `.EOF`, standard library
     only, failing loudly on any validation rule. The rules are file type, an
     Earth-fixed frame, the count, a 10 s monotonic step, the |r| and |v|
     bounds, quality, |dut1| under 0.9 s, and a constant TAI-UTC.
   - Stitching of the overlapping days.
   - `tools/fetch_poeorb.py` and the 730-file download.
   - Offline tests on a small fixture cut from a real file, with each rule
     shown to fail on a planted defect.
   - `Clock.dut1`.
2. **Forces.**
   - EGM2008 to 70x70 (a normalized recursion, vectorized over satellites).
   - Sun and Moon third-body gravity, and SRP with Cr.
   - Each one is a per-scenario switch.
   - Tests against published values and against conserved quantities, and
     `docs/SCOPE.md` and `docs/PHYSICS.md` updated.
3. **Fitting and burns.**
   - Batch least squares for the state, Cd and Cr (finite-difference partials
     flown as extra satellites in one vectorized run).
   - Burn detection and impulse fitting.
4. **Replay in the app.**
   - An ephemeris-replay orbit type and the truth-plus-twin view with the
     live RSW gap.
   - A Sentinel-1 replay scenario in `MISSION` on the start menu.
5. **Study and report.**
   - Accuracy arcs, the long-term series (LTAN, i, a, e and argp, the
     12-day repeat, 1A-1C phasing, burns) and the Markdown report.
   - `tests/test_sentinel1.py` moved onto the data.
6. **MATLAB.**
   - The port of the new forces and replay to `matlab_runtime.m`.
   - Study results as CSV/MAT plus a plotting function.

## 5. Open, to settle when it comes up

- The exact XML tag names, and whether TAI-UTC ever changes inside a file.
  This is checked on the first download.
- Sentinel-1's drag and SRP reference areas. They only scale the fitted Cd
  and Cr, so the fits absorb them; nominal starting values are about 10 m^2
  for drag and 25 m^2 for SRP.
- Whether the residual after fitting calls for polar motion or a better
  atmosphere. The report shows the error budget first.
