# Sentinel-1A and 1C from ESA precise orbits (POEORB) - plan

**Status (2026-11-26): every step is done, and the project is complete**
(see the README's status note and [../SCOPE.md](../SCOPE.md)). The plan is kept as
written, with one change since: on 2026-11-25 the study was cut to
2025-07-01 .. 2026-05-31, before ESA moved 1C to trail 1A by 207 deg in June
2026. Section 4's study figures (102 arcs, the year's means) come from the
full year. [../analysis/sentinel1/report.md](../analysis/sentinel1/report.md)
has the study as it stands (96 arcs).

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
- **Found in the first files (step 1):**
  - The tag names are as expected, with no XML namespace. Each file holds
    9,361 OSVs from 22:59:42 to 00:59:42, so samples fall on :02, :12, ...
    past each minute. TAI-UTC is 37 s. UT1-UTC was about +0.043 s in
    July 2025.
  - The quality flag can also be **`DEGRADED-MANOEUVRE`**, marking a window of
    about 61 samples (10 min) around each orbit-keeping burn, every 1-3 days.
    These samples are kept and flagged: they locate the burns for step 3.
    During a burn, the velocity departs from the positions' rate by up to
    22 m/s, so the velocity check skips those windows. Elsewhere the velocity
    agrees with the positions to within 0.15 m/s.
  - A third flag, **`DEGRADED-GAP`**, marks stretches with no tracking data
    that ESA bridged with its orbit model. They run from minutes to about
    11 h, in 5 S1A files and 9 S1C files. These samples are kept and flagged
    too, and the velocity check still applies to them.
  - **Whole year:** all 720 files pass every check. S1A is one unbroken
    365-day arc. S1C has **no POEORB (or RESORB) for 2026-06-09..18**, an
    outage, so it splits into two arcs and no fit crosses the outage.
    UT1-UTC ran from +0.011 to +0.095 s.
  - **Seams:** the 718 day-to-day overlaps agree to a median of 2 mm, and 95%
    are under 23 mm. Seven differ by 0.6-5.7 m with no burn, which is genuine
    disagreement between daily solutions. The limit is 10 m, and every seam's
    value is kept for the report.
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
- **Polar motion** is taken out of the truth (added 2026-11-19, step 3): the
  ITRF's z axis sits 0.3-0.5 arcsec off the spin axis, about 15 m at
  Sentinel-1's height. Left in, the simulator's frame wobbles by that once a
  day and a day's fit has ~8 m of cross-track error it cannot follow (with a
  negative Cd and a sideways burn soaking up the rest). `satflight/eop.py`
  reads the IERS `finals2000A.all` table (measured rows only;
  `tools/fetch_eop.py`, into `$SATFLIGHT_DATA/eop/`) and `poeorb.to_eci`
  applies W = R2(x_p) R1(y_p) before turning the Earth. The table's UT1-UTC
  agrees with the files' own to 0.2 ms. The simulator's gravity field itself
  still turns about the spin axis: that tilt changes the acceleration by under
  1e-7 m/s^2, centimetres a revolution.
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
   **Done 2026-11-18 (branch `forces-egm2008-sun-moon`):**
   - EGM2008 is cut to degree 70 from ICGEM's file (SHA-256 recorded in the
     header).
   - The field is the gradient of an independent Legendre potential to 5e-10.
     The Legendre functions are orthonormal to degree 70. A 70x70 evaluation
     takes about 1.6 ms.
   - Starting from one S1A precise state (2025-07-03), the gap from the truth
     after 6 h is 938 m with J2-J4 and 49 m with EGM70 + drag + Sun + Moon +
     SRP. After 24 h it is 670 m and 91 m (mostly along-track, which step 3
     fits). Cross-track after 24 h falls from 16 m to 3.4 m with the Sun and
     Moon.
   - `default.json` keeps J2-J4 + drag until step 4's data-based scenario
     replaces it. Switching it now would make the MATLAB export, which
     refuses the new forces until step 6, refuse the default scenario.
3. **Fitting and burns.**
   - Batch least squares for the state, Cd and Cr (finite-difference partials
     flown as extra satellites in one vectorized run).
   - Burn detection and impulse fitting.
   **Done 2026-11-19 (branch `fit-and-burns`):**
   - `satflight/fit.py`: Gauss-Newton on the start state, Cd*A/m, Cr*A/m and
     an impulse at the middle of each flagged burn window; the perturbed
     copies fly as rows of one batch. Any of them can be held; holding all but
     later burns is how a prediction meets the operators' next manoeuvres.
     `tools/fit_arc.py` runs it on a day (`--check H` flies H hours on).
   - Synthetic truth flown with the simulator's own forces gives back its
     state to under 1 m, its drag to 3 % and a 4 mm/s burn to 2 %.
   - Real S1A days, degree 30, 120 s samples, with polar motion:
     | arc | fit RMS | R / S / W | note |
     |---|---|---|---|
     | 2025-07-03, 12 h | 2.08 m | 0.89 / 1.71 / 0.76 m | without the pole: 5.83 m, W 5.25 m |
     | 2025-07-03, 24 h | 2.31 m | 0.74 / 1.53 / 1.56 m | Cd 3.15, Cr 5.7 at 10 m^2 |
     | 2025-07-05, 24 h, one burn | 3.76 m | 1.36 / 3.27 / 1.23 m | without the pole: 9.25 m and no convergence |
   - Flying the 2025-07-03 fit a further day against the truth: along-track
     RMS 72 m (worst 132 m), radial 1.7 m, cross-track 3.2 m. The fitted Cd
     and Cr (3.15 and 5.7, Cr above anything physical) are absorbing the
     day's density change, which the static table cannot follow near solar
     maximum. The along-track prediction is the error to measure in step 5's
     weekly arcs, with the table's density scale fitted.
   - The 2025-07-05 burn (a 10-minute window) fits as 2.0 / 4.1 / 3.0 mm/s
     R/S/W; its 3.3 m along-track residual is likely the impulse standing in
     for a finite burn. Finite burns stay open (section 5).
4. **Replay in the app.**
   - An ephemeris-replay orbit type and the truth-plus-twin view with the
     live RSW gap.
   - A Sentinel-1 replay scenario in `MISSION` on the start menu.
   **Done 2026-11-19 (branch `replay-in-app`):**
   - `satflight/replay.py`: orbit type `replay` (mission, `until` date) follows
     the recording, cubic Hermite between the 10 s samples (under 1 mm, 0.1
     mm/s), kept out of the integration; type `precise` starts a twin on the
     recorded state, flown by the force model. Both take out polar motion. A
     start at midnight takes the 2 s before the day's first sample from the day
     before. Missing files name the first missing day; the truth stops with
     status "end of data" when its recording ends.
   - The truth gets eclipses and ground contacts but no close approaches (its
     twin is metres away on purpose); it cannot burn; snapshots keep it a
     replay; the MATLAB export refuses it until step 6.
   - `Simulation.gap_to_truth` gives a twin's radial / along-track /
     cross-track gap; the Telemetry tab shows it (PRECISE ORBIT) and the list
     shows each twin's distance.
   - `scenarios/data/sentinel1_replay.json` (generated; in its own folder so
     the checks that load every bundled scenario do not need the data): 1A and
     1C, 3-9 July 2025, EGM2008 70x70 + drag + Sun + Moon + SRP, twins with the
     Cd fitted to 3 July (degree 70, Cr held at 1.3: 1.64 m RMS, 6 m RMS
     along-track over the next day). Second card under "Sentinel-1 mission".
   - Twins against the truth: 1A 0.9 / 4.7 / 28 / 77 m after 1 / 6 / 24 / 48
     h, 1C 2.3 / 11 / 56 / 131 m, almost all along-track (radial and
     cross-track under 7 m for two days). Then S1A's 5 July trim, which the
     twin does not fly, opens 0.85 km by +72 h. A simulated day of all four at
     70x70 takes about 17 s.
   - The fit at degree 70 with Cr held predicts a day ten times better than
     degree 30 with Cr free (6 m against 72 m RMS along-track): free Cr was
     soaking up model error, not measuring sunlight pressure.
5. **Study and report.**
   - Accuracy arcs, the long-term series (LTAN, i, a, e and argp, the
     12-day repeat, 1A-1C phasing, burns) and the Markdown report.
   - `tests/test_sentinel1.py` moved onto the data.
   **Done 2026-11-20 (branch `study-and-report`):**
   - `satflight/study.py` measures; `tools/sentinel1_study.py series|arcs|report`
     writes `docs/analysis/sentinel1/` (daily CSVs per satellite, `burns.csv`,
     `phasing.csv`, `arcs.csv`, `report.md` and three PNGs).
   - Revolution means run node to node (ascending nodes from a Newton solve on
     the Hermite z), integrated exactly over the span: a plain mean of the
     samples is off by up to 17 m in a, since J2 swings a by 10 km twice a
     revolution.
   - The year (2025-07-01 .. 2026-06-30): mean a 7070.980 km (1A) and
     7071.005 km (1C), i 98.1805 and 98.1822 deg, LTAN 18:01.2 and 18:00.2,
     e 1.184e-3 with argp 90 deg (frozen; e swings 1.173-1.192e-3 over the
     year). Each equator crossing lands within about 24 m (median daily RMS)
     of its relative orbit's usual crossing, 300 m at worst; 175 revolutions
     in 12 days to 0.005 revolutions a day.
   - 1C flies 176.2 deg behind 1A, not the nominal 180 (checked directly from
     the argument of latitude on 2025-09-01: 176.16 deg). ESA has no 1C files
     for 2026-06-09..18; afterwards 1C was raised 1.9 km, lowered again on 23
     June, and has flown 206.7 deg behind, 229 km off its old track.
   - Burns: 91 manoeuvres (129 windows) for 1A, 101 (200) for 1C, about
     every 4 days. Each is sized from the step in the revolution means
     (drag decay and the daily wobble fitted out); placebo windows between
     burns put the noise at 1.3-1.6 mm/s along-track and 22 mm/s cross-track,
     so the series sizes the along-track part (median about 4 mm/s) but not
     the cross-track part.
   - Accuracy arcs: weekly, a 24 h fit at 60 s (state, density scale, an
     impulse per burn; Cr 1.3; EGM2008 70x70 + drag + Sun + Moon + SRP) flown
     72 h on with only the later burns fitted. A burn needs half an hour of
     data after its window (`fit.MIN_AFTER_S`; `fit_arc` refuses one without):
     at the fit's end it ends the fit where its window begins and moves to the
     prediction, at the check's end it ends the check. Without that rule four
     arcs ran away (one impulse grew to 2.9 km/s, fitted to noise). 102 arcs
     (52 for 1A, 50 for 1C; 1C's 2026-06-09 and 06-16 fall in ESA's gap); 11
     fits and 19 checks were cut short at a burn (shortest 23.4 h and 71.4 h).
     The fit day leaves 0.4 m radial, 0.8 m along-track and 1.4 m cross-track
     RMS, so impulses stand in for the finite burns well enough. Flown on,
     the median along-track RMS is 9 m on day 1, 18-19 m on day 2 and
     23-24 m on day 3 (90th percentile 32-60, 69-102 and 94-127 m); radial
     2-5 m, cross-track 5-13 m. The density scale averages 1.23 (1A) and
     1.27 (1C), 0.43-4.05 week by week, the two satellites within a few
     percent of each other: a mean Cd of 2.7-2.8 at 10 m^2.
   - The large errors are space weather. Against GFZ's Kp index (not part of
     the pipeline; checked by hand on 2026-11-22): 44 of the 102 arcs see
     Kp >= 5 somewhere in their four days, and those have a day-3 along-track
     median of 42 m (90th percentile 302 m) against 16 m (91 m) for the rest;
     7 of the 10 worst arcs are storm weeks (2025-11-11/12 Kp 8.7: 1.3 km;
     2026-01-19/20 Kp 7.7: 1.6 km for 1A).
   - `tests/test_sentinel1.py` checks the defaults against the committed
     CSVs: the 12-day repeat, LTAN, inclination, argp and the 176/207 deg
     phasing.
   - `default.json` now flies what the data show: 1C 176.3 deg behind 1A
     (the July 2025 - May 2026 mean) instead of 180, and the measured mean
     e 1.1837e-3 at argp 90 instead of the first-order J2 + J3 1.0443e-3.
     `tools/make_scenarios.py` finds each satellite's osculating start by flying
     it and correcting until its revolution-mean e vector matches (osculating
     1.2695e-3 at 68.9 deg for 1A, 1.1865e-3 at 112.0 deg for 1C: the J2 swing
     depends on where each starts). The repeat still closes to 10 ms a period.
6. **MATLAB.**
   - The port of the new forces and replay to `matlab_runtime.m`.
   - Study results as CSV/MAT plus a plotting function.

## 5. Open, to settle when it comes up

- Sentinel-1's drag and SRP reference areas. They only scale the fitted Cd
  and Cr, so the fits absorb them; nominal starting values are about 10 m^2
  for drag and 25 m^2 for SRP.
- A better atmosphere: done 2026-11-23 as NRLMSIS 2.1 driven by CelesTrak's
  measured space weather (`"atmosphere": "msis"`). Six arcs fitted again with it
  (`docs/analysis/sentinel1/arcs_msis.csv`) cut the three-day along-track error
  in all four storm weeks (1347 to 508 m, 1627 to 1139 m, 445 to 226 m, 269 to
  130 m) and in one quiet week (18 to 4 m); the other quiet week went from 19 to
  22 m. The fitted density scale narrows from 0.85-4.05 to 0.78-1.36
  (PHYSICS.md, Atmosphere). The note as written before it:
  polar motion is now in (section 3). The study measured
  the remaining error: 9 m along-track after a day and 24 m after three in
  quiet and moderate weeks, up to 1.6 km over three days when a storm starts
  after the fit. A density model driven by Kp / F10.7 (NRLMSIS, JB2008) is
  what would close that, and would need space-weather data the way polar
  motion needs IERS's.
- Finite burns: not needed so far. The fit day's 0.8 m along-track RMS with
  impulses shows no residual worth modelling around the burns.
