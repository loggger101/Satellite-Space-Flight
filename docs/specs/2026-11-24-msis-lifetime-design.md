# NRLMSIS for the lifetime estimate and the Orbit tab - design

**Date:** 2026-11-24. **Branch:** `msis-lifetime` (off `physical-cd`).
**Roadmap item 4** of the drag work. Proof bar for this item: checked against
reference values (not the Sentinel-1 arcs).

## 1. Done: the Orbit tab's perigee air

Under `"atmosphere": "msis"` the Orbit tab's "Perigee density" and "Drag at
perigee" are now NRLMSIS's at the perigee point at the run's time (the
Earth turned by the run's gmst), times `density_scale`, and lose the
"(table)" mark. `orbit_info(..., atmosphere="msis", gmst=...)` refuses to run
without gmst. Tests: the density equals `msis_density` at the perigee point
(tests/test_msis.py), and the app passes the run's atmosphere and gmst
(tests/test_ui_headless.py). Planted defects (the satellite's position instead
of the perigee point, a time 6 h off, the app passing "table") each fail.
No decision was needed: the run already has measured space weather for its
own time.

## 2. The lifetime estimate: the open question is the future Sun

`lifetime.py` steps a and e forward for years with orbit-averaged drag. MSIS
needs F10.7, its 81-day mean and ap for every day of that future. The
measured table ends at today, and the code refuses predicted days (no silent
fallbacks). So MSIS lifetimes need an explicit model of the future Sun
(audit D37), and that choice is the user's.

What is available:
- NOAA SWPC's predicted solar cycle
  (`services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json`):
  monthly predicted F10.7 with 25/50/75% high and low bands, 2026-05 to
  2030-12 (fetched 2026-11-24: 130.8 sfu in 2026-05 falling to 70.3 in
  2030-12). It stops at the end of cycle 25; Sentinel-1 at 693 km lasts
  decades.
- The measured table itself (CelesTrak SW-All, back to 1957): past cycles, from
  which a mean cycle shape and its spread can be built for the years after
  2030.

Options:
- **A. SWPC forecast, then repeated mean cycles (recommended).** SWPC's
  monthly F10.7 to 2030-12, then a mean cycle built from the measured cycles
  (11-year period, F10.7 by cycle phase), repeated. The "At solar max / min"
  rows become SWPC's high and low bands, then the measured cycles' spread.
  ap held at its long-term mean (no storms in a forecast), stated in the tab.
  A new data file and fetch step, like `tools/fetch_space_weather.py`, with
  the file's date shown and a stale forecast refused.
- **B. Today's Sun held.** The measured 81-day mean F10.7 and mean ap of the
  run's date, held for the whole lifetime ("if the Sun stays as it is").
  Simple, no new data, but wrong by about the solar-cycle swing for anything
  that lasts more than a few years.
- **C. Fixed bands.** Lifetimes at three constant levels (for example F10.7
  70 / 140 / 220 sfu with ap 15), shown as low / mean / high, replacing the
  table's x0.3 / x3 rows. No forecast, honest about the spread, no single
  answer.

## 3. How the estimate changes (any option)

- `decay_rates` samples the ring at a time: it needs the date, the RAAN and
  its J2 drift (local solar time drives the MSIS day/night bulge; a
  sun-synchronous orbit keeps its local time), and the Earth's turn at that
  date. The perigee directions are still averaged.
- MSIS is called with the chosen indices instead of the measured table:
  `atmosphere.msis_state` gains a path that takes (F10.7, F10.7a, ap) given by
  the caller, still refusing non-finite results.
- The step is also capped at a few days so the solar-cycle change is
  followed. Cost to measure before building: about 400 MSIS points per
  evaluation, two evaluations per step.
- The table estimate stays the default under the table atmosphere, unchanged.

## 4. Reference checks (the proof bar for this item)

1. **Against a run flown to re-entry under MSIS**, with a synthetic measured
   table held at constant indices (the test fixture's way), the estimate
   with the same indices lands within a few percent, as the table estimate
   does today (tests/test_lifetime.py).
2. **Against a real decay, hindcast.** With measured space weather standing
   in for the forecast (past dates only), start from a published orbit of a
   satellite that has since re-entered and compare with its re-entry date.
   Candidate: ESA's ERS-2, lowered in 2011 and re-entered in February 2024,
   which spans a full solar cycle of measured data. Its starting orbit, mass,
   area and re-entry date need published sources before this is fixed as a
   test; Cd*A of a tumbling body is uncertain, so the tolerance would be wide
   (about 20%).
3. **Against the table:** at a constant F10.7 near the table's mean activity,
   the MSIS lifetime should land within the solar-cycle spread of the table's.

## 5. Questions for the user

- **Q1. The future Sun:** A (forecast then mean cycles), B (today held) or
  C (fixed bands)?
- **Q2. The real-decay check:** is ERS-2 a good reference, or is there a
  re-entry the user would rather match?

## 6. Outcome (2026-11-24)

- **Q1: A** (user). Built: `satflight/solarforecast.py`, the NRLMSIS path in
  `lifetime.py`, the Orbit tab's "Busy / quiet Sun" and "Future Sun" rows,
  and the SWPC download in `tools/fetch_space_weather.py`.
- **Check 1 passed:** within 8 % of runs flown to re-entry under NRLMSIS
  (tests/test_msis.py), including one through a storm, where holding the date
  fixed would be 17 % off. Planted defects (date frozen, node not turning, the
  app's node or levels wrong) each fail a test.
- **Check 2 (ERS-2):** done as a hindcast for the drag area each atmosphere
  needs, since ESA gives no area: NRLMSIS 79 m^2 Cd*A, the table 39 m^2
  (PHYSICS.md). Not a pass/fail test.
- **Check 3, not met as written:** the mean-Sun NRLMSIS lifetime is longer
  than the table's, by 6 % at 693 km (131 against 124 yr) and 51 % at 500 km
  (9.5 against 6.3 yr). At 500 km even the busy level (8.2 yr) outlasts the
  table: the table's mean-Sun air there is denser than NRLMSIS gives for the
  forecast cycle. Consistent with ERS-2, where the table needed half the drag
  area NRLMSIS did.
- Cost: near-circular orbits sample 16 x 2 points (same answers to 0.1 yr as
  16 x 8); Sentinel-1's three levels take about 17 s.
