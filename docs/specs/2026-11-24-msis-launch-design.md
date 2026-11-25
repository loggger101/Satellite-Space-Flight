# NRLMSIS for launches - design probe

**Date:** 2026-11-24. **Branch:** `msis-launch` (off `msis-lifetime`).
**Roadmap item 5** of the drag work. Proof bar: checked against reference
values.

## 1. What the probe found

The ascent model (`launch.py`) uses its own float copy of the orbit table
(`_density`, Vallado Table 8-4) for drag, and US Standard Atmosphere 1976 for
the ambient pressure that sets engine Isp.

**The Sun does not matter for an ascent.** At Cape Canaveral, NRLMSIS at
solar minimum (2019-12-15) and maximum (2024-08-15) differ by under 10 %
below 100 km, where ascent drag builds up; they part only above 150 km,
where an ascent's drag is negligible.

**The table is wrong low down.** It is a piecewise exponential fitted for
orbits; in the troposphere it is thin against both NRLMSIS and the 1976
standard (`ascent_probe.py`, scratchpad):

| height | table | US76 (p / R T) | US76 / table | NRLMSIS / table |
|---|---|---|---|---|
| 5 km | 0.615 | 0.736 | 1.20 | 1.15 |
| 10 km | 0.308 | 0.414 | 1.34 | 1.35 |
| 15 km | 0.155 | 0.195 | 1.26 | |
| 20 km | 0.0776 | 0.0889 | 1.15 | 1.20-1.22 |
| 30 km | 0.0177 | 0.0184 | 1.04 | 0.99-1.06 |
| 50 km | 1.06e-3 | 1.03e-3 | 0.97 | 0.92-1.02 |
| 70 km | 8.77e-5 | 8.28e-5 | 0.94 | 0.82-0.90 |

(kg/m^3; US76's published 10 km value is 0.41351 kg/m^3.) Max Q falls at
10-14 km, so it is under-read by about a third:

| ascent (planned afresh) | max Q table / US76 | drag loss table / US76 | insertion |
|---|---|---|---|
| Falcon 9, 400 km, 51.6 deg, Cape | 30.3 / 40.8 kPa | 18.4 / 21.7 m/s | same to 1 km |
| Falcon 9, 693 km SSO, Vandenberg | 28.2 / 37.7 kPa | 14.1 / 16.7 m/s | same to 1 km |

Guidance closes the orbit either way. The real Falcon 9 throttles down
through max Q to stay in the low 30s kPa; the model has no throttle bucket,
which the thin table has been hiding. (An Electron case to 500 km circular
SSO with 200 kg fails with both densities; that is not how Electron flies,
see docs/specs/2026-11-24-electron-to-500-km.md.)

## 2. Proposal

- **Table runs: US76 density below 86 km for ascents**, rho = p / (R T) from
  the same US76 layers that already give the pressure (R = 287.053 J/kg/K),
  the table above. Reference check: US76's published densities at the layer
  bases (1.225 at 0 km, 0.41351 at 10 km, 0.088910 at 20 km, 0.018410 at
  30 km, ...) to their printed digits, and continuity at 86 km.
- **NRLMSIS runs: an NRLMSIS column at the pad.** At liftoff, one NRLMSIS
  call gives the density at the pad's latitude and longitude on a height
  grid (0 to 1000 km), interpolated in log density in the float hot path.
  The ascent goes a few hundred km downrange in the dense part, so a column
  over the pad stands in for the air it meets. Checked against
  `msis_density` at the grid points and between them; planted defects
  (column at the wrong time or place) must fail a test.
- Max Q rises by about a third for every preset ascent under the table; the
  launch report, the log's "max Q" line and the MATLAB export (which has its
  own copy of the ascent density) change with it. MATLAB: the US76 density is
  a small port; the NRLMSIS column goes in the MATLAB follow-up.

## 3. Question for the user

- **Q1.** Build both (US76 as the new ascent default, the NRLMSIS column
  under NRLMSIS), only the NRLMSIS column (table ascents unchanged), or
  stop here?

## 4. Outcome (2026-11-24)

- **Q1: both** (user). Built: `launch.us76_density` (the ascent default, also
  ported to `matlab_runtime.m`) and `launch.msis_column`, used by every
  ascent under NRLMSIS (`AscentEnv.at_pad`).
- Checked: the 1976 standard's published densities at ten heights to 3e-4
  (tests/test_launch.py); the column against NRLMSIS at its grid heights to
  1e-6 and between them to 2 %, and a run's ascent flying the column for its
  own pad and liftoff time (tests/test_msis.py). Planted defects (the table
  in place of US76, the column at the run's start or with latitude and
  longitude swapped) each fail a test.
- An NRLMSIS run with a launch still to fly is refused on MATLAB export
  until the column is ported (matlab-export-gaps).
