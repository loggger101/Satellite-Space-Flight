# MATLAB export: what is still missing (2026-11-22)

Found while porting the Sun, Moon and sunlight pressure (step 6a of
[2026-11-18-sentinel1-poeorb-plan.md](2026-11-18-sentinel1-poeorb-plan.md),
branch `matlab-sun-moon-srp`, worktree `C:\Users\Loggg\worktrees\ssf-matlab-port`).
Read-only audit of everything the simulator models against what
`satflight/matlab_export.py` and `satflight/matlab_runtime.m` carry.

Nothing is dropped silently: whatever the exporter cannot carry it refuses
with a reason. Event log, statuses (re-entered, impacted, escaped), every
maneuver kind and timing, the three propagators, the four integrators,
launches (ascent, planner, mid-ascent snapshots) and since step 6a the Sun,
Moon and sunlight pressure with each object's Cr are all ported. The
simulator's clock never carries UT1-UTC (only replay code uses it), so the
export loses nothing there.

## Order of work

1. **Done 2026-11-22: no regression.** One test alone took 230-290 s against
   the 280 s session limit, on main as on the branch; sessions now get 900 s
   (a `time_limit` mark) and all 74 tests pass. The original note:
   **Check the step-6a MATLAB suite run.** On 2026-11-22, with the year
   study's 12 arc workers loading the machine, sessions export-3, export-6
   and export-7 failed (pytest's 300 s watchdog per session, probably time).
   Read `port1_suite.txt` in the session scratchpad; rerun the failed
   sessions when the machine is quiet. A real failure is a regression from
   step 6a (the `derivative` / `acceleration` signatures changed for every
   export) and comes first.
2. **Done 2026-11-22 (branch `launch-payload-cr`).** Launch payload Cr (a gap in the simulator itself). `LaunchSpec` has
   `payload_area` and `payload_cd` but no Cr, so with sunlight pressure on
   every launched payload is Cr 1.3.
   - Python: `LaunchSpec.payload_cr` (default 1.3, left out of `to_dict` when
     default so no bundled scenario changes); `CR_DEFAULT` moves to
     `constants.py` (scenario.py keeps the name; launch.py cannot import
     scenario.py); `validate` checks 0..2.5; `Simulation._finish_ascent` sets
     `sat.cr = spec.payload_cr`. Spent stages stay at the default.
   - MATLAB: `matlab_export.py` writes `L.payloadCr`; the runtime sets
     `s.cr(i) = L.payloadCr` at separation (next to `s.cd(i) = L.payloadCd`)
     and the launch defaults struct gets `"payloadCr", 1.3`; the toolkit's
     `addLaunch` takes `PayloadCr`.
   - Tests: Python (Cr reaches the separated satellite, default omitted, bad
     Cr refused, export carries `payloadCr`); MATLAB `ExportTest/launchInSunlight`
     (its own `EXPORT_PARTS` session): a launch with Sun + Moon + SRP,
     PayloadCr 1.7, tracked stages; positions held to the simulator and
     `out.objects.cr` 1.7 for the payload, 1.3 for the stages.
3. **Done 2026-11-22 (branch `matlab-egm2008`).** The MATLAB field agrees with
   `gravity.acceleration` and `gravity.potential` to 8e-16 relative; 70x70 costs
   1.6 ms a call in MATLAB as in Python. `ExportTest/egm2008Field` holds an
   export to under 1 mm over 6 h; a field one degree short fails it (12 m),
   and a potential one degree short fails its Jacobi check (1.4e-7 against
   1e-8; the simulator's own drift is 4.7e-9). The original note:
   **EGM2008 field (still refused).**
   - The exporter embeds the field's *unnormalized* C and S to degree N with
     repr() digits (Python's `gravity.load(N).c/.s`; ~120 kB at 70x70), so
     MATLAB uses the same doubles and a MATLAB-only script needs no data file.
   - Runtime: Cunningham's V/W recursion as `gravity.acceleration` runs it
     (degree by degree, every order and row at once); `acceleration` uses it
     instead of J2-J4/C22 when a degree is set; `derivative` passes GMST when
     the field is on, not only for C22 (Python: `forces.earth_fixed`).
   - Diagnostics: port `gravity.potential` (Legendre route) so the Jacobi
     constant is checked with the field on; the two routes check each other.
   - `ignoredTerms` must use Python's labels (`EGMn` in `ForceModel.active()`).
   - Toolkit: `newScenario(GravityDegree=N)`.
   - Tests: an export cross-check at degree 20 with Sun + Moon; mutant (field
     cut one degree short) must fail it.
   - Measure MATLAB's cost of a 70x70 evaluation (Python ~1.6 ms); a
     multi-day Sentinel-1 export may otherwise take minutes.
4. **Done 2026-11-22 (branch `matlab-replay`), the user's choice: MATLAB reads
   the POEORB files.** The script names each replay's unit and last day and
   each twin's unit; the runtime reads the files (unzip, the reader's checks
   and the 10 m seam rule), the IERS table (measured values only), takes polar
   motion out, turns the Earth back by GMST at UT1 and interpolates as
   `replay.Ephemeris` does. `ExportTest/preciseOrbitReplay` holds the replayed
   S1A to the simulator's within 1 mm through the end of its recording, with
   the twin's gap (`truthGap`); `replayWithoutItsFiles` refuses an empty data
   folder naming the day. Toolkit: `orbitReplay`, `orbitPrecise`. Found on the
   way, fixed in the simulator too: a recording that ended at a shown sample
   between two steps slipped back to the last step's state (380 km) and froze
   there (`Simulation._resume`). The original note:
   **Replay and precise orbits.** A satellite following a recording is
   refused; a precise twin exports as a plain state and loses its truth (and
   the R/S/W gap to it). **Needs the user's decision:** embed the recorded
   10 s states in the script (~26,000 rows per satellite for three days,
   several MB) or have MATLAB read ESA's POEORB files and apply polar motion
   itself. Toolkit builders for the `replay` / `precise` orbit types follow.
5. **Done 2026-11-23 (branch `matlab-study`).** `satflight.readStudy` reads
   the study's CSVs into four tables (daily, phasing, burns, arcs; MAT through
   `save -struct`), refusing a foreign column, and `satflight.plotStudy` draws
   the report's three figures. `SatflightToolkitTest/studyTablesMatchTheCsvs`
   checks every column's row count, filled cells and sum against Python's own
   reading of the files; `tests/test_matlab.py` holds readStudy's column lists
   to the study writer. Found on the way, fixed in the report too: the elements
   figure ticked the placebo windows as burns (`elements.png` regenerated; the
   report text and the other figures came out byte-identical). The original
   note: **Study results in MATLAB** (second half of step 6): the year study's
   CSVs (`docs/analysis/sentinel1/`) as tables / MAT plus a plotting function.

## Smaller items

- The app's launch form has no payload Cd or Cr field (only scenario files
  set them). Python UI, not the exporter.
- MATLAB toolkit: no replay / precise orbit builders (item 4). Done.
- `Simulation._perturbed` (and so `circular_velocity`) ignores the EGM2008
  field: a run with the field and J2 off treats a circularize burn as two-body.
  The MATLAB port copies it to stay in step; fix both together.
