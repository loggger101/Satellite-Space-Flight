# Physical drag coefficient from NRLMSIS - design

Drafted 2026-11-24. Item 1 of the drag roadmap the user set that day (physical
Cd, then a Sentinel-1 panel model, then winds, then MSIS for the lifetime
estimate / Orbit tab, then MSIS for launches). Serves audit B2 ("Cd stays a
constant per satellite") and D2 ("free-molecular Cd"). **No code until the
user approves this design.**

## 1. Why, and what it can and cannot fix

Drag is `a = -1/2 rho Cd (A/m) |v_rel| v_rel` with a fixed `Cd = 2.2`. In free
molecular flow (Sentinel-1's 693 km), Cd depends on the air: its temperature,
its mean molecular mass, and how much atomic oxygen is stuck to the surface.
Oxygen on the surface makes molecules bounce off diffusely ("accommodated").
At 693 km near solar maximum, published physical models give about 2.5-3.0.
The Sentinel-1 fits agree: with Cd held at 2.2 they need a mean scale of
1.23-1.27, an effective Cd of 2.7-2.8.

**What it cannot fix.** Each accuracy arc *fits* Cd·A/m over the first 24 h.
A Cd that is merely a better constant is absorbed by that fit and changes
nothing. Physical Cd helps only through its **variation**: over an orbit
(day/night, latitude) and, above all, from the fit day to the three predicted
days when a storm heats the air and changes its composition. So the expected
gain is modest and concentrated in storm arcs. This design may well end in "no
measurable change", and by the user's rule (section 6) it then does not
become the Sentinel-1 default.

**Probe (2026-11-24, scratch script, section 2's formulas before task 1
confirms them).** Physical Cd at 693 km, 7.5 km/s, over a lat/lon grid at
12:00 UTC:

| day | weather | Cd mean (min-max) | alpha | T |
|---|---|---|---|---|
| 2025-08-05 | quiet | 2.87 (2.81-2.93) | 0.07 | 1021 K |
| 2025-12-09 | quiet | 2.84 (2.73-2.93) | 0.14 | 1098 K |
| 2025-11-11 / 12 / 13 | storm | 2.83 / 2.76 / 2.79 | 0.16 / 0.30 / 0.23 | 1112 / 1296 / 1220 K |
| 2026-01-19 / 20 / 21 | storm | 2.82 / 2.76 / 2.79 | 0.17 / 0.28 / 0.24 | 1136 / 1315 / 1251 K |

- The mean level (about 2.8) matches the fits' effective Cd of 2.7-2.8, which
  supports the model.
- Across an orbit, Cd varies about ±3%.
- In a storm it **drops** about 2-3%: hotter air carries more oxygen, so
  more molecules are accommodated.

The storm arcs' remaining error comes from MSIS running *light*: the fits
want a scale of 1.36 in January 2026. A falling Cd pushes the drag the wrong
way, so physical Cd is unlikely to cut the storm errors and may slightly
raise them. The real gain, if any, is the ±3% within an orbit, which matters
for the panel model later. **This is a reason to reconsider the roadmap
order (section 7, Q0).** (The user chose to build it as designed.)

**Probe redone with the production code (rms reemission, 2026-11-24).** At
693 km and 7.5 km/s, the drag-weighted daily mean over a lat/lon grid every
3 h, for each arc's fit day (d0) and its three predicted days:

| arc | d0 Cd | d1 / d2 / d3 vs d0 | density d1 / d2 / d3 vs d0 |
|---|---|---|---|
| 2025-08-05 quiet | 3.19 | -0.2 / -0.1 / -0.4 % | x1.08 / 1.04 / 1.16 |
| 2025-11-04 storm | 3.15 | -1.1 / -1.6 / -2.0 % | x1.33 / 1.47 / 1.61 |
| 2025-11-11 storm | 3.12 | -3.1 / -1.6 / +1.0 % | x1.81 / 1.38 / 0.78 |
| 2025-12-09 quiet | 3.12 | -0.6 / -0.8 / -0.2 % | x1.15 / 1.23 / 1.05 |
| 2026-01-20 storm | 3.03 | +0.6 / +2.5 / +2.0 % | x0.87 / 0.59 / 0.66 |

- The level is now about 3.1, above the fits' effective 2.7-2.8 at 10 m². So
  the fitted correction should come out near 0.88. The level is absorbed by
  the fit either way.
- Day to day, Cd moves -3% to +2.5%, against the density: it dips when the
  air thickens. The same direction as the first probe.

## 2. The model

**Sphere, DRIA (diffuse reflection with incomplete accommodation), Sentman
closed form, summed over species.** This is the standard physical-Cd baseline
(Sutton 2009; Pilinski et al. 2013; Mehta et al. 2014). A sphere is a
stand-in until the panel model (roadmap item 2) replaces it with flat plates.
The panel model will reuse the per-species and accommodation code below.

For each species j (N2, O2, O, He, H, Ar, N, anomalous O; number densities
from pymsis), with molecular mass m_j:

- Speed ratio `s_j = |v_rel| / sqrt(2 k T / m_j)` (T is MSIS temperature).
- Sphere coefficient (Sentman):
  `Cd_j = e^(-s^2) (1 + 2 s^2) / (sqrt(pi) s^3) + (4 s^4 + 4 s^2 - 1) / (2 s^4) erf(s) + (2 sqrt(pi) / 3) (v_r / v_i)`
- Reflected-to-incident speed:
  `v_r / v_i = sqrt( 1/2 [1 + alpha (4 R T_w / v_i^2 - 1)] )`,
  with R the species' specific gas constant and wall temperature `T_w = 300 K`.
- Mixture: `Cd = sum_j rho_j Cd_j / rho` (mass-density weighted).

**Accommodation alpha (SESAM, Pilinski et al. 2010/2013, sphere fit):**
`alpha = K P / (1 + K P)`, `P = n_O T` (atomic-oxygen partial pressure / k),
`K = 7.50e-17 m^3/K`.

*To confirm before coding (task 1):* each formula and K against the papers'
own text and published check values. They are written here from memory of
the literature, and a wrong sign or factor would pass every self-consistency
test.

**Task 1 outcome (2026-11-24).**
- *Sources.* The papers that state the closed forms (Bernstein & Pilinski
  2022, Dey et al. 2025, Pilinski et al. 2013) could not be fetched: they
  return HTTP 403 or sit behind a bot check. Moe & Moe, "The Effects of Surface
  Composition and Treatment on Drag Coefficients of Spherical Satellites"
  (AAS 05-258, 2005), could be read. It does not print the sphere formula,
  but its appendices tabulate accommodation alpha and diffuse (Sentman) sphere
  Cd by height at solar minimum and maximum. Those numbers became the check
  values.
- *Reemission term.* The ½ form above misses Moe's Cd by up to 0.11 (2.27
  against 2.34 at alpha = 0.93). Moe's stated rms form,
  `v_r / v_i = sqrt(1 + alpha (3 R T_w / v^2 - 1))` (their p. 4, `≈ sqrt(1 - alpha)`),
  matches their ten rows from 250 to 500 km within 0.03 (pure O at 1000 K
  stands in for the air). **The rms form is used.** The 150-200 km rows (N2
  and cooler air) are left out of the check.
- *K = 7.50e-17 m^3/K.* With MSIS at fixed solar-maximum indices, the SESAM
  alpha matches Moe's solar-maximum alpha within 0.02 at 250-450 km. A K of
  2.5e-17 fails that check.
- *Species.* pymsis also returns NO (its column 9), now the ninth species. A
  species MSIS does not model at a height comes back NaN: O below about 55
  km, H below 80, N below 100, NO below 75, anomalous O below 150. It counts
  as none, but the species must then add up to MSIS's own total within 1%.
  The planted test, NaN atomic oxygen at 500 km, is refused.

## 3. Interface

- `atmosphere.msis_state(alt_km, lat, lon, jd, table=None) -> MsisState`.
  This is the one pymsis call `msis_density` makes today, keeping all 11
  outputs. Fields: `rho` (kg/m^3), `n` (N, 8) species number densities (m^-3),
  `T` (K). `msis_density` becomes a thin wrapper over it, bit-identical
  (checked by hash on the existing tests' outputs).
- `drag_coefficient.py` (new, pure numpy, no satflight imports):
  - `accommodation(n_O, T) -> alpha`
  - `sentman_sphere(s, vr_over_vi) -> Cd`
  - `sphere_cd(v_rel_ms, n, T, T_wall=300.0) -> Cd` (N,)
  - Takes arrays. Raises on a non-finite or negative input (no NaN passes
    silently: `strict.py` rules).
- `ForceModel.cd_model: "fixed" | "dria_sphere"` (scenario key
  `"cd_model"`, default `"fixed"`, so every existing scenario and test is
  untouched). `"dria_sphere"` **requires** `atmosphere == "msis"`: the table
  has no temperature or composition. Any other combination is refused when
  the scenario loads, saying why.
- In `accel_drag`, under `dria_sphere` the satellite's Cd·A/m is used as
  `Cd_sat·A/m · Cd_phys / CD_REF`, where `CD_REF = 2.2`. The satellite's Cd
  therefore becomes a **correction factor** (`Cd_sat / 2.2`), and a satellite
  left at 2.2 flies pure physics. Why this way: the fit, the batch runs and
  every UI field keep one per-satellite Cd·A/m number, and the study's
  `density_scale` column keeps its meaning. A scale of 1 means the physics
  was right. See open question Q1.
- `ForceModel.label()` / `active()`: `DRAG(MSIS,Cd)`. Physics dialog gets a
  "Drag coefficient: fixed / physical (sphere)" choice, disabled with a tip
  unless the atmosphere is MSIS. The Satellite dialog's Cd tip says what Cd
  means in each mode.
- Not built (kept for later): a physical-Cd line in `breakdown()`, and extra
  Cd columns in the study rows. The Cd along each arc is worked out from the
  truth orbit for the write-up instead.
- Study: `tools/sentinel1_study.py arcs --cd-model dria_sphere` (with
  `--atmosphere msis`), writing `arcs_msis_dria.csv`.
- MATLAB: out of this PR. It is a follow-up PR (`satflight.dragCoefficient`
  plus the export), as with #52/#53, added to the MATLAB gaps list meanwhile.

Out of scope: lifetime estimate, Orbit tab and launches (roadmap items 4-5,
still the table and fixed Cd); attitude, panels and lift (item 2); winds
(item 3).

## 4. Tasks (each a vertical slice, test first)

1. **Confirm the physics.** Check section 2 against the cited papers. Pin
   published check values: Sentman-sphere Cd at stated s, alpha and T_w, plus
   the SESAM alpha curve. Record them, with sources, in a test data file.
2. **`drag_coefficient.py`.** Tests:
   - The published values from task 1.
   - Limits: s→∞ with alpha=1 tends to `2 + (2 sqrt(pi)/3) sqrt(2 R T_w)/v`.
   - Cd falls as alpha rises.
   - The mixture of a single species equals that species.
   - Each check is first shown to fail on a planted defect (e.g. a dropped
     factor of 2 in s).
3. **`msis_state`.** `msis_density` stays bit-identical. Number densities
   times masses sum to `rho` (to pymsis's float32 precision). Typical 693 km
   values look sensible: Cd 2.4-3.0 across a day and a storm, shown in a
   table.
4. **`ForceModel.cd_model` + `accel_drag` + scenario keys + refusal of
   table+dria.** Tests:
   - `fixed` is bit-identical to today.
   - `dria_sphere` with a constant stub Cd equals `fixed` scaled.
   - The scenario round-trips.
   - A bad combination is refused with a message.
   - Fuzz/sweep tests extended.
5. **UI.** Physics dialog choice and Satellite dialog tip, checked with a
   headless screenshot.
6. **Study run and decision.** Re-fly the six MSIS arcs (four storm, two
   quiet) with `dria_sphere`, about 2 h each, run in the background with
   workers. Compare against `arcs_msis.csv` per section 6, then write up in
   PHYSICS.md and the audit's B2 status.

## 5. Failure modes (no silent fallbacks)

- pymsis missing, or space weather not measured for the date: same errors as
  MSIS today; the app pauses and says why.
- Negative or NaN number density / temperature from pymsis: raise, naming
  the place and time.
- `s` near 0 (speed ratio; impossible in orbit, but the formula divides by
  s^3): refuse below s = 0.1 rather than return inf.
- Above `TOP_KM` (rho = 0): Cd is not evaluated; drag is zero, as today.

## 6. Acceptance: the user's proof bar

It ships as the **Sentinel-1 default** only if, over the six arcs, the
along-track error at 72 h falls or holds:
- no arc worse by more than 10% or 20 m, whichever is larger;
- the storm-arc median falls.

The fitted correction factor (the `density_scale` column) should also move
toward 1 and spread less. That is reported, but it is not a gate.

If the arcs do not improve, the code can still merge as an opt-in choice,
with the result written up honestly in PHYSICS.md. The default stays `fixed`.
Either way, each commit, push and PR waits for the user's yes.

**Outcome (2026-11-24): opt-in; the default stays `fixed`.** The arcs end
just short of 72 h, so the third predicted day's along-track RMS stands in for
the 72 h value. Every arc is within the first gate (largest change +49 m,
+9.7%, on 1A 2025-11-11, where the limit is 51 m), but the storm-arc median
rises from 367 m to 397 m, so the second gate fails. The fitted correction
falls by about 30% on every arc (0.78-1.36 becomes 0.54-0.98) and spreads no
less. Full table in PHYSICS.md; rows in
`docs/analysis/sentinel1/arcs_msis_dria.csv`.

## 7. Open questions for the user

- **Q0. Still first?** The probe in section 1 says physical Cd will barely
  move the storm arcs. Two options:
  - Build it as designed. It is also the base the panel model needs, and it
    would be cheap: about a day of code plus a night of arc runs.
  - Go to the panel-model spec first and build Cd inside it, since the panel
    model needs the same per-species and accommodation code.

  Storm-time density itself (MSIS runs light; audit D36, density calibration
  from tracking) is a third candidate that targets the actual storm error.
- **Q1. Cd field meaning in physical mode.** Recommended: the satellite's Cd
  becomes a correction factor (`Cd/2.2`), as in section 3. The alternative is
  to ignore the satellite's Cd entirely in physical mode. That is simpler to
  explain, but it leaves the fit nothing to scale, so the study would need a
  separate correction field.
- **Q2. Wall temperature.** Recommended: fixed 300 K. The alternative is a
  per-satellite setting. The effect on Cd is under ~1%.
