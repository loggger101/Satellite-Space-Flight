# Physics audit: problems, additions and repos to integrate

**Date:** 2026-10-02 · **Code audited:** `main` at `f1e701b` · **No code was changed.**

This is an audit of the physical calculations only. It covers forces, frames
and time, orbital elements, the integrators and analytic propagators, maneuvers
and planners, events, the launch model, and the derived readouts in the Orbit
tab. UI code was checked only where it computes something physical. Nothing
here has been fixed.

The findings are in four parts:

- **[Part A: Problems](#part-a-problems-in-the-existing-code-and-physics)**
  covers what the code already does but gets wrong, gets inconsistent, or
  describes wrongly. Each item gives where it is, what is wrong, the evidence,
  the impact and a fix.
- **[Part B: Repo additions](#part-b-additions-that-need-work-inside-the-repo)**
  covers physics or features the simulator does not have yet and that have to
  be built inside satflight. Each item gives why it matters and the work in
  the repo.
- **[Part C: Repositories](#part-c-github-repositories-that-could-be-integrated)**
  lists GitHub repositories, mostly from your starred list, that could help
  with Parts A, B and D, and how each one could be used given its licence.
- **[Part D: Standalone additions](#part-d-standalone-additions-that-can-be-built-outside-the-repo)**
  covers self-contained pieces (models, solvers, data) that can be built and
  tested without satflight's code, each with the interface satflight will call
  it through.

Where fixing a problem needs a new feature, the Part A item points to the
Part B item, and a Part B item names the Part D pieces it uses.

Severity in Part A:

- **High:** gives wrong results that a normal user will run into.
- **Medium:** visible in readouts or plans.
- **Low:** small in magnitude, or only a claim in the docs.

Priority in Parts B and D:

- **High:** the largest remaining physical error, or needed by a Part A fix.
- **Medium:** clear value for accuracy or realism.
- **Low:** refinement.

"Measured" means the number came from running the code; the scripts are
described in [How the numbers were measured](#how-the-numbers-were-measured).
"Estimated" means the number is a hand calculation.

Scope reminder ([SCOPE.md](SCOPE.md)): only the Earth may act on an orbit.
Additions that would involve the Sun or the Moon are listed separately in
[B11](#b11-additions-that-need-a-scope-decision).

---

# Part A: Problems in the existing code and physics

## Summary of problems

| ID | Area | Sev. | One line |
|---|---|---|---|
| [A1](#a1-lambert-picks-the-wrong-way-round-for-polar-orbits) | Planner | **High** | Lambert's direction test uses the ECI z-axis; polar orbits can get the reverse transfer, ~12x the delta-v |
| [A2](#a2-planners-predict-with-two-body-motion-while-the-run-has-j2) | Planner | **High** | Rendezvous planned two-body in a J2 run misses by 11 km after 1 h and 90 km after 3 h |
| [A3](#a3-two-different-altitudes-are-used-side-by-side) | Cross-cutting | **High** | "Altitude" means ellipsoid height in some places and r - R_eq in others: up to 21 km apart, ×1.6 in density at 300 km |
| [A4](#a4-the-eci-frame-is-labelled-j2000-but-is-of-date) | Frames | Medium | ECI is really equator/equinox *of date*, not J2000: 0.37 deg apart in 2026 |
| [A5](#a5-ltan-is-apparent-solar-time-so-it-swings-through-the-year) | Readouts | Medium | A perfect SSO's LTAN reads 10:16 to 10:46 over a year; the glossary says it stays constant |
| [A6](#a6-hohmann-and-bi-elliptic-plans-assume-the-burn-is-at-an-apsis) | Planner | Medium | Hohmann from an eccentric orbit, off-apsis: 2,090 km circular instead of 2,000, with no warning |
| [A7](#a7-circularize-uses-the-two-body-circular-speed) | Maneuvers | Medium | "Circularize" under J2 leaves an 8-19 km radius swing |
| [A8](#a8-burns-can-spend-unlimited-delta-v) | Spacecraft | Medium | No dry-mass floor: any delta-v can be spent, and A/m grows without bound |
| [A9](#a9-orbit-tab-drag-readout-uses-inertial-speed-and-spherical-altitude) | Readouts | Medium | Drag at perigee uses inertial speed (+8 %) and spherical altitude |
| [A10](#a10-most-events-are-timed-to-the-step-not-to-the-crossing) | Events | Medium | Eclipse, AOS/LOS and re-entry are logged up to one step (60 s) late |
| [A11](#a11-altitude-means-mean-for-walker-shells-but-osculating-for-single-satellites) | Scenarios | Medium | The same "altitude" gives different orbits for a Walker shell and a single satellite |
| [A12](#a12-analytic-propagators-silently-drop-force-terms) | Propagators | Low | `kepler`/`j2mean` ignore drag, J3, J4 and C22 while the force label says they are on |
| [A13](#a13-sun-synchronous-inclination-ignores-j4) | Maneuvers | Low | "sso" drifts in LTAN in runs with J4 on |
| [A14](#a14-launch-window-regression-uses-the-osculating-semi-major-axis) | Launch | Low | Launch window into another plane uses the osculating a for J2 regression |
| [A15](#a15-launch-bookkeeping-details) | Launch | Low | Fairing mass can leave with the last stage; throttle floor can break the g-limit; two pressure models |
| [A16](#a16-solver-edge-cases) | Solvers | Low | Silent non-convergence, a division by zero at e = 1, ill-conditioning near 180 deg |
| [A17](#a17-sgp4-errors-fall-back-silently) | TLE | Low | A failed SGP4 call (e.g. a decayed object) silently uses the J2 fallback |
| [A18](#a18-wisdom-holman-with-drag-is-first-order-in-the-drag-term) | Integrators | Low | The docs call it second order with drag; the drag coupling is first order |
| [A19](#a19-one-dopri5-tolerance-for-km-and-kms) | Integrators | Low | One `atol` for km and km/s makes the velocity tolerance ~100x looser |
| [A20](#a20-drag-during-finite-burns-uses-the-mass-at-ignition) | Drag | Low | `Cd·A/m` is frozen at ignition mass for the whole burn |
| [A21](#a21-sun-ephemeris-time-scale) | Ephemeris | Low | The Sun series is fed UTC instead of TT (0.0008 deg) |

## High severity

### A1. Lambert picks the wrong way round for polar orbits

**Where:** [`maneuvers.py:65-68`](../satflight/maneuvers.py#L65),
[`planner.py:94`](../satflight/planner.py#L94),
[`planner.py:127`](../satflight/planner.py#L127)

**What:** The planner decides "prograde" from the sign of the chaser's
angular momentum z-component, `h_z >= 0`. `lambert` then picks the short or
long way from the sign of `(r1 x r2)_z`. For a polar orbit (i = 90 deg), `h_z`
is about 0, so both signs come from rounding and from which side the target
plane lies on. The test should be relative to the chaser's own orbit:
`dot(r1 x r2, h1) > 0` means the short way is prograde.

**Evidence (measured):** Chaser at 500 km and i = 90 deg; target at 520 km
with RAAN 0.3 deg away; time of flight 1500 s.

| target RAAN | dV1 chosen | the other direction |
|---|---|---|
| +0.3 deg | **14,678 m/s** (retrograde arc) | 1,197 m/s |
| -0.3 deg | 1,197 m/s | 14,678 m/s |

At i = 98 deg and 51.6 deg the right branch was picked in every case tried.
The failure band is near i = 90 deg, where `|h_z|` is small compared with the
plane offset between chaser and target.

**Impact:** Rendezvous plans for polar orbits can come out absurd, and
`scan_rendezvous` then reports "no safe arc" or a very expensive one.

**Fix:**
- Pass the chaser's `h` (or a direction) into `lambert` and choose the branch
  by `dot(cross(r1, r2), h1)`.
- Add a regression test with i = 90 deg and a target on each side of the plane.

### A2. Planners predict with two-body motion while the run has J2

**Where:** [`planner.py:23-29`](../satflight/planner.py#L23) (`_burn_state`),
[`planner.py:93`](../satflight/planner.py#L93),
[`planner.py:130`](../satflight/planner.py#L130),
[`maneuvers.py:169`](../satflight/maneuvers.py#L169) (`resolve_time`)

**What:** Burn times, the target's future position and the Lambert arc all
come from Kepler propagation of the *osculating* state. Under J2 the
osculating semi-major axis is several km off the mean one. PHYSICS.md already
notes this puts the ISS 816 km off after a day with plain Kepler, and here it
goes straight into the rendezvous aim point. The final `match_velocity` burn
matches velocity but leaves the position miss in place.

**Evidence (measured):** `scenarios/rendezvous.json`, replanned with
`plan_rendezvous` and run with Cowell:

| forces | time of flight | miss at arrival |
|---|---|---|
| two-body | 1.0 / 1.15 / 3.0 h | 0.00 km |
| J2 | 1.0 h | **11.0 km** |
| J2 | 1.15 h | **11.7 km** |
| J2 | 3.0 h | **90.1 km** |

`test_rendezvous_scenario_arrives_at_target` passes only because
`rendezvous.json` has J2 switched **off**.

**Impact:** Any rendezvous planned in a run with J2 (the default force model)
misses its target by kilometres to tens of kilometres.

**Fix:**
- Predict with `j2_mean_propagate` instead of `kepler_propagate` when
  `j2_acts(...)` is true. This is cheap and removes most of the error.
- Rerun the rendezvous test with J2 on.
- For metre-level arrival, see [B5](#b5-rendezvous-targeting-against-the-real-force-model)
  (differential correction against the real force model).

### A3. Two different altitudes are used side by side

Two definitions of "altitude" are mixed:

- **(a) Height above the WGS-84 ellipsoid.** `forces.approx_altitude`, the CSV
  `alt_km`, drag, re-entry, and the ground-track readouts use this one.
- **(b) `|r| - R_eq`,** distance above a sphere of *equatorial* radius. These
  use it:
  - orbit specs `altitude` / `perigee_alt` / `apogee_alt`
    ([`scenario.py:264-270`](../satflight/scenario.py#L264));
  - Walker `altitude` ([`scenario.py:345`](../satflight/scenario.py#L345));
  - the Hohmann/bi-elliptic `target_alt`
    ([`planner.py:36`](../satflight/planner.py#L36),
    [`planner.py:60`](../satflight/planner.py#L60));
  - the launch target perigee and apogee
    ([`launch.py:575`](../satflight/launch.py#L575));
  - the Orbit tab `rp_alt` / `ra_alt` and drag density at perigee
    ([`orbitinfo.py:178-180`](../satflight/orbitinfo.py#L178));
  - orbit classification "suborbital (intersects Earth)"
    ([`analysis.py:126`](../satflight/analysis.py#L126));
  - the batch summary table `alt km`
    ([`batch.py:98`](../satflight/batch.py#L98)), while the CSV from the same
    run uses (a);
  - the tooltip altitude in `ui/app.py`.

**Evidence (measured):** A point 300 km above the ellipsoid at latitude
89.9 deg has `|r| - R_eq` = 278.6 km. The table density at 278.6 km is
**1.6 times** the density at 300 km. A polar orbit's "perigee altitude" in the
Orbit tab is therefore up to 21 km lower than its real height when perigee is
near a pole, and the perigee drag readout is inflated accordingly.

**Impact:**
- Readouts disagree with each other: the batch table and the CSV give
  different altitudes for the same satellite.
- Drag and lifetime estimates for high-inclination, eccentric orbits are off.
- An orbit that clears the poles by 15 km is classified as intersecting the
  Earth.

**Fix:**
- Choose one meaning per use. Orbit *shape* quantities (perigee and apogee
  "altitude") conventionally use `r - R_eq`. Anything physical (density, the
  surface, re-entry) should use the ellipsoid height at the actual latitude.
- For the perigee readout, compute the ellipsoid height at the perigee's
  latitude (`asin(sin i · sin argp)`) rather than `rp - R_eq`.
- Label the two clearly in the UI and in PHYSICS.md.
- Add a test that the readouts, the CSV and the batch summary agree.
- An exact, fast geodetic conversion ([D50](#d50-closed-form-geodetic-conversion))
  would let every physical "altitude" use the true ellipsoid height cheaply.
- The 3-D globe is also a sphere of equatorial radius (see `render3d._occluded`
  notes). That is a display issue, but it is the same root cause.

## Medium severity

### A4. The ECI frame is labelled J2000 but is of-date

**Where:** [`frames.py:3-5`](../satflight/frames.py#L3),
[`tle.py:4-6`](../satflight/tle.py#L4),
[`ephemeris.py`](../satflight/ephemeris.py), PHYSICS.md "Frames and time"

**What:** Earth rotation is a single GMST (IAU-82) rotation, which takes the
mean/true equator and equinox *of date* to the Earth-fixed frame. The Sun
series (Vallado Alg. 29) is also mean-of-date, and SGP4 output (TEME) is
of-date too. Inside the program these are consistent, so the sim's inertial
frame is in fact a pseudo-inertial of-date frame (TEME-like), not J2000/GCRF.

- The tle.py docstring says TEME and ECI "differ by well under a milliradian".
  That holds against TOD. Against J2000 it is about **6.5 mrad (0.37 deg) in
  2026** (estimated: general precession 50.3"/yr × 26.75 yr).
- A user who exports the CSV and compares it with a J2000 tool (GMAT, Orekit,
  poliastro, astropy GCRS) will see positions rotated by about 0.37 deg: about
  45 km in LEO and about 270 km at GEO.
- RAANs typed into scenarios are of-date RAANs, not J2000 ones.

**Fix:** Relabel the frame as "TEME / mean-of-date, pseudo-inertial" in
frames.py, tle.py and PHYSICS.md, and say so in the CSV header. Keep the
integration frame as it is. Precession and nutation rotate the Earth's
*axis*, which is also the J2 symmetry axis, so an of-date integration frame is
the standard choice at this level of model. A real J2000 export is an
addition: [B8](#b8-import-and-export-j2000-states-and-ccsds-messages).

### A5. LTAN is apparent solar time, so it swings through the year

**Where:** [`orbitinfo.py:129-132`](../satflight/orbitinfo.py#L129),
`ui/glossary.py:81` ("A Sun-synchronous orbit keeps it the same all year")

**What:** `local_time_of_node` measures the node against the *true* Sun's
right ascension. A sun-synchronous orbit's node follows the *mean* Sun, at
0.9856 deg/day uniformly. The two differ by the equation of time, about ±16
min. LTAN is conventionally quoted as mean local solar time.

**Evidence (measured):** A node held fixed at 10:30 mean local time over 2026
reads **10:16 to 10:46** in the LTAN readout, a 30.6-minute swing. This
contradicts the glossary text.

**Fix:**
- Compute LTAN from the mean Sun:
  `RA_mean_sun ≈ 280.4606 + 36000.7701 T` (deg), or `GMST - UT`.
- Optionally show the apparent value next to it, labelled "apparent".
- Add a test that an SSO keeps a constant LTAN over a year.

### A6. Hohmann and bi-elliptic plans assume the burn is at an apsis

**Where:** [`planner.py:32-50`](../satflight/planner.py#L32),
[`planner.py:53-80`](../satflight/planner.py#L53)

**What:**
- Burn 1 is a prograde `dv1 = v_needed - |v|`, with `v_needed` from vis-viva
  for a = (r1 + r2)/2. The coast is half a period. Both are exact only if the
  velocity at the burn point is horizontal, i.e. a circular orbit or a burn
  at an apsis.
- With `timing="now"` on an eccentric orbit, the transfer orbit's apsides are
  not r1 and r2. Burn 2 (circularize) then fires at the wrong radius, after
  the wrong coast time.
- "Periapsis" and "apoapsis" timing on a nearly circular orbit uses
  *osculating* elements. Under J2 the osculating e of a "circular" LEO orbit
  is about 1e-3, so the osculating perigee can be anywhere.

**Evidence (measured):** 300 × 1,500 km orbit at nu = 90 deg, Hohmann to
2,000 km planned "now", two-body run. The result was a **2,090 × 2,090 km**
orbit, 90 km above the target, and nothing warned about it.

**Fix:**
- Warn, or refuse, when `|flight-path angle|` at the burn exceeds a threshold,
  or default the timing to the right apsis.
- Solve the first burn as a two-component (V and B) burn that puts the new
  apoapsis at r2, or use Lambert to the point 180 deg ahead at radius r2.
- For "periapsis"/"apoapsis" timing on near-circular orbits, use mean elements
  (J2 short-period terms removed), or refuse when e < 2e-3.

### A7. Circularize uses the two-body circular speed

**Where:** [`maneuvers.py:207-211`](../satflight/maneuvers.py#L207)

**What:** The target velocity is `sqrt(mu / r)` horizontal. Under J2 the
radial pull at the equator is `mu/r^2 (1 + 3/2 J2 (Re/r)^2)`, so the burn
leaves the orbit about 5 m/s slow at 700 km.

**Evidence (measured):** Started at 690 × 710 km, circularized at perigee,
J2 Cowell for 200 min:

| inclination | radius afterwards |
|---|---|
| 0 deg | 671 to 690 km (**19 km swing**) |
| 51.6 deg | 679 to 690 km |
| 98 deg | 683 to 690 km |

At i = 0 an exactly circular orbit does exist under J2, so the remaining
swing is pure error.

**Fix:**
- Equatorial: use `sqrt(mu/r (1 + 3/2 J2 (Re/r)^2))`, as
  `scenario.geostationary_radius` already does.
- Inclined: target a frozen or mean-circular orbit. That needs the full
  first-order mean-element conversion in
  [B4](#b4-mean-elements-in-the-propagator-scenarios-and-readouts).
- The same applies to Hohmann burn 2 and the launch `circularize` option.

### A8. Burns can spend unlimited delta-v

**Where:** [`simulation.py:565`](../satflight/simulation.py#L565),
[`simulation.py:580`](../satflight/simulation.py#L580),
[`maneuvers.py:243`](../satflight/maneuvers.py#L243)

**What:** A satellite has one `mass`. Impulsive burns shrink it through
`exp(-dv / (Isp g0))` with no floor. Finite burns are capped only at 95 % of
the current mass. A 500 kg satellite can therefore do 10 km/s (mass to 17 kg)
and keep going. Its `Cd A / m`, and so its drag, rises 30-fold.

**Fix:** This needs a propellant model, which the simulator does not have:
see [B3](#b3-propellant-budget). Until then, at least warn when a burn takes
the mass below some fraction of its starting value.

### A9. Orbit-tab drag readout uses inertial speed and spherical altitude

**Where:** [`orbitinfo.py:178-180`](../satflight/orbitinfo.py#L178),
[`orbitinfo.py:203`](../satflight/orbitinfo.py#L203)

**What:** `drag_perigee = 1/2 rho v_peri^2 Cd A/m` uses the *inertial*
perigee speed. The force model correctly uses `v - omega x r`. The density is
looked up at `rp - R_eq` (see A3).

**Evidence (measured):** ISS-like orbit at the equator crossing:
`|v|^2 / |v_rel|^2` = **1.083**, so the readout is 8 % high for prograde
orbits. Retrograde orbits read low.

**Fix:** Use the air-relative speed at perigee (perigee position × omega).
Look up density at the ellipsoid height at the perigee latitude.

### A10. Most events are timed to the step, not to the crossing

**Where:** [`simulation.py:812`](../satflight/simulation.py#L812) (eclipse),
[`simulation.py:823`](../satflight/simulation.py#L823) (AOS/LOS),
[`simulation.py:776-794`](../satflight/simulation.py#L776) (re-entry, impact
and escape)

**What:** Eclipse entry and exit, AOS/LOS, re-entry and impact are stamped
with the time of the step that first sees them. With `h_max` = 60 s that can
be up to a minute late. The doc admits this for eclipse and AOS/LOS.
- Close approaches already get sub-step accuracy from the Hermite curve.
- A trajectory can also dip below the 80 km re-entry altitude between steps
  and come back up. This is estimated to happen only for perigees within about
  1 km of the threshold at 60 s steps, but nothing checks for it.

**Fix:**
- Reuse the same Hermite interpolant (`_hermite_min` already has both ends of
  each step) to root-find:
  - the illuminated fraction crossing 0.5;
  - elevation minus mask for each station;
  - altitude minus 80 km;
  - altitude crossing 0.
- Stamp events at the root time. For eclipses, also log the penumbra and umbra
  boundaries separately.

### A11. Altitude means mean for Walker shells but osculating for single satellites

**Where:** [`scenario.py:258-292`](../satflight/scenario.py#L258) vs
[`scenario.py:337-354`](../satflight/scenario.py#L337)

**What:**
- A Walker constellation's `altitude` is a **mean** altitude: the J2
  short-period term is applied per member.
- A single satellite's `"altitude": 550` builds an **osculating** circular
  orbit at that point. Its mean semi-major axis therefore depends on where it
  starts (±6 km in LEO), so its period differs from the Walker member's by up
  to about 0.15 %.
- A single satellite and a one-plane "Walker" of the same altitude drift apart
  in phase by up to several degrees per day (estimated about 7 deg/day at
  the extremes of the ±6 km term).
- The osculating-circular start also carries a mean eccentricity of about 1e-3
  (see A7), i.e. a ±7 km altitude oscillation that users read as "my circular
  orbit isn't circular".

PHYSICS.md documents the choice; the problem is that the *same word* means two
things.

**Fix:** Make `"altitude"` (circular) consistently mean everywhere. Add an
explicit `"osculating": true` flag for users who want the raw state. Doing
this properly uses
[B4](#b4-mean-elements-in-the-propagator-scenarios-and-readouts).

## Low severity

### A12. Analytic propagators silently drop force terms

`kepler` and `j2mean` ignore drag, J3, J4 and C22 whatever the Physics dialog
says. The propagator tooltip says so, but the event log and the force label
(`2B+J2+DRAG`) do not, so a `j2mean` run with drag "on" never decays.

**Fix:** Log a warning when such a run starts, or grey out the terms that do
not apply.

### A13. Sun-synchronous inclination ignores J4

**Where:** [`maneuvers.py:102`](../satflight/maneuvers.py#L102)

With J4 on, the J4 and J2² node terms shift the required inclination by
hundredths of a degree (estimated). "sso" orbits therefore drift slowly in
LTAN in J2+J4 runs.

**Fix:** Include J4 in the node rate, or solve numerically against the
enabled force model.

### A14. Launch window regression uses the osculating semi-major axis

**Where:** [`simulation.py:420`](../satflight/simulation.py#L420)

`plane_of` computes the J2 regression rate from the osculating a, where the
Orbit tab uses the mean a. That is about 0.3 % rate error, roughly 0.03 deg of
node over a 2-day window search (estimated).

**Fix:** Use `osculating_to_mean_a`.

### A15. Launch bookkeeping details

- If cutoff happens below `fairing_alt`, the fairing's mass leaves with the
  last stage ([`launch.py:914`](../satflight/launch.py#L914)).
- The 40 % throttle floor can exceed `max_g` late in a burn.
- Ambient pressure for thrust uses `exp(-h / 7 km)`
  ([`launch.py:60`](../satflight/launch.py#L60)), independent of the density
  table's own scale heights.
- The liftoff thrust-to-weight check ignores centrifugal relief (0.3 %).

**Fix:** Jettison the fairing at cutoff if it is still on; let the throttle
go lower or log the overshoot; derive pressure from the same atmosphere table
as density, or from the US Standard Atmosphere
([D52](#d52-us-standard-atmosphere-1976)).

### A16. Solver edge cases

- `kepler_propagate` and `lambert` return the last iterate silently when they
  do not converge. Lambert raises only when it runs out of iterations.
- `mean_to_true` with e exactly 1 divides by zero in the hyperbolic branch.
- Lambert's psi bounds (±4π²) limit very fast hyperbolic arcs.
- Transfer angles near 180 deg are ill-conditioned: the plane is undefined,
  and a large out-of-plane delta-v appears (seen in the A1 runs).

**Fix:** Return a convergence flag and warn; handle e = 1 with Barker's
equation; warn near 180 deg. A more robust Lambert solver and a two-body
propagator that reports convergence are standalone additions:
[D5](#d5-lambert-solver) and [D51](#d51-robust-two-body-propagator).

### A17. SGP4 errors fall back silently

**Where:** [`tle.py:90-96`](../satflight/tle.py#L90)

A non-zero SGP4 error code (e.g. a decayed object) silently falls back to the
J2 path.

**Fix:** Warn in the event log.

### A18. Wisdom-Holman with drag is first order in the drag term

**Where:** [`integrators.py:86`](../satflight/integrators.py#L86)

The closing half-kick evaluates the velocity-dependent force at the
*pre-kick* velocity, so the kick-drift-kick sequence is not symmetric for
drag. The effect is negligible: |d a_drag / d v| is about 1e-10 /s at 400 km
and about 1e-7 /s at 150 km (estimated). The problem is only PHYSICS.md's
claim that it remains "an accurate second-order method" with drag.

**Fix:** Correct the wording, or use an implicit midpoint (one fixed-point
iteration) for the drag kick.

### A19. One DOPRI5 tolerance for km and km/s

**Where:** [`integrators.py:98`](../satflight/integrators.py#L98)

`atol` = 1e-6 applies to both position (1 mm) and velocity (1 mm/s). With
`rtol` = 1e-9, the velocity tolerance is about 100× looser *relative to the
velocity* than the position tolerance (estimated for LEO). Results are fine
(the test shows under 1 m/day), but the step-size choice is driven by an odd
mix.

**Fix:** Scale the velocity components by the orbital period (km/s × T), or
give separate atols.

### A20. Drag during finite burns uses the mass at ignition

**Where:** [`simulation.py:614`](../satflight/simulation.py#L614)

`Cd·A/m` is evaluated once per segment, so A/m stays at its ignition value
during a burn while the thrust uses the falling mass. Negligible except for
long, low-thrust burns at low altitude.

**Fix:** Compute `Cd·A/m` from `m(t)` inside the derivative for satellites
that are burning.

### A21. Sun ephemeris time scale

The Sun series is fed UTC where it expects TT (69 s, 0.0008 deg). There is no
aberration or light-time correction (20"). Lighting and eclipses only;
negligible.

**Fix:** Add TT = UTC + leap seconds + 32.184 s if anything ever needs it.

---

# Part B: Additions that need work inside the repo

These additions change satflight itself: its force model, simulation loop,
scenarios, planners, events or UI. Many of them use a self-contained piece
that can be built first, on its own, in
[Part D](#part-d-standalone-additions-that-can-be-built-outside-the-repo).
The **Uses** line of each item names those pieces. The **Repo work** list is
what remains to do in satflight's code.

## Summary of repo additions

| ID | Area | Priority | One line | Uses |
|---|---|---|---|---|
| [B1](#b1-spherical-harmonic-gravity-in-the-force-model) | Gravity | **High** | Full n×m gravity as a force-model term | D1 |
| [B2](#b2-a-better-atmosphere-in-the-drag-force) | Drag | **High** | Solar-activity density, winds and Cd in the drag force; a decay validation | D2, D3, D13, D36, D37 |
| [B3](#b3-propellant-budget) | Spacecraft | **High** | Dry mass and propellant; fixes A8 | D48 |
| [B4](#b4-mean-elements-in-the-propagator-scenarios-and-readouts) | Elements | Medium | Full mean elements in `j2mean`, orbit specs and readouts; fixes A7 and A11 | D4 |
| [B5](#b5-rendezvous-targeting-against-the-real-force-model) | Planner | Medium | Differential correction and a terminal approach | D5, D6, D40 |
| [B6](#b6-launch-vehicle-physics) | Launch | Medium | Mach-dependent drag, separation impulses, tracked fairings, winds, strap-ons | D7, D47, D52, D63 |
| [B7](#b7-tle-drag-and-an-sgp4-propagator) | TLE | Medium | B* used for drag; SGP4 as a propagator; `sgp4` as a dependency | D25, D46 |
| [B8](#b8-import-and-export-j2000-states-and-ccsds-messages) | Interop | Medium | J2000 states and CCSDS messages in and out | D8, D34 |
| [B9](#b9-smaller-models-plugged-in) | Various | Low | Eclipse geometry, refraction and masks, small forces, Earth orientation, geoid | D8-D12, D50 |
| [B10](#b10-conjunction-outcomes-and-statistics) | Events | Low | Collision events, per-satellite covariance, probability of collision | D14, D38, D39, D55, D56 |
| [B11](#b11-additions-that-need-a-scope-decision) | Scope | n/a | Diurnal bulge, albedo, tides, Sun and Moon gravity, SRP | |
| [B12](#b12-verification-additions) | Tests | **High** | Regression tests for Part A; comparisons against reference data | D15, D36, D58 |
| [B13](#b13-re-entry-to-the-ground) | Re-entry | Medium | Follow re-entering objects to the ground; impact point on the map | D16, D44, D53, D54, D59 |
| [B14](#b14-station-keeping-and-orbit-maintenance) | Operations | Medium | Automatic reboosts, GEO east-west control, constellation phasing | D41 |
| [B15](#b15-low-thrust-burns) | Maneuvers | Medium | Finite burns steered by a law instead of a fixed direction | D17, D48, D61 |
| [B16](#b16-more-maneuvers-in-the-planner) | Maneuvers | Medium | Phasing, RAAN-only change, combined plane change, deorbit, avoidance | D18, D40, D56 |
| [B17](#b17-orbit-design-keywords) | Design | Medium | `"repeat"`, `"frozen"` and LTAN keywords in orbit specs | D19, D42, D62 |
| [B18](#b18-orbital-lifetime-readout) | Readouts | Medium | Estimated lifetime and a deorbit-rule check in the Orbit tab | D20, D37 |
| [B19](#b19-coverage-passes-and-link-readouts) | Analysis | Medium | Coverage statistics, GNSS DOP, pass predictions, Doppler, inter-satellite visibility | D21, D22, D42, D45, D57, D62 |
| [B20](#b20-relative-motion-view) | Analysis | Medium | One satellite seen from another in the LVLH frame | D6, D43 |
| [B21](#b21-orbit-determination-in-the-simulation) | Estimation | Low | Simulated tracking from the stations, fitted and shown against truth | D23, D24, D38, D49, D57 |
| [B22](#b22-integrators-and-multi-rate-stepping) | Integration | Medium | New integrators and an averaged propagator as options; multi-rate stepping | D26, D27, D51 |
| [B23](#b23-monte-carlo-dispersions) | Uncertainty | Medium | Dispersed copies run as one ensemble | |
| [B24](#b24-attitude-in-the-ensemble) | Spacecraft | Low | Attitude state, attitude-dependent drag and lift | D28, D60 |
| [B25](#b25-magnetic-field-and-radiation-overlays) | Environment | Low | South Atlantic Anomaly and belt flags on the map and in the log | D29, D60 |
| [B26](#b26-power-and-thermal-readouts) | Spacecraft | Low | Power, battery and temperature per satellite | D30, D59 |
| [B27](#b27-launch-guidance-and-mission-additions) | Launch | Medium | Dogleg ascents, parking orbits with restarts, booster recovery, failures | D45, D63 |
| [B28](#b28-fragmentation-events-and-catalogue-scale-runs) | Debris | Medium | Breakups as events; tens of thousands of objects | D31, D39, D55 |
| [B29](#b29-differential-drag-and-formation-control) | Operations | Low | Phasing by switching drag area | D43 |
| [B30](#b30-aerobraking-and-aerocapture) | Maneuvers | Low | Lowering an orbit with the atmosphere | D16, D53 |
| [B31](#b31-imaging-targets) | Analysis | Medium | Ground targets with imaging windows | D32 |
| [B32](#b32-observer-visibility) | Analysis | Low | Visible passes and brightness for a ground observer | D33, D49 |
| [B33](#b33-deployment-docking-and-configuration-changes) | Spacecraft | Medium | Dispenser deployment, docking and undocking, area changes | D43 |
| [B34](#b34-scenarios-from-live-data) | Data | Medium | "What is up there now" scenarios | D3, D46 |
| [B35](#b35-relativistic-clock-readout) | Readouts | Low | Clock-rate offset per satellite (GPS: +38 µs/day) | D12 |
| [B36](#b36-ballistic-trajectory-planner) | Planner | Low | Aim a `surface` launch at a place on the map | D35 |

### B1. Spherical-harmonic gravity in the force model

**Why:** Only J2, J3, J4 and C22/S22 are modelled. Error estimates:
- LEO: the missing J5-J6 and tesserals C21/S21, C31, C33, ... cost roughly
  hundreds of metres per day of along-track error.
- GEO: C22 alone puts the stable longitudes at 75.07 E and 104.93 W, against
  about 75.1 E and 105.3 W observed. C31, S31, C33 and S33 make up the rest.
- Resonant orbits (GPS 2:1, repeat-ground-track LEO) are sensitive to specific
  tesserals (for GPS, J22 and J32), which are absent.

This is all Earth gravity, so it is **in scope**.

**Uses:** [D1](#d1-spherical-harmonic-gravity).

**Repo work:**
- A `harmonics` term in `ForceModel` with a degree and order setting
  (e.g. 8×8 for display, 20×20 or more for accuracy runs). Rotate positions to
  Earth-fixed with GMST, as `accel_c22` already does.
- Keep J2..J4 + C22 as a fast preset; turning on the full field replaces them.
- A switch and a degree/order field in the Physics dialog; the energy readout
  becomes the Jacobi integral, as it is with C22.
- Update `test_sun_and_moon_forces_are_out_of_scope`, which pins
  `ForceModel.TERMS`, and docs/SCOPE.md's in-scope list.

### B2. A better atmosphere in the drag force

**Why:** The density is the Vallado Table 8-4 piecewise exponential, a
function of altitude only, times a constant `density_scale`. It is the
largest *physical* error source in LEO:
- Solar-cycle variation is about ×10 at 400 km, and the scale stays fixed
  across a multi-year decay run.
- Geomagnetic storms add tens of % to ×2 for days.
- Seasonal and latitude variations are missing. (The day/night bulge is
  missing too, but needs a scope decision: see B11.)
- Drag uses a constant `Cd = 2.2`. Real free-molecular Cd varies about
  2.0-2.6+ with altitude and surface.
- The air is assumed to turn exactly with the Earth. Real thermospheric winds
  reach a few hundred m/s (estimated from typical HWM values), which changes
  drag by a few percent and slowly changes the inclination.
- Smaller points: the table is discontinuous at band edges (by under 1 %, e.g.
  at 25 km), which costs a few rejected DOPRI steps; the 1000-2500 km
  extrapolation then drops to zero.

**Uses:** [D2](#d2-atmospheric-density-model) for density,
[D3](#d3-data-access-space-weather-catalogues-and-re-entries) for F10.7 and
Ap, [D13](#d13-thermospheric-winds) for winds. Also [D36](#d36-density-calibration-from-tle-decay), [D37](#d37-solar-activity-forecast).

**Repo work:**
- An atmosphere choice in `ForceModel` (the current table stays as the fast
  default), with the space-weather source as a scenario setting: a fixed level,
  a schedule, or the D3 data files.
- `accel_drag` passes latitude, longitude and the Julian date as well as
  altitude, and adds the wind velocity to the air velocity.
- The ascent model in `launch.py` uses its own float copy of the table
  (`_density`); decide whether ascents keep the fast table.
- A decay validation test against the reference cases from
  [D15](#d15-reference-data-for-validation), so that "indicative" has a
  measured size.
- Scope: MSIS-type models depend on local solar time (the diurnal bulge).
  Until B11 is decided, run them with local time fixed or averaged.

### B3. Propellant budget

**Why:** Fixes [A8](#a8-burns-can-spend-unlimited-delta-v): today a satellite
can burn without limit.

**Uses:** [D48](#d48-thruster-performance-models).

**Repo work:**
- Add `dry_mass` (or a `propellant` field) to `Satellite`, `SatSpec` and the
  dialogs.
- Refuse or truncate burns that would go below dry mass, and log it.
- Show the remaining delta-v in the Orbit tab.
- Optionally give each satellite its own engine (Isp and thrust), instead of a
  per-maneuver Isp defaulting to 300 s. Engine and propellant presets can come
  from spacecost and prospector (C3).

### B4. Mean elements in the propagator, scenarios and readouts

**Why:**
- Mean-to-osculating conversion currently covers the semi-major axis only.
  `j2mean` re-reads osculating e, i and argp each step and treats them as
  mean, neglecting their J2 short-period terms (documented). J2² and J4
  secular terms (about 1e-3 relative in node rate) and J3 long-period terms
  (the frozen-orbit e/argp cycle) are also absent.
- A7 (circularize) and A11 (what "altitude" means) need a full conversion to
  be fixed properly.
- `rv2coe` uses classical elements, so near-equatorial and near-circular
  orbits (GEO above all) have noisy RAAN, argp and M in the readouts.

**Uses:** [D4](#d4-mean-elements-and-equinoctial-elements).

**Repo work:**
- `analysis.j2_mean_propagate` works on full mean elements, with J2², J4 and
  J3 terms.
- Orbit specs: `"altitude"` (circular) means mean altitude for single
  satellites as well as Walker shells (A11), with an `"osculating": true`
  escape hatch; `spread_along_orbit` and `walker_states` use the full
  conversion.
- `circularize` targets a mean-circular or frozen orbit (A7).
- The Orbit tab shows equinoctial or mean elements where classical ones are
  ill-defined.
- Re-check the tests in `tests/test_mean_elements.py`, which pin today's
  semi-major-axis-only behaviour.

### B5. Rendezvous targeting against the real force model

**Why:** After the quick fix in A2, two-body-plus-J2-mean predictions still
leave some miss. Lambert is zero-revolution only, and the current solver is
fragile near 180 deg (A16).

**Uses:** [D5](#d5-lambert-solver) for the first guess,
[D6](#d6-relative-motion-toolkit) for the terminal approach. Also [D40](#d40-transfer-grid-search).

**Repo work:**
- Differential correction in `planner.py`: after the Lambert guess, propagate
  the arc with the run's real force model (a scratch `Propagator`, as
  `_peek` already makes). Correct `dv1` by Newton/secant on the arrival miss,
  using a finite-difference state-transition matrix. Three or four iterations
  should reach metres.
- A terminal "null position" burn pair, or a CW/Hill final approach, as a
  maneuver kind.
- Replace `maneuvers.lambert` with D5, which also fixes A1's direction test.

### B6. Launch vehicle physics

**Why:** These are not in the model yet:
- `Cd` does not depend on Mach number. Real launchers peak at about ×2 near
  Mach 1-1.2, which is what sets max-Q and drag loss.
- Spent stages separate with zero relative velocity (no separation impulse),
  and keep the full-stack reference area.
- The fairing is dropped as mass but not tracked as an object.
- Already documented as missing: winds, strap-on boosters burning alongside a
  core, engine-out, and attitude dynamics.

**Uses:** [D7](#d7-launch-aerodynamics-and-vehicle-data). Also [D47](#d47-launch-performance-estimator). Also [D52](#d52-us-standard-atmosphere-1976), [D63](#d63-optimal-ascent-trajectories).

**Repo work:**
- Cd(Mach) in `launch._deriv`, keeping the float-only hot path fast (a small
  lookup table, not numpy).
- A separation delta-v per stage in `Ascent._drop`, and a per-stage area for
  spent stages.
- The fairing spawned as an object with its own area.
- Strap-on boosters: stages that burn in parallel, a change to `Stage` and to
  `Ascent._step`'s one-stage-at-a-time logic.
- More vehicle presets from D7's data.

### B7. TLE drag and an SGP4 propagator

**Why:**
- B* is parsed but not used. A TLE satellite gets the default
  mass/area/Cd (`Cd·A/m` = 0.022 m²/kg) whatever its real ballistic
  coefficient.
- `sgp4` is optional and commented out of `requirements.txt`, so most
  installs use the fallback, which also ignores drag.
- TLE satellites can only be propagated by Cowell, Kepler or `j2mean`, not
  by the theory their elements were made for.

**Uses:** [D25](#d25-tle-tools) for the B* conversion and TLE fitting. Also [D46](#d46-object-properties-dataset).

**Repo work:**
- Set `Cd·A/m` from B* when a TLE is loaded (`tle.py`, `scenario.orbit_state`).
- Make `sgp4` a listed dependency, or offer to install it.
- An `sgp4` propagator choice in `simulation.PROPAGATORS` for satellites that
  came from TLEs.
- Export a satellite as a TLE (D25's fitting).

### B8. Import and export: J2000 states and CCSDS messages

**Why:**
- The integration frame is of-date (A4). Tools such as GMAT, Orekit and
  astropy expect J2000/GCRF.
- Scenarios and the CSV export are satflight's own formats. Real orbit data
  is exchanged in CCSDS messages: OEM (ephemerides), OPM (a state with
  maneuvers), OMM (mean elements, the newer form of TLEs) and CDM
  (conjunctions).

**Uses:** [D8](#d8-precession-nutation-and-earth-orientation) for the frame
transform, [D34](#d34-ccsds-message-reader-and-writer) for the formats.

**Repo work:**
- A frame choice for `batch.py`'s CSV, stated in the header, with J2000
  output through D8.
- A `"frame": "J2000"` option for `state` orbit specs.
- Import OEM/OPM/OMM as satellites (OMM through SGP4, B7), and export
  ephemerides as OEM and close approaches as CDM (with B10).
- Optionally, a GMAT script or STK ephemeris export for the cross-checks in
  B12.
- Fix A4's labelling first.

### B9. Smaller models plugged in

**Uses:** [D50](#d50-closed-form-geodetic-conversion).

**Why:** Several small inaccuracies each have a standalone fix in Part D. In
satflight they are mostly a swap or a switch:

| What | Today | Uses | Repo work |
|---|---|---|---|
| Eclipse geometry | Sphere of equatorial radius, no atmosphere; the real shadow is about 2 % larger and the poles 21 km lower, which shifts entry and exit by a few seconds | [D10](#d10-eclipse-with-an-oblate-earth-and-atmosphere) | Swap `eclipse.shadow_fraction` |
| Station refraction and masks | No refraction (about 0.5 deg at the horizon, which matters for a 0-5 deg mask; Svalbard uses 5 deg); no terrain mask; spherical footprints | [D11](#d11-topocentric-refinements) | An azimuth/elevation mask per station in `GroundStation`; refraction in `look_angles`; ellipsoidal footprints in `groundtrack.py` |
| Small forces | General relativity (about 1.7e-8 m/s² in LEO, a few metres per day along track) and Earth infrared pressure (about 1e-8 m/s² at 0.01 m²/kg) are missing (estimated) | [D12](#d12-relativity-and-earth-infrared-pressure) | Two `ForceModel` switches; update the scope test |
| Earth orientation | No polar motion (about 10 m) or UT1-UTC (up to about 0.4 km along the equator) | [D8](#d8-precession-nutation-and-earth-orientation) | Optional EOP file in `timeutil`/`frames` |
| Geoid and terrain | Heights are above the ellipsoid (geoid up to about ±100 m); no terrain for impacts and masks | [D9](#d9-geoid-and-terrain-heights) | Mean-sea-level height readout; terrain for B13 impacts and station masks |

### B10. Conjunction outcomes and statistics

**Why:** Close approaches have no consequences: pairs pass through each
other. There is no collision or debris outcome, no covariance and no
probability of collision. The threshold is a fixed 10 km.

**Uses:** [D14](#d14-probability-of-collision). Also [D38](#d38-covariance-propagation), [D39](#d39-catalogue-scale-conjunction-screening). Also [D55](#d55-debris-environment-flux), [D56](#d56-collision-avoidance-optimiser).

**Repo work:**
- A configurable alert threshold (`Simulation.conjunction_km`) in the UI.
- A per-satellite position covariance (set by hand, or from B21's orbit
  determination), carried along and grown with time.
- Pc in the close-approach log, computed from the Hermite closest approach
  that `_approaches` already finds.
- A collision event when two objects actually meet, feeding B28's breakup.

### B11. Additions that need a scope decision

These are real gaps, but each needs a decision under [SCOPE.md](SCOPE.md)
because it involves the Sun or the Moon in an orbital calculation:

| Item | Why it matters | Scope conflict |
|---|---|---|
| Atmospheric diurnal bulge (Harris-Priester, Jacchia, MSIS) | ×2-3 density day/night at 400-500 km | Bulge direction follows the Sun |
| Solar-flux-driven density (F10.7, Ap) | ×10 over the solar cycle | Solar *activity* input, not position; arguably in scope as "atmosphere state" |
| Thermospheric winds from a full model (HWM14) | A few percent of drag; slow inclination change | Winds depend on local solar time |
| Ionospheric delays from a full model | Tens of metres on GPS ranges | Electron content is driven by sunlight |
| Sun-pointing solar panels with attitude-dependent drag | Drag area follows the Sun | The orbit would depend on the Sun's direction |
| Earth albedo radiation pressure | about 1e-8 m/s² at A/m = 0.01 m²/kg, dayside | Reflected sunlight |
| Solid Earth and ocean tides (changes in C20 and others) | metres to tens of metres per day in LEO | Raised by the Sun and Moon |
| Sun/Moon third body, SRP | Dominant at GEO and in HEO (inclination growth about 0.85 deg/yr at GEO) | Explicitly removed |

### B12. Verification additions

**Uses:** [D15](#d15-reference-data-for-validation),
[D36](#d36-density-calibration-from-tle-decay). Also [D58](#d58-physics-invariant-property-tests).

**Regression tests for Part A:**
- Rendezvous with J2 on (A2). `rendezvous.json` runs two-body, which hid the
  problem.
- Polar Lambert, target on each side of the plane (A1).
- Altitude readouts, CSV and batch summary agree (A3).
- An SSO keeps a constant LTAN over a year (A5).

**Comparisons against outside data**, using the files built in
[D15](#d15-reference-data-for-validation):
- **External reference ephemeris:** nothing compares a run against an
  independent propagator for the same force model. A test should load D15's
  reference states and compare at the 10 m level. That catches frame (A4)
  and force-model regressions.
- **Drag decay magnitude:** `test_drag_decays_low_orbit_and_detects_reentry`
  checks that decay *happens*, not how fast. Compare against D15's decay
  cases.
- **Launch against flight data:** fly each preset vehicle and compare with
  D15's flight events (e.g. Falcon 9 MECO at roughly T+150 s, 65-80 km and
  about 2.3 km/s on LEO missions), to bound "realistic in kind".

### B13. Re-entry to the ground

**Why:** A satellite is removed the moment it is below 80 km and descending
([`simulation.py:64`](../satflight/simulation.py#L64)). Today there is no
impact point, no peak heating and no peak deceleration.

**Uses:** [D16](#d16-entry-physics). Also [D44](#d44-component-demise-model). Also [D53](#d53-capsule-aerodynamics), [D54](#d54-parachute-descent-and-landing-dispersion), [D59](#d59-materials-dataset).

**Repo work:**
- Below 80 km, hand the object to D16's entry model instead of removing it,
  with smaller steps, down to the ground (or terrain, D9).
- Log peak heating, peak deceleration and breakup; show the impact point on
  the map. With B23 it becomes an impact footprint.
- Capsule properties (nose radius, lift-to-drag ratio) on `Satellite` and in
  the dialogs.

### B14. Station-keeping and orbit maintenance

**Why:** Real satellites hold their orbits; here they only drift.
Maintenance also shows the costs drag and C22 impose, which is the point of
modelling them.

**Uses:** [D41](#d41-station-keeping-budget-calculators).

**Repo work:**
- **LEO altitude band:** when the mean altitude falls below a set floor,
  schedule a reboost (two burns) back to the ceiling. Log the delta-v per
  month.
- **GEO east-west:** C22 pulls a satellite away from its slot. When the
  longitude drifts out of a deadband (e.g. ±0.05 deg), burn to reverse the
  drift. Typical cost is up to about 2 m/s per year, depending on longitude
  (estimated from published values). North-south control fights the Sun and
  Moon, so it is out of scope.
- **Constellation phasing:** keep each member's along-track position within a
  window of its slot.
- These run as rules checked on grid steps, like the existing events. Every
  burn draws on the propellant budget (B3), which gives each satellite a
  lifetime.

### B15. Low-thrust burns

**Why:** Finite burns point in one fixed direction in the VNB, RSW or ECI
frame. That suits a short chemical burn, but an electric thruster burns for
days or months, with the direction steered continuously.

**Uses:** [D17](#d17-low-thrust-steering-laws). Also [D48](#d48-thruster-performance-models). Also [D61](#d61-electrodynamic-tether-force).

**Repo work:**
- A `steering` option on finite burns, evaluated inside
  `Simulation._derivative` instead of `finite_direction`.
- Burns long enough to need a stop condition ("until a = ...") rather than a
  duration.
- Thrusting only in sunlight, using the shadow already computed. That only
  switches the engine off, so it stays within scope.
- Needs the propellant budget (B3).

### B16. More maneuvers in the planner

**Why:** The planners cover Hohmann, bi-elliptic, plane change at a node and
Lambert rendezvous. Several standard maneuvers are missing.

**Uses:** [D18](#d18-maneuver-calculators). Also [D40](#d40-transfer-grid-search). Also [D56](#d56-collision-avoidance-optimiser).

**Repo work:**
- New entries in the maneuver dialog and `planner.py` for phasing, RAAN-only
  change, combined Hohmann and plane change, apsidal rotation and deorbit
  burns, turning D18's numbers into scheduled `Maneuver`s.
- Collision avoidance: a burn planned from a logged close approach (B10).

### B17. Orbit design keywords

**Why:** Scenarios accept `"i": "sso"`, but the other classic orbit designs
have to be worked out by hand.

**Uses:** [D19](#d19-orbit-design-solvers). Also [D42](#d42-constellation-design-calculators). Also [D62](#d62-analytic-ground-track-predictor).

**Repo work:**
- `"repeat": "k/d"`, `"frozen": true` and `"ltan": "10:30"` in orbit specs
  (`scenario.orbit_state`) and in the Add-satellite dialog.
- Depends on full mean elements (B4) to place the result correctly.

### B18. Orbital lifetime readout

**Why:** Decay is visible only by running the simulation for months or years
at high warp.

**Uses:** [D20](#d20-lifetime-estimator). Also [D37](#d37-solar-activity-forecast).

**Repo work:**
- An estimated-lifetime row in the Orbit tab, shown as a range for low, mean
  and high solar activity (it is only as good as the atmosphere, B2).
- A check against the deorbit rules: the IADC 25-year guideline and the
  FCC's 5-year rule for US-licensed LEO satellites.

### B19. Coverage, passes and link readouts

**Why:**
- The simulator draws footprints and logs AOS/LOS, but cannot say how well a
  constellation covers the Earth. The GPS scenario cannot show what
  navigation quality it gives.
- Station passes have an elevation and a range but no range-rate, so no
  Doppler: about ±10 kHz at 437 MHz in LEO (estimated from about 7 km/s along
  the line of sight).
- `analysis.line_of_sight` exists but nothing calls it.

**Uses:** [D21](#d21-coverage-revisit-dop-and-pass-prediction),
[D22](#d22-link-geometry). Also [D42](#d42-constellation-design-calculators), [D45](#d45-launch-site-and-station-network-data). Also [D57](#d57-onboard-gnss-navigation), [D62](#d62-analytic-ground-track-predictor).

**Repo work:**
- A coverage view for constellations (percentage covered, revisit gaps)
  computed from the run's states.
- GNSS DOP at a chosen site, in the station panel.
- Predicted passes (next AOS, maximum elevation, LOS) in the station panel.
- Range-rate and Doppler for a chosen frequency in the station view.
- Inter-satellite visibility through `line_of_sight`, drawn as links for
  relay constellations.

### B20. Relative motion view

**Why:** Rendezvous is planned in inertial coordinates. Proximity operations
are understood in the target's local frame, where the chaser's approach
appears as the familiar cycloid.

**Uses:** [D6](#d6-relative-motion-toolkit). Also [D43](#d43-formation-design).

**Repo work:**
- A plot panel of one satellite relative to another in LVLH/RIC (radial,
  in-track, cross-track) coordinates, drawn from the history buffer.
- The Clohessy-Wiltshire prediction drawn beside the simulated path.
- It pairs with B5's terminal approach.

### B21. Orbit determination in the simulation

**Why:** The simulator knows every state exactly. Operators only estimate
them from tracking measurements. Simulating that would show why predictions
carry uncertainty, and it would feed the covariance B10 needs.

**Uses:** [D23](#d23-orbit-determination),
[D24](#d24-atmospheric-signal-delays). Also [D38](#d38-covariance-propagation), [D49](#d49-tracking-sensor-models). Also [D57](#d57-onboard-gnss-navigation).

**Repo work:**
- Generate measurements from the scenario's ground stations during a run,
  with D24's delays.
- Run D23's estimator with satflight's own propagator as the dynamics.
- Show the estimate, its covariance and its error against the truth.

### B22. Integrators and multi-rate stepping

**Why:**
- Higher-order or multistep integrators would make accurate runs cheaper:
  Gauss-Jackson needs about 1-2 force evaluations per step against DOPRI5's
  6 (estimated).
- High-eccentricity orbits crowd their steps into each perigee pass.
- Decay runs of months to years need an averaged propagator.
- The whole ensemble shares one step, set by its most demanding satellite
  (measured: 24 GPS satellites with `h_max` = 900 s take 140 steps in 12 h
  alone, and 808 once a single 400 km satellite joins). At the default 60 s
  cap this is hidden; it matters once `h_max` is raised. Adding one eccentric
  orbit to the 1584-satellite LEO shell cost only 1 extra step in 6 h
  (measured), so LEO-dominated runs gain nothing.

**Uses:** [D26](#d26-averaged-long-term-propagator),
[D27](#d27-integrators). Also [D51](#d51-robust-two-body-propagator).

**Repo work:**
- Register D27's methods in `integrators.METHODS` and the Physics dialog.
- An averaged-propagator choice in `simulation.PROPAGATORS`.
- Multi-rate stepping: group satellites by orbit type and integrate each
  group with its own step. The physics grid in `Simulation.advance` would
  then need one clock per group, and `tests/test_timestep.py` must still pin
  down frame-rate independence.

### B23. Monte Carlo dispersions

**Why:** Every result is a single deterministic run: one insertion orbit,
one decay date, one miss distance. Real numbers come with spread.

**Repo work:**
- Disperse the inputs: launch thrust, Isp and mass; `density_scale` and Cd;
  burn magnitude and pointing errors; initial-state errors from B21.
- Run the dispersed copies **as one ensemble.** The engine already integrates
  N satellites as one array, so 1000 copies cost one 1000-satellite run.
  Launches use plain-float ascents and would need a vectorised path, or run
  serially.
- Show the results as clouds: an impact footprint (B13), an insertion
  scatter, a decay-date histogram (B18), and a sampled probability of
  collision (B10).
- pymc (C3) can calibrate the input distributions against observed decays.

### B24. Attitude in the ensemble

**Why:** Every satellite is a sphere for drag (cannonball model, already
documented) and has no attitude. Area facing the flow depends on attitude,
and so do pointing, power (B26) and magnetic torques (B25).

**Uses:** [D28](#d28-attitude-dynamics-and-panel-drag). Also [D60](#d60-attitude-control-laws).

**Repo work:**
- An attitude state (quaternion and rate) beside the `(N, 6)` orbit state,
  integrated on the same grid, or simple pointing modes (nadir, velocity,
  inertial) that need no integration.
- Drag from D28's panel model instead of `Cd·A/m` for satellites that have
  panels.
- Scope: Sun-pointing panels need a decision (B11); nadir and velocity
  pointing do not.

### B25. Magnetic field and radiation overlays

**Why:** The Earth's magnetic field and its radiation belts are part of the
Earth environment, so they are in scope. They decide radiation dose, the
South Atlantic Anomaly passes that upset electronics, and the torques
magnetorquers use.

**Uses:** [D29](#d29-magnetic-field-and-radiation-belts). Also [D60](#d60-attitude-control-laws).

**Repo work:**
- A South Atlantic Anomaly / inner-belt overlay on the ground-track map, and
  entry/exit events in the log.
- Magnetic torques for B24's attitude.
- Solar particle events come from the Sun, so they stay out of scope.

### B26. Power and thermal readouts

**Why:** The simulator already knows when each satellite is in shadow and its
beta angle. Those numbers matter mainly for power and temperature, which are
not shown.

**Uses:** [D30](#d30-power-and-thermal-models). Also [D59](#d59-materials-dataset).

**Repo work:**
- Array area, efficiency, battery capacity and load as satellite properties.
- Power, battery state of charge and orbit-average temperature in the Orbit
  tab, from the illumination the eclipse code already computes.
- This uses the Sun's position only for things that do not change the orbit,
  which SCOPE.md allows.

### B27. Launch guidance and mission additions

**Why:** B6 covers the vehicle's physics. These are missing mission
capabilities, all inside `launch.py`'s ascent and planner:

**Uses:** [D45](#d45-launch-site-and-station-network-data). Also [D63](#d63-optimal-ascent-trajectories).

**Repo work:**
- **Dogleg ascents:** today an inclination below the pad's latitude is
  refused. A dogleg steers out of plane during the ascent and pays for it in
  delta-v.
- **Parking orbit, coast and restart:** guidance with several burns, for
  direct GTO and higher orbits.
- **Booster recovery:** a first-stage boostback and landing burn, or a
  downrange droneship landing. It needs B13's descent physics and leftover
  propellant planning.
- **Failures:** engine-out with the remaining engines, or an early shutdown,
  as outcomes the planner must survive.

### B28. Fragmentation events and catalogue-scale runs

**Why:** The debris side of space flight cannot be shown: a collision, a
battery explosion or an anti-satellite test, and the cloud that spreads
around the orbit afterwards.

**Uses:** [D31](#d31-breakup-fragment-generator). Also [D39](#d39-catalogue-scale-conjunction-screening). Also [D55](#d55-debris-environment-flux).

**Repo work:**
- Breakup events (collision from B10, explosion or test scheduled by the
  user) that add D31's fragments to the ensemble as ordinary objects with
  their own A/m. J2 then spreads the cloud's planes, and drag sorts it by
  A/m: a textbook picture this simulator could show well.
- **Catalogue scale:** tens of thousands of tracked objects. The grid pairing
  costs about 2 ms per step at 1584 satellites (PHYSICS.md). By linear
  scaling (estimated) it would be roughly 40 ms per step for 30,000, plus
  about 25 s of integration per simulated 6 h, scaled from 1.3 s measured for
  1584 satellites under Cowell. Rendering and the history buffer
  (`History.BUDGET`) would need checking at that size.

### B29. Differential drag and formation control

**Why:** Small satellites without thrusters, Planet's Dove constellation for
example, phase themselves by changing their drag area: flying edge-on or
face-on to the flow. The model is all Earth: the atmosphere and attitude.

**Uses:** [D43](#d43-formation-design).

**Repo work:**
- A per-satellite "drag mode" (low or high area) as a schedulable event; this
  needs a time-varying area per satellite.
- A control law that drives a satellite's along-track position toward its
  slot by switching modes.
- A proper version needs B24's attitude; a two-state switch is enough to
  start.

### B30. Aerobraking and aerocapture

**Why:** The atmosphere can remove orbital energy without propellant. Dipping
the perigee into the upper atmosphere on each pass lowers the apogee. The
Japanese probe Hiten demonstrated aerobraking at Earth in 1991.

**Uses:** [D16](#d16-entry-physics) for heating and deceleration limits and
for lift and bank control. Also [D53](#d53-capsule-aerodynamics).

**Repo work:**
- A planner that sets the perigee altitude from a target drag per pass, and
  raises it with a small burn at apogee when the limits would be exceeded.
- Aerocapture: a single deep pass that turns a hyperbolic or highly
  elliptical arrival into an orbit.

### B31. Imaging targets

**Why:** Earth observation is the main reason LEO satellites exist. The
simulator can show where a satellite is, but not when it can photograph a
place.

**Uses:** [D32](#d32-imaging-access-windows).

**Repo work:**
- Ground targets in scenarios, like stations but with imaging constraints:
  maximum off-nadir angle, minimum Sun elevation at the target (optical) or
  none (radar).
- Imaging windows over the coming days and revisit time per target, in a
  panel and on the map.
- The Sun is used only to light the target, which SCOPE.md allows.

### B32. Observer visibility

**Why:** Whether a satellite can be seen with the naked eye, and how bright it
is, is a popular question (Starlink trains, ISS passes). The simulator
already knows whether a satellite is sunlit, where the Sun is, and the look
angles from a site.

**Uses:** [D33](#d33-satellite-brightness-and-visibility). Also [D49](#d49-tracking-sensor-models).

**Repo work:**
- An "observer" kind of ground site, with visible passes and magnitudes.
- An albedo-area property per satellite.
- Lighting only, so within scope.

### B33. Deployment, docking and configuration changes

**Why:** Satellites are added only from elements or states. Real ones are
released from a dispenser or a station, and some dock and leave together. A
satellite's area and mass are fixed except for propellant.

**Uses:** [D43](#d43-formation-design).

**Repo work:**
- **Deployment:** release a satellite from a parent with a separation speed
  and direction. Examples: a Starlink stack released over a few minutes, or
  a CubeSat deployed from the ISS at about 1 m/s. The parent's state and mass
  change by the reaction.
- **Docking:** after a successful rendezvous (B5), merge two satellites into
  one with combined mass, area and propellant; **undocking** splits them
  again. The ensemble, history and close-approach bookkeeping must follow,
  as they do for launches.
- **Configuration events:** deploying a drag sail at end of life (a large
  area increase, which shortens the lifetime, B18), extending solar panels,
  jettisoning hardware.

### B34. Scenarios from live data

**Why:** Every scenario is synthetic or uses a pasted TLE. Live data would
make "what is up there now" scenarios, and give real outcomes to compare
predictions against.

**Uses:** [D3](#d3-data-access-space-weather-catalogues-and-re-entries). Also [D46](#d46-object-properties-dataset).

**Repo work:**
- A scenario builder: current TLEs or OMMs for a named constellation (GPS,
  Starlink, the ISS) or the whole catalogue (with B28's catalogue-scale work).
- Current F10.7 and Ap fed to the atmosphere (B2).
- Published re-entry predictions shown beside B13's and B18's.
- Network access must stay optional, since the app runs offline today.

### B35. Relativistic clock readout

**Why:** The best-known everyday effect of relativity is on satellite
clocks. A GPS clock runs fast relative to the ground by **+45.7 µs/day from
weaker gravity** and slow by **-7.2 µs/day from its speed: +38.4 µs/day net**
(computed with satflight's constants). Uncorrected, that would be about
11 km of ranging error per day.

**Uses:** [D12](#d12-relativity-and-earth-infrared-pressure).

**Repo work:** A per-satellite row in the Orbit tab and a glossary entry. It
changes no orbit.

### B36. Ballistic trajectory planner

**Why:** The `surface` orbit spec launches a point at a given speed, azimuth
and flight-path angle, but the user has to guess those values to reach a
place. The open-loop launch mode has the same problem for sounding rockets.

**Uses:** [D35](#d35-ballistic-trajectory-solver).

**Repo work:**
- A "target" option for `surface` orbit specs (a latitude and longitude to
  reach), filled in by D35.
- Report the burnout speed, flight time and apogee in the dialog.
- With drag on, correct the solution by differential correction against the
  real force model (as in B5).

---

# Part C: GitHub repositories that could be integrated

I went through all 98 repositories starred by `loggger101` (as of 2026-10-02).
20 of them could help with the items in Parts A, B and D. Five well-known
repositories you have not starred are listed separately in
[C5](#c5-not-starred-but-worth-considering).

## C1. How a repository can be used

satflight is MIT-licensed, so a repository's licence decides how it can be
used:

| Mode | Meaning | Licences that allow it |
|---|---|---|
| **Dependency** | `pip install` it and import it at run time | MIT, BSD, Apache-2.0, MPL-2.0 (if used unmodified) |
| **Port** | Translate an algorithm into satflight's numpy code, with credit | MIT, BSD, Apache-2.0 |
| **Test oracle** | Use it only in tests or offline scripts to produce reference numbers | any permissive licence |
| **Reference only** | Read it for ideas and compare results by hand; never copy code or import it from satflight | GPL-3.0, AGPL-3.0 (as already decided for celmech) |
| **Data** | Use its published data, under each dataset's own licence | per dataset |

One more constraint matters. satflight's hot path is vectorised numpy over the
whole `(N, 6)` ensemble, and the launch model uses plain floats for speed.
Libraries written in Rust, C++ or Java that are called once per satellite per
step would slow it down. Those work better as test oracles or as sources to
port from than as run-time dependencies.

## C2. Summary of repositories

| Repository | Licence | Best mode | Helps with |
|---|---|---|---|
| [duncaneddy/brahe](https://github.com/duncaneddy/brahe) | MIT | Test oracle, port | A4, A7, A10, A11, D1, D2, D3, D4, D6, D8, D10, D11, D12, D15, D21, D23, D25, D27, D29, D32, D34, D38 |
| [skyfielders/python-skyfield](https://github.com/skyfielders/python-skyfield) | MIT | Dependency | A4, A21, B7, D8, D11, D21, D32, D33 |
| [astropy/astropy](https://github.com/astropy/astropy) | BSD-3 | Test oracle | A4, A21, D8, D11, D15 |
| [astropy/astroquery](https://github.com/astropy/astroquery) | BSD-3 | Test oracle (data) | D15 |
| [esa/pykep](https://github.com/esa/pykep) | MPL-2.0 | Dependency or test oracle | A1, A16, D5, D51 |
| [nyx-space/hifitime](https://github.com/nyx-space/hifitime) | MPL-2.0 | Dependency | A21, D8 |
| [HIPS/autograd](https://github.com/HIPS/autograd) | MIT | Dependency | B5, B10, D23, D38 |
| [sympy/sympy](https://github.com/sympy/sympy) | BSD-3 | Dev tool | A13, D1, D4 |
| [esa/pygmo2](https://github.com/esa/pygmo2) | MPL-2.0 | Dependency (optional) | B6, D17, D18, D40, D42, D63 |
| [OpenSCvx/OpenSCvx](https://github.com/OpenSCvx/OpenSCvx) | Apache-2.0 | Offline experiments | B5, B6, B27, D17, D63 |
| [loggger101/spacecost](https://github.com/loggger101/spacecost) | MIT | Data, dependency | B3, D7, D15, D47, D48 |
| [Karmanplus/prospector](https://github.com/Karmanplus/prospector) | Apache-2.0 | Data | B3, D17, D48 |
| [juliensimon/space-datasets](https://github.com/juliensimon/space-datasets) | per dataset | Data | B7, B28, B34, D3, D7, D14, D15, D20, D36, D46 |
| [pymc-devs/pymc](https://github.com/pymc-devs/pymc) | Apache-2.0 | Offline experiments | B2, B7, B23, D36 |
| [hannorein/rebound](https://github.com/hannorein/rebound) | GPL-3.0 | Reference only | A18, D15, D27 |
| [dtamayo/reboundx](https://github.com/dtamayo/reboundx) | GPL-3.0 | Reference only | D12, D15 |
| [nyx-space/nyx](https://github.com/nyx-space/nyx) | AGPL-3.0 | Reference only | B10, D1, D2, D15 |
| [cuspaceflight/CamPyRoS](https://github.com/cuspaceflight/CamPyRoS) | GPL-3.0 | Reference only | B6, B23, D16, D28 |
| [shadden/celmech](https://github.com/shadden/celmech) | GPL-3.0 | Reference only | D4 |
| [matthewholman/assist](https://github.com/matthewholman/assist) | GPL-3.0 | Reference only | B11 (only if the Sun and Moon return) |

If you adopt only three, I'd pick:

1. **brahe** as the test oracle for D15's reference data, and as the source to port gravity,
   mean elements and the atmosphere from.
2. **skyfield** for time scales and J2000/TEME frames (A4, A21, D8). It is
   pure Python and numpy, like satflight.
3. **pymsis** (not starred, see C5) for the atmosphere (D2). It takes and
   returns numpy arrays, so it fits the ensemble hot path.

## C3. Starred repositories that fit well

### duncaneddy/brahe

**Licence:** MIT. **Install:** `pip install brahe` (Rust core with Python
bindings). **Units:** SI (metres), where satflight uses km.

This is the closest match in your starred list: a modern satellite-dynamics
library covering most of Part D. Its source tree contains:

- **Gravity (D1):** spherical-harmonic gravity with bundled EGM2008 (degree
  120), GGM05S and JGM3 coefficient files, plus an ICGEM file loader.
- **Atmosphere (D2, D3):** exponential, Harris-Priester and NRLMSISE-00 drag
  models, and bundled space-weather files (F10.7 flux table, CelesTrak `sw`
  file).
- **Mean elements (D4, A7, A11):** analytical Brouwer-Lyddane mean-to-
  osculating conversion in both directions, and equinoctial elements.
- **Frames (A4, D8):** GCRF, ITRF and TEME with bias-precession-nutation
  models and Earth orientation parameters.
- **Events and access (A10, D21, D32):** event detection during propagation, and
  ground-station access windows with elevation constraints.
- **Relativity (D12)** and **covariance propagation, RTN and relative orbital
  elements (D6, D14).**
- **SGP4 and a CelesTrak client (D3, D25).**
- **Beyond the core force model:** the IGRF and WMM-HR magnetic field models
  (D29); batch least squares, EKF and UKF orbit determination with
  azimuth/elevation/range measurements (D23); DP54, RKF45, RKF78 and RKN1210
  integrators (D27); Earth orientation parameters (D8); CCSDS messages (D34).
- Third body, SRP and tides as well, which fall under the B11 scope decision.

How to use it: as the **test oracle** for D15 and B12 (propagate the same state with
the same force model in both and compare), and as a source to **port**
algorithms from (the MIT licence allows it). Before using it at run time,
check whether its Python API vectorises over many satellites;
`initialize_eop()` also downloads Earth-orientation data.

### skyfielders/python-skyfield

**Licence:** MIT. Pure Python and numpy, built on the `sgp4` package.

- Time scales with leap seconds, TT and UT1 (A21).
- TEME, the true-of-date frames and GCRS, with IAU 2000 precession and
  nutation: exactly what a J2000 export needs (A4, D8).
- Atmospheric refraction for topocentric altitude (D11), and twilight and
  pass events for observers (D21, D33).
- An `EarthSatellite` (SGP4) class that would make SGP4 support routine (B7).

It is the lightest-weight way to fix the frame and time items, and matches
the project's "pure Python" preference.

### astropy/astropy

**Licence:** BSD-3. Heavier than skyfield, so best kept as a test-only
dependency:

- `Time` with every time scale, and IERS-based UT1 (A21).
- `TEME`, `GCRS` and `ITRS` frames (A4, D8): a test can check satflight's
  of-date frame against an independent implementation.
- `AltAz` with refraction (D11).

### astropy/astroquery

**Licence:** BSD-3. Its JPL Horizons interface can fetch reference
ephemerides of some Earth satellites (the ISS among them) as test data for
D15. Horizons' own satellite ephemerides come from tracking data or TLEs, so
they are a sanity check rather than truth.

### esa/pykep

**Licence:** MPL-2.0 (file-level copyleft: fine as an unmodified dependency).
**Install:** `pip install pykep`, or conda-forge (which still ships v1 while
v3 stabilises).

- Izzo's Lambert solver with multiple revolutions, robust near 180 deg: the
  D5 solver and the fix for the A16 edge cases.
- A test oracle for A1: compare satflight's chosen branch with pykep's
  prograde solution.

### nyx-space/hifitime

**Licence:** MPL-2.0. **Install:** `pip install hifitime`.

Leap-second-correct conversions between UTC, TAI, TT and GPS time, and UT1
when built with that feature. A small dependency for A21, and for the time
side of D8.

### HIPS/autograd

**Licence:** MIT. Differentiates numpy code automatically. It could produce
the state-transition matrix that differential correction needs (B5, D23) and
propagate covariance (B10) from satflight's existing numpy force model.

Caveat: autograd does not support in-place assignment into arrays, so
`ForceModel.acceleration` and the integrators would need checking. Finite
differences are the simpler fallback.

### sympy/sympy

**Licence:** BSD-3. A development tool, not a run-time dependency:

- Derive and check the Brouwer short-period expressions (D4).
- Generate or test higher-degree gravity terms symbolically (D1), as the
  existing tests do numerically for J2-J4.
- Derive the J4 term of the sun-synchronous condition (A13).

### esa/pygmo2

**Licence:** MPL-2.0. Global optimisation (differential evolution, CMA-ES,
etc.). The launch planner tunes only the kick angle by grid plus
golden-section search. pygmo could tune several ascent parameters together
(vertical-rise time, kick, throttle profile, coast lengths) for B6, or
optimise multi-impulse transfers and low-thrust laws (D18, D17). It would be an optional, opt-in
planner, since today's kick search takes about 0.3 s.

### OpenSCvx/OpenSCvx

**Licence:** Apache-2.0. Successive convexification on JAX: optimal
trajectories with constraints, such as fuel-optimal ascent or a rendezvous
approach with a keep-out zone (B5, B6, B27, D17). JAX is a heavy dependency, so this
suits offline experiments or a separate "optimise" tool rather than the core.

### loggger101/spacecost (your own)

**Licence:** MIT.

- The `propellants` table (vacuum Isp, bulk density, thruster device) is a
  ready source of engine and propellant presets for the propellant budget
  (B3).
- The `launch_vehicles` table (payload to LEO and GTO) gives a check for the
  launch model (D15, B12): fly each preset vehicle to its quoted payload and see
  whether it reaches orbit. It could also seed more vehicle presets (D7).

### Karmanplus/prospector

**Licence:** Apache-2.0. A heliocentric low-thrust mission designer, so its
physics does not apply here. Its YAML configs, though, hold real engines,
propellants and vehicles from flown missions (Dawn, Psyche, Hayabusa2, DART):
a source of electric-propulsion presets for finite burns once B3 exists (D17).

### juliensimon/space-datasets

**Licence:** none declared for the repository as a whole; each dataset
carries its source's terms. Published on Hugging Face as Parquet:

- **F10.7, Kp and CelesTrak space-weather files:** the input a solar-activity
  atmosphere needs (D3, B2).
- **Re-entry events (35K) and TLE history since 1959:** real decays to
  validate drag against (D15, D20, B12). The full TLE history is 10.9 GB, so take
  single objects.
- **CelesTrak SOCRATES conjunctions,** with miss distance and probability of
  collision: a sanity check for D14.
- **GCAT launch vehicles and stages:** data for more launch presets (D7).
- **Latest constellation TLEs:** ready-made scenario inputs (B34).

### pymc-devs/pymc

**Licence:** Apache-2.0 (GitHub shows "NOASSERTION"). Bayesian inference. It
could fit `density_scale`, or a satellite's `Cd·A/m`, to an object's observed
decay in its TLE history, with uncertainty (B2, B7, B23). An offline experiment,
not a dependency.

## C4. Starred repositories to use as reference only (GPL / AGPL)

These cannot be copied into or imported by MIT-licensed satflight. They can
be read for ideas, or run separately to produce numbers to compare against.

- **hannorein/rebound (GPL-3.0).** The source of the WHFast idea already
  used. Its IAS15 integrator (adaptive, accurate to machine precision) could
  produce offline reference trajectories (D15). Its symplectic-corrector notes
  bear on A18.
- **dtamayo/reboundx (GPL-3.0).** Its `gr` effects show how general
  relativity is usually implemented (D12). Its `gravitational_harmonics` and
  `gas_drag` effects can cross-check satflight's J2/J4 and drag.
- **nyx-space/nyx (AGPL-3.0).** A high-fidelity, validated astrodynamics
  toolkit (harmonics, drag, covariance, orbit determination). Use it as an
  offline cross-check for D1, D2, B10 and D15. Because of the AGPL, never
  vendor or link it.
- **cuspaceflight/CamPyRoS (GPL-3.0).** A 6-DOF rocket simulator with
  variable mass and inertia, live wind data and aerodynamic heating. A design
  reference for B6 and D7 (Mach-dependent drag, winds), D16 (aerodynamic
  heating) and D28 (attitude).
- **shadden/celmech (GPL-3.0).** Already used for ideas (mean vs osculating
  semi-major axis). Its averaging and secular-theory material bears on D4.
- **matthewholman/assist (GPL-3.0).** Ephemeris-quality perturbations from
  JPL DE ephemerides. Relevant only if the Sun and Moon are brought back into
  scope (B11).

## C5. Not starred but worth considering

| Repository | Licence | Why |
|---|---|---|
| [SWxTREC/pymsis](https://github.com/SWxTREC/pymsis) | MIT | NRLMSIS 2.x densities from numpy arrays in, arrays out. The most direct fit for D2 and B2's vectorised hot path |
| [brandon-rhodes/python-sgp4](https://github.com/brandon-rhodes/python-sgp4) | MIT | Already an optional dependency. Making it a required one fixes most of B7 and A17's fallback path, and D25 needs it |
| [pleiszenburg/hapsira](https://github.com/pleiszenburg/hapsira) | MIT | Maintained fork of poliastro. Pure Python (numba) Izzo Lambert and perturbation models, easier to port from than pykep (D5) |
| [CS-SI/Orekit](https://github.com/CS-SI/Orekit) | Apache-2.0 | Java, with Python access through `orekit_jpype`. The industry-standard reference propagator for D15 |
| [HypothesisWorks/hypothesis](https://github.com/HypothesisWorks/hypothesis) | MPL-2.0 (GitHub shows NOASSERTION) | Property-based testing: generates the random valid inputs D58's invariant tests need |

## C6. Starred repositories that do not help with the physics

- **Visualisation:** bilawalsidhu/gods-eye-view, typpo/spacekit,
  CelestiaProject/Celestia, julie-dujardin/space-map.
- **Heliocentric work:** loggger101/AsteroidCatalog, loggger101/economicspace,
  loggger101/asteroid-belt-gradient (asteroid science and mining).
- **General tools:** pytest is already used. d3, streamlit, nicegui, polars,
  pyomo, z3, mesa, pygame and the agent-skill, style and data-engineering
  lists are not physics.

---

# Part D: Standalone additions that can be built outside the repo

Each item here is a self-contained piece of physics or data: a function, a
small module or a dataset. It can be written and tested with no reference to
satflight's code, then plugged in by the Part B item it serves. Each one
could live in its own repository, as spacecost does, or in a `satflight/ext/`
folder written to these rules.

## D0. Conventions every Part D item follows

Following these makes a piece drop into satflight without adapters:

- **Units:** km, s, kg and radians. Accelerations in km/s²; densities in
  kg/m³; `Cd·A/m` in m²/kg; frequencies in Hz.
- **Arrays:** positions and velocities are numpy arrays of shape `(N, 3)`
  (a single `(3,)` vector also accepted). Functions are vectorised over N;
  scalars broadcast.
- **Frames:** "inertial" means satflight's ECI, which is mean/true-of-date and
  TEME-like (see A4). Earth-fixed inputs are ECEF; when a function needs
  both, it takes the GMST angle `theta` (rad) and rotates by
  `eci_to_ecef(r, theta) = Rz(theta) r`.
- **Time:** Julian dates as floats (UTC unless the item says TT), the way
  `timeutil.Clock.jd` gives them; durations in seconds.
- **Constants:** pass them as keyword arguments with satflight's values as
  defaults (`mu=398600.4418`, `re=6378.137`, `f=1/298.257223563`,
  `omega=7.292115146706979e-5`), so the caller can pass `satflight.constants`.
- **Dependencies:** numpy only, unless the item says otherwise. No imports
  from satflight. Data files are loaded from a path the caller gives; no
  global state.
- **Licence:** MIT-compatible. Code may be ported from the permissive
  repositories in Part C (with credit), never from the GPL/AGPL ones.
- **Tests:** each item says how to test it on its own, against published
  values or a Part C repository.

## Summary of standalone additions

| ID | Piece | Priority | Serves |
|---|---|---|---|
| [D1](#d1-spherical-harmonic-gravity) | Spherical-harmonic gravity from EGM-96/2008 coefficients | **High** | B1 |
| [D2](#d2-atmospheric-density-model) | Density from an NRLMSISE/Jacchia-type model; free-molecular Cd | **High** | B2 |
| [D3](#d3-data-access-space-weather-catalogues-and-re-entries) | Readers and cached fetchers for space weather, TLE/OMM catalogues, re-entry records | Medium | B2, B34 |
| [D4](#d4-mean-elements-and-equinoctial-elements) | Brouwer-Lyddane mean ↔ osculating elements; equinoctial elements | Medium | B4, B17 |
| [D5](#d5-lambert-solver) | Multi-revolution Lambert solver with an explicit direction | Medium | B5, A1, A16 |
| [D6](#d6-relative-motion-toolkit) | RIC frame, Clohessy-Wiltshire propagation and targeting, relative elements | Medium | B5, B20 |
| [D7](#d7-launch-aerodynamics-and-vehicle-data) | Speed of sound, Cd(Mach) curves, stage-level vehicle data | Medium | B6 |
| [D8](#d8-precession-nutation-and-earth-orientation) | J2000 ↔ of-date transform; EOP reader | Medium | B8, B9 |
| [D9](#d9-geoid-and-terrain-heights) | Geoid and terrain height lookups | Low | B9, B13 |
| [D10](#d10-eclipse-with-an-oblate-earth-and-atmosphere) | Shadow fraction with an oblate Earth and atmospheric enlargement | Low | B9 |
| [D11](#d11-topocentric-refinements) | Refraction, ellipsoidal footprints, azimuth/elevation masks | Low | B9 |
| [D12](#d12-relativity-and-earth-infrared-pressure) | Relativistic acceleration and clock rates; Earth IR pressure | Low | B9, B35 |
| [D13](#d13-thermospheric-winds) | Wind velocity in the upper atmosphere | Low | B2 |
| [D14](#d14-probability-of-collision) | Probability of collision from covariances | Low | B10 |
| [D15](#d15-reference-data-for-validation) | Reference ephemerides, decay cases and launch flight data | **High** | B12 |
| [D16](#d16-entry-physics) | Entry dynamics, heating, breakup, lift and bank | Medium | B13, B30 |
| [D17](#d17-low-thrust-steering-laws) | Tangential, Edelbaum and Q-law steering | Medium | B15 |
| [D18](#d18-maneuver-calculators) | Phasing, RAAN change, combined plane change, apsidal rotation, deorbit, avoidance | Medium | B16 |
| [D19](#d19-orbit-design-solvers) | Repeat ground track, frozen orbit, RAAN for an LTAN | Medium | B17 |
| [D20](#d20-lifetime-estimator) | Orbital lifetime from an atmosphere model | Medium | B18 |
| [D21](#d21-coverage-revisit-dop-and-pass-prediction) | Coverage and revisit statistics, DOP, pass prediction | Medium | B19 |
| [D22](#d22-link-geometry) | Range-rate, Doppler, path loss, link margin | Low | B19 |
| [D23](#d23-orbit-determination) | Measurement models; batch least squares and EKF | Low | B21 |
| [D24](#d24-atmospheric-signal-delays) | Tropospheric and ionospheric delays | Low | B21 |
| [D25](#d25-tle-tools) | B* ↔ Cd·A/m; fitting a TLE to states | Medium | B7 |
| [D26](#d26-averaged-long-term-propagator) | Mean-element propagator with averaged J2 and drag | Medium | B22 |
| [D27](#d27-integrators) | DOP853, RKF78, Gauss-Jackson, Sundman regularisation | Medium | B22 |
| [D28](#d28-attitude-dynamics-and-panel-drag) | Rigid-body attitude, gravity-gradient torque, flat-plate drag and lift | Low | B24 |
| [D29](#d29-magnetic-field-and-radiation-belts) | IGRF field, L-shell, South Atlantic Anomaly flag | Low | B25 |
| [D30](#d30-power-and-thermal-models) | Solar-array power, battery state of charge, one-node thermal | Low | B26 |
| [D31](#d31-breakup-fragment-generator) | NASA Standard Breakup Model fragments | Medium | B28 |
| [D32](#d32-imaging-access-windows) | Imaging windows with off-nadir and Sun-elevation limits | Medium | B31 |
| [D33](#d33-satellite-brightness-and-visibility) | Visual magnitude and visibility to an observer | Low | B32 |
| [D34](#d34-ccsds-message-reader-and-writer) | OEM, OPM, OMM and CDM in and out | Medium | B8 |
| [D35](#d35-ballistic-trajectory-solver) | Minimum-energy ballistic path between two places on the rotating Earth | Low | B36 |
| [D36](#d36-density-calibration-from-tle-decay) | A density scale fitted to an object's observed decay | Medium | B2, B12 |
| [D37](#d37-solar-activity-forecast) | F10.7 and Ap beyond the data: solar-cycle model or SWPC forecast | Medium | B2, B18 |
| [D38](#d38-covariance-propagation) | State uncertainty propagated by STM or unscented transform | Low | B10, B21 |
| [D39](#d39-catalogue-scale-conjunction-screening) | Hoots-style filters and a sieve for catalogue-scale screening | Medium | B28, B10 |
| [D40](#d40-transfer-grid-search) | Porkchop grid of Lambert transfers between two Earth orbits | Low | B5, B16 |
| [D41](#d41-station-keeping-budget-calculators) | Yearly delta-v for drag makeup and GEO east-west control | Medium | B14 |
| [D42](#d42-constellation-design-calculators) | Walker and streets-of-coverage sizing for coverage | Low | B19, B17 |
| [D43](#d43-formation-design) | Passively safe formations by e/i-vector separation; J2-aware drift | Low | B20, B29, B33 |
| [D44](#d44-component-demise-model) | Which parts of a re-entering satellite survive to the ground | Low | B13 |
| [D45](#d45-launch-site-and-station-network-data) | Launch sites with azimuth limits; tracking station networks | Low | B27, B19 |
| [D46](#d46-object-properties-dataset) | Mass, size and cross-section of catalogued objects | Medium | B7, B34 |
| [D47](#d47-launch-performance-estimator) | Quick payload-to-orbit estimate for a vehicle | Low | B6 |
| [D48](#d48-thruster-performance-models) | Chemical and electric thruster performance; real thruster table | Low | B3, B15 |
| [D49](#d49-tracking-sensor-models) | Radar and optical detection of objects by ground sensors | Low | B21, B32 |
| [D50](#d50-closed-form-geodetic-conversion) | Exact closed-form ECEF to geodetic conversion | Low | A3, B9 |
| [D51](#d51-robust-two-body-propagator) | Kepler propagation that reports convergence; e = 1 handled | Low | A16, B22 |
| [D52](#d52-us-standard-atmosphere-1976) | US Standard Atmosphere 1976: temperature, pressure, density, speed of sound | Medium | A15, B6 |
| [D53](#d53-capsule-aerodynamics) | Hypersonic Cd and L/D of capsules by modified Newtonian theory | Low | B13, B30 |
| [D54](#d54-parachute-descent-and-landing-dispersion) | Parachute descent, wind drift and landing dispersion | Low | B13 |
| [D55](#d55-debris-environment-flux) | Flux of small untracked debris through an orbit | Low | B28, B10 |
| [D56](#d56-collision-avoidance-optimiser) | Smallest maneuver that brings Pc below a threshold | Low | B10, B16 |
| [D57](#d57-onboard-gnss-navigation) | GPS pseudoranges and a PVT solution for a LEO satellite | Low | B21, B19 |
| [D58](#d58-physics-invariant-property-tests) | Property-based tests of physics invariants, reusable on any implementation | Medium | B12 |
| [D59](#d59-materials-dataset) | Thermal and optical properties of spacecraft materials | Low | B13, B26 |
| [D60](#d60-attitude-control-laws) | B-dot detumbling, momentum dumping and reaction-wheel pointing | Low | B24, B25 |
| [D61](#d61-electrodynamic-tether-force) | Lorentz force on an electrodynamic tether | Low | B15 |
| [D62](#d62-analytic-ground-track-predictor) | Ground tracks and node longitudes from mean elements alone | Low | B17, B19 |
| [D63](#d63-optimal-ascent-trajectories) | Offline optimal ascents as a benchmark for the guidance | Low | B27, B6 |

### D1. Spherical-harmonic gravity

**Serves:** [B1](#b1-spherical-harmonic-gravity-in-the-force-model).

**Build:** A gravity routine for a full n×m field: Cunningham/Montenbruck-Gill
recursion, or Pines' singularity-free formulation, which has no trouble at
the poles. A reader for ICGEM `.gfc` coefficient files (EGM-96, EGM-2008),
which are fully normalised.

**Interface:**
- `load_gfc(path, n_max, m_max) -> GravityField` with `C`, `S` (normalised,
  shape `(n_max+1, m_max+1)`), `mu`, `re`.
- `accel_harmonics(r_ecef, field, n_max, m_max, include_central=False) ->
  (N, 3)` km/s² in ECEF. The caller rotates to and from ECI.
- `potential_harmonics(r_ecef, field, n_max, m_max) -> (N,)` km²/s², for
  energy checks and gradient tests.

**Test on its own:**
- With only C20 set, it must equal the closed-form J2 acceleration.
- The acceleration equals the numerical gradient of the potential.
- Compare with brahe's spherical-harmonic gravity at random points, to
  1e-12 relative.

**Sources:** brahe (port, MIT) bundles EGM2008 to degree 120, GGM05S and
JGM3. sympy can generate low-degree terms for tests.

### D2. Atmospheric density model

**Serves:** [B2](#b2-a-better-atmosphere-in-the-drag-force).

**Build:** A density function driven by solar and geomagnetic activity:
NRLMSISE-00 or NRLMSIS 2.x (wrapping pymsis, or porting), or Jacchia-71 as
a lighter option. Optionally a free-molecular Cd as a function of altitude
and surface temperature.

**Interface:**
- `density(alt_km, lat, lon, jd, f107, f107a, ap) -> (N,)` kg/m³, with all
  arguments broadcast.
- Scope: the local-time dependence is the diurnal bulge (B11). Provide
  `local_time="model" | "fixed" | "average"` so satflight can switch it off
  until the decision is made.
- `cd_free_molecular(alt_km, ...) -> (N,)` (optional).

**Test on its own:** the NRLMSISE-00 distribution's published test cases;
pymsis results at the same inputs; the density at 400 km should rise about
×10 from solar minimum to maximum.

**Sources:** pymsis (C5, MIT, vectorised); brahe (NRLMSISE-00 and
Harris-Priester, MIT).

### D3. Data access: space weather, catalogues and re-entries

**Serves:** [B2](#b2-a-better-atmosphere-in-the-drag-force), [B34](#b34-scenarios-from-live-data).

**Build:** Readers for the standard files, plus optional fetchers that cache
to disk and work offline from the cache:
- CelesTrak space-weather files (`SW-All.csv` style: F10.7, its 81-day mean,
  Ap).
- GP data (TLE or OMM) by group or catalogue number.
- Re-entry records (object, decay date).

**Interface:**
- `load_space_weather(path) -> SpaceWeather`;
  `SpaceWeather.at(jd) -> (f107, f107a, ap)`.
- `fetch_gp(group_or_catnr, cache_dir, max_age_hours, fmt="tle"|"omm") ->
  str`; returns the cached copy when offline, and raises only if there is
  none.
- `load_reentries(path) -> list of records`.

**Test on its own:** parse small sample files kept as fixtures; a fetch with
the network disabled must return the cached copy.

**Sources:** brahe's CelesTrak and Space-Track clients; the space-datasets
collection (C3) for historic files.

### D4. Mean elements and equinoctial elements

**Serves:** [B4](#b4-mean-elements-in-the-propagator-scenarios-and-readouts), [B17](#b17-orbit-design-keywords).

**Build:** First-order Brouwer-Lyddane (or Kozai) mean ↔ osculating
conversion for all six elements, including J2² and J3 terms where the theory
has them. Conversions to and from equinoctial elements.

**Interface:**
- Elements as `(N, 6)` arrays `[a km, e, i, raan, argp, M]` (radians).
- `osc_to_mean(el, j2, j3=0.0, re=..., mu=...) -> (N, 6)` and
  `mean_to_osc(...)`.
- `coe_to_equinoctial(el) -> (N, 6)` `[a, h, k, p, q, lambda]` and the
  inverse.

**Test on its own:**
- The round trip mean → osculating → mean returns the input to first order
  in J2.
- Average an osculating J2 trajectory (integrated in the test with a simple
  RK4) over one revolution: the average equals the mean elements.
- Compare with brahe's `BROUWER_LYDDANE` method.

**Sources:** brahe (port, MIT); celmech for ideas only (GPL).

### D5. Lambert solver

**Serves:** [B5](#b5-rendezvous-targeting-against-the-real-force-model), [A1](#a1-lambert-picks-the-wrong-way-round-for-polar-orbits), [A16](#a16-solver-edge-cases).

**Build:** Izzo's (or Gooding's) Lambert solver, with multiple revolutions
and robust behaviour near 180 deg.

**Interface:**
- `lambert(r1, r2, tof, mu=..., direction=None, max_revs=0) -> list of
  (v1, v2, revs, branch)`.
- `direction` is a `(3,)` vector: the transfer goes the way round that makes
  `cross(r1, r2)` point along it. Passing the chaser's angular momentum
  removes A1's polar-orbit ambiguity.
- Raise a clear error with no solution; never return an unconverged answer.

**Test on its own:**
- Vallado's and Curtis' worked examples.
- Kepler-propagate `(r1, v1)` for `tof`: it must arrive at `r2`.
- Compare with pykep's `lambert_problem`, including multi-revolution
  branches.
- A polar case with the target plane on each side gives the same branch.

**Sources:** pykep (MPL-2.0); hapsira (C5, MIT, pure Python, easiest to
port).

### D6. Relative motion toolkit

**Serves:** [B5](#b5-rendezvous-targeting-against-the-real-force-model), [B20](#b20-relative-motion-view).

**Build:** The local frame of a target orbit and the linear relative-motion
models.

**Interface:**
- `ric_basis(r_t, v_t) -> (N, 3, 3)`.
- `eci_to_ric(r_c, v_c, r_t, v_t) -> (rho, rho_dot)` and its inverse. The
  rotating-frame velocity includes the frame's angular velocity.
- `cw_propagate(rho0, rho_dot0, n, dt) -> (rho, rho_dot)` for a circular
  target with mean motion `n`.
- `cw_two_impulse(rho0, rho_dot0, rho_f, n, tof) -> (dv1, dv2)`.
- `eci_to_roe(r_c, v_c, r_t, v_t)`: relative orbital elements (optional).

**Test on its own:** for small separations, `cw_propagate` matches a
nonlinear two-body propagation of both satellites over a fraction of an
orbit; textbook cases (a pure along-track offset stays still; a radial offset
drifts at `-3/2 n x0` per second).

**Sources:** brahe's relative-motion module (ECI ↔ RTN, ROE).

### D7. Launch aerodynamics and vehicle data

**Serves:** [B6](#b6-launch-vehicle-physics).

**Build:**
- The speed of sound and temperature from the US Standard Atmosphere 1976.
- Cd(Mach) curves for typical launcher shapes, as small tables.
- A stage-level dataset of real vehicles, in the same fields `launch.Stage`
  uses.

**Interface:**
- `speed_of_sound(alt_km) -> km/s`, plain floats (the ascent's hot path does
  not use numpy).
- `cd_of_mach(mach, table) -> float` (linear interpolation).
- Vehicle records as JSON: `{"name", "stages": [{"name", "thrust" (kN,
  vacuum), "isp_vac", "isp_sl", "propellant", "dry"}], "fairing",
  "fairing_alt", "diameter", "cd_table"}`.

**Test on its own:** US76 tabulated values; vehicle totals against the
sources' liftoff masses; each vehicle's ideal delta-v within the range
expected for its class.

**Sources:** GCAT launch vehicles and stages (space-datasets); spacecost's
launch-vehicle table for payload checks.

### D8. Precession, nutation and Earth orientation

**Serves:** [B8](#b8-import-and-export-j2000-states-and-ccsds-messages), [B9](#b9-smaller-models-plugged-in).

**Build:**
- The IAU-76/80 (or IAU-2006/2000B) precession-nutation matrix between J2000
  and the of-date frame, and TEME ↔ J2000.
- A reader and interpolator for IERS Earth orientation parameters (polar
  motion, UT1-UTC).

**Interface:**
- `j2000_to_of_date(r, v, jd_tt) -> (r, v)` and the inverse.
- `teme_to_j2000(r, v, jd_tt)` and the inverse.
- `load_eop(path) -> EOP`; `EOP.at(jd_utc) -> (xp, yp, dut1)` (rad, rad, s).
- `tt_from_utc(jd_utc) -> jd_tt` (leap seconds table included).

**Test on its own:** Vallado's worked IAU-76/FK5 reduction example
(chapter 3); comparison with astropy's GCRS/TEME frames or skyfield.

**Sources:** skyfield (MIT), astropy (BSD), brahe (MIT); hifitime for time
scales.

### D9. Geoid and terrain heights

**Serves:** [B9](#b9-smaller-models-plugged-in), [B13](#b13-re-entry-to-the-ground).

**Build:** Lookups of geoid undulation (EGM-96 15-arcminute grid) and terrain
height (a coarse global grid), with bilinear interpolation.

**Interface:** `geoid_height(lat, lon) -> (N,)` km above the WGS-84
ellipsoid; `terrain_height(lat, lon) -> (N,)` km above the geoid. Grids are
loaded from caller-given paths.

**Test on its own:** published geoid heights at known points (the EGM-96
range is about -105 to +85 m); terrain at a few known summits and ocean
points.

### D10. Eclipse with an oblate Earth and atmosphere

**Serves:** [B9](#b9-smaller-models-plugged-in).

**Build:** The conical shadow model with the Earth's polar flattening and an
effective atmospheric enlargement of the shadow (about 2 %, the
Danjon/Chauvenet figure).

**Interface:** the same as satflight's `eclipse.shadow_fraction(r_sat,
r_sun)`, with keyword options `oblate=True`, `atmosphere_km=...`, so it can
replace that function directly.

**Test on its own:** with both options off it must equal the spherical model
(copy satflight's function into the test); eclipse entry and exit times
against brahe's eclipse events.

### D11. Topocentric refinements

**Serves:** [B9](#b9-smaller-models-plugged-in).

**Build:** Atmospheric refraction, coverage footprints on the ellipsoid, and
azimuth/elevation masks.

**Interface:**
- `refraction(el, pressure_mbar=1010, temp_c=10) -> d_el` (rad), e.g.
  Bennett's formula.
- `footprint_ellipsoid(lat, lon, alt_km, min_el, n=90) -> (lats, lons)`.
- `visible(az, el, mask) -> bool array`, with `mask` as a table of
  azimuth → minimum elevation.

**Test on its own:** about 0.57 deg of refraction at the horizon and close to
zero at zenith; at the equator the ellipsoidal footprint equals the
spherical one; skyfield's refraction for comparison.

### D12. Relativity and Earth infrared pressure

**Serves:** [B9](#b9-smaller-models-plugged-in), [B35](#b35-relativistic-clock-readout).

**Build:** Three small functions.

**Interface:**
- `accel_relativity(r, v, mu=..., c=299792.458) -> (N, 3)` km/s², the
  post-Newtonian term
  `mu/(c^2 r^3) [(4 mu/r - v^2) r + 4 (r·v) v]`.
- `clock_rate_offset(r, v, mu=..., re=..., c=...) -> (N,)`: the fractional
  rate relative to a clock on the ground (gravitational minus velocity
  term). Multiply by 86400e6 for µs/day.
- `accel_earth_ir(r, area_over_mass, emissivity_flux=230.0) -> (N, 3)`
  km/s², a cannonball model with uniform Earth emission (W/m²).

**Test on its own:**
- GPS at 26,560 km must give +45.7 µs/day gravitational, -7.2 µs/day from
  velocity and +38.4 µs/day net.
- The relativistic acceleration in LEO is about 1.7e-8 m/s².
- The IR pressure is about 1e-8 m/s² at 0.01 m²/kg.
- Compare with brahe's relativity function.

### D13. Thermospheric winds

**Serves:** [B2](#b2-a-better-atmosphere-in-the-drag-force).

**Build:** A wind model for the upper atmosphere: a wrapper around HWM14, or
a simple super-rotation factor.

**Interface:** `wind_ecef(r_ecef, jd, ap=None, model="hwm14" | "rotation",
factor=1.0) -> (N, 3)` km/s, the air velocity relative to the rotating
Earth.

Scope: HWM14 depends on local solar time (B11); the rotation model does not.

**Test on its own:** the published HWM14 check values; the rotation model's
speed equals `(factor - 1) · omega · rho` at distance `rho` from the axis.

### D14. Probability of collision

**Serves:** [B10](#b10-conjunction-outcomes-and-statistics).

**Build:** The two-dimensional probability of collision in the encounter
plane (Foster or Alfano), for a hard-body radius and the combined position
covariance.

**Interface:** `collision_probability(r_rel, v_rel, cov_rel, hard_body_radius)
-> float`, with km and km² inputs; a vectorised version over many
encounters.

**Test on its own:** Alfano's published test cases (2009), which give Pc for a
range of geometries.

### D15. Reference data for validation

**Serves:** [B12](#b12-verification-additions).

**Build:** Files to compare satflight against, generated without satflight:
- **Reference ephemerides:** for a few orbits (ISS-like LEO, SSO, Molniya,
  GEO), states over a day or more, produced by Orekit, brahe or nyx with
  force models matching satflight's options (two-body, J2-J4, C22, the
  exponential atmosphere with a stated Cd·A/m). State the frame explicitly
  (of-date, per A4).
- **Decay cases:** objects with a TLE history and an observed re-entry date.
- **Launch flight data:** published event times, altitudes and speeds (MECO,
  SECO, max-Q) for the preset vehicles.

**Interface:** CSV files with `t_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms`
and a JSON header giving the epoch, frame, force model and generator version;
decay and launch data as small JSON records with their sources.

**Test on its own:** regenerate each file from its script and check it is
byte-identical.

**Sources:** Orekit (C5), brahe, nyx (offline only, AGPL); space-datasets for
re-entries and TLE histories; spacecost for vehicle figures.

### D16. Entry physics

**Serves:** [B13](#b13-re-entry-to-the-ground), [B30](#b30-aerobraking-and-aerocapture).

**Build:** Three-degree-of-freedom atmospheric entry over the rotating Earth.

**Interface:**
- `entry_derivative(t, y, params, density_fn) -> dy` with
  `y = (r, v)` in ECI and `params` holding mass, Cd·A, lift-to-drag ratio,
  nose radius and the bank angle (a value or a function of time and state).
  `density_fn(alt_km, lat, lon, jd)` is passed in, so D2 or satflight's table
  both work.
- `sutton_graves_heat_flux(rho, v, nose_radius) -> W/m²`: the Sutton-Graves
  stagnation-point heating `q = k sqrt(rho / r_n) v^3` (`k ≈ 1.74e-4` in SI
  units).
- `breakup(alt_km, heat_load, dynamic_pressure) -> bool`, with about 78 km as
  the usual altitude rule.

**Test on its own:** the Allen-Eggers ballistic-entry solution in an
exponential atmosphere (the peak deceleration and its altitude are closed
form); a lifting entry at L/D about 0.3 gives a lower peak deceleration than
a ballistic one from the same state.

**Sources:** CamPyRoS (ideas only, GPL) for aerodynamic heating.

### D17. Low-thrust steering laws

**Serves:** [B15](#b15-low-thrust-burns).

**Build:** Thrust-direction laws for continuous thrust.

**Interface:** each returns unit thrust directions `(N, 3)` in ECI:
- `steer_tangential(r, v)`.
- `steer_edelbaum(r, v, v0, v1, delta_i)` and
  `edelbaum_dv(v0, v1, delta_i) = sqrt(v0^2 + v1^2 - 2 v0 v1 cos(pi/2 ·
  delta_i))`.
- `steer_q_law(r, v, target_elements, weights, accel)`.

**Test on its own:** integrate a simple two-body + thrust model with each
law: Edelbaum's steering reaches the target with the analytic delta-v;
Q-law reproduces a published GTO-to-GEO case (Petropoulos 2004) to within
a few percent.

**Sources:** OpenSCvx or pygmo for optimal-control comparisons; prospector
for engine data.

### D18. Maneuver calculators

**Serves:** [B16](#b16-more-maneuvers-in-the-planner).

**Build:** Two-body maneuver sizing, returning delta-v vectors in VNB (or
RSW) and burn times.

**Interface:**
- `phasing(r, v, delta_u, revs) -> (dv1, dv2, t_wait)`.
- `raan_change(r, v, delta_raan) -> (dv, burn_u)` (Vallado Algorithms 39 and
  40).
- `combined_plane_change(r1, r2, delta_i) -> (dv1, dv2, split)` with the
  optimal split of the plane change.
- `apsidal_rotation(r, v, delta_argp) -> (dv, burn_nu)`.
- `deorbit(r, v, target_perigee_alt) -> dv`.
- `avoidance(r_rel, v_rel, target_miss_km, lead_time) -> dv` (along-track).

**Test on its own:** Vallado's chapter 6 worked examples; apply each delta-v
and Kepler-propagate to confirm the resulting orbit.

### D19. Orbit design solvers

**Serves:** [B17](#b17-orbit-design-keywords).

**Build:** The classic design formulas, with J2 secular rates.

**Interface:**
- `repeat_ground_track_a(revs, days, e, i, j2=..., re=..., mu=...) -> a` km,
  solved so the track repeats after `revs` nodal revolutions in `days`
  nodal days.
- `frozen_orbit(a, i, j2, j3) -> (e, argp)`, from
  `e ≈ -J3 Re sin i / (2 J2 a)` and argp = 90 deg (about 1e-3 at a 700 km
  SSO).
- `raan_for_ltan(ltan_hours, jd) -> raan` using the mean Sun (see A5).

**Test on its own:** Landsat 8 (233 revolutions in 16 days at about 705 km,
sun-synchronous); a frozen orbit integrated with J2 + J3 keeps e and argp
nearly constant.

### D20. Lifetime estimator

**Serves:** [B18](#b18-orbital-lifetime-readout).

**Build:** Orbital lifetime from a density model: King-Hele's analytic
theory, or a fast orbit-averaged decay integration.

**Interface:** `lifetime_days(a, e, cd_a_over_m, density_fn, jd0,
max_days=36500) -> float`, with `density_fn(alt_km, jd)` passed in so D2 or
the exponential table both work.

**Test on its own:** D15's decay cases; the analytic King-Hele result for an
exponential atmosphere.

### D21. Coverage, revisit, DOP and pass prediction

**Serves:** [B19](#b19-coverage-passes-and-link-readouts).

**Build:** Geometry analysis on a set of satellite positions.

**Interface:**
- Positions as a callable `positions_ecef(t) -> (N, 3)`, so the caller can
  supply a propagator, a history buffer or an ephemeris file.
- `coverage(positions_ecef, times, grid_lat, grid_lon, min_el) -> (percent,
  max_gap_s, mean_gap_s)` per grid point.
- `dop(site_ecef, sat_ecef, min_el) -> (gdop, pdop, hdop, vdop)`.
- `passes(site_lat, site_lon, site_alt, positions_ecef, t0, t1, min_el) ->
  list of (aos, t_max_el, max_el, los)`, with root-finding on the elevation
  crossing.

**Test on its own:**
- DOP for a symmetric toy geometry with a known answer; the 24-satellite GPS
  constellation gives PDOP of roughly 1.5-3 at mid-latitudes.
- Pass times against brahe's access windows or skyfield's
  `find_events`.

### D22. Link geometry

**Serves:** [B19](#b19-coverage-passes-and-link-readouts).

**Build:** Radio-link geometry for a ground site and a satellite.

**Interface:**
- `range_rate(site_ecef, r_ecef, v_ecef) -> (N,)` km/s, with the velocity
  relative to the rotating Earth.
- `doppler_shift(freq_hz, range_rate) -> Hz`.
- `free_space_path_loss_db(range_km, freq_hz)`.
- `link_margin_db(eirp_dbw, g_over_t, range_km, freq_hz, data_rate,
  required_eb_n0, losses_db=0)`.

**Test on its own:** a zenith pass has zero range-rate at closest approach;
a range-rate of 7 km/s at 437 MHz gives about 10 kHz of shift; path-loss
values from any link-budget reference.

### D23. Orbit determination

**Serves:** [B21](#b21-orbit-determination-in-the-simulation).

**Build:** Measurement models and estimators.

**Interface:**
- `measure_az_el_range(site_ecef, site_lat, site_lon, r_ecef, sigma, rng)`
  and `measure_range_rate(...)`: simulated measurements with noise.
- `batch_least_squares(obs, x0_guess, t0, propagate, h_fn) -> (x, cov)` and
  `ekf_step(...)`.
- `propagate(x (6,), t0, t1) -> x1` is passed in, so satflight's propagator
  can be used. The state-transition matrix comes from finite differences of
  `propagate` unless one is given.

**Test on its own:** a Kepler truth orbit with noisy synthetic measurements;
the estimate's error must be consistent with its covariance (chi-square test
over many runs).

**Sources:** brahe (batch least squares, EKF, UKF, az/el/range
measurements); autograd for Jacobians.

### D24. Atmospheric signal delays

**Serves:** [B21](#b21-orbit-determination-in-the-simulation).

**Build:** Tropospheric and ionospheric delays for radio ranging.

**Interface:**
- `tropo_delay_m(el, height_km, pressure_mbar, temp_c, humidity)`
  (Saastamoinen or Hopfield, with a mapping function).
- `klobuchar_delay_m(lat, lon, az, el, gps_seconds, alpha, beta)`, the model
  GPS broadcasts.

Scope: a full ionosphere model depends on sunlight (B11); Klobuchar with fixed
coefficients is the stand-in.

**Test on its own:** a zenith tropospheric delay of about 2.3-2.4 m at sea
level; the Klobuchar example in IS-GPS-200.

### D25. TLE tools

**Serves:** [B7](#b7-tle-drag-and-an-sgp4-propagator).

**Build:**
- The conversion between B* and `Cd·A/m`: B* = ρ0·B/2 with SGP4's reference
  density.
- Fitting a TLE to a set of states by least squares on SGP4.

**Interface:**
- `bstar_to_cd_a_over_m(bstar) -> m²/kg` and the inverse.
- `fit_tle(times_jd, states_teme (T, 6), satnum, epoch_jd) -> (line1, line2,
  rms_km)`. Needs the `sgp4` package.

**Test on its own:** take a real TLE, generate states with SGP4, fit, and
recover the same elements within the fit tolerance; check the B* conversion
against published ballistic coefficients.

**Sources:** python-sgp4 (C5); brahe's SGP4 tools.

### D26. Averaged long-term propagator

**Serves:** [B22](#b22-integrators-and-multi-rate-stepping).

**Build:** A mean-element propagator with J2 secular rates and orbit-averaged
drag (King-Hele or Liu theory), for runs of months to years in seconds.

**Interface:** `propagate_mean(el_mean (N, 6), dt, cd_a_over_m, density_fn,
j2=...) -> (N, 6)`, using D4's element layout.

**Test on its own:** against a Cowell run (written in the test with a simple
integrator) averaged over each revolution; against D15's decay cases.

### D27. Integrators

**Serves:** [B22](#b22-integrators-and-multi-rate-stepping).

**Build:** More integration methods, written to satflight's integrator
contract so they can be registered directly:
- **DOP853** and **RKF78** (adaptive, high order).
- **Gauss-Jackson** (fixed-step multistep, with an RK start-up).
- **Sundman time regularisation** (`dt/ds = r^k`) for high eccentricity.

**Interface:** a step function `step(f, t, y (N, 6), h) -> y`, or an adaptive
`integrate(f, t0, y0, t1, rtol, atol, h_max, callback, t_stop)` with
satflight's `Propagator.integrate` contract. That contract includes the rule
that steps are never shortened to land on `t_stop`, which keeps results
independent of the frame rate.

**Test on its own:**
- Error against an exact Kepler orbit versus step size shows the right
  order.
- Advancing in one call or in many chunks gives bit-identical results, as
  satflight's `tests/test_timestep.py` requires.
- Compare with brahe's RKF78 and RKN1210.

### D28. Attitude dynamics and panel drag

**Serves:** [B24](#b24-attitude-in-the-ensemble).

**Build:**
- Rigid-body attitude dynamics with the gravity-gradient torque, and simple
  pointing laws.
- A flat-plate macro-model for drag and lift.

**Interface:**
- `attitude_derivative(q, w, inertia, torque) -> (dq, dw)`; quaternions are
  scalar-first.
- `gravity_gradient_torque(r_body, inertia, mu=...)`, with the position in
  body axes.
- `panel_force(plates, q, v_rel, rho) -> (N, 3)`. Each plate has an area, a
  normal in body axes and coefficients.

**Test on its own:** small-angle pitch libration under gravity gradient at
`sqrt(3)·n`; one plate facing the flow gives the cannonball drag for the same
area and Cd.

**Sources:** CamPyRoS (ideas only, GPL) for 6-DOF structure.

### D29. Magnetic field and radiation belts

**Serves:** [B25](#b25-magnetic-field-and-radiation-overlays).

**Build:** The IGRF field model, the McIlwain L-shell, and a South Atlantic
Anomaly flag from field strength and L (or AE8/AP8 flux tables).

**Interface:**
- `igrf_field(r_ecef, jd) -> (N, 3)` nT, in ECEF.
- `l_shell(r_ecef, jd) -> (N,)`.
- `in_saa(r_ecef, jd, b_max_nt, l_range) -> bool array`.

**Test on its own:** NOAA's published IGRF values at a few points; the SAA
flag lights up over the South Atlantic at 400-1000 km.

**Sources:** brahe's IGRF and WMM-HR.

### D30. Power and thermal models

**Serves:** [B26](#b26-power-and-thermal-readouts).

**Build:** Spacecraft power and a one-node temperature, from an illumination
history.

**Interface:**
- `array_power(illum, sun_incidence, area_m2, efficiency, flux=1361.0) -> W`
  (arrays over time).
- `battery_soc(power_in_w, load_w, capacity_wh, dt_s, soc0=1.0) -> soc
  series`.
- `one_node_temperature(absorbed_w, area_m2, emissivity, heat_capacity_j_k,
  dt_s, t0_k) -> K series`, including the Earth's infrared and albedo heating
  as inputs.

**Test on its own:** constant illumination gives the steady-state
temperature `(Q / (eps sigma A))^(1/4)`; battery energy balance over an orbit
closes.

### D31. Breakup fragment generator

**Serves:** [B28](#b28-fragmentation-events-and-catalogue-scale-runs).

**Build:** The NASA Standard Breakup Model (Johnson et al. 2001):
- fragment counts for collisions, `N(>Lc) = 0.1 M^0.75 Lc^-1.71`;
- fragment counts for explosions, `N(>Lc) = 6 S Lc^-1.6`;
- the distributions of area-to-mass ratio and ejection speed;
- the catastrophic threshold of about 40 J/g of specific impact energy.

**Interface:** `breakup(kind, mass_kg, lc_min_m, rng, impactor_mass=None,
v_impact=None) -> (lc_m, area_over_mass, dv_kms (F, 3))`, with the velocity
increments in ECI given a parent velocity.

**Test on its own:** cumulative counts follow the power laws; the A/m and
speed distributions match the published means; how mass is (not exactly)
conserved, as in the model itself, is documented.

### D32. Imaging access windows

**Serves:** [B31](#b31-imaging-targets).

**Build:** When a satellite can image a ground target.

**Interface:** `imaging_windows(target_lat, target_lon, positions_ecef,
sun_ecef, t0, t1, max_off_nadir, min_sun_el=None) -> list of (start, end,
min_off_nadir)`. Positions and the Sun are callables of time, as in D21; with
`min_sun_el=None` (radar) the Sun is not used.

**Test on its own:** a satellite passing straight over the target has a
window centred on closest approach with zero off-nadir angle; the Sun
constraint removes night passes.

**Sources:** brahe's access module; skyfield.

### D33. Satellite brightness and visibility

**Serves:** [B32](#b32-observer-visibility).

**Build:** Visual magnitude of a satellite and whether an observer can see
it.

**Interface:**
- `visual_magnitude(r_sat, r_obs, r_sun, albedo_area_m2, model="sphere" |
  "plate") -> (N,)`.
- `observable(r_sat, r_obs, r_sun, illum, max_sun_el=-6 deg, min_el=10 deg)
  -> bool array` (sunlit satellite, dark sky, above the horizon).

**Test on its own:** the ISS comes out bright (around magnitude -3 overhead
in sunlight); published Starlink brightness ranges.

### D34. CCSDS message reader and writer

**Serves:** [B8](#b8-import-and-export-j2000-states-and-ccsds-messages).

**Build:** The CCSDS Orbit Data Messages and the Conjunction Data Message, in
KVN (and optionally XML).

**Interface:**
- `read_oem(path) -> list of segments (metadata, times_jd, states (T, 6))`
  and `write_oem(path, segments)`.
- `read_opm` / `write_opm`, `read_omm` / `write_omm`, `read_cdm` /
  `write_cdm`.
- CCSDS already uses km and km/s. Frames and time systems are passed through
  as metadata, not converted.

**Test on its own:** the examples in the CCSDS standards; read → write →
read round trips; files from brahe read correctly.

**Sources:** brahe reads and writes CCSDS in KVN, XML and JSON.

### D35. Ballistic trajectory solver

**Serves:** [B36](#b36-ballistic-trajectory-planner).

**Build:** The minimum-energy ballistic trajectory between two places on the
rotating Earth.

**Interface:** `ballistic(lat1, lon1, lat2, lon2, jd0, gmst_fn, mu=...) ->
(v_ecef, tof, apogee_alt)`, iterating Lambert (D5) on inertial points while
the target turns with the Earth during the flight.

**Test on its own:** without Earth rotation it must match the closed-form
minimum-energy range equation; with rotation, a two-body propagation from the
result must land on the target.

### D36. Density calibration from TLE decay

**Serves:** [B2](#b2-a-better-atmosphere-in-the-drag-force), [B12](#b12-verification-additions).

**Build:** An estimator that recovers how much denser (or thinner) the real
atmosphere was than a model, from an object's observed decay. Take the TLE
history, convert each mean motion to a mean semi-major axis, and fit a
density scale factor (or the product of scale and ballistic coefficient) so
that the model's decay rate matches the observed one, segment by segment.

**Interface:** `calibrate_density_scale(times_jd, a_mean_km, e, cd_a_over_m,
density_fn, window_days=5) -> (times_jd, scale, sigma)`, with
`density_fn(alt_km, jd)` passed in. Optionally a Bayesian version that
returns posterior samples (pymc).

**Test on its own:** decay generated with a known scale (by D26 or a simple
integrator) must give that scale back; real TLE histories should give scales
that rise and fall with the solar cycle.

**Sources:** space-datasets' TLE history; pymc for the Bayesian version.

### D37. Solar-activity forecast

**Serves:** [B2](#b2-a-better-atmosphere-in-the-drag-force), [B18](#b18-orbital-lifetime-readout).

**Build:** F10.7 and Ap for dates beyond the available data: a model of the
solar cycle (the Hathaway et al. 1994 cycle shape, fitted to the current
cycle), or a reader for NOAA SWPC's published cycle predictions, blended
with measured values where they exist.

**Interface:** `forecast(jd, source="model" | "swpc", level="mean" | "low"
| "high") -> (f107, f107a, ap)`, broadcast over `jd`. The `level` gives the
low/mean/high bands B18 shows.

Scope: solar activity is an input to the atmosphere, not a force; B11 lists
it as arguably in scope, pending a decision.

**Test on its own:** the model refitted to cycle 24 reproduces its peak date
and smoothed peak F10.7 to within the cycle's own scatter; values always
stay in the physical range (about 65-300 solar flux units).

### D38. Covariance propagation

**Serves:** [B10](#b10-conjunction-outcomes-and-statistics), [B21](#b21-orbit-determination-in-the-simulation).

**Build:** Propagation of a state's uncertainty: linear (through the
state-transition matrix), unscented (sigma points), and a Monte Carlo
reference.

**Interface:** `propagate_covariance(x (6,), P (6, 6), t0, t1, propagate,
method="stm" | "unscented") -> (x1, P1)`, with `propagate(x, t0, t1) -> x1`
passed in (the same callable as D23). The STM comes from finite differences
unless a function for it is given.

**Test on its own:** with a Kepler `propagate` and a small initial
covariance, the linear, unscented and Monte Carlo covariances agree; along-
track variance grows roughly with the square of time, as theory says.

**Sources:** brahe's covariance propagation examples; autograd for an exact
STM.

### D39. Catalogue-scale conjunction screening

**Serves:** [B28](#b28-fragmentation-events-and-catalogue-scale-runs), [B10](#b10-conjunction-outcomes-and-statistics).

**Build:** Screening filters that cut the number of pairs to check before any
propagation: the Hoots-Crawford-Roehrich apogee/perigee, orbit-path and time
filters (1984), and a "smart sieve" variant on propagated positions.

**Interface:** `screen(elements (N, 6), t0, t1, threshold_km) -> candidate
pairs (M, 2)`; `refine(positions_fn, pairs, t0, t1, threshold_km) -> list of
(i, j, tca, miss_km)`, with `positions_fn(t) -> (N, 3)` passed in.

**Test on its own:** on a few thousand random orbits, compare with a brute
force all-pairs search over fine time steps: no close approach may be missed,
and the share of pairs removed is reported.

### D40. Transfer grid search

**Serves:** [B5](#b5-rendezvous-targeting-against-the-real-force-model), [B16](#b16-more-maneuvers-in-the-planner).

**Build:** A "porkchop" search for impulsive transfers between two Earth
orbits: the total delta-v of a Lambert transfer for every departure time and
time of flight on a grid, with the minimum refined.

**Interface:** `transfer_grid(state1_fn, state2_fn, departures_s, tofs_s,
mu=...) -> (dv_total (D, T), dv1, dv2)`, where `state_fn(t) -> (r, v)` is
passed in (two-body, `j2mean`, or satflight's propagator), and
`best_transfer(...) -> (t_dep, tof, dv1_vec, dv2_vec)`. Uses D5.

**Test on its own:** between two coplanar circular orbits the minimum equals
the Hohmann delta-v, at half the transfer orbit's period.

**Sources:** pygmo for refinement with global optimisers.

### D41. Station-keeping budget calculators

**Serves:** [B14](#b14-station-keeping-and-orbit-maintenance).

**Build:** Annual delta-v estimates for keeping an orbit:
- LEO drag makeup: the delta-v per year to cancel the drag acceleration at a
  given altitude, ballistic coefficient and solar activity.
- GEO east-west: the longitude acceleration from the gravity field at a slot
  (C22 and, with D1, higher terms), and the delta-v per year to hold a
  deadband.

**Interface:**
- `drag_makeup_dv_per_year(alt_km, cd_a_over_m, density_fn, jd) -> m/s`.
- `geo_longitude_acceleration(lon, field=None) -> deg/day²` (C22/S22 alone
  when `field` is None).
- `geo_ew_dv_per_year(lon, deadband_deg, field=None) -> m/s`.

**Test on its own:** the longitude acceleration at 45 deg from a stable point
is close to PHYSICS.md's measured figure (3.06 deg in 60 days, i.e. about
0.0017 deg/day²); it is zero at the stable and unstable longitudes; the
yearly GEO cost peaks at about 2 m/s.

### D42. Constellation design calculators

**Serves:** [B19](#b19-coverage-passes-and-link-readouts), [B17](#b17-orbit-design-keywords).

**Build:** Sizing formulas for coverage constellations: Walker delta and
"streets of coverage" polar designs, giving the number of satellites and
planes for continuous single or multiple coverage at a given altitude and
elevation mask.

**Interface:** `walker_for_coverage(alt_km, min_el, fold=1, inclination=None)
-> list of (t, p, f, i)` and `streets_of_coverage(alt_km, min_el, fold=1) ->
(planes, per_plane)`.

**Test on its own:** each design returned must give full coverage when
checked numerically with D21; results should match published tables (e.g.
Walker's own results or Wertz's *Space Mission Engineering*).

**Sources:** pygmo for searching designs numerically.

### D43. Formation design

**Serves:** [B20](#b20-relative-motion-view), [B29](#b29-differential-drag-and-formation-control), [B33](#b33-deployment-docking-and-configuration-changes).

**Build:** Initial conditions for passively safe formations: the chaser
offset by separated relative eccentricity and inclination vectors (the "e/i
vector separation" of D'Amico and Montenbruck, 2006), so the radial and
cross-track separations never vanish together. A J2-aware linear model
(Schweighart-Sedwick) for how the formation drifts.

**Interface:** `safety_ellipse(r_t, v_t, along_km, radial_km, cross_km) ->
(r_c, v_c)`; `ss_propagate(rho0, rho_dot0, a, i, dt, j2=...) -> (rho,
rho_dot)`. Builds on D6.

**Test on its own:** propagate both satellites with a J2 integrator for many
orbits: the minimum distance perpendicular to the flight direction stays
above zero; the Schweighart-Sedwick prediction tracks the truth far better
than plain CW.

### D44. Component demise model

**Serves:** [B13](#b13-re-entry-to-the-ground).

**Build:** Which parts of a re-entering satellite burn up and which reach the
ground, in the spirit of NASA's DAS or ESA's DRAMA: components (tanks,
plates, reaction wheels) with material, shape and mass are released at
breakup, each heated along its own trajectory until it melts or lands.
Reports the surviving fragments and the casualty area.

**Interface:** `demise(components, breakup_state, entry_derivative,
density_fn) -> list of (component, demised, impact_lat_lon, impact_energy_j)`.
Material data (melting point, heat of fusion, specific heat) as a small
table. Uses D16 for the trajectory and heating.

**Test on its own:** aluminium plates should demise; titanium and stainless
steel tanks typically survive, as the published re-entry record shows.

### D45. Launch site and station network data

**Serves:** [B27](#b27-launch-guidance-and-mission-additions), [B19](#b19-coverage-passes-and-link-readouts).

**Build:** A dataset of real sites:
- launch sites with coordinates and the launch azimuth ranges range safety
  allows (Vandenberg, for example, launches south over the Pacific);
- tracking station networks (NASA's Near Earth Network, ESA's ESTRACK,
  KSAT) with coordinates and elevation masks.

**Interface:** JSON records `{"name", "lat", "lon", "alt_km", "azimuth_ranges"
| "min_el", "source"}`, readable into satflight's `LAUNCH_SITES` and
`GroundStation`.

**Test on its own:** each record cites its source; coordinates land on land
(or the right offshore platform) when plotted on a map.

### D46. Object properties dataset

**Serves:** [B7](#b7-tle-drag-and-an-sgp4-propagator), [B34](#b34-scenarios-from-live-data).

**Build:** Mass, size and an average cross-section for catalogued objects, so
TLE satellites get realistic ballistic coefficients instead of defaults.

**Interface:** `object_properties(satnum) -> {"mass_kg", "span_m",
"area_avg_m2", "shape", "source"}` from a bundled table, with
`cd_a_over_m(props, cd=2.2)`.

**Test on its own:** spot checks of well-known objects (the ISS at about
420 t); A/m values against D25's B*-derived ones, which should agree within
the scatter of B*.

**Sources:** GCAT (space-datasets) for masses and dimensions; ESA's DISCOS
holds cross-sections (it needs an account).

### D47. Launch performance estimator

**Serves:** [B6](#b6-launch-vehicle-physics).

**Build:** A quick payload-to-orbit estimate for a vehicle record: the
rocket equation over its stages against the delta-v a target orbit needs,
including typical gravity, drag and steering losses (together about
1.5-2 km/s to LEO).

**Interface:** `payload_capacity(vehicle_record, target="LEO" | "SSO" |
"GTO", site_lat=..., losses_kms=None) -> kg`, with vehicle records in D7's
format.

**Test on its own:** within about 20 % of published capacities for the
vehicles in spacecost's table. That also gives a starting guess for
satflight's ascent planner.

**Sources:** spacecost (payload figures); D7 (stage data).

### D48. Thruster performance models

**Serves:** [B3](#b3-propellant-budget), [B15](#b15-low-thrust-burns).

**Build:** Engine models for burns: chemical engines with a throttle range,
and electric thrusters whose thrust follows the power available,
`T = 2 eta P / (g0 Isp)`. A table of real thrusters (Hall-effect and
gridded-ion classes) with power, Isp, thrust and efficiency.

**Interface:** `electric_thrust(power_w, isp_s, efficiency) -> (thrust_n,
mdot_kg_s)`; `engine(name) -> record`. B15's burns and B26's power budget
would call these.

**Test on its own:** an SPT-100-class Hall thruster at 1.35 kW, Isp 1600 s
and efficiency 0.5 gives about 86 mN, close to its rated 83 mN.

**Sources:** prospector's engine configs; spacecost's propellant table.

### D49. Tracking sensor models

**Serves:** [B21](#b21-orbit-determination-in-the-simulation), [B32](#b32-observer-visibility).

**Build:** Whether a ground sensor can detect an object:
- radar: detection by signal-to-noise, which scales with radar cross-section
  over range to the fourth power;
- optical: the object must be sunlit against a dark sky and brighter than
  the telescope's limiting magnitude (uses D33).

**Interface:** `radar_detects(range_km, rcs_m2, ref_range_km, ref_rcs_m2) ->
bool array` (detection when the scaled SNR exceeds the reference);
`optical_detects(magnitude, limiting_magnitude, observable) -> bool array`.

**Test on its own:** halving the range lets the radar detect an object 16
times smaller in cross-section; optical detection never happens with the
object in shadow or the observer in daylight.

### D50. Closed-form geodetic conversion

**Serves:** [A3](#a3-two-different-altitudes-are-used-side-by-side), [B9](#b9-smaller-models-plugged-in).

**Build:** An exact, non-iterative conversion from Earth-fixed coordinates to
geodetic latitude, longitude and height (Vermeille 2002, or Zhu 1993),
vectorised and valid from the Earth's centre region out to beyond GEO,
including the poles. satflight's `frames.ecef_to_geodetic` uses six
fixed-point iterations instead, and the drag code uses an approximation
(`forces.approx_altitude`); one exact, fast routine would give every
"altitude" the same meaning (A3).

**Interface:** `ecef_to_geodetic(r_ecef, a=..., f=...) -> (lat, lon, h_km)`,
a drop-in for satflight's function, plus `geodetic_height(r_ecef)` for code
that needs only the height.

**Test on its own:** round trip with `geodetic_to_ecef` to under 1 mm for
random points from 0 to 50,000 km altitude, at the poles and on the equator;
faster than six iterations on 10^6 points.

### D51. Robust two-body propagator

**Serves:** [A16](#a16-solver-edge-cases), [B22](#b22-integrators-and-multi-rate-stepping).

**Build:** A universal-variable Kepler propagator that reports convergence
per satellite instead of returning the last iterate (A16), handles
near-parabolic and very long propagations (reducing the time by whole
periods), and solves Kepler's equation with a guaranteed-convergence
starter (e.g. Markley's or a bracketed Newton).

**Interface:** `kepler_propagate(r0, v0, dt, mu=...) -> (r, v, ok)`, the same
as satflight's plus an `(N,)` boolean `ok`; `mean_to_true(M, e)` covering
e = 1 exactly through Barker's equation.

**Test on its own:** forward then backward propagation returns the start to
1e-9 relative for e from 0 to 10, including e = 0.999999 and 1.000001; the
energy and angular momentum are conserved; comparison with pykep's
propagator.

### D52. US Standard Atmosphere 1976

**Serves:** [A15](#a15-launch-bookkeeping-details), [B6](#b6-launch-vehicle-physics).

**Build:** The full US Standard Atmosphere 1976: temperature, pressure,
density and speed of sound from sea level to 1000 km, with the geopotential
altitude conversion and the layer lapse rates below 86 km and the tabulated
upper atmosphere above.

**Interface:** `us76(alt_km) -> (T_K, p_Pa, rho_kg_m3, a_km_s)`, plain-float
and array versions (the ascent's hot path uses floats). It replaces the
launch model's `exp(-h / 7 km)` pressure (A15) and supplies D7's speed of
sound.

**Test on its own:** the published tables: 288.15 K, 101,325 Pa and
1.225 kg/m³ at sea level; 216.65 K and 22,632 Pa at 11 km geopotential
altitude.

### D53. Capsule aerodynamics

**Serves:** [B13](#b13-re-entry-to-the-ground), [B30](#b30-aerobraking-and-aerocapture).

**Build:** Hypersonic force coefficients for blunt entry shapes (sphere-cone
capsules, spheres, flat plates) by modified Newtonian theory,
`Cp = Cp_max sin^2(theta)`, with `Cp_max` from the normal-shock relation, and
the trim angle of attack and lift-to-drag ratio that follow.

**Interface:** `newtonian_coefficients(shape, alpha) -> (cd, cl, cm)` with
`shape` described by a few parameters (nose radius, cone half-angle, base
radius); `trim(shape, cg_offset) -> (alpha, cd, cl)`. D16's entry model takes
the resulting Cd and L/D.

**Test on its own:** `Cp_max` for γ = 1.4 at high Mach is 1.839; a sphere
gives Cd about 0.92; an Apollo-like capsule at its trim angle gives L/D of
about 0.3.

### D54. Parachute descent and landing dispersion

**Serves:** [B13](#b13-re-entry-to-the-ground).

**Build:** The last phase of a crewed or sample-return landing: drogue and
main parachute deployment by altitude or speed, descent at terminal speed
`sqrt(2 m g / (rho Cd A))`, and drift with a wind profile, giving a landing
point and a dispersion ellipse.

**Interface:** `descend(state, chutes, density_fn, wind_fn, g=...) ->
(landing_lat, landing_lon, time_s, touchdown_speed)`, with `chutes` a list of
`{"cd_area_m2", "deploy_alt_km" | "deploy_speed_km_s"}`; batch over
dispersed inputs for an ellipse.

**Test on its own:** touchdown speed equals the terminal-speed formula at
sea-level density; with no wind the landing point lies straight below the
deployment point.

### D55. Debris environment flux

**Serves:** [B28](#b28-fragmentation-events-and-catalogue-scale-runs), [B10](#b10-conjunction-outcomes-and-statistics).

**Build:** The flux of small, untracked debris through a satellite's orbit as
a function of altitude, inclination and minimum size, from published
environment model outputs (ORDEM or MASTER) or a fitted analytic form, and
the expected number of impacts over a mission.

**Interface:** `debris_flux(alt_km, inc, min_size_m, year) -> impacts per
m² per year`; `expected_impacts(orbit_mean_elements, area_m2, years,
min_size_m)`.

**Test on its own:** the spatial density peaks near 800-900 km, as the
published environment curves show; flux grows steeply as the minimum size
shrinks.

### D56. Collision avoidance optimiser

**Serves:** [B10](#b10-conjunction-outcomes-and-statistics), [B16](#b16-more-maneuvers-in-the-planner).

**Build:** The smallest maneuver that brings a predicted probability of
collision below a threshold (1e-4 is a common operator choice), given the
encounter geometry, the combined covariance and the time before closest
approach. Along-track burns are the usual answer; the optimiser can also
consider radial ones.

**Interface:** `avoidance_maneuver(r_rel, v_rel, cov_rel, hard_body_radius,
lead_time_s, pc_max=1e-4, mu=..., r_sat=..., v_sat=...) -> (dv_vnb, pc_after)`,
using D14 for Pc and two-body sensitivity of the miss vector to the burn.

**Test on its own:** with no burn the Pc equals D14's; the returned burn
achieves `pc_after <= pc_max` and is smaller than any burn found by a grid
search over directions and sizes.

### D57. Onboard GNSS navigation

**Serves:** [B21](#b21-orbit-determination-in-the-simulation), [B19](#b19-coverage-passes-and-link-readouts).

**Build:** A LEO satellite navigating with GPS: pseudoranges from the GPS
satellites in view (with a receiver clock bias and noise), and the standard
least-squares position-velocity-time solution, optionally smoothed with a
filter (D23).

**Interface:** `pseudoranges(r_user, gps_positions, clock_bias_m, sigma_m,
rng, mask=...) -> (ranges, visible)`; `pvt(ranges, gps_positions, x0) ->
(r_user, clock_bias_m, dop)`.

**Test on its own:** with the 24-satellite GPS constellation and no noise
the solution recovers the true position and clock exactly; with noise the
error scales with PDOP (D21) times the range noise.

### D58. Physics invariant property tests

**Serves:** [B12](#b12-verification-additions).

**Build:** A reusable property-based test suite for orbital code, generating
random but valid inputs (with the `hypothesis` library) and checking the
invariants any correct implementation must keep:
- element ↔ state round trips (`coe2rv(rv2coe(x)) == x`) for every conic;
- two-body propagation conserving energy and angular momentum, and forward
  then backward returning to the start;
- frame round trips (ECI ↔ ECEF, geodetic ↔ ECEF);
- force terms equal to the gradient of their potential;
- results that do not depend on how a run is chunked.

**Interface:** test functions that take the implementation as an argument
(e.g. `check_round_trip(rv2coe, coe2rv)`), so the same suite runs against
satflight's functions and every Part D replacement.

**Test on its own:** each property must fail on a planted defect (e.g. a sign
flipped in a J2 term) before it is trusted to pass.

### D59. Materials dataset

**Serves:** [B13](#b13-re-entry-to-the-ground), [B26](#b26-power-and-thermal-readouts).

**Build:** Thermal and optical properties of common spacecraft materials:
melting point, heat of fusion, specific heat, density, solar absorptivity and
infrared emissivity, each with its source.

**Interface:** `material(name) -> {"melt_k", "heat_of_fusion_j_kg",
"specific_heat_j_kg_k", "density_kg_m3", "absorptivity", "emissivity",
"source"}`. D44 (demise) and D30 (thermal) read it.

**Test on its own:** spot values from handbooks: aluminium melts at about
933 K, titanium at about 1940 K, stainless steel at about 1700 K.

### D60. Attitude control laws

**Serves:** [B24](#b24-attitude-in-the-ensemble), [B25](#b25-magnetic-field-and-radiation-overlays).

**Build:** The basic control laws for a small satellite: B-dot detumbling
with magnetorquers (`m = -k dB/dt`), magnetorquer momentum dumping, and
quaternion-feedback pointing with reaction wheels.

**Interface:** `bdot(b_body, b_body_prev, dt, k, m_max) -> dipole (3,)`;
`pointing_torque(q, w, q_target, kp, kd, torque_max) -> (3,)`; works with
D28's dynamics and D29's field.

**Test on its own:** a tumbling satellite under B-dot in a dipole field
slows to near orbit rate; quaternion feedback settles to the target with the
damping its gains imply.

### D61. Electrodynamic tether force

**Serves:** [B15](#b15-low-thrust-burns).

**Build:** The Lorentz force on a current-carrying tether in the Earth's
magnetic field, `F = I L × B`: a propellantless way to lower (or, with power,
raise) an orbit. It is an Earth-only force.

**Interface:** `tether_force(r_ecef, v_rel_ecef, tether_dir, length_km,
current_a, field_fn) -> (N, 3)` newtons; a simple current model from the
induced EMF `(v × B) · L` and a circuit resistance.

**Test on its own:** with 1 A in a 5 km tether in a 3e-5 T field the force is
about 0.15 N (estimated); the force reverses with the current; drag mode
(current driven by the EMF) always removes orbital energy.

**Sources:** D29 for the field.

### D62. Analytic ground-track predictor

**Serves:** [B17](#b17-orbit-design-keywords), [B19](#b19-coverage-passes-and-link-readouts).

**Build:** Ground tracks from mean elements alone, with no propagation: the
sub-satellite latitude from `sin(lat) = sin(i) sin(u)`, the longitude from
the node's longitude, the argument of latitude and the Earth's rotation, with
J2's secular rates. It also gives the maximum latitude, the longitude
shift per revolution and node crossing longitudes.

**Interface:** `ground_track(mean_elements, t, gmst0) -> (lat, lon)` over
time arrays; `node_longitudes(mean_elements, n_revs, gmst0)`.

**Test on its own:** matches a J2 Cowell ground track to within the
short-period terms (a few km) over a day; a Landsat 8-like repeat orbit
returns to the same node longitude after 233 revolutions.

### D63. Optimal ascent trajectories

**Serves:** [B27](#b27-launch-guidance-and-mission-additions), [B6](#b6-launch-vehicle-physics).

**Build:** Offline optimisation of launch ascents: the pitch profile, coast
lengths and staging that maximise payload (or propellant left) for a vehicle
and target orbit, with constraints on maximum dynamic pressure and
acceleration. The result is a reference against which satflight's
closed-loop guidance can be judged.

**Interface:** `optimal_ascent(vehicle_record, target_orbit, site, q_max_pa,
a_max_g) -> (pitch_profile, payload_kg, trajectory)`, with vehicle records
in D7's format and the atmosphere from D52.

**Test on its own:** with drag and limits switched off, the optimum
approaches the linear-tangent steering law's; payload figures within about
10 % of published capacities.

**Sources:** OpenSCvx (successive convexification), pygmo (global
optimisation).

---

# Appendices

## Checked and found correct

These were re-derived or cross-checked during the audit and need no work:

- **Gravity terms:**
  - J2, J3 and J4 accelerations match Vallado's closed forms; tests check them
    against the potential gradient.
  - EGM-96 coefficients are unnormalised correctly: J2 = 4.8417e-4·√5,
    C22 = 2.4394e-6·√(5/12).
  - The C22/S22 potential, its gradient and the ECEF-to-ECI rotation are
    right. The Jacobi integral `E - ω h_z` is the right conserved quantity.
- **Drag:** the unit factor 0.5e3 (km/s inputs, km/s² output) is correct,
  with a co-rotating atmosphere. The density table matches Vallado Table 8-4
  entry by entry.
- **J2 mean elements:**
  - The secular rates (Vallado eq. 9-41) are right.
  - The Kozai short-period term in a, and its exact quadratic inverse, are
    right.
  - The nodal period `2π / (n + Ṁ + ω̇)` is right.
  - `j2mean` keeps a geostationary orbit's rate equal to ω_E to first order.
- **Time and frames:**
  - GMST is IAU-82; the Julian date is Vallado Alg. 14; the TLE epoch, year
    pivot and checksum are right.
  - Geodetic conversion: iterative latitude with the pole-safe height formula.
  - ENU and look angles, and the RSW and VNB bases (B = V × N points radially
    outward), are right.
- **Eclipse:** the Montenbruck & Gill conical shadow (umbra, penumbra,
  annular overlap area) is implemented correctly.
- **Elements and solvers:**
  - `rv2coe` / `coe2rv` handle the circular and equatorial conventions.
  - The universal-variable Kepler solver uses Laguerre-Conway with correct
    Stumpff series.
  - The DOPRI5 tableau and its embedded error coefficients are right.
- **Maneuvers:**
  - The Hohmann and bi-elliptic formulas are right.
  - The plane-change rotation is right: rotating v about r rotates h, with
    the correct sign at both nodes.
  - Rocket-equation bookkeeping for impulsive and finite burns is right.
  - The sun-synchronous target rate uses the tropical year.
- **Other derived numbers:**
  - Walker delta and star phasing, and the coverage half-angle
    (90 - ε - η) and footprint small-circle are right.
  - The geostationary radius with J2 (force balance) is right.
- **Launch:**
  - Isp(p) varies linearly in ambient pressure, which is the right form
    (T = ṁ·vₑ + (pₑ − pₐ)·Aₑ).
  - The plumb-line vertical is right.
  - The inertial-to-Earth-relative launch azimuth correction is right.
  - The two-point boundary steering law reaches r_T with v_r = 0.
  - The energy cutoff is right.

## How the numbers were measured

Two throwaway scripts imported `satflight` read-only, from outside the repo:

| Finding | What the script did |
|---|---|
| A2 | Loaded `rendezvous.json`, dropped its stored maneuvers, set `ForceModel(j2=...)`, called `planner.plan_rendezvous(sim, "Chaser", "Station", tof, timing="delay", delay=300)`, scheduled the burns, advanced to arrival and measured the separation |
| A1 | Built a chaser at i = 90 deg / 500 km and a target at 520 km with RAAN ±0.3 deg, and called `lambert(...)` with the planner's `prograde` flag and with its opposite |
| A6 | 300 × 1500 km at nu = 90 deg, `plan_hohmann(..., 2000, timing="now")`, two-body |
| A7 | 690 × 710 km, a `circularize` maneuver at perigee, J2 Cowell, radius sampled every 60 s for 200 min |
| A5 | `local_time_of_node(raan, sun_position(jd))` with the node held at a constant offset from the mean Sun, every 5 days of 2026 |
| A3, A9 | `geodetic_to_ecef(89.9 deg, 0, 300 km)` vs `|r| - R_eq`; `|v|^2 / |v - omega x r|^2` for an ISS-like state |
