# Physics audit: problems, additions and repos to integrate

**Date:** 2026-10-02 · **Code audited:** `main` at `f1e701b` · **No code was changed.**

This is an audit of the physical calculations only. It covers forces, frames
and time, orbital elements, the integrators and analytic propagators, maneuvers
and planners, events, the launch model, and the derived readouts in the Orbit
tab. UI code was checked only where it computes something physical. Nothing
here has been fixed.

The findings are in three parts:

- **[Part A: Problems](#part-a-problems-in-the-existing-code-and-physics)**
  covers what the code already does but gets wrong, gets inconsistent, or
  describes wrongly. Each item gives where it is, what is wrong, the evidence,
  the impact and a fix.
- **[Part B: Additions](#part-b-additions-that-could-or-should-be-made)**
  covers physics or features the simulator does not have yet. Each item gives
  why it matters and how it could be built.
- **[Part C: Repositories](#part-c-github-repositories-that-could-be-integrated)**
  lists GitHub repositories, mostly from your starred list, that could help
  with Parts A and B, and how each one could be used given its licence.

Where fixing a problem needs a new feature, the Part A item points to the
Part B item.

Severity in Part A:

- **High:** gives wrong results that a normal user will run into.
- **Medium:** visible in readouts or plans.
- **Low:** small in magnitude, or only a claim in the docs.

Priority in Part B:

- **High:** the largest remaining physical error, or needed by a Part A fix.
- **Medium:** clear value for accuracy or realism.
- **Low:** refinement.

"Measured" means the number came from running the code; the scripts are
described in [How the numbers were measured](#how-the-numbers-were-measured).
"Estimated" means the number is a hand calculation.

Scope reminder ([SCOPE.md](SCOPE.md)): only the Earth may act on an orbit.
Additions that would involve the Sun or the Moon are listed separately in
[B13](#b13-additions-that-need-a-scope-decision).

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
- For metre-level arrival, see [B5](#b5-better-rendezvous-and-transfer-targeting)
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
addition: [B8](#b8-j2000-import-and-export).

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
  [B4](#b4-full-first-order-mean-elements-and-equinoctial-elements).
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
[B4](#b4-full-first-order-mean-elements-and-equinoctial-elements).

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
as density.

### A16. Solver edge cases

- `kepler_propagate` and `lambert` return the last iterate silently when they
  do not converge. Lambert raises only when it runs out of iterations.
- `mean_to_true` with e exactly 1 divides by zero in the hyperbolic branch.
- Lambert's psi bounds (±4π²) limit very fast hyperbolic arcs.
- Transfer angles near 180 deg are ill-conditioned: the plane is undefined,
  and a large out-of-plane delta-v appears (seen in the A1 runs).

**Fix:** Return a convergence flag and warn; handle e = 1 with Barker's
equation; warn near 180 deg. A more robust Lambert solver is an addition:
[B5](#b5-better-rendezvous-and-transfer-targeting).

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

# Part B: Additions that could or should be made

## Summary of additions

| ID | Area | Priority | One line |
|---|---|---|---|
| [B1](#b1-full-spherical-harmonic-gravity) | Gravity | **High** | Full n×m spherical-harmonic gravity (EGM-96/2008) |
| [B2](#b2-a-better-atmosphere-and-drag-model) | Drag | **High** | Solar-activity-driven density, a decay validation, variable Cd |
| [B3](#b3-propellant-budget) | Spacecraft | **High** | Dry mass and propellant; fixes A8 |
| [B4](#b4-full-first-order-mean-elements-and-equinoctial-elements) | Elements | Medium | Full Brouwer/Kozai short-period terms and equinoctial elements; fixes A7 and A11 properly |
| [B5](#b5-better-rendezvous-and-transfer-targeting) | Planner | Medium | Differential correction, terminal approach guidance, multi-revolution Lambert |
| [B6](#b6-launch-model-additions) | Launch | Medium | Mach-dependent drag, separation impulses, tracked fairings, winds, strap-ons |
| [B7](#b7-tle-ballistic-coefficient) | TLE | Medium | Map B* to `Cd·A/m`; list `sgp4` as a dependency |
| [B8](#b8-j2000-import-and-export) | Frames | Medium | A precession-nutation transform for J2000 export and import |
| [B9](#b9-eclipse-geometry) | Eclipse | Low | Oblate Earth and atmospheric shadow enlargement |
| [B10](#b10-ground-stations-and-coverage) | Stations | Low | Refraction, terrain masks, ellipsoidal footprints |
| [B11](#b11-small-earth-only-forces) | Forces | Low | General relativity and Earth infrared radiation pressure |
| [B12](#b12-conjunction-outcomes-and-statistics) | Events | Low | Collision outcomes, covariance and probability of collision |
| [B13](#b13-additions-that-need-a-scope-decision) | Scope | n/a | Diurnal bulge, albedo, tides, Sun and Moon gravity, SRP |
| [B14](#b14-verification-additions) | Tests | **High** | Regression tests for Part A and an external reference ephemeris |

### B1. Full spherical-harmonic gravity

**Why:** Only J2, J3, J4 and C22/S22 are modelled. Error estimates:
- LEO: the missing J5-J6 and tesserals C21/S21, C31, C33, ... cost roughly
  hundreds of metres per day of along-track error.
- GEO: C22 alone puts the stable longitudes at 75.07 E and 104.93 W, against
  about 75.1 E and 105.3 W observed. C31, S31, C33 and S33 make up the rest.
- Resonant orbits (GPS 2:1, repeat-ground-track LEO) are sensitive to specific
  tesserals (for GPS, J22 and J32), which are absent.

This is all Earth gravity, so it is **in scope**.

**How:**
- A general spherical-harmonic gravity routine (Cunningham/Montenbruck-Gill
  recursion, or Pines' singularity-free formulation) reading an EGM-96 or
  EGM-2008 coefficient file truncated to n×m, e.g. 8×8 for display, 20×20 or
  more for accuracy runs.
- It is vectorisable over the ensemble in numpy.
- Keep J2..J4 + C22 as a fast preset.
- Update `test_sun_and_moon_forces_are_out_of_scope`, which pins
  `ForceModel.TERMS`.

### B2. A better atmosphere and drag model

**Why:** The density is the Vallado Table 8-4 piecewise exponential, a
function of altitude only, times a constant `density_scale`. It is the
largest *physical* error source in LEO:
- Solar-cycle variation is about ×10 at 400 km, and the scale stays fixed
  across a multi-year decay run.
- Geomagnetic storms add tens of % to ×2 for days.
- Seasonal and latitude variations are missing. (The day/night bulge is
  missing too, but needs a scope decision: see B13.)
- Drag uses a constant `Cd = 2.2`. Real free-molecular Cd varies about
  2.0-2.6+ with altitude and surface.
- Smaller points: the table is discontinuous at band edges (by under 1 %, e.g.
  at 25 km), which costs a few rejected DOPRI steps; the 1000-2500 km
  extrapolation then drops to zero.

**How:**
- Time-varying solar activity: an F10.7 / Ap schedule, or a simple 11-year
  sinusoid, feeding a Jacchia-71, NRLMSISE-00 or JB2008 model. Pure-Python
  NRLMSISE ports exist (check licences).
- Validate decay against a published case, e.g. a known re-entry or a Vallado
  example, so that "indicative" has a measured size. An analytic King-Hele
  decay rate for the exponential atmosphere would make a cheap test.
- Optionally, an altitude-dependent Cd.

### B3. Propellant budget

**Why:** Fixes [A8](#a8-burns-can-spend-unlimited-delta-v): today a satellite
can burn without limit.

**How:**
- Add `dry_mass` (or a `propellant` field) to `Satellite`, `SatSpec` and the
  dialogs.
- Refuse or truncate burns that would go below dry mass, and log it.
- Show the remaining delta-v in the Orbit tab.
- Optionally give each satellite its own engine (Isp and thrust), instead of a
  per-maneuver Isp defaulting to 300 s.

### B4. Full first-order mean elements and equinoctial elements

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

**How:**
- First-order Brouwer-Lyddane (or Kozai) short-period terms for all six
  elements, both directions.
- J2², J4 and J3 terms in `j2mean`.
- Equinoctial elements for display and for `j2mean`, which removes the
  singularities.

### B5. Better rendezvous and transfer targeting

**Why:** After the quick fix in A2, two-body-plus-J2-mean predictions still
leave some miss. Lambert is zero-revolution only, and the current solver is
fragile near 180 deg (A16).

**How:**
- Differential correction: after the Lambert guess, propagate the arc with
  the run's real force model (a scratch `Propagator`). Correct `dv1` by
  Newton/secant on the arrival miss using a finite-difference state-transition
  matrix. Three or four iterations should reach metres.
- A terminal "null position" burn pair, or a CW/Hill-equation final approach.
- Izzo's or Gooding's Lambert solver, which handles multi-revolution cases and
  is robust near 180 deg.

### B6. Launch model additions

**Why:** These are not in the model yet:
- `Cd` does not depend on Mach number. Real launchers peak at about ×2 near
  Mach 1-1.2, which is what sets max-Q and drag loss.
- Spent stages separate with zero relative velocity (no separation impulse),
  and keep the full-stack reference area.
- The fairing is dropped as mass but not tracked as an object.
- Already documented as missing: winds, strap-on boosters burning alongside a
  core, engine-out, parking-orbit coasts with restarts, and attitude dynamics.

**How:** A Cd(Mach) table with a speed of sound from the atmosphere; a small
separation delta-v per stage; the fairing as a spawned object with its own
area; then the documented items, in whatever order matters for play.

### B7. TLE ballistic coefficient

**Why:** B* is parsed but not used. A TLE satellite gets the default
mass/area/Cd (`Cd·A/m` = 0.022 m²/kg) whatever its real ballistic
coefficient. `sgp4` is optional and commented out of `requirements.txt`, so
most installs use the fallback, which also ignores drag.

**How:** Derive `Cd·A/m` from B* (B* = ρ0·B/2 with SGP4's reference density).
List `sgp4` as a real dependency, or offer to install it.

### B8. J2000 import and export

**Why:** The integration frame is of-date (A4). Tools such as GMAT, Orekit
and astropy expect J2000/GCRF.

**How:** An IAU-76/80 precession + nutation transform (or IAU-2006/2000A),
`eci_j2000 <-> eci_of_date`, applied when exporting the CSV and when reading
J2000 states into a scenario.

### B9. Eclipse geometry

**Why:** The Earth is a sphere of *equatorial* radius with no atmosphere. In
reality, refraction and absorption make the effective shadow about 2 % larger
(the Danjon/Chauvenet enlargement), and the polar radius is 21 km smaller.
Together these shift entry and exit by a few seconds. Display only.

**How:** Use an effective shadow radius, and optionally an oblate shadow
cone.

### B10. Ground stations and coverage

**Why:** There is no atmospheric refraction (about 0.5 deg at the horizon,
which matters for a 0-5 deg mask; Svalbard uses 5 deg) and no terrain mask.
`coverage_half_angle` and `footprint` assume a spherical Earth of equatorial
radius.

**How:** A standard refraction correction to elevation; a per-station
azimuth/elevation mask; ellipsoidal footprints.

### B11. Small Earth-only forces

**Why:**
- General relativity (Schwarzschild): about 1.7e-8 m/s² in LEO; a few metres
  per day along track (estimated).
- Earth infrared radiation pressure: about 8e-7 m/s² per (m²/kg) of A/m, i.e.
  about 1e-8 m/s² at a typical 0.01 m²/kg (estimated from about 230 W/m² of
  outgoing IR).

Both are in scope and cheap.

**How:** The standard post-Newtonian term
`mu/(c^2 r^3) [(4 mu/r - v^2) r + 4 (r·v) v]`; a cannonball IR pressure with a
uniform Earth emission.

### B12. Conjunction outcomes and statistics

**Why:** Close approaches have no consequences: pairs pass through each
other. There is no collision or debris outcome, no covariance and no
probability of collision. The threshold is a fixed 10 km.

**How:** A configurable threshold; a collision event (with an optional simple
debris cloud); per-satellite position covariance and a Pc estimate (Foster
or Alfano).

### B13. Additions that need a scope decision

These are real gaps, but each needs a decision under [SCOPE.md](SCOPE.md)
because it involves the Sun or the Moon in an orbital calculation:

| Item | Why it matters | Scope conflict |
|---|---|---|
| Atmospheric diurnal bulge (Harris-Priester, Jacchia, MSIS) | ×2-3 density day/night at 400-500 km | Bulge direction follows the Sun |
| Solar-flux-driven density (F10.7, Ap) | ×10 over the solar cycle | Solar *activity* input, not position; arguably in scope as "atmosphere state" |
| Earth albedo radiation pressure | about 1e-8 m/s² at A/m = 0.01 m²/kg, dayside | Reflected sunlight |
| Solid Earth and ocean tides (changes in C20 and others) | metres to tens of metres per day in LEO | Raised by the Sun and Moon |
| Sun/Moon third body, SRP | Dominant at GEO and in HEO (inclination growth about 0.85 deg/yr at GEO) | Explicitly removed |

### B14. Verification additions

**Regression tests for Part A:**
- Rendezvous with J2 on (A2). `rendezvous.json` runs two-body, which hid the
  problem.
- Polar Lambert, target on each side of the plane (A1).
- Altitude readouts, CSV and batch summary agree (A3).
- An SSO keeps a constant LTAN over a year (A5).

**New kinds of check:**
- **External reference ephemeris:** nothing compares a run against an
  independent propagator (GMAT, Orekit, poliastro or astropy) for the same
  force model. A small fixture would catch frame (A4) and force-model
  regressions: for example, an ISS state propagated 1 day under J2-J4 plus a
  fixed exponential atmosphere, stored as a CSV, compared at the 10 m level.
- **Drag decay magnitude:** `test_drag_decays_low_orbit_and_detects_reentry`
  checks that decay *happens*, not how fast (see B2).
- **Launch against flight data:** e.g. Falcon 9 MECO at roughly T+150 s,
  65-80 km and about 2.3 km/s on LEO missions. This is to bound "realistic in
  kind".

---

# Part C: GitHub repositories that could be integrated

I went through all 98 repositories starred by `loggger101` (as of 2026-10-02).
20 of them could help with the items in Parts A and B. Four well-known
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
| [duncaneddy/brahe](https://github.com/duncaneddy/brahe) | MIT | Test oracle, port | A4, A7, A10, A11, B1, B2, B4, B5, B7, B8, B10, B11, B12, B14 |
| [skyfielders/python-skyfield](https://github.com/skyfielders/python-skyfield) | MIT | Dependency | A4, A21, B7, B8, B10 |
| [astropy/astropy](https://github.com/astropy/astropy) | BSD-3 | Test oracle | A4, A21, B8, B10, B14 |
| [astropy/astroquery](https://github.com/astropy/astroquery) | BSD-3 | Test oracle (data) | B14 |
| [esa/pykep](https://github.com/esa/pykep) | MPL-2.0 | Dependency or test oracle | A1, A16, B5 |
| [nyx-space/hifitime](https://github.com/nyx-space/hifitime) | MPL-2.0 | Dependency | A21, B8 |
| [HIPS/autograd](https://github.com/HIPS/autograd) | MIT | Dependency | B5, B12 |
| [sympy/sympy](https://github.com/sympy/sympy) | BSD-3 | Dev tool | A13, B1, B4 |
| [esa/pygmo2](https://github.com/esa/pygmo2) | MPL-2.0 | Dependency (optional) | B5, B6 |
| [OpenSCvx/OpenSCvx](https://github.com/OpenSCvx/OpenSCvx) | Apache-2.0 | Offline experiments | B5, B6 |
| [loggger101/spacecost](https://github.com/loggger101/spacecost) | MIT | Data, dependency | B3, B6, B14 |
| [Karmanplus/prospector](https://github.com/Karmanplus/prospector) | Apache-2.0 | Data | B3 |
| [juliensimon/space-datasets](https://github.com/juliensimon/space-datasets) | per dataset | Data | B2, B6, B7, B12, B14 |
| [pymc-devs/pymc](https://github.com/pymc-devs/pymc) | Apache-2.0 | Offline experiments | B2, B7 |
| [hannorein/rebound](https://github.com/hannorein/rebound) | GPL-3.0 | Reference only | A18, B14 |
| [dtamayo/reboundx](https://github.com/dtamayo/reboundx) | GPL-3.0 | Reference only | B11, B14 |
| [nyx-space/nyx](https://github.com/nyx-space/nyx) | AGPL-3.0 | Reference only | B1, B2, B12, B14 |
| [cuspaceflight/CamPyRoS](https://github.com/cuspaceflight/CamPyRoS) | GPL-3.0 | Reference only | B6 |
| [shadden/celmech](https://github.com/shadden/celmech) | GPL-3.0 | Reference only | B4 |
| [matthewholman/assist](https://github.com/matthewholman/assist) | GPL-3.0 | Reference only | B13 (only if the Sun and Moon return) |

If you adopt only three, I'd pick:

1. **brahe** as the test oracle for B14, and as the source to port gravity,
   mean elements and the atmosphere from.
2. **skyfield** for time scales and J2000/TEME frames (A4, A21, B8). It is
   pure Python and numpy, like satflight.
3. **pymsis** (not starred, see C5) for the atmosphere (B2). It takes and
   returns numpy arrays, so it fits the ensemble hot path.

## C3. Starred repositories that fit well

### duncaneddy/brahe

**Licence:** MIT. **Install:** `pip install brahe` (Rust core with Python
bindings). **Units:** SI (metres), where satflight uses km.

This is the closest match in your starred list: a modern satellite-dynamics
library covering most of Part B. Its source tree contains:

- **Gravity (B1):** spherical-harmonic gravity with bundled EGM2008 (degree
  120), GGM05S and JGM3 coefficient files, plus an ICGEM file loader.
- **Atmosphere (B2):** exponential, Harris-Priester and NRLMSISE-00 drag
  models, and bundled space-weather files (F10.7 flux table, CelesTrak `sw`
  file).
- **Mean elements (B4, A7, A11):** analytical Brouwer-Lyddane mean-to-
  osculating conversion in both directions, and equinoctial elements.
- **Frames (A4, B8):** GCRF, ITRF and TEME with bias-precession-nutation
  models and Earth orientation parameters.
- **Events and access (A10, B10):** event detection during propagation, and
  ground-station access windows with elevation constraints.
- **Relativity (B11)** and **covariance propagation, RTN and relative orbital
  elements (B5, B12).**
- **SGP4 and a CelesTrak client (B7).**
- Third body, SRP and tides as well, which fall under the B13 scope decision.

How to use it: as the **test oracle** for B14 (propagate the same state with
the same force model in both and compare), and as a source to **port**
algorithms from (the MIT licence allows it). Before using it at run time,
check whether its Python API vectorises over many satellites;
`initialize_eop()` also downloads Earth-orientation data.

### skyfielders/python-skyfield

**Licence:** MIT. Pure Python and numpy, built on the `sgp4` package.

- Time scales with leap seconds, TT and UT1 (A21).
- TEME, the true-of-date frames and GCRS, with IAU 2000 precession and
  nutation: exactly what a J2000 export needs (A4, B8).
- Atmospheric refraction for topocentric altitude (B10).
- An `EarthSatellite` (SGP4) class that would make SGP4 support routine (B7).

It is the lightest-weight way to fix the frame and time items, and matches
the project's "pure Python" preference.

### astropy/astropy

**Licence:** BSD-3. Heavier than skyfield, so best kept as a test-only
dependency:

- `Time` with every time scale, and IERS-based UT1 (A21).
- `TEME`, `GCRS` and `ITRS` frames (A4, B8): a test can check satflight's
  of-date frame against an independent implementation.
- `AltAz` with refraction (B10).

### astropy/astroquery

**Licence:** BSD-3. Its JPL Horizons interface can fetch reference
ephemerides of some Earth satellites (the ISS among them) as test data for
B14. Horizons' own satellite ephemerides come from tracking data or TLEs, so
they are a sanity check rather than truth.

### esa/pykep

**Licence:** MPL-2.0 (file-level copyleft: fine as an unmodified dependency).
**Install:** `pip install pykep`, or conda-forge (which still ships v1 while
v3 stabilises).

- Izzo's Lambert solver with multiple revolutions, robust near 180 deg: the
  B5 upgrade and the fix for the A16 edge cases.
- A test oracle for A1: compare satflight's chosen branch with pykep's
  prograde solution.

### nyx-space/hifitime

**Licence:** MPL-2.0. **Install:** `pip install hifitime`.

Leap-second-correct conversions between UTC, TAI, TT and GPS time, and UT1
when built with that feature. A small dependency for A21, and for the time
side of B8.

### HIPS/autograd

**Licence:** MIT. Differentiates numpy code automatically. It could produce
the state-transition matrix that differential correction needs (B5) and
propagate covariance (B12) from satflight's existing numpy force model.

Caveat: autograd does not support in-place assignment into arrays, so
`ForceModel.acceleration` and the integrators would need checking. Finite
differences are the simpler fallback.

### sympy/sympy

**Licence:** BSD-3. A development tool, not a run-time dependency:

- Derive and check the Brouwer short-period expressions (B4).
- Generate or test higher-degree gravity terms symbolically (B1), as the
  existing tests do numerically for J2-J4.
- Derive the J4 term of the sun-synchronous condition (A13).

### esa/pygmo2

**Licence:** MPL-2.0. Global optimisation (differential evolution, CMA-ES,
etc.). The launch planner tunes only the kick angle by grid plus
golden-section search. pygmo could tune several ascent parameters together
(vertical-rise time, kick, throttle profile, coast lengths) for B6, or
optimise multi-impulse transfers (B5). It would be an optional, opt-in
planner, since today's kick search takes about 0.3 s.

### OpenSCvx/OpenSCvx

**Licence:** Apache-2.0. Successive convexification on JAX: optimal
trajectories with constraints, such as fuel-optimal ascent or a rendezvous
approach with a keep-out zone (B5, B6). JAX is a heavy dependency, so this
suits offline experiments or a separate "optimise" tool rather than the core.

### loggger101/spacecost (your own)

**Licence:** MIT.

- The `propellants` table (vacuum Isp, bulk density, thruster device) is a
  ready source of engine and propellant presets for the propellant budget
  (B3).
- The `launch_vehicles` table (payload to LEO and GTO) gives a check for the
  launch model (B14): fly each preset vehicle to its quoted payload and see
  whether it reaches orbit. It could also seed more vehicle presets (B6).

### Karmanplus/prospector

**Licence:** Apache-2.0. A heliocentric low-thrust mission designer, so its
physics does not apply here. Its YAML configs, though, hold real engines,
propellants and vehicles from flown missions (Dawn, Psyche, Hayabusa2, DART):
a source of electric-propulsion presets for finite burns once B3 exists.

### juliensimon/space-datasets

**Licence:** none declared for the repository as a whole; each dataset
carries its source's terms. Published on Hugging Face as Parquet:

- **F10.7, Kp and CelesTrak space-weather files:** the input a solar-activity
  atmosphere needs (B2).
- **Re-entry events (35K) and TLE history since 1959:** real decays to
  validate drag against (B2, B14). The full TLE history is 10.9 GB, so take
  single objects.
- **CelesTrak SOCRATES conjunctions,** with miss distance and probability of
  collision: a sanity check for B12.
- **GCAT launch vehicles and stages:** data for more launch presets (B6).
- **Latest constellation TLEs:** ready-made scenario inputs (B7).

### pymc-devs/pymc

**Licence:** Apache-2.0 (GitHub shows "NOASSERTION"). Bayesian inference. It
could fit `density_scale`, or a satellite's `Cd·A/m`, to an object's observed
decay in its TLE history, with uncertainty (B2, B7). An offline experiment,
not a dependency.

## C4. Starred repositories to use as reference only (GPL / AGPL)

These cannot be copied into or imported by MIT-licensed satflight. They can
be read for ideas, or run separately to produce numbers to compare against.

- **hannorein/rebound (GPL-3.0).** The source of the WHFast idea already
  used. Its IAS15 integrator (adaptive, accurate to machine precision) could
  produce offline reference trajectories (B14). Its symplectic-corrector notes
  bear on A18.
- **dtamayo/reboundx (GPL-3.0).** Its `gr` effects show how general
  relativity is usually implemented (B11). Its `gravitational_harmonics` and
  `gas_drag` effects can cross-check satflight's J2/J4 and drag.
- **nyx-space/nyx (AGPL-3.0).** A high-fidelity, validated astrodynamics
  toolkit (harmonics, drag, covariance, orbit determination). Use it as an
  offline cross-check for B1, B2, B12 and B14. Because of the AGPL, never
  vendor or link it.
- **cuspaceflight/CamPyRoS (GPL-3.0).** A 6-DOF rocket simulator with
  variable mass and inertia, live wind data and aerodynamic heating. A design
  reference for B6 (Mach-dependent drag, winds, attitude).
- **shadden/celmech (GPL-3.0).** Already used for ideas (mean vs osculating
  semi-major axis). Its averaging and secular-theory material bears on B4.
- **matthewholman/assist (GPL-3.0).** Ephemeris-quality perturbations from
  JPL DE ephemerides. Relevant only if the Sun and Moon are brought back into
  scope (B13).

## C5. Not starred but worth considering

| Repository | Licence | Why |
|---|---|---|
| [SWxTREC/pymsis](https://github.com/SWxTREC/pymsis) | MIT | NRLMSIS 2.x densities from numpy arrays in, arrays out. The most direct fit for B2's vectorised hot path |
| [brandon-rhodes/python-sgp4](https://github.com/brandon-rhodes/python-sgp4) | MIT | Already an optional dependency. Making it a required one fixes most of B7 and A17's fallback path |
| [pleiszenburg/hapsira](https://github.com/pleiszenburg/hapsira) | MIT | Maintained fork of poliastro. Pure Python (numba) Izzo Lambert and perturbation models, easier to port from than pykep (B5) |
| [CS-SI/Orekit](https://github.com/CS-SI/Orekit) | Apache-2.0 | Java, with Python access through `orekit_jpype`. The industry-standard reference propagator for B14 |

## C6. Starred repositories that do not help with the physics

- **Visualisation:** bilawalsidhu/gods-eye-view, typpo/spacekit,
  CelestiaProject/Celestia, julie-dujardin/space-map.
- **Heliocentric work:** loggger101/AsteroidCatalog, loggger101/economicspace,
  loggger101/asteroid-belt-gradient (asteroid science and mining).
- **General tools:** pytest is already used. d3, streamlit, nicegui, polars,
  pyomo, z3, mesa, pygame and the agent-skill, style and data-engineering
  lists are not physics.

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
