# Physics and numerical models

Units throughout: km, s, kg (accelerations km/s^2). The Earth is the central
body; every state is geocentric. Only the Earth acts on the satellites: Sun
and Moon gravity and solar radiation pressure are out of the present scope
(see [SCOPE.md](SCOPE.md)). The Sun's position is still computed for
lighting, eclipses, beta angle and local solar time.

## Frames and time

| frame | definition | notes |
|---|---|---|
| ECI | geocentric pseudo-inertial frame *of date* (TEME-like) | not J2000/GCRF: precession has turned it ~0.37 deg since 2000 (about 45 km in LEO, 270 km at GEO against a J2000 tool); typed RAANs are of-date |
| ECEF | ECI rotated by GMST about +Z | polar motion neglected |
| geodetic | WGS-84 ellipsoid | Heikkinen's exact closed form (millimetre-exact, no iteration) |
| ENU | local east-north-up | station azimuth / elevation / range |
| RSW | radial, along-track, orbit normal | |
| VNB | velocity, orbit normal, V x N | the burn frame used by the planner |

The Earth turns into ECI by GMST alone, the Sun series is mean-of-date and
SGP4 returns TEME, so the integration frame is the equator and equinox of the
date. That is the usual frame at this level of model: precession and nutation
turn the Earth's spin axis, which is also J2's symmetry axis. The CSV holds
states in this frame.

Time: UTC stands in for UT1 (|UT1-UTC| < 0.9 s, i.e. < 14 arcsec of Earth
rotation) and for TT in the Sun series (69 s, 0.0008 deg: well inside the
series' own 0.01 deg). GMST uses the IAU-82 expression (Vallado eq. 3-47); a
test reproduces Vallado Example 3-5 to 1e-6 deg.

**Altitude: one meaning per use.** The size of an orbit is quoted as
`r - R_eq` (6378.137 km): orbit specs (`altitude`, `perigee_alt`,
`apogee_alt`), Walker shells, Hohmann and launch targets, the Orbit tab's
perigee and apogee altitudes and the regime names. Anything physical uses the
height above the WGS-84 ellipsoid at the actual point: drag, re-entry,
impact, the CSV's `alt_km`, the batch table, the tooltips, the satellite
list, and the air
density at perigee (looked up at the perigee point's real height). The two
differ by up to 21.4 km over the poles, where the table density at the
smaller figure is 1.6 times the real one at 300 km. Whether an orbit hits the
Earth is decided by its lowest height above the ellipsoid, so a polar orbit
clearing the poles by 15 km is not "suborbital".

## Equations of motion

    r'' = -mu r / r^3 + a_J2 + a_J3 + a_J4 + a_C22 + a_drag + a_thrust

**Zonal harmonics** - accelerations are the gradients of

    Phi = mu/r [1 - sum_n Jn (Re/r)^n Pn(sin phi)],   n = 2, 3, 4

with EGM-96 J2 = 1.0826e-3, J3 = -2.533e-6, J4 = -1.620e-6. The tests check
each closed-form term against a numerical gradient of Phi, and that the total
energy `v^2/2 - Phi` is conserved to 1e-10 over a day.

**Elliptical equator (C22/S22)**, off by default - the gradient of

    U22 = 3 mu Re^2 (C22 (x^2 - y^2) + 2 S22 x y) / r^5

in Earth-fixed x, y (EGM-96 C22 = 1.5745e-6, S22 = -9.038e-7). It turns with
the Earth, so the forces depend on time through GMST and energy is no longer
conserved; the energy readouts then show the Jacobi integral
`E - omega h_z`, conserved to 1e-11 over a day in the tests. Its visible
effect is on geostationary satellites: they accelerate in longitude toward
75 E or 105 W (up to ~0.0017 deg/day^2; 3.06 deg in 60 days from 45 deg
away, `geo_drift.json`). Launch ascents and the analytic propagators leave
it out.

**Geostationary slots.** A `geo` orbit sits where the equatorial pull
`mu / r^2 (1 + 3/2 J2 (Re/r)^2)` matches the Earth's rotation when J2 acts,
~0.5 km above the two-body R_GEO; at R_GEO it drifted east 0.8 deg a month.

**Drag** - `a = -1/2 rho (Cd A / m) |v_rel| v_rel` with `v_rel = v - omega x r`
(atmosphere co-rotating with the Earth), at the exact height above the
ellipsoid. A craft burning its engine gets lighter, so its `Cd A / m` follows
the falling mass. Density: piecewise-exponential model
of Vallado Table 8-4 (CIRA-72, mean solar activity) from 0 to 1000 km,
extrapolated to 2500 km and zero above. `density_scale` multiplies it to mimic
solar maximum (~3x at 400 km) or minimum (~0.3x); real densities vary by an
order of magnitude over the solar cycle, so decay times are indicative.

**Sun position (no force)** - Astronomical Almanac series (~0.01 deg,
Vallado Algorithm 29). It drives the Earth's lighting and the eclipse model:
`nu`, the visible fraction of the solar disc from the conical umbra /
penumbra model (Montenbruck & Gill 3.4.2), dims satellites in shadow and
times eclipse entry and exit.

**Thrust** - finite burns add `T/m(t)` along a direction fixed in the VNB,
RSW or ECI frame, with `m(t) = m0 - T t / (Isp g0)`. Impulsive burns spend
propellant through `m1 = m0 exp(-dv / (Isp g0))`.

**Propellant budget.** A satellite may have a `dry_mass`: what is left when
the propellant is gone. An impulsive burn then delivers at most
`Isp g0 ln(m / m_dry)` and a finite burn ends when the mass reaches it; either
way the log says how much of the burn was flown. The Orbit tab shows the
delta-v left (at Isp 300 s). Without a dry mass (0, the default) there is no
limit, as before, except that a finite burn still stops at 5 % of the mass
(and says so).

## Integration

The whole ensemble is one `(N, 6)` state, advanced with one shared step.

- **DOPRI5** (Dormand-Prince 5(4), FSAL) with error control per satellite:
  the step is accepted only if the worst satellite's RMS scaled error is
  below 1 (`atol + rtol |y|`, defaults 1e-6 and 1e-9). Velocities are weighed
  in km through each orbit's own time scale `sqrt(r^3/mu)` (~900 s in LEO), so
  one `atol` means the same for position and velocity; taken in km/s as they
  are, they were held ~100x less tightly than positions. A maximum step
  (default 60 s) keeps trails smooth. At the defaults the two-body error after
  a day is about 19 m in LEO and 2 m on a GTO; at `rtol` 1e-12 it is below 1 m
  (`tests/test_forces_integrators.py`).
- **RK4** fixed step.
- **Wisdom-Holman** (`wh`), fixed step: kick-drift-kick in which the drift
  is the exact Kepler orbit (the universal-variable solver below) and only
  the perturbations - J2-J4, drag, thrust - are kicks, the splitting behind
  REBOUND's WHFast (Wisdom & Holman 1991). The splitting error scales with
  the perturbations, about 1e-3 of gravity in LEO, instead of with gravity.
  Position error after one day, ISS orbit under J2-J4:

  | step | leapfrog | RK4 | Wisdom-Holman |
  |---|---|---|---|
  | 60 s | 1,000 km | 2.2 km | 1.2 km |
  | 120 s | 3,930 km | 65 km | 4.9 km |

  It is symplectic for the conservative forces (bounded energy error).
  Drag and thrust depend on velocity or time; the closing half-kick reads the
  drag at the velocity before that kick, so the drag coupling is first order
  in the step. The effect is negligible: |d a_drag / d v| is about 1e-10 /s at
  400 km and 1e-7 /s at 150 km.
- **Leapfrog** (kick-drift-kick with a straight-line drift), symplectic:
  bounded energy error, but a phase error that grows quickly. It needs steps
  of a few seconds in LEO; it is kept for comparison.

Integration stops exactly at every scheduled maneuver and burn boundary, and
restarts whenever a satellite leaves the active set (re-entry, impact,
escape).

The steps do not depend on the frame rate. A frame asks the simulation to
move on by warp x frame time, but the integration never shortens a step to
end on a frame: it takes whole steps up to the last one that fits, and the
state drawn for the frame is integrated from there on the side and discarded.
Advancing an hour in one call or in thousands of frames gives the same
trajectory and the same events (`tests/test_timestep.py`). A change made
between frames (adding, removing or editing a satellite) restarts the physics
from the state as shown.

Analytic alternatives: **kepler** (universal-variable Kepler solution with a
Laguerre-Conway root finder, valid for every conic) and **j2mean** (the same
plus the secular J2 rates

    dRAAN/dt = -3/2 n J2 (Re/p)^2 cos i
    dargp/dt =  3/4 n J2 (Re/p)^2 (4 - 5 sin^2 i)
    dM/dt    =  n + 3/4 n J2 (Re/p)^2 sqrt(1-e^2) (2 - 3 sin^2 i)

applied to mean elements).

The period has to come from the *mean* semi-major axis. J2 makes the
osculating one swing over each revolution by the first-order short-period
term (Kozai 1959)

    a_osc - a_mean = 3/2 J2 Re^2 / a [2/3 (1 - 3/2 sin^2 i)((a/r)^3 - (1-e^2)^-3/2)
                                      + sin^2 i (a/r)^3 cos 2(argp + nu)]

which is about +-6 km in LEO (and over 100 km at a Molniya perigee). Taking
the osculating value as mean put the ISS 816 km off after a day, farther
than plain Kepler; with the mean value j2mean stays within 10 km (ISS) to
50 km (700 km SSO) of a J2 Cowell run over a week, and a Molniya orbit within
230 km instead of 24,000 km (`tests/test_mean_elements.py`). The other elements
are taken as given: their short-period terms leave periodic errors of a few
to a few tens of km but no drift. With the term written as `K / a_mean`, the
conversion back from osculating to mean is the exact root of a quadratic,
so re-reading the state every step gives back the same mean value and the
result does not depend on how the run is chunked.

**Formations and orbit specs.** When J2 acts, the size of an orbit given by
elements (`altitude`, `a`, or `perigee_alt` / `apogee_alt`) is the mean one,
for a single satellite as for a Walker shell: each satellite gets the
osculating value for its own place in the orbit. A single satellite and a
one-plane shell at the same altitude therefore start on the same orbit (they
used to drift apart by degrees a day). With equal osculating radii instead,
the members' periods differ by up to ~0.3 % and a 1584-satellite shell at
550 km shears from 16.4 deg in-plane gaps to 4-28 deg within three days under
Cowell; with mean values the gaps stay within 0.05 deg. `"osculating": true`
in an orbit spec places the satellite exactly as given, as does a path whose
perigee is under the ground (it has no revolution to average over). Without J2 mean and
osculating are the same. Copies spread along one orbit (`count` > 1) share
the first copy's mean semi-major axis.

**Readouts and the `sso` keyword.** The Orbit tab's J2 node and perigee
drifts, nodal period, ground-track shift and sun-synchronous inclination are
computed from the mean semi-major axis and the mean inclination (Kozai's
first-order short-period term in i,
`3/8 J2 (Re/p)^2 sin 2i [cos 2u + e cos(2 argp + nu) + e/3 cos(2 argp + 3 nu)]`,
~0.02 deg in LEO, removed). They hold steady around the orbit and match what
a J2 run does: the ISS node drift is 0.08 % below the drift fitted to a 5-day
Cowell run wherever in the orbit it is read. That residual is J2's
second-order secular term, which is not modeled; with the osculating
inclination the readout also swung by 0.09 % around the orbit, and with the
osculating a it was 0.3 % off. `"i": "sso"` gives the inclination that makes
the *mean* orbit sun-synchronous, as the osculating one around it for the
satellite's starting place. In a Cowell run with J2 that inclination is
tuned to the run's whole zonal field: one revolution is flown from an
ascending node to the next (the short-period terms repeat there, so the
node's change is all secular) and the inclination is solved by the secant
method. The first-order J2 value left a 700 km orbit drifting 0.0008 deg/day
from the mean Sun with J2 alone (J2's second-order term) and 0.003 deg/day
with J3 and J4 (1 to 4.5 min of LTAN a year); tuned, it is within 3e-5
deg/day, 0.0065 deg (J2) or 0.024 deg (J2-J4) higher. The Orbit tab's
"Sun-sync inclination" stays the first-order value.

**LTAN** is mean local solar time at the ascending node, `UT + (RAAN - GMST)`,
because the node of a sun-synchronous orbit follows the mean Sun. The true
Sun runs up to 16 min ahead of it or behind it over the year (the equation of
time), so a node held at 10:30 used to read 10:16 to 10:46.

## Orbital elements

`rv2coe` / `coe2rv` follow Vallado Algorithms 9 and 10. For undefined
elements the conventions are: equatorial orbits use RAAN = 0 with the node
line along +X (so the argument of perigee becomes the longitude of perigee);
circular orbits use argp = 0 (so the true anomaly becomes the argument of
latitude). With these conventions the two functions round-trip exactly,
including retrograde-equatorial and hyperbolic orbits.

## Maneuver planning

Planners predict with the run's own propagator and forces
(`Simulation.coast_states`: J2-J4, C22 and drag when they are on), never with
plain two-body motion while the run has more.

- **Burn timing.** "Periapsis" and "apoapsis" are the lowest and highest
  points of the coming revolution of the trajectory the run will fly (the
  extreme of 180 samples, then the root of `r . v` by bisection to 1 us);
  "ascending/descending node" is the real equator crossing (Newton on z).
  Under J2 a "circular" orbit's osculating perigee can be anywhere, and the
  osculating node is seconds off.
- **Hohmann**: the first burn leaves the satellite moving level, at the speed
  that puts the far apsis at the target radius under the run's forces
  (vis-viva first, then Newton on the apsis radius found by flying the
  transfer: J2 alone pulls a two-body LEO-to-GEO apogee ~180 km short). Off an
  apsis the burn also levels the flight path, and the summary says what that
  costs; from 300 x 1,500 km at nu = 90 deg the old prograde-only burn ended
  90 km above a 2,000 km target. The second burn circularizes at the apsis.
- **Bi-elliptic**: the same aim for each leg, then a circularization.
  Aiming a transfer flies it (a LEO-to-GEO one is ~5,000 force evaluations),
  so the Plan maneuver dialog previews the two-body estimate while the form
  is edited, says so, and aims when the burns are scheduled. For the same
  reason planners look at most 7 days ahead (a LEO day is ~0.6 s of
  coasting): a later burn or rendezvous arrival is refused with a note to
  plan it nearer the time. Single burns (impulse, finite, circularize,
  plane change) have no such limit; they are worked out when they fire.
- **Circularize** gives the satellite the velocity, in its present plane, of a
  circular orbit for the run's forces: no once-per-revolution term in the
  radius over the coming revolution (found by Newton on the radial and
  along-track speed, flying the next revolution). Under J2 that is ~5 m/s
  faster than `sqrt(mu/r)` at the equator, and an inclined orbit keeps J2's
  forced ripple at twice the orbital frequency (1.9 km peak to peak at
  51.6 deg, 700 km); the two-body speed left 19 km swings at the equator.
- **Plane change**: velocity rotated about the radius vector; at the
  ascending node a positive angle raises the inclination.
- **Lambert**: universal-variable zero-revolution solver with bisection
  (Vallado Alg. 58). The way round is the chaser's own sense of motion,
  `(r1 x r2) . h1 > 0` for the short way: the old test on the ECI z-axis
  picked the reverse arc (~12x the delta-v) for polar orbits. Rendezvous
  planning optionally scans time of flight for the lowest total delta-v whose
  arc stays above 150 km, with the target predicted by the run's forces; the
  chosen arc is then corrected by Newton on the arrival miss (three trial arcs
  flown together for the sensitivities) until it arrives within 1 m of the
  target's predicted position, and a velocity-matching burn follows. With J2 a
  two-body arc missed by 11 km after 1 h and 90 km after 3 h; now the miss is
  metres even with drag. An arc through the Earth is refused, and one within
  2 deg of half a turn (plane barely defined) is flagged.

## Launch and ascent

`satflight/launch.py`. A launch is a stack of stages plus fairing and
payload on a pad at geodetic latitude, longitude and height. Before liftoff
its state is the pad's, rotating with the Earth. From liftoff it leaves the
ensemble and is integrated on its own with RK4 at 1 s steps (shortened to end
exactly at burnout, cutoff and ignition), under

    r'' = -mu r / r^3 + a_J2..J4 + a_drag + (T / m) u
    T   = mdot g0 Isp(h),   mdot = throttle T_vac / (g0 Isp_vac)
    Isp(h) = Isp_vac - (Isp_vac - Isp_sl) p(h)/p0

with `p(h)/p0` from the US Standard Atmosphere 1976 (seven layers to 86 km,
within 1e-4 of its published table; isothermal above). The mass flow is fixed
by the vacuum rating, so thrust rises with altitude as the nozzle
back-pressure falls. Drag uses the vehicle's `Cd` and frontal area and is
always applied during ascent, whatever the scenario's drag switch. An
acceleration limit, if set, throttles between 40 and 100 %; where even 40 %
exceeds it the log says so once per stage. The liftoff thrust-to-weight is
measured against the weight along the plumb line, gravity less the Earth's
centrifugal relief.

**Steering.**

1. Vertical rise along the local plumb line: minus (gravity + centrifugal
   acceleration), the geodetic vertical with J2. The geocentric radius is
   about 0.2 deg off at mid-latitudes, enough to tip a rocket with a
   thrust-to-weight of 1.2 over before it has any speed.
2. Pitch kick: thrust leans by the kick angle toward the launch azimuth until
   the air-relative velocity has leaned as far.
3. Gravity turn: thrust along the air-relative velocity (zero angle of attack).
4. Closed-loop guidance (orbit launches), from the first staging or 50 km.
   With `r`, `v_r`, `v_h` the radius and the vertical and in-plane horizontal
   speeds, the net radial acceleration is shaped as `A + B t`, with

       B = -12 (r_T - r - v_r t_go / 2) / t_go^3,   A = (-v_r - B t_go^2 / 2) / t_go

   so that `r(t_go) = r_T` and `v_r(t_go) = 0`. The thrust's radial share is
   `(A + mu/r^2 - v_h^2/r) / (T/m)`. The same law, with target 0, drives the
   position and velocity out of the target plane to zero. The time to go
   comes from the rocket equation over the remaining stages, with coasts
   between them, for the velocity still to gain plus the gravity to hold off.
   Steering is frozen for the last 4 s. The engines cut off when the
   specific energy reaches the target orbit's, `-mu / (2 a_T)`, so the payload
   is at perigee of the requested orbit.

Open-loop launches skip step 4 and burn every stage along a fixed azimuth.

**Plane and azimuth.** The orbit plane is the one of the requested
inclination that contains the pad at liftoff and crosses it northbound or
southbound. That is possible only for `|lat| <= i <= 180 - |lat|` (geocentric),
because dogleg ascents are not modeled. A **window** launch waits until the
pad rotates into a fixed plane (given RAAN), or into another satellite's
plane regressing with J2 (at its mean orbit's rate), found by scanning the pad's signed distance from
the plane and bisecting its zero crossings. The azimuth flown relative to the
rotating Earth, `atan2(v sin(b) - w R cos(lat), v cos(b))`, turns the inertial
in-plane azimuth `b` into one that ends in the plane once the orbital speed
`v` is reached. Guidance removes the remaining out-of-plane error.

**Kick optimization.** Left on `auto`, the kick is chosen by flying
candidates from 0.05 to 16 deg (grid, then golden-section) and keeping the
one that reaches the target with the most propellant left. For open-loop
flights it keeps the highest perigee, or, for sub-orbital ones, the highest
apogee. One flight takes about 10 ms: the ascent uses plain Python floats,
not numpy.

**Bookkeeping.** Staging drops each stage's dry mass. The fairing goes at
its jettison height, or at cutoff if the ascent never got that high (it used
to leave with the last stage). At cutoff the last stage, with its leftover
propellant, separates from the payload. Spent stages
are coasted to the ensemble's time and added as satellites of the same
launch "family"; close-approach alerts inside a family are suppressed. The
delta-v budget integrates `T/m` (ideal), `-g . v_hat` (gravity loss), drag
along `v_hat`, and `(T/m)(1 - cos alpha)` (steering loss, measured against
the inertial velocity, so it includes the vertical rise across the Earth's
rotation). Ideal minus losses equals the speed gained, to about 1 %.

**Accuracy.** For the preset vehicles, insertion is within 5 km of the
target perigee and apogee (0.2 % on GTO apogee) and 0.05 deg in inclination.
Launch windows put the RAAN within 0.1 deg. Vehicle figures are approximate
public numbers: performance is realistic in kind, not for mission design.

## Events and limits

- Re-entry: height < 80 km while descending, including a dip below 80 km
  between two steps that has come back up by the end of the step. Impact:
  height <= 0. Launches starting low are not flagged while they climb.
- Escape: distance > 924,000 km (Earth's sphere of influence w.r.t. the Sun).
  Beyond it an Earth-centered model is no longer appropriate.
- Every event is stamped when it happens, not at the end of the step that
  sees it: the step's cubic Hermite curve (below) is searched for the
  crossing of half the Sun's disc (eclipse), the station's elevation mask
  (AOS/LOS), 80 km, 0 km or the sphere of influence. Against a run with 1 s
  steps the times agree to 8 ms; at the end of the step they were up to 60 s
  late.
- Close approaches (below 10 km) are found between steps too: each pair's
  relative motion over a step is the cubic Hermite curve through both ends'
  positions and velocities, and its minimum gives the distance and time of
  closest approach that are logged. The curve stays within
  4/27 (|m0 - c| + |m1 - c|) of its chord c (m = velocity x step), and the
  same bound per satellite rules out nearly every pair from the current
  distances alone, so the search costs little.
  Up to 300 satellites every pair is measured. Larger ensembles never
  measure all N^2 pairs: during a step each satellite stays within half its
  chord plus that bound of the chord's middle, so the middles are binned in
  a grid of cubes (alert distance + twice the largest such spread, ~470 km
  in LEO at 60 s steps) and only satellites in the same or neighboring cubes
  are paired. For the 1584-satellite shell that is ~2 ms per step. The
  footer's "closest pair" is then the closest among those candidates.
- Eclipses, AOS/LOS and close approaches are looked for at every accepted
  step, not at every frame (per-satellite detail for ensembles up to 40
  satellites, otherwise for the selected one), so an entry can reach the log
  up to one step after the view shows it, stamped with the time it happened.

## Known simplifications

- Of the non-zonal harmonics only C22/S22 (no C21/S21, C31, ...).
- The inertial frame is of date (see Frames); no polar motion or UT1-UTC.
- The analytic propagators apply only two-body motion (kepler) or two-body
  plus J2 (j2mean); a run that has other terms switched on logs a warning
  naming them.
- No Sun or Moon gravity and no solar radiation pressure (out of scope,
  [SCOPE.md](SCOPE.md)); the low-precision Sun ephemeris is used for lighting
  and eclipses only.
- Static exponential atmosphere (no diurnal bulge, no space-weather input).
- Launch vehicles: point mass with instantaneous attitude (no rotational
  dynamics or aerodynamic loads, no winds), serial staging only (no strap-on
  boosters burning alongside a core), no engine-out, a single burn to orbit
  (no parking-orbit coast and restart; schedule maneuvers for that).
- Cannonball drag (no attitude-dependent areas).
- Without the optional `sgp4` package, a TLE is read as J2 mean elements:
  its mean motion is taken as the mean anomaly's rate under J2 (true to
  first order, as in SGP4), which fixes the mean semi-major axis; the
  short-period term above makes it osculating, and j2mean carries the state
  from the TLE epoch, so the plane regresses. For the ISS that is 0.004 deg
  of RAAN after 3 days (pure Kepler from the epoch was 15 deg off) and
  0.13 deg along track after a day of Cowell. Drag (B*) and SGP4's
  higher-order terms are not applied. With `sgp4`, a TLE SGP4 cannot carry to
  the requested time (a decayed object, say) is an error, not a quiet switch
  to the J2 reading.
- Mean elements: the semi-major axis everywhere and the inclination in the
  readouts (first-order J2). The short-period terms of e, RAAN, argp and M
  are neglected, so a satellite started "circular" from its elements carries
  an osculating-start eccentricity of ~1e-3 (a few km of radius swing), and
  j2mean, which puts the whole short-period term into a, shows a radius
  ripple of +-6 km at 51.6 deg where Cowell has +-1 km.
- The j2mean propagator and the Orbit tab's node drift use first-order J2
  secular rates; J2's second-order term adds ~0.08 % to the node drift.
