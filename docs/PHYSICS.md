# Physics and numerical models

Units throughout: km, s, kg (accelerations km/s^2). The Earth is the central
body; every state is geocentric. Only the Earth acts on the satellites: Sun
and Moon gravity and solar radiation pressure are out of the present scope
(see [SCOPE.md](SCOPE.md)). The Sun's position is still computed for
lighting, eclipses, beta angle and local solar time.

## Frames and time

| frame | definition | notes |
|---|---|---|
| ECI | geocentric inertial, treated as J2000/GCRF | precession and nutation neglected (< 0.4 deg over 2000-2030) |
| ECEF | ECI rotated by GMST about +Z | polar motion neglected |
| geodetic | WGS-84 ellipsoid | iterative latitude, pole-safe height formula |
| ENU | local east-north-up | station azimuth / elevation / range |
| RSW | radial, along-track, orbit normal | |
| VNB | velocity, orbit normal, V x N | the burn frame used by the planner |

Time: UTC stands in for UT1 (|UT1-UTC| < 0.9 s, i.e. < 14 arcsec of Earth
rotation) and for TT in the ephemerides. GMST uses the IAU-82 expression
(Vallado eq. 3-47); a test reproduces Vallado Example 3-5 to 1e-6 deg.

## Equations of motion

    r'' = -mu r / r^3 + a_J2 + a_J3 + a_J4 + a_drag + a_thrust

**Zonal harmonics** - accelerations are the gradients of

    Phi = mu/r [1 - sum_n Jn (Re/r)^n Pn(sin phi)],   n = 2, 3, 4

with EGM-96 J2 = 1.0826e-3, J3 = -2.533e-6, J4 = -1.620e-6. The tests check
each closed-form term against a numerical gradient of Phi, and that the total
energy `v^2/2 - Phi` is conserved to 1e-10 over a day.

**Drag** - `a = -1/2 rho (Cd A / m) |v_rel| v_rel` with `v_rel = v - omega x r`
(atmosphere co-rotating with the Earth). Density: piecewise-exponential model
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

## Integration

The whole ensemble is one `(N, 6)` state, advanced with one shared step.

- **DOPRI5** (Dormand-Prince 5(4), FSAL) with error control per satellite:
  the step is accepted only if the worst satellite's RMS scaled error is
  below 1 (`atol + rtol |y|`, defaults 1e-6 and 1e-9). A maximum step (default
  60 s) keeps trails smooth and event detection responsive. Against the exact
  Kepler solution the error after one day is below 1 m.
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

  It is symplectic for the conservative forces (bounded energy error);
  drag and thrust depend on velocity or time, so with them it is only an
  accurate second-order method.
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

**Formations.** Walker constellations take their altitude as the mean one,
and copies spread along one orbit (`count` > 1) share the first copy's mean
semi-major axis: each member gets the osculating value for its own place in
the orbit. With equal osculating radii instead, the members' periods differ
by up to ~0.3 % and a 1584-satellite shell at 550 km shears from 16.4 deg
in-plane gaps to 4-28 deg within three days under Cowell; with mean values the
gaps stay within 0.05 deg. A single satellite given by elements is placed
exactly as given (osculating).

## Orbital elements

`rv2coe` / `coe2rv` follow Vallado Algorithms 9 and 10. For undefined
elements the conventions are: equatorial orbits use RAAN = 0 with the node
line along +X (so the argument of perigee becomes the longitude of perigee);
circular orbits use argp = 0 (so the true anomaly becomes the argument of
latitude). With these conventions the two functions round-trip exactly,
including retrograde-equatorial and hyperbolic orbits.

## Maneuver planning

- **Hohmann**: first burn along V sized by vis-viva from the actual radius at
  the burn point; the second burn is a *circularize* maneuver computed from
  the true state on arrival, so perturbations during the coast are absorbed.
- **Bi-elliptic**: two prograde burns plus a final circularization.
- **Plane change**: velocity rotated about the radius vector; at the
  ascending node a positive angle raises the inclination.
- **Lambert**: universal-variable zero-revolution solver with bisection
  (Vallado Alg. 58). Rendezvous planning optionally scans time of flight for
  the lowest total delta-v whose arc stays above 150 km, then schedules a
  velocity-matching burn at arrival.

## Launch and ascent

`satflight/launch.py`. A launch is a stack of stages plus fairing and
payload on a pad at geodetic latitude, longitude and height. Before liftoff
its state is the pad's, rotating with the Earth. From liftoff it leaves the
ensemble and is integrated on its own with RK4 at 1 s steps (shortened to end
exactly at burnout, cutoff and ignition), under

    r'' = -mu r / r^3 + a_J2..J4 + a_drag + (T / m) u
    T   = mdot g0 Isp(h),   mdot = throttle T_vac / (g0 Isp_vac)
    Isp(h) = Isp_vac - (Isp_vac - Isp_sl) p(h)/p0,   p(h)/p0 = exp(-h / 7 km)

The mass flow is fixed by the vacuum rating, so thrust rises with altitude as
the nozzle back-pressure falls. Drag uses the vehicle's `Cd` and frontal area
and is always applied during ascent, whatever the scenario's drag switch.
An acceleration limit, if set, throttles between 40 and 100 %.

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
plane regressing with J2, found by scanning the pad's signed distance from
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

**Bookkeeping.** Staging drops each stage's dry mass. At cutoff the last
stage, with its leftover propellant, separates from the payload. Spent stages
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

- Re-entry: altitude < 80 km while descending. Impact: altitude <= 0.
  Launches starting low are not flagged while they climb.
- Escape: distance > 924,000 km (Earth's sphere of influence w.r.t. the Sun).
  Beyond it an Earth-centered model is no longer appropriate.
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
- Eclipses, AOS/LOS and close approaches are evaluated at every accepted
  step, not at every frame, so an entry can reach the log up to one step
  after the view shows it (per-satellite detail for ensembles up to 40 satellites, otherwise for
  the selected one).

## Known simplifications

- No tesseral/sectoral harmonics (only zonal J2-J4).
- No precession/nutation, polar motion or UT1-UTC.
- No Sun or Moon gravity and no solar radiation pressure (out of scope,
  [SCOPE.md](SCOPE.md)); the low-precision Sun ephemeris is used for lighting
  and eclipses only.
- Static exponential atmosphere (no diurnal bulge, no space-weather input).
- Launch vehicles: point mass with instantaneous attitude (no rotational
  dynamics or aerodynamic loads, no winds), serial staging only (no strap-on
  boosters burning alongside a core), no engine-out, a single burn to orbit
  (no parking-orbit coast and restart; schedule maneuvers for that), and an
  ambient-pressure model of `exp(-h / 7 km)`.
- Cannonball drag (no attitude-dependent areas).
- Without the optional `sgp4` package, TLE mean elements are used as
  osculating elements (a few km of error).
- Mean elements cover the semi-major axis only (first-order J2); the
  short-period terms of e, i, RAAN, argp and M are neglected.
