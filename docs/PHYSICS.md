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
- **Leapfrog** (kick-drift-kick), symplectic: bounded energy error for
  conservative forces over arbitrarily long runs, as in REBOUND's WHFast
  family.

Integration stops exactly at every scheduled manoeuvre and burn boundary, and
restarts whenever a satellite leaves the active set (re-entry, impact,
escape).

Analytic alternatives: **kepler** (universal-variable Kepler solution with a
Laguerre-Conway root finder, valid for every conic) and **j2mean** (the same
plus the secular J2 rates

    dRAAN/dt = -3/2 n J2 (Re/p)^2 cos i
    dargp/dt =  3/4 n J2 (Re/p)^2 (4 - 5 sin^2 i)
    dM/dt    =  n + 3/4 n J2 (Re/p)^2 sqrt(1-e^2) (2 - 3 sin^2 i)

applied to the osculating elements treated as mean elements).

## Orbital elements

`rv2coe` / `coe2rv` follow Vallado Algorithms 9 and 10. For undefined
elements the conventions are: equatorial orbits use RAAN = 0 with the node
line along +X (so the argument of perigee becomes the longitude of perigee);
circular orbits use argp = 0 (so the true anomaly becomes the argument of
latitude). With these conventions the two functions round-trip exactly,
including retrograde-equatorial and hyperbolic orbits.

## Manoeuvre planning

- **Hohmann**: first burn along V sized by vis-viva from the actual radius at
  the burn point; the second burn is a *circularize* manoeuvre computed from
  the true state on arrival, so perturbations during the coast are absorbed.
- **Bi-elliptic**: two prograde burns plus a final circularisation.
- **Plane change**: velocity rotated about the radius vector; at the
  ascending node a positive angle raises the inclination.
- **Lambert**: universal-variable zero-revolution solver with bisection
  (Vallado Alg. 58). Rendezvous planning optionally scans time of flight for
  the lowest total delta-v whose arc stays above 150 km, then schedules a
  velocity-matching burn at arrival.

## Events and limits

- Re-entry: altitude < 80 km while descending. Impact: altitude <= 0.
  Launches starting low are not flagged while they climb.
- Escape: distance > 924,000 km (Earth's sphere of influence w.r.t. the Sun).
  Beyond it an Earth-centred model is no longer appropriate.
- Eclipses, AOS/LOS and close approaches are evaluated at every accepted
  step (per-satellite detail for ensembles up to 40 satellites, otherwise for
  the selected one; pairwise approaches up to 300 satellites).

## Known simplifications

- No tesseral/sectoral harmonics (only zonal J2-J4).
- No precession/nutation, polar motion or UT1-UTC.
- No Sun or Moon gravity and no solar radiation pressure (out of scope,
  [SCOPE.md](SCOPE.md)); the low-precision Sun ephemeris is used for lighting
  and eclipses only.
- Static exponential atmosphere (no diurnal bulge, no space-weather input).
- Cannonball drag (no attitude-dependent areas).
- Without the optional `sgp4` package, TLE mean elements are used as
  osculating elements (a few km of error).
