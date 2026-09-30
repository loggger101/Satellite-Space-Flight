# Scope

**Decision (2026-09-29):** the simulation models the Earth and the satellites
only. The Sun, the Moon and every other body exert no influence on any
satellite's orbit. Everything listed under "Removed" was declared **out of
the present scope for now**; it was removed deliberately, not lost, and can be
restored from git history (the last commit that has all of it is `c079ded`).

## In scope

- Earth gravity: point mass plus the zonal harmonics J2, J3 and J4.
- Atmospheric drag (the atmosphere co-rotates with the Earth).
- Thrust: impulsive and finite burns.
- **The Sun's position**, used only for things that do not change an orbit:
  - lighting of the Earth: day/night terminator, twilight band, ocean glint
    and the reddened sunset limb;
  - the Sun drawn in the 3-D sky, and the sub-solar point with day/night
    shading on the ground-track map;
  - Earth's shadow: satellites dimming in eclipse, eclipse entry/exit events,
    the sunlight/eclipse timeline and "time in shadow" in the Orbit tab;
  - beta angle and local time of the ascending node (LTAN).
- The sun-synchronous inclination (preset `"i": "sso"` and the Orbit tab
  value). It is a J2 property of the Earth's field that matches a fixed rate
  of 0.9856 deg/day; it does not use the Sun's position.
- The escape threshold of 924,000 km (Earth's sphere of influence) stays as
  a distance limit for an Earth-centered model; nothing is computed about the
  Sun there.
- The star field and Milky Way remain as a decorative backdrop (no physics).

## Removed (out of the present scope)

### Physics

| removed | what it did | where it lived |
|---|---|---|
| Sun third-body gravity | Sun's tidal pull on every satellite | `forces.accel_third_body`, `ForceModel.sun`, constant `MU_SUN` |
| Moon third-body gravity | Moon's tidal pull on every satellite | `forces.accel_third_body`, `ForceModel.moon`, constant `MU_MOON` |
| Solar radiation pressure | sunlight pushing on the satellite (cannonball model, shadowed in eclipse) | `forces.accel_srp`, `ForceModel.srp`, constant `P_SRP` |
| Reflectivity coefficient `Cr` | satellite property used only by SRP | `Satellite.cr`, `SatSpec.cr`, `ConstellationSpec.cr`, presets |
| Moon ephemeris | geocentric Moon position (truncated Brown series) | `ephemeris.moon_position`, constant `R_MOON` |

`ForceModel.acceleration` and `ForceModel.breakdown` no longer take the Julian
date or `Cr*A/m`: with only the Earth acting, the forces do not depend on time.

### Display

- The Moon in the 3-D view (phased disc, depth-sorted against the Earth).
- The sub-lunar point on the ground-track map.
- The *Sun third-body*, *Moon third-body* and *Solar radiation pressure*
  switches in the Physics dialog.
- The `a_sun`, `a_moon` and `a_srp` rows of the Telemetry tab's acceleration
  budget.
- The *Reflectivity Cr* field in the Add satellite and Edit satellite
  dialogs; the *SRP acceleration* row and `Cr` in the Orbit tab.

### Scenarios and presets

- `scenarios/third_body.json` ("Third-body perturbations": a lunar transfer
  orbit, a GEO satellite and an HEO under Sun, Moon and SRP) and its
  generator in `tools/make_scenarios.py`.
- The *Lunar transfer (TLI)* satellite preset (a 200 x 384,000 km transfer
  orbit whose purpose is to reach the Moon).
- `gps_constellation.json` and `molniya_tundra.json` had Sun and Moon gravity
  switched on; they now run with the Earth only.

### Tests

- `test_moon_phases` (Moon ephemeris against known new and full moons).
- The third-body scenario's entry in the bundled-scenario run.
- Added `test_sun_and_moon_forces_are_out_of_scope`.

## Older scenario files

A scenario file that still switches on `sun`, `moon` or `srp`, or gives a
satellite a `cr`, loads normally: those settings are ignored and the event
log shows *"Ignored (out of scope): ..."*. Saving it again writes it without them.
