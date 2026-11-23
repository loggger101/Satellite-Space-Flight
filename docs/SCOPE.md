# Scope

**Decision (2026-11-18), revising 2026-09-29:** the project's goal is to map
Sentinel-1A and 1C against ESA's precise orbits, accurate to a few cm. For that the
user brought back the Sun's and the Moon's gravity and sunlight pressure, and
added the full EGM2008 gravity field. Each is a **per-scenario switch**: the
Sentinel-1 work turns them on, and the bundled demos keep the Earth-only J2-J4
model they were built for, at the same speed. The plan is in
[specs/2026-11-18-sentinel1-poeorb-plan.md](specs/2026-11-18-sentinel1-poeorb-plan.md).

The 2026-09-29 decision ("the Earth and the satellites only") still covers
the display: the Moon is not drawn, and the items listed under "Still
removed" below stay out. The last commit that had all of them is `c079ded`.

## In scope

- **Earth gravity:** point mass plus either
  - the separate terms J2, J3, J4 and C22/S22 (the elliptical equator), or
  - **EGM2008 to degree and order N** (2..70, `ForceModel.gravity_degree`;
    fully normalized, tide-free, from ICGEM). It stands in for the separate
    terms in the motion, while the J switches still drive the analytic
    helpers. The coefficients are in `satflight/data/egm2008_70.gfc`
    (`tools/make_gravity.py` makes them).
- **Atmospheric drag:** the atmosphere co-rotates with the Earth.
- **Third-body gravity** of the Sun and the Moon (`sun`, `moon`), from the
  low-precision almanac series in `ephemeris.py`.
- **Solar radiation pressure** (`srp`): a cannonball model with `Cr` per
  satellite (default 1.3, same area as drag), dimmed in the Earth's penumbra
  and umbra.
- **Thrust:** impulsive and finite burns.
- **The Sun's position** for lighting, eclipses, beta angle and LTAN.
- **The sun-synchronous inclination** (`"i": "sso"`). It is a J2 property of
  the Earth's field and does not use the Sun's position.
- **The escape threshold** of 924,000 km, a distance limit for an
  Earth-centered model.
- **The star field and Milky Way**, a decorative backdrop (no physics).

Not modelled, and named in the Sentinel-1 report as what is left:
- polar motion in the simulator's own Earth (the truth is corrected for it;
  the field's 0.5 arcsec tilt is centimetres);
- solid-Earth and ocean tides;
- relativity;
- Earth albedo and infrared pressure;
- a space-weather-driven atmosphere (the table's density scale is fitted
  instead).

## Still removed (since 2026-09-29)

### Display

- The Moon in the 3-D view (phased disc, depth-sorted against the Earth).
- The sub-lunar point on the ground-track map.
- The *Reflectivity Cr* field in the Add satellite and Edit satellite
  dialogs, and the *SRP acceleration* row and `Cr` in the Orbit tab. `Cr` is
  set in scenario files (`"cr"`). The Physics dialog (P) has the force
  switches, and the Telemetry tab lists each force's acceleration.

### Scenarios and presets

- `scenarios/third_body.json` ("Third-body perturbations": a lunar transfer
  orbit, a GEO satellite and an HEO under Sun, Moon and SRP) and its
  generator.
- The *Lunar transfer (TLI)* satellite preset.
- `gps_constellation.json` and `molniya_tundra.json` still run Earth-only.

## Older scenario files

Files written between 2026-09-29 and 2026-11-18 that still switch on `sun`,
`moon` or `srp`, or give a satellite a `cr`, now get those forces: they are
no longer ignored. The MATLAB export flies every one of these terms: the Sun,
the Moon, sunlight pressure and the EGM2008 field (its coefficients written
into the script, so it needs no data file). A replayed Sentinel-1 unit exports
too: the script reads the same precise orbit files when it runs.
