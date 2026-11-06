# MATLAB toolkit

Build Satellite Space Flight scenarios in MATLAB, run them through the
simulator without graphics, and analyse the results as MATLAB tables and
plots. The simulator itself stays in Python: these functions write the same
scenario JSON the app opens, call `python -m satflight.batch`, and read the CSV
it writes back in. No MATLAB toolbox is needed.

```matlab
addpath("path/to/Satellite Space Flight/matlab")

sc = satflight.newScenario("Polar imager", Drag=true, ...
    Epoch=datetime(2026,9,29,12,0,0,TimeZone="UTC"));
sc = satflight.addSatellite(sc, "Imager", satflight.orbitElements(Altitude=550, I="sso"));
sc = satflight.addStation(sc, "Svalbard", 78.23, 15.39, MinEl=5);
path = satflight.saveScenario(sc, fullfile(satflight.repoRoot, "scenarios", "user", "imager.json"));

satflight.checkScenario(path);                       % the simulator's verdict
T = satflight.runScenario(path, Duration="1d", Step=60);
satflight.plotGroundTrack(T, Scenario=sc);
```

Files saved to `scenarios/user/` show up in the app's Open list (Ctrl+O). Two
complete scripts are in `examples/`: `buildScenarioExample.m` (build, check,
run, plot) and `analyzeRunExample.m` (fits J2 nodal regression from a run and
compares it with theory).

## Building scenarios

| function | what it adds |
|---|---|
| `newScenario(name, ...)` | epoch, propagator, force terms (`J2`, `J3`, `J4`, `C22`, `Drag`), integrator settings |
| `addSatellite(sc, name, orbit, ...)` | a satellite, or `Count` copies along one orbit; `Mass`, `Area`, `Cd`, `Color` |
| `addConstellation(sc, name, ...)` | a Walker shell: `Altitude`, `Inclination`, `Total`, `Planes`, `Phasing`, `Pattern` |
| `addStation(sc, name, lat, lon, ...)` | a ground station; `Alt`, `MinEl` |
| `addManeuver(sc, sat, kind, ...)` | a scheduled burn: `Timing`, `T`/`Delay`, `Dv` + `Frame`, `DeltaI`, finite-burn `Thrust`/`Isp`/`Duration` |
| `addLaunch(sc, name, ...)` | a rocket launch: `Vehicle` preset or struct, pad, timing, target orbit, payload |

Orbits for `addSatellite`:

| function | orbit |
|---|---|
| `orbitElements(...)` | `A` + `E`, or `Altitude`, or `PerigeeAlt` + `ApogeeAlt`; `I` (deg or `"sso"`), `Raan`, `Argp`, `Nu` or `M` |
| `orbitState(r, v)` | ECI position (km) and velocity (km/s) |
| `orbitGeo(lon)` | geostationary slot |
| `orbitSurface(...)` | from a point over the ground: `Lat`, `Lon`, `Alt`, `Speed`, `Azimuth`, `Fpa` |
| `orbitTle(line1, line2)` | a two-line element set |

Every function has `help`, e.g. `help satflight.addManeuver`. Options you
leave out are not written to the file, so the simulator's own default applies.

`loadScenario(path)` reads any scenario (the bundled ones included) into a
struct whose lists are cell arrays, to change and save under a new name.

**Checking.** The simulator is the one judge of what a scenario may contain.
`checkScenario` hands it the file; an unknown option, a value out of range or
an orbit inside the Earth comes back as a MATLAB error (`satflight:python`)
with the simulator's reason. MATLAB itself refuses only what cannot be written
faithfully: a NaN or Inf, an epoch without a time zone, a misspelt option name.

## Running and analysing

| function | |
|---|---|
| `[T, report] = runScenario(scenario, ...)` | propagate a struct or file: `Duration` (`"12h"`, `"3d"` or seconds), `Step`, `Out` (keep the CSV), `Propagator` |
| `T = readRun(csv)` | read a CSV from `runScenario`, `python -m satflight.batch --out`, or the app's Scenarios > Export CSV |
| `plotGroundTrack(T, ...)` | tracks on a coastline map; `Names`, `Scenario` (marks its stations) |
| `plotAltitude(T, ...)` | altitude against time |
| `plotElements(T, ...)` | a, e, i, RAAN, argument of perigee (unwrapped) and altitude |
| `plotOrbit3D(T, ...)` | trajectories in ECI around a to-scale Earth |

`T` has one row per satellite per sample: `t_s`, `utc` (datetime, UTC),
`name`, `status`, ECI `x_km`..`vz_kms`, `alt_km`, `lat_deg`, `lon_deg`, and the
osculating `a_km`, `e`, `i_deg`, `raan_deg`, `argp_deg`, `nu_deg`. Select a
satellite with `T(T.name == "ISS", :)`.

## Exporting a running simulation

The app's Scenarios dialog (Ctrl+O) has an **Export MATLAB** button. It writes
a live script to `scenarios/user/exports/` that runs the simulation again in
MATLAB on its own, starting at the current moment or at the scenario's start
(your choice). Open it in the Live Editor and press Run. Its sections:

1. **Settings**: `duration`, `step`, and `crossCheck`.
2. **Scenario**: epoch, force model, propagator, each satellite's exact ECI
   state and drag properties, the scheduled burns, ground stations.
3. **Propagate**: `satflightRun` returns the same table as `readRun` above.
4. **Plots**: ground track, altitude, elements, 3-D orbits.
5. **Cross-check**: runs the Python simulator on the same scenario through
   this toolkit and tabulates and plots the position difference. It needs
   this folder and Python; set `crossCheck = false` to skip it elsewhere.

The physics after the sections is a port of the simulator's own (forces J2-J4,
C22, drag; every burn kind and timing; finite burns; re-entry at 80 km; the
Kepler and J2-mean propagators). For numerical runs it uses `ode45` with the
scenario's tolerances. `tests/ExportTest.m` exports several scenarios that
between them use every feature, and requires every satellite still in orbit to
match the simulator within 10 m. One difference is expected: a satellite that
re-enters stops at the instant it drops below 80 km here, but at the end of the
integrator step in the simulator, so the two frozen positions differ by that
step's travel. Rocket ascents are not ported. Launches still on the pad are
listed in the script and appear only in the cross-check, and a vehicle exported
in powered flight continues as a free-flying object.

## Which Python

The functions run `python` from the PATH with this checkout first on
`PYTHONPATH`. If that is not the Python with numpy installed, point at the
right one before calling them:

```matlab
setenv("SATFLIGHT_PYTHON", "C:\path\to\python.exe")
```

## Tests

```bash
matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests'))"
```

`tests/test_matlab.py` runs the same suite under pytest when MATLAB is on the
PATH (and is skipped, saying so, when it is not). It reads every bundled
scenario through MATLAB and back and has the simulator confirm nothing changed,
builds a scenario with every kind of entry and runs it, and checks that the
simulator's refusals reach MATLAB as errors.
