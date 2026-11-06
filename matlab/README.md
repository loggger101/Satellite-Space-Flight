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
| `[T, report, E] = runScenario(scenario, ...)` | propagate a struct or file: `Duration` (`"12h"`, `"3d"` or seconds), `Step`, `Out` (keep the CSV), `Events` (keep the event log), `Propagator`; `E` is the event log (`t_s`, `utc`, `kind`, `text`) |
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
(your choice), either **with a cross-check** against the simulator (the
default) or **MATLAB only** (`--no-compare` from the command line): no
cross-check section, no `crossCheck` setting and none of the comparison code,
so it needs nothing but MATLAB. Open it in the Live Editor and press Run. Its
sections:

1. **Settings**: `duration`, `step`, `crossCheck` (with a cross-check), with launches
   `planLaunches` (false: fly the simulator's plan; true: find the launch
   window and search the pitch kick here, as the simulator does when a
   scenario loads, e.g. after editing a launch), `animate` and
   `animationFile` (see Animation).
2. **Scenario**: epoch, force model, propagator, each satellite's exact ECI
   state (with a comment saying what orbit it is: perigee x apogee altitude
   and inclination, sub-orbital or escaping) and drag properties, the
   scheduled burns, ground stations.
3. **Launches**: per launch the vehicle's stages, the site, the target orbit
   and timing, the simulator's plan and, for a vehicle exported in powered
   flight, its flight state.
4. **Run**: `[T, events, launches] = satflightRun(...)`. `T` is the same table
   as `readRun` above, a payload from its pad and each spent stage from its
   separation; `events` the event log, line for line as the simulator writes
   it; `launches` per launch the outcome, insertion orbit, delta-v budget and
   losses, max Q, and the ascent profile (`profile`, `marks`, and the plan's).
   Shown: `keyEvents` (launches, burns, alerts), `atEnd` (where each object is
   at the end of the run) and `launchTable`.
5. **Plots**: ground track, altitude, elements, 3-D orbits (around the Earth
   with its coastlines, turned as it is at the end of the run), and each ascent
   (`satflightPlotAscent`), of the objects in `shown`: all of them up to 12,
   past that those still in orbit first. Set `shown` to the names you want.
6. **Passes and eclipses**, from the event log: `passes` (with ground
   stations) has each station pass, its AOS and LOS, minutes in view and the
   highest elevation at the samples in between; `eclipses` each object's
   number of eclipses, minutes in shadow, longest, and share of its time in
   the run; and a timeline draws both for the objects in `shown`. A pass or
   eclipse under way when the run (or the object) ends has LOS or stop `NaT`.
   The simulator logs these only while a run has at most 40 objects; past
   that the section says so.
7. **Animation** (`animate = false` to skip it): plays the run back, the
   ground track beside the orbits around the turning Earth, each object over
   its last orbit and marked where it is (o in orbit, x come down or gone),
   with the latest lines of the event log. In a figure window it plays in
   20 s, or as fast as its frames draw if that is slower.
   `animationFile = "run.gif"` also writes every frame to an animated GIF
   that plays in the same time; drawing and writing each frame takes a while.
8. **Cross-check** (not in a MATLAB-only script): runs the Python simulator on the same scenario through
   this toolkit, tabulates and plots the position difference
   (`satflightCompare`) and pairs up the two event logs (`satflightCompareEvents`).
   It needs this folder and Python; set `crossCheck = false` to skip it elsewhere.

The code after the sections is a port of the simulator's own: forces J2-J4,
C22 and drag; every burn kind and timing; finite burns; the Kepler and J2-mean
propagators; the rocket ascent step for step (fixed 1 s RK4, thrust varying
with altitude, drag, throttling, vertical rise, pitch kick, gravity turn,
closed-loop guidance, staging, fairing jettison, cutoff on the target energy,
spent stages, circularization at apogee); the launch planner (launch windows
into a RAAN or another satellite's plane, the pitch-kick search); the
simulator's step boundaries (liftoffs, ascent syncs every second, burns); and
its events (eclipses with the conical shadow, station AOS/LOS, close
approaches, re-entry at 80 km, escape). For numerical runs it uses `ode45` with
the scenario's tolerances, starting, as the simulator's integrator does, with a
10 s step and carrying the step size from one stretch of the run to the next.

`tests/ExportTest.m` exports several scenarios that between them use every
feature (`launch_day.json` from the start and mid-ascent among them), and
`tests/BundledExportTest.m` every scenario in `scenarios/`, from its start and
an hour in. Every script they export must pass MATLAB's Code Analyzer with
nothing to mark (`codeIssues`), and its passes and eclipses must match its
event log line for line, with the highest elevation of every pass of 10
minutes or more above the station's mask (the elevation the simulator
measures). Both require the same objects at the same samples, the same
statuses, positions within 10 m while in orbit, launch and maneuver events
identical to the millisecond, and every other event within one integrator step.
Three differences are expected. Events found at the end of a step (eclipses,
passes, re-entries) fall at the end of each integrator's own steps; ode45 steps
are not the simulator's. For the same reason a pass shorter than a step (a
satellite grazing a station's elevation mask for under a minute, a brief shadow)
can be in one log only: the two lines of such a pass are allowed to be missing
from the other log together. And an object coming down through the dense air (a sounding
rocket's payload, a spent stage) can differ by tens of metres: the atmosphere
table jumps slightly at its layer boundaries, and with drag near 100 m/s^2
where each integrator's steps straddle a jump matters. Such objects are held
to 50 m.

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
