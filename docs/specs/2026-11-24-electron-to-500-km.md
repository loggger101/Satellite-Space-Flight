# Electron to 500 km SSO - finding (no defect)

**Date:** 2026-11-24. Found while probing ascent air (docs/specs/
2026-11-24-msis-launch-design.md): the Electron preset with 200 kg failed to
reach a 500 km circular sun-synchronous orbit by direct insertion, under
either density. That is not a defect: it is not how Electron flies, and the
simulator flies the real profile.

## What the probe shows

`electron_probe.py` (session scratchpad), the largest payload `plan_launch`
puts into each orbit (bisection on the plan succeeding):

| target (Mahia LC-1 unless noted) | largest payload |
|---|---|
| 500 km circular SSO, direct insertion | ~30 kg |
| 500 km circular SSO, from Vandenberg | ~30 kg |
| 200 x 500 km SSO (stage 2 only) | ~210 kg |
| 300 km circular SSO | ~157 kg |
| 500 km circular, 45 deg | ~85 kg |

The 200 kg direct attempt runs out of propellant 396 s in: a low-thrust upper
stage (about 1 g at ignition) asked to reach 500 km circular in one burn pays
2.03 km/s in steering and 2.07 km/s to gravity, against 9.59 km/s ideal.

## Why that is right

The preset's stages match the public figures (224 kN, 311/282 s and 9250 kg
on stage 1; 25.8 kN, 343 s and 2050 kg on stage 2). Rocket Lab's ~200 kg to
500 km SSO is for its real profile: stage 2 to an ellipse, then the Curie kick
stage circularises at apogee. The simulator flies that with a launch's
`circularize` option (the payload circularizes at its first apogee), as
`scenarios/launch_day.json`'s "Rocket Lab sat" does (200 x 500 km, 97.4 deg,
`circularize: true`). The preset carries about 210 kg to that ellipse, which
matches.

## Left as is

The circularization is charged to the payload as an impulsive burn, without
a Curie stage's own mass; about 0.09 km/s at 500 km. A kick stage as a real
stage, with a coast and a relight, is audit B27's "parking orbit, coast and
restart".
