# Thermospheric winds - design probe

**Date:** 2026-11-24. **Branch:** `winds` (off `physical-cd`).
**Roadmap item 3** of the drag work. The user's rule: try HWM14; if it does
not install or does not pass, fall back to super-rotation and say so loudly.

## 1. HWM14 does not install here

**HWM14 is not available on this machine.**

- The only PyPI package is `pyhwm2014` 1.1. It has no binary wheel for any
  platform pip offers here (`pip download --only-binary=:all:` finds none).
- Its source build fails before compiling, at "getting requirements to build
  wheel": it builds with `numpy.distutils`, which numpy removed (NumPy 2 and
  Python 3.12+).
- There is no Fortran or C compiler on this machine (no gfortran, ifx, flang,
  gcc or cl), so neither the package nor the original Fortran could be built.
- No other HWM14 package exists on PyPI (`hwm14`, `pyhwm14`). `hwm93` 0.9.1
  is the 1993 model, also Fortran.

So the fallback applies: super-rotation.

## 2. Probe: what winds could change for Sentinel-1

`wind_probe.py` (scratchpad): the circular Sentinel-1 orbit (693 km,
i = 98.18 deg), along-track drag `|v_rel| v_rel` with
`v_rel = v - Lambda (omega x r) - wind`, against today's co-rotating air
(speed through the air 7590 m/s, inertial 7508 m/s).

| change | along-track drag, orbit mean | range around the orbit |
|---|---|---|
| air rotating 1.1 x the Earth | +0.22% | +0.19 .. +0.24% |
| air rotating 1.2 x | +0.44% | +0.39 .. +0.49% |
| air rotating 1.3 x | +0.66% | +0.58 .. +0.74% |
| 100 m/s equatorward wind, both hemispheres | +0.02% | -2.6 .. +2.7% |
| 200 m/s equatorward wind, both hemispheres | +0.07% | -5.3 .. +5.4% |
| 50 / 100 / 200 m/s head wind everywhere | +1.3 / +2.7 / +5.4% | constant |

- **Super-rotation is a constant.** The orbit is nearly polar (cos i = -0.14),
  so extra rotation of the air adds 0.2-0.7% to the drag, the same all round
  the orbit. The arc fits absorb that into the density scale, so it cannot
  change the arcs. It also tilts the drag slightly out of the orbit plane, a
  slow inclination drift; over three days that is negligible next to the
  along-track error.
- **Meridional winds cancel.** A pattern that is the same in both hemispheres
  (the storm-time equatorward surge, or the quiet-time poleward flow at
  dawn and dusk) is a head wind on one half of each pass and a tail wind on the
  other: under 0.1% net per orbit.
- **Only a net head wind over the whole orbit that changes after the fit day
  would show.** To move the storm arcs by what the density does (the fits need
  scales from 0.54 to 1.36), it would have to change by several hundred m/s
  everywhere, which winds do not do.

## 3. Conclusion

Winds are at least an order of magnitude below the density error for the
Sentinel-1 arcs. Super-rotation as a setting is cheap (a `ForceModel` factor
on `omega` in `accel_drag`), but by section 2 it would fly the six arcs
unchanged. Building it and spending a night of arc runs to measure that is
the user's call (options below).

## 4. Options for the user

- **A. Skip winds; go to roadmap item 4** (MSIS for the lifetime estimate and
  the Orbit tab), checked against reference values.
- **B. Build super-rotation anyway** as an opt-in (`air_rotation`, default
  1.0), with tests, documented as having no measurable effect on the arcs,
  without re-flying them (the probe is the evidence).
- **C. Port HWM14 to Python** (the Fortran plus its coefficient files, the
  quasi-dipole conversion and the DWM07 storm part). A large job that needs
  published reference outputs to check against, for an effect section 2
  bounds as small.
