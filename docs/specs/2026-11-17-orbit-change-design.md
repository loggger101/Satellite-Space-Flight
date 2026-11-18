# Changing a satellite's orbit by user input - design

Approved 2026-11-17. One of the project's main points is to model an established
satellite transfer into a different orbit. Plan maneuver (key B) already flies
Hohmann, bi-elliptic, circularize, plane change, rendezvous and finite burns; this
work adds a goal-orbit planner, direct 3-D editing of burns, a 3-D preview of
every plan, clearer explanations, and a GTO-to-GEO showcase.

## 1. 3-D transfer preview

- While the Plan maneuver dialog (or a burn node, part 3) is open, the 3-D view
  draws the planned path: the transfer arc dashed, each burn point marked with its
  delta-v, the final orbit faded.
- Path = the satellite coasted to each burn, the burn applied, coasted on; ends
  one period of the final orbit after the last burn.
- While typing: two-body (Kepler) path from the planner's estimate. Once the form
  stops changing, the aimed plan (run's forces) is computed on a
  `background.Latest` worker and replaces it; the frame never waits for it.
- Elements are named and have hover tips, like the other 3-D overlays.

## 2. "Change orbit to..." (target-orbit planner)

- New maneuver kind. Inputs: target perigee altitude, apogee altitude,
  inclination (deg), and where the first burn fires (the usual timing choices).
- Two burns. Burn 1 (at the chosen point) leaves the satellite level at the speed
  whose far apsis is the target apsis on that side (the far side gets the other
  target apsis), and tilts the plane by a share `f` of the inclination change.
  Burn 2 (at that far apsis) sets the speed for the other target apsis and does
  the remaining `(1 - f)` of the tilt.
- `f` minimises total two-body delta-v (bounded scalar search, golden section);
  the aimed plan flies the run's forces with the existing `_transfer` aiming and
  computes burn 2 from the predicted state at the far apsis.
- Which target apsis goes where: the far apsis of the transfer goes to the target
  apsis that is not at the burn-1 radius; if burn 1 is not at an apsis it levels
  the flight path (as Hohmann does) and says what that costs.
- Out of scope for now: choosing RAAN / argument of perigee (needs multi-orbit
  phasing; J2 moves both anyway).
- Burns are scheduled as ordinary `impulse` maneuvers (VNB), so the MATLAB engine
  needs no new kind and the export tests keep holding.

## 3. Drag burns in 3-D

- With a satellite selected, a burn-node tool (key / button) and a click on its
  orbit drop a node at that point (time = when the satellite gets there).
- Six handles: prograde / retrograde, normal / anti-normal, radial out / in.
  Dragging a handle along its axis changes that delta-v component; the resulting
  orbit redraws live through part 1's preview. The node can slide along the orbit.
- A small readout shows the components, total delta-v and resulting
  perigee / apogee / inclination.
- Enter schedules it (absolute-time `impulse`, VNB); Esc discards.

## 4. Clearer explanations

- The maneuver dialog shows a table: each burn's time, place (periapsis, node...),
  delta-v split into speed and tilt, coast between; then orbit before -> after
  (perigee, apogee, inclination, period), total delta-v, propellant at the Isp.
- Each planner adds a one-line reason (e.g. why the tilt is done at apogee).
- Glossary / tips / docs updated (README, docs/PHYSICS.md).

## 5. GTO-to-GEO showcase

- Built in `tools/make_scenarios.py`: a satellite in a 185 x 35,786 km, 28.5 deg
  transfer orbit (a Cape Canaveral launch), with the target-orbit planner's burns
  to GEO scheduled (a small share of the tilt at perigee, most at apogee). README
  and welcome-card blurb explain it.

## Testing

- Planner: two-body numbers match the textbook GTO -> GEO split (~1.8 km/s total,
  ~2 deg at perigee); with J2 the final perigee / apogee / inclination land within
  tolerance; impossible targets raise ValueError with a reason.
- Dialog: every option driven through submit; every field has a tip.
- Burn nodes: synthetic mouse events place, drag, slide, schedule, discard.
- Screenshot checks; layout tests also with `SATFLIGHT_FONTS=bundled`; MATLAB
  export suite.
