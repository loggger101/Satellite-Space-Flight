"""Plain-language explanations shown as hover tips (see :mod:`.tips`): every
row label of the right panel's property sheets, their section titles, the
labels in the orbit diagrams and the kinds of event in the log."""

from __future__ import annotations

OMEGA, omega, nu = "\N{GREEK CAPITAL LETTER OMEGA}", "\N{GREEK SMALL LETTER OMEGA}", \
    "\N{GREEK SMALL LETTER NU}"
GAMMA = "\N{GREEK SMALL LETTER GAMMA}"

SECTIONS = {
    "STATE": "Where the satellite is and how fast it moves, right now.",
    "ORBIT": "The osculating orbit: the ellipse the satellite would follow from here if only "
             "the Earth's central gravity acted.",
    "ENVIRONMENT": "Sunlight and the size of each force acting on the satellite.",
    "LAUNCH": "The rocket that carried this payload, and how its ascent went.",
    "SPACECRAFT": "The spacecraft's own properties, and any maneuvers planned for it.",
    "GROUND CONTACT": "Ground stations that can see the satellite now: azimuth (compass "
                      "bearing), elevation above the horizon and slant range.",
    "SHAPE": "The size and shape of the orbit's ellipse.",
    "ORIENTATION": "How the orbit is tilted and turned in space, and where on it the "
                   "satellite is.",
    "TIMING": "How long a revolution takes and when the next special points come up.",
    "SPEED AND ENERGY": "Speeds along the orbit and the energy of the orbit.",
    "PERTURBATIONS AND LIGHT": "Slow changes caused by the Earth's bulge (J2), and how much "
                               "of each orbit is spent in the Earth's shadow.",
    "ASCENT": "Live readings from the rocket while it climbs.",
    "STAGES": "Propellant left in each stage. The highlighted one is burning or next to burn.",
}

ROWS = {
    # --- state
    "Altitude": "Height above the Earth's surface (the WGS-84 ellipsoid).",
    "Latitude / Longitude": "The point on the ground directly below the satellite. Positive "
                            "latitude is north, positive longitude east.",
    "Radius": "Distance from the center of the Earth.",
    "Speed (inertial)": "Speed relative to the stars (the non-rotating ECI frame). This is "
                        "the speed orbital mechanics works with.",
    "Speed (ground-rel.)": "Speed relative to the rotating Earth's surface, the way someone "
                           "on the ground would measure it.",
    "Flight-path angle": "Angle of the velocity above the local horizontal. Positive while "
                         "climbing toward apogee, negative while descending, zero on a "
                         "circular orbit.",
    # --- shape
    "Regime": "Which family of orbit this is: low Earth orbit, geostationary, highly "
              "elliptical and so on.",
    "Semi-major axis a": "Half the longest diameter of the ellipse the satellite is on at "
                         "this instant. It sets the orbit's energy; the Earth's bulge makes "
                         "it swing by several km around each revolution.",
    "Mean semi-major axis": "The semi-major axis averaged over a revolution, without the "
                            "swing the Earth's bulge (J2) gives it. This is what sets the "
                            "period: satellites meant to fly in formation share it.",
    "Semi-minor axis b": "Half the shortest diameter of the ellipse.",
    "Eccentricity e": "How stretched the orbit is: 0 is a circle, close to 1 a long thin "
                      "ellipse, 1 or more an open escape path.",
    "Semi-latus rectum p": "The orbit's radius when the satellite is 90 degrees from "
                           "perigee. It sets the shape through r = p / (1 + e cos(nu)).",
    "Perigee alt / radius": "Lowest point of the orbit: its height above the surface, then "
                            "its distance from the Earth's center.",
    "Apogee alt / radius": "Highest point of the orbit: its height above the surface, then "
                           "its distance from the Earth's center.",
    "Focus offset c = ae": "How far the ellipse's center sits from the Earth's center (the "
                           "Earth is at one focus, not in the middle).",
    "Perigee / apogee alt": "Height of the orbit's lowest and highest points above the "
                            "surface.",
    "Perigee alt": "Height of the lowest point of the trajectory above the surface.",
    # --- orientation
    "Inclination i": "Tilt of the orbit against the equator: 0 is equatorial, 90 polar, "
                     "more than 90 retrograde (against the Earth's spin).",
    "RAAN": "Right ascension of the ascending node: the direction, measured east from the "
            "vernal equinox, where the orbit crosses the equator going north.",
    "Arg. of perigee": "Angle from the ascending node to the perigee, measured along the "
                       "orbit in the direction of motion.",
    "True anomaly": "Angle from the perigee to the satellite, seen from the Earth's center.",
    "Mean anomaly M": "The fraction of the period since perigee, as an angle. It grows at "
                      "a steady rate, unlike the true anomaly.",
    "Arg. of latitude": "Angle from the ascending node to the satellite along the orbit. "
                        "It is used for circular orbits, which have no perigee.",
    "Arg. of latitude u": "Angle from the ascending node to the satellite along the orbit. "
                          "It is used for circular orbits, which have no perigee.",
    "Local time of AN":"Local solar time on the ground under the ascending node. A "
                        "Sun-synchronous orbit keeps it the same all year.",
    # --- timing
    "Period": "Time for one full revolution.",
    "Nodal period (J2)": "Time from one ascending node to the next, including the Earth's "
                         "bulge. It is the period the ground track repeats with.",
    "Revolutions per day": "How many orbits the satellite makes in 24 hours.",
    "Revs per day": "How many orbits the satellite makes in 24 hours.",
    "Track shift per rev": "How far west (or east) the ground track moves each revolution, "
                           "as the Earth turns beneath the orbit.",
    "Next perigee in": "Time until the satellite next passes its lowest point.",
    "Next apogee in": "Time until the satellite next passes its highest point.",
    "Next AN / DN in": "Time until the satellite next crosses the equator going north "
                       "(ascending node) and going south (descending node).",
    # --- speed and energy
    "Speed now": "Current speed relative to the stars (inertial).",
    "Speed at perigee": "Fastest speed on this orbit, at its lowest point.",
    "Speed at apogee": "Slowest speed on this orbit, at its highest point.",
    "Circular speed here": "Speed that would keep a circular orbit at the current radius.",
    "Escape speed here": "Speed that would leave the Earth for good from the current "
                         "radius (the square root of 2 times the circular speed).",
    "C3 = 2 x energy": "Characteristic energy: negative on a closed orbit, zero on the "
                       "edge of escape, positive when escaping. Mission planners quote it "
                       "for departures.",
    "Angular momentum h": "Radius times the sideways speed. It stays constant under central "
                          "gravity alone, and points along the orbit normal.",
    "Angular momentum": "Radius times the sideways speed. It stays constant under central "
                        "gravity alone.",
    "Specific energy": "Kinetic plus potential energy per kg: v^2/2 - mu/r. Negative on "
                       "closed orbits, and fixed by the semi-major axis.",
    "Hyperbolic excess v": "Speed left over far from the Earth on an escape trajectory.",
    # --- perturbations and light
    "J2 node drift": "How fast the Earth's equatorial bulge turns the orbital plane "
                     "around the pole (RAAN change per day).",
    "J2 perigee drift": "How fast the bulge turns the ellipse within its plane (argument "
                        "of perigee change per day). It stops at the critical inclination "
                        "of 63.4 degrees.",
    "J2 dRAAN / dargp": "Drift per day of the orbit's node (RAAN) and of its perigee, "
                        "caused by the Earth's equatorial bulge.",
    "Sun-sync inclination": "The inclination at which the J2 node drift would match the "
                            "Earth's yearly trip around the Sun (0.9856 degrees a day) for "
                            "this a and e. The orbit then keeps the same angle to the Sun.",
    "Beta angle": "Angle between the orbital plane and the direction of the Sun. Near 0 the "
                  "orbit passes through the Earth's shadow; high values mean little or no "
                  "eclipse.",
    "Time in shadow / rev": "Share of each revolution spent in the Earth's shadow.",
    "Illumination": "Whether the satellite is in sunlight, in the partial shadow "
                    "(penumbra) or in full shadow (umbra), and how much of the Sun it sees.",
    "Energy drift": "Relative change of the conserved orbital energy since the reference "
                    "time. It shows the integrator's error when no drag or burns act (it "
                    "should stay tiny).",
    # --- spacecraft
    "Mass": "Current mass of the spacecraft (it falls as burns use propellant).",
    "Area / Cd": "Cross-section facing the airflow, and the drag coefficient.",
    "Area-to-mass": "Cross-section divided by mass: the higher it is, the harder air drag "
                    "slows the spacecraft.",
    "Ballistic coeff m/CdA": "Mass over drag area. The higher it is, the longer the "
                             "spacecraft resists drag.",
    "Air density at perigee": "Density of the thin upper atmosphere at the orbit's lowest "
                              "point, from the exponential model with the chosen solar "
                              "activity scale.",
    "Drag at perigee": "Deceleration from air drag at perigee.",
    "Delta-v spent": "Total change of velocity from all burns so far: the usual measure of "
                     "propellant used.",
    "dV spent": "Total change of velocity from all burns so far.",
    "Horizon distance": "Distance to the horizon as seen from the satellite.",
    "Coverage radius": "Ground distance from the point below the satellite to where it "
                       "stands 10 degrees above the horizon (the footprint drawn on the "
                       "map).",
    "Cd*A/m": "Drag coefficient times area over mass, the number the drag force uses.",
    "BURNING": "A finite burn is firing right now: thrust and time left.",
    "in": "A maneuver scheduled for this satellite, and the time until it executes.",
    # --- forces
    "a_gravity": "Acceleration from the Earth's central (point-mass) gravity.",
    "a_J2": "Acceleration from the Earth's equatorial bulge (the J2 term), the largest "
            "perturbation for most orbits.",
    "a_J3": "Acceleration from the Earth's slight pear shape (J3).",
    "a_J4": "Acceleration from the J4 zonal term of the Earth's gravity field.",
    "a_C22": "Acceleration from the Earth's slightly elliptical equator (the C22/S22 "
             "term). It turns with the Earth and makes geostationary satellites drift "
             "toward 75 E or 105 W.",
    "a_drag": "Deceleration from air drag. '(off)' means the force is shown but not applied "
              "(turn it on in Physics).",
    # --- launch
    "Phase": "What the rocket is doing now: on the pad, vertical rise, pitch kick, gravity "
             "turn or closed-loop guidance.",
    "Mission time": "Time since liftoff (T-minus before it).",
    "Stage": "The stage now in use, and whether its engine is burning.",
    "Propellant in stage": "Propellant left in the current stage.",
    "Vehicle": "The launch vehicle that flew this payload.",
    "Site": "Where the rocket lifted off.",
    "Outcome": "How the ascent ended: in orbit, sub-orbital, escaped, or failed.",
    "Cutoff": "Mission time when the last engine shut down.",
    "Insertion orbit": "Perigee x apogee altitude and inclination at engine cutoff.",
    "Propellant left": "Propellant still in the last stage at cutoff: the margin.",
    "Max Q": "Highest aerodynamic pressure on the vehicle during the climb, and when.",
    "Ascent dV / losses": "Delta-v the engines delivered, and how much of it gravity, air "
                          "drag and steering took.",
    "Downrange": "Ground distance from the launch pad.",
    "Speed inertial / air": "Speed relative to the stars, and relative to the air (which "
                            "turns with the Earth).",
    "Vertical speed": "Rate of climb.",
    "Dynamic pressure": "Pressure of the oncoming air, 0.5 rho v^2. It peaks at 'max Q' a "
                        "minute or so after liftoff.",
    "Acceleration / throttle": "Thrust acceleration in g, and the engine throttle (it drops "
                               "when the vehicle's acceleration limit is reached).",
    "Pitch above horizon": "Angle of the thrust above the local horizontal.",
    "Vehicle mass": "Mass of the rocket and payload now.",
    "Time to cutoff": "Guidance's estimate of the time left until the engines shut down "
                       "in the target orbit.",
    "Orbit now": "The orbit the vehicle would coast on if the engines stopped now.",
    "Target": "The orbit the guidance is aiming for.",
    "Launch azimuth / kick": "Compass heading of the ascent, and the pitch-over angle that "
                             "starts the gravity turn.",
    "dV spent (ideal)": "Delta-v the engines have delivered so far (the rocket equation).",
    "gravity / drag loss": "Delta-v lost to gravity pulling back while climbing, and to "
                           "air drag.",
    "steering loss": "Delta-v lost because the thrust does not point along the velocity.",
    "Liftoff in": "Time left on the countdown.",
    "Liftoff at": "Date and time of liftoff (UTC).",
    "Liftoff mass": "Mass of the fueled rocket and payload on the pad.",
    "Planned outcome": "What the preview flight predicted for this ascent.",
}

# the labels drawn inside the orbit diagrams (by their first word)
LABELS = {
    "Pe": "Perigee: the orbit's lowest point, where the satellite moves fastest.",
    "Ap": "Apogee: the orbit's highest point, where the satellite moves slowest.",
    "AN": "Ascending node: where the orbit crosses the equator going north.",
    "DN": "Descending node: where the orbit crosses the equator going south.",
    "Sun": "Direction of the Sun.",
    "h": "Angular momentum vector: perpendicular to the orbit (the orbit normal).",
    "N": "The Earth's north pole (spin axis).",
    "equator": "The equatorial plane, seen edge-on.",
    GAMMA: "Vernal equinox: the fixed direction RAAN is measured from.",
    OMEGA: "RAAN: the angle from the vernal equinox to the ascending node. LTAN: local "
           "solar time at the ascending node.",
    "prograde": "The satellite orbits the same way the Earth spins (inclination below "
                "90 degrees).",
    "retrograde": "The satellite orbits against the Earth's spin (inclination above "
                  "90 degrees).",
    "polar": "The orbit passes over both poles (inclination near 90 degrees).",
    "equatorial": "The orbit lies in the equatorial plane.",
}

EVENTS = {
    "info": "General news from the simulation.",
    "maneuver": "A maneuver was scheduled, started or finished.",
    "eclipse": "A satellite entered or left the Earth's shadow.",
    "station": "A ground station gained or lost sight of a satellite (AOS / LOS).",
    "alert": "Something went wrong or needs attention: a close approach, a re-entry, an "
             "impact.",
    "warn": "Something the simulator skipped or ignored, such as a maneuver it could not "
            "run.",
    "launch": "A launch event: liftoff, staging, fairing jettison, orbit insertion.",
}


def row(label: str) -> str:
    """The explanation of a property-sheet row ``label`` ('' if there is none)."""
    key = label.strip()
    if key.endswith(" (off)"):
        key = key[:-6]
    if key in ROWS:
        return ROWS[key]
    for sym in (f" {OMEGA}", f" {omega}", f" {nu}", " (10\N{DEGREE SIGN})"):
        if key.endswith(sym) and key[: -len(sym)] in ROWS:     # "RAAN Ω", "True anomaly ν"
            return ROWS[key[: -len(sym)]]
    if key.startswith("Energy drift"):
        return ROWS["Energy drift"]
    first = key.split(" ")[0]
    return ROWS.get(first, "")


def section(title: str) -> str:
    """The explanation of a section title such as 'ORBIT  (low Earth orbit)'."""
    return SECTIONS.get(title.split("  (")[0].strip(), "")


def label(text: str) -> str:
    """The explanation of a diagram label, looked up by its first word."""
    return LABELS.get(text.split(" ")[0], "") if text else ""
