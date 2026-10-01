"""Physical constants used throughout the simulation.

Units are kilometers, seconds and kilograms unless a name says otherwise.
Earth values follow WGS-84 for shape and EGM-96 / EGM-2008 for gravity, the
same families used by Vallado's *Fundamentals of Astrodynamics* and by
astropy / skyfield.
"""

import math

# --- Earth --------------------------------------------------------------
MU_EARTH = 398600.4418            # km^3/s^2  (EGM-96 / WGS-84 GM)
R_EARTH = 6378.137                # km        equatorial radius (WGS-84)
F_EARTH = 1.0 / 298.257223563     # flattening (WGS-84)
E2_EARTH = F_EARTH * (2.0 - F_EARTH)   # first eccentricity squared
R_EARTH_POLAR = R_EARTH * (1.0 - F_EARTH)
OMEGA_EARTH = 7.292115146706979e-5     # rad/s, sidereal rotation rate

# Unnormalized zonal harmonics (EGM-96)
J2 = 1.08262668355e-3
J3 = -2.53265648533e-6
J4 = -1.61962159137e-6
# sectoral degree-2 order-2 coefficients (EGM-96, unnormalized): the equator's ellipticity
C22 = 1.57446037456e-6
S22 = -9.03803806639e-7

# --- Sun (position only: lighting, eclipses, beta angle) -------------------
# The Sun and Moon exert no force in this simulator; see docs/SCOPE.md.
R_SUN = 696000.0                  # km
AU = 149597870.7                  # km
SOI_EARTH = 924000.0              # km, Earth's sphere of influence w.r.t. the Sun

# --- Time ---------------------------------------------------------------
JD_J2000 = 2451545.0
SECONDS_PER_DAY = 86400.0
SIDEREAL_DAY = 2.0 * math.pi / OMEGA_EARTH   # ~86164.09 s

# --- Useful derived radii -------------------------------------------------
R_GEO = (MU_EARTH * (SIDEREAL_DAY / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)  # ~42164 km

DEG = math.pi / 180.0
RAD = 180.0 / math.pi
