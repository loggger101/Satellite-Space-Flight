"""Physical constants used throughout the simulation.

Units are kilometres, seconds and kilograms unless a name says otherwise.
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

# Unnormalised zonal harmonics (EGM-96)
J2 = 1.08262668355e-3
J3 = -2.53265648533e-6
J4 = -1.61962159137e-6

# --- Sun and Moon -------------------------------------------------------
MU_SUN = 1.32712440018e11         # km^3/s^2
MU_MOON = 4902.800066             # km^3/s^2
R_SUN = 696000.0                  # km
R_MOON = 1737.4                   # km
AU = 149597870.7                  # km
P_SRP = 4.56e-6                   # N/m^2, solar radiation pressure at 1 AU
SOI_EARTH = 924000.0              # km, Earth's sphere of influence w.r.t. the Sun

# --- Time ---------------------------------------------------------------
JD_J2000 = 2451545.0
SECONDS_PER_DAY = 86400.0
SIDEREAL_DAY = 2.0 * math.pi / OMEGA_EARTH   # ~86164.09 s

# --- Useful derived radii -------------------------------------------------
R_GEO = (MU_EARTH * (SIDEREAL_DAY / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)  # ~42164 km

DEG = math.pi / 180.0
RAD = 180.0 / math.pi
