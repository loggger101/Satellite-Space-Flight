"""Derived orbital and physical properties of a single satellite, gathered
for the orbit inspector: orbit shape, apsis speeds, timing of the next apsis
and node passes, the sunlight profile of the coming revolution, local time of
the ascending node, J2 drift and the craft's drag characteristics.

Everything is two-body plus J2 secular theory evaluated from the osculating
state, so the numbers describe the orbit *now*; the numerical propagator
remains the source of truth for where the satellite actually goes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis import (
    beta_angle,
    classify,
    coverage_half_angle,
    j2_secular_rates,
    osculating_to_mean_a,
)
from .atmosphere import density
from .constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from .eclipse import shadow_fraction
from .elements import Elements, coe2rv, mean_to_true, rv2coe, time_to_true_anomaly
from .ephemeris import sun_position
from .forces import approx_altitude
from .maneuvers import sun_synchronous_inclination

CIRCULAR_E = 1e-3        # below this, apsides are not meaningful for display
EQUATORIAL_SIN_I = 1e-3  # below this, nodes are not meaningful
TIMELINE_SAMPLES = 361


@dataclass
class OrbitInfo:
    """The orbit inspector's numbers for one satellite; built by :func:`orbit_info`."""

    el: Elements
    regime: str
    closed: bool
    circular: bool
    equatorial: bool
    retrograde: bool
    # state
    radius: float            # km
    altitude: float          # km above the ellipsoid
    speed: float             # km/s
    fpa: float               # flight-path angle, rad
    # shape
    b: float                 # semi-minor axis (km); nan if open
    c: float                 # center-to-focus distance a*e (km); nan if open
    a_mean: float            # J2 orbit-averaged semi-major axis (km); nan if open
    rp_alt: float            # km
    ra_alt: float            # km (inf if open)
    # speeds
    v_peri: float
    v_apo: float             # nan if open
    v_circ: float            # circular speed at the current radius
    v_esc: float             # escape speed at the current radius
    c3: float                # 2 * energy, km^2/s^2
    # timing
    period: float            # s (inf if open)
    nodal_period: float      # s, with J2 (nan if open or equatorial)
    revs_per_day: float
    ground_shift: float      # deg of longitude the ground track moves west per rev (-180..180)
    t_peri: float            # s until the next periapsis (inf if never / circular)
    t_apo: float
    t_an: float              # s until the next ascending node
    t_dn: float
    # perturbations and lighting
    raan_dot: float          # rad/s (J2 secular)
    argp_dot: float          # rad/s
    sso_inclination: float   # rad needed for a sun-synchronous orbit of this a, e (nan if none)
    ltan: float              # local solar time of the ascending node, hours (nan if n/a)
    beta: float              # rad
    timeline_t: np.ndarray   # s from now over one revolution
    timeline_lit: np.ndarray  # illuminated fraction at those times
    eclipse_fraction: float  # fraction of the revolution spent in shadow
    # coverage
    horizon_km: float        # straight-line distance to the horizon
    footprint_km: float      # ground radius seen above 10 deg elevation
    # spacecraft
    mass: float              # kg
    area: float              # m^2
    cd: float
    area_to_mass: float      # m^2/kg
    ballistic: float         # m / (Cd A), kg/m^2
    rho_perigee: float       # kg/m^3 at perigee altitude
    drag_perigee: float      # m/s^2 drag deceleration at perigee

    @property
    def eclipse_duration(self) -> float:
        """Seconds per revolution spent in shadow (nan if open)."""
        return self.eclipse_fraction * self.period if self.closed else float("nan")


def next_passes(r, v, el: Elements | None = None):
    """Seconds until the next periapsis, apoapsis, ascending and descending
    node (two-body). ``inf`` where the event is undefined or never reached."""
    el = el if el is not None else rv2coe(r, v)
    inf = float("inf")
    circular = el.e < CIRCULAR_E
    equatorial = math.sin(el.i) < EQUATORIAL_SIN_I
    t_peri = inf if circular else time_to_true_anomaly(r, v, 0.0)
    t_apo = inf if circular or el.e >= 1 else time_to_true_anomaly(r, v, math.pi)
    # the argument of latitude u = argp + nu is 0 at the ascending node
    if equatorial:
        t_an = t_dn = inf
    else:
        t_an = time_to_true_anomaly(r, v, (-el.argp) % (2 * math.pi))
        t_dn = time_to_true_anomaly(r, v, (math.pi - el.argp) % (2 * math.pi))
    return t_peri, t_apo, t_an, t_dn


def sunlight_timeline(el: Elements, r_sun, samples: int = TIMELINE_SAMPLES):
    """Illuminated fraction along the next revolution of a closed orbit,
    sampled uniformly in time from now. The Sun is held fixed (it moves
    about 1 deg per day)."""
    t = np.linspace(0.0, el.period, samples)
    nu = mean_to_true(el.M + el.n * t, el.e)
    pos, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, nu)
    return t, shadow_fraction(pos, r_sun)


def local_time_of_node(raan: float, r_sun) -> float:
    """Local (apparent) solar time, hours, at the ascending node."""
    ra_sun = math.atan2(r_sun[1], r_sun[0])
    return (12.0 + math.degrees(raan - ra_sun) / 15.0) % 24.0


def orbit_info(r, v, jd: float, sat=None, density_scale: float = 1.0) -> OrbitInfo:
    """Everything the orbit inspector shows for one ECI state at ``jd``.
    ``sat`` supplies the physical properties (mass, area, cd)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    el = rv2coe(r, v)
    nan, inf = float("nan"), float("inf")
    rm, vm = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    closed = bool(el.e < 1.0)
    circular = bool(el.e < CIRCULAR_E)
    equatorial = bool(math.sin(el.i) < EQUATORIAL_SIN_I)
    r_sun = sun_position(jd)

    if closed:
        b = el.a * math.sqrt(1.0 - el.e ** 2)
        c = el.a * el.e
        v_apo = el.h / el.ra
        rd, wd, md = (float(x) for x in j2_secular_rates(el.a, el.e, el.i))
        nodal_period = nan if equatorial else 2 * math.pi / (el.n + md + wd)
        rev = el.period if equatorial else nodal_period
        ground_shift = (math.degrees((OMEGA_EARTH - rd) * rev) + 180.0) % 360.0 - 180.0
        try:
            sso = sun_synchronous_inclination(el.a, el.e)
        except ValueError:
            sso = nan
        t_line, lit = sunlight_timeline(el, r_sun)
        eclipse = float(np.mean(1.0 - lit[:-1]))   # the last sample repeats the first
    else:
        b = c = v_apo = nodal_period = sso = nan
        rd = wd = 0.0
        ground_shift = nan
        t_line, lit = np.zeros(0), np.zeros(0)
        eclipse = nan
    t_peri, t_apo, t_an, t_dn = next_passes(r, v, el)

    mass = getattr(sat, "mass", nan)
    area = getattr(sat, "area", nan)
    cd = getattr(sat, "cd", nan)
    am = area / mass if mass else nan
    rp_alt = el.rp - R_EARTH
    v_peri = el.h / el.rp
    rho_p = float(density(rp_alt, density_scale))
    alt = float(approx_altitude(r))
    return OrbitInfo(
        el=el, regime=classify(el), closed=closed, circular=circular, equatorial=equatorial,
        retrograde=bool(el.i > math.pi / 2),
        radius=rm, altitude=alt, speed=vm,
        fpa=math.asin(float(np.clip(np.dot(r, v) / (rm * vm), -1, 1))),
        b=b, c=c, rp_alt=rp_alt, ra_alt=(el.ra - R_EARTH) if closed else inf,
        a_mean=float(osculating_to_mean_a(el.a, el.e, el.i, el.argp, el.nu)) if closed else nan,
        v_peri=v_peri, v_apo=v_apo,
        v_circ=math.sqrt(MU_EARTH / rm), v_esc=math.sqrt(2 * MU_EARTH / rm), c3=2 * el.energy,
        period=el.period if closed else inf, nodal_period=nodal_period,
        revs_per_day=86400.0 / el.period if closed else nan, ground_shift=ground_shift,
        t_peri=t_peri, t_apo=t_apo, t_an=t_an, t_dn=t_dn,
        raan_dot=rd, argp_dot=wd, sso_inclination=sso,
        ltan=nan if equatorial else local_time_of_node(el.raan, r_sun),
        beta=beta_angle(r, v, r_sun),
        timeline_t=t_line, timeline_lit=lit, eclipse_fraction=eclipse,
        horizon_km=math.sqrt(max(rm * rm - R_EARTH * R_EARTH, 0.0)),
        footprint_km=coverage_half_angle(alt, math.radians(10.0)) * R_EARTH,
        mass=mass, area=area, cd=cd, area_to_mass=am,
        ballistic=mass / (cd * area) if cd and area else nan,
        rho_perigee=rho_p,
        drag_perigee=0.5 * rho_p * (v_peri * 1e3) ** 2 * cd * am,
    )


__all__ = ["OrbitInfo", "orbit_info", "next_passes", "sunlight_timeline", "local_time_of_node"]
