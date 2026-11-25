"""Orbital lifetime under drag: roughly how long until a satellite re-enters.

The semi-major axis and eccentricity are decayed with Gauss's equations,
averaged over each revolution. The drag in them is the simulator's own
(:func:`forces.accel_drag`), so it uses the same density table, height above
the ellipsoid and turning air as the run.
- Revolutions are averaged by quadrature in eccentric anomaly.
- The perigee is averaged over every direction in the plane: J2 turns it
  through all of them many times in a lifetime (in LEO it takes weeks), so
  its latitude, and with it the perigee's height above the oblate Earth, is
  averaged out.
- J2 and the other gravity terms do not change a or e on average, so they
  are left out; the inclination is held fixed.
- The step is set so the perigee moves at most a tenth of the local scale
  height, and the apogee at most a few percent of its height above the
  perigee.
- The decay ends when the apogee is below ``END_KM``; from there the object
  comes down within about a day.

The answer is only as good as the atmosphere it uses. The table is for mean
solar activity, and real densities swing about x10 over the solar cycle
(atmosphere.py). Because the decay rates are proportional to the density,
the lifetime scales as exactly 1 / ``density_scale``.

With ``sun`` (NRLMSIS), the air is sampled at a date: the ring of points is
placed at the orbit's RAAN, which J2 turns (so the local solar time, and with
it the day/night bulge, is right; a sun-synchronous orbit keeps its local
time), with the Earth turned to that date, and NRLMSIS takes its solar and
geomagnetic indices from ``sun`` (the measured table, or a
:class:`satflight.solarforecast.SolarForecast` for the years ahead). Steps are
capped at ``MSIS_MAX_STEP`` so the solar cycle is followed. Near-circular
orbits use fewer points (``_msis_grid``): the air changes smoothly around them,
and below ``ROUND_E`` the perigee's direction no longer matters (a e under
about 7 km, a tenth of the scale height at 700 km). Past ``MSIS_MAX_YEARS``
the lifetime is reported as infinite:
repeated mean solar cycles say nothing that far ahead.
"""

from __future__ import annotations

import math

import numpy as np

from . import atmosphere
from .analysis import j2_secular_rates
from .constants import J2, MU_EARTH, R_EARTH
from .elements import coe2rv
from .forces import accel_drag
from .timeutil import gmst

N_E = 48            # eccentric-anomaly nodes per revolution (midpoint rule: spectral for periodic)
N_W = 8             # argument-of-perigee nodes
END_KM = 120.0      # stop when the apogee is below this (r_a - R_eq): down within about a day
MAX_YEARS = 1000.0  # past this the lifetime is reported as infinite
YEAR = 365.25 * 86400.0
MAX_STEPS = 20000   # a safety net; ordinary decays take a few hundred steps
N_E_CIRCULAR = 16   # NRLMSIS, e below CIRCULAR_E: eccentric-anomaly nodes
CIRCULAR_E = 0.01
ROUND_E = 0.001     # NRLMSIS, e below this: two argument-of-perigee nodes
MSIS_MAX_STEP = 30 * 86400.0   # s: follows the solar cycle and the monthly forecast
MSIS_MAX_YEARS = 200.0         # NRLMSIS: past this the lifetime is reported as infinite


def _grid(n_e: int, n_w: int = N_W):
    e, w = np.meshgrid((np.arange(n_e) + 0.5) * 2 * math.pi / n_e,
                       (np.arange(n_w) + 0.5) * 2 * math.pi / n_w)
    return e.ravel(), w.ravel()


_E, _W = _grid(N_E)
_GRIDS = {"table": (_E, _W), "eccentric": (_E, _W), "circular": _grid(N_E_CIRCULAR),
          "round": _grid(N_E_CIRCULAR, 2)}


def _msis_grid(sun, e: float) -> str:
    if sun is None:
        return "table"
    return "round" if e < ROUND_E else "circular" if e < CIRCULAR_E else "eccentric"

# rules for objects left in low Earth orbit after their mission
DEORBIT_RULES = (("FCC", 5.0), ("IADC", 25.0))   # years


def decay_rates(a: float, e: float, i: float, cd_a_over_m: float,
                density_scale: float = 1.0, j2: bool = True, *, sun=None,
                jd: float | None = None, raan: float = 0.0) -> tuple[float, float]:
    """Orbit-averaged da/dt (km/s) and de/dt (1/s) from drag, for a closed
    orbit with mean semi-major axis ``a`` (km), eccentricity ``e``,
    inclination ``i`` (rad) and Cd*A/m ``cd_a_over_m`` (m^2/kg).

    With ``j2``, the drag is sampled at the radius the satellite actually flies
    at, not on the mean ellipse. J2's first-order short-period terms for a
    near-circular orbit (Kozai 1959) move it by

        dr = k [-3/2 (1 - 3/2 sin^2 i) + 1/4 sin^2 i cos 2u],  k = J2 Re^2 / a,

    where u is the argument of latitude. That is several km in LEO: a polar
    orbit flies about 5 km above its mean semi-major axis, and highest over
    the poles. This is the circular theory: it is faded out once the
    eccentricity's own swing, a e, passes ~10 k (about 0.01 in LEO). Beyond
    that, the osculating e and a that the estimate starts from already put the
    perigee where it is flown, to a few km. Measured against runs to re-entry
    (tests/test_lifetime.py), circular orbits come out within a few percent and
    eccentric ones within about 10 %.

    With ``sun`` the air is NRLMSIS's at UTC Julian date ``jd``, with the
    orbit's node at ``raan`` (rad, inertial of date); see the module docstring."""
    if not (0.0 <= e < 1.0 and a > 0.0):
        raise ValueError(f"decay_rates needs a closed orbit; got a = {a!r} km, e = {e!r}")
    if sun is not None and jd is None:
        raise ValueError("the NRLMSIS lifetime needs the date (jd) to start from")
    ee, ww = _GRIDS[_msis_grid(sun, e)]
    nu = 2.0 * np.arctan2(math.sqrt(1.0 + e) * np.sin(ee / 2), math.sqrt(1.0 - e) * np.cos(ee / 2))
    p = a * (1.0 - e * e)
    r, v = coe2rv(a, e, i, raan if sun is not None else 0.0, ww, nu, p=p)
    rr = np.linalg.norm(r, axis=-1)
    rhat = r / rr[:, None]
    r_air = r                                # where the drag is sampled
    if j2:
        s2 = math.sin(i) ** 2
        k = J2 * R_EARTH ** 2 / a
        fade = 1.0 / (1.0 + (a * e / (10.0 * k)) ** 2)    # circular theory only
        dr = fade * k * (-1.5 * (1.0 - 1.5 * s2) + 0.25 * s2 * np.cos(2 * (ww + nu)))
        r_air = r + dr[:, None] * rhat
    cd = np.full(len(ee), float(cd_a_over_m))
    if sun is None:
        acc = accel_drag(r_air, v, cd, density_scale)
    else:
        acc = accel_drag(r_air, v, cd, density_scale, "msis", gmst=float(gmst(jd)), jd=jd,
                         table=sun)
    h = math.sqrt(MU_EARTH * p)
    shat = np.cross(np.cross(r, v) / h, rhat)
    R = np.sum(acc * rhat, axis=-1)          # radial
    S = np.sum(acc * shat, axis=-1)          # along the motion, in the plane
    sn, cn = np.sin(nu), np.cos(nu)
    dadt = 2.0 * a * a / h * (e * sn * R + p / rr * S)
    dedt = (p * sn * R + ((p + rr) * cn + rr * e) * S) / h
    w = rr / a                               # dM = (r / a) dE: a time average
    return float(np.sum(w * dadt) / np.sum(w)), float(np.sum(w * dedt) / np.sum(w))


def _scale_height(alt_km: float) -> float:
    k = int(np.clip(np.searchsorted(atmosphere.H0, alt_km, side="right") - 1, 0,
                    len(atmosphere.H0) - 1))
    return float(atmosphere.SCALE_H[k])


def lifetime(a: float, e: float, i: float, cd_a_over_m: float,
             density_scale: float = 1.0, j2: bool = True, *, sun=None,
             jd: float | None = None, raan: float = 0.0) -> float:
    """Seconds until the orbit decays into the atmosphere (see the module
    docstring). ``a``, ``e`` and ``i`` should be mean elements; ``j2`` says
    whether the run has J2 (see :func:`decay_rates`). Returns 0.0
    if the apogee is already below ``END_KM``, and ``inf`` if there is no
    air at the perigee or the decay takes longer than ``MAX_YEARS``
    (``MSIS_MAX_YEARS`` under NRLMSIS).

    ``sun``, ``jd`` (UTC Julian date now) and ``raan`` (rad): the NRLMSIS
    estimate (module docstring); without ``sun``, the table's."""
    if not (cd_a_over_m >= 0.0 and density_scale >= 0.0):
        raise ValueError(f"lifetime needs Cd*A/m >= 0 and a density scale >= 0; got "
                         f"{cd_a_over_m!r} m^2/kg and {density_scale!r}")
    if sun is not None and jd is None:
        raise ValueError("the NRLMSIS lifetime needs the date (jd) to start from")
    horizon = (MSIS_MAX_YEARS if sun is not None else MAX_YEARS) * YEAR

    def node_rate(a_, e_):
        return float(j2_secular_rates(a_, e_, i)[0]) if (sun is not None and j2) else 0.0

    def rates(a_, e_, t_, raan_):
        if sun is None:
            return decay_rates(a_, e_, i, cd_a_over_m, density_scale, j2)
        return decay_rates(a_, e_, i, cd_a_over_m, density_scale, j2, sun=sun,
                           jd=jd + t_ / 86400.0, raan=raan_)

    t = 0.0
    for _ in range(MAX_STEPS):
        rp, ra = a * (1.0 - e) - R_EARTH, a * (1.0 + e) - R_EARTH
        if ra < END_KM:
            return t
        if t > horizon:
            return math.inf
        da, de = rates(a, e, t, raan)
        drp, dra = da * (1 - e) - a * de, da * (1 + e) + a * de
        if drp >= 0.0 and dra >= 0.0:
            return math.inf                  # no air at this height: it never comes down
        hp = _scale_height(rp)
        dt = min(0.1 * hp / abs(drp) if drp else math.inf,
                 max(0.1 * hp, 0.03 * (ra - rp)) / abs(dra) if dra else math.inf)
        if sun is not None:
            dt = min(dt, MSIS_MAX_STEP)
        # midpoint step
        am, em = a + 0.5 * dt * da, max(e + 0.5 * dt * de, 0.0)
        wdot = node_rate(a, e)
        da, de = rates(am, em, t + 0.5 * dt, raan + 0.5 * dt * wdot)
        raan += dt * node_rate(am, em)
        a, e, t = a + dt * da, min(max(e + dt * de, 0.0), 0.999), t + dt
    raise RuntimeError(f"lifetime: no answer after {MAX_STEPS} steps "
                       f"(a = {a:.1f} km, e = {e:.5f}, t = {t / YEAR:.2f} yr)")


def rule_check(seconds: float) -> str:
    """Which deorbit rules (``DEORBIT_RULES``) a lifetime meets, briefly:
    ``"meets FCC 5 yr, IADC 25 yr"``, ``"meets IADC 25 yr only"`` or
    ``"meets neither rule"``."""
    met = [f"{name} {lim:g} yr" for name, lim in DEORBIT_RULES if seconds <= lim * YEAR]
    if not met:
        return "meets neither rule"
    return "meets " + ", ".join(met) + ("" if len(met) == len(DEORBIT_RULES) else " only")


__all__ = ["decay_rates", "lifetime", "rule_check", "DEORBIT_RULES", "END_KM", "MAX_YEARS",
           "MSIS_MAX_YEARS"]
