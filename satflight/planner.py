"""Mission planners that turn a goal ("go to 35786 km", "meet satellite B in
3 h") into concrete scheduled maneuvers for a running simulation.

Burn points and the target's future position are predicted with the run's
own propagator and forces (``Simulation.coast_states``), so a run with J2,
drag or C22 is planned with them. The rendezvous intercept starts from a
two-body Lambert arc and is then corrected by shooting through that same
force model until it arrives on the target. The final ``circularize`` and
``match_velocity`` burns are computed from the actual state when they
execute.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .elements import kepler_propagate
from .frames import unit, vnb_basis
from .maneuvers import Maneuver, lambert, plane_change_dv, resolve_time
from .scenario import MAX_ALT
from .simulation import _hermite_at
from .strict import within

SAFE_PERIGEE_ALT = 150.0   # km; lower transfer arcs would re-enter
AIM_TOL = 1e-3             # km; a rendezvous arc must arrive within 1 m of the target
AIM_ITER = 12              # differential-correction passes before giving up
AIM_STEP = 1e-5            # km/s; finite-difference step of the correction (1 cm/s)
HALF_TURN_DEG = 2.0       # deg: a Lambert arc this close to 180 deg gets a warning
LEVEL_SIN_FPA = 1e-3      # flatter than this, a transfer burn is purely along V
APSIS_TOL = 0.01          # km; a transfer's far apsis must land this close
ESTIMATE = "two-body estimate, aimed on Schedule: "   # how an unaimed preview reads
APSIS_STEP = 1e-4         # km/s; finite-difference step when aiming it
PLAN_HORIZON = 7 * 86400.0   # s; planners fly the run's forces this far ahead at most
#                              (a LEO day is ~0.6 s of coasting; a week keeps it seconds)


def _within_horizon(span: float, what: str = "the burn"):
    """ValueError when ``span`` seconds is past ``PLAN_HORIZON``: planners fly
    the force model up to the burn, and a coast of months would freeze the app."""
    if span > PLAN_HORIZON:
        raise ValueError(f"{what} is {span / 86400:,.1f} days ahead; planners look at most "
                         f"{PLAN_HORIZON / 86400:.0f} days ahead (they fly the run's forces "
                         "there), so plan it nearer the time")


def _burn_state(sim, sat: str, timing: str, delay: float = 0.0, aim: bool = True,
                horizon: bool = True):
    """Time of the first burn and the state ``sat`` will have then, coasting
    under the run's force model (two-body, for a quick estimate, without ``aim``).
    A burn past ``PLAN_HORIZON`` is refused unless ``horizon`` is False (for a
    two-body estimate that flies nothing)."""
    i = sim.require(sat)
    m = Maneuver(sat, timing=timing, delay=delay)
    m.check()                            # a bad delay is refused before anything is flown
    if horizon and timing == "delay":
        _within_horizon(delay)
    if not aim:
        t_b = resolve_time(m, sim.t, sim.y[i, :3], sim.y[i, 3:])
        r, v = kepler_propagate(sim.y[i, :3], sim.y[i, 3:], t_b - sim.t)
        return t_b, np.asarray(r), np.asarray(v)
    t_b = sim.resolve_time(m)
    yb = sim.predict(sat, t_b) if t_b > sim.t else sim.y[i].copy()
    return t_b, yb[:3].copy(), yb[3:].copy()


def _level(r, v) -> np.ndarray:
    """Unit vector level with the ground, in the orbit plane, along the motion."""
    return unit(np.cross(np.cross(r, v), r))


def _apsis_burn(r, v, speed: float):
    """VNB delta-v that leaves (r, v) moving level at ``speed``, which makes the
    burn point an apsis of the new orbit. Returns (dv_vnb, signed size km/s, the
    part of the cost spent turning the flight path level km/s). Fired at an
    apsis this is the usual purely prograde burn."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    sin_fpa = float(np.dot(r, v)) / (np.linalg.norm(r) * np.linalg.norm(v))
    if abs(sin_fpa) < LEVEL_SIN_FPA:                     # level already: one clean component
        d = speed - float(np.linalg.norm(v))
        return (d, 0.0, 0.0), d, 0.0
    dv = speed * _level(r, v) - v
    dv_vnb = vnb_basis(r, v) @ dv
    size = float(np.linalg.norm(dv))
    signed = size if dv_vnb[0] >= 0 else -size
    return tuple(float(x) for x in dv_vnb), signed, size - abs(speed - float(np.linalg.norm(v)))


def _far_apsis(sim, i: int, y0: np.ndarray, t0: float, tof: float, highest: bool):
    """(radius km, absolute time, state) of the far apsis of each row of ``y0``
    flown together from ``t0`` with the run's forces. The apsis is bracketed
    by the sign of r.v among samples around the two-body transfer time (0.9 to
    1.15 of it, then wider), placed on the cubic Hermite curve through the two
    samples either side (sub-metre over that span), and the state there is
    flown to once more."""
    out = []
    for span in ((0.9, 1.15, 11), (0.5, 1.6, 45)):
        times = t0 + tof * np.linspace(*span)
        track = sim._coast_samples([i] * len(y0), y0, t0, times, h_max=tof / 200)
        rv = np.einsum("nkj,nkj->nk", track[:, :, :3], track[:, :, 3:])
        # past a highest point the radius falls (r.v < 0); past a lowest one it rises
        turned = [np.flatnonzero(row < 0 if highest else row > 0) for row in rv]
        if all(t.size and t[0] > 0 for t in turned):
            break
    else:
        raise ValueError("the transfer orbit has no apsis near where it should be")
    flow = sim.propagator == "cowell"      # j2mean re-reads mean elements: always from t0
    frac = np.linspace(0.0, 1.0, 2001)
    for n, row in enumerate(turned):
        k = int(row[0])
        ya, yb, h = track[n, k - 1], track[n, k], float(times[k] - times[k - 1])
        rad = np.linalg.norm(_hermite_at(ya, yb, h, frac), axis=1)
        j = int(np.argmax(rad) if highest else np.argmin(rad))
        j = min(max(j, 1), len(frac) - 2)
        a, b, c = rad[j - 1], rad[j], rad[j + 1]
        curv = a - 2.0 * b + c
        shift = 0.5 * (a - c) / curv if curv else 0.0
        t_x = float(times[k - 1]) + h * (frac[j] + shift * (frac[1] - frac[0]))
        y = (sim.coast_states([i], ya[None], float(times[k - 1]), t_x) if flow
             else sim.coast_states([i], y0[n:n + 1], t0, t_x))[0]
        out.append((float(np.linalg.norm(y[:3])), t_x, y))
    return out


def _transfer(sim, i: int, t0: float, r, v, r_goal: float, aim: bool = True):
    """Speed to leave (r, v) at ``t0`` with, level, so the far apsis lands at
    radius ``r_goal`` under the run's own forces (J2 alone moves a LEO-to-GEO
    apogee ~180 km). Vis-viva gives the first guess; the first pass also flies
    a neighbouring speed for the sensitivity, and Newton's step from there
    lands within ``APSIS_TOL`` at once (checked by flying it). Returns
    (speed km/s, time of the far apsis, state there)."""
    r = np.asarray(r, dtype=float)
    r1 = float(np.linalg.norm(r))
    level = _level(r, v)
    speed = math.sqrt(MU_EARTH * (2 / r1 - 2 / (r1 + r_goal)))
    tof = math.pi * math.sqrt(((r1 + r_goal) / 2) ** 3 / MU_EARTH)
    highest = r_goal > r1
    if not aim:                          # the two-body estimate: vis-viva, half a period
        r_x, v_x = kepler_propagate(r, speed * level, tof)
        return speed, t0 + tof, np.concatenate([r_x, v_x])
    step = APSIS_STEP
    trial = np.array([np.concatenate([r, s * level]) for s in (speed, speed + step)])
    (ra, _, _), (rb, _, _) = _far_apsis(sim, i, trial, t0, tof, highest)
    slope = (rb - ra) / step
    miss = ra - r_goal
    for _ in range(AIM_ITER):
        speed -= miss / slope
        (R, t_x, y_x), = _far_apsis(sim, i, np.concatenate([r, speed * level])[None], t0, tof,
                                     highest)
        miss = R - r_goal
        if abs(miss) < APSIS_TOL:
            return speed, t_x, y_x
    raise ValueError(f"could not aim the transfer at {r_goal - R_EARTH:,.0f} km "
                     f"(still {miss:,.1f} km off)")


def _level_note(extra: float) -> str:
    """Summary text for the cost of levelling the flight path, if any."""
    if extra < 5e-4:
        return ""
    return (f"; {extra * 1000:.1f} m/s of it levels the flight path because the burn is "
            "not at an apsis (burning at the next periapsis or apoapsis is cheaper)")


def plan_hohmann(sim, sat: str, target_alt: float, timing: str = "now", delay: float = 0.0,
                 aim: bool = True):
    """Raise or lower ``sat`` to a circular orbit at ``target_alt`` km. Returns (burns, summary).

    Burn 1 leaves the satellite moving level, fast enough (or slow enough) that
    the far apsis is at the target radius with the run's forces, so it works
    from any point of an eccentric orbit; burn 2 circularizes there. Without
    ``aim`` it is the two-body estimate (fast, for a preview: a LEO-to-GEO
    transfer flown with J2 takes ~5,000 force evaluations to aim)."""
    t_b, r, v = _burn_state(sim, sat, timing, delay, aim)
    i = sim.require(sat)
    r2 = R_EARTH + within(target_alt, 0.0, MAX_ALT, "target altitude (km)")
    speed, t_2, y2 = _transfer(sim, i, t_b, r, v, r2, aim)
    dv1_vnb, dv1, extra = _apsis_burn(r, v, speed)
    dv2 = math.sqrt(MU_EARTH / r2) - float(np.linalg.norm(y2[3:]))
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=dv1_vnb, frame="VNB",
                 label=f"Hohmann burn 1 ({dv1 * 1000:+.1f} m/s)"),
        Maneuver(sat, "circularize", timing="absolute", t=t_2,
                 label="Hohmann burn 2 (circularize)"),
    ]
    summary = (f"dV1 {dv1 * 1000:+.1f} m/s, dV2 ~{dv2 * 1000:+.1f} m/s, "
               f"total ~{(abs(dv1) + abs(dv2)) * 1000:.1f} m/s, coast {(t_2 - t_b) / 3600:.2f} h"
               + _level_note(extra))
    return burns, summary if aim else ESTIMATE + summary


def plan_bielliptic(sim, sat: str, rb_alt: float, target_alt: float, timing: str = "now",
                    delay: float = 0.0, aim: bool = True):
    """Three-burn transfer to ``target_alt`` km via apoapsis ``rb_alt`` km, each
    leg aimed with the run's forces (two-body without ``aim``). Returns (burns,
    summary)."""
    t_b, r, v = _burn_state(sim, sat, timing, delay, aim)
    i = sim.require(sat)
    r1 = float(np.linalg.norm(r))
    rb = R_EARTH + within(rb_alt, 0.0, MAX_ALT, "intermediate apoapsis (km)")
    r2 = R_EARTH + within(target_alt, 0.0, MAX_ALT, "target altitude (km)")
    if rb < max(r1, r2):
        raise ValueError("intermediate apoapsis must be above both orbits")
    s1, t_1, yb = _transfer(sim, i, t_b, r, v, rb, aim)
    dv1_vnb, dv1, extra = _apsis_burn(r, v, s1)
    s2, t_2, y2 = _transfer(sim, i, t_1, yb[:3], yb[3:], r2, aim)
    dv2 = s2 - float(np.linalg.norm(yb[3:]))
    dv3 = math.sqrt(MU_EARTH / r2) - float(np.linalg.norm(y2[3:]))
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=dv1_vnb, frame="VNB",
                 label=f"Bi-elliptic burn 1 ({dv1 * 1000:+.1f} m/s)"),
        Maneuver(sat, "impulse", timing="absolute", t=t_1, dv=(dv2, 0, 0), frame="VNB",
                 label=f"Bi-elliptic burn 2 ({dv2 * 1000:+.1f} m/s)"),
        Maneuver(sat, "circularize", timing="absolute", t=t_2,
                 label="Bi-elliptic burn 3 (circularize)"),
    ]
    summary = (f"dV {dv1 * 1000:+.0f} / {dv2 * 1000:+.0f} / ~{dv3 * 1000:+.0f} m/s, total "
               f"~{(abs(dv1) + abs(dv2) + abs(dv3)) * 1000:.0f} m/s, {(t_2 - t_b) / 3600:.1f} h"
               + _level_note(extra))
    return burns, summary if aim else ESTIMATE + summary


def aim(sim, sat: str, r1, v_guess, t_b: float, tof: float, r2):
    """Departure velocity that takes ``sat`` from ``r1`` at ``t_b`` to ``r2`` after
    ``tof`` s under the run's own forces: Newton iteration on the arrival miss,
    with the 3x3 sensitivity from three neighbouring trial arcs flown in the
    same batch. Returns (v1, arrival state). Raises ValueError if it does not
    converge to within ``AIM_TOL``."""
    i = sim.require(sat)
    r1 = np.asarray(r1, dtype=float)
    v = np.asarray(v_guess, dtype=float).copy()
    miss = math.inf
    for _ in range(AIM_ITER):
        trial = np.repeat(np.concatenate([r1, v])[None], 4, axis=0)
        trial[1:, 3:] += AIM_STEP * np.eye(3)
        y = sim.coast_states([i] * 4, trial, t_b, t_b + tof)
        err = y[0, :3] - r2
        miss = float(np.linalg.norm(err))
        if miss < AIM_TOL:
            return v, y[0]
        jac = (y[1:, :3] - y[0, :3]).T / AIM_STEP
        v = v - np.linalg.solve(jac, err)
        if not np.all(np.isfinite(v)):
            break
    raise ValueError(f"rendezvous aiming did not converge (still {miss:,.3f} km off after "
                     f"{AIM_ITER} corrections); try another time of flight")


def plan_rendezvous(sim, sat: str, target: str, tof: float, timing: str = "now",
                    delay: float = 0.0):
    """Intercept ``target`` after ``tof`` seconds, then match its velocity.

    The intercept is a Lambert arc corrected against the run's force model, so
    it arrives on the target's predicted position. Returns (burns, summary);
    the summary warns when the arc would re-enter.
    """
    if sat == target:
        raise ValueError("choose a different target satellite")
    tof = within(tof, 1.0, PLAN_HORIZON, "time of flight (s)")
    t_b, r1, v1 = _burn_state(sim, sat, timing, delay)
    _within_horizon(t_b + tof - sim.t, "the arrival")
    tgt = sim.predict(target, t_b + tof)
    r2, v2t = tgt[:3], tgt[3:]
    lv1, _ = lambert(r1, r2, tof, normal=np.cross(r1, v1))
    low = _arc_perigee_alt(r1, lv1, tof)
    if low < 0:
        raise ValueError(f"the transfer arc for a {tof / 60:,.0f} min flight passes "
                         f"{-low:,.0f} km below the surface; choose another time of flight")
    av1, arrive = aim(sim, sat, r1, lv1, t_b, tof, r2)
    dv1 = av1 - v1
    dv2 = v2t - arrive[3:]
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=tuple(float(x) for x in dv1),
                 frame="ECI", label=f"Intercept burn ({np.linalg.norm(dv1) * 1000:.1f} m/s)"),
        Maneuver(sat, "match_velocity", timing="absolute", t=t_b + tof, target=target,
                 label=f"Match velocity with {target}"),
    ]
    perigee_alt = _arc_perigee_alt(r1, av1, tof)
    summary = (f"dV1 {np.linalg.norm(dv1) * 1000:.1f} m/s, dV2 ~{np.linalg.norm(dv2) * 1000:.1f} "
               f"m/s, total ~{(np.linalg.norm(dv1) + np.linalg.norm(dv2)) * 1000:.1f} m/s")
    if perigee_alt < SAFE_PERIGEE_ALT:
        summary = f"WARNING: arc dips to {perigee_alt:,.0f} km altitude - " + summary
    sweep = math.degrees(math.acos(float(np.clip(
        np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0))))
    if np.dot(np.cross(r1, r2), np.cross(r1, v1)) < 0:
        sweep = 360.0 - sweep               # the long way round, as the chaser flies it
    if abs(sweep - 180.0) < HALF_TURN_DEG:
        summary = (f"WARNING: the arc sweeps {sweep:.1f} deg, so close to half a turn that its "
                   "plane is barely defined (another time of flight is cheaper) - " + summary)
    return burns, summary


def _arc_perigee_alt(r1, v1, tof: float) -> float:
    """Lowest altitude actually flown on the transfer arc (not merely the
    orbit's periapsis, which may lie beyond the arrival point)."""
    ts = np.linspace(0.0, tof, 64)
    r, _ = kepler_propagate(np.repeat(np.asarray(r1)[None], len(ts), 0),
                            np.repeat(np.asarray(v1)[None], len(ts), 0), ts)
    return float(np.min(np.linalg.norm(r, axis=1))) - R_EARTH


def scan_rendezvous(sim, sat: str, target: str, tof_min: float = 600.0, tof_max: float = 14400.0,
                    steps: int = 120, timing: str = "now", delay: float = 0.0):
    """Cheapest total delta-v time of flight whose arc stays above
    ``SAFE_PERIGEE_ALT``. Returns (tof_seconds, total_dv_km_s).

    The target is predicted with the run's force model at every candidate
    arrival time; the arcs themselves are two-body Lambert solutions (the
    force-model correction changes delta-v by well under a percent, so it is
    left to :func:`plan_rendezvous` for the chosen time only)."""
    t_b, r1, v1 = _burn_state(sim, sat, timing, delay)
    _within_horizon(t_b + tof_max - sim.t, "the latest arrival searched")
    j = sim.require(target)
    tofs = np.linspace(tof_min, tof_max, steps)
    tgt = _track(sim, j, t_b + tofs)
    normal = np.cross(r1, v1)
    best = (None, math.inf)
    for tof, row in zip(tofs, tgt, strict=True):
        r2, v2t = row[:3], row[3:]
        try:
            lv1, lv2 = lambert(r1, r2, tof, normal=normal)
        except ValueError:
            continue
        if _arc_perigee_alt(r1, lv1, tof) < SAFE_PERIGEE_ALT:
            continue
        total = float(np.linalg.norm(lv1 - v1) + np.linalg.norm(v2t - lv2))
        if total < best[1]:
            best = (float(tof), total)
    if best[0] is None:
        raise ValueError("no safe rendezvous arc in the searched window")
    return best


def _track(sim, j: int, times) -> np.ndarray:
    """States of satellite ``j`` at the increasing absolute ``times``, coasting
    from now under the run's force model (one pass, leg by leg)."""
    out = np.empty((len(times), 6))
    t, y = sim.t, sim.y[j:j + 1].copy()
    for k, tk in enumerate(times):
        y = sim.coast_states([j], y, t, float(tk))
        t = float(tk)
        out[k] = y[0]
    return out


def plane_change_summary(sim, sat: str, delta_i: float, timing: str, delay: float = 0.0):
    """Delta-v estimate for turning the orbit plane by ``delta_i`` deg at
    ``timing`` (the two-body speed there: a preview, shown while typing)."""
    _, _, v = _burn_state(sim, sat, timing, delay, aim=False, horizon=False)
    return f"dV ~{plane_change_dv(float(np.linalg.norm(v)), math.radians(delta_i)) * 1000:.1f} m/s"
