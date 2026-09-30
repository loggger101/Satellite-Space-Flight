"""Mission planners that turn a goal ("go to 35786 km", "meet satellite B in
3 h") into concrete scheduled manoeuvres for a running simulation.

Burn points are predicted with two-body motion; perturbations acting during
the coast make the arrival slightly imperfect, exactly as in real
operations - hence the dynamic ``circularize`` / ``match_velocity`` final
burns, which are computed from the actual state when they execute.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .elements import kepler_propagate
from .maneuvers import Maneuver, hohmann, lambert, plane_change_dv, resolve_time

SAFE_PERIGEE_ALT = 150.0   # km; lower transfer arcs would re-enter


def _burn_state(sim, sat: str, timing: str, delay: float = 0.0):
    """Time of the first burn and the two-body state ``sat`` will have then."""
    i = sim.index_of(sat)
    r, v = sim.y[i, :3], sim.y[i, 3:]
    t_b = resolve_time(Maneuver(sat, timing=timing, delay=delay), sim.t, r, v)
    rb, vb = kepler_propagate(r, v, t_b - sim.t) if t_b > sim.t else (r.copy(), v.copy())
    return t_b, np.asarray(rb), np.asarray(vb)


def plan_hohmann(sim, sat: str, target_alt: float, timing: str = "now", delay: float = 0.0):
    """Raise or lower ``sat`` to a circular orbit at ``target_alt`` km. Returns (burns, summary)."""
    t_b, r, v = _burn_state(sim, sat, timing, delay)
    r1 = float(np.linalg.norm(r))
    r2 = R_EARTH + target_alt
    speed = float(np.linalg.norm(v))
    v_needed = math.sqrt(MU_EARTH * (2 / r1 - 2 / (r1 + r2)))
    dv1 = v_needed - speed
    tof = math.pi * math.sqrt(((r1 + r2) / 2) ** 3 / MU_EARTH)
    _, dv2, _ = hohmann(r1, r2)
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=(dv1, 0.0, 0.0), frame="VNB",
                 label=f"Hohmann burn 1 ({dv1 * 1000:+.1f} m/s)"),
        Maneuver(sat, "circularize", timing="absolute", t=t_b + tof,
                 label="Hohmann burn 2 (circularize)"),
    ]
    summary = (f"dV1 {dv1 * 1000:+.1f} m/s, dV2 ~{dv2 * 1000:+.1f} m/s, "
               f"total ~{(abs(dv1) + abs(dv2)) * 1000:.1f} m/s, coast {tof / 3600:.2f} h")
    return burns, summary


def plan_bielliptic(sim, sat: str, rb_alt: float, target_alt: float, timing: str = "now",
                    delay: float = 0.0):
    """Three-burn transfer to ``target_alt`` km via apoapsis ``rb_alt`` km.
    Returns (burns, summary)."""
    t_b, r, v = _burn_state(sim, sat, timing, delay)
    r1 = float(np.linalg.norm(r))
    rb = R_EARTH + rb_alt
    r2 = R_EARTH + target_alt
    if rb < max(r1, r2):
        raise ValueError("intermediate apoapsis must be above both orbits")
    speed = float(np.linalg.norm(v))
    dv1 = math.sqrt(MU_EARTH * (2 / r1 - 2 / (r1 + rb))) - speed
    t1 = math.pi * math.sqrt(((r1 + rb) / 2) ** 3 / MU_EARTH)
    dv2 = (math.sqrt(MU_EARTH * (2 / rb - 2 / (rb + r2)))
           - math.sqrt(MU_EARTH * (2 / rb - 2 / (r1 + rb))))
    t2 = math.pi * math.sqrt(((rb + r2) / 2) ** 3 / MU_EARTH)
    dv3 = math.sqrt(MU_EARTH / r2) - math.sqrt(MU_EARTH * (2 / r2 - 2 / (rb + r2)))
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=(dv1, 0, 0), frame="VNB",
                 label=f"Bi-elliptic burn 1 ({dv1 * 1000:+.1f} m/s)"),
        Maneuver(sat, "impulse", timing="absolute", t=t_b + t1, dv=(dv2, 0, 0), frame="VNB",
                 label=f"Bi-elliptic burn 2 ({dv2 * 1000:+.1f} m/s)"),
        Maneuver(sat, "circularize", timing="absolute", t=t_b + t1 + t2,
                 label="Bi-elliptic burn 3 (circularize)"),
    ]
    summary = (f"dV {dv1 * 1000:+.0f} / {dv2 * 1000:+.0f} / ~{dv3 * 1000:+.0f} m/s, total "
               f"~{(abs(dv1) + abs(dv2) + abs(dv3)) * 1000:.0f} m/s, {(t1 + t2) / 3600:.1f} h")
    return burns, summary


def plan_rendezvous(sim, sat: str, target: str, tof: float, timing: str = "now",
                    delay: float = 0.0):
    """Lambert intercept of ``target`` after ``tof`` seconds, then velocity match.

    Returns (burns, summary); the summary warns when the arc would re-enter.
    """
    if sat == target:
        raise ValueError("choose a different target satellite")
    t_b, r1, v1 = _burn_state(sim, sat, timing, delay)
    j = sim.index_of(target)
    r2, v2t = kepler_propagate(sim.y[j, :3], sim.y[j, 3:], t_b + tof - sim.t)
    prograde = np.cross(r1, v1)[2] >= 0
    lv1, lv2 = lambert(r1, r2, tof, prograde=prograde)
    dv1 = lv1 - v1
    dv2 = np.asarray(v2t) - lv2
    burns = [
        Maneuver(sat, "impulse", timing="absolute", t=t_b, dv=tuple(float(x) for x in dv1),
                 frame="ECI", label=f"Intercept burn ({np.linalg.norm(dv1) * 1000:.1f} m/s)"),
        Maneuver(sat, "match_velocity", timing="absolute", t=t_b + tof, target=target,
                 label=f"Match velocity with {target}"),
    ]
    perigee_alt = _arc_perigee_alt(r1, lv1, tof)
    summary = (f"dV1 {np.linalg.norm(dv1) * 1000:.1f} m/s, dV2 ~{np.linalg.norm(dv2) * 1000:.1f} "
               f"m/s, total ~{(np.linalg.norm(dv1) + np.linalg.norm(dv2)) * 1000:.1f} m/s")
    if perigee_alt < SAFE_PERIGEE_ALT:
        summary = f"WARNING: arc dips to {perigee_alt:,.0f} km altitude - " + summary
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
    ``SAFE_PERIGEE_ALT``. Returns (tof_seconds, total_dv_km_s)."""
    t_b, r1, v1 = _burn_state(sim, sat, timing, delay)
    j = sim.index_of(target)
    prograde = np.cross(r1, v1)[2] >= 0
    best = (None, math.inf)
    for tof in np.linspace(tof_min, tof_max, steps):
        r2, v2t = kepler_propagate(sim.y[j, :3], sim.y[j, 3:], t_b + tof - sim.t)
        try:
            lv1, lv2 = lambert(r1, r2, tof, prograde=prograde)
        except ValueError:
            continue
        if _arc_perigee_alt(r1, lv1, tof) < SAFE_PERIGEE_ALT:
            continue
        total = float(np.linalg.norm(lv1 - v1) + np.linalg.norm(np.asarray(v2t) - lv2))
        if total < best[1]:
            best = (float(tof), total)
    if best[0] is None:
        raise ValueError("no safe rendezvous arc in the searched window")
    return best


def plane_change_summary(sim, sat: str, delta_i: float, timing: str, delay: float = 0.0):
    """Delta-v estimate for turning the orbit plane by ``delta_i`` deg at ``timing``."""
    _, _, v = _burn_state(sim, sat, timing, delay)
    return f"dV ~{plane_change_dv(float(np.linalg.norm(v)), math.radians(delta_i)) * 1000:.1f} m/s"
