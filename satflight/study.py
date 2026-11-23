"""Measurements for the Sentinel-1 study (plan step 5), from the precise orbits.

* :func:`revolutions` - one row per revolution, ascending node to ascending
  node: the node's time, Earth-fixed longitude and local time (LTAN), and the
  revolution's mean a, eccentricity vector, inclination.
* :func:`track_offsets` - how far each node lands east or west of where its
  relative orbit (one of the 175 in the 12-day repeat) usually crosses: the
  ground-track control the operators fly.
* :func:`phasing` - how far 1C trails 1A around the orbit (180 deg nominal, 176 measured).
* :func:`burn_sizes` - each flagged manoeuvre sized from the jump in the mean
  elements across it.
* :func:`accuracy_arc` - one of the weekly accuracy arcs: fit a day, fly three.

Means are time averages of the osculating elements over exactly one
revolution, node to node (:func:`span_mean`; a plain average of the samples is
biased by up to 17 m in a): the short-period J2 terms cancel, the long-period
ones (frozen eccentricity) and the secular ones (drag) stay.
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta

import numpy as np

from .constants import MU_EARTH, R_EARTH
from .elements import rv2coe

REPEAT_ORBITS = 175          # revolutions in Sentinel-1's 12-day repeat cycle
M_PER_DEG = R_EARTH * 1e3 * math.pi / 180.0     # along the equator


def _hermite(t, r, v, k, s):
    """Cubic Hermite position and velocity at fraction ``s`` of interval ``k``."""
    h = t[k + 1] - t[k]
    r0, r1, m0, m1 = r[k], r[k + 1], v[k] * h, v[k + 1] * h
    s2, s3 = s * s, s * s * s
    p = ((2 * s3 - 3 * s2 + 1) * r0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * r1
         + (s3 - s2) * m1)
    d = ((6 * s2 - 6 * s) * r0 + (3 * s2 - 4 * s + 1) * m0 + (-6 * s2 + 6 * s) * r1
         + (3 * s2 - 2 * s) * m1) / h
    return p, d


def _us(seconds) -> np.ndarray:
    return np.round(np.asarray(seconds, float) * 1e6).astype("timedelta64[us]")


def span_mean(t, y, t0: float, t1: float, y0, y1):
    """Time-mean of ``y`` sampled at ``t`` over exactly [t0, t1], with the
    values at the ends given (``y0``, ``y1``): trapezoids through the samples
    inside. A revolution's ends fall between samples; dropping or adding the
    nearest sample instead biases the mean of anything that swings twice a
    revolution (osculating a: ~10 km with J2, so ~10 m)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    k = np.flatnonzero((t > t0) & (t < t1))
    tt = np.concatenate([[t0], t[k], [t1]])
    yy = np.concatenate([np.asarray(y0, float)[None], y[k], np.asarray(y1, float)[None]])
    return np.trapezoid(yy, tt, axis=0) / (t1 - t0)


def revolutions(arc, eop=None) -> dict:
    """One row per whole revolution in ``arc`` (a ``poeorb.Arc``), as columns:

    ``node_utc`` (datetime64[us]) ascending node, found in the inertial frame
    to the microsecond; ``period_s`` to the next node; ``orbit`` the files'
    absolute orbit number mid-revolution; ``lon_deg`` the node's Earth-fixed
    longitude; ``ltan_h`` the local mean solar time there (UT1 + lon/15);
    ``a_km``, ``ex``, ``ey`` (e cos/sin argp), ``i_deg`` the revolution's
    means; ``burn`` / ``gap`` whether any sample in it is flagged
    DEGRADED-MANOEUVRE / DEGRADED-GAP. ``eop`` takes polar motion out first.
    """
    r, v = arc.eci(eop)
    t = arc.seconds()
    ks = np.flatnonzero((r[:-1, 2] < 0.0) & (r[1:, 2] >= 0.0))
    tn, lon, node_states = [], [], []
    for k in ks:
        s = -r[k, 2] / (r[k + 1, 2] - r[k, 2])
        for _ in range(20):                       # Newton on z(s) = 0 within the interval
            p, d = _hermite(t, r, v, k, s)
            ds = -p[2] / (d[2] * (t[k + 1] - t[k]))
            s += ds
            if abs(ds) < 1e-12:
                break
        tn.append(t[k] + s * (t[k + 1] - t[k]))
        node_states.append(np.concatenate(_hermite(t, r, v, k, s)))
        pf, _ = _hermite(t, arc.r, arc.v, k, s)
        lon.append(math.degrees(math.atan2(pf[1], pf[0])))
    tn, lon = np.array(tn), np.array(lon)
    def columns(el):
        return np.stack([np.atleast_1d(el.a), np.atleast_1d(el.e * np.cos(el.argp)),
                         np.atleast_1d(el.e * np.sin(el.argp)), np.atleast_1d(el.i)], 1)

    cols = columns(rv2coe(r, v))
    if len(ks):
        ns = np.array(node_states)
        at_node = columns(rv2coe(ns[:, :3], ns[:, 3:]))
    flagged_burn = np.concatenate([[0], np.cumsum(arc.manoeuvre)])
    flagged_gap = np.concatenate([[0], np.cumsum(arc.gap)])
    rows = {c: [] for c in ("a_km", "ex", "ey", "i_deg", "orbit", "burn", "gap")}
    for j in range(len(ks) - 1):
        lo, hi = ks[j] + 1, ks[j + 1] + 1         # samples inside the revolution
        m = span_mean(t[lo:hi], cols[lo:hi], tn[j], tn[j + 1], at_node[j], at_node[j + 1])
        rows["a_km"].append(m[0])
        rows["ex"].append(m[1])
        rows["ey"].append(m[2])
        rows["i_deg"].append(math.degrees(m[3]))
        rows["orbit"].append(int(arc.abs_orbit[(lo + hi) // 2]))
        # the samples either side of the nodes count too: a burn there touches it
        rows["burn"].append(bool(flagged_burn[hi + 1] - flagged_burn[lo - 1]))
        rows["gap"].append(bool(flagged_gap[hi + 1] - flagged_gap[lo - 1]))
    n = max(len(ks) - 1, 0)
    node_utc = arc.utc[0] + _us(tn[:n])
    ut1_h = ((node_utc - node_utc.astype("datetime64[D]")) / np.timedelta64(1, "h")
             + np.interp(tn[:n], t, arc.dut1) / 3600.0)
    out = {"node_utc": node_utc, "period_s": np.diff(tn), "lon_deg": lon[:n],
           "ltan_h": (ut1_h + lon[:n] / 15.0) % 24.0}
    for c, vals in rows.items():
        out[c] = np.array(vals, dtype=bool if c in ("burn", "gap") else
                          int if c == "orbit" else float)
    return out


def _wrap180(deg):
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def track_offsets(orbit, lon_deg, cycle: int = REPEAT_ORBITS) -> np.ndarray:
    """East-west offset (m, at the equator) of each node from the median
    crossing of its relative orbit (``orbit`` mod ``cycle``)."""
    orbit = np.asarray(orbit, int)
    lon = np.asarray(lon_deg, float)
    out = np.full(len(lon), np.nan)
    rel = orbit % cycle
    for q in np.unique(rel):
        m = rel == q
        d = _wrap180(lon[m] - lon[m][0])
        out[m] = (d - np.median(d)) * M_PER_DEG
    return out


def phasing(lead: dict, trail: dict):
    """(times, deg): at each of ``lead``'s nodes, how far around the orbit
    ``trail`` is behind it, from when ``trail`` reaches the node next (as a
    fraction of its revolution then); nodes where ``trail`` has no node within
    a revolution (its data has a gap) are left out. Sentinel-1C flies about 176 deg
    behind 1A."""
    ta = lead["node_utc"]
    tc = trail["node_utc"]
    j = np.searchsorted(tc, ta)
    ok = j < len(trail["period_s"])
    j = j[ok]
    dt = (tc[j] - ta[ok]) / np.timedelta64(1, "s")
    near = dt < trail["period_s"][j]          # else the trailer has a gap there (an outage)
    return ta[ok][near], 360.0 * dt[near] / trail["period_s"][j][near]


def group_windows(windows, within_s: float = 3 * 3600.0):
    """Manoeuvre windows (start, end) gathered into manoeuvres: burns less than
    ``within_s`` apart (Sentinel-1 often burns twice about an orbit apart) are
    one, since no clean revolution separates them."""
    groups = []
    for t0, t1 in sorted((np.datetime64(a, "us"), np.datetime64(b, "us")) for a, b in windows):
        if groups and t0 - groups[-1][-1][1] < _us(within_s):
            groups[-1].append((t0, t1))
        else:
            groups.append([(t0, t1)])
    return groups


def placebo_windows(windows, min_gap_days: float = 2.0, length_s: float = 600.0):
    """A 10-minute window midway between each pair of manoeuvres at least
    ``min_gap_days`` apart: sized like a burn, it shows the method's noise,
    since nothing burned there."""
    groups = group_windows(windows)
    out = []
    for g0, g1 in zip(groups, groups[1:], strict=False):
        a, b = g0[-1][1], g1[0][0]
        if (b - a) / np.timedelta64(1, "s") >= min_gap_days * 86400.0:
            mid = a + (b - a) // 2
            out.append((mid, mid + _us(length_s)))
    return out


def _step(x, lon, y, after):
    """The step at ``after`` in y, fitted with a line in time and the daily
    wobble: harmonics 1 and 2 of the node's Earth-fixed longitude (Earth-fixed
    gravity seen along a ground track that moves 25 deg west a revolution)."""
    lr = np.radians(lon)
    A = np.stack([np.ones_like(x), x, after.astype(float), np.cos(lr), np.sin(lr),
                  np.cos(2 * lr), np.sin(2 * lr)], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[2])


def burn_sizes(revs: dict, windows, n: int = 15, min_revs: int = 4) -> list[dict]:
    """Each manoeuvre (windows grouped by :func:`group_windows`) sized from the
    step in the revolution means across it: up to ``n`` clean revolutions each
    side, stopping at another burn, fitted together as a line in time plus the
    daily wobble plus the step. Gives the steps in a (m), i (mdeg) and the
    eccentricity vector (1e-6), and the delta-v they need: along-track
    v da / 2a and at least v |di| cross-track (exactly that only at a node).
    Too few revolutions on a side (the data's edge, burns close together)
    gives NaN and says why."""
    node = revs["node_utc"]
    end_t = node + _us(revs["period_s"])
    ref = node[0] if len(node) else np.datetime64("2000-01-01", "us")
    x = ((node + _us(revs["period_s"] / 2.0)) - ref) / np.timedelta64(1, "s") / 86400.0
    out = []
    for group in group_windows(windows):
        t0, t1 = group[0][0], group[-1][1]
        row = {"start_utc": t0, "end_utc": t1, "windows": len(group), "da_m": math.nan,
               "di_mdeg": math.nan, "de_1e6": math.nan, "dv_along_mm_s": math.nan,
               "dv_cross_mm_s": math.nan, "revs_before": 0, "revs_after": 0, "note": ""}
        # this manoeuvre's own flagged revolutions (it touches them) are skipped;
        # another manoeuvre's end the run
        mine = (end_t > t0 - _us(revs["period_s"])) & (node < t1 + _us(revs["period_s"]))
        b = list(_clean_run(np.flatnonzero(node < t0)[::-1], revs["burn"], mine, end_t, t0,
                            True))[:n]
        a = list(_clean_run(np.flatnonzero(end_t > t1), revs["burn"], mine, node, t1,
                            False))[:n]
        row["revs_before"], row["revs_after"] = len(b), len(a)
        if len(b) < min_revs or len(a) < min_revs:
            row["note"] = (f"only {len(b)} clean revolutions before and {len(a)} after "
                           f"(need {min_revs})")
            out.append(row)
            continue
        k = np.array(b[::-1] + a)
        after = np.arange(len(k)) >= len(b)
        jump = {c: _step(x[k], revs["lon_deg"][k], revs[c][k], after)
                for c in ("a_km", "i_deg", "ex", "ey")}
        a_km = float(np.mean(revs["a_km"][b]))
        v = math.sqrt(MU_EARTH / a_km)
        row.update(da_m=jump["a_km"] * 1e3, di_mdeg=jump["i_deg"] * 1e3,
                   de_1e6=math.hypot(jump["ex"], jump["ey"]) * 1e6,
                   dv_along_mm_s=v / (2.0 * a_km) * jump["a_km"] * 1e6,
                   dv_cross_mm_s=v * abs(math.radians(jump["i_deg"])) * 1e6)
        out.append(row)
    return out


def _clean_run(order, burn, mine, edge, at, before):
    """Clean revolutions from ``order`` (wholly before/after ``at``), skipping
    flagged ones near this manoeuvre and stopping at any other flagged one."""
    for k in order:
        if burn[k]:
            if mine[k]:
                continue
            return
        if (edge[k] <= at) if before else (edge[k] >= at):
            yield int(k)


def arc_starts(first: date, last: date, days: int = 4, every: int = 7) -> list[date]:
    """Weekly arc starts from ``first`` whose ``days`` days all fit by ``last``."""
    out, d = [], first
    while d + timedelta(days=days - 1) <= last:
        out.append(d)
        d += timedelta(days=every)
    return out


def _rms(x):
    x = np.asarray(x, float)
    return np.sqrt((x ** 2).mean(0)) if len(x) else np.full(3, np.nan)


def accuracy_arc(mission: str, day: date, model, mass: float, area: float = 10.0,
                 cd: float = 2.2, cr: float = 1.3, fit_hours: float = 24.0,
                 check_hours: float = 72.0, step: float = 60.0, eop_table=None,
                 root=None) -> dict:
    """One weekly accuracy arc: fit ``fit_hours`` from 00:00 UTC on ``day``
    (start state, Cd*A/m with Cr held at ``cr``, an impulse per flagged burn),
    then fly the solution ``check_hours`` on with only the later burns fitted
    to the truth. Returns a flat row: the fit's RMS (R/S/W), the drag as a
    density scale on the static table at ``cd`` (Cd and the scale are one
    number to the fit), and the prediction's error by day and at each 24 h.

    A burn needs ``fit.MIN_AFTER_S`` of data after its window. One too near the end of
    the fit day ends the fit where its window begins (the prediction, starting
    there, fits it); one too near the end of the check ends the check there.
    ``fit_hours`` / ``check_hours`` in the row are the spans flown; a fit cut
    below half its length is a FitError."""
    from . import fit, poeorb
    t_start = time.time()
    last = day + timedelta(days=int(math.ceil((fit_hours + check_hours) / 24.0)) - 1)
    arc = poeorb.load_arc(mission, day, last, root)
    t_end = fit_hours * 3600.0
    windows = fit.burn_windows(arc, hours=fit_hours + check_hours)
    while True:
        edge = [a for a, b, _ in windows if a < t_end and b + fit.MIN_AFTER_S > t_end]
        if not edge:
            break
        t_end = math.floor(min(edge) / step) * step
        if t_end < fit_hours * 1800.0:
            raise fit.FitError(f"burn at {min(edge) / 3600:.1f} h leaves under half of the "
                               f"{fit_hours:g} h fit")
    obs = fit.observations(arc, step=step, hours=t_end / 3600.0, eop=eop_table)
    burns = [m for a, _, m in windows if a < t_end]
    am = area / mass
    res = fit.fit_arc(obs, model, fit.first_state(arc, eop=eop_table), cd_am=cd * am,
                      cr_am=cr * am, fit_cr=False, burns=burns)
    used = res.residuals_rsw[obs.use]
    rf = _rms(used)
    row = {"mission": mission, "start": day.isoformat(), "fit_rms_m": res.rms_m,
           "fit_r_m": rf[0], "fit_s_m": rf[1], "fit_w_m": rf[2], "cd_am": res.cd_am,
           "density_scale": res.cd_am / (cd * am), "burns_fit": len(burns),
           "iterations": res.iterations, "fit_hours": t_end / 3600.0}
    # the prediction: the fitted orbit from the end of the fit, everything held
    # but the later burns, which only the truth can tell (the operators' plans
    # are not in the files); it starts on the first sample at or after t_end
    t_all = arc.seconds()
    t0 = float(t_all[np.searchsorted(t_all, t_end)])
    end = fit.fly_batch(model, obs.jd0, obs.dut1, res.y0[None], np.array([0.0, t0]),
                        [res.cd_am], [res.cr_am], burns, res.dv[None])[0, -1]
    chk = fit.observations(arc, step=step, hours=check_hours, start=t0, eop=eop_table)
    stop = float(chk.t[-1])
    later = [(a - t0, b - t0, max(m - t0, 0.0)) for a, b, m in windows if t_end <= a < t0 + stop]
    while (cut := [a for a, b, _ in later if b + fit.MIN_AFTER_S > stop]):
        # truth after a burn that cannot be fitted is not the model's error
        stop = math.floor(min(cut) / step) * step
        later = [w for w in later if w[0] < stop]
    chk = fit.Observations(chk.t, chk.r, chk.use & (chk.t <= stop), chk.jd0, chk.dut1)
    held = fit.fit_arc(chk, model, end, res.cd_am, res.cr_am, fit_state=False, fit_cd=False,
                       fit_cr=False, burns=[m for _, _, m in later])
    e = held.residuals_rsw
    row["burns_check"] = len(later)
    row["check_hours"] = check_hours if stop >= chk.t[-1] else stop / 3600.0
    for d in range(1, int(check_hours // 24) + 1):
        lo, hi = 24 * (d - 1) * 3600.0, 24 * d * 3600.0
        m = (chk.t > lo) & (chk.t <= hi) & chk.use
        rr = _rms(e[m])
        row.update({f"day{d}_r_m": rr[0], f"day{d}_s_m": rr[1], f"day{d}_w_m": rr[2]})
        k = np.flatnonzero(chk.use & (chk.t <= hi))
        row[f"at{24 * d}h_s_m"] = float(e[k[-1], 1]) if k.size and hi <= stop else math.nan
    row["seconds"] = time.time() - t_start
    return row
