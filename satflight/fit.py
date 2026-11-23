"""Fitting an orbit to observed positions: batch least squares.

Given positions over an arc (Sentinel-1's precise orbits, see ``poeorb``),
find the start state, the drag and sunlight-pressure strengths (Cd*A/m and
Cr*A/m) and the velocity change of each burn inside the arc, so that the
simulator's force model flies the closest orbit to them. Then whatever is
left over (the residuals) is the model's own error, and the fitted solution
flown forward predicts the orbit.

Gauss-Newton with a finite-difference Jacobian: every parameter's nudged
copy is flown as an extra row of one vectorized integration, so a fit of
eight parameters costs about as much as flying two satellites. Drag strength
and the atmosphere's density scale cannot be told apart; the fit reports
Cd*A/m at the model's ``density_scale``.

A burn is an impulse at a given time (the middle of the ESA files'
``DEGRADED-MANOEUVRE`` window): Sentinel-1's trims last seconds to a few
minutes, far shorter than an orbit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forces import ForceModel
from .integrators import Propagator
from .timeutil import gmst

RTOL = ATOL = 1e-11      # tight: the integration must sit far below the cm-level truth
H_MAX = 60.0
MIN_AFTER_S = 1800.0     # usable data a fitted burn needs after it (a third of a revolution)


class FitError(ValueError):
    """A fit that did not converge or ran away; says which and how far it got."""


@dataclass(frozen=True)
class Observations:
    """Positions to fit: ``t`` (s from the arc's start, increasing), ``r`` (N, 3)
    km in the simulator's ECI, ``use`` (N,) which ones count; ``jd0`` the UTC
    Julian date of t = 0 and ``dut1`` UT1 - UTC (s) for the Earth's rotation."""
    t: np.ndarray
    r: np.ndarray
    use: np.ndarray
    jd0: float
    dut1: float


@dataclass(frozen=True)
class Burn:
    """A fitted impulse: time (s), delta-v in ECI (km/s) and in the orbit's
    radial / along-track / cross-track frame (mm/s)."""
    t: float
    dv: np.ndarray
    rsw: np.ndarray


@dataclass
class FitResult:
    y0: np.ndarray                  # (6,) km, km/s at t = 0
    cd_am: float                    # m^2/kg
    cr_am: float
    dv: np.ndarray                  # (k, 3) km/s, one row per burn
    burns: list[Burn]
    rms_m: float                    # 3-D position RMS over the used samples
    residuals_rsw: np.ndarray       # (N, 3) m, observed minus fitted, every sample
    used: int
    iterations: int
    converged: bool
    sigma: dict = field(default_factory=dict)   # 1-sigma of each parameter (formal)


# --- flying a batch ---------------------------------------------------------------------------

def _derivative(model: ForceModel, jd0: float, dut1: float, cd_am, cr_am):
    cd_am = np.asarray(cd_am, float)
    cr_am = np.asarray(cr_am, float)

    def f(t, y):
        jd = jd0 + t / 86400.0
        theta = gmst(jd + dut1 / 86400.0)
        a = model.acceleration(y[:, :3], y[:, 3:], cd_am, theta, jd, cr_am)
        return np.concatenate([y[:, 3:], a], axis=1)
    return f


def fly_batch(model, jd0, dut1, Y0, t, cd_am, cr_am, burn_times=(), DV=None):
    """Fly B start states ``Y0`` (B, 6) to the times ``t`` (increasing, t[0] =
    0), each with its own Cd*A/m, Cr*A/m (B,) and burn delta-vs ``DV`` (B, k, 3)
    at ``burn_times`` (k,). Returns the states (B, N, 6)."""
    Y0 = np.atleast_2d(np.asarray(Y0, float))
    B = len(Y0)
    t = np.asarray(t, float)
    burn_times = list(burn_times)
    DV = np.zeros((B, len(burn_times), 3)) if DV is None else np.asarray(DV, float)
    f = _derivative(model, jd0, dut1, np.broadcast_to(cd_am, (B,)),
                    np.broadcast_to(cr_am, (B,)))
    prop = Propagator("dopri5", rtol=RTOL, atol=ATOL, h_max=H_MAX)
    stops = sorted(set(t.tolist()) | set(burn_times))
    want = {tt: k for k, tt in enumerate(t.tolist())}
    burn_at = {tt: k for k, tt in enumerate(burn_times)}
    out = np.empty((B, len(t), 6))
    y, now = Y0.copy(), 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for s in stops:
            if s > now:
                y = prop.integrate(f, now, y, s)[1]
                now = s
            if s in want:
                out[:, want[s]] = y
            if s in burn_at:                    # after recording: the burn starts here
                y = y.copy()
                y[:, 3:] += DV[:, burn_at[s]]
    return out


def fly(model, jd0, dut1, y0, t, cd_am, cr_am, burns=(), states=False):
    """One orbit flown to the times ``t`` with ``burns`` [(t, dv km/s)]: the
    positions (N, 3), or with ``states`` (N, 2, 3) rows of (r, v)."""
    times = [b[0] for b in burns]
    DV = np.array([[b[1] for b in burns]]) if burns else None
    s = fly_batch(model, jd0, dut1, np.asarray(y0, float)[None], t, [cd_am], [cr_am], times,
                  DV)[0]
    return s.reshape(len(s), 2, 3) if states else s[:, :3]


# --- the fit ----------------------------------------------------------------------------------

def rsw_frame(r, v):
    """Rows R, S, W (radial, along-track, cross-track unit vectors) per state."""
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    W = np.cross(r, v)
    W = W / np.linalg.norm(W, axis=-1, keepdims=True)
    S = np.cross(W, R)
    return np.stack([R, S, W], axis=-2)


def fit_arc(obs: Observations, model: ForceModel, y0, cd_am: float, cr_am: float,
            fit_cr: bool = True, burns=(), max_iter: int = 10, tol_km: float = 1e-6,
            fit_state: bool = True, fit_cd: bool = True) -> FitResult:
    """Fit the start state, Cd*A/m and Cr*A/m (each unless held with
    ``fit_state`` / ``fit_cd`` / ``fit_cr`` False) and an impulse at each of
    ``burns`` (s) to ``obs``. Holding the rest and fitting only burns is how a
    prediction meets the operators' later burns. Converged when the state moves
    under ``tol_km`` and the RMS settles; FitError if it diverges or
    ``max_iter`` passes. With nothing free it only flies (iterations 0)."""
    burns = [float(b) for b in burns]
    k = len(burns)
    p = np.concatenate([np.asarray(y0, float), [cd_am, cr_am], np.zeros(3 * k)])
    free = np.ones(len(p), bool)
    free[:6] = fit_state
    free[6] = fit_cd
    free[7] = fit_cr
    names = ["x", "y", "z", "vx", "vy", "vz", "cd_am", "cr_am"] + [
        f"burn{j}_{c}" for j in range(k) for c in "xyz"]
    step = np.array([1e-4] * 3 + [1e-7] * 3 + [max(1e-3 * abs(cd_am), 1e-6),
                                              max(1e-3 * abs(cr_am), 1e-6)] + [1e-7] * 3 * k)
    idx = np.flatnonzero(free)
    use = np.asarray(obs.use, bool)
    if use.sum() * 3 < len(idx):
        raise FitError(f"{use.sum()} usable positions cannot fix {len(idx)} parameters")
    for b in burns:
        # an impulse needs a good part of a revolution after it; without it, its
        # column is noise that the normalised solve blows up
        after = obs.t[use & (obs.t > b)]
        if not after.size or after[-1] - b < MIN_AFTER_S:
            raise FitError(f"burn at {b:.0f} s has under {MIN_AFTER_S:.0f} s of usable "
                           "observations after it; end the arc before its window")

    def unpack(rows):
        return (rows[:, :6], rows[:, 6], rows[:, 7], rows[:, 8:].reshape(len(rows), k, 3))

    rms_prev = np.inf
    it, J, scale = 0, None, None
    for it in range(1, max_iter + 1 if len(idx) else 1):
        rows = np.repeat(p[None], 1 + len(idx), axis=0)
        for j, q in enumerate(idx):
            rows[1 + j, q] += step[q]
        Y0, cd, cr, DV = unpack(rows)
        try:
            r = fly_batch(model, obs.jd0, obs.dut1, Y0, obs.t, cd, cr, burns, DV)[:, :, :3]
        except FloatingPointError as exc:
            raise FitError(f"diverged at iteration {it}: the orbit left the model's range "
                           f"({exc})") from None
        res = (obs.r - r[0])[use]                                     # km
        rms = float(np.sqrt((res ** 2).sum(1).mean())) * 1e3
        if not np.isfinite(rms) or rms > 1e6:
            raise FitError(f"diverged at iteration {it}: RMS {rms:.3g} m")
        J = ((r[1:] - r[0][None]) / step[idx][:, None, None])[:, use]  # (P, n, 3)
        J = J.reshape(len(idx), -1).T                                   # (3n, P)
        scale = np.linalg.norm(J, axis=0)
        scale[scale == 0] = 1.0
        dp, *_ = np.linalg.lstsq(J / scale, res.reshape(-1), rcond=None)
        dp = dp / scale
        p[idx] += dp
        # settled: the state stopped moving and the RMS stopped falling (by 0.1 %,
        # or by a mm once it is down at the integration's own noise)
        moved = float(np.linalg.norm(dp[:3])) if fit_state else 0.0
        if moved < tol_km and abs(rms - rms_prev) < max(1e-3 * rms, 1e-3):
            break
        rms_prev = rms
    else:
        if len(idx):
            raise FitError(f"did not converge in {max_iter} iterations (RMS {rms:.3g} m, "
                           f"last state change {moved * 1e3:.3g} m)")
        it = 0

    # the converged solution, flown once more for residuals and burn frames
    Y0, cd, cr, DV = unpack(p[None])
    s = fly_batch(model, obs.jd0, obs.dut1, Y0, obs.t, cd, cr, burns, DV)[0]
    diff = (obs.r - s[:, :3]) * 1e3
    frames = rsw_frame(s[:, :3], s[:, 3:])
    rsw = np.einsum("nij,nj->ni", frames, diff)
    rms = float(np.sqrt((diff[use] ** 2).sum(1).mean()))
    out_burns = []
    if k:
        at = fly_batch(model, obs.jd0, obs.dut1, Y0, np.array([0.0, *burns]), cd, cr, burns,
                       DV)[0, 1:]
        for j, tb in enumerate(burns):
            fr = rsw_frame(at[j, :3], at[j, 3:])
            out_burns.append(Burn(tb, DV[0, j].copy(), fr @ DV[0, j] * 1e6))
    sigma = {}
    if J is not None:
        J_scaled = J / scale
        try:
            cov = np.linalg.inv(J_scaled.T @ J_scaled) * (rms * 1e-3) ** 2
            sig = np.sqrt(np.maximum(np.diag(cov), 0.0)) / scale
            sigma = {names[q]: float(sig[j]) for j, q in enumerate(idx)}
        except np.linalg.LinAlgError:
            sigma = {}
    return FitResult(p[:6].copy(), float(p[6]), float(p[7]), DV[0].copy(), out_burns, rms, rsw,
                     int(use.sum()), it, True, sigma)


# --- from the precise orbit files -------------------------------------------------------------

def observations(arc, step: float = 60.0, hours: float | None = None, start: float = 0.0,
                 eop=None):
    """Positions from a ``poeorb.Arc`` every ``step`` s (a multiple of 10) from
    ``start`` s for ``hours``, in the simulator's ECI (polar motion taken out
    with ``eop``, an ``eop.Table``), with samples flagged DEGRADED (burns,
    tracking gaps) left out of the fit."""
    from .poeorb import STEP, julian_dates
    every = int(round(step / STEP))
    if every < 1 or abs(every * STEP - step) > 1e-9:
        raise ValueError(f"step {step} s is not a multiple of the files' {STEP:g} s")
    t_all = arc.seconds()
    sel = np.flatnonzero(t_all >= start)
    if hours is not None:
        sel = sel[t_all[sel] <= start + hours * 3600.0]
    sel = sel[::every]
    if len(sel) < 3:
        raise ValueError("fewer than 3 samples in that span")
    r, _ = arc.eci(eop)
    t = t_all[sel] - t_all[sel[0]]
    use = ~(arc.manoeuvre[sel] | arc.gap[sel])
    return Observations(t, r[sel], use, float(julian_dates(arc.utc[sel[:1]])[0]),
                        float(arc.dut1[sel[0]]))


def first_state(arc, start: float = 0.0, eop=None) -> np.ndarray:
    """The arc's ECI state (6,) at ``start`` s, as a fit's first guess."""
    k = int(np.searchsorted(arc.seconds(), start))
    r, v = arc.eci(eop)
    return np.concatenate([r[k], v[k]])


def burn_windows(arc, start: float = 0.0, hours: float | None = None):
    """[(t_start, t_end, t_mid)] s of each DEGRADED-MANOEUVRE run in the span,
    times from ``start``."""
    t = arc.seconds() - start
    flag = arc.manoeuvre.astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag, [0]])))
    out = []
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        t0, t1 = float(t[a]), float(t[b - 1])
        if t1 < 0 or (hours is not None and t0 > hours * 3600.0):
            continue
        out.append((t0, t1, 0.5 * (t0 + t1)))
    return out
