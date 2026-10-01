"""Numerical integrators for the ensemble state ``y`` of shape ``(N, 6)``.

* ``dopri5``   Dormand-Prince 5(4) with adaptive step control and FSAL - the
               default; the same scheme as scipy's RK45 and MATLAB's ode45.
* ``rk4``      classic fixed-step 4th-order Runge-Kutta.
* ``wh``       Wisdom-Holman: kick-drift-kick where the drift is the exact
               Kepler orbit and only the small perturbations (J2..J4, drag,
               thrust) are kicks - the splitting behind REBOUND's WHFast. At a
               60 s step it is ~1000x more accurate than leapfrog in LEO.
* ``leapfrog`` kick-drift-kick with a straight-line drift (REBOUND's
               LEAPFROG). Symplectic, so its energy error stays bounded, but
               the phase error grows fast: it needs steps of a few seconds.

All satellites share one step (ensemble integration). The adaptive error
norm takes the worst satellite, so every orbit individually meets the
tolerance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH
from .elements import kepler_propagate

Deriv = Callable[[float, np.ndarray], np.ndarray]

METHODS = ("dopri5", "rk4", "wh", "leapfrog")

# Dormand-Prince 5(4) tableau
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
                187 / 2100, 1 / 40])
_E = _B5 - _B4


@dataclass
class StepStats:
    """Running counters shown in the UI's integrator readout."""

    accepted: int = 0
    rejected: int = 0
    evaluations: int = 0
    last_h: float = 0.0


def rk4_step(f: Deriv, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """One classic fourth-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def leapfrog_step(f: Deriv, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Kick-drift-kick. Acceleration is read from the derivative function."""
    a0 = f(t, y)[:, 3:]
    v_half = y[:, 3:] + 0.5 * h * a0
    r_new = y[:, :3] + h * v_half
    a1 = f(t + h, np.concatenate([r_new, v_half], axis=1))[:, 3:]
    v_new = v_half + 0.5 * h * a1
    return np.concatenate([r_new, v_new], axis=1)


def _perturbation(f: Deriv, t: float, y: np.ndarray) -> np.ndarray:
    """Acceleration minus the Earth's point-mass term."""
    r = y[:, :3]
    rn = np.sqrt(np.sum(r * r, axis=1))[:, None]
    return f(t, y)[:, 3:] + MU_EARTH * r / rn ** 3


def wh_step(f: Deriv, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Wisdom-Holman kick-drift-kick: half a kick of the perturbations, the
    exact two-body motion over ``h``, then the other half kick. The Kepler
    drift carries the orbit itself, so the splitting error scales with the
    perturbations (~1e-3 of gravity) instead of with gravity. With velocity
    dependent forces (drag) it is no longer exactly symplectic."""
    v_half = y[:, 3:] + 0.5 * h * _perturbation(f, t, y)
    r1, v1 = kepler_propagate(y[:, :3], v_half, h)
    v1 = v1 + 0.5 * h * _perturbation(f, t + h, np.concatenate([r1, v1], axis=1))
    return np.concatenate([r1, v1], axis=1)


def _error_norm(err, y0, y1, rtol, atol):
    """Scaled RMS error of the worst satellite (<= 1 means the step is accepted)."""
    sc = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
    per_sat = np.sqrt(np.mean((err / sc) ** 2, axis=1))
    return float(np.max(per_sat)) if per_sat.size else 0.0


class Propagator:
    """Advances ``y`` from ``t0`` to ``t1`` with the configured method,
    calling ``callback(t, y)`` after every accepted step. A callback that
    returns True stops the integration early at that step.

    With ``t_stop`` < ``t1`` the integration heads for ``t1`` but ends before
    the first step that would pass ``t_stop``. Steps are never shortened to
    land on ``t_stop``, so reaching ``t1`` through any sequence of stops takes
    exactly the steps a single call would: the step grid belongs to the
    physics, not to whoever asks for the time."""

    def __init__(self, method: str = "dopri5", rtol: float = 1e-9, atol: float = 1e-6,
                 h_max: float = 60.0, h_fixed: float = 10.0):
        if method not in METHODS:
            raise ValueError(f"unknown integrator {method!r}; choose from {METHODS}")
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.h_max = h_max
        self.h_fixed = h_fixed
        self.h = min(10.0, h_max)       # current adaptive step suggestion
        self.stats = StepStats()

    def integrate(self, f: Deriv, t0: float, y0: np.ndarray, t1: float,
                  callback: Callable[[float, np.ndarray], bool | None] | None = None,
                  t_stop: float | None = None):
        """Returns ``(t_reached, y)``; ``t_reached < t1`` only if stopped."""
        stop = t1 if t_stop is None else min(t1, t_stop)
        if t1 <= t0 or y0.size == 0:
            return max(t0, stop), y0
        if self.method == "dopri5":
            return self._dopri5(f, t0, y0, t1, callback, stop)
        step = {"rk4": rk4_step, "wh": wh_step, "leapfrog": leapfrog_step}[self.method]
        evals = 4 if self.method == "rk4" else 2
        t, y = t0, y0
        while t < t1:
            h = min(self.h_fixed, t1 - t)
            if stop < t1 and t + h > stop:
                break
            y = step(f, t, y, h)
            t = t1 if t1 - (t + h) < 1e-9 else t + h
            self.stats.accepted += 1
            self.stats.evaluations += evals
            self.stats.last_h = h
            if callback and callback(t, y):
                break
        return t, y

    def _dopri5(self, f, t0, y0, t1, callback, t_stop):
        """Adaptive Dormand-Prince; the step size carries over between calls."""
        t, y = t0, y0
        k = [None] * 7
        k[0] = f(t, y)
        self.stats.evaluations += 1
        h = min(self.h, self.h_max)
        while t1 - t > 1e-9 * max(1.0, abs(t)):
            last = h >= t1 - t
            h_try = t1 - t if last else h
            if t_stop < t1 and t + h_try > t_stop:
                break
            for s in range(1, 7):
                acc = y.copy()
                for j, aij in enumerate(_A[s]):
                    if aij:
                        acc += h_try * aij * k[j]
                k[s] = f(t + _C[s] * h_try, acc)
            y_new = acc                 # stage 7 sits at the 5th-order solution (FSAL)
            self.stats.evaluations += 6
            err_vec = h_try * sum(_E[j] * k[j] for j in range(7) if _E[j])
            err = _error_norm(err_vec, y, y_new, self.rtol, self.atol)
            if err <= 1.0 or h_try < 1e-6:
                t = t1 if last else t + h_try
                y = y_new
                k[0] = k[6]
                self.stats.accepted += 1
                self.stats.last_h = h_try
                stop = bool(callback and callback(t, y))
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * err ** -0.2))
                if not last:
                    h = min(self.h_max, h_try * fac)
                else:
                    # a short final step says nothing about the natural step size
                    h = min(self.h_max, max(h, h_try * fac))
                if stop:
                    break
            else:
                self.stats.rejected += 1
                h = h_try * max(0.2, 0.9 * err ** -0.2)
        self.h = h
        return t, y
