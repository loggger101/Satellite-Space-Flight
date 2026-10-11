"""Strict reading of scenario dicts: a key the program does not know is an
error, never skipped. A typo such as ``"inclination"`` for ``"i"`` would
otherwise fly the default (0 deg) without a word.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, fields


def check_keys(d, allowed, where: str, required=()):
    """Raise ValueError if ``d`` is not a dict, has a key outside ``allowed``,
    or lacks one of ``required``. ``where`` names the place for the message
    (e.g. "satellite 'ISS' orbit")."""
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object, got {type(d).__name__}")
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key{'s' if len(unknown) > 1 else ''} "
                         f"{', '.join(map(repr, unknown))} "
                         f"(known: {', '.join(sorted(allowed))})")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(map(repr, missing))}")


def build(cls, d, where: str):
    """Dataclass ``cls`` from ``d``; unknown or missing required keys raise ValueError."""
    fs = fields(cls)
    required = [f.name for f in fs if f.default is MISSING and f.default_factory is MISSING]
    check_keys(d, [f.name for f in fs], where, required)
    return cls(**d)


def finite(x, what: str) -> float:
    """``x`` as a float; ValueError if it is not a finite number."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {x!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"{what} must be a finite number, got {x!r}")
    return f


def within(x, lo: float, hi: float, what: str) -> float:
    """``x`` as a float; ValueError unless lo <= x <= hi."""
    f = finite(x, what)
    if not lo <= f <= hi:
        raise ValueError(f"{what} must be between {lo:,.6g} and {hi:,.6g}, got {f:,.6g}")
    return f


def positive(x, what: str) -> float:
    """``x`` as a float; ValueError unless it is finite and > 0."""
    f = finite(x, what)
    if f <= 0:
        raise ValueError(f"{what} must be positive, got {f:,.6g}")
    return f


def whole(x, lo: int, hi: int, what: str) -> int:
    """``x`` as an int; ValueError unless it is a whole number in [lo, hi]."""
    f = finite(x, what)
    if f != int(f) or not lo <= f <= hi:
        raise ValueError(f"{what} must be a whole number from {lo:,} to {hi:,}, got {x!r}")
    return int(f)


MAX_SPAN = 100 * 365.25 * 86400.0   # s; the longest wait (delay) anything may ask for

# drag properties a body may have: 1 g to 10,000 t, Cd*A/m up to 100 m^2/kg (the
# drag_decay balloon has 1.5); beyond that the drag force swamps any integrator
MAX_CDA_OVER_M = 100.0


def drag_props(mass, area, cd, what: str):
    """Raise ValueError unless mass (kg), area (m^2) and Cd describe a body the
    drag model can fly."""
    within(mass, 1e-3, 1e7, f"{what} mass (kg)")
    within(area, 0.0, 1e5, f"{what} area (m^2)")
    within(cd, 0.0, 10.0, f"{what} drag coefficient")
    if cd * area / mass > MAX_CDA_OVER_M:
        raise ValueError(f"{what}: Cd*A/m = {cd * area / mass:,.4g} m^2/kg is beyond "
                         f"{MAX_CDA_OVER_M:g} (too light for its area)")


def one_of(value, choices, what: str):
    """``value`` if it is one of ``choices``, else ValueError listing them."""
    if value not in choices:
        raise ValueError(f"unknown {what} {value!r} (choose from {', '.join(map(str, choices))})")
    return value
