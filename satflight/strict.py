"""Strict reading of scenario dicts: a key the program does not know is an
error, never skipped. A typo such as ``"inclination"`` for ``"i"`` would
otherwise fly the default (0 deg) without a word.
"""

from __future__ import annotations

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


def one_of(value, choices, what: str):
    """``value`` if it is one of ``choices``, else ValueError listing them."""
    if value not in choices:
        raise ValueError(f"unknown {what} {value!r} (choose from {', '.join(map(str, choices))})")
    return value
