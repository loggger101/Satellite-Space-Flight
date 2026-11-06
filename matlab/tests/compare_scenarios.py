"""Check that pairs of scenario files describe the same scenario.

    python compare_scenarios.py original1.json copy1.json [original2.json copy2.json ...]

Each file is read with the simulator's strict loader and the two ``to_dict()``
forms must be equal. Used by the MATLAB tests to prove that a scenario read
and written back by the MATLAB toolkit is unchanged. Exit status 1 names every
pair that differs and the first keys where it does.
"""

import sys

from satflight.scenario import Scenario


def differences(a, b, where="scenario"):
    """Paths where the JSON-like values ``a`` and ``b`` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{where}.{k}: only in one file" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += differences(a[k], b[k], f"{where}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} entries vs {len(b)}"]
        pairs = enumerate(zip(a, b, strict=True))
        return [d for k, (x, y) in pairs for d in differences(x, y, f"{where}[{k}]")]
    return [] if a == b else [f"{where}: {a!r} vs {b!r}"]


def main(argv):
    if not argv or len(argv) % 2:
        sys.exit("give pairs of scenario files")
    bad = 0
    for first, second in zip(argv[::2], argv[1::2], strict=True):
        diff = differences(Scenario.load(first).to_dict(), Scenario.load(second).to_dict())
        if diff:
            bad += 1
            print(f"DIFFERENT {first} vs {second}:")
            for line in diff[:10]:
                print("   ", line)
        else:
            print(f"same      {first}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
