"""The Sentinel-1A/1C study (plan step 5): a year of ESA precise orbits, measured
and modelled, written to docs/analysis/sentinel1/.

    python tools/sentinel1_study.py series      # the long-term series, ~10 min
    python tools/sentinel1_study.py arcs        # the weekly accuracy arcs, hours; resumable
    python tools/sentinel1_study.py report      # Markdown and PNGs from the CSVs, seconds

``series`` reads every day of both satellites and writes, per satellite, the
daily mean elements and ground-track offset (``daily_S1A.csv``), the 1A-1C
phasing (``phasing.csv``) and every flagged burn sized from the jump in the
mean elements (``burns.csv``). The per-revolution tables it builds them from
stay in the data folder (``$SATFLIGHT_DATA/study/``).

``arcs`` fits a day of each satellite every week (start state, drag, an
impulse per burn; EGM2008 70x70 + drag + Sun + Moon + SRP) and flies it three
days on against the truth, one row per arc in ``arcs.csv``, written as each
finishes; a rerun skips the arcs already there. An arc the data cannot
support (the S1C outage) is a row with its reason.

``report`` needs only the committed CSVs, so it reruns anywhere.

Needs tools/fetch_poeorb.py and tools/fetch_eop.py for ``series`` and ``arcs``.
Contains modified Copernicus Sentinel data 2025-2026.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satflight import eop, fit, poeorb, study  # noqa: E402

OUT = ROOT / "docs" / "analysis" / "sentinel1"
FIRST, LAST = date(2025, 7, 1), date(2026, 6, 30)
MISSIONS = ("S1A", "S1C")
MASS = {"S1A": 2164.0, "S1C": 2200.0}       # kg, as the replay scenario uses
CHUNK_DAYS = 14
OFF_TRACK_M = 1000.0        # a day farther than this from the usual track: moved
CREDIT = "Contains modified Copernicus Sentinel data 2025-2026 (ESA POEORB precise orbits)."


# --- CSV -------------------------------------------------------------------------------------
def _fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return "" if math.isnan(x) else f"{float(x):.10g}"
    if isinstance(x, np.datetime64):
        return str(x.astype("datetime64[s]"))
    return x


def write_csv(path: Path, rows: list[dict], columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in columns])


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def column(rows, name, kind=float):
    out = []
    for r in rows:
        v = r.get(name, "")
        out.append(kind(v) if v not in ("", None) else (math.nan if kind is float else None))
    return np.array(out, dtype=float if kind is float else object)


# --- series ----------------------------------------------------------------------------------
def runs_of_days(have, first: date, last: date) -> list[tuple[date, date]]:
    """Contiguous runs of days with a file, within first..last."""
    out, start, d = [], None, first
    while d <= last + timedelta(days=1):
        if d <= last and d in have:
            start = start or d
        elif start is not None:
            out.append((start, d - timedelta(days=1)))
            start = None
        d += timedelta(days=1)
    return out


def check_orbit_numbers(mission, revs):
    """The ground-track offsets group revolutions by orbit number: each
    revolution must be numbered one more than the one before it (more across
    a data outage, by the revolutions missed)."""
    dn = np.diff(revs["orbit"])
    missed = np.round(np.diff(revs["node_utc"]) / np.timedelta64(1, "us") / 1e6
                      / revs["period_s"][:-1]).astype(int)
    bad = np.flatnonzero(dn != missed)
    if bad.size:
        k = bad[0]
        raise ValueError(f"{mission}: revolution at {revs['node_utc'][k + 1]} is numbered "
                         f"{revs['orbit'][k + 1]} after {revs['orbit'][k]}, "
                         f"{missed[k]} revolution(s) later")


def mission_revolutions(mission, first, last, table, root=None, out=print):
    """Per-revolution table over first..last, read CHUNK_DAYS at a time with a
    day's overlap (so no revolution is cut), plus each burn window (UTC)."""
    have = poeorb.local_files(mission, root)
    parts, windows, latest = [], [], None
    for a, b in runs_of_days(have, first, last):
        d = a
        while True:
            e = min(d + timedelta(days=CHUNK_DAYS), b)
            arc = poeorb.load_arc(mission, d, e, root)
            revs = study.revolutions(arc, table)
            # the overlap day's revolutions came with the chunk before
            keep = (np.ones(len(revs["node_utc"]), bool) if latest is None else
                    revs["node_utc"] > latest + np.timedelta64(60, "s"))
            if keep.any():
                latest = revs["node_utc"][keep][-1]
            parts.append({k: v[keep] for k, v in revs.items()})
            windows += [(arc.utc[0] + study._us(t0), arc.utc[0] + study._us(t1))
                        for t0, t1, _ in fit.burn_windows(arc)]
            out(f"  {mission} {d}..{e}: {keep.sum()} revolutions")
            if e == b:
                break
            d = e
    revs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(revs["node_utc"])
    return {k: v[order] for k, v in revs.items()}, merge_windows(windows)


def merge_windows(windows, slack_s: float = 10.0):
    """Burn windows (start, end) with the overlapping ones (the same burn read
    from two chunks, each cut at the chunk's edge) joined."""
    out = []
    for t0, t1 in sorted(windows):
        if out and t0 <= out[-1][1] + study._us(slack_s):
            out[-1] = (out[-1][0], max(out[-1][1], t1))
        else:
            out.append((t0, t1))
    return out


def daily_rows(revs: dict, track_m: np.ndarray) -> list[dict]:
    day = revs["node_utc"].astype("datetime64[D]")
    rows = []
    for d in np.unique(day):
        m = day == d
        clean = m & ~revs["burn"]
        ex, ey = revs["ex"][m].mean(), revs["ey"][m].mean()
        rows.append({
            "date": str(d), "revs": int(m.sum()), "burn_revs": int(revs["burn"][m].sum()),
            "gap_revs": int(revs["gap"][m].sum()),
            "a_km": revs["a_km"][clean].mean() if clean.any() else math.nan,
            "e": math.hypot(ex, ey), "argp_deg": math.degrees(math.atan2(ey, ex)),
            "i_deg": revs["i_deg"][m].mean(), "ltan_h": revs["ltan_h"][m].mean(),
            "track_rms_m": float(np.sqrt(np.mean(track_m[m] ** 2))),
            "track_max_m": float(np.abs(track_m[m]).max())})
    return rows


def run_series(first=FIRST, last=LAST, root=None, out=print, dest=OUT):
    table = eop.read()
    cache = poeorb.data_root() / "study" if root is None else Path(root) / "study"
    cache.mkdir(parents=True, exist_ok=True)
    all_revs, burns = {}, []
    for mission in MISSIONS:
        revs, windows = mission_revolutions(mission, first, last, table, root, out)
        check_orbit_numbers(mission, revs)
        np.savez(cache / f"revs_{mission}.npz", **revs)
        all_revs[mission] = revs
        track = study.track_offsets(revs["orbit"], revs["lon_deg"])
        write_csv(dest / f"daily_{mission}.csv", daily_rows(revs, track))
        for row in study.burn_sizes(revs, windows):
            burns.append({"mission": mission, "kind": "burn", **row})
        for row in study.burn_sizes(revs, study.placebo_windows(windows)):
            burns.append({"mission": mission, "kind": "placebo", **row})
        out(f"{mission}: {len(revs['orbit'])} revolutions, {len(windows)} burns")
    write_csv(dest / "burns.csv", burns)
    t, ph = study.phasing(all_revs["S1A"], all_revs["S1C"])
    day = t.astype("datetime64[D]")
    rows = [{"date": str(d), "mean_deg": ph[day == d].mean(), "min_deg": ph[day == d].min(),
             "max_deg": ph[day == d].max()} for d in np.unique(day)]
    write_csv(dest / "phasing.csv", rows)
    return all_revs


# --- arcs ------------------------------------------------------------------------------------
ARC_COLUMNS = ["mission", "start", "fit_rms_m", "fit_r_m", "fit_s_m", "fit_w_m", "cd_am",
               "density_scale", "burns_fit", "burns_check", "iterations",
               "fit_hours", "check_hours",
               "day1_r_m", "day1_s_m", "day1_w_m", "day2_r_m", "day2_s_m", "day2_w_m",
               "day3_r_m", "day3_s_m", "day3_w_m", "at24h_s_m", "at48h_s_m", "at72h_s_m",
               "seconds", "error"]


_WORKER = {}


def _arc_row(m, d, degree, one=None, atmosphere="table", cd_model="fixed", iers=False):
    """One arc's row; what the data cannot support is a row with the reason."""
    try:
        if one is not None:
            return one(m, d)
        key = (degree, atmosphere, cd_model, iers)
        if key not in _WORKER:                             # once per process
            from satflight.forces import ForceModel
            _WORKER[key] = (ForceModel(
                gravity_degree=degree, drag=True, sun=True, moon=True, srp=True,
                atmosphere=atmosphere, cd_model=cd_model, tides=iers, relativity=iers),
                           eop.read())
        model, table = _WORKER[key]
        return study.accuracy_arc(m, d, model, MASS[m], eop_table=table)
    except (poeorb.PoeorbError, fit.FitError) as exc:
        return {"mission": m, "start": d.isoformat(), "error": str(exc)}


CD_FILE_TAG = {"fixed": "", "dria_sphere": "_dria"}


def arcs_file(atmosphere="table", cd_model="fixed", iers=False):
    """The arcs' file name: arcs.csv for the table, arcs_<atmosphere>.csv
    otherwise, with "_dria" for the physical drag coefficient and "_iers" for
    the solid-Earth tides and relativity."""
    base = "arcs" if atmosphere == "table" else f"arcs_{atmosphere}"
    return f"{base}{CD_FILE_TAG[cd_model]}{'_iers' if iers else ''}.csv"


def run_arcs(first=FIRST, last=LAST, degree=70, limit=None, missions=MISSIONS, out=print,
             dest=OUT, one=None, workers=1, atmosphere="table", only=None,
             cd_model="fixed", iers=False):
    """Every weekly arc not already in ``dest``/arcs.csv, ``workers`` at a time
    (processes), each row written as it finishes; ``one`` (for tests) replaces
    :func:`study.accuracy_arc` and runs in this process. ``atmosphere`` "msis"
    flies them with NRLMSIS into arcs_msis.csv, and ``cd_model``
    "dria_sphere" with its physical Cd into arcs_msis_dria.csv; ``only``
    [(mission, day)] runs just those (ValueError for one that is not a weekly
    arc, or for a Cd model the atmosphere cannot support). ``iers`` adds the
    solid-Earth tides and relativity (into arcs_..._iers.csv)."""
    from satflight.forces import ForceModel
    ForceModel(drag=True, atmosphere=atmosphere, cd_model=cd_model)   # refuse a bad pair now
    path = dest / arcs_file(atmosphere, cd_model, iers)
    done = {(r["mission"], r["start"]) for r in read_csv(path)} if path.exists() else set()
    if not path.exists():
        write_csv(path, [], ARC_COLUMNS)
    starts = study.arc_starts(first, last)
    for m, d in only or ():
        if d not in starts or m not in missions:
            raise ValueError(f"{m} {d.isoformat()} is not a weekly arc of the study "
                             f"({starts[0]}, {starts[1]}, ... for {', '.join(missions)})")
    todo = [(m, d) for d in starts for m in missions
            if (m, d.isoformat()) not in done and (not only or (m, d) in only)]
    todo = todo[:limit] if limit else todo
    pool = None
    if one is not None or workers <= 1:
        rows = (_arc_row(m, d, degree, one, atmosphere, cd_model, iers) for m, d in todo)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(_arc_row, m, d, degree, None, atmosphere, cd_model, iers)
                   for m, d in todo]
        rows = (f.result() for f in as_completed(futures))
    try:
        _write_arc_rows(path, rows, len(todo), out)
    finally:
        if pool is not None:     # an unexpected error: stop, without hours of waiting
            pool.shutdown(wait=False, cancel_futures=True)
    write_csv(path, read_arcs(dest, atmosphere, cd_model, iers), ARC_COLUMNS)   # out of order: sort


def _write_arc_rows(path, rows, total, out):
    for n, row in enumerate(rows):
        m, d = row["mission"], row["start"]
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow([_fmt(row.get(c, "")) for c in
                                                         ARC_COLUMNS])
        msg = row.get("error") or (f"fit {row['fit_rms_m']:.2f} m, day 1/2/3 along-track "
                                   f"{row['day1_s_m']:.0f}/{row['day2_s_m']:.0f}/"
                                   f"{row['day3_s_m']:.0f} m, {row['seconds']:.0f} s")
        out(f"[{n + 1}/{total}] {m} {d}: {msg}")


def read_arcs(dest=OUT, atmosphere="table", cd_model="fixed", iers=False):
    """``dest``/arcs.csv (or the ``atmosphere``'s and ``cd_model``'s file) in
    date order (then satellite), [] if there is none."""
    path = dest / arcs_file(atmosphere, cd_model, iers)
    rows = read_csv(path) if path.exists() else []
    return sorted(rows, key=lambda r: (r["start"], MISSIONS.index(r["mission"])))


# --- report ----------------------------------------------------------------------------------
def _dates(rows, name="date"):
    return np.array([np.datetime64(r[name]) for r in rows], "datetime64[D]")


def _gapped(t, y):
    """Dates and values with a NaN in each gap of more than a day, so a line
    plot breaks over a data outage instead of bridging it."""
    t, y = np.asarray(t), np.asarray(y, float)
    k = np.flatnonzero(np.diff(t) > np.timedelta64(1, "D")) + 1
    return np.insert(t, k, t[k - 1] + np.timedelta64(1, "D")), np.insert(y, k, np.nan)


def _pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else math.nan


def burn_days(burns, mission):
    """The days ``mission`` burned; the placebo windows are where nothing did."""
    return [np.datetime64(r["start_utc"][:10]) for r in burns
            if r["mission"] == mission and r["kind"] == "burn"]


def _figures(dest, daily, phasing, burns, arcs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"S1A": "#2a6fdb", "S1C": "#d9622b"}
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(dest / name, dpi=110)
        plt.close(fig)
        made.append(name)

    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    a_all = []
    for m, rows in daily.items():
        t = _dates(rows)
        kw = {"color": colours[m], "label": f"Sentinel-1{m[-1]}", "lw": 1}
        a = column(rows, "a_km") - 7000.0
        a_all.append(a)
        ax[0].plot(*_gapped(t, a), **kw)
        ax[1].plot(*_gapped(t, column(rows, "i_deg")), **kw)
        ax[2].plot(*_gapped(t, (column(rows, "ltan_h") - 18.0) * 60.0), **kw)
        ax[3].plot(*_gapped(t, column(rows, "e") * 1e3), **kw)
    # the station-kept band; a satellite moved off it (1C, June 2026) runs off scale
    a_all = np.concatenate(a_all)
    lo, hi = np.nanpercentile(a_all, [1, 99])
    pad = 0.6 * (hi - lo)
    ax[0].set_ylim(lo - pad, hi + pad)
    if np.nanmax(a_all) > hi + pad:
        ax[0].text(0.01, 0.92, f"off scale: up to {np.nanmax(a_all):+.2f}",
                   transform=ax[0].transAxes, fontsize=8)
    for m in daily:
        bt = burn_days(burns, m)
        ax[0].plot(bt, np.full(len(bt), ax[0].get_ylim()[0]), "|", color=colours[m], ms=8)
    ax[0].set_ylabel("mean a - 7000 km")
    ax[1].set_ylabel("mean inclination (deg)")
    ax[2].set_ylabel("LTAN - 18:00 (min)")
    ax[3].set_ylabel("eccentricity (x1e-3)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Revolution-mean elements, daily (burns marked on the a axis)")
    save(fig, "elements.png")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for m, rows in daily.items():
        y = column(rows, "track_rms_m")
        y[column(rows, "track_max_m") > OFF_TRACK_M] = np.nan
        ax[0].plot(*_gapped(_dates(rows), y), color=colours[m], lw=1,
                   label=f"Sentinel-1{m[-1]}")
    ax[0].set_ylabel("node offset RMS (m)")
    ax[0].set_title("Ground track: daily RMS east-west offset of the equator crossings "
                    f"(days within {OFF_TRACK_M / 1e3:g} km)", fontsize=10)
    ax[0].legend(loc="upper right")
    if phasing:
        ax[1].plot(*_gapped(_dates(phasing), column(phasing, "mean_deg")), color="k", lw=1)
    ax[1].set_ylabel("1C behind 1A (deg)")
    ax[1].set_title("Phasing")
    save(fig, "track_phasing.png")

    ok = [r for r in arcs if not r.get("error")]
    if ok:
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for m in MISSIONS:
            rows = [r for r in ok if r["mission"] == m]
            if not rows:
                continue
            t = _dates(rows, "start")
            for d, mk in ((1, "o"), (3, "^")):
                ax[0].semilogy(t, column(rows, f"day{d}_s_m"), mk, color=colours[m], ms=4,
                               label=f"1{m[-1]} day {d}")
            ax[1].plot(t, column(rows, "day1_r_m"), "o", color=colours[m], ms=4,
                       label=f"1{m[-1]} radial")
            ax[1].plot(t, column(rows, "day1_w_m"), "x", color=colours[m], ms=4,
                       label=f"1{m[-1]} cross-track")
            ax[2].plot(t, column(rows, "density_scale"), "o-", color=colours[m], ms=3, lw=0.8,
                       label=f"Sentinel-1{m[-1]}")
        ax[0].set_ylabel("along-track RMS (m)")
        ax[0].set_title("Prediction error after a one-day fit")
        ax[0].legend(ncol=4, fontsize=8)
        ax[1].set_ylabel("day-1 RMS (m)")
        ax[1].legend(ncol=4, fontsize=8)
        ax[2].set_ylabel("fitted density scale")
        ax[2].set_title("Drag the fit needs, as a scale on the static table (Cd 2.2)")
        save(fig, "accuracy.png")
    return made


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_report(dest=OUT, figures=True) -> Path:
    daily = {m: read_csv(dest / f"daily_{m}.csv") for m in MISSIONS
             if (dest / f"daily_{m}.csv").exists()}
    if not daily:
        raise FileNotFoundError(f"no daily_*.csv in {dest}: run the series first")
    phasing = read_csv(dest / "phasing.csv") if (dest / "phasing.csv").exists() else []
    burns = read_csv(dest / "burns.csv") if (dest / "burns.csv").exists() else []
    arcs = read_arcs(dest)
    made = _figures(dest, daily, phasing, burns, arcs) if figures else []

    lines = ["# Sentinel-1A and 1C: a year of precise orbits", "",
             f"Generated by `tools/sentinel1_study.py report` from the CSVs beside it. {CREDIT}",
             "", "## Long-term series", ""]
    rows, moved = [], []
    for m, d in daily.items():
        a = column(d, "a_km")
        span = f"{d[0]['date']} .. {d[-1]['date']}"
        # offsets are from the year's usual crossing of each relative orbit: a
        # satellite moved to another ground track is reported, not averaged in
        off = column(d, "track_max_m") > OFF_TRACK_M
        on = ~off
        if off.any():
            k = np.flatnonzero(off)
            moved.append(f"- Sentinel-1{m[-1]} is more than {OFF_TRACK_M / 1e3:g} km off its "
                         f"usual track on {k.size} days, {d[k[0]]['date']} .. "
                         f"{d[k[-1]]['date']} (up to "
                         f"{np.nanmax(column(d, 'track_max_m')) / 1e3:.0f} km).")
        rows.append([f"Sentinel-1{m[-1]}", span, len(d),
                     f"{np.nanmean(a):.3f}", f"{np.nanmin(a):.3f} .. {np.nanmax(a):.3f}",
                     f"{np.mean(column(d, 'i_deg')):.4f}",
                     f"{(np.mean(column(d, 'ltan_h')) - 18) * 60:+.1f}",
                     f"{np.mean(column(d, 'e')) * 1e3:.3f}",
                     f"{np.mean(column(d, 'argp_deg')):.1f}",
                     f"{_pct(column(d, 'track_rms_m')[on], 50):.0f} / "
                     f"{_pct(column(d, 'track_max_m')[on], 100):.0f}"])
    lines += [_table(["satellite", "span", "days", "mean a (km)", "a range (km)", "i (deg)",
                      "LTAN - 18:00 (min)", "e (1e-3)", "argp (deg)",
                      "track offset median RMS / max (m)"], rows), ""]
    lines += [f"Track offset: how far east or west each equator crossing lands from its "
              f"relative orbit's usual (median) crossing, on the days within "
              f"{OFF_TRACK_M / 1e3:g} km of it.", ""] + moved + ([""] if moved else [])
    if phasing:
        ph = column(phasing, "mean_deg")
        lines += [f"1C flies {np.mean(ph):.3f} deg behind 1A on average "
                  f"(daily means {np.min(ph):.3f} .. {np.max(ph):.3f}; 180 nominal).", ""]
    if burns:
        lines += ["## Burns", "",
                  "The DEGRADED-MANOEUVRE windows, those under 3 h apart taken as one "
                  "manoeuvre, each sized from the step in the revolution means across it "
                  "(fitted with the drag's decay and the daily wobble; `burns.csv`). "
                  "Cross-track is the least the change in inclination needs.", ""]
        rows, noise = [], []
        for m in daily:
            b = [r for r in burns if r["mission"] == m and r.get("kind", "burn") == "burn"]
            pl = [r for r in burns if r["mission"] == m and r.get("kind") == "placebo"]
            if pl:
                # robust spread (1.48 x median |x|): a satellite drifting to a new
                # slot (1C, June 2026) is not noise
                rms = {c: 1.4826 * float(np.nanmedian(np.abs(column(pl, c))))
                       for c in ("da_m", "di_mdeg", "dv_along_mm_s", "dv_cross_mm_s")}
                noise.append(f"- Sentinel-1{m[-1]}: {len(pl)} placebo windows (midway "
                             f"between manoeuvres, where nothing burned) size to a spread of "
                             f"{rms['da_m']:.1f} m in a ({rms['dv_along_mm_s']:.2f} mm/s "
                             f"along-track) and {rms['di_mdeg']:.2f} mdeg in i "
                             f"({rms['dv_cross_mm_s']:.1f} mm/s cross-track): the method's "
                             f"noise.")
            along = column(b, "dv_along_mm_s")
            cross = column(b, "dv_cross_mm_s")
            sized = np.isfinite(along)
            days = len(daily[m])
            win = column(b, "windows")
            rows.append([f"Sentinel-1{m[-1]}", len(b),
                         int(np.where(np.isfinite(win), win, 1).sum()),
                         f"{days / max(len(b), 1):.1f}",
                         int(sized.sum()), f"{_pct(np.abs(along), 50):.2f}",
                         f"{np.nansum(np.abs(along)):.0f}", f"{_pct(cross, 50):.2f}",
                         f"{np.nansum(cross):.0f}"])
        lines += [_table(["satellite", "manoeuvres", "burns", "days apart", "sized",
                          "median along (mm/s)",
                          "year along (mm/s)", "median cross (mm/s)", "year cross (mm/s)"],
                         rows), ""] + noise + ([""] if noise else []) + [
            "So the series sizes the along-track part of a burn (a few mm/s, against "
            "about 2 mm/s of noise), but not the cross-track part: Sentinel-1's "
            "inclination trims are a few mm/s and the mean inclination's own wobble is "
            "worth about 30. The accuracy arcs' fitted impulses resolve both.", ""]
    ok = [r for r in arcs if not r.get("error")]
    if ok:
        lines += ["## Accuracy arcs", "",
                  "Each week, a day of each satellite is fitted (start state, drag as a "
                  "density scale, an impulse per burn; Cr held at 1.3; EGM2008 70x70, "
                  "drag, Sun, Moon, SRP) and flown three days on against the truth, with "
                  "only the later burns fitted. RMS in metres, median (90th percentile) "
                  "over the arcs:", ""]
        rows = []
        for m in MISSIONS:
            rr = [r for r in ok if r["mission"] == m]
            if not rr:
                continue
            for lab, key in (("fit day", "fit"), ("day 1", "day1"), ("day 2", "day2"),
                             ("day 3", "day3")):
                rows.append([f"Sentinel-1{m[-1]}", lab] + [
                    f"{_pct(column(rr, f'{key}_{c}_m'), 50):.1f} "
                    f"({_pct(column(rr, f'{key}_{c}_m'), 90):.1f})" for c in "rsw"])
        lines += [_table(["satellite", "span", "radial", "along-track", "cross-track"], rows),
                  ""]
        for m in MISSIONS:
            rr = [r for r in ok if r["mission"] == m]
            if rr:
                sc = column(rr, "density_scale")
                lines.append(f"- Sentinel-1{m[-1]}: {len(rr)} arcs; fitted density scale "
                             f"{np.mean(sc):.2f} mean ({np.min(sc):.2f} .. {np.max(sc):.2f}), "
                             f"so the year's mean Cd at 10 m^2 on the static table is "
                             f"{2.2 * np.mean(sc):.2f}.")
        short_fit = column(ok, "fit_hours")[column(ok, "fit_hours") < 24.0]
        short_chk = column(ok, "check_hours")[column(ok, "check_hours") < 72.0]
        if short_fit.size or short_chk.size:
            lines.append(f"- Spans cut at a burn (an impulse needs half an hour of data after "
                         f"its window): {short_fit.size} fits shorter than a day"
                         + (f" (shortest {short_fit.min():.1f} h)" if short_fit.size else "")
                         + f", {short_chk.size} checks shorter than three days"
                         + (f" (shortest {short_chk.min():.1f} h)" if short_chk.size else "")
                         + ".")
        failed = [r for r in arcs if r.get("error")]
        for r in failed:
            lines.append(f"- {r['mission']} {r['start']}: not run ({r['error']})")
        lines.append("")
    if made:
        lines += ["## Figures", ""] + [f"![{n}]({n})" for n in made] + [""]
    path = dest / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _mission_day(text):
    m, _, d = text.partition(":")
    try:
        return m, date.fromisoformat(d)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected MISSION:YYYY-MM-DD") from None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["series", "arcs", "report"])
    p.add_argument("--limit", type=int, help="arcs: run at most this many")
    p.add_argument("--degree", type=int, default=70, help="arcs: EGM2008 degree (default 70)")
    p.add_argument("--workers", type=int, default=1, help="arcs: processes at once (default 1)")
    p.add_argument("--atmosphere", choices=["table", "msis"], default="table",
                   help="arcs: the drag's atmosphere; msis writes arcs_msis.csv")
    p.add_argument("--cd-model", choices=["fixed", "dria_sphere"], default="fixed",
                   help="arcs: the drag coefficient; dria_sphere (needs --atmosphere msis) "
                        "writes arcs_msis_dria.csv")
    p.add_argument("--iers", action="store_true",
                   help="arcs: add the solid-Earth tides and relativity (writes ..._iers.csv)")
    p.add_argument("--only", action="append", type=_mission_day, metavar="MISSION:DAY",
                   help="arcs: just this arc, e.g. S1A:2026-01-20 (repeatable)")
    a = p.parse_args(argv)
    try:
        if a.command == "series":
            run_series()
        elif a.command == "arcs":
            run_arcs(degree=a.degree, limit=a.limit, workers=a.workers,
                     atmosphere=a.atmosphere, only=a.only, cd_model=a.cd_model,
                     iers=a.iers)
        else:
            print(run_report())
    except (poeorb.PoeorbError, eop.EopError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
