"""Ground-track keeping: the controller on its own (synthetic crossings), then
flown in the simulator."""
import math
from pathlib import Path

import numpy as np
import pytest

from satflight.constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from satflight.keeping import TrackKeeper, TrackKeeping

A = 7071.0                                   # km, Sentinel-1
P = 2 * math.pi * math.sqrt(A ** 3 / MU_EARTH)
SHIFT = 360.0 * 12 / 175                     # deg west per orbit on a 175/12 repeat
KM_PER_DEG = R_EARTH * math.pi / 180.0
ROOT = Path(__file__).resolve().parents[1]


I_SSO = math.radians(98.18)


def state():
    """A circular sun-synchronous state at A, at its ascending node."""
    v = math.sqrt(MU_EARTH / A)
    return np.array([A, 0.0, 0.0]), np.array([0.0, v * math.cos(I_SSO), v * math.sin(I_SSO)])


def rate_change_km_day(da_km):
    """How much a change ``da_km`` in a turns the crossings' east-west drift
    (km/day), by finite differences of the period: a longer orbit lets the Earth
    turn further under it, so its crossings move west."""
    p1 = 2 * math.pi * math.sqrt((A + da_km) ** 3 / MU_EARTH)
    rel = OMEGA_EARTH - 2 * math.pi / (365.2421897 * 86400.0)   # sun-synchronous node
    return -rel * (p1 - P) * R_EARTH * 86400.0 / P


def fly(keeper, days, acc_km_day2, e0=0.0, rate0=0.0, lon0=-40.0):
    """Feed ``keeper`` the crossings of a track whose offset accelerates east at
    ``acc_km_day2`` (drag), burns turning its drift by rate_change_km_day of the
    a change each along-track burn makes (dv = v da / 2a). Returns the
    (t days, offset km) of every crossing and the burns (t, dv km/s)."""
    v = math.sqrt(MU_EARTH / A)
    e, rate, t, out, burns, due = e0, rate0, 0.0, [], [], None
    r, vv = state()
    n = 0
    while t < days * 86400.0:
        lon = (lon0 - n * SHIFT + e / KM_PER_DEG + 180.0) % 360.0 - 180.0
        out.append((t / 86400.0, e))
        got = keeper.crossing(t, lon, r, vv)
        if got is not None:
            due = got
            burns.append(got)
        dt = P / 86400.0
        if due is not None and due[0] < t + P:
            da = 2 * A * due[1] / v
            # the burn half an orbit on: half the step at each drift rate
            e += rate * dt / 2 + acc_km_day2 * (dt / 2) ** 2 / 2
            rate += acc_km_day2 * dt / 2 + rate_change_km_day(da)
            e += rate * dt / 2 + acc_km_day2 * (dt / 2) ** 2 / 2
            rate += acc_km_day2 * dt / 2
            due = None
        else:
            e += rate * dt + acc_km_day2 * dt * dt / 2
            rate += acc_km_day2 * dt
        t += P
        n += 1
    return np.array(out), burns


def test_crossings_on_the_grid_need_no_burn():
    k = TrackKeeper(TrackKeeping("S"))
    track, burns = fly(k, 6, 0.0)
    assert burns == []
    assert np.abs(np.array(k.offsets)[:, 1]).max() < 1e-6


def test_the_reference_is_the_first_crossing_after_the_start():
    k = TrackKeeper(TrackKeeping("S", start=3 * P + 1.0))
    fly(k, 1, 0.0, e0=0.05)
    assert k.offsets[0][0] == pytest.approx(4 * P)
    assert np.abs(np.array(k.offsets)[:, 1]).max() < 1e-6     # measured from there


def test_it_waits_for_a_day_of_crossings_before_deciding():
    k = TrackKeeper(TrackKeeping("S", band_km=0.1))
    _, burns = fly(k, 0.9, 0.0, rate0=1.0)       # 1 km/day: out of the band in hours
    assert burns == []
    _, burns = fly(TrackKeeper(TrackKeeping("S", band_km=0.1)), 1.2, 0.0, rate0=1.0)
    assert len(burns) == 1


def test_drag_drift_is_held_in_the_band():
    acc = 0.010                                   # km/day^2 east, as drag gives Sentinel-1
    k = TrackKeeper(TrackKeeping("S", band_km=0.1))
    track, burns = fly(k, 40, acc)
    after = track[track[:, 0] > 1.5]
    assert np.abs(after[:, 1]).max() < 0.1 * 1.05
    assert len(burns) >= 3
    gaps = np.diff([b[0] for b in burns]) / 86400.0
    assert 2.0 < np.median(gaps) < 15.0
    dv = np.array([b[1] for b in burns[1:]])
    assert np.all(dv > 0)                         # raising: drag lowers it
    assert 0.5e-6 < np.median(dv) < 10e-6         # mm/s, as ESA's burns


def test_a_burn_turns_the_drift_so_it_peaks_at_the_far_edge():
    acc = 0.010
    k = TrackKeeper(TrackKeeping("S", band_km=0.1))
    track, burns = fly(k, 12, acc)
    t1 = burns[0][0] / 86400.0
    t2 = burns[1][0] / 86400.0 if len(burns) > 1 else track[-1, 0]
    arc = track[(track[:, 0] > t1 + P / 86400.0) & (track[:, 0] < t2)]
    assert arc[:, 1].min() == pytest.approx(-0.1, abs=0.02)


def test_the_sensitivity_is_the_period_s():
    # the controller's d(drift)/da against the finite difference above
    k = TrackKeeper(TrackKeeping("S"))
    r, v = state()
    assert k.drift_per_km(r, v) == pytest.approx(rate_change_km_day(1e-3) / 1e-3, rel=1e-3)


@pytest.mark.parametrize("bad", [dict(band_km=0.0), dict(orbits=12, days=12),
                                 dict(days=0), dict(isp=0.0), dict(start=-1.0)])
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        TrackKeeping("S", **bad).check()


# --- flown in the simulator --------------------------------------------------------

def s1_pair(keep=True):
    """The default scenario's Sentinel-1 pair, as built (keeping on or off)."""
    from satflight.scenario import Scenario
    sc = Scenario.load(ROOT / "scenarios" / "default.json")
    if not keep:
        sc.track_keeping = []
    return sc


@pytest.fixture(scope="module")
def kept_run():
    from satflight.simulation import Simulation
    sim = Simulation(s1_pair())
    sim.advance(9 * 86400.0)
    return sim


def test_the_default_pair_keeps_its_tracks(kept_run):
    sim = kept_run
    assert set(sim.keepers) == {"Sentinel-1A", "Sentinel-1C"}
    for name, k in sim.keepers.items():
        off = np.array(k.offsets)
        assert len(off) > 120                         # every crossing of 9 days
        late = off[off[:, 0] > 1.5 * 86400.0, 1]
        assert np.abs(late).max() < 0.05 * 1.1, name
        # flown as maneuvers: the propellant ledger holds exactly the keeper's burns
        done = [dv for t, dv in k.burns if t <= sim.t]
        assert len(done) >= 1, name
        assert sim.sats[sim.index_of(name)].dv_used == pytest.approx(sum(done), rel=1e-9)
    assert any(e.kind == "maneuver" and ": executed track keeping - dV " in e.text
               for e in sim.events)          # (the log keeps its latest lines only)


def test_without_keeping_the_tracks_drift_off():
    # the same pair with a band too wide ever to burn: drag walks its crossings
    # well out of 50 m
    from satflight.simulation import Simulation
    sc = s1_pair()
    for k in sc.track_keeping:
        k.band_km = 1000.0
    sim = Simulation(sc)
    sim.advance(9 * 86400.0)
    for name, k in sim.keepers.items():
        assert k.burns == []
        assert np.abs(np.array(k.offsets)[:, 1]).max() > 0.1, name


def test_track_keeping_round_trips_and_refuses_strangers(tmp_path):
    from satflight.scenario import Scenario
    from satflight.simulation import Simulation
    sc = s1_pair()
    assert [k.band_km for k in sc.track_keeping] == [0.05, 0.05]
    sc.save(tmp_path / "s.json")
    again = Scenario.load(tmp_path / "s.json")
    assert again.track_keeping == sc.track_keeping
    sc.track_keeping.append(TrackKeeping("Sentinel-1B"))
    sim = Simulation(sc)                        # not in the scenario: dropped, and said so
    assert set(sim.keepers) == {"Sentinel-1A", "Sentinel-1C"}
    assert any(e.kind == "warn" and e.text == "Sentinel-1B: track keeping dropped "
               "(not in the scenario)" for e in sim.events)
    sc.track_keeping.append(TrackKeeping("Sentinel-1A"))
    with pytest.raises(ValueError, match="twice"):
        Simulation(sc)
    plain = Scenario.load(ROOT / "scenarios" / "hohmann_to_geo.json")
    assert "track_keeping" not in plain.to_dict()          # absent unless used


def test_a_saved_run_keeps_its_keepers(kept_run):
    snap = kept_run.snapshot_scenario()
    assert [k.sat for k in snap.track_keeping] == ["Sentinel-1A", "Sentinel-1C"]
    assert [k.start for k in snap.track_keeping] == [0.0, 0.0]


def test_a_crossing_missed_does_not_shift_the_grid():
    # every crossing fed but the 40th: the offsets stay on the same tracks
    k = TrackKeeper(TrackKeeping("S", band_km=1000.0))
    r, v = state()
    for n in range(80):
        if n != 40:
            k.crossing(n * P, (-40.0 - n * SHIFT + 180.0) % 360.0 - 180.0, r, v)
    assert np.abs(np.array(k.offsets)[:, 1]).max() < 1e-6
