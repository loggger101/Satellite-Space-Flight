"""Events are stamped where they happen, not at the end of the step (audit A10)."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from satflight.constants import MU_EARTH, R_EARTH
from satflight.eclipse import shadow_fraction
from satflight.elements import coe2rv, kepler_propagate, mean_to_true
from satflight.ephemeris import sun_position
from satflight.forces import ForceModel
from satflight.frames import eci_to_ecef
from satflight.scenario import DEFAULT_STATIONS, SatSpec, Scenario
from satflight.simulation import DECAYED, Simulation

EPOCH = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def _sim():
    return Simulation(Scenario(
        name="t", epoch=EPOCH, forces=ForceModel(j2=True), stations=list(DEFAULT_STATIONS),
        satellites=[SatSpec("A", {"type": "elements", "altitude": 420, "i": 51.6, "nu": 40}),
                    SatSpec("B", {"type": "elements", "perigee_alt": 300, "apogee_alt": 3000,
                                  "i": 63, "argp": 20, "nu": 10})]))


def test_eclipse_and_pass_events_straddle_the_real_crossing():
    sim = _sim()
    sim.advance(4 * 3600.0)
    ref = _sim()                                 # a twin to look up states at any time
    events = [e for e in sim.events if e.kind in ("eclipse", "station") and e.t > 1.0]
    assert len(events) > 10
    stations = {st.name: st for st in ref.stations}
    for e in events:
        name = e.text.split()[1] if e.kind == "station" else e.text.split()[0]
        side = []
        for dt in (-0.5, 0.5):
            r = ref.predict(name, e.t + dt)[:3]
            if e.kind == "eclipse":
                side.append(float(np.ravel(shadow_fraction(r, sun_position(ref.jd(e.t))))[0])
                            >= 0.5)
            else:
                st = stations[e.text.split("@ ")[1]]
                el = st.look_angles(eci_to_ecef(r, ref.gmst(e.t + dt)))[1]
                side.append(float(el) >= math.radians(st.min_el))
        assert side[0] != side[1], (e.t, e.text)    # it changed within a second


def test_a_dip_below_the_reentry_height_between_two_steps_is_caught():
    rp = R_EARTH + 79.7                            # equator: height = r - R_eq here
    ra = R_EARTH + 2000.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    n = math.sqrt(MU_EARTH / a ** 3)
    t_peri = 630.0                                  # mid-step for 60 s steps from t = 0
    nu0 = float(mean_to_true(-n * t_peri, e))
    r, v = coe2rv(a, e, 0.0, 0.0, 0.0, nu0)
    sim = Simulation(Scenario(name="t", epoch=EPOCH, forces=ForceModel(j2=False),
                              propagator="kepler", satellites=[SatSpec(
                                  "D", {"type": "state", "r": list(r), "v": list(v)})]))
    sim.advance(700.0)
    assert sim.sats[0].status == DECAYED
    lo, hi = 600.0, t_peri                          # the descent through 80 km
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rm = np.linalg.norm(kepler_propagate(r, v, mid)[0])
        lo, hi = (mid, hi) if rm - R_EARTH > 80.0 else (lo, mid)
    event = next(ev for ev in sim.events if "re-entered" in ev.text)
    assert event.t == pytest.approx(lo, abs=0.05)
