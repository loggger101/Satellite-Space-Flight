"""Smaller physics-audit fixes (docs/PHYSICS_AUDIT.md A12-A15): warnings for terms a
propagator ignores, sun-synchronous orbits under the full zonal field, the launch
window's plane drift, and launch bookkeeping."""

import math
from datetime import datetime

import numpy as np
import pytest

from satflight.analysis import SSO_RATE
from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.launch import (
    LAUNCH_SITES,
    AscentEnv,
    LaunchSpec,
    _pressure,
    fly,
    resolve,
    vehicle_preset,
)
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.timeutil import UTC, Clock

EPOCH = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
CLOCK = Clock(EPOCH)
ENV = AscentEnv(gmst=lambda t: float(CLOCK.gmst(t)))
CAPE = LAUNCH_SITES["Cape Canaveral SLC-40 (USA)"]


def _sim(orbit, forces, propagator="cowell"):
    return Simulation(Scenario(name="t", epoch=EPOCH, forces=forces, propagator=propagator,
                               satellites=[SatSpec("S", orbit)]))


@pytest.mark.parametrize("propagator, forces, named", [
    ("j2mean", ForceModel(j2=True, drag=True, j3=True), "J3, DRAG"),
    ("kepler", ForceModel(j2=True), "J2"),
])
def test_an_analytic_run_says_which_terms_it_ignores(propagator, forces, named):
    # A12: a j2mean run with drag "on" never decayed, and nothing said so
    sim = _sim({"type": "elements", "altitude": 500}, forces, propagator)
    warns = [e.text for e in sim.events if e.kind == "warn"]
    assert warns == [f"Propagator {propagator} ignores {named}: switch to cowell to "
                     "integrate them"]
    assert not [e for e in _sim({"type": "elements", "altitude": 500}, forces).events
                if e.kind == "warn"]


@pytest.mark.parametrize("forces", [ForceModel(j2=True), ForceModel(j2=True, j3=True, j4=True)])
def test_an_sso_orbit_follows_the_mean_sun_under_the_full_zonal_field(forces):
    # A13: the first-order inclination drifted 0.0008 deg/day (J2) and 0.003 (J2-J4)
    sim = _sim({"type": "elements", "altitude": 700, "i": "sso", "nu": 30}, forces)
    ts, raans = [], []
    for _ in range(3 * 24):
        ts.append(sim.t)
        raans.append(rv2coe(sim.y[0, :3], sim.y[0, 3:]).raan)
        sim.advance(3600.0)
    rate = np.polyfit(ts, np.unwrap(raans), 1)[0]
    assert math.degrees(rate - SSO_RATE) * 86400 == pytest.approx(0.0, abs=1e-4)  # deg/day


def test_the_launch_window_plane_drifts_with_the_mean_orbit():
    # A14: the osculating a and i put the drift ~0.3 % off depending on where it is read
    for nu in (0.0, 45.0, 90.0):
        sim = _sim({"type": "elements", "altitude": 420, "i": 51.6, "raan": 30, "nu": nu},
                   ForceModel(j2=True))
        _, normal_at = sim.plane_of("S")
        sim.advance(2 * 86400.0)
        h = np.cross(sim.y[0, :3], sim.y[0, 3:])
        h /= np.linalg.norm(h)
        # the osculating plane wobbles by ~0.01 deg; the secular drift is 10 deg
        assert math.degrees(math.acos(np.clip(np.dot(h, normal_at(sim.t)), -1, 1))) < 0.02


def test_us_standard_atmosphere_pressure():
    # A15: thrust used exp(-h / 7 km); US76 table values (p / 101325 Pa)
    for h, p in ((0.0, 1.0), (11.019, 0.22336), (20.063, 0.054033), (32.162, 0.0085666),
                 (47.350, 0.0010945), (51.413, 6.6063e-4), (71.802, 3.9046e-5),
                 (86.0, 3.6850e-6)):
        assert _pressure(h) == pytest.approx(p, rel=1e-4), h
    assert _pressure(-0.5) == 1.0 and _pressure(200.0) < 1e-9


def _spec(**kw):
    kw.setdefault("payload_mass", 5000.0)
    lat, lon, alt = CAPE
    return LaunchSpec(lat=lat, lon=lon, alt=alt, perigee_alt=200.0, apogee_alt=200.0,
                      inclination=28.6, **kw)


def test_a_fairing_still_on_at_cutoff_goes_alone():
    # A15: it used to leave with the last stage, counted in that stage's mass
    veh = vehicle_preset("Falcon 9 (approx.)")
    veh.fairing_alt = 400.0                        # higher than the 200 km cutoff
    s = _spec(vehicle=veh, kick=2.5)
    f = fly(s, resolve(s, ENV, 0.0), ENV, 2.5)
    assert f.outcome == "orbit"
    assert any("fairing jettison at cutoff" in text for _, text, _ in f.events)
    last = [m for _, name, _, m, _ in f.spawns if name.endswith("Stage 2")]
    assert last == [pytest.approx(veh.stages[1].dry + f.insertion["prop_left"])]


def test_the_throttle_floor_says_when_it_breaks_the_g_limit():
    veh = vehicle_preset("Falcon 9 (approx.)")
    veh.max_g = 2.0                    # stage 1 near burnout gives more even at 40 %
    s = _spec(vehicle=veh, kick=2.5)
    f = fly(s, resolve(s, ENV, 0.0), ENV, 2.5)
    warns = [text for _, text, kind in f.events if kind == "warn"]
    assert warns == ["Payload: Stage 1 exceeds the 2 g limit even at 40 % throttle (2.01 g)"]


@pytest.mark.parametrize("tw, lifts", [(1.002, True), (0.998, False)])
def test_liftoff_weighs_the_vehicle_on_the_turning_earth(tw, lifts):
    # A15: weight was m mu / r^2, 0.34 % more than the plumb-line weight at the equator
    lat, lon, alt = LAUNCH_SITES["Kourou ELA-3 (French Guiana)"]
    s2 = math.sin(math.radians(lat)) ** 2      # WGS-84 normal gravity (Somigliana)
    g = 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - 0.00669437999013 * s2)
    veh = vehicle_preset("Falcon 9 (approx.)")
    st = veh.stages[0]
    thrust = st.mdot * 9.80665 * st.isp(_pressure(alt))
    payload = thrust / (g * tw) - veh.liftoff_mass(0.0)
    s = LaunchSpec(lat=lat, lon=lon, alt=alt, vehicle=veh, payload_mass=payload,
                   perigee_alt=200.0, apogee_alt=200.0, inclination=10.0, kick=1.0)
    f = fly(s, resolve(s, ENV, 0.0), ENV, 1.0)
    assert (f.outcome != "no_liftoff") == lifts, f.events[0]
