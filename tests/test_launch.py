"""Launch vehicles: guidance accuracy, launch windows, failure modes, mass
and delta-v bookkeeping, and launches run inside the simulation."""

import math
from datetime import datetime

import numpy as np
import pytest

from satflight.constants import R_EARTH
from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.frames import eci_to_ecef
from satflight.maneuvers import Maneuver
from satflight.launch import (G0_M, LAUNCH_SITES, AscentEnv, LaunchSpec, Stage, Vehicle, fly,
                              plan_launch, resolve, vehicle_preset)
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.timeutil import UTC, Clock

EPOCH = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
CLOCK = Clock(EPOCH)
ENV = AscentEnv(gmst=lambda t: float(CLOCK.gmst(t)))
CAPE = LAUNCH_SITES["Cape Canaveral SLC-40 (USA)"]
VAFB = LAUNCH_SITES["Vandenberg SLC-4E (USA)"]


def spec(**kw):
    kw.setdefault("vehicle", vehicle_preset("Falcon 9 (approx.)"))
    kw.setdefault("payload_mass", 5000.0)
    lat, lon, alt = kw.pop("site_xyz", CAPE)
    return LaunchSpec(lat=lat, lon=lon, alt=alt, **kw)


# --- guidance ---------------------------------------------------------------------------

@pytest.mark.parametrize("hp, ha, inc, direction, site", [
    (400.0, 400.0, 51.6, "north", CAPE),        # ISS-like
    (250.0, 35786.0, 28.6, "north", CAPE),      # GTO: insertion at perigee
    (550.0, 550.0, 97.6, "south", VAFB),        # retrograde sun-synchronous
])
def test_closed_loop_guidance_reaches_the_target_orbit(hp, ha, inc, direction, site):
    plan = plan_launch(spec(perigee_alt=hp, apogee_alt=ha, inclination=inc, direction=direction,
                            site_xyz=site), ENV, 0.0)
    ins = plan.flight.insertion
    assert plan.ok and plan.flight.outcome == "orbit"
    assert ins["hp"] == pytest.approx(hp, abs=5.0)
    assert ins["ha"] == pytest.approx(ha, abs=max(5.0, 2e-3 * ha))
    assert ins["i"] == pytest.approx(inc, abs=0.05)
    assert ins["prop_left"] > 0
    # the vehicle crossed the pad's latitude in the requested direction
    az = math.degrees(plan.res.azimuth) % 360
    assert (az < 90 or az > 270) if direction == "north" else 90 < az < 270


def test_every_preset_vehicle_flies_its_kind_of_mission():
    cases = [("Falcon 9 (approx.)", 15000, CAPE, 300, 300, 28.6),
             ("Electron (approx.)", 150, LAUNCH_SITES["Mahia LC-1 (New Zealand)"], 200, 500, 97.4),
             ("Saturn V (approx.)", 45000, LAUNCH_SITES["Kennedy LC-39A (USA)"], 190, 190, 32.5)]
    for name, payload, site, hp, ha, inc in cases:
        south = inc > 90
        p = plan_launch(spec(vehicle=vehicle_preset(name), payload_mass=payload, site_xyz=site,
                             perigee_alt=hp, apogee_alt=ha, inclination=inc,
                             direction="south" if south else "north"), ENV, 0.0)
        assert p.ok, (name, p.flight.outcome)
        assert p.flight.insertion["hp"] == pytest.approx(hp, abs=5.0), name
    hop = plan_launch(spec(vehicle=vehicle_preset("Sounding rocket"), payload_mass=150, guidance="open",
                           site_xyz=LAUNCH_SITES["Andoya (Norway)"], kick=1.0), ENV, 0.0)
    assert hop.flight.outcome == "suborbital" and 200 < hop.flight.insertion["ha"] < 500


def test_low_thrust_vehicle_needs_and_gets_a_tiny_kick():
    """Saturn V lifts off at T/W 1.19: only a pitch kick of a fraction of a
    degree avoids tipping over, and it must be measured from the true local
    vertical (the geocentric radius alone tips it into the ground)."""
    s = spec(vehicle=vehicle_preset("Saturn V (approx.)"), payload_mass=45000, perigee_alt=190,
             apogee_alt=190, inclination=32.5, site_xyz=LAUNCH_SITES["Kennedy LC-39A (USA)"])
    res = resolve(s, ENV, 0.0)
    assert fly(s, res, ENV, 2.0).outcome == "crash"
    plan = plan_launch(s, ENV, 0.0)
    assert plan.ok and plan.kick < 0.5


def test_unreachable_inclination_is_refused_with_the_reachable_range():
    baikonur = LAUNCH_SITES["Baikonur Site 1 (Kazakhstan)"]
    with pytest.raises(ValueError, match="cannot be reached directly"):
        resolve(spec(inclination=28.5, site_xyz=baikonur), ENV, 0.0)
    with pytest.raises(ValueError, match="cannot be reached directly"):
        resolve(spec(inclination=140.0, site_xyz=baikonur), ENV, 0.0)
    resolve(spec(inclination=51.6, site_xyz=baikonur), ENV, 0.0)


@pytest.mark.parametrize("bad, match", [
    (dict(vehicle=Vehicle("empty", [])), "no stages"),
    (dict(perigee_alt=60.0), "perigee"),
    (dict(vehicle=Vehicle("slow", [Stage("S", 1.0, 300.0, 0.0, 5000.0, 100.0)])), "burn"),
    (dict(timing="whenever"), "timing"),
])
def test_invalid_specs_are_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        resolve(spec(**bad), ENV, 0.0)


def test_failures_are_outcomes_not_exceptions():
    heavy = plan_launch(spec(payload_mass=40000.0, perigee_alt=300, apogee_alt=300, inclination=28.6),
                        ENV, 0.0)
    assert heavy.flight.outcome == "short" and not heavy.ok
    assert heavy.flight.insertion["hp"] < 0                    # it falls back
    grounded = fly(spec(payload_mass=900000.0, kick=1.0), resolve(spec(), ENV, 0.0), ENV, 1.0)
    assert grounded.outcome == "no_liftoff"


# --- bookkeeping -----------------------------------------------------------------------------

def test_mass_flow_staging_and_the_delta_v_budget():
    s = spec(vehicle=vehicle_preset("Falcon 9 (approx.)"), payload_mass=5000.0)
    veh = s.vehicle
    f = fly(s, resolve(s, ENV, 0.0), ENV, 3.0)
    assert f.outcome == "orbit"
    # stage 1 burned for exactly propellant / mass flow
    marks = {label: t for t, _, _, label in f.stage_marks}
    assert marks["Stage 1 sep."] == pytest.approx(veh.stages[0].burn_time, abs=1e-6)
    assert marks["Stage 2 ign."] - marks["Stage 1 sep."] == pytest.approx(veh.stage_coast, abs=1e-6)
    # dropped: stage 1 dry mass, then stage 2 dry mass plus what it had left
    drops = [(name, m) for _, name, _, m, _ in f.spawns]
    assert drops[0] == (f"{s.name} Stage 1", veh.stages[0].dry)
    assert drops[1][1] == pytest.approx(veh.stages[1].dry + f.insertion["prop_left"])
    assert f.mass == s.payload_mass
    # total mass: lift-off = payload + fairing + stage drops + propellant burned
    burned = veh.stages[0].propellant + veh.stages[1].propellant - f.insertion["prop_left"]
    assert veh.liftoff_mass(s.payload_mass) == pytest.approx(
        s.payload_mass + veh.fairing + sum(m for _, m in drops) + burned)
    # delta-v: ideal - losses = speed gained (rectangle-rule accounting, 1 s steps)
    speed = math.sqrt(sum(v * v for v in f.y[3:]))
    assert f.dv_ideal - sum(f.losses.values()) == pytest.approx(speed - f.v_start, rel=0.02)
    # and never more than the rocket equation allows in vacuum
    m0 = veh.liftoff_mass(s.payload_mass)
    ideal = 0.0
    for k, st in enumerate(veh.stages):
        used = st.propellant if k == 0 else st.propellant - f.insertion["prop_left"]
        ideal += st.isp_vac * G0_M * math.log(m0 / (m0 - used)) / 1000
        m0 -= st.propellant + st.dry + (veh.fairing if k == 0 else 0)
    assert f.dv_ideal <= ideal + 1e-6


def test_launch_window_puts_the_orbit_on_the_requested_node():
    s = spec(timing="raan", raan=200.0, inclination=51.6, perigee_alt=300, apogee_alt=300)
    plan = plan_launch(s, ENV, 0.0)
    assert 0 < plan.res.t0 < 86400
    el = rv2coe(np.array(plan.flight.y[:3]), np.array(plan.flight.y[3:]))
    assert math.degrees(el.raan) == pytest.approx(200.0, abs=0.1)
    assert math.degrees(el.i) == pytest.approx(51.6, abs=0.05)
    south = plan_launch(spec(timing="raan", raan=200.0, inclination=51.6, perigee_alt=300,
                             apogee_alt=300, direction="south"), ENV, 0.0)
    assert south.res.t0 != pytest.approx(plan.res.t0, abs=600)   # the other pass of the day
    el = rv2coe(np.array(south.flight.y[:3]), np.array(south.flight.y[3:]))
    assert math.degrees(el.raan) == pytest.approx(200.0, abs=0.1)


# --- inside the simulation ------------------------------------------------------------------------

def _scenario(propagator="cowell", launches=(), sats=()):
    return Scenario(name="launch test", epoch=EPOCH, propagator=propagator,
                    forces=ForceModel(j2=True, drag=True), satellites=list(sats), launches=list(launches))


def test_vehicle_waits_on_the_pad_turning_with_the_earth_then_flies():
    sim = Simulation(_scenario(launches=[spec(name="Sat", timing="delay", delay=900.0)]))
    i = sim.index_of("Sat")
    asc = sim.ascent_of(i)
    assert asc.phase == "pad" and asc.t0 == pytest.approx(900.0)
    pad0 = eci_to_ecef(sim.y[i, :3], sim.gmst())
    sim.advance(600.0)
    assert np.allclose(eci_to_ecef(sim.y[i, :3], sim.gmst()), pad0, atol=1e-6)
    assert sim.sats[i].status == "active"
    with pytest.raises(ValueError, match="launch vehicle"):
        sim.schedule(Maneuver("Sat", "impulse", dv=(0.01, 0.0, 0.0)))
    sim.advance(900.0)                                         # T+10 min: guided upper stage
    assert sim.ascent_of(i) is None or sim.ascent_of(i).phase == "guided"
    sim.advance(600.0)
    assert sim.ascent_of(i) is None
    el = rv2coe(sim.y[i, :3], sim.y[i, 3:])
    assert el.rp - R_EARTH > 380 and el.ra - R_EARTH < 420
    names = [s.name for s in sim.sats]
    assert names[-2:] == ["Sat Stage 1", "Sat Stage 2"]
    kinds = [e.text for e in sim.events if e.kind == "launch"]
    assert any("lift-off" in t for t in kinds) and any("SECO" in t for t in kinds)
    assert sim.sats[i].launch_report["outcome"] == "orbit"
    assert sim.sats[i].mass == 5000.0 and sim.sats[i].area == 5.0


@pytest.mark.parametrize("propagator", ["cowell", "j2mean", "kepler"])
def test_launch_into_another_satellites_plane(propagator):
    iss = SatSpec("ISS", {"type": "elements", "altitude": 420, "i": 51.64, "raan": 30})
    sim = Simulation(_scenario(propagator, [spec(name="Dragon", timing="plane", target="ISS",
                                                 perigee_alt=300, apogee_alt=300, payload_mass=12000)],
                               [iss]))
    asc = sim.ascents[0]
    sim.advance(asc.t0 + 1200.0)
    a, b = sim.index_of("ISS"), sim.index_of("Dragon")
    h1 = np.cross(sim.y[a, :3], sim.y[a, 3:])
    h2 = np.cross(sim.y[b, :3], sim.y[b, 3:])
    angle = math.degrees(math.acos(np.dot(h1, h2) / np.linalg.norm(h1) / np.linalg.norm(h2)))
    assert angle < 0.1
    # payload and its stage 2 fly together without close-approach alarms
    assert not any("Close approach" in e.text and "Dragon" in e.text for e in sim.events)


def test_stage_one_falls_back_and_the_payload_can_circularise():
    sim = Simulation(_scenario(launches=[spec(name="GTO sat", perigee_alt=250, apogee_alt=35786,
                                              inclination=28.6, circularize=True, payload_mass=4000)]))
    sim.advance(1200.0)
    assert sim.sats[sim.index_of("GTO sat Stage 1")].status in ("re-entered", "impacted")
    assert any(m.sat == "GTO sat" and m.kind == "circularize" for m in sim.maneuvers)
    sim.advance(6 * 3600.0)
    i = sim.index_of("GTO sat")
    el = rv2coe(sim.y[i, :3], sim.y[i, 3:])
    assert el.e < 0.01 and el.a - R_EARTH == pytest.approx(35786, rel=0.01)


def test_scenario_round_trip_and_snapshot_keep_pending_launches(tmp_path):
    sc = _scenario(launches=[spec(name="Later", timing="delay", delay=3600.0, kick=2.5)])
    path = tmp_path / "launch.json"
    sc.save(path)
    loaded = Scenario.load(path)
    assert loaded.launches[0].vehicle.stages[1].thrust == 981.0
    sim = Simulation(loaded)
    sim.advance(600.0)
    snap = sim.snapshot_scenario("snap")
    assert [s.name for s in snap.satellites] == []            # the pad vehicle is a launch, not a sat
    assert snap.launches[0].timing == "absolute" and snap.launches[0].t0 == pytest.approx(3000.0)
    again = Simulation(Scenario.from_dict(snap.to_dict()))
    assert again.ascents[0].t0 == pytest.approx(3000.0) and again.ascents[0].kick_deg == 2.5


def test_bad_launch_in_a_scenario_is_scrubbed_not_fatal():
    sim = Simulation(_scenario(launches=[spec(name="Nope", inclination=10.0)]))
    assert not sim.ascents
    assert any(e.kind == "alert" and "scrubbed" in e.text for e in sim.events)


def test_adding_objects_does_not_repeat_station_acquisitions():
    from satflight.scenario import DEFAULT_STATIONS
    sc = _scenario(sats=[SatSpec("GEO", {"type": "geo", "lon": -75})])
    sc.stations = list(DEFAULT_STATIONS)
    sim = Simulation(sc)
    sim.advance(60.0)
    before = sum("AOS GEO" in e.text for e in sim.events)
    assert before >= 1
    sim.launch(spec(name="New"))
    sim.advance(900.0)                                    # stages are spawned along the way
    assert sum("AOS GEO" in e.text for e in sim.events) == before
