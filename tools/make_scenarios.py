"""Regenerate the example scenarios in ``scenarios/``.

    python tools/make_scenarios.py

Each function below builds one scenario (``ALL`` maps file names to them).
Edit these, not the JSON files: scenarios with planned maneuvers or launches
run the planners here and store the results.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satflight.forces import ForceModel  # noqa: E402
from satflight.launch import LAUNCH_SITES, LaunchSpec, pad_state, vehicle_preset  # noqa: E402
from satflight.planner import (  # noqa: E402
    plan_hohmann,
    plan_rendezvous,
    plan_target_orbit,
    scan_rendezvous,
)
from satflight.scenario import (  # noqa: E402
    DEFAULT_STATIONS,
    ConstellationSpec,
    GroundStation,
    IntegratorSettings,
    SatSpec,
    Scenario,
    preset_spec,
)
from satflight.simulation import Simulation  # noqa: E402
from satflight.timeutil import UTC, Clock  # noqa: E402

EPOCH = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
OUT = ROOT / "scenarios"


def el(**kw):
    """Orbit spec of type ``elements``."""
    return {"type": "elements", **kw}


# Sentinel-1 reference orbit (ESA): sun-synchronous, 175 revolutions in a
# 12-day repeat cycle, ascending node at 18:00 local mean solar time (dawn-dusk),
# frozen eccentricity with the perigee held over the North Pole (argp 90 deg).
# Sentinel-1C flies 1A's orbit 180 deg behind it (where 1B was), so the pair
# covers each ground track every 6 days.
S1_REVS, S1_DAYS = 175, 12
S1_LTAN = 18.0                     # h, local mean solar time of the ascending node
# Mean altitude (km above R_EARTH) whose node-to-node period, flown with J2, J3
# and J4 in full on the "sso" inclination they pick, is 12 d / 175 = 5924.571 s:
# the repeat closes to ~2 ms after 175 revolutions (tests/test_sentinel1.py
# measures it). The published mean semi-major axis is 7071 km (693 km).
S1_ALT = 692.827267
# Frozen eccentricity under J2 + J3 (first order): e = -J3 Re sin(i) / (2 J2 a)
S1_E = 0.0010443


def s1_raan(epoch: datetime) -> float:
    """RAAN (deg) that puts the ascending node at S1_LTAN local mean time at ``epoch``."""
    clock = Clock(epoch)
    ut = epoch.hour + epoch.minute / 60 + epoch.second / 3600
    return (math.degrees(clock.gmst(0.0)) + (S1_LTAN - ut) * 15.0) % 360.0


def default():
    """Sentinel-1A and -1C on ESA's reference orbit (until real orbit data replaces it)."""
    raan = s1_raan(EPOCH)

    def s1(name, u, mass):             # u: argument of latitude at the epoch, deg
        return SatSpec(name, el(altitude=S1_ALT, e=S1_E, i="sso", raan=round(raan, 6),
                                argp=90.0, nu=(u - 90.0) % 360.0),
                       color=None, mass=mass, area=10.0, cd=2.2)

    return Scenario(
        name="Sentinel-1A and 1C",
        description="ESA's Copernicus radar pair on the Sentinel-1 reference orbit: "
                    "693 km, sun-synchronous (98.18 deg), dawn-dusk with the ascending "
                    "node at 18:00, frozen perigee over the North Pole, 175 orbits in a "
                    "12-day repeat. 1C flies 180 deg behind 1A, so together they revisit "
                    "each ground track every 6 days. Open the map (M) to watch the tracks "
                    "interleave. Nominal orbit, not tracking data.",
        epoch=EPOCH, forces=ForceModel(j2=True, j3=True, j4=True, drag=True), warp=120,
        satellites=[s1("Sentinel-1A", 0.0, 2164.0), s1("Sentinel-1C", 180.0, 2200.0)],
        stations=[GroundStation("Svalbard", 78.23, 15.39, 0.5, 5.0),
                  GroundStation("Matera", 40.65, 16.70, 0.5, 5.0),
                  GroundStation("Maspalomas", 27.76, -15.63, 0.2, 5.0),
                  GroundStation("Kiruna", 67.86, 20.96, 0.4, 5.0)])


# The replay's twins carry the drag fitted to 2025-07-03 (tools/fit_arc.py S1x
# 2025-07-03 --degree 70 --step 120 --no-cr): Cd*A/m 0.00867 (S1A) and 0.00862
# (S1C) m^2/kg with Cr held at 1.3, fits of 1.64 m RMS that stray 6 m (RMS
# along-track) over the next day. Step 5 replaces these with the year's mean.
REPLAY_EPOCH = datetime(2025, 7, 3, tzinfo=UTC)
REPLAY_UNTIL = "2025-07-09"
REPLAY_CD_AM = {"S1A": 0.00867, "S1C": 0.00862}


def sentinel1_replay():
    """S1A and S1C following ESA's precise orbits, each with a twin flown by the
    full force model from the same recorded state (needs the downloaded data)."""
    sats = []
    for mission, mass, color, twin_color in (("S1A", 2164.0, [255, 196, 64], [120, 200, 255]),
                                             ("S1C", 2200.0, [255, 120, 90], [140, 255, 170])):
        name = f"Sentinel-1{mission[-1]}"
        cd = round(REPLAY_CD_AM[mission] * mass / 10.0, 4)
        sats += [SatSpec(f"{name} (ESA)", {"type": "replay", "mission": mission,
                                           "until": REPLAY_UNTIL},
                         color=color, mass=mass, area=10.0, cd=cd),
                 SatSpec(f"{name} twin", {"type": "precise", "mission": mission},
                         color=twin_color, mass=mass, area=10.0, cd=cd)]
    return Scenario(
        name="Sentinel-1 precise-orbit replay",
        description="Sentinel-1A and 1C where ESA's precise orbit files put them, 3 to 9 July "
                    "2025, each beside a twin: a simulated copy started on the same recorded "
                    "state and flown by the full force model (EGM2008 70x70, drag with the "
                    "fitted Cd, the Sun, the Moon and sunlight pressure). Select a twin to "
                    "see its radial, along-track and cross-track gap to the truth grow; the "
                    "twins fly no burns, so they part from the truth at each orbit trim "
                    "(S1A on 5 July). Needs the data from tools/fetch_poeorb.py and "
                    "tools/fetch_eop.py.",
        epoch=REPLAY_EPOCH, warp=60,
        forces=ForceModel(j2=True, j3=True, j4=True, drag=True, gravity_degree=70, sun=True,
                          moon=True, srp=True),
        satellites=sats,
        stations=[GroundStation("Svalbard", 78.23, 15.39, 0.5, 5.0),
                  GroundStation("Matera", 40.65, 16.70, 0.5, 5.0),
                  GroundStation("Maspalomas", 27.76, -15.63, 0.2, 5.0),
                  GroundStation("Kiruna", 67.86, 20.96, 0.4, 5.0)])


def orbit_tour():
    names =["ISS (LEO 420 km, 51.6 deg)", "Hubble (540 km, 28.5 deg)",
             "Sun-synchronous 700 km", "GPS MEO (20200 km, 55 deg)", "Geostationary (GEO)",
             "Molniya (12 h, 63.4 deg)", "GTO (200 x 35786 km, 27 deg)"]
    return Scenario(
        name="Earth orbit tour",
        description="One satellite in each classic regime: LEO, SSO, MEO, GEO, Molniya and GTO.",
        epoch=EPOCH, satellites=[preset_spec(n) for n in names], stations=list(DEFAULT_STATIONS),
        forces=ForceModel(j2=True), warp=120)


def hohmann():
    sc = Scenario(
        name="Hohmann transfer LEO to GEO",
        description="Equatorial 300 km parking orbit raised to GEO with two scheduled burns; "
                    "a companion stays behind for comparison.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=300,
        satellites=[
            SatSpec("Transfer", el(altitude=300, i=0.0, raan=0, nu=0), mass=4000, area=15),
            SatSpec("Parking", el(altitude=300, i=0.0, raan=0, nu=-20), mass=4000, area=15)])
    sim = Simulation(sc)
    sim.advance(600)
    burns, summary = plan_hohmann(sim, "Transfer", 35786.0, "now")
    sc.maneuvers = burns
    sc.description += f" Plan: {summary}."
    return sc


def gto_to_geo():
    """The classic way to geostationary orbit from Cape Canaveral (28.5 deg), two ways."""
    gto = el(perigee_alt=185.0, apogee_alt=35786.0, i=28.5, raan=40.0, argp=180.0, nu=120.0)
    sc = Scenario(
        name="GTO to GEO",
        description="Two satellites go from a Cape Canaveral launch (28.5 deg) to "
                    "geostationary orbit with the Change-orbit planner. 'Apogee kick' rides "
                    "a 185 x 35,786 km transfer orbit and fires once at apogee, turning the "
                    "whole 28.5 deg there, where it moves slowest. 'From parking' starts in "
                    "a 185 km circular orbit: its first burn (the transfer injection) takes "
                    "a little of the turn for almost nothing, the second does the rest at "
                    "apogee.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=600,
        satellites=[
            SatSpec("Apogee kick", gto, mass=4000, area=15),
            SatSpec("From parking", el(altitude=185.0, i=28.5, raan=40.0, nu=300.0),
                    mass=6000, area=15)])
    sim = Simulation(sc)
    kick, kick_summary = plan_target_orbit(sim, "Apogee kick", 35786.0, 35786.0, 0.0,
                                           "apoapsis")
    park, park_summary = plan_target_orbit(sim, "From parking", 35786.0, 35786.0, 0.0,
                                           "ascending_node")
    sc.maneuvers = kick + park
    sc.description += f" Apogee kick: {kick_summary}. From parking: {park_summary}."
    return sc


def rendezvous():
    sc = Scenario(
        name="Lambert rendezvous",
        description="A chaser 70 km below and 6 deg behind a station intercepts it with a "
                    "Lambert transfer (time of flight chosen for minimum delta-v with the arc "
                    "kept above the atmosphere), aimed through J2, then matches velocity.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=30,
        satellites=[
            SatSpec("Station", el(altitude=420, i=51.6, raan=30, nu=40), mass=420000, area=1600),
            SatSpec("Chaser", el(altitude=350, i=51.6, raan=30, nu=34), mass=8000, area=20)],
        stations=list(DEFAULT_STATIONS[:2]))
    sim = Simulation(sc)
    sim.advance(300)
    tof, _ = scan_rendezvous(sim, "Chaser", "Station", 1200.0, 3 * 3600.0, 200)
    burns, summary = plan_rendezvous(sim, "Chaser", "Station", tof)
    sc.maneuvers = burns
    sc.description += f" Plan (TOF {tof / 60:.0f} min): {summary}."
    return sc


def starlink():
    return Scenario(
        name="Starlink-like shell (1584)",
        description="Walker delta 53 deg: 1584/72/17 at 550 km, propagated with the analytic "
                    "J2-secular model for speed.",
        epoch=EPOCH, propagator="j2mean", forces=ForceModel(j2=True), warp=60,
        constellations=[ConstellationSpec("Shell", 550, 53.0, 1584, 72, 17, "delta",
                                          [120, 200, 255])],
        stations=list(DEFAULT_STATIONS))


def gps():
    return Scenario(
        name="GPS-like constellation",
        description="Walker delta 55 deg: 24/6/1 at 20180 km, Cowell propagation with J2.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=600,
        constellations=[ConstellationSpec("GPS", 20180, 55.0, 24, 6, 1, "delta", [255, 196, 64],
                                          mass=2000, area=20)],
        stations=list(DEFAULT_STATIONS))


def iridium():
    return Scenario(
        name="Polar star constellation",
        description="Iridium-like Walker star: 66/6/2 at 780 km, 86.4 deg - planes span "
                    "180 deg so there is a counter-rotating seam.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=120,
        constellations=[ConstellationSpec("Polar", 780, 86.4, 66, 6, 2, "star",
                                          [140, 240, 140])])


def drag():
    return Scenario(
        name="Atmospheric drag decay",
        description="Three spacecraft at 300 km with different ballistic coefficients: a "
                    "dense probe, a CubeSat and a balloon. The balloon re-enters within about "
                    "eight hours.",
        epoch=EPOCH, forces=ForceModel(j2=True, drag=True), warp=3600,
        satellites=[
            SatSpec("Dense probe", el(altitude=300, i=51.6, raan=0, nu=0), mass=2000, area=1.0),
            SatSpec("CubeSat 3U", el(altitude=300, i=51.6, raan=0, nu=120), mass=4, area=0.03),
            SatSpec("Balloon", el(altitude=300, i=51.6, raan=0, nu=240), mass=60, area=40)])


def j2():
    incs = [0.0, 30.0, 63.4, 90.0, "sso", 130.0]
    sats = [SatSpec(f"i = {i if isinstance(i, str) else f'{i:g}'}",
                    el(perigee_alt=600, apogee_alt=1400, i=i, raan=0, argp=30, nu=0),
                    mass=500, area=4)
            for i in incs]
    return Scenario(
        name="J2 precession study",
        description="Identical 600 x 1400 km orbits at six inclinations: RAAN regresses "
                    "fastest at low inclination, stands still at 90 deg, and the perigee "
                    "freezes at 63.4 deg. "
                    "Open the plot panel (G) and pick RAAN or Arg. of perigee.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=3600, satellites=sats)


def heo():
    return Scenario(
        name="Molniya and Tundra",
        description="Three Molniya and three Tundra satellites sharing ground tracks - best "
                    "viewed in the Earth-fixed frame (E) with the map (M).",
        epoch=EPOCH, forces=ForceModel(j2=True, j3=True, j4=True), warp=1200,
        satellites=[SatSpec("Molniya", el(a=26554, e=0.72, i=63.4, raan=250, argp=270, nu=0),
                            count=3, mass=1600, area=10),
                    SatSpec("Tundra", el(a=42164, e=0.25, i=63.4, raan=120, argp=270, nu=0),
                            count=3, mass=2000, area=15)],
        stations=[GroundStation("Moscow", 55.75, 37.62, 0.15, 10.0),
                  GroundStation("Fairbanks", 64.84, -147.72, 0.14, 10.0)])


def launches():
    cape = dict(type="surface", lat=28.5, lon=-80.6, alt=100.0, azimuth=90.0)   # burnout state
    return Scenario(
        name="Launches and ballistic arcs",
        description="Burnout states: sub-orbital hops at 15/45/75 deg from 100 km over the "
                    "Cape, an orbit insertion over Kourou and a hyperbolic escape at 11.6 km/s.",
        epoch=EPOCH, forces=ForceModel(j2=True, drag=True), warp=30,
        satellites=[SatSpec("Hop 15 deg", dict(cape, speed=4.0, fpa=15.0), mass=500, area=1),
                    SatSpec("Hop 45 deg", dict(cape, speed=4.0, fpa=45.0), mass=500, area=1),
                    SatSpec("Hop 75 deg", dict(cape, speed=4.0, fpa=75.0), mass=500, area=1),
                    SatSpec("Insertion", {"type": "surface", "lat": 5.2, "lon": -52.8,
                                          "alt": 200.0, "speed": 7.4, "azimuth": 90.0,
                                          "fpa": 0.0}, mass=5000, area=10),
                    SatSpec("Escape", {"type": "surface", "lat": 28.5, "lon": -80.6, "alt": 300.0,
                                       "speed": 11.6, "azimuth": 90.0, "fpa": 5.0},
                            mass=1000, area=4)],
        stations=[DEFAULT_STATIONS[0], DEFAULT_STATIONS[4]])


def launch_day():
    def site(name, **kw):
        lat, lon, alt = LAUNCH_SITES[name]
        return dict(site=name, lat=lat, lon=lon, alt=alt, **kw)

    launches_ = [
        LaunchSpec("Aurora probe", vehicle_preset("Sounding rocket"), guidance="open",
                   azimuth=0.0, timing="delay", delay=60.0, kick=1.0, payload_mass=150.0,
                   payload_area=0.3, **site("Andoya (Norway)")),
        LaunchSpec("Starlink batch", vehicle_preset("Falcon 9 (approx.)"), timing="delay",
                   delay=120.0, perigee_alt=300.0, apogee_alt=300.0, inclination=53.0,
                   payload_mass=15000.0, payload_area=20.0, **site("Cape Canaveral SLC-40 (USA)")),
        LaunchSpec("Rocket Lab sat", vehicle_preset("Electron (approx.)"), timing="delay",
                   delay=300.0, perigee_alt=200.0, apogee_alt=500.0, inclination=97.4,
                   direction="south", circularize=True, payload_mass=150.0, payload_area=1.0,
                   **site("Mahia LC-1 (New Zealand)")),
        LaunchSpec("Apollo stack", vehicle_preset("Saturn V (approx.)"), timing="delay",
                   delay=480.0, perigee_alt=190.0, apogee_alt=190.0, inclination=32.5,
                   payload_mass=45000.0, payload_area=30.0, **site("Kennedy LC-39A (USA)")),
        LaunchSpec("Comsat", vehicle_preset("Falcon 9 (approx.)"), timing="delay", delay=720.0,
                   perigee_alt=250.0, apogee_alt=35786.0, inclination=6.0, circularize=True,
                   payload_mass=5500.0, payload_area=15.0,
                   **site("Kourou ELA-3 (French Guiana)")),
        LaunchSpec("Crew Dragon", vehicle_preset("Falcon 9 (approx.)"), timing="plane",
                   target="ISS", perigee_alt=300.0, apogee_alt=300.0, payload_mass=12500.0,
                   payload_area=12.0, **site("Cape Canaveral SLC-40 (USA)")),
    ]
    # put the ISS plane over the Cape (northbound) 25 minutes in, so the Crew
    # Dragon's launch window comes up during the show
    iss = preset_spec("ISS (LEO 420 km, 51.6 deg)")
    r, _ = pad_state(*LAUNCH_SITES["Cape Canaveral SLC-40 (USA)"],
                     float(Clock(EPOCH).gmst(1500.0)))
    inc = math.radians(iss.orbit["i"])
    u = math.asin(r[2] / math.hypot(*r) / math.sin(inc))    # argument of latitude over the pad
    node = math.atan2(r[1], r[0]) - math.atan2(math.cos(inc) * math.sin(u), math.cos(u))
    iss.orbit["raan"] = round(math.degrees(node) % 360, 3)
    sc = Scenario(
        name="Launch day",
        description="Six launches from four continents: a sounding rocket over Norway, "
                    "Falcon 9 to a 53 deg shell, Electron to sun-synchronous orbit, a Saturn V "
                    "parking orbit, a Kourou GTO that circularizes at apogee, and a Crew Dragon "
                    "that waits for the window into the ISS plane. Spent stages fall back or "
                    "stay in orbit.",
        epoch=EPOCH, forces=ForceModel(j2=True, drag=True), warp=10, record_dt=5.0,
        satellites=[iss],
        stations=[DEFAULT_STATIONS[0], DEFAULT_STATIONS[2], DEFAULT_STATIONS[4]],
        launches=launches_)
    # fly every launch once and store the optimized pitch kick, so loading is quick
    sim = Simulation(sc)
    kicks = {a.spec.name: a.kick_deg for a in sim.ascents}
    for spec in sc.launches:
        spec.kick = round(kicks[spec.name], 3)
    return sc


def geo_drift():
    lons = [-165, -120, -75, -30, 15, 60, 105, 150]
    return Scenario(
        name="Geostationary drift",
        description="Eight geostationary satellites 45 deg apart with no station-keeping. "
                    "The slightly elliptical equator (C22/S22) pulls each one toward 75 E or "
                    "105 W, where it would rock back and forth. Best in the Earth-fixed frame "
                    "(E) with the map (M); a day passes every second.",
        epoch=EPOCH, forces=ForceModel(j2=True, c22=True), warp=86400,
        integrator=IntegratorSettings(h_max=600.0),
        satellites=[SatSpec(f"GEO {lon:+d}", {"type": "geo", "lon": lon}, mass=3000, area=30)
                    for lon in lons])


ALL = {"default": default, "data/sentinel1_replay": sentinel1_replay,
       "orbit_tour": orbit_tour, "hohmann_to_geo": hohmann,
       "gto_to_geo": gto_to_geo, "rendezvous": rendezvous,
       "starlink_shell": starlink, "gps_constellation": gps, "polar_star": iridium,
       "drag_decay": drag, "j2_precession": j2, "molniya_tundra": heo,
       "launches_and_arcs": launches, "launch_day": launch_day, "geo_drift": geo_drift}

# Planner results differ in the last bits between platforms (libm, numpy SIMD
# paths): 1e-14 m/s on a 7.5 km/s subtraction. Any edit to a scenario changes
# a value by far more than this.
FLOAT_TOL = 1e-9


def differences(built, saved, where: str = "") -> list[str]:
    """Paths at which a generated scenario and a saved one differ; floats
    within FLOAT_TOL (relative or absolute) count as equal."""
    if isinstance(built, bool) or isinstance(saved, bool):
        return [] if built is saved else [where or "/"]
    if isinstance(built, (int, float)) and isinstance(saved, (int, float)):
        same = math.isclose(built, saved, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)
        return [] if same else [f"{where}: {built!r} != {saved!r}"]
    if isinstance(built, dict) and isinstance(saved, dict):
        out = [f"{where}/{k}: missing from one side" for k in sorted(built.keys() ^ saved.keys())]
        for k in built.keys() & saved.keys():
            out += differences(built[k], saved[k], f"{where}/{k}")
        return out
    if isinstance(built, list) and isinstance(saved, list):
        if len(built) != len(saved):
            return [f"{where}: {len(built)} items != {len(saved)}"]
        out = []
        for i, (a, b) in enumerate(zip(built, saved, strict=True)):
            out += differences(a, b, f"{where}/{i}")
        return out
    return [] if built == saved and type(built) is type(saved) else [
        f"{where}: {built!r} != {saved!r}"]


def stale(name: str) -> list[str]:
    """How scenarios/<name>.json differs from what the generator makes now."""
    path = OUT / f"{name}.json"
    if not path.exists():
        return [f"{path.name} is missing"]
    built = json.loads(json.dumps(ALL[name]().to_dict()))
    return differences(built, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in ALL.items():
        sc = fn()
        sc.save(OUT / f"{name}.json")
        print(f"wrote scenarios/{name}.json  ({sc.name})")
