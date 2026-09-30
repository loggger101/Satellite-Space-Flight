"""Regenerate the example scenarios in ``scenarios/``.

    python tools/make_scenarios.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satflight.forces import ForceModel  # noqa: E402
from satflight.planner import plan_hohmann, plan_rendezvous, scan_rendezvous  # noqa: E402
from satflight.scenario import (DEFAULT_STATIONS, ConstellationSpec, GroundStation,  # noqa: E402
                                SatSpec, Scenario, preset_spec)
from satflight.simulation import Simulation  # noqa: E402
from satflight.timeutil import UTC  # noqa: E402

EPOCH = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
OUT = ROOT / "scenarios"


def el(**kw):
    return {"type": "elements", **kw}


def default():
    names = ["ISS (LEO 420 km, 51.6 deg)", "Hubble (540 km, 28.5 deg)", "Sun-synchronous 700 km",
             "GPS MEO (20200 km, 55 deg)", "Geostationary (GEO)", "Molniya (12 h, 63.4 deg)",
             "GTO (200 x 35786 km, 27 deg)"]
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
        satellites=[SatSpec("Transfer", el(altitude=300, i=0.0, raan=0, nu=0), mass=4000, area=15),
                    SatSpec("Parking", el(altitude=300, i=0.0, raan=0, nu=-20), mass=4000, area=15)])
    sim = Simulation(sc)
    sim.advance(600)
    burns, summary = plan_hohmann(sim, "Transfer", 35786.0, "now")
    sc.maneuvers = burns
    sc.description += f" Plan: {summary}."
    return sc


def rendezvous():
    sc = Scenario(
        name="Lambert rendezvous",
        description="A chaser 70 km below and 6 deg behind a station intercepts it with a "
                    "Lambert transfer (time of flight chosen for minimum delta-v with the arc "
                    "kept above the atmosphere), then matches velocity.",
        epoch=EPOCH, forces=ForceModel(j2=False), warp=30,
        satellites=[SatSpec("Station", el(altitude=420, i=51.6, raan=30, nu=40), mass=420000, area=1600),
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
        constellations=[ConstellationSpec("Shell", 550, 53.0, 1584, 72, 17, "delta", [120, 200, 255])],
        stations=list(DEFAULT_STATIONS))


def gps():
    return Scenario(
        name="GPS-like constellation",
        description="Walker delta 55 deg: 24/6/1 at 20180 km, Cowell propagation with J2, Sun and Moon.",
        epoch=EPOCH, forces=ForceModel(j2=True, sun=True, moon=True), warp=600,
        constellations=[ConstellationSpec("GPS", 20180, 55.0, 24, 6, 1, "delta", [255, 196, 64], mass=2000,
                                          area=20)],
        stations=list(DEFAULT_STATIONS))


def iridium():
    return Scenario(
        name="Polar star constellation",
        description="Iridium-like Walker star: 66/6/2 at 780 km, 86.4 deg - planes span 180 deg "
                    "so there is a counter-rotating seam.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=120,
        constellations=[ConstellationSpec("Polar", 780, 86.4, 66, 6, 2, "star", [140, 240, 140])])


def drag():
    return Scenario(
        name="Atmospheric drag decay",
        description="Three spacecraft at 300 km with different ballistic coefficients: a dense "
                    "probe, a CubeSat and a balloon. The balloon re-enters in about a day.",
        epoch=EPOCH, forces=ForceModel(j2=True, drag=True), warp=3600,
        satellites=[SatSpec("Dense probe", el(altitude=300, i=51.6, raan=0, nu=0), mass=2000, area=1.0),
                    SatSpec("CubeSat 3U", el(altitude=300, i=51.6, raan=0, nu=120), mass=4, area=0.03),
                    SatSpec("Balloon", el(altitude=300, i=51.6, raan=0, nu=240), mass=60, area=40)])


def j2():
    incs = [0.0, 30.0, 63.4, 90.0, "sso", 130.0]
    sats = [SatSpec(f"i = {i if isinstance(i, str) else f'{i:g}'}",
                    el(perigee_alt=600, apogee_alt=1400, i=i, raan=0, argp=30, nu=0), mass=500, area=4)
            for i in incs]
    return Scenario(
        name="J2 precession study",
        description="Identical 600 x 1400 km orbits at six inclinations: RAAN regresses fastest at low "
                    "inclination, stands still at 90 deg, and the perigee freezes at 63.4 deg. "
                    "Open the plot panel (G) and pick RAAN or Arg. of perigee.",
        epoch=EPOCH, forces=ForceModel(j2=True), warp=3600, satellites=sats)


def heo():
    return Scenario(
        name="Molniya and Tundra",
        description="Three Molniya and three Tundra satellites sharing ground tracks - best viewed "
                    "in the Earth-fixed frame (E) with the map (M).",
        epoch=EPOCH, forces=ForceModel(j2=True, j3=True, j4=True, sun=True, moon=True), warp=1200,
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
        description="Burnout states: sub-orbital hops at 15/45/75 deg from 100 km over the Cape, "
                    "an orbit insertion over Kourou and a hyperbolic escape at 11.6 km/s.",
        epoch=EPOCH, forces=ForceModel(j2=True, drag=True), warp=30,
        satellites=[SatSpec("Hop 15 deg", dict(cape, speed=4.0, fpa=15.0), mass=500, area=1),
                    SatSpec("Hop 45 deg", dict(cape, speed=4.0, fpa=45.0), mass=500, area=1),
                    SatSpec("Hop 75 deg", dict(cape, speed=4.0, fpa=75.0), mass=500, area=1),
                    SatSpec("Insertion", {"type": "surface", "lat": 5.2, "lon": -52.8, "alt": 200.0,
                                          "speed": 7.4, "azimuth": 90.0, "fpa": 0.0}, mass=5000, area=10),
                    SatSpec("Escape", {"type": "surface", "lat": 28.5, "lon": -80.6, "alt": 300.0,
                                       "speed": 11.6, "azimuth": 90.0, "fpa": 5.0}, mass=1000, area=4)],
        stations=[DEFAULT_STATIONS[0], DEFAULT_STATIONS[4]])


def lunar():
    return Scenario(
        name="Third-body perturbations",
        description="A lunar transfer orbit, a GEO satellite and a high-inclination HEO with the Sun, "
                    "Moon and solar radiation pressure switched on.",
        epoch=EPOCH, forces=ForceModel(j2=True, sun=True, moon=True, srp=True), warp=21600,
        satellites=[preset_spec("Lunar transfer (TLI)", "Lunar transfer"),
                    preset_spec("Geostationary (GEO)", "GEO"),
                    SatSpec("HEO", el(perigee_alt=1000, apogee_alt=120000, i=60, raan=40, argp=90, nu=0),
                            mass=800, area=12, cr=1.8)])


ALL = {"default": default, "hohmann_to_geo": hohmann, "rendezvous": rendezvous,
       "starlink_shell": starlink, "gps_constellation": gps, "polar_star": iridium,
       "drag_decay": drag, "j2_precession": j2, "molniya_tundra": heo,
       "launches_and_arcs": launches, "third_body": lunar}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in ALL.items():
        sc = fn()
        sc.save(OUT / f"{name}.json")
        print(f"wrote scenarios/{name}.json  ({sc.name})")
