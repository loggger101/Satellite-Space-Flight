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
from satflight.keeping import TrackKeeping  # noqa: E402
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
# Sentinel-1C flies 1A's orbit about half a revolution behind it (where 1B
# was), so the pair covers each ground track every 6 days: 180 deg nominally,
# 176.3 deg in ESA's precise orbits (mean July 2025 - May 2026, 175.9-176.7;
# docs/analysis/sentinel1/phasing.csv).
S1_REVS, S1_DAYS = 175, 12
S1_LTAN = 18.0                     # h, local mean solar time of the ascending node
# Mean altitude (km above R_EARTH) whose node-to-node period, flown with J2, J3
# and J4 in full on the "sso" inclination they pick, is 12 d / 175 = 5924.571 s:
# the repeat closes to ~2 ms after 175 revolutions (tests/test_sentinel1.py
# measures it). The published mean semi-major axis is 7071 km (693 km).
S1_ALT = 692.827267
# Frozen eccentricity, the mean over a revolution as the precise orbits show it
# (both satellites, July 2025 - May 2026; daily_S1*.csv), argp 90 deg. The
# first-order J2 + J3 value, -J3 Re sin(i) / (2 J2 a) = 1.0443e-3, is 12 % low.
S1_E_MEAN = 0.0011837
S1_PHASE = 176.3                   # deg, 1C behind 1A


def s1_plane_offset(phase: float = S1_PHASE) -> float:
    """How far (deg of RAAN, + east) 1C's plane turns from 1A's so that 1C,
    trailing by ``phase`` deg, flies 1A's ground tracks. Each orbit's node lands
    360 * S1_DAYS / S1_REVS deg (12 tracks) farther west, so a trailer half a
    revolution back lands exactly 6 tracks over; one ``phase`` back lands
    (phase/360 - 1/2) of that short, which the plane makes up: -0.254 deg
    (1C's node a minute of local time behind 1A's), as the precise orbits show."""
    return (phase / 360.0 - 0.5) * 360.0 * S1_DAYS / S1_REVS


def _mean_e_vector(sim, k, revs=2, step=10.0):
    """Satellite ``k``'s (e cos argp, e sin argp) averaged over whole
    revolutions, node to node, flown with the scenario's forces."""
    import numpy as np

    from satflight.elements import rv2coe
    from satflight.fit import fly
    s = sim.scenario.satellites[k]
    t = np.arange(0.0, (revs + 1.1) * S1_DAYS * 86400.0 / S1_REVS, step)
    rv = fly(sim.scenario.forces, sim.clock.jd(0.0), 0.0, sim.y[k], t,
             s.cd * s.area / s.mass, 0.0, states=True)
    z = rv[:, 0, 2]
    up = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))
    lo, hi = up[0] + 1, up[revs] + 1
    e = rv2coe(rv[lo:hi, 0], rv[lo:hi, 1])
    span = t[hi - 1] - t[lo]
    return np.array([np.trapezoid(e.e * np.cos(e.argp), t[lo:hi]) / span,
                     np.trapezoid(e.e * np.sin(e.argp), t[lo:hi]) / span])


def frozen_start(build, k, target, tol=1e-7, max_iter=8):
    """The osculating (e, argp deg) at the epoch that flies the mean
    eccentricity vector ``target``: J2 swings e by ~1e-3 twice a revolution,
    so the start differs from the mean, and by where the satellite starts.
    ``build(osc)`` makes the scenario with osculating vectors ``osc`` (one per
    satellite); the others stay as given. Fixed point: the short-period part
    hardly changes as the start moves by 1e-4."""
    import numpy as np
    target = np.asarray(target, float)
    osc = [target.copy() for _ in range(2)]
    for _ in range(max_iter):
        miss = target - _mean_e_vector(Simulation(build(osc)), k)
        osc[k] = osc[k] + miss
        if np.linalg.norm(miss) < tol:
            return math.hypot(*osc[k]), math.degrees(math.atan2(osc[k][1], osc[k][0]))
    raise RuntimeError(f"satellite {k}: mean e not within {tol} after {max_iter} passes")


def s1_raan(epoch: datetime) -> float:
    """RAAN (deg) that puts the ascending node at S1_LTAN local mean time at ``epoch``."""
    clock = Clock(epoch)
    ut = epoch.hour + epoch.minute / 60 + epoch.second / 3600
    return (math.degrees(clock.gmst(0.0)) + (S1_LTAN - ut) * 15.0) % 360.0


def default():
    """Sentinel-1A and -1C on ESA's reference orbit, with the eccentricity and
    phasing their precise orbits show."""
    raan = s1_raan(EPOCH)
    raans = [raan, (raan + s1_plane_offset()) % 360.0]   # 1C on 1A's ground tracks
    us = [0.0, (-S1_PHASE) % 360.0]   # argument of latitude at the epoch, deg

    def build(osc):
        def s1(name, u, raan, mass, v):
            e, argp = math.hypot(*v), math.degrees(math.atan2(v[1], v[0]))
            return SatSpec(name, el(altitude=S1_ALT, e=round(e, 9), i="sso",
                                    raan=round(raan, 6), argp=round(argp, 6),
                                    nu=round((u - argp) % 360.0, 6)),
                           color=None, mass=mass, area=10.0, cd=2.2)
        return _default_scenario([s1("Sentinel-1A", us[0], raans[0], 2164.0, osc[0]),
                                  s1("Sentinel-1C", us[1], raans[1], 2200.0, osc[1])])

    def vec(e, w):
        return e * math.cos(math.radians(w)), e * math.sin(math.radians(w))

    target = (0.0, S1_E_MEAN)          # argp 90 deg
    first = vec(*frozen_start(build, 0, target))
    # The phase and the plane offset are what the precise orbits measure from
    # the nodes (when each satellite crosses, and where on the ground); the start
    # sets osculating elements, whose J2 short-period swing differs at u = 0 and
    # u = 183.7 deg (0.27 deg of phase, 2 km on the ground). So fly a day and move
    # 1C along its orbit and turn its plane until its nodes come S1_PHASE behind
    # 1A's and land on 1A's tracks (and find 1C's frozen start again where it is).
    for _ in range(6):
        osc = [first, vec(*frozen_start(build, 1, target))]
        phase, miss = _flown_phase_and_miss(Simulation(build(osc)))
        if abs(phase - S1_PHASE) < 0.002 and abs(miss) * 111.32 < 0.05:
            sc = build(osc)            # calibrated free of burns, then kept on track
            sc.track_keeping = s1_keeping([s.name for s in sc.satellites])
            return sc
        us[1] = (us[1] + phase - S1_PHASE) % 360.0
        raans[1] = (raans[1] - miss) % 360.0
    raise RuntimeError(f"Sentinel-1C still {phase:.3f} deg behind 1A and "
                       f"{miss * 111.32:.2f} km off its tracks")


# ESA holds each Sentinel-1 on its reference tracks: on a typical day of the
# precise orbits its worst crossing lands 31-32 m off (the median of the daily
# largest offsets), with along-track trims every 3.5-4 days of about 7 m in a
# (docs/analysis/sentinel1/report.md). A +-50 m band holds the simulated pair as
# tight; the table atmosphere's milder drag spaces its burns about a week apart.
S1_TRACK_BAND_KM = 0.05


def s1_keeping(names, start=0.0):
    """Ground-track keeping for the Sentinel-1 satellites ``names`` on the
    175-orbit, 12-day reference cycle, from ``start`` (s after the epoch)."""
    return [TrackKeeping(n, band_km=S1_TRACK_BAND_KM, orbits=175, days=12,
                         start=start, isp=S1_HYDRAZINE_ISP) for n in names]


def _flown_phase_and_miss(sim, hours=26.0, step=10.0):
    """Flown ``hours`` with the scenario's forces, from the ascending nodes: how
    far (deg) 1C trails 1A (when it crosses after 1A, over 1A's node-to-node
    period) and how far east (deg) its nodes land from 1A's 175-track grid
    (tracks 360/175 deg apart), each averaged over the day."""
    import numpy as np

    from satflight.fit import fly_batch
    from satflight.timeutil import gmst
    sats = sim.scenario.satellites
    t = np.arange(0.0, hours * 3600.0, step)
    jd0 = sim.clock.jd(0.0)
    Y = fly_batch(sim.scenario.forces, jd0, 0.0, sim.y, t,
                  [s.cd * s.area / s.mass for s in sats], [0.0] * len(sats))
    tn, lon = [], []
    for k in range(2):
        r = Y[k, :, :3]
        up = np.flatnonzero((r[:-1, 2] < 0) & (r[1:, 2] >= 0))
        f = -r[up, 2] / (r[up + 1, 2] - r[up, 2])
        rn = r[up] + f[:, None] * (r[up + 1] - r[up])
        tn.append(t[up] + f * step)
        lon.append(np.degrees(np.arctan2(rn[:, 1], rn[:, 0]) - gmst(jd0 + tn[k] / 86400.0)))
    period = np.diff(tn[0]).mean()
    after = tn[1] - tn[0][np.searchsorted(tn[0], tn[1]) - 1]
    after = after[tn[1] > tn[0][0]]
    grid = 360.0 / S1_REVS
    return (float(360.0 * after.mean() / period),
            float(np.mean((lon[1] - lon[0][0] + grid / 2) % grid - grid / 2)))


def _default_scenario(satellites):
    return Scenario(
        name="Sentinel-1A and 1C",
        description="ESA's Copernicus radar pair on the Sentinel-1 reference orbit: "
                    "693 km, sun-synchronous (98.18 deg), dawn-dusk with the ascending "
                    "node at 18:00, frozen perigee over the North Pole, 175 orbits in a "
                    "12-day repeat. 1C flies about half an orbit (176 deg) behind 1A, its "
                    "plane turned 0.25 deg west to put it on 1A's ground tracks, so "
                    "together they revisit each track every 6 days (Telemetry, PHASING). "
                    "Each keeps within 50 m of its tracks with small along-track burns "
                    "about a week apart, as ESA keeps them. "
                    "Open the map (M) to watch the tracks interleave. Reference orbit with the "
                    "eccentricity and spacing ESA's precise orbits show (2025-26), not "
                    "tracking data.",
        epoch=EPOCH, forces=ForceModel(j2=True, j3=True, j4=True, drag=True), warp=120,
        satellites=satellites,
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


# --- Sentinel-1 launch ------------------------------------------------------------------------
# The study's first moment: the first sample of ESA's precise orbits for 1 July 2025.
S1_DATA_START = datetime(2025, 7, 1, 0, 0, 2, tzinfo=UTC)
# 1A and 1C there, in the simulator's ECI frame (km, km/s), read from the precise orbits
# (replay.for_orbit at S1_DATA_START; tests/test_sentinel1_launch.py checks them against
# the downloaded files when they are there).
S1_DATA_START_STATE = {
    "S1A": ((-5249.440911, -1545.403036, -4494.047597), (-4.874137275, 0.003528182, 5.697960208)),
    "S1C": ((5581.886148, 1511.438843, 4068.991851), (4.457220354, -0.14653567, -6.042050241)),
}
S1_LAUNCH_EPOCH = datetime(2025, 6, 25, tzinfo=UTC)
S1_LAUNCH_SITE = "Vandenberg SLC-4E (USA)"
S1_PARKING_ALT = 600.0       # km: Falcon 9 leaves each satellite below its slot
S1_HYDRAZINE_ISP = 220.0     # s, the satellites' own thrusters
S1_TRANSFER_DEG = 170.0      # the last arc's sweep (a half turn is singular to aim)


def _s1_launch_scenario(launches, maneuvers):
    return Scenario(
        name="Sentinel-1 launch",
        description="Sentinel-1A and 1C launched from Vandenberg on Falcon 9 a few days "
                    "before the study's precise orbits start (not the real dates): each "
                    "lifts off southbound into the plane it will fly, is left at 600 km, "
                    "and with three burns of its own thrusters (a phasing burn, an aimed "
                    "transfer and an arrival burn) reaches the place ESA's precise orbit "
                    "puts it at 00:00:02 UTC on 1 July 2025, from where the study begins. "
                    "From then on each keeps within 50 m of its ground tracks with small "
                    "along-track burns. Watch the PHASING rows settle at 176 deg once "
                    "both have arrived.",
        epoch=S1_LAUNCH_EPOCH, forces=ForceModel(j2=True, j3=True, j4=True, drag=True),
        warp=60, record_dt=10.0, satellites=[], launches=launches, maneuvers=maneuvers,
        stations=[GroundStation("Svalbard", 78.23, 15.39, 0.5, 5.0),
                  GroundStation("Matera", 40.65, 16.70, 0.5, 5.0),
                  GroundStation("Maspalomas", 27.76, -15.63, 0.2, 5.0),
                  GroundStation("Kiruna", 67.86, 20.96, 0.4, 5.0)])


def _s1_launch_spec(name, mass, t0):
    lat, lon, alt = LAUNCH_SITES[S1_LAUNCH_SITE]
    return LaunchSpec(name, vehicle_preset("Falcon 9 (approx.)"), site=S1_LAUNCH_SITE,
                      lat=lat, lon=lon, alt=alt, timing="absolute", t0=round(t0, 3),
                      perigee_alt=S1_PARKING_ALT, apogee_alt=S1_PARKING_ALT,
                      inclination=98.18, direction="south", payload_mass=mass,
                      payload_area=10.0, payload_cd=2.2)


def _s1_window(raan_deg, after, clock):
    """Liftoff time (s after S1_LAUNCH_EPOCH) of the first southbound window
    from Vandenberg after ``after`` into the plane of RAAN ``raan_deg`` at i
    98.18 deg, the plane fixed in space."""
    from satflight.launch import next_window, plane_normal
    lat, lon, alt = LAUNCH_SITES[S1_LAUNCH_SITE]
    n = plane_normal(math.radians(98.18), math.radians(raan_deg))
    return next_window(lat, lon, alt, lambda t: n, after, lambda t: float(clock.gmst(t)),
                       "south")


def _s1_signed_angle(r_from, r_to, h):
    """Angle (deg, 0..360) from ``r_from`` to ``r_to`` the way the orbit of
    angular momentum ``h`` runs."""
    import numpy as np
    a = math.degrees(math.atan2(np.dot(np.cross(r_from, r_to), h / np.linalg.norm(h)),
                                np.dot(r_from, r_to)))
    return a % 360.0


def _s1_plan_arrival(sim, name, target):
    """The three burns that take ``name`` (separated, in its parking orbit) to
    the state ``target`` at S1_DATA_START: a phasing burn along the velocity an
    orbit after now, sized so that at the transfer's start the slot lies
    S1_TRANSFER_DEG ahead; the transfer, aimed through the run's forces to
    arrive on the slot's position; and the arrival burn onto its velocity.
    Returns (burns [(t, dv ECI)], the plane's miss deg: payload RAAN at the
    transfer minus the slot's carried back there)."""
    import numpy as np

    from satflight.analysis import j2_secular_rates
    from satflight.constants import MU_EARTH
    from satflight.maneuvers import propellant_after
    i = sim.index_of(name)
    m0 = sim.sats[i].mass           # each arc is flown at the mass left after its burn:
    #                                 drag goes as area over mass, and the burns spend 5 %
    T = (S1_DATA_START - S1_LAUNCH_EPOCH).total_seconds()
    r2, v2 = (np.array(x, float) for x in target)
    t1 = sim.t + 2 * math.pi * math.sqrt(np.linalg.norm(sim.y[i, :3]) ** 3 / MU_EARTH)
    y1 = sim.coast_states([i], sim.y[i][None], sim.t, t1)[0]

    # the transfer starts at a fixed moment: a 170-deg arc from the parking radius
    a_tr = 0.5 * (np.linalg.norm(y1[:3]) + np.linalg.norm(r2))
    tof = S1_TRANSFER_DEG / 360.0 * 2 * math.pi * math.sqrt(a_tr ** 3 / MU_EARTH)
    tb = T - tof
    along = y1[3:] / np.linalg.norm(y1[3:])

    def misses(dvs):
        """For each phasing burn (km/s along the velocity), flown together to the
        transfer: how far (deg) the slot then lies from S1_TRANSFER_DEG ahead."""
        sim.sats[i].mass = propellant_after(m0, float(np.mean(dvs)), S1_HYDRAZINE_ISP)
        Y = np.repeat(y1[None], len(dvs), axis=0)
        Y[:, 3:] += np.outer(dvs, along)
        Yb = sim.coast_states([i] * len(dvs), Y, t1, tb)
        f = np.array([_s1_signed_angle(y[:3], r2, np.cross(y[:3], y[3:])) for y in Yb])
        return (f - S1_TRANSFER_DEG + 180.0) % 360.0 - 180.0, Yb

    def bracket(lo, hi, last=True):
        grid = np.linspace(lo, hi, 25)
        f, _ = misses(grid)
        roots = [k for k in range(len(grid) - 1)
                 if f[k] * f[k + 1] <= 0 and abs(f[k] - f[k + 1]) < 180.0]
        if not roots:
            return None
        k = roots[-1] if last else roots[0]
        return grid[k], grid[k + 1], f[k], f[k + 1]

    # the highest phasing orbit (up to 48 m/s: still below the slot) that brings it round
    b = bracket(0.0, 0.048)
    if b is None:
        raise RuntimeError(f"{name}: no phasing burn up to 48 m/s brings it to its slot")
    for _ in range(2):                         # narrow it twice, flying 25 together
        b = bracket(b[0], b[1], last=False)
    lo, hi, flo, fhi = b
    dv1 = lo - flo * (hi - lo) / (fhi - flo) if fhi != flo else lo
    _, Yb = misses(np.array([dv1]))
    sim.sats[i].mass = m0
    r1, v1 = Yb[0][:3], Yb[0][3:]
    # the plane: the payload's RAAN at the transfer against the slot's carried back there
    from satflight.elements import rv2coe
    el_p, el_s = rv2coe(r1, v1), rv2coe(r2, v2)
    rd = float(j2_secular_rates(el_s.a, el_s.e, el_s.i)[0])
    miss_raan = math.degrees(math.remainder(el_p.raan - (el_s.raan - rd * (T - tb)), 2 * math.pi))
    return (round(t1, 3), dv1 * along), round(tb, 3), miss_raan


def _s1_aim_transfer(sim, name, target, tb):
    """The transfer and arrival burns [(t, dv ECI)] for ``name``, the run standing
    at the transfer's start ``tb``: aimed from the state the run has flown it to
    (five days of phasing coast drift some 170 m from any prediction, while the
    45-minute transfer lands within centimetres of its own)."""
    import numpy as np

    from satflight.constants import MU_EARTH
    from satflight.maneuvers import propellant_after
    from satflight.planner import aim
    i = sim.index_of(name)
    T = (S1_DATA_START - S1_LAUNCH_EPOCH).total_seconds()
    r2, v2 = (np.array(x, float) for x in target)
    r1, v1 = sim.y[i, :3].copy(), sim.y[i, 3:].copy()
    m1 = sim.sats[i].mass
    a_tr = 0.5 * (np.linalg.norm(r1) + np.linalg.norm(r2))
    speed = math.sqrt(MU_EARTH * (2.0 / np.linalg.norm(r1) - 1.0 / a_tr))
    v_dep = v1 * speed / np.linalg.norm(v1)
    for _ in range(3):                   # the transfer's own burn sets the arc's mass
        sim.sats[i].mass = propellant_after(m1, float(np.linalg.norm(v_dep - v1)),
                                            S1_HYDRAZINE_ISP)
        v_dep, arrival = aim(sim, name, r1, v_dep, tb, T - tb, r2)
    sim.sats[i].mass = m1
    return [(tb, v_dep - v1), (T, v2 - arrival[3:])]


def sentinel1_launch():
    """Sentinel-1A and 1C launched a few days before the study starts and flown
    onto their precise-orbit states at S1_DATA_START (see the scenario text)."""
    import numpy as np

    from satflight.maneuvers import Maneuver
    clock = Clock(S1_LAUNCH_EPOCH)
    T = (S1_DATA_START - S1_LAUNCH_EPOCH).total_seconds()
    sso = math.degrees(2 * math.pi / (365.2421897 * 86400.0))        # deg/s
    plan = {}
    for name, mission, mass, day in (("Sentinel-1A", "S1A", 2164.0, 0),
                                     ("Sentinel-1C", "S1C", 2200.0, 1)):
        from satflight.elements import rv2coe
        r, v = S1_DATA_START_STATE[mission]
        raan_T = math.degrees(float(rv2coe(r, v).raan))
        # the plane at liftoff: the slot's carried back sun-synchronously, plus what
        # flying it shows the parking orbit's own regression adds
        t0, fix = day * 86400.0, 0.0
        for _ in range(6):
            for _ in range(2):                     # the window moves the plane a little
                t0 = _s1_window(raan_T - sso * (T - t0) + fix, day * 86400.0, clock)
            spec = _s1_launch_spec(name, mass, t0)
            sim = Simulation(_s1_launch_scenario([spec], []))
            spec.kick = round(sim.ascents[0].kick_deg, 3)
            sim.advance(t0 + 1800.0 - sim.t)       # separated, in the parking orbit
            if sim.ascent_of(sim.index_of(name)) is not None:
                raise RuntimeError(f"{name}: not in orbit 30 min after liftoff")
            phase, tb, miss = _s1_plan_arrival(sim, name, S1_DATA_START_STATE[mission])
            print(f"  {name}: liftoff {t0:.0f} s, plane miss {miss:+.4f} deg, phasing "
                  f"{1e3 * float(np.linalg.norm(phase[1])):.1f} m/s")
            if abs(miss) < 0.002:
                break
            fix -= miss
        else:
            raise RuntimeError(f"{name}: the plane still misses by {miss:.4f} deg")
        plan[name] = (spec, phase, tb, mission)

    def burn(name, t, dv, label):
        return Maneuver(name, "impulse", timing="absolute", t=round(t, 3),
                        dv=tuple(round(float(x), 9) for x in dv), frame="ECI",
                        isp=S1_HYDRAZINE_ISP, label=label)

    # both flown together with their phasing burns, as the scenario will fly them;
    # each transfer is aimed from where that run has them when it starts
    specs = [plan[n][0] for n in plan]
    maneuvers = [burn(n, *plan[n][1], "phasing burn") for n in plan]
    sim = Simulation(_s1_launch_scenario(specs, maneuvers))
    for name in sorted(plan, key=lambda n: plan[n][2]):
        _, _, tb, mission = plan[name]
        sim.advance(tb - sim.t)
        (t2, dv2), (t3, dv3) = _s1_aim_transfer(sim, name, S1_DATA_START_STATE[mission], tb)
        print(f"  {name}: transfer {1e3 * float(np.linalg.norm(dv2)):.1f} m/s, arrival "
              f"{1e3 * float(np.linalg.norm(dv3)):.1f} m/s")
        new = [burn(name, t2, dv2, "transfer to the slot"),
               burn(name, t3, dv3, "arrival in the slot")]
        for m in new:
            sim.schedule(Maneuver.from_dict(m.to_dict()))   # as a loaded run would
        maneuvers += new
    sc = _s1_launch_scenario(specs, sorted(maneuvers, key=lambda m: (m.sat, m.t)))
    sc.track_keeping = s1_keeping(list(plan), start=T)      # once they are in their slots
    return sc


ALL = {"default": default, "data/sentinel1_replay": sentinel1_replay,
       "orbit_tour": orbit_tour, "hohmann_to_geo": hohmann,
       "gto_to_geo": gto_to_geo, "rendezvous": rendezvous,
       "starlink_shell": starlink, "gps_constellation": gps, "polar_star": iridium,
       "drag_decay": drag, "j2_precession": j2, "molniya_tundra": heo,
       "launches_and_arcs": launches, "launch_day": launch_day, "geo_drift": geo_drift,
       "sentinel1_launch": sentinel1_launch}
# builders that fly days to calibrate (default: the pair's measured phasing) or to aim
# (sentinel1_launch: both ascents, the phasing and the transfers): minutes each to
# rebuild, so selfcheck --quick leaves them out and says so
SLOW = ("default", "sentinel1_launch")

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
