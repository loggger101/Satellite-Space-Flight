"""The orbital lifetime estimate (satflight/lifetime.py) against runs flown to
re-entry, its exact scalings, its edge cases and its rows in the Orbit tab."""

import math
import os
import time
from pathlib import Path

import pytest

from satflight import lifetime as lt
from satflight.analysis import osculating_to_mean_a
from satflight.constants import R_EARTH
from satflight.elements import rv2coe
from satflight.forces import ForceModel
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import ACTIVE, Simulation

ROOT = Path(__file__).resolve().parents[1]


# the seven-regime tour these tests were written against (the app itself opens
# scenarios/default.json, Sentinel-1, when given no file)
TOUR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scenarios", "orbit_tour.json")


def _estimate_and_flight(orbit, mass, area):
    """(estimated, flown) seconds to re-entry for one satellite with J2 and drag."""
    sim = Simulation(Scenario(name="t", satellites=[SatSpec("x", orbit, mass=mass, area=area)],
                              forces=ForceModel(j2=True, drag=True)))
    s = sim.sats[0]
    el = rv2coe(sim.y[0, :3], sim.y[0, 3:])
    a_mean = float(osculating_to_mean_a(el.a, el.e, el.i, el.argp, el.nu))
    est = lt.lifetime(a_mean, el.e, el.i, s.cd * s.area / s.mass)
    while s.status == ACTIVE and sim.t < 2 * est:
        sim.advance(6 * 3600)
    down = [e.t for e in sim.events if "re-entered" in e.text]
    assert down, f"still up after {sim.t / 86400:.1f} days (estimate {est / 86400:.2f})"
    return est, down[0]


@pytest.mark.parametrize("orbit, mass, area", [
    ({"type": "elements", "altitude": 300, "i": 97}, 20, 2),       # polar: J2 lifts it ~5 km
    ({"type": "elements", "altitude": 250, "i": 51.6}, 50, 4),
    ({"type": "elements", "perigee_alt": 200, "apogee_alt": 600, "i": 28.5}, 30, 6),
], ids=["polar", "iss-like", "eccentric"])
def test_the_estimate_matches_a_run_flown_to_reentry(orbit, mass, area):
    est, flown = _estimate_and_flight(orbit, mass, area)
    assert est == pytest.approx(flown, rel=0.08), (est / 86400, flown / 86400)


def test_the_lifetime_goes_exactly_as_one_over_the_density_and_the_drag_area():
    """The decay rates are proportional to rho Cd A / m, so the same path in
    (a, e) is flown at a different speed: what the solar max / min row uses."""
    a, e, i = R_EARTH + 420, 0.004, math.radians(51.6)
    base = lt.lifetime(a, e, i, 0.01)
    assert lt.lifetime(a, e, i, 0.01, density_scale=3.0) == pytest.approx(base / 3, rel=1e-9)
    assert lt.lifetime(a, e, i, 0.04) == pytest.approx(base / 4, rel=1e-9)
    assert 0.5 * lt.YEAR < base < 10 * lt.YEAR          # a few years from 420 km at 0.01 m^2/kg


def test_higher_orbits_last_longer():
    i = math.radians(53)
    lives = [lt.lifetime(R_EARTH + h, 0.0, i, 0.02) for h in (300, 400, 500, 600)]
    assert lives == sorted(lives) and lives[-1] > 20 * lives[0]


def test_edge_cases():
    i = math.radians(30)
    assert lt.lifetime(R_EARTH + 110, 0.0, i, 0.02) == 0.0           # already coming down
    assert lt.lifetime(R_EARTH + 3000, 0.0, i, 0.02) == math.inf     # no air up there
    assert lt.lifetime(R_EARTH + 1200, 0.0, i, 0.001) == math.inf    # > MAX_YEARS
    assert lt.lifetime(R_EARTH + 400, 0.0, i, 0.0) == math.inf       # no drag area
    with pytest.raises(ValueError):
        lt.lifetime(R_EARTH + 400, 0.0, i, -0.01)
    with pytest.raises(ValueError):
        lt.decay_rates(R_EARTH + 400, 1.2, i, 0.01)


def test_the_deorbit_rule_check():
    assert lt.rule_check(3 * lt.YEAR) == "meets FCC 5 yr, IADC 25 yr"
    assert lt.rule_check(12 * lt.YEAR) == "meets IADC 25 yr only"
    assert lt.rule_check(30 * lt.YEAR) == lt.rule_check(math.inf) == "meets neither rule"


def _settled(app, timeout=30.0):
    """app.lifetime() once the background estimate has arrived."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        life = app.lifetime()
        if life != "estimating...":
            return life
        time.sleep(0.02)
    raise AssertionError("no lifetime estimate within the time limit")


def test_the_orbit_tab_shows_the_lifetime(tmp_path):
    pygame = pytest.importorskip("pygame")
    from satflight.ui import glossary
    from satflight.ui.app import App
    from satflight.ui.orbitpanel import lifetime_rows, property_sections

    app = App(TOUR, size=(1400, 850))
    app.user_scenario_dir = tmp_path / "user"
    try:
        app.load_scenario(ROOT / "scenarios" / "drag_decay.json")
        app.select(app.sim.index_of("Balloon"))
        life = _settled(app)
        assert life == pytest.approx(8.3 * 3600, rel=0.08)   # flown: down at 8.3 h
        app.sim.advance(3600)
        later = _settled(app)
        assert later < life
        rows = lifetime_rows(later, app.sim.forces.density_scale)
        assert [r[0] for r in rows] == ["Lifetime",
                                        "At solar max / min", "Deorbit rules"]
        assert rows[0][1].endswith("hours") and rows[2][1].startswith("meets FCC")
        info = app.orbit_info()
        sheet = [r for _, rs in property_sections(info, 0.0, life=later) for r in rs]
        assert all(glossary.row(label) for label, _ in sheet)

        app.load_scenario(ROOT / "scenarios" / "orbit_tour.json")
        app.select(0)
        assert app.lifetime() == "no drag in this run"
    finally:
        pygame.quit()
