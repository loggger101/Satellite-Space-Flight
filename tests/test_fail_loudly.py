"""Failures surface: a broken state or step raises, and work the simulation
drops is written to the event log. Each test here passed silently before."""

import math

import numpy as np
import pytest

from satflight.forces import ForceModel
from satflight.integrators import Propagator
from satflight.maneuvers import Maneuver, propellant_after
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation


def leo(name="A", altitude=500.0, **kw):
    return SatSpec(name, {"type": "elements", "altitude": altitude, "i": 51.6}, **kw)


def sim_of(*sats, **kw):
    return Simulation(Scenario(name="t", satellites=list(sats), **kw))


def test_a_nan_derivative_stops_the_integrator():
    def f(t, y):
        return np.full_like(y, np.nan) if t > 5.0 else np.ones_like(y)
    with pytest.raises(FloatingPointError, match="step size"):
        Propagator("dopri5").integrate(f, 0.0, np.zeros((1, 6)), 60.0)


def test_a_nan_state_stops_the_simulation():
    for propagator in ("cowell", "kepler", "j2mean"):
        sim = sim_of(leo("A"), leo("B", 700), propagator=propagator)
        sim.advance(60)
        sim.y[1, 4] = np.nan
        with pytest.raises(FloatingPointError, match="B"):
            sim.advance(120)


@pytest.mark.parametrize("dt", [-1.0, math.nan, math.inf])
def test_advance_refuses_impossible_spans(dt):
    with pytest.raises(ValueError):
        sim_of(leo()).advance(dt)


def test_many_ascents_in_one_advance_reach_the_end():
    """Ascents sync every second; one long advance used to give up after
    10000 boundaries and leave the physics behind the reported time."""
    launches = [dict(name=f"L{k}", vehicle="Falcon 9 (approx.)", timing="delay",
                     delay=60.0 + 400.0 * k, track_stages=False, kick=1.5) for k in range(30)]
    sc = Scenario.from_dict({"name": "busy pad", "launches": launches})
    one, chunked = Simulation(sc), Simulation(sc.copy())
    one.advance(4 * 3600)
    for _ in range(24):
        chunked.advance(600)
    assert one._grid[0] == chunked._grid[0]
    assert np.array_equal(one.y, chunked.y)


def test_a_reentry_reports_the_maneuvers_it_cancels():
    sim = sim_of(leo("Low", altitude=95.0, mass=100, area=50), forces=ForceModel(drag=True))
    sim.schedule(Maneuver("Low", "impulse", timing="delay", delay=3 * 3600, dv=(0.01, 0, 0)))
    sim.advance(3 * 3600)
    assert any("1 pending maneuver cancelled" in e.text for e in sim.events)


def test_match_velocity_with_a_removed_target_is_reported():
    sim = sim_of(leo("A"), leo("B", 520))
    sim.schedule(Maneuver("A", "match_velocity", timing="delay", delay=600, target="B"))
    sim.remove(sim.index_of("B"))
    sim.advance(900)
    assert any("B no longer exists" in e.text and e.kind == "warn" for e in sim.events)


def test_a_removed_satellite_takes_its_maneuvers_and_burns_along():
    sim = sim_of(leo("A"), leo("B", 520))
    sim.schedule(Maneuver("A", "finite", dv=(1, 0, 0), thrust=400, duration=600))
    sim.advance(60)
    assert sim.burns
    sim.remove(sim.index_of("A"))
    assert not sim.burns and not sim.maneuvers
    sim.advance(900)                    # nothing left that names A


def test_rocket_equation_refuses_a_zero_isp():
    with pytest.raises(ValueError, match="Isp"):
        propellant_after(100.0, 0.1, 0.0)
