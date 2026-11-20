"""Orbit fitting (satflight.fit): batch least squares for the start state,
Cd*A/m, Cr*A/m and impulsive burns, checked on synthetic truth flown with the
simulator's own forces (so the fit must recover the values it was made
with), then on the real Sentinel-1 files when they are on this machine."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pytest

from satflight import fit
from satflight.elements import coe2rv
from satflight.forces import ForceModel
from satflight.timeutil import UTC, julian_date

JD0 = julian_date(datetime(2025, 7, 3, tzinfo=UTC))
MODEL = ForceModel(gravity_degree=4, drag=True, sun=True, moon=True, srp=True)
R0, V0 = coe2rv(7071.0, 0.0011, math.radians(98.18), math.radians(278.3), math.radians(90.0),
                0.4)
Y0 = np.concatenate([R0, V0])
CD_AM, CR_AM = 0.0102, 0.0060           # m^2/kg, about Sentinel-1's


def truth(hours, burns=(), cd_am=CD_AM, cr_am=CR_AM, step=120.0):
    """Synthetic observations: the orbit flown with known values."""
    t = np.arange(0.0, hours * 3600.0 + 1e-9, step)
    r = fit.fly(MODEL, JD0, 0.03, Y0, t, cd_am, cr_am, burns)
    return fit.Observations(t, r, np.ones(len(t), bool), JD0, 0.03)


@pytest.fixture(scope="module")
def clean():
    """Six clean hours fitted from a start 100 m and 0.1 m/s off, drag 40 % off."""
    obs = truth(6)
    guess = Y0 + np.array([0.1, -0.08, 0.05, 1e-4, -8e-5, 5e-5])
    return obs, fit.fit_arc(obs, MODEL, guess, cd_am=CD_AM * 1.4, cr_am=CR_AM, fit_cr=False)


def test_a_clean_arc_gives_back_the_state_and_drag_it_was_made_with(clean):
    obs, res = clean
    assert np.linalg.norm(res.y0[:3] - Y0[:3]) < 1e-3                 # km: under 1 m
    assert np.linalg.norm(res.y0[3:] - Y0[3:]) < 1e-6                 # km/s: under 1 mm/s
    assert res.cd_am == pytest.approx(CD_AM, rel=0.03)
    assert res.rms_m < 0.05 and res.converged
    assert res.residuals_rsw.shape == (len(obs.t), 3)


def test_a_burn_inside_the_arc_is_fitted_as_an_impulse():
    dv = np.array([0.0, 0.0, 0.0])
    t_burn = 2.5 * 3600.0
    # 4 mm/s along the velocity at the burn, the size of a Sentinel-1 orbit trim
    r_b, v_b = fit.fly(MODEL, JD0, 0.03, Y0, np.array([0.0, t_burn]), CD_AM, CR_AM, (),
                       states=True)[-1]
    dv = 4e-6 * v_b / np.linalg.norm(v_b)
    obs = truth(6, burns=[(t_burn, dv)])
    plain = fit.fit_arc(obs, MODEL, Y0, cd_am=CD_AM, cr_am=CR_AM, fit_cr=False)
    assert plain.rms_m > 3.0                                          # a burn it cannot explain
    res = fit.fit_arc(obs, MODEL, Y0, cd_am=CD_AM, cr_am=CR_AM, fit_cr=False, burns=[t_burn])
    assert res.rms_m < 0.05
    assert np.linalg.norm(res.dv[0] - dv) < 0.02 * np.linalg.norm(dv)
    assert res.burns[0].t == t_burn
    assert res.burns[0].rsw[1] == pytest.approx(4.0, rel=0.02)        # mm/s along track


def test_masked_samples_do_not_count():
    obs = truth(3)
    bad = obs.r.copy()
    bad[40:60] += 5.0                                                 # 5 km wrong
    use = obs.use.copy()
    use[40:60] = False
    res = fit.fit_arc(fit.Observations(obs.t, bad, use, JD0, 0.03), MODEL, Y0, cd_am=CD_AM,
                      cr_am=CR_AM, fit_cr=False)
    assert res.rms_m < 0.05 and res.used == len(obs.t) - 20


def test_prediction_flies_the_fitted_solution_on(clean):
    _, res = clean
    later = np.array([12.0, 24.0]) * 3600.0
    r = fit.fly(MODEL, JD0, 0.03, res.y0, np.concatenate([[0.0], later]), res.cd_am,
                res.cr_am, ())[1:]
    want = fit.fly(MODEL, JD0, 0.03, Y0, np.concatenate([[0.0], later]), CD_AM, CR_AM, ())[1:]
    assert np.linalg.norm(r - want, axis=1).max() < 0.01              # km after a day


def test_later_burns_are_estimated_with_everything_else_held():
    """Prediction past the fitted span: the operators' burns are known to have
    happened, so each gets its delta-v from the truth while the fitted state,
    drag and reflectivity stay as fitted. What is left is the model's error."""
    t_burn = 4.0 * 3600.0
    v_b = fit.fly(MODEL, JD0, 0.03, Y0, np.array([0.0, t_burn]), CD_AM, CR_AM, (),
                  states=True)[-1][1]
    dv = -3e-6 * v_b / np.linalg.norm(v_b)
    obs = truth(6, burns=[(t_burn, dv)])
    first = obs.t <= 3 * 3600.0
    early = fit.Observations(obs.t[first], obs.r[first], obs.use[first], JD0, 0.03)
    res = fit.fit_arc(early, MODEL, Y0, cd_am=CD_AM, cr_am=CR_AM, fit_cr=False)
    held = fit.fit_arc(obs, MODEL, res.y0, cd_am=res.cd_am, cr_am=res.cr_am, fit_state=False,
                       fit_cd=False, fit_cr=False, burns=[t_burn])
    assert np.array_equal(held.y0, res.y0) and held.cd_am == res.cd_am
    assert np.linalg.norm(held.dv[0] - dv) < 0.02 * np.linalg.norm(dv)
    assert held.rms_m < 0.5
    none = fit.fit_arc(early, MODEL, res.y0, cd_am=res.cd_am, cr_am=res.cr_am,
                       fit_state=False, fit_cd=False, fit_cr=False)
    assert none.iterations == 0 and none.rms_m == pytest.approx(res.rms_m, abs=1e-3)


def test_a_fit_that_cannot_converge_says_so():
    obs = truth(3)
    wild = Y0 + np.array([300.0, 0, 0, 0, 0.3, 0])                    # far from any orbit
    with pytest.raises(fit.FitError, match="did not converge|diverged"):
        fit.fit_arc(obs, MODEL, wild, cd_am=CD_AM, cr_am=CR_AM, max_iter=3)


# --- the real files --------------------------------------------------------------------------

def test_a_real_sentinel_1a_arc_fits_to_metres():
    from satflight import poeorb
    have = poeorb.local_files("S1A")
    day = date(2025, 7, 3)                                            # no burn that day
    if day not in have:
        pytest.skip(f"no S1A file for {day} under {poeorb.mission_dir('S1A')}")
    arc = poeorb.load_arc("S1A", day, day)
    model = ForceModel(gravity_degree=30, drag=True, sun=True, moon=True, srp=True)

    def cross_track(table):
        obs = fit.observations(arc, step=120.0, hours=12.0, eop=table)
        res = fit.fit_arc(obs, model, fit.first_state(arc, eop=table), cd_am=0.0102,
                          cr_am=0.006, fit_cr=False)
        assert res.converged
        return res, np.sqrt((res.residuals_rsw[obs.use, 2] ** 2).mean())

    from satflight import eop
    try:
        table = eop.read()
    except eop.EopError as exc:
        pytest.skip(str(exc))
    res, w = cross_track(table)
    assert res.rms_m < 3.0 and w < 2.0
    # left uncorrected, the ITRF's tilt from the spin axis comes back as a daily
    # cross-track wobble the fit cannot follow
    _, w_raw = cross_track(None)
    assert w_raw > 2.0 * w
