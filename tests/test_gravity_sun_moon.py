"""The EGM2008 field (satflight.gravity), the Sun's and Moon's gravity and
sunlight pressure (satflight.forces), and how scenarios switch them on."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from satflight import gravity
from satflight.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SRP, R_EARTH
from satflight.ephemeris import moon_position, sun_position
from satflight.forces import (
    ForceModel,
    accel_c22,
    accel_j2,
    accel_srp,
    accel_third_body,
    conservative_energy,
)
from satflight.integrators import Propagator
from satflight.scenario import SatSpec, Scenario
from satflight.simulation import Simulation
from satflight.timeutil import UTC, julian_date

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
R = rng.normal(size=(6, 3))
R = R / np.linalg.norm(R, axis=1, keepdims=True) * np.array([[7071.0]] * 3 + [[26560.0]] * 3)


# --- the field ------------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [2, 5, 20, 70])
def test_the_field_is_the_gradient_of_its_potential(degree):
    """Two independent routes: the V/W recursion against normalized Legendre sums."""
    a = gravity.acceleration(R, 0.7, degree)
    h = 1e-3
    g = np.zeros_like(R)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[:, k] = (gravity.potential(R + e, 0.7, degree)
                   - gravity.potential(R - e, 0.7, degree)) / (2 * h)
    assert np.abs(a - g).max() < 1e-7 * np.abs(a).max()


def test_the_legendre_functions_are_fully_normalized_to_degree_70():
    """Geodesy's normalization: the mean square of P_nm(sin phi) cos(m lon) over
    the sphere is 1 (so the integral over t = sin phi is 2 for m = 0, 4
    otherwise), and different degrees are orthogonal. The potential checks the
    recursion through its gradient; this checks the normalization itself."""
    t, w = np.polynomial.legendre.leggauss(80)            # exact for degree <= 159
    P = gravity.legendre(70, t)                           # (80, 71, 71)
    gram = np.einsum("k,kim,kjm->mij", w, P, P)           # per order m: <P_im, P_jm>
    for m in (0, 1, 2, 35, 69, 70):
        g = gram[m, m:, m:]
        assert np.allclose(np.diag(g), 2.0 if m == 0 else 4.0, rtol=1e-10), m
        assert np.abs(g - np.diag(np.diag(g))).max() < 1e-10, m
    assert gravity.legendre(2, np.array(0.5))[2, 0] == pytest.approx(
        math.sqrt(5) * (3 * 0.25 - 1) / 2)                 # sqrt(5) P2 by hand


def test_degree_two_is_j2_and_c22_with_egm2008s_values():
    f = gravity.load(2)
    assert f.gm == pytest.approx(398600.4415) and f.re == pytest.approx(6378.1363)
    assert -f.c[2, 0] == pytest.approx(1.0826e-3, rel=1e-4)            # J2
    ref = (accel_j2(R, mu=f.gm, re=f.re, j2=-f.c[2, 0])
           + accel_c22(R, 0.3, mu=f.gm, re=f.re, c22=f.c[2, 2], s22=f.s[2, 2]))
    # all that differs is C21/S21 (~1e-9: the pole sits near the axis)
    assert np.abs(gravity.acceleration(R, 0.3, 2) - ref).max() < 1e-5 * np.abs(ref).max()


def test_the_field_turns_with_the_earth():
    a0 = gravity.acceleration(R[:1], 0.0, 20)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # +90 deg about z
    a1 = gravity.acceleration(R[:1] @ rot.T, math.pi / 2, 20)
    assert np.allclose(a1, a0 @ rot.T, rtol=1e-12, atol=1e-20)


def test_the_committed_coefficients_say_where_they_came_from():
    text = (ROOT / "satflight" / "data" / "egm2008_70.gfc").read_text(encoding="ascii")
    assert "source sha256 " in text and "tide_system                 tide_free" in text
    assert "max_degree                  70" in text
    f = gravity.load(70)
    assert f.cbar[70, 70] == pytest.approx(0.298214665798648e-09)       # the last row
    assert f.cbar[1].sum() == 0 and f.cbar[0, 0] == 0                  # point mass apart


def test_the_cut_reproduces_from_icgems_file_when_it_is_here():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import make_gravity

    from satflight.poeorb import data_root
    src = data_root() / "gravity" / "EGM2008.zip"
    if not src.is_file():
        pytest.skip(f"no {src} (download it from ICGEM, see tools/make_gravity.py)")
    assert make_gravity.cut(src, 70) == (ROOT / "satflight" / "data" / "egm2008_70.gfc"
                                         ).read_text(encoding="ascii")


@pytest.mark.parametrize("bad", [0, 1, 71, True, 2.0, "70"])
def test_only_degrees_two_to_seventy_exist(bad):
    with pytest.raises(ValueError, match="degree"):
        gravity.load(bad)
    if bad != 0:
        with pytest.raises(ValueError, match="gravity_degree"):
            ForceModel(gravity_degree=bad)


def test_a_coefficient_file_of_the_wrong_kind_is_refused(tmp_path):
    good = (ROOT / "satflight" / "data" / "egm2008_70.gfc").read_text(encoding="ascii")
    zero_tide = tmp_path / "zt.gfc"
    zero_tide.write_text(good.replace("tide_free", "zero_tide"), encoding="ascii")
    with pytest.raises(ValueError, match="tide_system"):
        gravity.load(5, zero_tide)
    holey = tmp_path / "holey.gfc"
    holey.write_text("\n".join(ln for ln in good.splitlines()
                               if not ln.startswith("gfc     3    2")), encoding="ascii")
    with pytest.raises(ValueError, match="degree 3 order 2"):
        gravity.load(5, holey)


def test_the_jacobi_integral_holds_in_the_full_field():
    """The field turns with the Earth: E - omega h_z is what it conserves."""
    from satflight.elements import coe2rv
    model = ForceModel(gravity_degree=30)
    omega = 7.292115146706979e-5
    r0, v0 = coe2rv(7071.0, 0.001, math.radians(98.18), 1.0, 1.5, 0.3)
    y0 = np.concatenate([r0, v0]).reshape(1, 6)

    def f(t, y):
        return np.concatenate([y[:, 3:], model.acceleration(
            y[:, :3], y[:, 3:], np.zeros(1), omega * t)], axis=1)

    e0 = conservative_energy(y0[:, :3], y0[:, 3:], model, 0.0)
    y = Propagator("dopri5", rtol=1e-12, atol=1e-12, h_max=30.0).integrate(f, 0.0, y0, 3 * 3600)[1]
    e1 = conservative_energy(y[:, :3], y[:, 3:], model, omega * 3 * 3600)
    assert abs((e1 - e0) / e0) < 1e-11


# --- Sun, Moon and sunlight -------------------------------------------------------------------

def test_the_moon_is_where_the_almanac_puts_it():
    """Vallado example 5-3 (28 April 1994, 0h TDB): r = (-134240.626,
    -311571.590, -126693.785) km; the low-precision series is good to ~0.3 deg."""
    jd = julian_date(datetime(1994, 4, 28, tzinfo=UTC))
    m = moon_position(jd)
    ref = np.array([-134240.626, -311571.590, -126693.785])
    ang = math.degrees(math.acos(m @ ref / np.linalg.norm(m) / np.linalg.norm(ref)))
    assert ang < 0.3 and abs(np.linalg.norm(m) - np.linalg.norm(ref)) < 1500


def test_moon_phases_and_distance():
    def elongation(dt):
        jd = julian_date(dt)
        s, m = sun_position(jd), moon_position(jd)
        return math.degrees(math.acos(s @ m / np.linalg.norm(s) / np.linalg.norm(m)))
    assert elongation(datetime(2024, 1, 25, 17, 54, tzinfo=UTC)) > 174    # full moon
    assert elongation(datetime(2024, 1, 11, 11, 57, tzinfo=UTC)) < 6      # new moon
    jd0 = julian_date(datetime(2025, 1, 1, tzinfo=UTC))
    assert all(355000 < np.linalg.norm(moon_position(jd0 + d)) < 407500 for d in range(60))


def test_third_body_tides_have_the_textbook_size_and_shape():
    r = np.array([[7071.0, 0.0, 0.0]])
    for mu, d in ((MU_MOON, 384400.0), (MU_SUN, AU)):
        body = np.array([d, 0.0, 0.0])
        rel = 2 * 7071.0 / d        # the tide formula drops terms of order (3/2) r/d
        along = accel_third_body(r, body, mu)[0]
        assert along[0] == pytest.approx(2 * mu * 7071.0 / d ** 3, rel=rel)    # stretched
        across = accel_third_body(np.array([[0.0, 7071.0, 0.0]]), body, mu)[0]
        assert across[1] == pytest.approx(-mu * 7071.0 / d ** 3, rel=rel)      # squeezed
    moon = 2 * MU_MOON / 384400.0 ** 3
    sun = 2 * MU_SUN / AU ** 3
    assert 2.0 < moon / sun < 2.4                       # the Moon's tide ~2.2x the Sun's


def test_sunlight_pushes_away_from_the_sun_and_stops_in_the_shadow():
    sun = np.array([AU, 0.0, 0.0])
    r = np.array([[0.0, 7071.0, 0.0], [-7071.0, 0.0, 0.0]])     # in sunlight, behind Earth
    a = accel_srp(r, sun, np.array([0.01, 0.01]))
    assert a[0] @ (r[0] - sun) > 0                               # away from the Sun
    assert np.linalg.norm(a[0]) == pytest.approx(P_SRP * 0.01 * 1e-3, rel=1e-6)
    assert np.linalg.norm(a[1]) == 0.0


# --- the switches -----------------------------------------------------------------------------

def test_the_field_stands_in_for_the_separate_shape_terms():
    jd = julian_date(datetime(2025, 7, 3, tzinfo=UTC))
    v = np.array([[0.0, 7.5, 0.0]] * len(R))
    base = ForceModel(j2=True, j3=True, j4=True, c22=True, gravity_degree=70)
    a = base.acceleration(R, v, np.zeros(len(R)), 0.4)
    from satflight.forces import accel_point_mass
    assert np.allclose(a - accel_point_mass(R), gravity.acceleration(R, 0.4, 70), rtol=0,
                       atol=1e-18)
    assert base.label() == "2B+EGM70" and ForceModel().label() == "2B+J2"
    everything = ForceModel(gravity_degree=70, drag=True, sun=True, moon=True, srp=True)
    assert everything.label() == "2B+EGM70+DRAG+SUN+MOON+SRP"
    with pytest.raises(ValueError, match="need the date"):
        everything.acceleration(R, v, np.zeros(len(R)), 0.0)
    with pytest.raises(ValueError, match="Cr"):
        everything.acceleration(R, v, np.zeros(len(R)), 0.0, jd)
    shown = everything.breakdown(R[:1], v[:1], 0.01, 0.0, jd, 0.006)
    assert {"EGM70 - J2", "Sun", "Moon", "sunlight"} <= set(shown)
    assert shown["J2"][1] is False and shown["EGM70 - J2"][1] is True    # the field has J2


def test_scenarios_carry_the_switches_and_cr():
    sc = Scenario(name="s", forces=ForceModel(gravity_degree=40, sun=True, moon=True, srp=True),
                  satellites=[SatSpec("A", {"type": "elements", "altitude": 693, "i": 98.18},
                                      cr=1.5), SatSpec("B", {"type": "elements", "altitude": 700})])
    d = json.loads(json.dumps(sc.to_dict()))
    assert d["forces"]["gravity_degree"] == 40 and d["forces"]["srp"] is True
    assert d["satellites"][0]["cr"] == 1.5 and "cr" not in d["satellites"][1]   # default 1.3
    back = Scenario.from_dict(d)
    assert back.to_dict() == sc.to_dict()
    sim = Simulation(back)
    assert [s.cr for s in sim.sats] == [1.5, 1.3]
    for bad in ({"cr": -0.1}, {"cr": 3.0}):
        d2 = json.loads(json.dumps(d))
        d2["satellites"][0].update(bad)
        with pytest.raises(ValueError, match="reflectivity Cr"):
            Scenario.from_dict(d2)
    d["forces"]["gravity_degree"] = 99
    with pytest.raises(ValueError, match="gravity_degree"):
        Scenario.from_dict(d)


def test_the_sun_and_moon_move_a_simulated_orbit():
    """Same start, 6 h: the extra forces shift a 700 km orbit by metres or
    more, and the full field by kilometres - each switch reaches the motion."""
    orbit = {"type": "elements", "altitude": 693, "i": 98.18, "raan": 278.3}

    def run(**forces):
        sim = Simulation(Scenario(name="x", forces=ForceModel(j2=True, **forces),
                                  satellites=[SatSpec("S1", orbit, mass=2164, area=10)]))
        sim.advance(6 * 3600)
        return sim.y[0, :3]

    base = run()
    for on, least in (({"sun": True}, 0.002), ({"moon": True}, 0.01), ({"srp": True}, 0.0005),
                      ({"gravity_degree": 20}, 0.5)):
        assert np.linalg.norm(run(**on) - base) > least, on


def test_a_sentinel_like_orbit_feels_forces_of_the_expected_size():
    jd = julian_date(datetime(2025, 7, 3, tzinfo=UTC))
    r = np.array([[7071.0, 0.0, 0.0]])
    v = np.array([[0.0, 1.1, 7.4]])
    shown = ForceModel(gravity_degree=70, drag=True, sun=True, moon=True, srp=True).breakdown(
        r, v, 2.2 * 10 / 2164, 0.0, jd, 1.3 * 10 / 2164)
    mps2 = {k: m * 1e3 for k, (m, _) in shown.items()}
    assert mps2["gravity"] == pytest.approx(MU_EARTH / 7071.0 ** 2 * 1e3)
    assert 3e-5 < mps2["EGM70 - J2"] < 1e-3          # the field beyond J2 (J3, tesserals)
    # tides below their maxima 2 mu r / d^3: 5.6e-7 (Sun), 1.2e-6 m/s^2 (Moon)
    assert 1e-7 < mps2["Sun"] < 5.7e-7 and 2e-7 < mps2["Moon"] < 1.25e-6
    assert 2e-8 < mps2["sunlight"] < 5e-8
    assert R_EARTH < 7071.0
