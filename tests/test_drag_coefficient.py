"""Physical drag coefficient (satflight.drag_coefficient): Sentman's sphere with
diffuse reemission and incomplete energy accommodation, the accommodation from
the SESAM Langmuir fit, held to published values.

Check values: Moe & Moe, "The Effects of Surface Composition and Treatment on
Drag Coefficients of Spherical Satellites", AAS 05-258 (2005), Appendices A
and B, the "Diffuse" column: Sentman's sphere with orbit-measured accommodation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from satflight import drag_coefficient as dc

K_B = 1.380649e-23
M_O = 15.9994 * 1.66053907e-27

# (alpha, diffuse Cd) from Moe & Moe 2005, Appendix A (solar minimum, 250-350
# km) and B (solar maximum, 250-500 km); the 150 and 200 km rows are left out:
# there N2 dominates and the air is cooler, which the O-at-1000-K stand-in
# below does not represent (it misses them by 0.04-0.06)
MOE_2005 = [(0.97, 2.24), (0.93, 2.34), (0.89, 2.42),
            (0.99, 2.16), (0.98, 2.21), (0.97, 2.25), (0.95, 2.31), (0.93, 2.36),
            (0.90, 2.42)]


def _atomic_oxygen_sphere(alpha, v=7600.0, temperature=1000.0, wall=300.0):
    """Cd of a sphere in pure atomic oxygen at ``temperature`` K."""
    s = v / math.sqrt(2.0 * K_B * temperature / M_O)
    return dc.sentman_sphere(np.array([s]), dc.reemission_ratio(
        np.array([alpha]), np.array([v]), M_O, wall))[0]


@pytest.mark.parametrize("alpha, cd", MOE_2005)
def test_sphere_matches_moe_and_moe(alpha, cd):
    assert _atomic_oxygen_sphere(alpha) == pytest.approx(cd, abs=0.035)


def _msis_mean_accommodation(alt_km, f107=200.0, ap=15.0):
    """SESAM accommodation averaged over a lat/lon grid at fixed solar-maximum
    indices (F10.7 = its 81-day mean = 200, ap 15), 2001-11-15 12:00 UTC."""
    pymsis = pytest.importorskip("pymsis")
    lat, lon = np.meshgrid(np.linspace(-60, 60, 7), np.linspace(-180, 150, 12))
    n = lat.size
    out = pymsis.calculate(np.full(n, np.datetime64("2001-11-15T12:00")), lon.ravel(),
                           lat.ravel(), np.full(n, float(alt_km)), np.full(n, f107),
                           np.full(n, f107), np.full((n, 7), ap))
    return dc.accommodation(out[:, 1 + dc.SPECIES.index("O")], out[:, 10]).mean()


# Moe & Moe 2005, Appendix B: orbit-derived accommodation at solar maximum
@pytest.mark.parametrize("alt_km, alpha", [(250, 0.99), (300, 0.98), (350, 0.97),
                                           (400, 0.95), (450, 0.93)])
def test_sesam_accommodation_matches_moe_and_moe_at_solar_maximum(alt_km, alpha):
    assert _msis_mean_accommodation(alt_km) == pytest.approx(alpha, abs=0.03)


def test_accommodation_rises_with_oxygen_and_stays_below_one():
    a = dc.accommodation(np.array([0.0, 1e12, 1e13, 1e14, 1e16]), np.full(5, 1000.0))
    assert a[0] == 0.0 and np.all(np.diff(a) > 0) and a[-1] < 1.0


def _species(**n):
    row = np.zeros((1, len(dc.SPECIES)))
    for name, value in n.items():
        row[0, dc.SPECIES.index(name.replace("_", " "))] = value
    return row


def test_air_of_one_species_has_that_species_coefficient():
    v, t = np.array([7500.0]), np.array([1100.0])
    n = _species(O=3e13)
    alpha = dc.accommodation(n[:, 2], t)
    s = v / np.sqrt(2.0 * K_B * t / M_O)
    want = dc.sentman_sphere(s, dc.reemission_ratio(alpha, v, M_O))
    assert dc.sphere_cd(v, n, t) == pytest.approx(want, rel=1e-12)


def test_a_mixture_is_weighted_by_mass_density():
    v, t = np.array([7500.0]), np.array([1100.0])
    both = _species(O=3e13, He=6e13)
    alpha = dc.accommodation(both[:, 2], t)    # the oxygen sets the accommodation of all
    cd = []
    for i in (2, 3):
        m = dc.MASS_KG[i]
        s = v / np.sqrt(2.0 * K_B * t / m)
        cd.append(dc.sentman_sphere(s, dc.reemission_ratio(alpha, v, m))[0])
    rho_o, rho_he = 3e13 * dc.MASS_KG[2], 6e13 * dc.MASS_KG[3]
    want = (rho_o * cd[0] + rho_he * cd[1]) / (rho_o + rho_he)
    assert dc.sphere_cd(v, both, t)[0] == pytest.approx(want, rel=1e-12)


def test_helium_rich_air_drags_harder_than_oxygen():
    v, t = np.array([7500.0, 7500.0]), np.array([1000.0, 1000.0])
    n = np.vstack([_species(O=1e13), _species(O=1e13, He=1e14)])
    cd = dc.sphere_cd(v, n, t)
    assert cd[1] > cd[0]


@pytest.mark.parametrize("bad", [
    dict(v=[np.nan], n=_species(O=1e13), t=[1000.0]),
    dict(v=[7500.0], n=_species(O=-1.0), t=[1000.0]),
    dict(v=[7500.0], n=_species(O=1e13), t=[np.inf]),
    dict(v=[0.0], n=_species(O=1e13), t=[1000.0]),
])
def test_bad_air_is_refused_not_passed_on(bad):
    with pytest.raises(ValueError, match="drag coefficient"):
        dc.sphere_cd(np.array(bad["v"]), bad["n"], np.array(bad["t"]))


def test_no_air_is_refused():
    with pytest.raises(ValueError, match="no air"):
        dc.sphere_cd(np.array([7500.0]), _species(), np.array([1000.0]))
