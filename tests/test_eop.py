"""Earth orientation (satflight.eop): the IERS finals2000A table read strictly,
polar motion interpolated to any time, and the pole correction applied when the
Sentinel-1 files (ITRF) come into the simulator's frame."""

from __future__ import annotations

import math

import numpy as np
import pytest

from satflight import eop, poeorb

AS = math.pi / (180.0 * 3600.0)                                       # rad per arcsec


def _line(mjd, xp, yp, dut1, flag="I", ut1_flag=None):
    """One row in the finals2000A layout (IERS readme: PM-x cols 19-27, PM-y
    38-46, UT1-UTC 59-68, flags in cols 17 and 58)."""
    return (f"25 7 5 {mjd:8.2f} {flag} {xp:9.6f} 0.000012 {yp:9.6f} 0.000015  "
            f"{ut1_flag or flag}{dut1:10.7f} 0.0000136 -0.8290 0.0081  I     0.402    0.400"
            "    -0.151    0.137  0.167817  0.438664  0.0456628     0.415    -0.137")


def _table(tmp_path, rows, tail=("28 1 8 61778.00",)):
    p = tmp_path / "finals2000A.all"
    p.write_text("\n".join([*(_line(*r) for r in rows), *tail]) + "\n", encoding="ascii")
    return p


def test_the_layout_helper_matches_the_real_columns():
    s = _line(60861.0, 0.16783, 0.438608, 0.045637)
    assert s[16] == "I" and s[18:27] == " 0.167830" and s[37:46] == " 0.438608"
    assert s[57] == "I" and s[58:68] == " 0.0456370"


def test_a_table_reads_pole_and_ut1_and_stops_at_the_blank_future(tmp_path):
    t = eop.read(_table(tmp_path, [(60861.0, 0.16783, 0.438608, 0.045637),
                                   (60862.0, 0.16900, 0.437000, 0.045000)]))
    assert t.mjd.tolist() == [60861.0, 60862.0]
    assert t.xp[0] == pytest.approx(0.16783 * AS) and t.yp[1] == pytest.approx(0.437 * AS)
    assert t.dut1.tolist() == [0.045637, 0.045]


def test_the_pole_is_interpolated_linearly_in_time(tmp_path):
    t = eop.read(_table(tmp_path, [(60861.0, 0.1, 0.4, 0.04), (60862.0, 0.2, 0.3, 0.05)]))
    utc = np.array(["2025-07-05T06:00", "2025-07-05T18:00"], "datetime64[us]")   # MJD 60861
    xp, yp = t.pole(utc)
    assert xp / AS == pytest.approx([0.125, 0.175]) and yp / AS == pytest.approx([0.375, 0.325])


def test_times_outside_the_measured_table_are_refused(tmp_path):
    t = eop.read(_table(tmp_path, [(60861.0, 0.1, 0.4, 0.04), (60862.0, 0.2, 0.3, 0.05),
                                   (60863.0, 0.2, 0.3, 0.05, "P")]))
    with pytest.raises(eop.EopError, match="2025-07-04"):
        t.pole(np.array(["2025-07-04T23:00"], "datetime64[us]"))
    with pytest.raises(eop.EopError, match="predicted|measured"):
        t.pole(np.array(["2025-07-06T12:00"], "datetime64[us]"))       # needs the P row


@pytest.mark.parametrize("bad", ["25 7 5 60861.00 I  0.16x830", "garbage", ""])
def test_a_broken_row_is_named(tmp_path, bad):
    p = tmp_path / "finals2000A.all"
    p.write_text(_line(60861.0, 0.1, 0.4, 0.04) + "\n" + (bad or " " * 30) + "\n"
                 + _line(60863.0, 0.1, 0.4, 0.04) + "\n", encoding="ascii")
    with pytest.raises(eop.EopError, match="line 2"):
        eop.read(p)


def test_days_must_follow_one_another(tmp_path):
    with pytest.raises(eop.EopError, match="line 2"):
        eop.read(_table(tmp_path, [(60861.0, 0.1, 0.4, 0.04), (60863.0, 0.1, 0.4, 0.04)]))


def test_a_missing_table_says_where_it_looked_and_how_to_get_it(tmp_path):
    with pytest.raises(eop.EopError, match="fetch_eop"):
        eop.read(tmp_path / "nope.all")
    assert eop.table_path(tmp_path) == tmp_path / "eop" / "finals2000A.all"


def test_the_spin_axis_sits_at_xp_and_minus_yp_in_the_itrf():
    """IERS Conventions (2010) ch. 5: x_p, y_p are the CIP's coordinates in the
    ITRS, x_p toward Greenwich and y_p toward 90 deg W (so -y_p along the ITRS
    y axis, which points to 90 deg E). The rotation into the terrestrial
    intermediate frame (GCRS = Q R W ITRS, this is W) must carry that
    direction onto its z axis, and so the ITRS z axis lands at (-x_p, +y_p)."""
    xp, yp = 0.3 * AS, 0.4 * AS
    W = eop.itrf_to_tirs(np.array([xp]), np.array([yp]))[0]
    cip = np.array([xp, -yp, 1.0])
    assert np.allclose(W @ (cip / np.linalg.norm(cip)), [0.0, 0.0, 1.0], atol=1e-12)
    z = W @ np.array([0.0, 0.0, 1.0])
    assert z[0] == pytest.approx(-xp, rel=1e-6) and z[1] == pytest.approx(yp, rel=1e-6)
    assert np.allclose(W @ W.T, np.eye(3), atol=1e-15)


def test_to_eci_applies_the_pole_before_turning_the_earth():
    utc = np.array(["2025-07-05T00:00", "2025-07-05T06:00"], "datetime64[us]")
    r = np.array([[7071.0, 0.0, 0.0], [0.0, 0.0, 7071.0]])
    v = np.array([[0.0, 7.5, 0.0], [7.5, 0.0, 0.0]])
    r0, v0 = poeorb.to_eci(utc, 0.04, r, v)
    pole = (np.full(2, 0.3 * AS), np.full(2, 0.4 * AS))
    r1, v1 = poeorb.to_eci(utc, 0.04, r, v, pole=pole)
    # the pole point moves by the whole 0.5 arcsec tilt (17 m at 7071 km); a point on
    # Greenwich's meridian only by x_p, the tilt about the y axis (10 m)
    moved = np.linalg.norm(r1 - r0, axis=1)
    assert moved == pytest.approx(np.array([0.3, 0.5]) * AS * 7071.0, rel=1e-3)
    assert abs(np.linalg.norm(r1[1]) - 7071.0) < 1e-9
    W = eop.itrf_to_tirs(*pole)
    r2, _ = poeorb.to_eci(utc, 0.04, np.einsum("nij,nj->ni", W, r),
                          np.einsum("nij,nj->ni", W, v))
    assert np.allclose(r1, r2, atol=1e-12)


def test_the_downloaded_table_has_the_iers_values():
    p = eop.table_path()
    if not p.exists():
        pytest.skip(f"no EOP table at {p}")
    t = eop.read(p)
    xp, yp = t.pole(np.array(["2025-07-05T00:00"], "datetime64[us]"))
    assert xp[0] / AS == pytest.approx(0.167830) and yp[0] / AS == pytest.approx(0.438608)
    # the Sentinel-1 files carry their own UT1-UTC: it must agree with the IERS one
    have = poeorb.local_files("S1A")
    from datetime import date
    if date(2025, 7, 5) in have:
        f = poeorb.read_file(have[date(2025, 7, 5)])
        k = np.flatnonzero(f.utc == np.datetime64("2025-07-05T00:00:02", "us"))
        assert k.size
        mjd = 60861.0 + 2.0 / 86400.0
        assert np.interp(mjd, t.mjd, t.dut1) == pytest.approx(f.dut1[k[0]], abs=2e-4)


def test_fetch_keeps_a_good_table_and_refuses_a_broken_one(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import fetch_eop
    good = "\n".join(_line(60861.0 + k, 0.1, 0.4, 0.04) for k in range(3)) + "\n"
    p = fetch_eop.fetch(tmp_path, get=lambda url: good.encode(), out=lambda s: None)
    assert p == eop.table_path(tmp_path) and eop.read(p).mjd.size == 3
    with pytest.raises(eop.EopError):
        fetch_eop.fetch(tmp_path, get=lambda url: b"<html>maintenance</html>\n",
                        out=lambda s: None)
    assert eop.read(p).mjd.size == 3 and not list(tmp_path.glob("eop/*.part"))
