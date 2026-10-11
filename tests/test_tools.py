"""The scripts in tools/: the scenario generator, the README pictures and the
asset fetcher (its network-free parts)."""

import json
import os
import shutil
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path

import numpy as np
import pygame
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fetch_assets  # noqa: E402
import make_scenarios  # noqa: E402
import make_screenshots  # noqa: E402
import selfcheck  # noqa: E402

from satflight.simulation import Simulation  # noqa: E402


@pytest.mark.parametrize("name", list(make_scenarios.ALL))
def test_the_committed_scenarios_are_what_the_generator_makes(name):
    """scenarios/*.json are generated: an edit to one file, or to the generator
    without rerunning it, would otherwise drift apart unnoticed."""
    diff = make_scenarios.stale(name)
    assert not diff, f"rerun: python tools/make_scenarios.py  ({name}.json is stale: {diff})"


def test_the_staleness_check_ignores_only_float_noise():
    """Last-bit differences between platforms pass; any real edit does not."""
    saved = json.loads((ROOT / "scenarios" / "hohmann_to_geo.json").read_text(encoding="utf-8"))
    diff = make_scenarios.differences
    t = saved["maneuvers"][1]["t"]
    noisy = json.loads(json.dumps(saved))
    noisy["maneuvers"][1]["t"] = t * (1 + 4e-16)
    assert diff(noisy, saved) == []
    for change in (lambda s: s["maneuvers"][1].update(t=t + 1e-3),
                   lambda s: s["satellites"][0].update(mass=4001),
                   lambda s: s["forces"].update(drag=True),
                   lambda s: s["satellites"][0]["orbit"].update(i=1e-6 + 1e-3),
                   lambda s: s.update(name=s["name"] + " "),
                   lambda s: s["maneuvers"].pop(),
                   lambda s: s.update(extra=1)):
        edited = json.loads(json.dumps(saved))
        change(edited)
        assert diff(edited, saved), change


def test_every_bundled_scenario_comes_from_the_generator():
    files = {p.stem for p in (ROOT / "scenarios").glob("*.json")}
    assert files == set(make_scenarios.ALL)


@pytest.mark.parametrize("name", list(make_screenshots.SCENES))
def test_every_readme_picture_renders(name, tmp_path):
    path = make_screenshots.render(name, frames=2, out=tmp_path)
    img = pygame.surfarray.array3d(pygame.image.load(str(path)))
    committed = ROOT / "docs" / "images" / f"{name}.png"
    assert img.shape[:2] == pygame.image.load(str(committed)).get_size()
    assert img.std() > 10                   # a picture, not a blank or single-colour frame


def test_every_readme_picture_is_a_scene():
    pictures = {p.stem for p in (ROOT / "docs" / "images").glob("*.png")}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    used = {p for p in pictures if f"images/{p}.png" in readme}
    assert used <= set(make_screenshots.SCENES), used - set(make_screenshots.SCENES)


def test_an_unknown_picture_name_is_refused():
    with pytest.raises(SystemExit, match="unknown scene"):
        make_screenshots.main(["overveiw"])


def test_the_coastline_conversion_reads_both_line_kinds():
    gj = {"features": [
        {"geometry": {"type": "LineString", "coordinates": [[1.23456, 2.0], [3.0, 4.0]]}},
        {"geometry": {"type": "MultiLineString", "coordinates": [[[5, 6], [7, 8]], [[9, 9]]]}},
    ]}
    assert fetch_assets.coastlines_from_geojson(gj) == [
        [[1.235, 2.0], [3.0, 4.0]], [[5, 6], [7, 8]], [[9, 9]]]


@pytest.mark.parametrize("gj", [{"features": []},
                                {"features": [{"geometry": {"type": "Polygon",
                                                            "coordinates": [[[0, 0]]]}}]}])
def test_the_coastline_conversion_refuses_what_it_cannot_use(gj):
    with pytest.raises(ValueError):
        fetch_assets.coastlines_from_geojson(gj)


def test_a_downloaded_texture_is_checked_before_it_is_saved(tmp_path):
    with pytest.raises(ValueError, match="not an image"):
        fetch_assets.check_texture(b"<html>404 Not Found</html>")
    pygame.init()
    good = tmp_path / "t.png"
    pygame.image.save(pygame.surfarray.make_surface(np.zeros((64, 32, 3), np.uint8)), str(good))
    assert fetch_assets.check_texture(good.read_bytes()) == (64, 32)
    square = tmp_path / "s.png"
    pygame.image.save(pygame.surfarray.make_surface(np.zeros((32, 32, 3), np.uint8)),
                      str(square))
    with pytest.raises(ValueError, match="2:1"):
        fetch_assets.check_texture(square.read_bytes())


def test_the_committed_assets_pass_the_fetchers_checks():
    assert fetch_assets.check_texture((ROOT / "assets" / "earth.jpg").read_bytes())
    lines = json.loads((ROOT / "assets" / "coastlines.json").read_text(encoding="utf-8"))
    assert lines and all(len(p) >= 1 and all(len(xy) == 2 for xy in p) for p in lines)


# --- tools/selfcheck.py -----------------------------------------------------------------------


def test_selfcheck_passes_here(capsys):
    assert selfcheck.main(["--quick", "default", "geo_drift"]) == 0
    out = capsys.readouterr().out
    assert "All checks passed." in out and "FAIL" not in out


class Jittery(Simulation):
    """A simulation whose result depends on how it is advanced (a planted bug)."""

    def advance(self, dt):
        super().advance(dt)
        self.y[:, 3:] *= 1.0 + 1e-9


@pytest.mark.parametrize("words", ["CHUNKED RUN DIFFERS", "energy drift"])
def test_selfcheck_fails_on_a_planted_physics_bug(capsys, monkeypatch, words):
    monkeypatch.setattr(selfcheck, "Simulation", Jittery)
    assert selfcheck.main(["--quick", "geo_drift"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  geo_drift" in out and words in out


def test_selfcheck_fails_on_a_broken_scenario_file(capsys, monkeypatch, tmp_path):
    shutil.copytree(ROOT / "scenarios", tmp_path / "scenarios",
                    ignore=shutil.ignore_patterns("user"))
    (tmp_path / "scenarios" / "user").mkdir()
    (tmp_path / "scenarios" / "user" / "mine.json").write_text('{"satelites": []}',
                                                               encoding="utf-8")
    monkeypatch.setattr(selfcheck, "ROOT", tmp_path)
    assert selfcheck.main(["--quick", "geo_drift"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  scenarios" in out and "'satelites'" in out
