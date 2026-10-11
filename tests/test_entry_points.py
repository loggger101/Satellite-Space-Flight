"""The ways in: the command line, the batch runner, files on disk (scenarios,
the Earth assets, the fonts). A bad argument or file stops with the reason; it
is never dropped, replaced by a default or swapped for a stand-in."""

import csv
import json
import math
import os
import shutil

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path

import pygame
import pytest

from satflight import batch
from satflight.__main__ import main as cli
from satflight.scenario import Scenario
from satflight.ui import theme
from satflight.ui.app import App
from satflight.ui.earth import EarthRenderer
from satflight.ui.welcome import StartScreen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = str(ROOT / "scenarios" / "default.json")


@pytest.fixture
def app(tmp_path):
    a = App(size=(1400, 850))
    a.user_scenario_dir = tmp_path / "user"
    yield a
    pygame.quit()


def refused(capsys, fn, *args) -> str:
    """Run ``fn(*args)``; it must exit with status 2 (argparse); returns stderr."""
    with pytest.raises(SystemExit) as stop:
        fn(*args)
    assert stop.value.code == 2
    return capsys.readouterr().err


# --- command line ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv, words", [
    (["no_such.json"], "cannot open"),
    (["--size", "big"], "WxH"),
    (["--size", "0x600"], "positive"),
    (["--advance", "3x"], "duration"),
    (["--advance=-1h"], "negative"),       # argparse before 3.14 reads a bare "-1h" as an option
    (["--warp", "0"], "--warp"),
    (["--distance", "-5"], "--distance"),
    (["--frames", "0"], "--frames"),
    (["--show", "map,mpa"], "'mpa'"),
    (["--select", "Nobody"], "Nobody"),
])
def test_bad_command_lines_stop_with_the_reason(capsys, argv, words):
    args = [DEFAULT, "--headless", "--frames", "1", *argv]
    if argv[0].endswith(".json"):
        args = ["--headless", *argv]
    assert words in refused(capsys, cli, args)


def test_a_scenario_with_a_typo_is_refused_on_the_command_line(capsys, tmp_path):
    bad = tmp_path / "typo.json"
    bad.write_text(json.dumps({"name": "x", "satelites": []}), encoding="utf-8")
    assert "'satelites'" in refused(capsys, cli, [str(bad), "--headless"])


def test_every_command_line_option_takes_effect(tmp_path):
    shot = tmp_path / "out.png"
    cli([DEFAULT, "--headless", "--frames", "2", "--screenshot", str(shot), "--size", "1000x700",
         "--show", "map,plot,vectors", "--select", "Hubble", "--advance", "10m",
         "--frame", "ECEF", "--follow", "--distance", "30000", "--yaw", "10", "--pitch", "20",
         "--warp", "120"])
    img = pygame.image.load(str(shot))
    assert img.get_size() == (1000, 700)


# --- batch runs -----------------------------------------------------------------------------

@pytest.mark.parametrize("argv, words", [
    (["no_such.json"], "no_such.json"),
    ([DEFAULT, "--step", "0"], "--step"),
    ([DEFAULT, "--step", "abc"], "duration"),
    ([DEFAULT, "--duration", "-1h"], "--duration"),
    ([DEFAULT, "--duration", "inf"], "finite"),
])
def test_bad_batch_runs_stop_with_the_reason(capsys, argv, words):
    assert words in refused(capsys, batch.main, argv)


def test_run_refuses_a_step_that_would_never_end():
    with pytest.raises(ValueError, match="step"):
        batch.run(Scenario.load(DEFAULT), 3600, 0.0, quiet=True)


def test_batch_csv_has_every_satellite_at_every_step_and_only_numbers(tmp_path, capsys):
    out = tmp_path / "run.csv"
    batch.main([DEFAULT, "--duration", "2h", "--step", "600", "--out", str(out)])
    text = capsys.readouterr().out
    assert "Wrote" in text and "Hubble" in text
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    n_sats = len(Scenario.load(DEFAULT).satellites)
    assert len(rows) == n_sats * 13                          # t = 0, 600, ..., 7200
    for r in rows:
        for k, v in r.items():
            if k not in ("utc", "name", "status"):
                assert math.isfinite(float(v)), (k, r)


def test_history_export_writes_only_numbers(app, tmp_path):
    app.sim.advance(1800)
    n = batch.export_history_csv(app.sim, tmp_path / "h.csv")
    rows = list(csv.reader((tmp_path / "h.csv").read_text(encoding="utf-8").splitlines()))
    assert n == len(rows) - 1 > 0
    assert all(math.isfinite(float(x)) for r in rows[1:] for x in r[4:])


# --- assets and fonts -----------------------------------------------------------------------

def test_absent_earth_assets_are_announced(tmp_path):
    earth = EarthRenderer(tmp_path)
    assert earth.missing == ["earth.jpg", "coastlines.json"] and not earth.has_image
    a = App(Scenario(name="x"), size=(1000, 700), root=tmp_path)
    try:
        assert any("plain Earth" in m for _, m in a.toasts)
    finally:
        pygame.quit()


def test_bundled_earth_assets_load():
    earth = EarthRenderer(ROOT / "assets")
    assert earth.has_image and len(earth.coastlines) > 100 and not earth.missing


@pytest.mark.parametrize("name, content", [("earth.jpg", b"not a jpeg"),
                                           ("coastlines.json", b"{\"a\": 1}"),
                                           ("coastlines.json", b"[[1, 2, 3]]")])
def test_a_broken_earth_asset_names_itself(tmp_path, name, content):
    pygame.init()
    (tmp_path / name).write_bytes(content)
    with pytest.raises(RuntimeError, match=name.replace(".", r"\.")):
        EarthRenderer(tmp_path)


def test_a_missing_bundled_font_is_an_error(monkeypatch, tmp_path):
    pygame.font.init()
    monkeypatch.setenv("SATFLIGHT_FONTS", "bundled")
    monkeypatch.setattr(theme, "FONT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="DejaVuSans.ttf"):
        theme.load_font("sans", 15)


# --- loading and saving in the app ----------------------------------------------------------

def test_a_missing_scenario_file_changes_nothing(app, tmp_path):
    sim = app.sim
    with pytest.raises(FileNotFoundError):
        app.load_scenario(tmp_path / "gone.json")
    assert app.sim is sim


def test_a_failed_load_keeps_the_start_menu_open(app, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"satelites": []}), encoding="utf-8")
    app.open("start")
    with pytest.raises(ValueError, match="satelites"):
        app.load_scenario(bad)
    assert app.in_menu


def test_an_unreadable_file_in_scenarios_gets_a_card_that_says_why(tmp_path):
    shutil.copytree(ROOT / "scenarios", tmp_path / "scenarios",
                    ignore=shutil.ignore_patterns("user"))
    (tmp_path / "scenarios" / "zz_broken.json").write_text("{not json", encoding="utf-8")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    a = App(size=(1400, 850), root=tmp_path, welcome=True)
    try:
        menu = a.dialogs[-1]
        assert isinstance(menu, StartScreen)
        k = next(k for k, c in enumerate(menu.cards) if c[0].stem == "zz_broken")
        assert menu.cards[k][3] == "unreadable" and "Cannot be opened" in menu.cards[k][2]
        menu.choose(k)
        assert a.in_menu and any("Could not open zz_broken.json" in m for _, m in a.toasts)
    finally:
        pygame.quit()


def test_a_broken_bundled_scenario_stops_the_menu(tmp_path):
    shutil.copytree(ROOT / "scenarios", tmp_path / "scenarios",
                    ignore=shutil.ignore_patterns("user"))
    (tmp_path / "scenarios" / "geo_drift.json").write_text("{}x", encoding="utf-8")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    with pytest.raises(ValueError):
        App(size=(1400, 850), root=tmp_path, welcome=True)
    pygame.quit()


def test_saving_where_it_cannot_write_says_so(app, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    app.user_scenario_dir = blocker / "user"
    app.quick_save()
    assert any("Could not save" in m for _, m in app.toasts)
    app.open("scenario")
    dlg = app.dialogs[-1]
    next(b for b in dlg.buttons if b.text == "Save snapshot").callback()
    assert dlg in app.dialogs and dlg.error


def test_reset_after_the_file_went_away_says_so(app, tmp_path):
    path = tmp_path / "s.json"
    shutil.copy(DEFAULT, path)
    app.load_scenario(path)
    path.unlink()
    app.reset()
    assert any("Could not reload s.json" in m for _, m in app.toasts)
