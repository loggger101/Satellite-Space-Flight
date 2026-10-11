"""Every dialog, every option of every choice, bad text in every field: the
form takes the input or says why (a ValueError shown in the dialog). Any other
exception is a bug, and it is no longer caught on the way to the user, so it
fails here. After each accepted form the simulation runs on, so a bad value
cannot hide until later either."""

import contextlib
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"          # before pygame is imported
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import pytest

from satflight.scenario import Scenario
from satflight.ui.app import App
from satflight.ui.widgets import Checkbox, Choice, TextField

DIALOGS = ("add", "walker", "maneuver", "station", "physics", "edit", "launch")
BAD = ("", "abc", "nan", "inf", "-1", "0", "1e300", "2.5")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = App(size=(1400, 850))
    a.user_scenario_dir = tmp_path_factory.mktemp("user")
    yield a
    pygame.quit()


def fresh(app, scenario, name):
    """A new simulation of ``scenario`` with dialog ``name`` open on it (None if
    the dialog does not open there, e.g. Edit with nothing selected)."""
    app.load_scenario(scenario if isinstance(scenario, Scenario) else app.scenario_dir / scenario)
    app.select(0 if app.sim.n else -1)
    app.dialogs.clear()
    app.open(name)
    return app.dialogs[-1] if app.dialogs else None


def settle(app, dlg):
    """Draw it; if the form was accepted, run the simulation a while."""
    if dlg in app.dialogs:
        dlg.draw(app.screen)
    else:
        app.sim.advance(1800)
        app.draw()


def choices(dlg):
    return [(k, w) for k, w in dlg.widgets.items() if isinstance(w, Choice | Checkbox)]


def visible_text_fields(dlg):
    raw = dlg.raw()
    return [s.key for s in dlg.specs
            if isinstance(dlg.widgets.get(s.key), TextField) and dlg._visible(s, raw)]


def cases(app, scenario, name):
    """(choice key, option) settings: the defaults, then each option alone."""
    dlg = fresh(app, scenario, name)
    if dlg is None:
        return []
    out = [(None, None)]
    for key, w in choices(dlg):
        opts = w.options if isinstance(w, Choice) else [not w.value]
        out += [(key, o) for o in opts]
    return out


@contextlib.contextmanager
def case(*what):
    """Name the dialog, option, field and value in any failure (the original
    exception stays chained below it)."""
    try:
        yield
    except Exception as exc:
        raise AssertionError(f"case {what!r}: {type(exc).__name__}: {exc}") from exc


def apply(dlg, key, option):
    if key is not None:
        dlg.set(key, option)
        dlg.changed(key)


@pytest.mark.parametrize("scenario", ["default.json", Scenario(name="Empty")],
                         ids=["default", "empty"])
@pytest.mark.parametrize("name", DIALOGS)
def test_every_option_of_every_dialog_submits_or_says_why(app, scenario, name):
    for key, option in cases(app, scenario, name):
        with case(name, key, option):
            dlg = fresh(app, scenario, name)
            apply(dlg, key, option)
            dlg.submit()
            assert dlg not in app.dialogs or dlg.error, "kept open with no message"
            settle(app, dlg)


@pytest.mark.parametrize("name", DIALOGS)
def test_bad_text_in_any_field_is_refused_with_a_reason(app, name):
    seen = set()
    for key, option in cases(app, "default.json", name):
        dlg = fresh(app, "default.json", name)
        apply(dlg, key, option)
        for field in visible_text_fields(dlg):
            for bad in BAD:
                if (field, bad) in seen:
                    continue
                seen.add((field, bad))
                with case(name, key, option, field, bad):
                    dlg = fresh(app, "default.json", name)
                    apply(dlg, key, option)
                    dlg.set(field, bad)
                    dlg.changed(field)
                    dlg.submit()
                    if dlg in app.dialogs:
                        assert dlg.error, "refused with no message"
                    settle(app, dlg)
