"""A small immediate-feel widget toolkit for pygame: buttons, text fields,
choice cyclers, checkboxes and a declarative modal form dialog."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pygame

from . import theme, tips
from .theme import px


class Widget:
    """Base class: the owner sets ``rect``, forwards events to :meth:`handle` and
    calls :meth:`draw` every frame."""

    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.hover = False

    def handle(self, ev) -> bool:
        """React to a pygame event; True if it was consumed."""
        return False

    def draw(self, surf, fonts):
        """Draw the widget into ``surf`` at ``rect``."""


class Button(Widget):
    """Push button. ``active()`` highlights it as a toggle that is on; ``tooltip``
    records the matching keyboard shortcut and ``hint`` says what the button
    does (both shown in its hover tip, see :mod:`.tips`); ``accent`` marks the
    primary action."""

    def __init__(self, text: str, callback: Callable[[], Any],
                 active: Callable[[], bool] | None = None, tooltip: str = "",
                 accent: bool = False, hint: str = ""):
        super().__init__()
        self.text = text
        self.callback = callback
        self.active = active
        self.tooltip = tooltip
        self.hint = hint
        self.accent = accent

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and self.rect.collidepoint(ev.pos)):
            self.callback()
            return True
        return False

    def draw(self, surf, fonts):
        on = self.active() if self.active else False
        bg = theme.ACCENT_DARK if (on or self.accent) else theme.FIELD
        if self.hover:
            bg = theme.mix(bg, theme.ACCENT, 0.35)
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(5))
        edge = theme.ACCENT if on else theme.PANEL_EDGE
        pygame.draw.rect(surf, edge, self.rect, 1, border_radius=px(5))
        if self.text:
            fonts.draw(surf, self.text, self.rect.center, theme.TEXT, fonts.ui, "center")
        tips.add(self.rect, self.hint, self.tooltip)


class TextField(Widget):
    """Single-line text entry; ``kind`` (float, floatx, int, text) sets how
    :meth:`parsed` reads the value. Ctrl+Backspace clears, Ctrl+V pastes."""

    def __init__(self, value: str = "", kind: str = "float"):
        super().__init__()
        self.value = str(value)
        self.kind = kind
        self.focused = False
        self.error = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.focused = self.rect.collidepoint(ev.pos)
            if self.focused:
                pygame.key.start_text_input()
            return self.focused
        if not self.focused:
            return False
        if ev.type == pygame.TEXTINPUT:
            self.value += ev.text
            self.error = False
            return True
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_BACKSPACE:
                self.value = "" if ev.mod & pygame.KMOD_CTRL else self.value[:-1]
                return True
            if ev.key == pygame.K_v and ev.mod & pygame.KMOD_CTRL:
                with contextlib.suppress(Exception):     # no clipboard on this platform
                    pygame.scrap.init()
                    txt = pygame.scrap.get_text() if hasattr(pygame.scrap, "get_text") else ""
                    self.value += (txt or "").strip()
                return True
        return False

    def parsed(self):
        """The value converted per ``kind``; raises ValueError if it does not parse."""
        s = self.value.strip()
        if self.kind == "text":
            return s
        if self.kind == "int":
            return int(float(s))
        if self.kind == "floatx":        # float or keyword (e.g. "sso")
            try:
                return float(s)
            except ValueError:
                return s
        return float(s)

    def draw(self, surf, fonts):
        bg = theme.FIELD_FOCUS if self.focused else theme.FIELD
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(4))
        edge = theme.BAD if self.error else (theme.ACCENT if self.focused else theme.PANEL_EDGE)
        pygame.draw.rect(surf, edge, self.rect, 1, border_radius=px(4))
        clip = surf.get_clip()
        surf.set_clip(self.rect.inflate(-px(6), 0))
        txt = fonts.render(self.value, theme.TEXT, fonts.mono)
        x = self.rect.x + px(6)
        if txt.get_width() > self.rect.w - px(14):
            x = self.rect.right - px(8) - txt.get_width()
        surf.blit(txt, (x, self.rect.centery - txt.get_height() // 2))
        if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = min(self.rect.right - px(6), x + txt.get_width() + 1)
            pygame.draw.line(surf, theme.TEXT, (cx, self.rect.y + px(5)),
                             (cx, self.rect.bottom - px(5)), px(1))
        surf.set_clip(clip)


class Choice(Widget):
    """Click (or wheel) to cycle through options; right-click goes back."""

    def __init__(self, options: list[str], index: int = 0, on_change=None):
        super().__init__()
        self.options = list(options)
        self.index = max(0, min(index, len(self.options) - 1))
        self.on_change = on_change

    @property
    def value(self):
        """The selected option (empty string if there are none)."""
        return self.options[self.index] if self.options else ""

    def set(self, value):
        """Select ``value`` if it is one of the options."""
        if value in self.options:
            self.index = self.options.index(value)

    def _step(self, d):
        """Move ``d`` options along (wrapping) and notify ``on_change``."""
        if not self.options:
            return
        self.index = (self.index + d) % len(self.options)
        if self.on_change:
            self.on_change(self.value)

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            if ev.button in (1, 5):
                self._step(1)
                return True
            if ev.button in (3, 4):
                self._step(-1)
                return True
        elif ev.type == pygame.MOUSEWHEEL and self.hover:
            self._step(-1 if ev.y > 0 else 1)
            return True
        return False

    def draw(self, surf, fonts):
        bg = theme.mix(theme.FIELD, theme.ACCENT, 0.2) if self.hover else theme.FIELD
        pygame.draw.rect(surf, bg, self.rect, border_radius=px(4))
        pygame.draw.rect(surf, theme.PANEL_EDGE, self.rect, 1, border_radius=px(4))
        clip = surf.get_clip()
        # clip on the right only (for the arrows): a symmetric inset cut the first letter
        surf.set_clip(pygame.Rect(self.rect.x + px(2), self.rect.y, self.rect.w - px(26),
                                  self.rect.h))
        fonts.draw(surf, self.value, (self.rect.x + px(8), self.rect.centery), theme.TEXT,
                   fonts.ui, "midleft")
        surf.set_clip(clip)
        fonts.draw(surf, "<>", (self.rect.right - px(8), self.rect.centery), theme.DIM,
                   fonts.small, "midright")


class Checkbox(Widget):
    """A checkbox; ``on_change(value)`` fires on every click."""

    def __init__(self, value: bool = False, on_change=None):
        super().__init__()
        self.value = bool(value)
        self.on_change = on_change

    def handle(self, ev):
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos)):
            self.value = not self.value
            if self.on_change:
                self.on_change(self.value)
            return True
        return False

    def draw(self, surf, fonts):
        box = pygame.Rect(self.rect.x, self.rect.centery - px(9), px(18), px(18))
        pygame.draw.rect(surf, theme.FIELD, box, border_radius=px(3))
        edge = theme.ACCENT if self.value else theme.PANEL_EDGE
        pygame.draw.rect(surf, edge, box, 1, border_radius=px(3))
        if self.value:
            x, y = box.topleft
            tick = [(x + px(4), box.centery), (x + px(8), box.bottom - px(4)),
                    (box.right - px(4), y + px(4))]
            pygame.draw.lines(surf, theme.ACCENT, False, tick, px(2))


# --- Form dialog ------------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """One row of a :class:`FormDialog`; ``visible(raw_values)`` hides it conditionally.
    ``tip`` explains the field on hover; ``option_tips`` adds a line about the
    option a choice currently shows."""

    key: str
    label: str
    kind: str = "float"            # float | floatx | int | text | choice | bool | info
    default: Any = ""
    options: list = field(default_factory=list)
    visible: Callable[[dict], bool] | None = None
    wide: bool = False             # text spanning the full row (e.g. TLE lines)
    tip: str = ""
    option_tips: dict = field(default_factory=dict)


CHOICE_HELP = "Click or scroll to change it; right-click goes back."


class FormDialog:
    """A modal form. ``side`` is an optional panel drawn to the right of the
    fields: any object with ``width``, ``min_height``, ``draw(surf, rect,
    dialog)`` and ``handle(ev, dialog) -> bool``. It is left out when the
    window is too narrow for it. ``extra_buttons`` are (text, callback) or
    (text, callback, hint); ``ok_hint`` explains the OK button."""

    ROW = 32                       # design px, like ``width`` and the side panel's sizes
    MIN_ROW = 22

    def __init__(self, app, title: str, specs: list[FieldSpec], on_ok, ok_text: str = "OK",
                 width: int = 520, on_change=None, extra_buttons=None, subtitle: str = "",
                 side=None, ok_hint: str = ""):
        self.app = app
        self.title = title
        self.subtitle = subtitle
        self.specs = specs
        self.on_ok = on_ok
        self.on_change = on_change
        self.width = width
        self.side = side
        self.side_rect = None
        self.row_h = px(self.ROW)
        self.error = ""
        self.info: dict[str, str] = {}
        self.widgets: dict[str, Widget] = {}
        for s in specs:
            if s.kind == "choice":
                opts = list(s.options)
                idx = opts.index(s.default) if s.default in opts else 0
                self.widgets[s.key] = Choice(opts, idx,
                                             on_change=lambda v, k=s.key: self.changed(k))
            elif s.kind == "bool":
                self.widgets[s.key] = Checkbox(bool(s.default),
                                               on_change=lambda v, k=s.key: self.changed(k))
            elif s.kind == "info":
                self.info[s.key] = str(s.default)
            else:
                self.widgets[s.key] = TextField(s.default, s.kind)
        self.buttons = [Button(ok_text, self.submit, accent=True, tooltip="Enter",
                               hint=ok_hint or f"{ok_text} with these settings"),
                        Button("Cancel", self.close, tooltip="Esc",
                               hint="Close without changing anything")]
        for text, cb, *hint in (extra_buttons or []):
            self.buttons.insert(-1, Button(text, lambda cb=cb: self._extra(cb),
                                           hint=hint[0] if hint else ""))
        self.rect = pygame.Rect(0, 0, px(width), px(100))
        self.layout()

    # --- values -----------------------------------------------------------------------
    def raw(self) -> dict:
        """Unparsed widget values by key (text as typed)."""
        return {k: w.value for k, w in self.widgets.items()}

    def set(self, key, value):
        """Set a field's value, whatever kind of widget it is."""
        w = self.widgets.get(key)
        if isinstance(w, TextField):
            w.value = str(value)
        elif isinstance(w, Choice):
            w.set(value)
        elif isinstance(w, Checkbox):
            w.value = bool(value)
        elif key in self.info:
            self.info[key] = str(value)

    def values(self) -> dict:
        """Parsed values of *visible* fields; raises ValueError naming the field."""
        raw = self.raw()
        out = {}
        for s in self.specs:
            if s.kind == "info" or not self._visible(s, raw):
                continue
            w = self.widgets[s.key]
            if isinstance(w, TextField):
                try:
                    out[s.key] = w.parsed()
                    w.error = False
                except ValueError:
                    w.error = True
                    raise ValueError(f"'{s.label}' needs a number") from None
            else:
                out[s.key] = w.value
        return out

    def _visible(self, s: FieldSpec, raw) -> bool:
        return s.visible is None or bool(s.visible(raw))

    def changed(self, key):
        """A field changed: tell ``on_change`` and re-layout (visibility may change)."""
        if self.on_change:
            try:
                self.on_change(self, key)
            except Exception as exc:  # keep the dialog alive on bad intermediate input
                self.error = str(exc)
        self.layout()

    def _extra(self, cb):
        """Run an extra button's callback; its return value (or error) is shown."""
        try:
            msg = cb(self)
            self.error = msg or ""
        except Exception as exc:
            self.error = str(exc)

    # --- layout -----------------------------------------------------------------------
    def layout(self):
        """Size and center the dialog and place every visible row and button."""
        raw = self.raw()
        rows = [s for s in self.specs if self._visible(s, raw)]
        sw, sh = self.app.screen.get_size()
        width = px(self.width)
        side_w = 0
        if self.side is not None:
            side_w = min(px(self.side.width), sw - width - px(34))
            side_w = side_w if side_w >= px(260) else 0
        total_w = width + (side_w + px(6) if side_w else 0)
        top = px(52 + (22 if self.subtitle else 0))
        avail = sh - px(20) - top - px(90)
        # squeeze the rows (never below MIN_ROW) rather than run off the screen
        row = max(px(self.MIN_ROW), min(px(self.ROW), avail // max(1, len(rows))))
        body = row * len(rows) + px(12)
        if side_w:
            body = max(body, min(px(self.side.min_height), avail + px(12)))
        h = min(top + body + px(78), sh - px(20))
        self.row_h = row
        self.rect = pygame.Rect((sw - total_w) // 2, max(px(10), (sh - h) // 2), total_w, h)
        self.side_rect = (pygame.Rect(self.rect.x + width, self.rect.y + top - px(6), side_w,
                                      self.rect.bottom - px(56) - (self.rect.y + top - px(6)))
                          if side_w else None)
        y = self.rect.y + top
        lx = self.rect.x + px(18)
        right = self.rect.x + width - px(18)
        wx = self.rect.x + int(width * 0.44)
        ww = right - wx
        gap = px(6)                                     # between rows
        self._rows = []
        for s in rows:
            if s.kind == "info":
                self._rows.append((s, pygame.Rect(lx, y, width - px(36), row - gap)))
            else:
                w = self.widgets[s.key]
                if s.wide:
                    w.rect = pygame.Rect(lx + px(60), y, right - lx - px(60), row - gap)
                elif s.kind == "bool":
                    w.rect = pygame.Rect(wx, y, px(24), row - gap)
                else:
                    w.rect = pygame.Rect(wx, y, ww, row - gap)
                self._rows.append((s, w.rect))
            y += row
        bx = self.rect.right - px(18)
        for b in reversed(self.buttons):
            bw = max(px(90), self.app.fonts.ui.size(b.text)[0] + px(28))
            b.rect = pygame.Rect(bx - bw, self.rect.bottom - px(46), bw, px(32))
            bx -= bw + px(10)
        self._visible_keys = {s.key for s in rows}

    # --- events -----------------------------------------------------------------------
    def submit(self):
        """Parse and pass the values to ``on_ok``; close unless it returns a message."""
        try:
            vals = self.values()
            msg = self.on_ok(vals)
        except Exception as exc:
            self.error = str(exc)
            return
        if msg:
            self.error = msg
        else:
            self.close()

    def close(self):
        """Close without submitting (Cancel, Esc)."""
        pygame.key.stop_text_input()
        self.app.close_dialog(self)

    def _text_fields(self):
        return [self.widgets[s.key] for s in self.specs
                if s.key in self._visible_keys
                and isinstance(self.widgets.get(s.key), TextField)]

    def handle(self, ev) -> bool:
        """Keyboard shortcuts (Esc, Enter, Tab), then buttons, side panel and fields."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.close()
                return True
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.submit()
                return True
            if ev.key == pygame.K_TAB:
                tf = self._text_fields()
                if tf:
                    cur = next((i for i, w in enumerate(tf) if w.focused), -1)
                    for w in tf:
                        w.focused = False
                    step = -1 if ev.mod & pygame.KMOD_SHIFT else 1
                    tf[(cur + step) % len(tf)].focused = True
                return True
        for b in self.buttons:
            if b.handle(ev):
                return True
        if self.side_rect is not None and self.side.handle(ev, self):
            return True
        visible = [s for s in self.specs if s.key in self._visible_keys and s.key in self.widgets]
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            # every widget sees clicks so text fields can lose focus
            for s in visible:
                self.widgets[s.key].handle(ev)
            return True
        for s in visible:
            w = self.widgets[s.key]
            if w.handle(ev):
                if isinstance(w, TextField):
                    self.changed(s.key)
                return True
        return True    # modal: swallow everything

    def field_tip(self, s: FieldSpec, raw: dict) -> str:
        """The hover tip of field ``s``: its ``tip``, what the chosen option means,
        and how to change a choice."""
        parts = [s.tip]
        if s.kind == "choice":
            parts += [s.option_tips.get(raw.get(s.key), ""), CHOICE_HELP]
        return "\n".join(p for p in parts if p)

    def draw(self, surf):
        """Shade the window, then draw the frame, rows, side panel, buttons and error."""
        fonts = self.app.fonts
        shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        surf.blit(shade, (0, 0))
        tips.block(surf.get_rect())             # nothing under the shade explains itself
        theme.panel(surf, self.rect, (14, 20, 36, 245), theme.ACCENT)
        head = fonts.draw(surf, self.title, (self.rect.x + px(18), self.rect.y + px(14)),
                          theme.TEXT, fonts.title)
        tips.add(head, "Enter applies the form, Esc closes it, Tab moves between the text "
                       "fields. Rest the mouse on a field to see what it means.")
        if self.subtitle:
            sub = fonts.draw(surf, self.subtitle, (self.rect.x + px(18), self.rect.y + px(42)),
                             theme.DIM, fonts.small)
            tips.add(sub, self.subtitle)
        raw = self.raw()
        for s, r in self._rows:
            if s.kind == "info":
                text = self.info.get(s.key, "")
                fonts.draw(surf, text, (r.x, r.centery), theme.ACCENT, fonts.small, "midleft")
                tips.add(r, "\n".join(p for p in (text, s.tip) if p))
                continue
            fonts.draw(surf, s.label, (self.rect.x + px(18), r.centery), theme.DIM, fonts.ui,
                       "midleft")
            self.widgets[s.key].draw(surf, fonts)
            tips.add(pygame.Rect(self.rect.x + px(18), r.y, r.right - self.rect.x - px(18), r.h),
                     self.field_tip(s, raw))
        if self.side_rect is not None:
            self.side.draw(surf, self.side_rect, self)
        for b in self.buttons:
            b.draw(surf, fonts)
        if self.error:
            err = fonts.draw(surf, self.error[:90], (self.rect.x + px(18),
                                                     self.rect.bottom - px(30)),
                             theme.BAD, fonts.small, "midleft")
            tips.add(err, self.error)
