"""Orbit geometry drawn in the 3-D view around the selected satellite.

``basic``  translucent orbital plane, line of nodes (AN/DN), line of apsides
           with perigee/apogee altitudes
``full``   adds the equatorial plane and angle arcs for the inclination i,
           RAAN (from the vernal equinox), argument of perigee and true
           anomaly, the angular-momentum vector h, the radius vector and a
           callout card beside the satellite

Filled planes are split along the Earth's silhouette plane: the far part is
drawn before the ray-cast Earth (which covers what it hides) and the near
part after it, so the plane visibly slices through the globe.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..constants import R_EARTH
from ..elements import perifocal_axes
from ..orbitinfo import OrbitInfo
from . import glossary, theme
from .orbitpanel import APSIS, EQUATOR, H_VEC, NODE, countdown, deg, num
from .render3d import Line
from .theme import px
from .tips import add as add_tip

MODES = ("full", "basic", "off")
ARC_I = (255, 150, 90)
ARC_RAAN = (120, 200, 255)
ARC_ARGP = (205, 150, 255)
R_FILL_IN = R_EARTH * 1.001
ROUND_E = 0.01          # below this, arcs use the argument of latitude
LEGEND_TITLE = ("ORBIT GEOMETRY KEY  - click to hide", "ORBIT GEOMETRY KEY  + click to show")
LEGEND_TIP = ("Key to the lines, planes and angles drawn around the selected satellite's "
              "orbit. Click to fold or open it; D switches the drawing between full, basic "
              "and off.")
LEGEND_ROWS = {         # hover tips of the key's rows, by the row's name
    "orbital plane": "The flat plane the orbit lies in, shaded in the satellite's color.",
    "equatorial plane": "The Earth's equator extended into space. The orbit's tilt against "
                        "it is the inclination.",
    "line of apsides": "Joins the perigee (lowest point) and the apogee (highest point); "
                       "their altitudes are labeled at the ends.",
    "line of nodes": "Where the orbital plane cuts the equator: AN where the satellite "
                     "crosses going north, DN going south.",
    "u arg. of latitude": glossary.ROWS["Arg. of latitude"],
    "velocity": "Direction the satellite is moving now.",
    "h ang. momentum": glossary.LABELS["h"],
}
X_HAT, Y_HAT, Z_HAT = np.eye(3)
ORIGIN = np.zeros(3)


def _unit(v):
    return v / np.linalg.norm(v)


def _arc(center, u, w, radius, a0, a1, n=48):
    """Points of a circular arc in the plane of unit vectors ``u``, ``w`` from angle
    ``a0`` to ``a1`` (``n`` points per half turn)."""
    t = np.linspace(a0, a1, max(3, int(n * abs(a1 - a0) / math.pi) + 3))
    return center + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)


def _dashed(p0, p1, color, n=24, width=2):
    """A dashed straight line: one Line whose dashes are separated by NaN
    points (the renderer breaks polylines at non-finite points)."""
    t = np.linspace(0.0, 1.0, 2 * n)
    pts = p0 + t[:, None] * (p1 - p0)
    pts = np.insert(pts, np.arange(2, 2 * n, 2), np.nan, axis=0)
    return [Line(pts, color, width, 0.4)]


def _visible_near(cam, pts, k, view=None):
    """Index of the point of ``pts`` nearest index ``k`` that lies in ``view``
    (a screen rect; default the whole window), with room for a label, and is
    not hidden by the Earth (``k`` itself when none is)."""
    sx, sy, _ = cam.project(pts)
    x0, y0, x1, y1 = (0, 0, cam.width, cam.height) if view is None else (
        view.left + px(12), view.top + px(16), view.right - px(12), view.bottom - px(16))
    with np.errstate(invalid="ignore"):
        seen = (np.isfinite(sx) & np.isfinite(sy) & (sx >= x0) & (sx < x1)
                & (sy >= y0) & (sy < y1) & ~cam.hidden_by_sphere(pts))
    idx = np.flatnonzero(seen)
    return int(idx[np.argmin(np.abs(idx - k))]) if idx.size else k


def _angle(lines, markers, fills, world, seen, center, u, w, radius, angle, color, label):
    """Add an angle measured from direction ``u`` turning toward ``w``: the wedge
    it spans, shaded (clear of the Earth when centered on it) with its two edges,
    an arc with an arrowhead pointing the way it is measured, and ``label`` tied
    to the arc at the visible point nearest its middle (``seen(pts, k)`` finds it)."""
    t = np.linspace(0.0, angle, max(8, int(48 * abs(angle) / math.pi) + 3))
    dirs = np.cos(t)[:, None] * u + np.sin(t)[:, None] * w
    arc = center + radius * dirs
    at_center = not center.any()
    if at_center:           # the part inside the Earth is left out, like the orbital plane
        inner = R_FILL_IN * dirs[::-1]
        edges = (R_FILL_IN * dirs[0], R_FILL_IN * dirs[-1])
    else:
        inner = center[None, :]
        edges = (center, center)
    fills.append((world(np.concatenate([arc, inner])), (*color[:3], 38)))
    edge_col = theme.dim(color, 0.8)
    lines.append(Line(world(np.linspace(edges[0], arc[0], 12)), edge_col, 1, 0.4))
    lines.append(Line(world(np.linspace(edges[1], arc[-1], 12)), edge_col, 1, 0.4))
    lines.append(Line(world(arc), color, 2, 0.4))
    # arrowhead at the end of the arc, in the angle's plane
    a = float(t[-1])
    tangent = math.copysign(1.0, angle) * (-math.sin(a) * u + math.cos(a) * w)
    outward = math.cos(a) * u + math.sin(a) * w
    head = min(0.14 * radius, 0.6 * abs(angle) * radius)
    tips = np.stack([arc[-1] - head * tangent + 0.45 * head * outward, arc[-1],
                     arc[-1] - head * tangent - 0.45 * head * outward])
    lines.append(Line(world(tips), color, 2, 0.4))
    k = seen(world(arc), len(arc) // 2)
    markers.append((world(arc[k]), color, 0, label, False))


def legend_rect(fonts, rows, bounds: pygame.Rect, shown: bool = True):
    """Where the key to the overlay goes: the top-left corner of ``bounds``
    (None if it does not fit); only its title bar when not ``shown``."""
    if not rows:
        return None
    title_w = fonts.small.size(LEGEND_TITLE[0])[0] + px(20)
    if not shown:
        rect = pygame.Rect(bounds.x + px(8), bounds.y + px(8), title_w, px(24))
    else:
        name_w = max(fonts.small.size(r[2])[0] for r in rows)
        desc_w = max(fonts.small.size(r[3])[0] for r in rows)
        w = max(px(34) + name_w + px(12) + desc_w + px(10), title_w)
        rect = pygame.Rect(bounds.x + px(8), bounds.y + px(8), w, px(28) + px(17) * len(rows))
    return rect if bounds.contains(rect) else None


def draw_legend(surf, fonts, rows, rect: pygame.Rect, shown: bool = True):
    """The key: a sample of each element of the overlay, its name and what it
    spans or points at (just the title bar when not ``shown``)."""
    theme.panel(surf, rect, (10, 16, 30, 228), theme.PANEL_EDGE, 6)
    fonts.draw(surf, LEGEND_TITLE[0 if shown else 1], (rect.x + px(10), rect.y + px(5)),
               theme.ACCENT, fonts.small)
    add_tip(rect, LEGEND_TIP, "D")
    if not shown:
        return
    name_w = max(fonts.small.size(r[2])[0] for r in rows)
    y, lw = rect.y + px(26), px(2)
    for col, kind, name, desc in rows:
        x0, mid = rect.x + px(10), y + px(8)
        if kind == "fill":
            sw = pygame.Rect(x0, y + px(3), px(18), px(11))
            theme.panel(surf, sw, (*col[:3], 90), col, 2)
        elif kind == "dash":
            for dx in (0, 7, 14):
                pygame.draw.line(surf, col, (x0 + px(dx), mid), (x0 + px(dx + 4), mid), lw)
        elif kind == "arrow":
            pygame.draw.line(surf, col, (x0, mid), (x0 + px(17), mid), lw)
            pygame.draw.lines(surf, col, False, [(x0 + px(12), mid - px(4)), (x0 + px(17), mid),
                                                 (x0 + px(12), mid + px(4))], lw)
        else:                                   # an angle: a shaded wedge and its arc
            c, r = (x0, y + px(15)), px(18)
            t = np.linspace(0.0, 0.8, 8)
            pts = [(c[0] + r * math.cos(a), c[1] - r * math.sin(a)) for a in t]
            theme_poly = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(theme_poly, (*col[:3], 90),
                                [(u - rect.x, v - rect.y) for u, v in [c] + pts])
            surf.blit(theme_poly, rect.topleft)
            pygame.draw.lines(surf, col, False, pts, lw)
        tx = x0 + px(24)
        fonts.draw(surf, name, (tx, y), theme.mix(col, (255, 255, 255), 0.35), fonts.small)
        fonts.draw(surf, desc, (tx + name_w + px(12), y), theme.DIM, fonts.small)
        add_tip(pygame.Rect(rect.x, y, rect.w, px(17)), legend_tip(name, desc))
        y += px(17)


def legend_tip(name: str, desc: str) -> str:
    """What a row of the key means: its own tip, or the property it draws."""
    if name in LEGEND_ROWS:
        return LEGEND_ROWS[name]
    prop = {"i": "Inclination i", "\N{GREEK CAPITAL LETTER OMEGA}": "RAAN",
            "\N{GREEK SMALL LETTER OMEGA}": "Arg. of perigee",
            "\N{GREEK SMALL LETTER NU}": "True anomaly"}.get(name.split(" ")[0])
    return glossary.ROWS.get(prop, desc) if prop else desc


def _arrow(p0, p1, color, side, width=2):
    """Shaft and head of an arrow; the head opens across ``side`` (the view axis)."""
    d = p1 - p0
    L = np.linalg.norm(d)
    if L == 0:
        return []
    d = d / L
    s = _unit(np.cross(d, side)) if np.linalg.norm(np.cross(d, side)) > 1e-9 else np.zeros(3)
    head = min(0.08 * L, 900.0)
    tips = np.stack([p1 - head * d + 0.5 * head * s, p1, p1 - head * d - 0.5 * head * s])
    return [Line(np.linspace(p0, p1, 16), color, width, 0.4), Line(tips, color, width, 0.4)]


def sector_polygon(P, Q, p, e, nu, r_in=R_FILL_IN):
    """Closed 3-D polygon of the region between radius ``r_in`` and the conic
    r = p / (1 + e cos nu) over the true-anomaly samples ``nu``: the outline
    forward, then the inner arc back (a keyhole when ``nu`` spans 2 pi)."""
    r_out = p / (1.0 + e * np.cos(nu))
    dirs = np.cos(nu)[:, None] * P + np.sin(nu)[:, None] * Q
    inner = np.minimum(r_in, r_out)[:, None] * dirs
    return np.concatenate([r_out[:, None] * dirs, inner[::-1]])


def clip_polygon(poly, n, d):
    """Sutherland-Hodgman: the part of a closed polygon where p . n >= d."""
    if len(poly) == 0:
        return poly
    f = poly @ n - d
    nxt = np.roll(poly, -1, axis=0)
    fn = np.roll(f, -1)
    inside = f >= 0
    cross = inside != (fn >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, f / (f - fn), 0.0)
    both = np.stack([poly, poly + t[:, None] * (nxt - poly)], axis=1)
    return both[np.stack([inside, cross], axis=1)]


def draw_fill(surf, cam, poly_world, rgba, part: str):
    """Draw the ``near`` or ``far`` part of a translucent planar polygon.

    The Earth's silhouette seen from the camera lies in the plane
    p . c_hat = R^2 / |c|. Outside the Earth, everything in front of that
    plane is visible, so the near part is drawn after the Earth; the far part
    is drawn before it and the Earth then covers exactly the hidden piece.
    """
    c = cam.position
    cn = float(np.linalg.norm(c))
    c_hat = c / cn
    k = R_EARTH ** 2 / cn
    sign = 1.0 if part == "near" else -1.0
    poly = clip_polygon(poly_world, sign * c_hat, sign * k)
    poly = clip_polygon(poly, cam.forward, float(cam.forward @ c) + 2 * cam.near)
    if len(poly) < 3:
        return
    sx, sy, _ = cam.project(poly)
    ok = np.isfinite(sx) & np.isfinite(sy)
    if not ok.all():
        return
    w, h = surf.get_size()
    sx = np.clip(sx, -4 * w, 5 * w)
    sy = np.clip(sy, -4 * h, 5 * h)
    x0, x1 = max(0, int(sx.min())), min(w, int(sx.max()) + 2)
    y0, y1 = max(0, int(sy.min())), min(h, int(sy.max()) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    layer = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
    pygame.draw.polygon(layer, rgba, np.stack([sx - x0, sy - y0], 1).tolist())
    surf.blit(layer, (x0, y0))


def build(info: OrbitInfo, r, v, color, W, cam, mode: str, view=None):
    """Lines, label markers, fills (polygon_world, rgba) and, in ``full`` mode,
    the legend's rows (color, sample kind, name, what it spans) for one orbit."""
    lines, markers, fills, legend = [], [], [], []
    if mode == "off":
        return lines, markers, fills, legend
    el = info.el
    P, Q = perifocal_axes(el.i, el.raan, el.argp)
    Hn = np.cross(P, Q)
    e, p = float(el.e), float(el.p)
    side = cam.forward

    def world(x):
        return np.asarray(x) @ W.T

    def seen(pts, k):           # where along ``pts`` a label can go
        return _visible_near(cam, pts, k, view)

    # orbital plane
    if info.closed:
        nu = np.linspace(0.0, 2 * np.pi, 181)
    else:
        rcap = max(3.0e5, 1.5 * info.radius)
        lim = min(math.acos(np.clip((p / rcap - 1.0) / e, -1, 1)), math.acos(-1.0 / e) - 1e-3)
        nu = np.linspace(-lim, lim, 181)
    fills.append((world(sector_polygon(P, Q, p, e, nu)), (*color[:3], 40)))
    legend.append((color, "fill", "orbital plane", "the satellite's plane"))

    def at(nu_):
        den = 1.0 + e * math.cos(nu_)
        if den <= 1e-6:
            return None
        rr = p / den
        if not info.closed and rr > 3e5:
            return None
        return rr * (math.cos(nu_) * P + math.sin(nu_) * Q)

    # line of apsides
    pe = at(0.0)
    if not info.circular:
        ap = at(math.pi) if info.closed else None
        lines += _dashed(world(pe), world(ap if ap is not None else ORIGIN),
                         theme.dim(APSIS, 0.75))
        markers.append((world(pe), APSIS, -2, f"perigee {num(info.rp_alt, 'km', 0)}", False))
        if ap is not None:
            markers.append((world(ap), APSIS, -2, f"apogee {num(info.ra_alt, 'km', 0)}",
                            False))
        legend.append((APSIS, "dash", "line of apsides",
                       "perigee \N{LEFT RIGHT ARROW} apogee"))

    # line of nodes
    an = dn = None
    n_hat = np.array([math.cos(el.raan), math.sin(el.raan), 0.0])
    v_hat = np.cross(Hn, n_hat)          # in the orbit plane, 90 deg past the node
    if not info.equatorial:
        an, dn = at(-el.argp), at(math.pi - el.argp)
        a_w = world(an) if an is not None else ORIGIN
        d_w = world(dn) if dn is not None else ORIGIN
        lines += _dashed(d_w, a_w, NODE, 30)
        if an is not None:
            markers.append((a_w, NODE, -2, "AN ascending node", False))
        if dn is not None:
            markers.append((d_w, NODE, -2, "DN descending node", False))
        legend.append((NODE, "dash", "line of nodes", "AN north-bound, DN south"))

    if mode != "full":
        return lines, markers, fills, []

    # equatorial plane out past the node crossings, and the reference directions
    node_r = [np.linalg.norm(x) for x in (an, dn) if x is not None]
    r_eq = min(1.12 * max(node_r + [info.radius, 1.6 * R_EARTH]), 3e5)
    if not info.equatorial:
        a = np.linspace(0, 2 * np.pi, 145)
        fills.append((world(sector_polygon(X_HAT, Y_HAT, r_eq, 0.0, a)), (*EQUATOR, 24)))
        ring = world(_arc(ORIGIN, X_HAT, Y_HAT, r_eq, 0, 2 * np.pi, 96))
        lines.append(Line(ring, theme.dim(EQUATOR, 0.8), 1, 0.4))
        markers.append((ring[seen(ring, 0)], EQUATOR, 0, "equatorial plane", False))
        legend.insert(1, (EQUATOR, "fill", "equatorial plane", "the equator extended"))

    base = max(R_EARTH, 0.35 * el.rp)
    rn, rw, rv = 1.80 * base, 1.55 * base, 1.30 * base     # arc radii: RAAN, argp, u / nu
    bright = theme.mix(color, (255, 255, 255), 0.35)
    ang = (lines, markers, fills, world, seen)
    # RAAN: in the equator from the vernal equinox (+X) to the ascending node
    if not info.equatorial:
        lines += _dashed(ORIGIN, world(X_HAT * rn * 1.12), ARC_RAAN, 10)
        markers.append((world(X_HAT * rn * 1.12), ARC_RAAN, 0,
                        "\N{GREEK SMALL LETTER GAMMA} vernal equinox", False))
        if el.raan > 1e-3:
            _angle(*ang, ORIGIN, X_HAT, Y_HAT, rn, el.raan, ARC_RAAN,
                   f"\N{GREEK CAPITAL LETTER OMEGA} RAAN {deg(el.raan, 1)}")
            legend.append((ARC_RAAN, "angle", "\N{GREEK CAPITAL LETTER OMEGA} RAAN",
                           "\N{GREEK SMALL LETTER GAMMA} \N{RIGHTWARDS ARROW} AN, in the equator"))
    # near-circular orbits have an ill-defined perigee: show the argument of
    # latitude (node -> satellite) instead of the perigee and true anomaly
    round_ = el.e < ROUND_E
    if round_ and not info.equatorial and el.u > 1e-3:
        _angle(*ang, ORIGIN, n_hat, v_hat, rv, el.u, bright,
               f"u arg. of latitude {deg(el.u, 1)}")
        legend.append((bright, "angle", "u arg. of latitude",
                       "AN \N{RIGHTWARDS ARROW} satellite"))
    # argument of perigee: in the orbit plane from the node to the perigee
    if not info.equatorial and not round_ and el.argp > 1e-3:
        _angle(*ang, ORIGIN, n_hat, v_hat, rw, el.argp, ARC_ARGP,
               f"\N{GREEK SMALL LETTER OMEGA} arg. of perigee {deg(el.argp, 1)}")
        legend.append((ARC_ARGP, "angle", "\N{GREEK SMALL LETTER OMEGA} arg. of perigee",
                       "AN \N{RIGHTWARDS ARROW} perigee"))
    # true anomaly: from the perigee (or node, if circular) to the satellite
    nu0 = float(el.nu) if info.closed else (float(el.nu) + np.pi) % (2 * np.pi) - np.pi
    if abs(nu0) > 1e-3 and not (round_ and not info.equatorial):
        _angle(*ang, ORIGIN, P, Q, rv, nu0, bright,
               f"\N{GREEK SMALL LETTER NU} true anomaly {deg(el.nu, 1)}")
        legend.append((bright, "angle", "\N{GREEK SMALL LETTER NU} true anomaly",
                       "perigee \N{RIGHTWARDS ARROW} satellite"))
    # inclination at the ascending node, from the equator (east) to the track
    if an is not None:
        east = _unit(np.cross(Z_HAT, n_hat))
        rho = max(0.45 * R_EARTH, 0.12 * np.linalg.norm(an))
        lines.append(Line(world(np.stack([an - 0.4 * rho * east, an + 1.5 * rho * east])),
                          EQUATOR, 1, 0.4))
        lines += _dashed(world(an), world(an + 1.5 * rho * v_hat), bright, 6)
        _angle(*ang, an, east, Z_HAT, rho, el.i, ARC_I, f"i inclination {deg(el.i, 2)}")
        legend.append((ARC_I, "angle", "i inclination",
                       "equator \N{RIGHTWARDS ARROW} track, at AN"))
    # angular momentum, radius and velocity vectors
    h_len = 2.1 * base
    lines += _arrow(ORIGIN, world(Hn * h_len), H_VEC, side)
    shaft = world(np.linspace(ORIGIN, Hn * h_len, 16))
    markers.append((shaft[seen(shaft, 15)], H_VEC, 0, "h orbit normal", False))
    legend.append((H_VEC, "arrow", "h ang. momentum", "orbit normal"))
    lines.append(Line(np.linspace(ORIGIN, world(r), 24), theme.dim(color, 0.8), 1, 0.3))
    tip = r + _unit(v) * 0.5 * base
    lines += _arrow(world(r), world(tip), (255, 255, 255), side)
    shaft = world(np.linspace(r, tip, 8))
    markers.append((shaft[seen(shaft, 7)], (255, 255, 255), 0, "velocity", False))
    legend.append(((255, 255, 255), "arrow", "velocity", "direction of motion"))
    return lines, markers, fills, legend


def draw_callout(surf, app, info: OrbitInfo, sat, pos, bounds: pygame.Rect):
    """Small card beside the selected satellite with its key numbers."""
    fonts = app.fonts
    rows = [f"alt {num(info.altitude, 'km', 0):>10}   v {info.speed:.3f} km/s"]
    if info.closed and not info.circular:
        rows.append(f"Pe {num(info.rp_alt, 'km', 0):>11}   Ap {num(info.ra_alt, 'km', 0)}")
    elif info.closed:
        rows.append(f"circular, e = {info.el.e:.5f}")
    else:
        rows.append(f"Pe {num(info.rp_alt, 'km', 0):>11}   C3 {info.c3:.2f}")
    period = countdown(info.period) if info.closed else "open"
    rows.append(f"i  {deg(info.el.i, 2):>11}   T {period}")
    passes = ((info.t_peri, "Pe"), (info.t_apo, "Ap"), (info.t_an, "AN"), (info.t_dn, "DN"))
    nxt = [(t, n) for t, n in passes if math.isfinite(t)]
    if nxt:
        t, n = min(nxt)
        rows.append(f"next {n} in {countdown(t)}")
    w = max(fonts.small.size(r)[0] for r in rows) + px(20)
    w = max(w, fonts.bold.size(sat.name)[0] + px(30))
    h = px(30) + px(16) * len(rows)
    x, y = int(pos[0]) + px(26), int(pos[1]) - h - px(18)
    rect = pygame.Rect(x, y, w, h)
    if rect.right > bounds.right:
        rect.right = int(pos[0]) - px(26)
    if rect.top < bounds.top:
        rect.top = int(pos[1]) + px(18)
    rect.clamp_ip(bounds)
    corner = (rect.left if rect.centerx > pos[0] else rect.right,
              rect.bottom if rect.centery < pos[1] else rect.top)
    pygame.draw.aaline(surf, theme.dim(sat.color, 0.9), pos, corner)
    theme.panel(surf, rect, (10, 16, 30, 225), theme.dim(sat.color, 0.8), 6)
    pygame.draw.circle(surf, sat.color, (rect.x + px(12), rect.y + px(13)), px(5))
    fonts.draw(surf, sat.name, (rect.x + px(22), rect.y + px(4)), theme.TEXT, fonts.bold)
    yy = rect.y + px(26)
    for r_ in rows:
        fonts.draw(surf, r_, (rect.x + px(10), yy), theme.TEXT, fonts.small)
        yy += px(16)
    add_tip(rect, f"{sat.name} at a glance: altitude and speed, perigee (Pe) and apogee (Ap) "
                  "altitudes, inclination i and period T, and the next special point of the "
                  "orbit. The right panel has the full list.")
    return rect
