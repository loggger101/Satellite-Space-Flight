"""The "Orbit" tab of the right-hand panel: diagrams and a property sheet
for the selected satellite.

* the orbital plane seen face-on, to scale, with the Earth's shadow cut
  through it, apsides, line of nodes, the satellite and its velocity
* the inclination seen edge-on along the line of nodes
* the orbit seen from above the north pole: RAAN, the vernal equinox and
  the Sun direction (local time of the ascending node)
* a timeline of the next revolution: sunlight, eclipse, apsis and node passes
* altitude and speed gauges between their perigee and apogee values
* grouped orbital, timing, perturbation and spacecraft properties
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..constants import R_EARTH
from ..eclipse import shadow_fraction
from ..elements import perifocal_axes
from ..ephemeris import sun_position
from ..orbitinfo import OrbitInfo
from . import glossary, theme, tips
from .theme import px

SUNLIT = (255, 214, 120)
SHADOW = (40, 48, 92)
NODE = (255, 206, 110)
APSIS = (225, 230, 245)
EQUATOR = (100, 160, 230)
EARTH_DAY = (58, 118, 196)
EARTH_NIGHT = (24, 40, 74)
EARTH_EDGE = (120, 170, 230)
H_VEC = (150, 240, 150)
BOX = (8, 12, 24, 200)
LABEL_BACK = (4, 7, 16, 200)      # behind diagram labels


def num(x, unit="", prec=1):
    """``x`` with thousands separators, ``prec`` decimals and ``unit``; '-' if missing."""
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:,.{prec}f}" + (f" {unit}" if unit else "")


def countdown(s: float) -> str:
    """Compact duration such as ``2d 3h 04m``, ``1h 05m`` or ``4m 09s``."""
    if not math.isfinite(s):
        return "-"
    s = int(round(s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def deg(x, prec=2):
    """Angle ``x`` (rad) in degrees with a degree sign."""
    return num(math.degrees(x), "", prec) + "\N{DEGREE SIGN}" if math.isfinite(x) else "-"


def hours_clock(h: float) -> str:
    """Hours of the day as ``HH:MM``."""
    if not math.isfinite(h):
        return "-"
    m = int(round(h * 60)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# --- small drawing helpers ------------------------------------------------------------------

def _alpha_poly(surf, color, alpha, pts, clip: pygame.Rect):
    """Translucent filled polygon, drawn through a layer the size of ``clip``."""
    if len(pts) < 3:
        return
    layer = theme.scratch(clip.size)
    pygame.draw.polygon(layer, (*color[:3], alpha), [(x - clip.x, y - clip.y) for x, y in pts])
    surf.blit(layer, clip.topleft)


def _dashed(surf, color, a, b, dash=5, gap=4, width=1):
    """Dashed screen-space line from ``a`` to ``b`` (``dash``, ``gap``, ``width`` in
    design px)."""
    dash, gap, width = px(dash), px(gap), px(width)
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    t = 0.0
    while t < length:
        t1 = min(t + dash, length)
        pygame.draw.line(surf, color, (ax + ux * t, ay + uy * t), (ax + ux * t1, ay + uy * t1),
                         width)
        t = t1 + gap


def _arrow(surf, color, a, b, head=6, width=1):
    """Screen-space arrow from ``a`` to ``b`` (``head`` and ``width`` in design px)."""
    head, width = px(head), px(width)
    pygame.draw.line(surf, color, a, b, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-1, 1):
        pygame.draw.line(surf, color, b, (b[0] - head * math.cos(ang + s * 0.45),
                                          b[1] - head * math.sin(ang + s * 0.45)), width)


def _arc(surf, color, center, radius, a0, a1, width=1):
    """Arc from angle a0 to a1 (rad, counter-clockwise on screen, y up); ``radius``
    in screen px, ``width`` in design px."""
    width = px(width)
    n = max(4, int(abs(a1 - a0) / 0.06))
    t = np.linspace(a0, a1, n)
    pts = np.stack([center[0] + radius * np.cos(t), center[1] - radius * np.sin(t)], 1)
    pygame.draw.lines(surf, color, False, pts.tolist(), width)


DIAGRAM_TIPS = {        # by the first word of the diagram's title
    "ORBITAL": "The orbit seen face-on and to scale, perigee to the right, the satellite "
               "moving counterclockwise. The line is bright in sunlight and dark in the Earth's "
               "shadow (the dark band); the arrow is the velocity; e is the eccentricity.",
    "INCLINATION": "The orbit seen edge-on, looking along the line of nodes: the dashed line "
                   "is the equator, the colored one the orbital plane, the arc between them "
                   "the inclination. h is the orbit normal, N the north pole.",
    "FROM": "The orbit seen from above the north pole. The arc from the vernal equinox "
            "(\N{GREEK SMALL LETTER GAMMA}) to the ascending node (AN) is the RAAN; the "
            "dashed yellow line points to the Sun. LTAN is the local solar time at the node.",
    "NEXT": "The next revolution, from now (left) to one period later (right): yellow in "
            "sunlight, blue in the Earth's shadow, with the times until perigee (Pe), apogee "
            "(Ap) and the ascending and descending nodes (AN, DN).",
}


def _box(surf, rect, fonts, title, right=None):
    """Diagram frame with a title (and optional right-aligned note)."""
    tips.add(rect, DIAGRAM_TIPS.get(title.split(" ")[0], ""), clip=surf.get_clip())
    theme.panel(surf, rect, BOX, theme.PANEL_EDGE, 6)
    fonts.draw(surf, title, (rect.x + px(8), rect.y + px(5)), theme.ACCENT, fonts.small)
    if right:
        fonts.draw(surf, right, (rect.right - px(8), rect.y + px(5)), theme.DIM, fonts.small,
                   "topright")


def _label(surf, fonts, text, pos, color, anchor="center", bounds=None, placed=None):
    """Small label on a dark backing, so it reads over lines and the Earth; kept
    inside ``bounds`` and, given the list of labels ``placed`` so far, moved up or
    down clear of them (then added to it)."""
    txt = fonts.render(text, color, fonts.small)
    r = txt.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
    if bounds is not None:
        r.clamp_ip(bounds)
    if placed is not None:
        step = r.h + px(2)
        for dy in (0, -step, step, -2 * step, 2 * step):
            cand = r.move(0, dy)
            if bounds is not None:
                cand.clamp_ip(bounds)
            if cand.inflate(px(6), 0).collidelist(placed) < 0:
                r = cand
                break
        placed.append(r)
    theme.panel(surf, r.inflate(px(6), 0), LABEL_BACK, None, 3)
    surf.blit(txt, r)
    tips.add(r.inflate(px(6), px(2)), glossary.label(text), clip=surf.get_clip())
    return r


# --- diagrams --------------------------------------------------------------------------------

def draw_orbit_plane(surf, rect, info: OrbitInfo, color, r_sun, fonts):
    """The orbital plane face-on (perigee to the right, motion counter-clockwise)."""
    el = info.el
    _box(surf, rect, fonts, "ORBITAL PLANE  (face-on)", f"e {el.e:.4f}")
    P, Q = perifocal_axes(el.i, el.raan, el.argp)
    e, p = float(el.e), float(el.p)
    if info.closed:
        nu = np.linspace(0.0, 2 * np.pi, 241)
    else:
        rcap = max(4.0 * el.rp, 1.5 * info.radius)
        lim = min(math.acos(np.clip((p / rcap - 1.0) / e, -1, 1)), math.acos(-1.0 / e) - 1e-3)
        nu = np.linspace(-lim, lim, 241)
    rr = p / (1.0 + e * np.cos(nu))
    x, y = rr * np.cos(nu), rr * np.sin(nu)

    area = pygame.Rect(rect.x + px(10), rect.y + px(24), rect.w - px(20), rect.h - px(34))
    xmin, xmax = min(x.min(), -R_EARTH), max(x.max(), R_EARTH)
    ymin, ymax = min(y.min(), -R_EARTH), max(y.max(), R_EARTH)
    scale = min(area.w / (xmax - xmin), area.h / (ymax - ymin)) * 0.9
    cx = area.centerx - scale * (xmin + xmax) / 2
    cy = area.centery + scale * (ymin + ymax) / 2

    def S(u, v):
        return (cx + scale * u, cy - scale * v)

    clip = surf.get_clip()
    surf.set_clip(rect.inflate(-px(2), -px(2)).clip(clip))
    sun = _sun_in_plane(r_sun, P, Q)
    u_hat, _, _, spn = sun
    if spn > 0.02:
        _draw_plane_shadow(surf, rect, S, sun, (xmax - xmin + ymax - ymin) * 2)

    # orbit interior, then the Earth with its lit half toward the Sun
    outline = [S(a, b) for a, b in zip(x, y, strict=True)]
    _alpha_poly(surf, color, 38, outline + ([S(0, 0)] if not info.closed else []), rect)
    ecen = S(0, 0)
    _draw_plane_earth(surf, S, max(px(3), int(R_EARTH * scale)), sun)

    # orbit colored by illumination
    pos3 = x[:, None] * P + y[:, None] * Q
    lit = shadow_fraction(pos3, r_sun)
    pts = np.stack([cx + scale * x, cy - scale * y], 1)
    dark = theme.dim(color, 0.35)
    for k in range(len(pts) - 1):
        c = color if lit[k] > 0.5 else dark
        pygame.draw.line(surf, c, pts[k], pts[k + 1], px(2))

    labels = _plane_apsides_nodes(surf, info, S)
    if spn > 0.1:                               # the Sun's direction, at the edge
        far = np.array(ecen) + np.array([u_hat[0], -u_hat[1]]) * 1e4
        edge = _ray_to_rect(ecen, far, area)
        pygame.draw.circle(surf, SUNLIT, edge, px(5))
        labels.insert(0, ("Sun", (edge[0], edge[1] - px(12)), SUNLIT, "center"))
    sat = _draw_plane_satellite(surf, S, el, color)
    # labels last, so no line crosses them, and never over the satellite
    placed = [pygame.Rect(0, 0, px(18), px(18)).move(sat[0] - px(9), sat[1] - px(9))]
    for text, pos, col, anchor in labels:
        _label(surf, fonts, text, pos, col, anchor, rect.inflate(-px(6), -px(6)), placed)
    surf.set_clip(clip)
    _legend(surf, fonts, rect, color, dark)


def _sun_in_plane(r_sun, P, Q):
    """The Sun seen in the orbit plane (perifocal axes ``P``, ``Q``):
    ``(u_hat, w_hat, sin_b, spn)``, the 2-D unit vector toward the Sun's projection
    (``P`` when the Sun is over the orbit's pole), the one 90 deg on from it, the
    sine of the beta angle and the projection's length."""
    s_hat = np.asarray(r_sun) / np.linalg.norm(r_sun)
    sp = np.array([s_hat @ P, s_hat @ Q])
    sin_b = float(s_hat @ np.cross(P, Q))
    spn = float(np.linalg.norm(sp))
    u_hat = sp / spn if spn > 1e-9 else np.array([1.0, 0.0])
    return u_hat, np.array([-u_hat[1], u_hat[0]]), sin_b, spn


def _draw_plane_shadow(surf, rect, S, sun, span):
    """The Earth's (cylindrical) shadow where it cuts the plane: a half-ellipse of
    semi-axes R/|sin beta| (anti-Sun, at most ``span``) and R."""
    u_hat, w_hat, sin_b, _ = sun
    A = min(R_EARTH / max(abs(sin_b), 1e-9), span)
    t = np.linspace(-np.pi / 2, np.pi / 2, 60)
    uu, ww = -A * np.cos(t), R_EARTH * np.sin(t)
    pts = [S(*(a * u_hat + b * w_hat)) for a, b in zip(uu, ww, strict=True)]
    _alpha_poly(surf, (0, 0, 12), 150, pts, rect)


def _draw_plane_earth(surf, S, er, sun):
    """The Earth, ``er`` px in radius, with the half facing the Sun lit (all of it
    when the Sun is above the orbit's north pole)."""
    u_hat, w_hat, sin_b, spn = sun
    ecen = S(0, 0)
    pygame.draw.circle(surf, EARTH_NIGHT, ecen, er)
    if spn > 0.02:
        t = np.linspace(-np.pi / 2, np.pi / 2, 40)
        day = [S(*(R_EARTH * (math.cos(a) * u_hat + math.sin(a) * w_hat))) for a in t]
        term = [S(*(-R_EARTH * sin_b * math.cos(a) * u_hat + R_EARTH * math.sin(a) * w_hat))
                for a in t[::-1]]
        pygame.draw.polygon(surf, EARTH_DAY, day + term)
    elif sin_b > 0:
        pygame.draw.circle(surf, EARTH_DAY, ecen, er)
    pygame.draw.circle(surf, EARTH_EDGE, ecen, er, 1)


def _draw_plane_satellite(surf, S, el, color):
    """The satellite, its radius vector and its velocity arrow; returns its screen point."""
    e, p, nu0 = float(el.e), float(el.p), float(el.nu)
    r0 = p / (1.0 + e * math.cos(nu0))
    sat = S(r0 * math.cos(nu0), r0 * math.sin(nu0))
    pygame.draw.line(surf, theme.dim(color, 0.7), S(0, 0), sat)
    vx, vy = -math.sin(nu0), e + math.cos(nu0)
    vn = math.hypot(vx, vy) or 1.0
    arm = px(26)
    _arrow(surf, (255, 255, 255), sat, (sat[0] + arm * vx / vn, sat[1] - arm * vy / vn), 6, 2)
    pygame.draw.circle(surf, color, sat, px(5))
    pygame.draw.circle(surf, (255, 255, 255), sat, px(8), 1)
    return sat


def _plane_apsides_nodes(surf, info: OrbitInfo, S):
    """Orbit-plane diagram: the line of apsides (with the ellipse center) and the
    line of nodes; returns the labels of their end points as (text, pos, color,
    anchor). ``S(x, y)`` maps perifocal km to screen."""
    el = info.el
    e, p = float(el.e), float(el.p)
    ecen, pe = S(0, 0), S(el.rp, 0)
    ap = S(-el.ra, 0) if info.closed else None
    if info.closed:
        _dashed(surf, theme.dim(APSIS, 0.6), ap, pe)
        c0 = S(-info.c, 0)
        k = px(3)
        pygame.draw.line(surf, APSIS, (c0[0] - k, c0[1]), (c0[0] + k, c0[1]))
        pygame.draw.line(surf, APSIS, (c0[0], c0[1] - k), (c0[0], c0[1] + k))
    else:
        _dashed(surf, theme.dim(APSIS, 0.6), ecen, pe)
    labels = []
    if not info.circular:
        pygame.draw.circle(surf, APSIS, pe, px(3))
        labels.append((f"Pe {num(info.rp_alt, 'km', 0)}", (pe[0], pe[1] + px(7)), APSIS,
                       "midtop"))
        if ap is not None:
            pygame.draw.circle(surf, APSIS, ap, px(3))
            labels.append((f"Ap {num(info.ra_alt, 'km', 0)}", (ap[0], ap[1] + px(7)), APSIS,
                           "midtop"))
    if not info.equatorial:
        ends = []
        for nu_n in (-el.argp, math.pi - el.argp):          # ascending, descending node
            den = 1.0 + e * math.cos(nu_n)
            ends.append(None if den <= 1e-6           # beyond an open orbit's asymptote
                        else (p / den * math.cos(nu_n), p / den * math.sin(nu_n)))
        an, dn = ends
        a_s = S(*an) if an else ecen
        d_s = S(*dn) if dn else ecen
        _dashed(surf, theme.dim(NODE, 0.8), a_s, d_s)
        if an:
            pygame.draw.circle(surf, NODE, a_s, px(4))
            labels.append(("AN", (a_s[0], a_s[1] - px(7)), NODE, "midbottom"))
        if dn:
            pygame.draw.circle(surf, NODE, d_s, px(4), 1)
            labels.append(("DN", (d_s[0], d_s[1] - px(7)), NODE, "midbottom"))
    return labels


def _legend(surf, fonts, rect, color, dark):
    """Key to the orbit-plane diagram's line styles."""
    y = rect.bottom - px(13)
    x = rect.x + px(8)
    ly, sw = y + px(6), px(14)                  # the sample line's height and length
    tips.add(pygame.Rect(rect.x, y - px(2), rect.w, px(15)),
             "Key: the orbit in sunlight and in shadow; the dashed line of apsides joins "
             "perigee and apogee, the line of nodes the two equator crossings.",
             clip=surf.get_clip())
    for c, text, dashed in ((color, "sunlit", False), (dark, "shadow", False),
                            (APSIS, "apsides", True), (NODE, "nodes", True)):
        if dashed:
            _dashed(surf, c, (x, ly), (x + sw, ly), 3, 2)
        else:
            pygame.draw.line(surf, c, (x, ly), (x + sw, ly), px(2))
        x = fonts.draw(surf, text, (x + px(18), y), theme.FAINT, fonts.small).right + px(10)


def _ray_to_rect(a, b, rect: pygame.Rect):
    """Point where the ray a->b leaves ``rect`` (a inside)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    ts = []
    if dx:
        ts += [(rect.left + px(6) - ax) / dx, (rect.right - px(6) - ax) / dx]
    if dy:
        ts += [(rect.top + px(14) - ay) / dy, (rect.bottom - px(18) - ay) / dy]
    t = min(t for t in ts if t > 0) if any(t > 0 for t in ts) else 0.0
    return (ax + dx * t, ay + dy * t)


def draw_inclination(surf, rect, info: OrbitInfo, color, fonts):
    """Edge-on view along the line of nodes: equator vs orbit plane."""
    el = info.el
    _box(surf, rect, fonts, "INCLINATION")
    c = (rect.centerx, rect.centery)
    L = rect.w * 0.42
    er = px(15)
    pygame.draw.circle(surf, EARTH_NIGHT, c, er)
    pygame.draw.circle(surf, EARTH_EDGE, c, er, 1)
    _dashed(surf, EQUATOR, (c[0] - L, c[1]), (c[0] + L, c[1]))
    _arrow(surf, theme.DIM, (c[0], c[1] - er), (c[0], c[1] - er - px(16)), 4)
    i = float(el.i)
    d = (math.cos(i), math.sin(i))
    pygame.draw.line(surf, color, (c[0] - L * d[0], c[1] + L * d[1]),
                     (c[0] + L * d[0], c[1] - L * d[1]), px(2))
    hx, hy = -d[1], d[0]
    hl = px(40)
    _arrow(surf, H_VEC, (c[0] + er * hx, c[1] - er * hy), (c[0] + hl * hx, c[1] - hl * hy), 5, 2)
    kind = ("equatorial" if info.equatorial else "polar" if abs(math.degrees(i) - 90) < 2
            else "retrograde" if info.retrograde else "prograde")
    labels = []
    if i > 1e-3:
        _arc(surf, NODE, c, px(30), 0.0, i, 2)
        mid, lr = i / 2, px(36)
        labels.append((deg(i), (c[0] + lr * math.cos(mid), c[1] - lr * math.sin(mid)), NODE,
                       "midleft" if math.cos(mid) >= 0 else "midright"))
    labels += [("h", (c[0] + px(50) * hx, c[1] - px(50) * hy), H_VEC, "center"),
               ("N", (c[0] + px(5), c[1] - er - px(13)), theme.DIM, "midleft"),
               ("equator", (c[0] + L, c[1] + px(4)), EQUATOR, "topright"),
               (kind, (rect.centerx, rect.bottom - px(5)), theme.DIM, "midbottom")]
    placed = [pygame.Rect(rect.x, rect.y, rect.w, px(22))]         # the title
    for text, pos, col, anchor in labels:
        _label(surf, fonts, text, pos, col, anchor, rect.inflate(-px(6), -px(6)), placed)


def draw_north_view(surf, rect, info: OrbitInfo, color, r_sun, fonts):
    """The orbit projected on the equator, seen from above the north pole."""
    el = info.el
    _box(surf, rect, fonts, "FROM NORTH")
    c = (rect.centerx, rect.centery)
    half = min(rect.w, rect.h - px(44)) * 0.5 - px(6)
    P, Q = perifocal_axes(el.i, el.raan, el.argp)
    e, p = float(el.e), float(el.p)
    if info.closed:
        nu = np.linspace(0, 2 * np.pi, 181)
    else:
        lim = math.acos(-1.0 / e) - 0.05
        nu = np.linspace(-lim, lim, 181)
    rr = p / (1.0 + e * np.cos(nu))
    pts = (rr * np.cos(nu))[:, None] * P + (rr * np.sin(nu))[:, None] * Q
    ext = max(float(np.max(np.abs(pts[:, :2]))), R_EARTH * 1.2)
    if not info.closed:
        ext = min(ext, 3 * info.radius)
    k = half / ext
    er = max(px(4), int(R_EARTH * k))
    clip = surf.get_clip()
    surf.set_clip(rect.inflate(-px(2), -px(2)).clip(clip))
    s = np.asarray(r_sun) / np.linalg.norm(r_sun)
    sa = math.atan2(s[1], s[0])
    pygame.draw.circle(surf, EARTH_NIGHT, c, er)
    tt = np.linspace(sa - np.pi / 2, sa + np.pi / 2, 30)
    pygame.draw.polygon(surf, EARTH_DAY,
                        [(c[0] + er * math.cos(a), c[1] - er * math.sin(a)) for a in tt])
    pygame.draw.circle(surf, EARTH_EDGE, c, er, 1)
    scr = np.stack([c[0] + k * pts[:, 0], c[1] - k * pts[:, 1]], 1)
    pygame.draw.lines(surf, color, False, scr.tolist(), px(2))
    # vernal equinox and Sun directions, line of nodes and RAAN
    _arrow(surf, theme.AXIS_X, c, (c[0] + half, c[1]), 5)
    sun_end = (c[0] + half * math.cos(sa), c[1] - half * math.sin(sa))
    _dashed(surf, SUNLIT, (c[0] + er * math.cos(sa), c[1] - er * math.sin(sa)), sun_end, 4, 3)
    labels = [("Sun", (sun_end[0], sun_end[1] - px(7)), SUNLIT, "midbottom")]
    if not info.equatorial:
        n = (math.cos(el.raan), math.sin(el.raan))
        a_s = (c[0] + half * 0.92 * n[0], c[1] - half * 0.92 * n[1])
        d_s = (c[0] - half * 0.92 * n[0], c[1] + half * 0.92 * n[1])
        _dashed(surf, NODE, d_s, a_s)
        if el.raan > 1e-3:
            _arc(surf, NODE, c, half * 0.45, 0.0, float(el.raan), 2)
        pygame.draw.circle(surf, NODE, a_s, px(4))
        pygame.draw.circle(surf, NODE, d_s, px(4), 1)
        labels.append(("AN", (a_s[0], a_s[1] - px(7)), NODE, "midbottom"))
    labels.append(("\N{GREEK SMALL LETTER GAMMA}", (c[0] + half, c[1] + px(5)), theme.AXIS_X,
                   "topright"))
    r0 = p / (1.0 + e * math.cos(el.nu))
    now = r0 * (math.cos(el.nu) * P + math.sin(el.nu) * Q)
    sat = (c[0] + k * now[0], c[1] - k * now[1])
    pygame.draw.circle(surf, color, sat, px(4))
    pygame.draw.circle(surf, (255, 255, 255), sat, px(7), 1)
    placed = [pygame.Rect(rect.x, rect.y, rect.w, px(22)),
              pygame.Rect(sat[0] - px(8), sat[1] - px(8), px(16), px(16))]
    for text, pos, col, anchor in labels:
        _label(surf, fonts, text, pos, col, anchor, rect.inflate(-px(6), -px(26)), placed)
    surf.set_clip(clip)
    _label(surf, fonts, f"\N{GREEK CAPITAL LETTER OMEGA} {deg(el.raan, 1)}   "
           f"LTAN {hours_clock(info.ltan)}", (rect.centerx, rect.bottom - px(5)), theme.DIM,
           "midbottom")


def draw_timeline(surf, rect, info: OrbitInfo, fonts):
    """Sunlight and events over the next revolution, now at the left."""
    if not info.closed:
        _box(surf, rect, fonts, "NEXT REVOLUTION")
        fonts.draw(surf, "open trajectory - no further revolutions",
                   (rect.x + px(8), rect.y + px(26)), theme.FAINT, fonts.small)
        return
    ecl = info.eclipse_fraction
    right = ("no eclipse" if ecl < 1e-4
             else f"shadow {ecl * 100:.0f}% = {countdown(info.eclipse_duration)}")
    _box(surf, rect, fonts, f"NEXT {countdown(info.period)}", right)
    bar = pygame.Rect(rect.x + px(10), rect.y + px(24), rect.w - px(20), px(12))
    lit = np.interp(np.linspace(0, 1, bar.w), info.timeline_t / info.period, info.timeline_lit)
    cols = np.asarray(SHADOW, float) * (1 - lit[:, None]) + np.asarray(SUNLIT, float) * lit[:, None]
    strip = pygame.Surface((bar.w, 1))
    pygame.surfarray.blit_array(strip, cols.astype(np.uint8)[:, None, :])
    surf.blit(pygame.transform.scale(strip, bar.size), bar.topleft)
    pygame.draw.rect(surf, theme.PANEL_EDGE, bar, 1)
    events = [(info.t_peri, "Pe", APSIS), (info.t_apo, "Ap", APSIS),
              (info.t_an, "AN", NODE), (info.t_dn, "DN", NODE)]
    events = sorted((t, n, c) for t, n, c in events if math.isfinite(t) and t <= info.period)
    xs = [bar.x + int(t / info.period * (bar.w - 1)) for t, _, _ in events]
    for x, (_, _, col) in zip(xs, events, strict=True):
        pygame.draw.line(surf, col, (x, bar.y - px(3)), (x, bar.bottom + px(2)), px(2))
    # each label takes the first of three rows under the bar where it fits
    placed = []
    area = rect.inflate(-px(6), -px(4))
    for x, (t, name, col) in zip(xs, events, strict=True):
        text = f"{name} {countdown(t)}"
        w = fonts.small.size(text)[0]
        for row in range(3):
            r = pygame.Rect(0, 0, w, fonts.small.get_height())
            r.midtop = (x, bar.bottom + px(4) + (r.h + 1) * row)
            r.clamp_ip(area)
            if row == 2 or r.inflate(px(8), 0).collidelist(placed) < 0:
                break
        placed.append(r)
        theme.panel(surf, r.inflate(px(6), 0), LABEL_BACK, None, 3)
        fonts.draw(surf, text, r.topleft, col, fonts.small)
        tips.add(r.inflate(px(6), px(2)), f"{glossary.label(name)} Next pass in {countdown(t)}.",
                 clip=surf.get_clip())


def draw_gauge(surf, rect, fonts, label, lo, hi, val, unit, prec, color):
    """Horizontal bar from ``lo`` to ``hi`` with a marker at ``val``."""
    what = label.split(" (")[0].lower()
    tips.add(rect, f"Where the satellite's {what} is now (the dot) between its lowest and "
                   f"highest values on this orbit (the numbers at the ends).",
             clip=surf.get_clip())
    fonts.draw(surf, label, (rect.x, rect.y), theme.DIM, fonts.small)
    fonts.draw(surf, f"{val:,.{prec}f} {unit}", (rect.right, rect.y), theme.TEXT, fonts.small,
               "topright")
    bar = pygame.Rect(rect.x, rect.y + px(17), rect.w, px(6))
    pygame.draw.rect(surf, (30, 40, 64), bar, border_radius=px(3))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo < 1e-9 * max(1.0, abs(hi)):
        f = 0.5
        fonts.draw(surf, "constant", (bar.centerx, bar.bottom + 1), theme.FAINT, fonts.small,
                   "midtop")
    else:
        f = min(1.0, max(0.0, (val - lo) / (hi - lo)))
        fonts.draw(surf, f"{lo:,.{prec}f}", (bar.x, bar.bottom + 1), theme.FAINT, fonts.small)
        fonts.draw(surf, f"{hi:,.{prec}f}", (bar.right, bar.bottom + 1), theme.FAINT, fonts.small,
                   "topright")
    fill = bar.copy()
    fill.w = max(px(6), int(bar.w * f))
    pygame.draw.rect(surf, theme.dim(color, 0.6), fill, border_radius=px(3))
    x = bar.x + int(bar.w * f)
    pygame.draw.circle(surf, (255, 255, 255), (x, bar.centery), px(5))
    pygame.draw.circle(surf, color, (x, bar.centery), px(3))


# --- property sheet ----------------------------------------------------------------------------

def draw_section(surf, fonts, x, y, w, title, rows) -> int:
    """A titled block of (label, value) rows, values right-aligned at ``x + w``;
    returns the y just below it."""
    clip = surf.get_clip()                      # the panel's scrolled body
    about = glossary.section(title)
    tips.add(fonts.draw(surf, title, (x, y), theme.ACCENT, fonts.small), about, clip=clip)
    y += px(18)
    for label, value in rows:
        fonts.draw(surf, label, (x + px(4), y), theme.DIM, fonts.small)
        fonts.draw(surf, value, (x + w, y), theme.TEXT, fonts.small, "topright")
        tips.add(pygame.Rect(x, y, w, px(16)), glossary.row(label) or about, clip=clip)
        y += px(16)
    return y + px(6)


def property_sections(info: OrbitInfo, dv_used: float):
    """The Orbit tab's property sheet: [(section title, [(label, value)])]."""
    el = info.el
    D = math.degrees
    closed = info.closed
    shape = [
        ("Regime", info.regime),
        ("Semi-major axis a", num(el.a, "km", 1) if closed else f"{el.a:,.1f} km (open)"),
        ("Mean semi-major axis", num(info.a_mean, "km", 1)),
        ("Semi-minor axis b", num(info.b, "km", 1)),
        ("Eccentricity e", f"{el.e:.6f}"),
        ("Semi-latus rectum p", num(el.p, "km", 1)),
        ("Perigee alt / radius", f"{num(info.rp_alt, '', 1)} / {num(el.rp, 'km', 0)}"),
        ("Apogee alt / radius",
         f"{num(info.ra_alt, '', 1)} / {num(el.ra, 'km', 0)}" if closed else "-"),
        ("Focus offset c = ae", num(info.c, "km", 1)),
    ]
    orient = [
        ("Inclination i", deg(el.i, 4)),
        ("RAAN \N{GREEK CAPITAL LETTER OMEGA}", "-" if info.equatorial else deg(el.raan, 4)),
        ("Arg. of perigee \N{GREEK SMALL LETTER OMEGA}", "-" if info.circular else deg(el.argp, 3)),
        ("True anomaly \N{GREEK SMALL LETTER NU}", deg(el.nu, 3)),
        ("Mean anomaly M", deg(el.M, 3) if closed else "-"),
        ("Arg. of latitude u", deg(el.u, 3)),
        ("Flight-path angle", deg(info.fpa, 3)),
        ("Local time of AN", hours_clock(info.ltan)),
    ]
    timing = [
        ("Period", countdown(info.period) if closed else "open"),
        ("Nodal period (J2)", countdown(info.nodal_period)),
        ("Revolutions per day", num(info.revs_per_day, "", 4)),
        ("Track shift per rev", "-" if not math.isfinite(info.ground_shift) else
         f"{abs(info.ground_shift):.2f} deg {'W' if info.ground_shift >= 0 else 'E'}"),
        ("Next perigee in", countdown(info.t_peri)),
        ("Next apogee in", countdown(info.t_apo)),
        ("Next AN / DN in", f"{countdown(info.t_an)} / {countdown(info.t_dn)}"),
    ]
    speeds = [
        ("Speed now", num(info.speed, "km/s", 4)),
        ("Speed at perigee", num(info.v_peri, "km/s", 4)),
        ("Speed at apogee", num(info.v_apo, "km/s", 4)),
        ("Circular speed here", num(info.v_circ, "km/s", 4)),
        ("Escape speed here", num(info.v_esc, "km/s", 4)),
        ("C3 = 2 x energy", num(info.c3, "km^2/s^2", 3)),
        ("Angular momentum h", num(el.h, "km^2/s", 0)),
    ]
    sso = info.sso_inclination
    sso_txt = deg(sso, 2) if math.isfinite(sso) else "none"
    if math.isfinite(sso) and abs(sso - el.i) < math.radians(0.05):
        sso_txt += "  (SSO)"
    pert = [
        ("J2 node drift", num(D(info.raan_dot) * 86400, "deg/day", 4)),
        ("J2 perigee drift", num(D(info.argp_dot) * 86400, "deg/day", 4)),
        ("Sun-sync inclination", sso_txt),
        ("Beta angle", deg(info.beta, 2)),
        ("Time in shadow / rev", f"{info.eclipse_fraction * 100:.1f}%" if closed else "-"),
    ]
    craft = [
        ("Mass", num(info.mass, "kg", 1)),
        ("Area / Cd", f"{info.area:g} m^2 / {info.cd:g}"),
        ("Area-to-mass", num(info.area_to_mass, "m^2/kg", 4)),
        ("Ballistic coeff m/CdA", num(info.ballistic, "kg/m^2", 1)),
        ("Air density at perigee",
         f"{info.rho_perigee:.3e} kg/m^3" if info.rho_perigee else "none"),
        ("Drag at perigee", f"{info.drag_perigee:.3e} m/s^2" if info.rho_perigee else "none"),
        ("Delta-v spent", num(dv_used * 1000, "m/s", 2)),
        ("Horizon distance", num(info.horizon_km, "km", 0)),
        ("Coverage radius (10\N{DEGREE SIGN})", num(info.footprint_km, "km", 0)),
    ]
    return [("SHAPE", shape), ("ORIENTATION", orient), ("TIMING", timing),
            ("SPEED AND ENERGY", speeds), ("PERTURBATIONS AND LIGHT", pert), ("SPACECRAFT", craft)]


def draw_orbit_tab(surf, x, y, w, app, i) -> int:
    """Draw the tab from (x, y); returns the y just below the content."""
    sim, fonts = app.sim, app.fonts
    info = app.orbit_info()
    s = sim.sats[i]
    if info is None:
        fonts.draw(surf, f"{s.name} is {s.status}", (x, y), theme.FAINT, fonts.ui)
        return y + px(24)
    r_sun = sun_position(sim.jd())
    gap = px(6)
    draw_orbit_plane(surf, pygame.Rect(x, y, w, px(236)), info, s.color, r_sun, fonts)
    y += px(242)
    half = (w - gap) // 2
    draw_inclination(surf, pygame.Rect(x, y, half, px(172)), info, s.color, fonts)
    draw_north_view(surf, pygame.Rect(x + half + gap, y, w - half - gap, px(172)), info, s.color,
                    r_sun, fonts)
    y += px(178)
    draw_timeline(surf, pygame.Rect(x, y, w, px(96)), info, fonts)
    y += px(104)
    if info.closed and not info.circular:
        gauge = pygame.Rect(x + px(4), y, w - px(8), px(36))
        draw_gauge(surf, gauge, fonts, "Altitude (perigee -> apogee)",
                   info.rp_alt, info.ra_alt, info.radius - R_EARTH, "km", 0, s.color)
        y += px(46)
        draw_gauge(surf, gauge.move(0, px(46)), fonts, "Speed (apogee -> perigee)",
                   info.v_apo, info.v_peri, info.speed, "km/s", 3, s.color)
        y += px(50)
    for title, rows in property_sections(info, s.dv_used):
        y = draw_section(surf, fonts, x, y, w, title, rows)
    return y
