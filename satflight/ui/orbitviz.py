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
from . import theme
from .orbitpanel import APSIS, EQUATOR, H_VEC, NODE, countdown, deg, num
from .render3d import Line

MODES = ("full", "basic", "off")
ARC_I = (255, 150, 90)
ARC_RAAN = (120, 200, 255)
ARC_ARGP = (205, 150, 255)
R_FILL_IN = R_EARTH * 1.001
ROUND_E = 0.01          # below this, arcs use the argument of latitude
X_HAT, Y_HAT, Z_HAT = np.eye(3)
ORIGIN = np.zeros(3)


def _unit(v):
    return v / np.linalg.norm(v)


def _arc(center, u, w, radius, a0, a1, n=48):
    """Points of a circular arc in the plane of unit vectors ``u``, ``w`` from angle
    ``a0`` to ``a1`` (``n`` points per half turn)."""
    t = np.linspace(a0, a1, max(3, int(n * abs(a1 - a0) / math.pi) + 3))
    return center + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)


def _dashed(p0, p1, color, n=24, width=1):
    """A dashed straight line: one Line whose dashes are separated by NaN
    points (the renderer breaks polylines at non-finite points)."""
    t = np.linspace(0.0, 1.0, 2 * n)
    pts = p0 + t[:, None] * (p1 - p0)
    pts = np.insert(pts, np.arange(2, 2 * n, 2), np.nan, axis=0)
    return [Line(pts, color, width, 0.4)]


def _angle(lines, markers, world, center, u, w, radius, angle, color, label, label_r):
    """Add an angle: an arc of ``radius`` from direction ``u`` turning toward ``w``,
    and ``label`` at its mid-angle, ``label_r`` from ``center``."""
    mid = angle / 2
    lines.append(Line(world(_arc(center, u, w, radius, 0.0, angle)), color, 2, 0.4))
    at = center + label_r * (math.cos(mid) * u + math.sin(mid) * w)
    markers.append((world(at), color, 0, label, False))


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


def build(info: OrbitInfo, r, v, color, W, cam, mode: str):
    """Lines, label markers and fills (polygon_world, rgba) for one orbit."""
    lines, markers, fills = [], [], []
    if mode == "off":
        return lines, markers, fills
    el = info.el
    P, Q = perifocal_axes(el.i, el.raan, el.argp)
    Hn = np.cross(P, Q)
    e, p = float(el.e), float(el.p)
    side = cam.forward

    def world(x):
        return np.asarray(x) @ W.T

    # orbital plane
    if info.closed:
        nu = np.linspace(0.0, 2 * np.pi, 181)
    else:
        rcap = max(3.0e5, 1.5 * info.radius)
        lim = min(math.acos(np.clip((p / rcap - 1.0) / e, -1, 1)), math.acos(-1.0 / e) - 1e-3)
        nu = np.linspace(-lim, lim, 181)
    fills.append((world(sector_polygon(P, Q, p, e, nu)), (*color[:3], 26)))

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
        markers.append((world(pe), APSIS, -2, f"Pe {num(info.rp_alt, 'km', 0)}", False))
        if ap is not None:
            markers.append((world(ap), APSIS, -2, f"Ap {num(info.ra_alt, 'km', 0)}", False))

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
            markers.append((a_w, NODE, -2, "AN", False))
        if dn is not None:
            markers.append((d_w, NODE, -2, "DN", False))

    if mode != "full":
        return lines, markers, fills

    # equatorial plane out past the node crossings, and the reference directions
    node_r = [np.linalg.norm(x) for x in (an, dn) if x is not None]
    r_eq = min(1.12 * max(node_r + [info.radius, 1.6 * R_EARTH]), 3e5)
    if not info.equatorial:
        a = np.linspace(0, 2 * np.pi, 145)
        fills.append((world(sector_polygon(X_HAT, Y_HAT, r_eq, 0.0, a)), (*EQUATOR, 14)))
        lines.append(Line(world(_arc(ORIGIN, X_HAT, Y_HAT, r_eq, 0, 2 * np.pi, 96)),
                          theme.dim(EQUATOR, 0.8), 1, 0.4))

    base = max(R_EARTH, 0.35 * el.rp)
    rn, rw, rv = 1.80 * base, 1.55 * base, 1.30 * base     # arc radii: RAAN, argp, u / nu
    bright = theme.mix(color, (255, 255, 255), 0.35)
    ang = (lines, markers, world)
    # RAAN: in the equator from the vernal equinox (+X) to the ascending node
    if not info.equatorial:
        lines += _dashed(ORIGIN, world(X_HAT * rn * 1.12), ARC_RAAN, 10)
        markers.append((world(X_HAT * rn * 1.16), ARC_RAAN, 0, "\N{GREEK SMALL LETTER GAMMA}",
                        False))
        if el.raan > 1e-3:
            _angle(*ang, ORIGIN, X_HAT, Y_HAT, rn, el.raan, ARC_RAAN,
                   f"\N{GREEK CAPITAL LETTER OMEGA} {deg(el.raan, 1)}", rn * 1.08)
    # near-circular orbits have an ill-defined perigee: show the argument of
    # latitude (node -> satellite) instead of the perigee and true anomaly
    round_ = el.e < ROUND_E
    if round_ and not info.equatorial and el.u > 1e-3:
        _angle(*ang, ORIGIN, n_hat, v_hat, rv, el.u, bright, f"u {deg(el.u, 1)}", rv * 1.1)
    # argument of perigee: in the orbit plane from the node to the perigee
    if not info.equatorial and not round_ and el.argp > 1e-3:
        _angle(*ang, ORIGIN, n_hat, v_hat, rw, el.argp, ARC_ARGP,
               f"\N{GREEK SMALL LETTER OMEGA} {deg(el.argp, 1)}", rw * 1.08)
    # true anomaly: from the perigee (or node, if circular) to the satellite
    nu0 = float(el.nu) if info.closed else (float(el.nu) + np.pi) % (2 * np.pi) - np.pi
    if abs(nu0) > 1e-3 and not (round_ and not info.equatorial):
        _angle(*ang, ORIGIN, P, Q, rv, nu0, bright,
               f"\N{GREEK SMALL LETTER NU} {deg(el.nu, 1)}", rv * 1.1)
    # inclination at the ascending node, between the equator and the track
    if an is not None:
        east = _unit(np.cross(Z_HAT, n_hat))
        rho = max(0.45 * R_EARTH, 0.12 * np.linalg.norm(an))
        lines.append(Line(world(np.stack([an - 0.4 * rho * east, an + 1.5 * rho * east])),
                          EQUATOR, 1, 0.4))
        _angle(*ang, an, east, Z_HAT, rho, el.i, ARC_I, f"i {deg(el.i, 2)}", 1.15 * rho)
    # angular momentum, radius and velocity vectors
    h_len = 2.1 * base
    lines += _arrow(ORIGIN, world(Hn * h_len), H_VEC, side)
    markers.append((world(Hn * h_len * 1.06), H_VEC, 0, "h", False))
    lines.append(Line(np.linspace(ORIGIN, world(r), 24), theme.dim(color, 0.8), 1, 0.3))
    lines += _arrow(world(r), world(r + _unit(v) * 0.5 * base), (255, 255, 255), side)
    return lines, markers, fills


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
    w = max(fonts.small.size(r)[0] for r in rows) + 20
    w = max(w, fonts.bold.size(sat.name)[0] + 30)
    h = 30 + 16 * len(rows)
    x, y = int(pos[0]) + 26, int(pos[1]) - h - 18
    rect = pygame.Rect(x, y, w, h)
    if rect.right > bounds.right:
        rect.right = int(pos[0]) - 26
    if rect.top < bounds.top:
        rect.top = int(pos[1]) + 18
    rect.clamp_ip(bounds)
    corner = (rect.left if rect.centerx > pos[0] else rect.right,
              rect.bottom if rect.centery < pos[1] else rect.top)
    pygame.draw.aaline(surf, theme.dim(sat.color, 0.9), pos, corner)
    theme.panel(surf, rect, (10, 16, 30, 225), theme.dim(sat.color, 0.8), 6)
    pygame.draw.circle(surf, sat.color, (rect.x + 12, rect.y + 13), 5)
    fonts.draw(surf, sat.name, (rect.x + 22, rect.y + 4), theme.TEXT, fonts.bold)
    yy = rect.y + 26
    for r_ in rows:
        fonts.draw(surf, r_, (rect.x + 10, yy), theme.DIM, fonts.small)
        yy += 16
    return rect
