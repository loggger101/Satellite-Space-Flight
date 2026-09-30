"""3-D scene renderer: stars, Sun, Earth, axes, orbits, trails,
satellites, ground stations, rocket exhaust and annotations.

Occlusion by the Earth is handled with two passes: every line segment or
marker whose sight line is blocked by the Earth sphere is drawn *before*
the ray-cast Earth (which then paints over exactly the covered part);
everything else is drawn after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from ..analysis import coverage_half_angle, footprint
from ..constants import R_EARTH, R_GEO
from ..eclipse import shadow_fraction
from ..elements import coe2rv, conic_points, rv2coe
from ..ephemeris import sun_position
from ..frames import eci_to_ecef, rot3
from ..simulation import ACTIVE
from ..timeutil import format_duration
from . import theme
from .earth import EarthRenderer

CLIP = 30000.0      # px: screen coordinates are clamped here before pygame draws them


@dataclass
class Line:
    """A world-space polyline; the part hidden by the Earth is drawn dimmed by
    ``behind_dim`` (0 hides it)."""

    pts: np.ndarray          # (M, 3) world km
    color: tuple
    width: int = 1
    behind_dim: float = 0.55


class SceneRenderer:
    """Draws the 3-D view each frame; see :meth:`draw` for the pass order."""

    def __init__(self, asset_dir):
        self.earth = EarthRenderer(asset_dir)
        self._build_sky()
        self._sprites: dict = {}
        self._sat_lit = np.zeros(0)
        self.label_rects: list[pygame.Rect] = []    # where this frame's 3-D labels went
        self._sat_start = 0         # index of the first satellite marker
        self._overlay = range(0)    # markers of the selected orbit's geometry overlay
        self.legend_rect = None     # where the overlay's key went (clicking it folds it)
        self.sat_screen = None      # (sx, sy, visible) of last frame, for picking

    def _build_sky(self):
        """Random field stars with stellar colours plus a Milky Way band laid
        along the true galactic plane (J2000 galactic pole and centre)."""
        rng = np.random.default_rng(7)
        v = rng.normal(size=(2400, 3))
        dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        mag = rng.random(2400) ** 3.2
        palette = np.array([[165, 190, 255], [215, 225, 255], [255, 255, 255],
                            [255, 244, 214], [255, 214, 160], [255, 180, 140]], np.float32)
        tint = palette[rng.choice(len(palette), 2400, p=[0.12, 0.2, 0.3, 0.2, 0.12, 0.06])]
        bright = 0.18 + 0.82 * mag
        star_col = np.clip(tint * bright[:, None], 0, 255).astype(np.uint8)

        def radec(ra, dec):
            ra, dec = np.radians(ra), np.radians(dec)
            return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        zg = radec(192.85948, 27.12825)              # north galactic pole
        xg = radec(266.40499, -28.93617)             # galactic centre
        xg = xg - zg * (xg @ zg)
        xg /= np.linalg.norm(xg)
        yg = np.cross(zg, xg)
        n = 16000
        lon = np.where(rng.random(n) < 0.45, rng.normal(0.0, 0.6, n),
                       rng.uniform(-np.pi, np.pi, n))
        sig = np.radians(3.5 + 7.0 * np.exp(-(lon / 0.35) ** 2))
        lat = rng.normal(0.0, 1.0, n) * sig
        mw = ((np.cos(lat) * np.cos(lon))[:, None] * xg + (np.cos(lat) * np.sin(lon))[:, None] * yg
              + np.sin(lat)[:, None] * zg)
        glow = (10 + 34 * rng.random(n) ** 2) * (0.6 + 0.4 * np.exp(-(lon / 0.8) ** 2))
        mw_col = np.clip(np.array([0.82, 0.86, 1.0]) * glow[:, None], 0, 255).astype(np.uint8)
        self.sky_dirs = np.vstack([mw, dirs])
        self.sky_col = np.vstack([mw_col, star_col])
        self.sky_big = np.r_[np.zeros(n, bool), mag > 0.8]

    def _sprite(self, kind: str, color, radius: int):
        """Cached radial-glow sprite for additive blending."""
        key = (kind, tuple(color), radius)
        spr = self._sprites.get(key)
        if spr is None:
            size = radius * 2 + 1
            yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
            r = np.hypot(xx, yy).T.astype(np.float32)
            if kind == "sun":
                inten = (np.exp(-(r / (radius * 0.07)) ** 2) * 1.6
                         + 0.55 / (1 + (r / (radius * 0.12)) ** 2)
                         + 0.25 * np.exp(-np.abs(yy.T) / 1.2)
                         * np.exp(-np.abs(xx.T) / (radius * 0.5)))
            else:
                inten = 0.9 / (1 + (r / (radius * 0.28)) ** 2) * np.clip(1 - r / radius, 0, 1)
            arr = np.clip(inten[..., None] * np.asarray(color, np.float32)[None, None, :], 0, 255)
            spr = pygame.Surface((size, size))
            pygame.surfarray.blit_array(spr, arr.astype(np.uint8))
            if len(self._sprites) > 256:
                self._sprites.clear()
            self._sprites[key] = spr
        return spr

    # --- helpers -------------------------------------------------------------------------
    @staticmethod
    def world_rotation(sim, frame: str) -> np.ndarray:
        """Matrix W with world = W @ eci (identity in ECI view)."""
        if frame == "ECEF":
            return rot3(sim.gmst())
        return np.eye(3)

    def _draw_lines(self, surf, cam, lines: list[Line], behind: bool, cache):
        """Draw the hidden (``behind``) or visible parts of ``lines``; the projection is
        shared between both passes through ``cache``."""
        if not lines:
            return
        if cache.get("proj") is None:
            pts = np.concatenate([ln.pts for ln in lines], axis=0)
            sx, sy, z = cam.project(pts)
            hid = cam.hidden_by_sphere(pts)
            ok = np.isfinite(sx) & np.isfinite(sy)
            cache["proj"] = (np.clip(sx, -CLIP, CLIP), np.clip(sy, -CLIP, CLIP), ok, hid)
        sx, sy, ok, hid = cache["proj"]
        off = 0
        for ln in lines:
            m = len(ln.pts)
            s = slice(off, off + m)
            off += m
            h = hid[s]
            if behind:
                if not h.any():
                    continue
                near = h.copy()
                near[1:] |= h[:-1]
                near[:-1] |= h[1:]
                mask = ok[s] & near
                color = theme.dim(ln.color, ln.behind_dim)
            else:
                mask = ok[s] & ~h
                color = ln.color
            _runs(surf, color, sx[s], sy[s], mask, ln.width)

    # --- main entry -----------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, app):
        """Sky and Sun, then everything hidden by the Earth, the Earth itself, then
        everything in front of it; finally the selected satellite's callout.

        The scene is first collected as world-space ``lines`` and ``markers``
        (``(position, color, radius, label, is_selected)``; radius > 0 is a dot,
        < 0 a square, 0 a bare label). List order is draw order, and the
        satellites are always the last markers (picking relies on it).
        """
        sim, cam, opts = app.sim, app.camera, app.opts
        W = self.world_rotation(sim, opts.frame)
        world_to_ecef = rot3(sim.gmst()) @ W.T
        sun = sun_position(sim.jd())
        sun_dir = W @ (sun / np.linalg.norm(sun))
        sel = app.selected if 0 <= app.selected < sim.n else -1

        surf.fill(theme.BG)
        self.label_rects = []
        self._stars(surf, cam)
        self._sun(surf, cam, sun_dir, app)

        lines: list[Line] = []
        markers: list = []
        self._guides(opts, world_to_ecef, lines, markers)
        self._orbits(app, W, sel, lines)
        if opts.trails:
            lines.extend(self._trails(sim, app, W))
        if opts.stations and sim.stations:
            self._stations(sim, W, world_to_ecef, sel, lines, markers)
        from . import orbitviz  # imports Line from this module
        fills, legend = [], []
        self._overlay = range(0)
        info = app.orbit_info() if sel >= 0 else None
        if info is not None:
            lines.extend(self._selected_extras(sim, sel, W, world_to_ecef, opts, markers))
            ol, om, fills, legend = orbitviz.build(info, sim.y[sel, :3], sim.y[sel, 3:],
                                                   sim.sats[sel].color, W, cam, opts.geometry,
                                                   app.view_rect())
            lines.extend(ol)
            self._overlay = range(len(markers), len(markers) + len(om))
            markers.extend(om)
        self._satellites(sim, opts, W, sun, sel, lines, markers)
        key = self.legend_rect = orbitviz.legend_rect(app.fonts, legend, app.view_rect(),
                                                      opts.legend)
        if key is not None:
            self.label_rects.append(key)        # labels keep clear of the key

        # hidden pass, the Earth (orbit-plane fills split around it), visible pass
        cache: dict = {}
        mpos = self._project_markers(cam, markers)
        self._draw_lines(surf, cam, lines, True, cache)
        self._draw_markers(surf, app, markers, mpos, behind=True)
        for poly, rgba in fills:
            orbitviz.draw_fill(surf, cam, poly, rgba, "far")
        self.earth.render(surf, cam, world_to_ecef, sun_dir)
        for poly, rgba in fills:
            orbitviz.draw_fill(surf, cam, poly, rgba, "near")
        self._draw_lines(surf, cam, lines, False, cache)
        self._draw_markers(surf, app, markers, mpos, behind=False)
        self._exhaust(surf, sim, cam, W)

        sx, sy, vis = mpos
        n = sim.n
        self.sat_screen = (sx[-n:], sy[-n:], vis[-n:]) if n else None
        if info is not None and opts.geometry == "full" and self.sat_screen is not None:
            self._callout(surf, app, info, sel)
        if key is not None:
            orbitviz.draw_legend(surf, app.fonts, legend, key, opts.legend)

    # --- scene content ------------------------------------------------------------------------
    def _guides(self, opts, world_to_ecef, lines, markers):
        """Frame axes, GEO ring, equator and coastlines, as the options ask."""
        if opts.axes:
            if opts.frame == "ECI":
                names = ("X (vernal equinox)", "Y", "Z (north pole)")
            else:
                names = ("X (Greenwich)", "Y (90E)", "Z (north pole)")
            for k, col in enumerate((theme.AXIS_X, theme.AXIS_Y, theme.AXIS_Z)):
                end = np.zeros(3)
                end[k] = 2.3 * R_EARTH
                lines.append(Line(np.linspace(np.zeros(3), end, 24), col, 2, 0.5))
                markers.append((end, col, 0, names[k], False))
        a = np.linspace(0, 2 * np.pi, 181)
        circle = np.stack([np.cos(a), np.sin(a), 0 * a], 1)
        if opts.geo_ring:
            lines.append(Line(circle * R_GEO, (60, 70, 100), 1, 0.7))
        if opts.equator:
            lines.append(Line(circle * R_EARTH * 1.002, (70, 120, 170), 1, 0.0))
        if self.earth.coastlines and opts.coastlines:
            for cl in self.earth.coastlines:     # ECEF row vectors: world = p @ world_to_ecef
                lines.append(Line(cl @ world_to_ecef * (R_EARTH * 1.002), (150, 200, 150), 1, 0.0))

    def _orbits(self, app, W, sel, lines):
        """Osculating conics of the satellites picked by :meth:`_orbit_indices`;
        the selected one bright and thicker."""
        sim = app.sim
        idx = np.array(self._orbit_indices(app), int)
        if not idx.size:
            return
        r_far = float(np.max(np.linalg.norm(sim.y[idx, :3], axis=1)))
        pts = conic_points(sim.y[idx, :3], sim.y[idx, 3:], n=180, r_max=max(3e5, 1.5 * r_far))
        for k, i in enumerate(idx):
            col = sim.sats[i].color
            if i == sel:
                lines.append(Line(pts[k] @ W.T, col, 2))
            else:
                lines.append(Line(pts[k] @ W.T, theme.dim(col, 0.75), 1))

    def _stations(self, sim, W, world_to_ecef, sel, lines, markers):
        """Ground-station squares and a link line to every active satellite in
        view (only the selected one's in large constellations)."""
        r_ecef = eci_to_ecef(sim.y[:, :3], sim.gmst()) if sim.n else np.zeros((0, 3))
        for st in sim.stations:
            p_ecef = st.ecef()
            p_ecef = p_ecef * (1 + 5 / np.linalg.norm(p_ecef))   # 5 km up, above the globe
            p_world = p_ecef @ world_to_ecef
            markers.append((p_world, st.color or (120, 255, 160), -3, st.name, False))
            if not sim.n:
                continue
            for i in np.flatnonzero(st.sees(r_ecef) & sim.active):
                if sim.n > 60 and i != sel:
                    continue
                lines.append(Line(np.linspace(p_world, W @ sim.y[i, :3], 12),
                                  (80, 220, 130), 1, 0.4))

    def _satellites(self, sim, opts, W, sun, sel, lines, markers):
        """A marker per satellite (dimmed in the Earth's shadow, grey once
        inactive, a countdown on rockets still on the pad) and velocity arrows."""
        self._sat_lit = shadow_fraction(sim.y[:, :3], sun) if sim.n else np.zeros(0)
        self._sat_start = len(markers)
        pads = {sim.sats.index(a.sat): a for a in sim.ascents if a.phase == "pad"}
        for i, s in enumerate(sim.sats):
            p = W @ sim.y[i, :3]
            few = sim.n <= 40 or i == sel           # label only when it will not clutter
            if s.status != ACTIVE:
                markers.append((p, (110, 110, 110), 2, f"{s.name} ({s.status})" if few else None,
                                i == sel))
                continue
            label = s.name if opts.labels and few else None
            if label and i in pads:
                label += f"  T-{format_duration(pads[i].t0 - sim.t).split('.')[0]}"
            col = theme.mix(theme.dim(s.color, 0.32), s.color, float(self._sat_lit[i]))
            markers.append((p, col, 5 if i == sel else 3, label, i == sel))
            if opts.vectors and few:
                tip = sim.y[i, :3] + sim.y[i, 3:] * 180.0     # 3 minutes of velocity
                lines.append(Line(np.linspace(p, W @ tip, 8),
                                  theme.mix(s.color, (255, 255, 255), 0.4), 2))

    def _callout(self, surf, app, info, sel):
        """The orbit-data callout beside the selected satellite, if it is on screen."""
        from . import orbitviz
        sx, sy, vis = self.sat_screen
        bounds = app.view_rect()
        if vis[sel] and np.isfinite(sx[sel]) and bounds.collidepoint(sx[sel], sy[sel]):
            orbitviz.draw_callout(surf, app, info, app.sim.sats[sel], (sx[sel], sy[sel]), bounds)

    def _orbit_indices(self, app):
        """Satellites whose osculating orbit is drawn, per the ``orbits`` option."""
        sim, mode = app.sim, app.opts.orbits
        pads = {id(a.sat) for a in sim.ascents if a.phase == "pad"}
        act = [i for i, s in enumerate(sim.sats) if s.status == ACTIVE and id(s) not in pads]
        if mode == "none":
            return []
        if mode == "all" or (mode == "auto" and sim.n <= 60):
            return act
        return [i for i in act if i == app.selected]

    def _trails(self, sim, app, W):
        """History trails, fading toward the past (drawn in chunks of rising brightness)."""
        times, data = sim.history.series()
        if len(times) < 2:
            return []
        if sim.n > 60:
            ids = [app.selected] if 0 <= app.selected < sim.n else []
        else:
            ids = list(range(sim.n))
        if not ids:
            return []
        out = []
        pos = data[ids, :, :3]
        if app.opts.frame == "ECEF":
            pos = eci_to_ecef(pos, sim.clock.gmst(times)[None, :])
        # append the live position so the trail reaches the satellite
        live = np.stack([W @ sim.y[i, :3] for i in ids])[:, None, :]
        pos = np.concatenate([pos, live], axis=1)
        m = pos.shape[1]
        chunks = 6 if len(ids) <= 20 else 1
        for k, i in enumerate(ids):
            col = sim.sats[i].color
            for c in range(chunks):
                a = int(c * (m - 1) / chunks)
                b = int((c + 1) * (m - 1) / chunks) + 1
                if b - a < 2:
                    continue
                f = 0.25 + 0.75 * (c + 1) / chunks
                out.append(Line(pos[k, a:b], theme.mix(theme.BG, col, f), 1, 0.5))
        return out

    def _selected_extras(self, sim, i, W, world_to_ecef, opts, markers):
        """Nadir line, apsis/node markers and the coverage footprint of satellite ``i``."""
        out = []
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        col = sim.sats[i].color
        # nadir line
        rn = r / np.linalg.norm(r)
        out.append(Line(np.linspace(W @ r, W @ (rn * R_EARTH), 16), theme.dim(col, 0.6), 1, 0.4))
        # apsides and ascending node (the orbit-geometry overlay labels these itself)
        plain = opts.geometry == "off"
        if plain and el.e > 1e-4:
            pe, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, 0.0)
            markers.append((W @ pe, (255, 255, 255), -2, "perigee", False))
            if el.e < 1:
                ap, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, math.pi)
                markers.append((W @ ap, (255, 255, 255), -2, "apogee", False))
        if plain and math.sin(el.i) > 1e-3:
            nu_an = (-el.argp) % (2 * math.pi)
            rr = el.p / (1 + el.e * math.cos(nu_an))
            if rr > 0 and (el.e < 1 or math.cos(nu_an) > -1 / el.e):
                an, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, nu_an)
                markers.append((W @ an, (255, 220, 120), -2, "ascending node", False))
        # coverage footprint on the ground
        if opts.footprint:
            theta = sim.gmst()
            re = eci_to_ecef(r, theta)
            lat0 = math.asin(re[2] / np.linalg.norm(re))
            lon0 = math.atan2(re[1], re[0])
            alt = np.linalg.norm(r) - R_EARTH
            lam = coverage_half_angle(alt, math.radians(10.0))
            lat, lon = footprint(lat0, lon0, lam, 120)
            ring = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
            out.append(Line(ring * R_EARTH * 1.003 @ world_to_ecef,
                            theme.mix(col, (255, 255, 255), 0.3), 2, 0.0))
        return out

    # --- drawing ------------------------------------------------------------------------------
    def _exhaust(self, surf, sim, cam, W):
        """A flame behind every vehicle whose engines are running, sized to
        the view so it shows at any zoom."""
        for a in sim.ascents:
            if not a.burning:
                continue
            p = W @ sim.y[sim.sats.index(a.sat), :3]
            d = W @ np.asarray(a.thrust_dir)
            length = cam.distance * 0.03 * (0.5 + 0.5 * a.throttle)
            pts = np.stack([p, p - d * length, p - d * length * 0.45])
            sx, sy, _ = cam.project(pts)
            if not np.all(np.isfinite(sx)) or _occluded(cam, pts[:1])[0]:
                continue
            if np.any(np.abs(sx) > CLIP) or np.any(np.abs(sy) > CLIP):
                continue
            head, tail, mid = ((int(x), int(y)) for x, y in zip(sx, sy, strict=True))
            flicker = 0.85 + 0.15 * math.sin(pygame.time.get_ticks() * 0.05)
            pygame.draw.line(surf, theme.dim((255, 110, 40), flicker), head, tail, 6)
            pygame.draw.line(surf, (255, 200, 90), head, mid, 3)
            pygame.draw.circle(surf, (255, 245, 210), head, 3)

    def _project_markers(self, cam, markers):
        """Screen x, y and visibility of every marker position."""
        if not markers:
            return (np.zeros(0), np.zeros(0), np.zeros(0, bool))
        pts = np.stack([m[0] for m in markers])
        sx, sy, _ = cam.project(pts)
        hid = _occluded(cam, pts)
        ok = np.isfinite(sx) & np.isfinite(sy)
        return sx, sy, ok & ~hid

    def _draw_markers(self, surf, app, markers, mpos, behind: bool):
        """Draw hidden or visible markers: dots (radius > 0) with glows for sunlit
        satellites, squares (radius < 0) and labels."""
        sx, sy, vis = mpos
        start = self._sat_start
        glow = len(markers) - start <= 300
        labels = []
        for k, (_, col, rad, label, selected) in enumerate(markers):
            if not np.isfinite(sx[k]) or not np.isfinite(sy[k]):
                continue
            if abs(sx[k]) > CLIP or abs(sy[k]) > CLIP:
                continue
            if behind == bool(vis[k]):
                # the overlay's point labels (apsides, nodes) still show, dimmed, behind the Earth
                if not behind and label and rad < 0 and k in self._overlay:
                    labels.append((k, int(sx[k]), int(sy[k]), theme.dim(col, 0.6), label, False,
                                   True, rad))
                continue
            x, y = int(sx[k]), int(sy[k])
            if rad > 0 and glow and k >= start and not behind:
                lit = float(self._sat_lit[k - start]) if k - start < len(self._sat_lit) else 0.0
                if lit > 0.2:
                    g = self._sprite("glow", theme.dim(col, lit), 5 * rad)
                    surf.blit(g, (x - 5 * rad, y - 5 * rad), special_flags=pygame.BLEND_ADD)
            if rad > 0:
                pygame.draw.circle(surf, col, (x, y), rad)
                if selected:
                    pygame.draw.circle(surf, (255, 255, 255), (x, y), rad + 5, 1)
            elif rad < 0:
                pygame.draw.rect(surf, col, (x + rad, y + rad, -2 * rad, -2 * rad), 1)
            if label and not behind:
                tie = rad if k in self._overlay else None
                labels.append((k, x, y, col, label, selected, False, tie))
        if labels:
            self._draw_labels(surf, app.fonts, labels, start, app.view_rect())

    def _draw_labels(self, surf, fonts, labels, start, bounds):
        """Place each label beside its marker, inside ``bounds`` (clear of the
        panels) and overlapping no label placed before it, trying right, left,
        below-right and below-left; a label with no free spot is left out. The
        selected satellite goes first, then the other satellites, the selected
        orbit's annotations, stations and axes; labels of points hidden by the
        Earth come last. The orbit overlay's labels (``tie`` is their marker's
        radius) may also sit further off, joined to their point by a leader line."""
        placed = self.label_rects
        order = sorted(labels, key=lambda m: (m[6], not m[5], m[0] < start,
                                              m[0] if m[0] >= start else -m[0]))
        for _, x, y, col, label, selected, _, tie in order:
            txt = fonts.render(label, theme.mix(col, (255, 255, 255), 0.35), fonts.small)
            w, h = txt.get_size()
            spots = [(x + 8, y - 8), (x - 8 - w, y - 8), (x + 8, y + 4), (x - 8 - w, y + 4)]
            if tie is not None:
                spots += [(x + 16, y - 26), (x - 16 - w, y - 26), (x + 16, y + 14),
                          (x - 16 - w, y + 14), (x - w // 2, y - 34), (x - w // 2, y + 22)]
            rect = next((r for r in (pygame.Rect(sx_, sy_, w, h) for sx_, sy_ in spots)
                         if bounds.contains(r) and r.inflate(4, 0).collidelist(placed) < 0),
                        None)
            if rect is None:
                if not selected:
                    continue
                rect = pygame.Rect(spots[0], (w, h))    # the selected name always shows
            placed.append(rect)
            if tie is not None:
                # a leader from the labelled point to the label's nearest edge
                end = (min(max(x, rect.left - 3), rect.right + 3),
                       min(max(y, rect.top), rect.bottom))
                pygame.draw.aaline(surf, theme.dim(col, 0.9), (x, y), end)
                if tie == 0:
                    pygame.draw.circle(surf, col, (x, y), 3)
            # a dark backing keeps the text legible over the Earth and bright orbits
            theme.panel(surf, rect.inflate(6, 0), (4, 7, 16, 190), None, 3)
            surf.blit(txt, rect)

    def _stars(self, surf, cam):
        """Plot the star field directly into the pixel buffer (bright stars as 2x2)."""
        sx, sy, _ = cam.project_dirs(self.sky_dirs)
        w, h = surf.get_size()
        ok = np.isfinite(sx) & (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
        x = sx[ok].astype(np.intp)
        y = sy[ok].astype(np.intp)
        col = self.sky_col[ok]
        big = self.sky_big[ok]
        px = pygame.surfarray.pixels3d(surf)
        px[x, y] = np.maximum(px[x, y], col)
        for dx, dy in ((1, 0), (0, 1), (1, 1)):
            px[x[big] + dx, y[big] + dy] = np.maximum(px[x[big] + dx, y[big] + dy],
                                                      (col[big] * 0.55).astype(np.uint8))
        del px

    def _sun(self, surf, cam, sun_dir, app):
        """Additive Sun glare sprite and label."""
        sx, sy, _ = cam.project_dirs(sun_dir[None, :])
        if not np.isfinite(sx[0]):
            return
        x, y = int(sx[0]), int(sy[0])
        if -200 < x < cam.width + 200 and -200 < y < cam.height + 200:
            spr = self._sprite("sun", (255, 236, 190), 150)
            surf.blit(spr, (x - 150, y - 150), special_flags=pygame.BLEND_ADD)
            app.fonts.draw(surf, "Sun", (x + 18, y + 12), theme.SUN, app.fonts.small)


def _occluded(cam, pts):
    """Earth occlusion of markers. The globe is drawn as a sphere of
    equatorial radius, but the WGS-84 surface is up to 21 km lower: a launch
    pad (or ground station) away from the equator sits inside that sphere, so
    such points are tested against a sphere just beneath them."""
    hid = cam.hidden_by_sphere(pts)
    rm = np.linalg.norm(pts, axis=-1)
    for k in np.flatnonzero(rm < R_EARTH * 1.002):
        hid[k] = cam.hidden_by_sphere(pts[k:k + 1], min(R_EARTH, rm[k]) * 0.998)[0]
    return hid


def _runs(surf, color, xs, ys, mask, width):
    """Draw the polyline through the points where ``mask`` holds, broken at gaps."""
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, idx.size]
    pts = np.stack([xs, ys], axis=1)
    for s, e in zip(starts, ends, strict=True):
        if e - s < 2:
            continue
        seg = pts[idx[s:e]].tolist()
        if width == 1:
            pygame.draw.aalines(surf, color, False, seg)
        else:
            pygame.draw.lines(surf, color, False, seg, width)

