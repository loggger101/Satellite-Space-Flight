"""3-D scene renderer: stars, Sun, Moon, Earth, axes, orbits, trails,
satellites, ground stations and annotations.

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

from ..constants import R_EARTH, R_GEO, R_MOON
from ..elements import coe2rv, conic_points, rv2coe
from ..ephemeris import moon_position, sun_position
from ..frames import eci_to_ecef, look_angles, rot3
from ..analysis import coverage_half_angle, footprint
from ..simulation import ACTIVE
from . import theme
from .earth import EarthRenderer

CLIP = 30000.0


@dataclass
class Line:
    pts: np.ndarray          # (M, 3) world km
    color: tuple
    width: int = 1
    behind_dim: float = 0.55


class SceneRenderer:
    def __init__(self, asset_dir):
        self.earth = EarthRenderer(asset_dir)
        rng = np.random.default_rng(7)
        v = rng.normal(size=(1800, 3))
        self.star_dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        mag = rng.random(1800) ** 3
        self.star_bright = (40 + 215 * mag).astype(int)
        self.star_size = np.where(mag > 0.85, 2, 1)
        self.sat_screen = None      # (sx, sy, visible) of last frame, for picking

    # --- helpers -------------------------------------------------------------------------
    @staticmethod
    def world_rotation(sim, frame: str) -> np.ndarray:
        """Matrix W with world = W @ eci (identity in ECI view)."""
        if frame == "ECEF":
            return rot3(sim.gmst())
        return np.eye(3)

    def _draw_lines(self, surf, cam, lines: list[Line], behind: bool, cache):
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
        sim, cam, opts = app.sim, app.camera, app.opts
        W = self.world_rotation(sim, opts.frame)
        world_to_ecef = rot3(sim.gmst()) @ W.T
        jd = sim.jd()
        sun = sun_position(jd)
        sun_dir = W @ (sun / np.linalg.norm(sun))

        surf.fill(theme.BG)
        self._stars(surf, cam)
        self._sun(surf, cam, sun_dir, app)

        lines: list[Line] = []
        markers = []   # (pos world, color, radius, label, is_selected)
        sel = app.selected if 0 <= app.selected < sim.n else -1

        if opts.axes:
            names = ("X (vernal equinox)", "Y", "Z (north pole)") if opts.frame == "ECI" else \
                    ("X (Greenwich)", "Y (90E)", "Z (north pole)")
            for k, col in enumerate((theme.AXIS_X, theme.AXIS_Y, theme.AXIS_Z)):
                end = np.zeros(3)
                end[k] = 2.3 * R_EARTH
                lines.append(Line(np.linspace(np.zeros(3), end, 24), col, 2, 0.5))
                markers.append((end, col, 0, names[k], False))
        if opts.geo_ring:
            a = np.linspace(0, 2 * np.pi, 181)
            ring = np.stack([np.cos(a), np.sin(a), 0 * a], 1) * R_GEO
            lines.append(Line(ring, (60, 70, 100), 1, 0.7))
        if opts.equator:
            a = np.linspace(0, 2 * np.pi, 181)
            ring = np.stack([np.cos(a), np.sin(a), 0 * a], 1) * R_EARTH * 1.002
            lines.append(Line(ring, (70, 120, 170), 1, 0.0))

        if self.earth.coastlines and opts.coastlines:
            M = world_to_ecef           # ecef row-vector -> world: p @ M
            for cl in self.earth.coastlines:
                lines.append(Line(cl @ M * (R_EARTH * 1.002), (150, 200, 150), 1, 0.0))

        # orbits and trails
        active = sim.active
        show_orbit = self._orbit_indices(app)
        if show_orbit:
            idx = np.array(show_orbit)
            pts = conic_points(sim.y[idx, :3], sim.y[idx, 3:], n=180,
                               r_max=max(3e5, 1.5 * float(np.max(np.linalg.norm(sim.y[idx, :3], axis=1)))))
            for k, i in enumerate(idx):
                col = sim.sats[i].color
                w = 2 if i == sel else 1
                colr = col if i == sel else theme.dim(col, 0.75)
                lines.append(Line(pts[k] @ W.T, colr, w))
        if opts.trails:
            lines.extend(self._trails(sim, app, W))

        # ground stations + links
        if opts.stations and sim.stations:
            theta = sim.gmst()
            r_ecef = eci_to_ecef(sim.y[:, :3], theta) if sim.n else np.zeros((0, 3))
            for st in sim.stations:
                p_ecef = st.ecef()
                p_world = (p_ecef / np.linalg.norm(p_ecef) * (np.linalg.norm(p_ecef) + 5)) @ world_to_ecef
                markers.append((p_world, st.color or (120, 255, 160), -3, st.name, False))
                if sim.n:
                    _, el, _ = look_angles(p_ecef, math.radians(st.lat), math.radians(st.lon), r_ecef)
                    for i in np.flatnonzero((el >= math.radians(st.min_el)) & active):
                        if sim.n > 60 and i != sel:
                            continue
                        lines.append(Line(np.linspace(p_world, W @ sim.y[i, :3], 12),
                                          (80, 220, 130), 1, 0.4))

        # selected-satellite annotations
        if sel >= 0 and sim.sats[sel].status == ACTIVE:
            lines.extend(self._selected_extras(sim, sel, W, world_to_ecef, opts, markers))

        # satellites
        for i, s in enumerate(sim.sats):
            p = W @ sim.y[i, :3]
            if s.status != ACTIVE:
                markers.append((p, (110, 110, 110), 2, s.name + f" ({s.status})"
                                if (sim.n <= 40 or i == sel) else None, i == sel))
                continue
            label = s.name if (opts.labels and (sim.n <= 40 or i == sel)) else None
            markers.append((p, s.color, 5 if i == sel else 3, label, i == sel))
            if opts.vectors and (sim.n <= 40 or i == sel):
                v = sim.y[i, 3:]
                tip = sim.y[i, :3] + v * 180.0
                lines.append(Line(np.linspace(p, W @ tip, 8), theme.mix(s.color, (255, 255, 255), 0.4), 2))

        # the Moon (drawn in depth order relative to Earth)
        moon_world = W @ moon_position(jd)
        cache: dict = {}
        mpos = self._project_markers(cam, markers)
        self._draw_lines(surf, cam, lines, True, cache)
        self._draw_markers(surf, app, markers, mpos, behind=True)
        earth_depth = float(cam.to_camera(np.zeros(3))[2])
        moon_depth = float(cam.to_camera(moon_world)[2])
        if opts.moon and moon_depth > earth_depth:
            self._moon(surf, cam, moon_world, sun_dir, app)
        self.earth.render(surf, cam, world_to_ecef, sun_dir)
        if opts.moon and moon_depth <= earth_depth:
            self._moon(surf, cam, moon_world, sun_dir, app)
        self._draw_lines(surf, cam, lines, False, cache)
        self._draw_markers(surf, app, markers, mpos, behind=False)

        # picking data
        # satellites are the last markers appended
        nsat = sim.n
        if nsat:
            sx, sy, vis = mpos
            self.sat_screen = (sx[-nsat:], sy[-nsat:], vis[-nsat:])
        else:
            self.sat_screen = None

    # --- pieces -----------------------------------------------------------------------------
    def _orbit_indices(self, app):
        sim, mode = app.sim, app.opts.orbits
        act = [i for i, s in enumerate(sim.sats) if s.status == ACTIVE]
        if mode == "none":
            return []
        if mode == "all" or (mode == "auto" and sim.n <= 60):
            return act
        return [i for i in act if i == app.selected]

    def _trails(self, sim, app, W):
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
        out = []
        r, v = sim.y[i, :3], sim.y[i, 3:]
        el = rv2coe(r, v)
        col = sim.sats[i].color
        # nadir line
        rn = r / np.linalg.norm(r)
        out.append(Line(np.linspace(W @ r, W @ (rn * R_EARTH), 16), theme.dim(col, 0.6), 1, 0.4))
        # apsides and ascending node
        if el.e > 1e-4:
            pe, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, 0.0)
            markers.append((W @ pe, (255, 255, 255), -2, "Pe", False))
            if el.e < 1:
                ap, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, math.pi)
                markers.append((W @ ap, (255, 255, 255), -2, "Ap", False))
        if math.sin(el.i) > 1e-3:
            nu_an = (-el.argp) % (2 * math.pi)
            rr = el.p / (1 + el.e * math.cos(nu_an))
            if rr > 0 and (el.e < 1 or math.cos(nu_an) > -1 / el.e):
                an, _ = coe2rv(el.a, el.e, el.i, el.raan, el.argp, nu_an)
                markers.append((W @ an, (255, 220, 120), -2, "AN", False))
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
            out.append(Line(ring * R_EARTH * 1.003 @ world_to_ecef, theme.mix(col, (255, 255, 255), 0.3), 2, 0.0))
        return out

    def _project_markers(self, cam, markers):
        if not markers:
            return (np.zeros(0), np.zeros(0), np.zeros(0, bool))
        pts = np.stack([m[0] for m in markers])
        sx, sy, _ = cam.project(pts)
        hid = cam.hidden_by_sphere(pts)
        ok = np.isfinite(sx) & np.isfinite(sy)
        return sx, sy, ok & ~hid

    def _draw_markers(self, surf, app, markers, mpos, behind: bool):
        sx, sy, vis = mpos
        fonts = app.fonts
        for k, (p, col, rad, label, selected) in enumerate(markers):
            if not np.isfinite(sx[k]) or not np.isfinite(sy[k]):
                continue
            if abs(sx[k]) > CLIP or abs(sy[k]) > CLIP:
                continue
            if behind == bool(vis[k]):
                continue
            x, y = int(sx[k]), int(sy[k])
            if rad > 0:
                pygame.draw.circle(surf, col, (x, y), rad)
                if selected:
                    pygame.draw.circle(surf, (255, 255, 255), (x, y), rad + 5, 1)
            elif rad < 0:
                pygame.draw.rect(surf, col, (x + rad, y + rad, -2 * rad, -2 * rad), 1)
            if label and not behind:
                fonts.draw(surf, label, (x + 8, y - 8), theme.mix(col, (255, 255, 255), 0.35),
                           fonts.small)

    def _stars(self, surf, cam):
        sx, sy, z = cam.project_dirs(self.star_dirs)
        ok = np.isfinite(sx) & (sx >= 0) & (sx < cam.width) & (sy >= 0) & (sy < cam.height)
        for x, y, b, s in zip(sx[ok].astype(int), sy[ok].astype(int), self.star_bright[ok],
                              self.star_size[ok]):
            surf.fill((b, b, min(255, b + 20)), (x, y, s, s))

    def _sun(self, surf, cam, sun_dir, app):
        sx, sy, z = cam.project_dirs(sun_dir[None, :])
        if not np.isfinite(sx[0]):
            return
        x, y = int(sx[0]), int(sy[0])
        if -200 < x < cam.width + 200 and -200 < y < cam.height + 200:
            glow = pygame.Surface((120, 120), pygame.SRCALPHA)
            for rr, a in ((60, 18), (40, 35), (24, 70), (12, 255)):
                pygame.draw.circle(glow, (*theme.SUN, a), (60, 60), rr)
            surf.blit(glow, (x - 60, y - 60))
            app.fonts.draw(surf, "Sun", (x + 16, y + 10), theme.SUN, app.fonts.small)

    def _moon(self, surf, cam, moon_world, sun_dir, app):
        sx, sy, z = cam.project(moon_world[None, :])
        if not np.isfinite(sx[0]):
            return
        rad = max(3, int(cam.screen_radius(moon_world, R_MOON)))
        x, y = int(sx[0]), int(sy[0])
        if -rad < x < cam.width + rad and -rad < y < cam.height + rad and rad < 4000:
            pygame.draw.circle(surf, (70, 70, 78), (x, y), rad)
            # lit half: offset disc towards the Sun's screen direction
            sdx = float(sun_dir @ cam.right)
            sdy = -float(sun_dir @ cam.up)
            n = math.hypot(sdx, sdy) or 1.0
            lit = pygame.Surface((2 * rad + 2, 2 * rad + 2), pygame.SRCALPHA)
            pygame.draw.circle(lit, (*theme.MOON, 255), (rad + 1, rad + 1), rad)
            mask = pygame.Surface((2 * rad + 2, 2 * rad + 2), pygame.SRCALPHA)
            off = (int(rad * 0.9 * sdx / n), int(rad * 0.9 * sdy / n))
            pygame.draw.circle(mask, (255, 255, 255, 255), (rad + 1 + off[0], rad + 1 + off[1]), rad)
            lit.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            surf.blit(lit, (x - rad - 1, y - rad - 1))
            app.fonts.draw(surf, "Moon", (x + rad + 6, y), theme.MOON, app.fonts.small)


def _runs(surf, color, xs, ys, mask, width):
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, idx.size]
    pts = np.stack([xs, ys], axis=1)
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        seg = pts[idx[s:e]].tolist()
        if width == 1:
            pygame.draw.aalines(surf, color, False, seg)
        else:
            pygame.draw.lines(surf, color, False, seg, width)

