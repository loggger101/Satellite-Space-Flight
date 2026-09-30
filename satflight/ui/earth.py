"""Ray-cast Earth.

Every frame the Earth's screen footprint is ray-traced with numpy at a
capped resolution: each pixel's view ray is intersected with the sphere,
the hit point is rotated into ECEF for texturing, and lit by the true Sun
direction (soft terminator, night side, atmospheric rim and halo). The
result is a per-pixel-alpha surface blitted between the "behind Earth" and
"in front of Earth" drawing passes, which gives exact occlusion.

An equirectangular texture at ``assets/earth.jpg`` and coastlines at
``assets/coastlines.json`` are used when present (see
``tools/fetch_assets.py``); otherwise a procedural globe with a graticule
is drawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pygame

from ..constants import R_EARTH

HALO = 1.035          # atmosphere halo radius in Earth radii
NIGHT_TINT = np.array([10, 18, 40], np.float32)
RIM_COLOR = np.array([90, 150, 255], np.float32)
HALO_COLOR = np.array([110, 175, 255], np.float32)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class EarthRenderer:
    def __init__(self, asset_dir: Path | None = None, quality: int = 120_000):
        self.quality = quality
        self.texture = None                        # (W, H, 3) uint8, equirectangular
        self.has_image = False
        self.coastlines: list[np.ndarray] = []     # unit ECEF polylines
        self._geom_key = None
        self._geom = None
        if asset_dir is not None:
            self.load_assets(Path(asset_dir))
        if self.texture is None:
            self.texture = self._procedural_texture()

    # --- assets -------------------------------------------------------------------
    def load_assets(self, asset_dir: Path):
        for name in ("earth.jpg", "earth.png"):
            p = asset_dir / name
            if p.exists():
                try:
                    img = pygame.image.load(str(p))
                    self.texture = pygame.surfarray.array3d(img).astype(np.uint8)
                    self.has_image = True
                except pygame.error:
                    self.texture = None
                break
        p = asset_dir / "coastlines.json"
        if p.exists():
            try:
                lines = json.loads(p.read_text(encoding="utf-8"))
                for line in lines:
                    ll = np.radians(np.asarray(line, dtype=float))
                    lon, lat = ll[:, 0], ll[:, 1]
                    self.coastlines.append(np.stack([np.cos(lat) * np.cos(lon),
                                                     np.cos(lat) * np.sin(lon),
                                                     np.sin(lat)], axis=1))
            except (ValueError, OSError):
                self.coastlines = []

    # --- colouring -----------------------------------------------------------------------
    @staticmethod
    def _procedural_texture(w: int = 1440, h: int = 720) -> np.ndarray:
        """Ocean-blue globe with a 15-degree graticule and polar ice, baked once."""
        lond = np.linspace(-180, 180, w, endpoint=False) + 180.0 / w
        latd = np.linspace(90, -90, h, endpoint=False) - 90.0 / h
        lon, lat = np.meshgrid(np.radians(lond), np.radians(latd), indexing="ij")
        lond, latd = np.degrees(lon), np.degrees(lat)
        k = np.abs(np.sin(lat))[..., None]
        col = (np.array([18, 58, 128], np.float32) * (1 - k)
               + np.array([26, 84, 150], np.float32) * k)
        gl = np.minimum(np.mod(latd, 15.0), 15.0 - np.mod(latd, 15.0)) < 0.22
        gm = np.minimum(np.mod(lond, 15.0), 15.0 - np.mod(lond, 15.0)) < 0.22 * np.maximum(
            1.0, 1.0 / np.maximum(np.cos(lat), 0.08))
        grid = (gl | gm)[..., None]
        col = np.where(grid, col * 0.55 + np.array([70, 140, 210], np.float32) * 0.45, col)
        eq = ((np.abs(latd) < 0.3) | (np.abs(lond) < 0.3 / np.maximum(np.cos(lat), 0.1)))[..., None]
        col = np.where(eq, np.array([110, 200, 255], np.float32), col)
        ice = _smoothstep(72.0, 80.0, np.abs(latd))[..., None]
        col = col * (1 - ice) + np.array([225, 235, 245], np.float32) * ice
        return np.clip(col, 0, 255).astype(np.uint8)

    def _base_color(self, lat, lon):
        w, h = self.texture.shape[:2]
        u = ((lon + np.pi) * (w / (2 * np.pi))).astype(np.int32) % w
        v = np.clip(((np.pi / 2 - lat) * (h / np.pi)).astype(np.int32), 0, h - 1)
        return self.texture[u, v].astype(np.float32)

    # --- rendering -----------------------------------------------------------------------
    def _bbox(self, cam):
        c = cam.position
        dist = float(np.linalg.norm(c))
        rh = R_EARTH * HALO
        full = (0, 0, cam.width, cam.height)
        if dist <= rh * 1.001:
            return full
        w = -c / dist
        L = math.sqrt(dist * dist - rh * rh)
        alpha = math.asin(rh / dist)
        center = c + w * L * math.cos(alpha)
        rad = L * math.sin(alpha)
        e1 = np.cross(w, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(w, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pts = center + rad * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
        sx, sy, z = cam.project(pts)
        if np.any(z <= cam.near) or not np.all(np.isfinite(sx)):
            return full
        x0 = max(0, int(np.floor(sx.min())) - 2)
        y0 = max(0, int(np.floor(sy.min())) - 2)
        x1 = min(cam.width, int(np.ceil(sx.max())) + 2)
        y1 = min(cam.height, int(np.ceil(sy.max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    def render(self, target: pygame.Surface, cam, world_to_ecef: np.ndarray,
               sun_dir: np.ndarray):
        """Draw the Earth. ``world_to_ecef`` rotates world vectors into ECEF;
        ``sun_dir`` is the unit Sun direction in world coordinates."""
        if float(np.linalg.norm(cam.position)) <= R_EARTH * 1.0005:
            return
        bbox = self._bbox(cam)
        if bbox is None:
            return
        key = (cam.state_key(), bbox, self.quality)
        if key != self._geom_key:
            self._geom = self._geometry(cam, bbox)
            self._geom_key = key
        surf = self._shade(self._geom, world_to_ecef.astype(np.float32),
                           np.asarray(sun_dir, np.float32))
        target.blit(surf, bbox[:2])

    def _geometry(self, cam, bbox):
        """Camera-dependent part: which pixels hit the Earth, their normals,
        limb (rim) factors and the atmosphere halo."""
        x0, y0, bw, bh = bbox
        s = min(1.0, math.sqrt(self.quality / float(bw * bh)))
        nw, nh = max(2, int(bw * s)), max(2, int(bh * s))
        xs = x0 + (np.arange(nw) + 0.5) * (bw / nw)
        ys = y0 + (np.arange(nh) + 0.5) * (bh / nh)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        d = cam.ray_dirs(gx, gy)                        # (nw, nh, 3)
        c = cam.position
        b = d @ c
        cc = float(c @ c)
        disc = b * b - (cc - R_EARTH * R_EARTH)
        hit = disc > 0
        t = -b - np.sqrt(np.where(hit, disc, 0.0))
        hit &= t > 0
        dh = d[hit]
        n = (c + t[hit][:, None] * dh) / R_EARTH
        rim = (1.0 - np.clip(-np.sum(dh * n, axis=1), 0.0, 1.0)) ** 3
        m = np.sqrt(np.maximum(cc - b * b, 0.0))
        halo = ~hit & (b < 0) & (m < R_EARTH * HALO)
        q = c + (-b[halo])[:, None] * d[halo]
        qn = q / np.maximum(m[halo], 1.0)[:, None]
        g = (1.0 - (m[halo] - R_EARTH) / (R_EARTH * (HALO - 1.0))) ** 2
        alpha = np.zeros((nw, nh), np.uint8)
        alpha[hit] = 255
        return dict(size=(nw, nh), out=(bw, bh), hit=hit, n=n.astype(np.float32),
                    rim=rim.astype(np.float32), halo=halo, qn=qn.astype(np.float32),
                    g=g.astype(np.float32), alpha=alpha)

    def _shade(self, geo, world_to_ecef, sun_dir):
        """Time-dependent part: texture lookup under Earth rotation + lighting."""
        nw, nh = geo["size"]
        rgb = np.zeros((nw, nh, 3), np.float32)
        alpha = geo["alpha"].copy()
        n = geo["n"]
        if len(n):
            ne = n @ world_to_ecef.T
            lat = np.arcsin(np.clip(ne[:, 2], -1, 1))
            lon = np.arctan2(ne[:, 1], ne[:, 0])
            base = self._base_color(lat, lon)
            ndl = n @ sun_dir
            day = _smoothstep(-0.10, 0.12, ndl)
            light = 0.13 + 0.87 * day * (0.35 + 0.65 * np.clip(ndl, 0.0, 1.0))
            col = base * light[:, None] + NIGHT_TINT * ((1 - day) * 0.6)[:, None]
            col += geo["rim"][:, None] * RIM_COLOR * (0.25 + 0.6 * day)[:, None]
            rgb[geo["hit"]] = col
        if len(geo["qn"]):
            hl = np.clip(0.2 + 0.9 * (geo["qn"] @ sun_dir + 0.25), 0.08, 1.0)
            rgb[geo["halo"]] = HALO_COLOR * hl[:, None]
            alpha[geo["halo"]] = np.clip(230.0 * geo["g"] * hl, 0, 255).astype(np.uint8)
        surf = pygame.Surface((nw, nh), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(surf)
        px[...] = np.clip(rgb, 0, 255).astype(np.uint8)
        del px
        pa = pygame.surfarray.pixels_alpha(surf)
        pa[...] = alpha
        del pa
        if (nw, nh) != geo["out"]:
            surf = pygame.transform.smoothscale(surf, geo["out"])
        return surf
