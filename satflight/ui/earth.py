"""Ray-cast Earth.

The Earth's screen footprint is ray-traced with numpy at a capped
resolution. The camera-dependent part (which pixels hit the sphere, their
normals, view vectors, limb factors and the atmosphere halo) is cached until
the camera moves; every frame only re-textures the hit points under the
Earth's rotation and relights them for the current Sun direction:

* Lambert day side with a soft terminator and a dim blue night side
* orange twilight band along the terminator
* Blinn-Phong sun glint on water only (ocean mask derived from the texture)
* atmospheric limb brightening and an outer halo that reddens at sunset

The result is a per-pixel-alpha surface blitted between the "behind Earth"
and "in front of Earth" passes, which gives exact occlusion.

``assets/earth.jpg`` (NASA Blue Marble, via ``tools/fetch_assets.py``) is
used when present; otherwise a procedural globe with a graticule is drawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pygame

from ..constants import R_EARTH

HALO = 1.035          # atmosphere halo radius in Earth radii
NIGHT_TINT = np.array([6, 12, 30], np.float32)
RIM_COLOR = np.array([95, 160, 255], np.float32)
HALO_COLOR = np.array([105, 170, 255], np.float32)
TWILIGHT = np.array([255, 120, 50], np.float32)
GLINT = np.array([255, 238, 205], np.float32)
_TINTS = np.stack([NIGHT_TINT, TWILIGHT, RIM_COLOR])      # rows weighted in Earth._lighting

OCEAN_DEEP = np.array([5, 24, 66], np.float32)
OCEAN_SHALLOW = np.array([24, 98, 142], np.float32)


def smoothstep(e0, e1, x):
    """Hermite ramp from 0 at ``e0`` to 1 at ``e1`` (clamped outside)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _blur(mask: np.ndarray, factor: int = 16) -> np.ndarray:
    """Cheap wide blur of a (W, H) 0..1 array via down- and up-scaling."""
    w, h = mask.shape
    surf = pygame.Surface((w, h))
    gray = (mask * 255).astype(np.uint8)[..., None]
    pygame.surfarray.blit_array(surf, np.repeat(gray, 3, axis=2))
    small = pygame.transform.smoothscale(surf, (max(2, w // factor), max(2, h // factor)))
    big = pygame.transform.smoothscale(small, (w, h))
    return pygame.surfarray.array3d(big)[..., 0].astype(np.float32) / 255.0


class EarthRenderer:
    """Draws the textured, lit globe; ``quality`` caps the ray-traced pixel count."""

    def __init__(self, asset_dir: Path | None = None, quality: int = 230_000):
        self.quality = quality
        self.texture = None                        # (W, H, 3) uint8, equirectangular
        self.ocean = None                          # (W, H) float32, 1 = water
        self.has_image = False
        self.coastlines: list[np.ndarray] = []     # unit ECEF polylines
        self._geom_key = None
        self._geom = None
        self._tex = self._light = self._final = None
        self._last_cam = None
        self._moving = 0           # consecutive frames with a moving camera
        self._spinning = 0         # consecutive frames that needed a re-texture
        self._packed_src = self._packed_arr = None
        if asset_dir is not None:
            self.load_assets(Path(asset_dir))
        if self.texture is None:
            self.texture, self.ocean = self._procedural_texture()

    # --- assets -------------------------------------------------------------------
    def load_assets(self, asset_dir: Path):
        """Load ``earth.jpg``/``earth.png`` and ``coastlines.json`` when present."""
        for name in ("earth.jpg", "earth.png"):
            p = asset_dir / name
            if p.exists():
                try:
                    img = pygame.image.load(str(p))
                    self.texture, self.ocean = self._prepare(pygame.surfarray.array3d(img))
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

    @staticmethod
    def _prepare(tex: np.ndarray):
        """Find the water in a Blue-Marble style image (flat dark navy) and
        repaint it with depth-like shading: turquoise shallows near the
        coasts fading to deep blue offshore. Land is brightened slightly."""
        t = tex.astype(np.float32)
        r, g, b = t[..., 0], t[..., 1], t[..., 2]
        water = ((b > 1.5 * r + 8) & (b > 1.3 * g + 6) & (r < 80) & (g < 90)).astype(np.float32)
        land_near = 1.0 - _blur(water, 24)
        coast = np.clip(land_near * 2.4, 0.0, 1.0) * water
        h = t.shape[1]
        lat = np.linspace(90, -90, h)[None, :]
        cold = smoothstep(45.0, 75.0, np.abs(lat))[..., None]
        sea = OCEAN_DEEP * (1 - coast[..., None]) + OCEAN_SHALLOW * coast[..., None]
        sea = sea * (1 - 0.25 * cold) + np.array([40, 60, 80], np.float32) * 0.25 * cold
        land = np.clip(t * 1.12 + 4.0, 0, 255)
        # soft coastline (anti-aliased blend over ~2 texels instead of a hard switch)
        soft = np.clip((_blur(water, 3) - 0.5) * 1.6 + 0.5, 0.0, 1.0)
        out = sea * soft[..., None] + land * (1.0 - soft[..., None])
        return np.clip(out, 0, 255).astype(np.uint8), soft

    @staticmethod
    def _procedural_texture(w: int = 1440, h: int = 720):
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
        ice = smoothstep(72.0, 80.0, np.abs(latd))
        col = col * (1 - ice[..., None]) + np.array([225, 235, 245], np.float32) * ice[..., None]
        return np.clip(col, 0, 255).astype(np.uint8), (1.0 - ice).astype(np.float32)

    def _packed(self):
        """Texture + water mask packed as one uint32 per texel (R, G, B, water)
        in a flat array indexed ``u * H + v``: a single gather per tap."""
        if self._packed_src is not self.texture:
            w, h = self.texture.shape[:2]
            arr = np.empty((w, h, 4), np.uint8)
            arr[..., :3] = self.texture
            arr[..., 3] = np.clip(self.ocean * 255.0, 0, 255).astype(np.uint8)
            self._packed_arr = arr.reshape(-1).view(np.uint32)
            self._packed_src = self.texture
        return self._packed_arr

    # --- rendering -----------------------------------------------------------------------
    def _bbox(self, cam):
        """Screen rectangle (x, y, w, h) enclosing the Earth and its halo; the whole
        window from inside the halo, None if off screen."""
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
        """Draw the Earth. ``world_to_ecef`` (a rotation about +Z in both the
        ECI and ECEF views) rotates world vectors into ECEF; ``sun_dir`` is the
        unit Sun direction in world coordinates.

        Three caches keep this cheap: the ray geometry (until the camera
        moves), the texture sample (until the Earth has turned by half a
        buffer pixel) and the lighting terms (until the Sun moves).
        """
        if float(np.linalg.norm(cam.position)) <= R_EARTH * 1.0005:
            return
        bbox = self._bbox(cam)
        if bbox is None:
            return
        # while the camera keeps moving (dragging, follow mode) trace at a third
        # resolution; the full-quality image returns once it settles
        cam_key = (cam.state_key(), bbox)
        self._moving = self._moving + 1 if cam_key != self._last_cam else 0
        self._last_cam = cam_key
        if self._moving >= 3:
            quality = self.quality // 3
        elif self._spinning >= 3:          # high time warp: the globe turns every frame
            quality = self.quality // 2
        else:
            quality = self.quality
        key = (cam_key, quality)
        if key != self._geom_key:
            self._geom = self._geometry(cam, bbox, quality)
            self._geom_key = key
            self._tex = self._light = self._final = None
        geo = self._geom
        theta = math.atan2(float(world_to_ecef[0, 1]), float(world_to_ecef[0, 0]))
        tol = geo["tex_tol"]
        if self._tex is None or abs(math.remainder(theta - self._tex[0], math.tau)) > tol:
            self._tex = (theta, *self._sample(geo, theta))
            self._final = None
            self._spinning += 1
        else:
            self._spinning = 0
        sun = np.asarray(sun_dir, float)
        # relight only after the Sun has moved ~0.05 deg (sub-pixel terminator shift)
        if self._light is None or float(np.dot(self._light[0], sun)) < math.cos(1e-3):
            self._light = (sun, *self._lighting(geo, sun.astype(np.float32)))
            self._final = None
        if self._final is None:
            self._final = self._compose(geo)
        target.blit(self._final, bbox[:2])

    def _geometry(self, cam, bbox, quality: int):
        """Camera-dependent part: which pixels hit the Earth, their normals,
        view vectors and latitudes, limb (rim) factors and the halo.

        The camera basis (forward f, right r, up u) is orthonormal, so the
        ray through pixel offsets (dx, dy) is (f + dx r + dy u) / sqrt(1 + dx^2
        + dy^2): the ray-sphere test runs on 2-D scalar grids and full 3-D
        vectors are only built for the pixels that hit the Earth or halo."""
        x0, y0, bw, bh = bbox
        s = min(1.0, math.sqrt(quality / float(bw * bh)))
        nw, nh = max(2, int(bw * s)), max(2, int(bh * s))
        dx = ((x0 + (np.arange(nw) + 0.5) * (bw / nw) - cam.cx) / cam.focal)[:, None]
        dy = (-(y0 + (np.arange(nh) + 0.5) * (bh / nh) - cam.cy) / cam.focal)[None, :]
        c = cam.position
        f, r, u = cam.forward, cam.right, cam.up
        inv_len = 1.0 / np.sqrt(1.0 + dx * dx + dy * dy)
        b = (float(f @ c) + dx * float(r @ c) + dy * float(u @ c)) * inv_len   # d . c
        cc = float(c @ c)
        disc = b * b - (cc - R_EARTH * R_EARTH)
        hit = disc > 0
        hit &= (-b - np.sqrt(np.where(hit, disc, 0.0))) > 0
        m = np.sqrt(np.maximum(cc - b * b, 0.0))
        halo = ~hit & (b < 0) & (m < R_EARTH * HALO)
        basis = np.stack([f, r, u])

        def rays(mask):
            """Pixels of ``mask`` as (flat index, (row, col), unit rays): each
            ray is (k, k dx, k dy) in the camera basis, one matrix product."""
            ix, iy = np.nonzero(mask)
            k = inv_len[ix, iy]
            w = np.column_stack([k, k * dx[ix, 0], k * dy[0, iy]])
            return ix * nh + iy, (ix, iy), w, w @ basis

        hit_idx, (ix, iy), w, dh = rays(hit)
        sq = np.sqrt(disc[ix, iy])
        t = -b[ix, iy] - sq
        # n = (c + t d) / R, built from the same weights as the ray d
        n = (w * (t / R_EARTH)[:, None]) @ basis + c / R_EARTH
        ndv = sq / R_EARTH            # n . view = -(d . c + t) / R = sqrt(disc) / R
        cos_view = np.clip(ndv, 0.0, 1.0)
        rim = (1.0 - cos_view) * (1.0 - cos_view) * (1.0 - cos_view)
        halo_idx, (jx, jy), _, dq = rays(halo)
        q = c - b[jx, jy][:, None] * dq
        mh = m[jx, jy]
        qn = q / np.maximum(mh, 1.0)[:, None]
        g = (1.0 - (mh - R_EARTH) / (R_EARTH * (HALO - 1.0))) ** 2
        alpha = np.zeros(nw * nh, np.uint8)
        # soften the limb over the outermost buffer pixels
        alpha[hit_idx] = (255 * np.clip(cos_view * max(nw, nh) * 0.35, 0, 1)).astype(np.uint8)
        # half a buffer pixel, as an angle on the sphere below the camera
        km_per_px = (bw / nw) / cam.focal * max(float(np.linalg.norm(c)) - R_EARTH, 1.0)
        return dict(size=(nw, nh), out=(bw, bh), hit_idx=hit_idx, halo_idx=halo_idx,
                    n=n.astype(np.float32), view=(-dh).astype(np.float32),
                    ndv=ndv.astype(np.float32),
                    rim=rim.astype(np.float32), qn=qn.astype(np.float32),
                    g=g.astype(np.float32), alpha=alpha.reshape(nw, nh),
                    lat=np.arcsin(np.clip(n[:, 2], -1, 1)), lonw=np.arctan2(n[:, 1], n[:, 0]),
                    tex_tol=0.5 * km_per_px / R_EARTH, km_per_px=km_per_px)

    def _sample(self, geo, theta):
        """Texture color and water fraction under Earth rotation ``theta``.

        ECEF longitude = world longitude - theta while latitude is unchanged,
        so texture rows and their weights are fixed per geometry and only the
        column index shifts. Bilinear when the texture is magnified on
        screen, a single nearest tap otherwise."""
        if not len(geo["n"]):
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        packed = self._packed()
        w, h = self.texture.shape[:2]
        if "rows" not in geo:
            fv = (np.pi / 2 - geo["lat"]) * (h / np.pi) - 0.5
            v0 = np.floor(fv)
            geo["rows"] = (np.clip(v0, 0, h - 1).astype(np.int32),
                           np.clip(v0 + 1, 0, h - 1).astype(np.int32),
                           (fv - v0).astype(np.float32)[:, None],
                           np.clip(np.rint(fv), 0, h - 1).astype(np.int32))
            # world longitude in texel columns, offset to stay positive
            geo["fuw"] = (geo["lonw"] + np.pi) * (w / (2 * np.pi)) - 0.5 + 2 * w
        v0, v1, dv, vn = geo["rows"]
        fu = geo["fuw"] - theta * (w / (2 * np.pi))
        texel_km = 2 * math.pi * R_EARTH / w
        if geo["km_per_px"] >= 1.5 * texel_km:
            u = (fu + 0.5).astype(np.int32) % w
            px = packed[u * h + vn].view(np.uint8).reshape(-1, 4).astype(np.float32)
            return px[:, :3], px[:, 3] * (1 / 255.0)
        ui = fu.astype(np.int32)
        du = (fu - ui).astype(np.float32)[:, None]
        u0 = (ui % w) * h
        u1 = ((ui + 1) % w) * h

        def tap(idx):
            return packed[idx].view(np.uint8).reshape(-1, 4).astype(np.float32)
        px = ((tap(u0 + v0) * (1 - du) + tap(u1 + v0) * du) * (1 - dv)
              + (tap(u0 + v1) * (1 - du) + tap(u1 + v1) * du) * dv)
        return px[:, :3], px[:, 3] * (1 / 255.0)

    def _lighting(self, geo, sun_dir):
        """Sun-dependent terms so that color = base * A + B + water * C GLINT
        (C per pixel, nonzero only in the glint), plus the halo canvas and alpha.
        Everything per pixel is a scalar or one (N, 3) x (3, 3) product: this runs
        on every frame while the camera moves."""
        n = geo["n"]
        ndl = n @ sun_dir
        day = smoothstep(-0.12, 0.10, ndl)
        light = 0.085 + 0.915 * day * (0.28 + 0.72 * np.clip(ndl, 0.0, 1.0) ** 0.85)
        twilight = np.exp(-((ndl - 0.015) / 0.05) ** 2)
        # sun glint (Blinn-Phong); negligible outside n.h > 0.85, so only the
        # small glint region is evaluated. With view and Sun unit vectors,
        # n.h = (n.v + n.s) / |v + s| and |v + s|^2 = 2 + 2 v.s
        spec = np.zeros(len(n), np.float32)
        near = np.flatnonzero(ndl > 0.0)
        if len(near):
            vs = geo["view"][near] @ sun_dir
            nh_ = (geo["ndv"][near] + ndl[near]) / np.sqrt(np.maximum(2.0 + 2.0 * vs, 1e-12))
            sel = nh_ > 0.85
            idx, x = near[sel], nh_[sel]
            spec[idx] = (0.8 * x ** 600 + 0.07 * x ** 40) * np.clip(ndl[idx] * 5.0, 0.0, 1.0)
        rim = geo["rim"]
        keep = 1.0 - 0.3 * rim
        A = (light * keep).astype(np.float32)
        # night tint, twilight and rim glow as weights of their three colors
        tw = twilight * (0.10 * keep + 0.36 * rim)
        w = np.stack([(1.0 - day) * keep, tw, (0.12 + 0.88 * day) * rim * 0.8], axis=1)
        B = (w.astype(np.float32) @ _TINTS).astype(np.float32)
        C = (spec * keep).astype(np.float32)

        nw, nh = geo["size"]
        canvas = np.zeros((nw * nh, 3), np.uint8)
        alpha = geo["alpha"].copy().reshape(-1)
        if len(geo["qn"]):
            mu = geo["qn"] @ sun_dir
            hl = np.clip(0.15 + 0.95 * (mu + 0.25), 0.04, 1.0)
            dusk = np.exp(-(mu / 0.22) ** 2)[:, None]
            color = (HALO_COLOR * (1 - 0.55 * dusk) + TWILIGHT * 0.55 * dusk) * hl[:, None]
            canvas[geo["halo_idx"]] = np.clip(color, 0, 255).astype(np.uint8)
            alpha[geo["halo_idx"]] = np.clip(235.0 * geo["g"] * hl, 0, 255).astype(np.uint8)
        return A, B, C, canvas, alpha.reshape(nw, nh)

    def _compose(self, geo):
        """Combine the cached texture sample and lighting into the output surface."""
        _, base, water = self._tex
        _, A, B, C, canvas, alpha = self._light
        nw, nh = geo["size"]
        rgb = canvas.copy()
        if len(base):
            col = base * A[:, None] + B
            g = np.flatnonzero(C)
            if len(g):
                col[g] += GLINT * (C[g] * water[g])[:, None]
            rgb[geo["hit_idx"]] = np.clip(col, 0, 255, out=col).astype(np.uint8)
        surf = pygame.Surface((nw, nh), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(surf)
        px[...] = rgb.reshape(nw, nh, 3)
        del px
        pa = pygame.surfarray.pixels_alpha(surf)
        pa[...] = alpha
        del pa
        if (nw, nh) != geo["out"]:
            surf = pygame.transform.smoothscale(surf, geo["out"])
        return surf
