"""Perspective orbit camera and the projection/occlusion math used by the
renderer. World units are kilometers with +Z the Earth's rotation axis."""

from __future__ import annotations

import math

import numpy as np

from ..constants import R_EARTH


class Camera:
    """Orbits ``target`` at ``distance`` km from direction (``yaw``, ``pitch``);
    call :meth:`update` after changing them to rebuild the view basis."""

    MIN_DIST = R_EARTH * 1.02
    MAX_DIST = 2.5e6
    FLOOR = R_EARTH * 1.002      # the camera never goes below ~13 km

    def __init__(self, width: int, height: int, fov_deg: float = 45.0):
        self.yaw = math.radians(-60.0)
        self.pitch = math.radians(22.0)
        self.distance = 42000.0
        self.target = np.zeros(3)
        self.fov = math.radians(fov_deg)
        self.near = 1.0
        self.resize(width, height)
        self.update()

    def resize(self, width: int, height: int):
        """Adapt the projection to a new window size."""
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.focal = (height / 2.0) / math.tan(self.fov / 2.0)

    # --- controls ---------------------------------------------------------------
    def rotate(self, dyaw: float, dpitch: float):
        """Turn the view (radians); pitch stops short of the poles."""
        self.yaw = (self.yaw + dyaw) % (2 * math.pi)
        self.pitch = max(-1.55, min(1.55, self.pitch + dpitch))

    def zoom(self, factor: float, min_dist: float | None = None):
        """Scale the distance, clamped to [``min_dist`` or ``MIN_DIST``, ``MAX_DIST``]."""
        lo = min_dist if min_dist is not None else self.MIN_DIST
        self.distance = max(lo, min(self.MAX_DIST, self.distance * factor))

    def state_key(self):
        """Hashable summary of the view, for render caches."""
        return (round(self.yaw, 6), round(self.pitch, 6), round(self.distance, 3),
                tuple(np.round(self.target, 3)), self.width, self.height)

    # --- basis ----------------------------------------------------------------------
    def update(self):
        """Recompute position and the right/up/forward basis from the controls."""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        offset = self._above_ground(np.array([cp * cy, cp * sy, sp])) * self.distance
        self.position = self.target + offset
        f = -offset / np.linalg.norm(offset)
        r = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(r) < 1e-9:
            r = np.array([1.0, 0.0, 0.0])
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        self.right, self.up, self.forward = r, u, f
        self.basis = np.stack([r, u, f])       # rows: camera x, y, z (depth)
        self.near = max(0.5, self.distance * 1e-4)

    def _above_ground(self, d: np.ndarray) -> np.ndarray:
        """Unit view offset ``d``, tilted toward the target's local vertical
        just enough that the camera is above the ground and sees the target
        past the Earth. Following a rocket on its pad would otherwise put the
        camera inside the planet, or behind it, for most viewing directions."""
        t = self.target
        tn = float(np.linalg.norm(t))
        if tn < 1.0:
            return d
        rb = min(R_EARTH, tn * 0.998)          # sphere the sight line must clear

        def clear(pos):
            if np.linalg.norm(pos) < self.FLOOR:
                return False
            seg = t - pos                          # camera -> target
            a = float(seg @ seg)
            b = 2.0 * float(seg @ pos)
            c = float(pos @ pos) - rb * rb
            disc = b * b - 4.0 * a * c
            if disc <= 0:
                return True
            s1 = (-b - math.sqrt(disc)) / (2.0 * a)
            return not 0.0 < s1 < 1.0

        if clear(t + d * self.distance):
            return d
        up = t / tn

        def blend(a):
            v = (1.0 - a) * d + a * up
            n = np.linalg.norm(v)
            return v / n if n > 1e-9 else up

        if not clear(t + up * self.distance):
            return up
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if clear(t + blend(mid) * self.distance):
                hi = mid
            else:
                lo = mid
        return blend(hi)

    # --- projection -----------------------------------------------------------------
    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        """World points -> camera coordinates (x right, y up, z depth)."""
        return (np.asarray(pts, dtype=float) - self.position) @ self.basis.T

    def project(self, pts: np.ndarray):
        """World points (..., 3) -> screen x, y and depth arrays; points
        behind the near plane get depth <= near and must be discarded."""
        c = self.to_camera(pts)
        z = c[..., 2]
        zs = np.where(z > self.near, z, np.nan)
        sx = self.cx + self.focal * c[..., 0] / zs
        sy = self.cy - self.focal * c[..., 1] / zs
        return sx, sy, z

    def project_dirs(self, dirs: np.ndarray):
        """Directions at infinity (stars, the Sun) -> screen coordinates."""
        c = np.asarray(dirs) @ self.basis.T
        z = c[..., 2]
        zs = np.where(z > 1e-6, z, np.nan)
        return self.cx + self.focal * c[..., 0] / zs, self.cy - self.focal * c[..., 1] / zs, z

    def pixel_scale(self, depth) -> np.ndarray:
        """Screen pixels per km at a given depth."""
        return self.focal / np.maximum(depth, self.near)

    def ray_dirs(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Unit world-space view rays through screen pixels (broadcasting)."""
        dx = (xs - self.cx) / self.focal
        dy = -(ys - self.cy) / self.focal
        d = (self.forward[None, None, :] + dx[..., None] * self.right[None, None, :]
             + dy[..., None] * self.up[None, None, :])
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def hidden_by_sphere(self, pts: np.ndarray, radius: float = R_EARTH) -> np.ndarray:
        """True where the sight line camera -> point is blocked by the sphere."""
        c = self.position
        d = np.asarray(pts, dtype=float) - c
        a = np.sum(d * d, axis=-1)
        b = 2.0 * (d @ c)
        cc = float(c @ c) - radius * radius
        if cc <= 0:
            return np.ones(d.shape[:-1], dtype=bool)
        disc = b * b - 4.0 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            t1 = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
        return (disc > 0) & (t1 > 0) & (t1 < 1.0)

    def screen_radius(self, center: np.ndarray, radius_km: float) -> float:
        """Apparent radius (px) of a sphere at ``center``; 0 if behind the camera."""
        depth = float(self.to_camera(center)[2])
        if depth <= self.near:
            return 0.0
        return radius_km * self.focal / depth
