"""Perspective orbit camera and the projection/occlusion maths used by the
renderer. World units are kilometres with +Z the Earth's rotation axis."""

from __future__ import annotations

import math

import numpy as np

from ..constants import R_EARTH


class Camera:
    MIN_DIST = R_EARTH * 1.02
    MAX_DIST = 2.5e6

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
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.focal = (height / 2.0) / math.tan(self.fov / 2.0)

    # --- controls ---------------------------------------------------------------
    def rotate(self, dyaw: float, dpitch: float):
        self.yaw = (self.yaw + dyaw) % (2 * math.pi)
        self.pitch = max(-1.55, min(1.55, self.pitch + dpitch))

    def zoom(self, factor: float, min_dist: float | None = None):
        lo = min_dist if min_dist is not None else self.MIN_DIST
        self.distance = max(lo, min(self.MAX_DIST, self.distance * factor))

    def state_key(self):
        return (round(self.yaw, 6), round(self.pitch, 6), round(self.distance, 3),
                tuple(np.round(self.target, 3)), self.width, self.height)

    # --- basis ----------------------------------------------------------------------
    def update(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        offset = np.array([cp * cy, cp * sy, sp]) * self.distance
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

    # --- projection -----------------------------------------------------------------
    def to_camera(self, pts: np.ndarray) -> np.ndarray:
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
        depth = float(self.to_camera(center)[2])
        if depth <= self.near:
            return 0.0
        return radius_km * self.focal / depth
