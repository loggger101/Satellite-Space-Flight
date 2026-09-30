"""2-D equirectangular ground-track map with day/night shading, station
markers, sub-satellite points and the selected satellite's coverage."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..analysis import coverage_half_angle, footprint
from ..ephemeris import subsolar_point
from ..frames import ecef_to_geodetic, eci_to_ecef
from ..simulation import ACTIVE
from . import theme
from .earth import smoothstep


class GroundTrackView:
    """The map panel (key M). The background and night shading are cached surfaces,
    rebuilt only when the size or the Sun moves."""

    def __init__(self, earth):
        self.earth = earth              # EarthRenderer, for texture and coastlines
        self._bg_key = None
        self._bg = None
        self._night_key = None
        self._night = None
        self.rect = pygame.Rect(0, 0, 0, 0)

    def _background(self, size):
        """Texture (or a latitude gradient), graticule and coastlines at ``size``."""
        if self._bg_key == size:
            return self._bg
        w, h = size
        if self.earth.has_image:
            tex = pygame.surfarray.make_surface(self.earth.texture)
            bg = pygame.transform.smoothscale(tex, size)
            shade = pygame.Surface(size, pygame.SRCALPHA)
            shade.fill((0, 0, 0, 60))
            bg.blit(shade, (0, 0))
        else:
            bg = pygame.Surface(size)
            k = np.abs(np.sin(np.radians(np.linspace(90, -90, h))))
            col = np.stack([18 + 8 * k, 50 + 30 * k, 110 + 40 * k], axis=1)
            arr = np.repeat(col[None, :, :], w, axis=0).astype(np.uint8)
            pygame.surfarray.blit_array(bg, arr)
        for d in range(-180, 181, 30):
            x = int((d + 180) / 360 * (w - 1))
            pygame.draw.line(bg, (60, 90, 130) if d else (110, 170, 220), (x, 0), (x, h))
        for d in range(-90, 91, 30):
            y = int((90 - d) / 180 * (h - 1))
            pygame.draw.line(bg, (60, 90, 130) if d else (110, 170, 220), (0, y), (w, y))
        for cl in self.earth.coastlines:
            lat = np.degrees(np.arcsin(np.clip(cl[:, 2], -1, 1)))
            lon = np.degrees(np.arctan2(cl[:, 1], cl[:, 0]))
            coast = (95, 125, 100) if self.earth.has_image else (160, 210, 160)
            self._polyline(bg, coast, lat, lon, (0, 0, w, h))
        self._bg_key, self._bg = size, bg
        return bg

    def _night_overlay(self, size, lat_s, lon_s):
        """Translucent night side for the sub-solar point (rad), soft at the terminator."""
        key = (size, round(lat_s, 3), round(lon_s, 3))
        if key == self._night_key:
            return self._night
        gw, gh = 240, 120
        lon = np.radians(np.linspace(-180, 180, gw))[:, None]
        lat = np.radians(np.linspace(90, -90, gh))[None, :]
        cosz = np.sin(lat) * math.sin(lat_s) + np.cos(lat) * math.cos(lat_s) * np.cos(lon - lon_s)
        night = 1.0 - smoothstep(-0.10, 0.05, cosz)
        small = pygame.Surface((gw, gh), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(small)
        px[...] = np.array([2, 4, 16], np.uint8)
        del px
        pa = pygame.surfarray.pixels_alpha(small)
        pa[...] = (night * 150).astype(np.uint8)
        del pa
        self._night = pygame.transform.smoothscale(small, size)
        self._night_key = key
        return self._night

    @staticmethod
    def _xy(lat_deg, lon_deg, r):
        """Map pixel coordinates of lat/lon (deg) inside rectangle ``r`` (x, y, w, h)."""
        x = r[0] + (np.asarray(lon_deg) + 180.0) / 360.0 * (r[2] - 1)
        y = r[1] + (90.0 - np.asarray(lat_deg)) / 180.0 * (r[3] - 1)
        return x, y

    def _polyline(self, surf, color, lat, lon, r, width=1):
        """Draw a lat/lon track, breaking it where it wraps across the date line."""
        if len(lat) < 2:
            return
        x, y = self._xy(lat, lon, r)
        breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
        for seg in np.split(np.arange(len(lat)), breaks):
            if len(seg) >= 2:
                pts = np.stack([x[seg], y[seg]], 1).tolist()
                if width == 1:
                    pygame.draw.aalines(surf, color, False, pts)
                else:
                    pygame.draw.lines(surf, color, False, pts, width)

    def draw(self, surf, rect: pygame.Rect, app):
        sim, fonts = app.sim, app.fonts
        self.rect = rect
        theme.panel(surf, rect)
        fonts.draw(surf, "GROUND TRACK", (rect.x + 10, rect.y + 5), theme.FAINT, fonts.small)
        mh = rect.h - 26
        mw = min(rect.w - 16, 2 * mh)
        mh = mw // 2
        mr = pygame.Rect(rect.centerx - mw // 2, rect.y + 22, mw, mh)
        surf.blit(self._background(mr.size), mr.topleft)
        jd = sim.jd()
        theta = sim.gmst()
        lat_s, lon_s = subsolar_point(jd, theta)
        surf.blit(self._night_overlay(mr.size, float(lat_s), float(lon_s)), mr.topleft)
        r = (mr.x, mr.y, mr.w, mr.h)
        clip = surf.get_clip()
        surf.set_clip(mr)

        # sub-solar point
        sx, sy = self._xy(math.degrees(lat_s), math.degrees(lon_s), r)
        pygame.draw.circle(surf, theme.SUN, (int(sx), int(sy)), 6)

        # stations
        for st in sim.stations:
            x, y = self._xy(st.lat, st.lon, r)
            pygame.draw.rect(surf, (120, 255, 160), (int(x) - 3, int(y) - 3, 7, 7), 1)
            if mw > 500:
                fonts.draw(surf, st.name, (int(x) + 6, int(y) - 6), (120, 220, 150), fonts.small)

        # launch pads with a vehicle waiting or climbing (labels stack when pads share a site)
        taken: list[pygame.Rect] = []
        for a in sim.ascents:
            x, y = self._xy(a.spec.lat, a.spec.lon, r)
            x, y = int(x), int(y)
            pygame.draw.polygon(surf, (255, 170, 90), [(x, y - 7), (x - 5, y + 4), (x + 5, y + 4)])
            if a.phase == "pad" and mw > 500:
                text = f"{a.spec.name} T-{int(a.t0 - sim.t) // 60} min"
                box = pygame.Rect((x + 7, y + 6), fonts.small.size(text))
                while box.collidelist(taken) >= 0:
                    box.y += box.h
                taken.append(box)
                fonts.draw(surf, text, box.topleft, (255, 190, 120), fonts.small)

        # tracks
        times, data = sim.history.series()
        if sim.n <= 30:
            ids = list(range(sim.n))
        else:           # large ensembles: only the selected satellite's track
            ids = [app.selected] if 0 <= app.selected < sim.n else []
        if len(times) > 1 and ids:
            pos = eci_to_ecef(data[ids, :, :3], sim.clock.gmst(times)[None, :])
            lat, lon, _ = ecef_to_geodetic(pos)
            for k, i in enumerate(ids):
                col = sim.sats[i].color
                col = col if i == app.selected else theme.dim(col, 0.7)
                self._polyline(surf, col, np.degrees(lat[k]), np.degrees(lon[k]), r,
                               2 if i == app.selected else 1)

        # current sub-satellite points
        if sim.n:
            re = eci_to_ecef(sim.y[:, :3], theta)
            lat, lon, alt = ecef_to_geodetic(re)
            x, y = self._xy(np.degrees(lat), np.degrees(lon), r)
            for i in range(sim.n):
                if sim.sats[i].status != ACTIVE:
                    continue
                rad = 5 if i == app.selected else 3
                pygame.draw.circle(surf, sim.sats[i].color, (int(x[i]), int(y[i])), rad)
            i = app.selected
            if 0 <= i < sim.n and sim.sats[i].status == ACTIVE:
                lam = coverage_half_angle(float(alt[i]), math.radians(10.0))
                flat, flon = footprint(float(lat[i]), float(lon[i]), lam, 180)
                self._polyline(surf, theme.mix(sim.sats[i].color, (255, 255, 255), 0.4),
                               np.degrees(flat), np.degrees(flon), r)
        surf.set_clip(clip)
        pygame.draw.rect(surf, theme.PANEL_EDGE, mr, 1)
