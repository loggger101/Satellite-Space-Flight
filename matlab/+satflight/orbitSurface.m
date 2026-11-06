function orbit = orbitSurface(opts)
%ORBITSURFACE Start from a point above the ground with a given velocity.
%   Lat, Lon  geodetic degrees      Alt      km above the ellipsoid
%   Speed     km/s relative to the rotating Earth
%   Azimuth   degrees from north    Fpa      flight-path angle above the horizon (deg)
%
%   satflight.orbitSurface(Lat=28.5, Lon=-80.6, Alt=200, Speed=7.4, Azimuth=90)
arguments
    opts.Lat (1,1) double
    opts.Lon (1,1) double
    opts.Alt (1,1) double
    opts.Speed (1,1) double
    opts.Azimuth (1,1) double
    opts.Fpa (1,1) double
end
orbit = struct("type", "surface");
orbit = copyOptions(orbit, opts, ["Lat", "lat"; "Lon", "lon"; "Alt", "alt"; ...
    "Speed", "speed"; "Azimuth", "azimuth"; "Fpa", "fpa"]);
end
