function orbit = orbitGeo(lon)
%ORBITGEO Geostationary slot over east longitude LON (deg).
%   satflight.orbitGeo(-75)
arguments
    lon (1,1) double
end
orbit = struct("type", "geo", "lon", lon);
end
