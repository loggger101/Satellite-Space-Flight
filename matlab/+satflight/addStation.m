function sc = addStation(sc, name, lat, lon, opts)
%ADDSTATION Add a ground station at geodetic LAT, LON (deg).
%   Options: Alt (km), MinEl (deg; passes count above this elevation,
%   default 10), Color ([r g b]).
%
%   sc = satflight.addStation(sc, "Svalbard", 78.23, 15.39, MinEl=5);
arguments
    sc (1,1) struct
    name (1,1) string
    lat (1,1) double
    lon (1,1) double
    opts.Alt (1,1) double
    opts.MinEl (1,1) double
    opts.Color (1,3) double
end
g = struct("name", char(name), "lat", lat, "lon", lon);
g = copyOptions(g, opts, ["Alt", "alt"; "MinEl", "min_el"; "Color", "color"]);
sc = appendTo(sc, "stations", g);
end
