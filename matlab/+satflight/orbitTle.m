function orbit = orbitTle(line1, line2)
%ORBITTLE Orbit from a two-line element set.
%   satflight.orbitTle("1 25544U 98067A ...", "2 25544  51.6416 ...")
arguments
    line1 (1,1) string
    line2 (1,1) string
end
orbit = struct("type", "tle", "line1", char(line1), "line2", char(line2));
end
