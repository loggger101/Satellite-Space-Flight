function orbit = orbitState(r, v)
%ORBITSTATE Orbit from an ECI position r (km) and velocity v (km/s).
%   satflight.orbitState([7000 0 0], [0 7.546 0])
arguments
    r (1,3) double
    v (1,3) double
end
orbit = struct("type", "state", "r", r, "v", v);
end
