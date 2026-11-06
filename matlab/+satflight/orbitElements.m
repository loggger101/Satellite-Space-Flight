function orbit = orbitElements(opts)
%ORBITELEMENTS Orbit from classical elements (km and degrees).
%   Give the size exactly one way:
%     A                      semi-major axis (km), with E for the eccentricity
%     Altitude               circular orbit this high above the equator (km)
%     PerigeeAlt, ApogeeAlt  altitudes of the low and high points (km)
%   and the orientation:
%     I      inclination (deg), or "sso" for the sun-synchronous one
%     Raan   right ascension of the ascending node (deg)
%     Argp   argument of perigee (deg)
%     Nu     true anomaly (deg), or M for the mean anomaly - not both
%
%   Examples:
%     satflight.orbitElements(Altitude=550, I="sso", Raan=30)
%     satflight.orbitElements(PerigeeAlt=600, ApogeeAlt=39700, I=63.4, Argp=270)
arguments
    opts.A (1,1) double
    opts.E (1,1) double
    opts.Altitude (1,1) double
    opts.PerigeeAlt (1,1) double
    opts.ApogeeAlt (1,1) double
    opts.I
    opts.Raan (1,1) double
    opts.Argp (1,1) double
    opts.Nu (1,1) double
    opts.M (1,1) double
end
if isfield(opts, "I") && ~(isnumeric(opts.I) && isscalar(opts.I)) ...
        && ~(isStringScalar(opts.I) || ischar(opts.I))
    error("satflight:badOrbit", "I must be a number of degrees or ""sso"".");
end
orbit = struct("type", "elements");
orbit = copyOptions(orbit, opts, ["A", "a"; "E", "e"; "Altitude", "altitude"; ...
    "PerigeeAlt", "perigee_alt"; "ApogeeAlt", "apogee_alt"; "I", "i"; "Raan", "raan"; ...
    "Argp", "argp"; "Nu", "nu"; "M", "M"]);
end
