function sc = addConstellation(sc, name, opts)
%ADDCONSTELLATION Add a Walker constellation (inclination: total/planes/phasing).
%   Options: Altitude (km), Inclination (deg), Total, Planes, Phasing,
%   Pattern ("delta": nodes over 360 deg, or "star": over 180 deg),
%   Raan0 (deg, first plane's node), Color ([r g b]), Mass, Area, Cd, DryMass
%   (kg each: burns stop there; unset, no limit).
%
%   sc = satflight.addConstellation(sc, "Shell A", Altitude=1200, ...
%       Inclination=55, Total=24, Planes=6, Phasing=1);
arguments
    sc (1,1) struct
    name (1,1) string
    opts.Altitude (1,1) double
    opts.Inclination (1,1) double
    opts.Total (1,1) double
    opts.Planes (1,1) double
    opts.Phasing (1,1) double
    opts.Pattern (1,1) string
    opts.Color (1,3) double
    opts.Mass (1,1) double
    opts.Area (1,1) double
    opts.Cd (1,1) double
    opts.Raan0 (1,1) double
    opts.DryMass (1,1) double
end
c = struct("name", char(name));
c = copyOptions(c, opts, ["Altitude", "altitude"; "Inclination", "inclination"; ...
    "Total", "total"; "Planes", "planes"; "Phasing", "phasing"; "Pattern", "pattern"; ...
    "Color", "color"; "Mass", "mass"; "Area", "area"; "Cd", "cd"; "Raan0", "raan0"; ...
    "DryMass", "dry_mass"]);
sc = appendTo(sc, "constellations", c);
end
