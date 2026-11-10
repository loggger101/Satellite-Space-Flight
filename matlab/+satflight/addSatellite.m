function sc = addSatellite(sc, name, orbit, opts)
%ADDSATELLITE Add a satellite (or Count copies spread along one orbit).
%   sc = satflight.addSatellite(sc, name, orbit) with an orbit from
%   satflight.orbitElements, orbitState, orbitGeo, orbitSurface or orbitTle.
%
%   Options: Mass (kg), Area (m^2, drag cross-section), Cd (drag coefficient),
%   DryMass (kg left when the propellant is gone: burns stop there; 0 or unset,
%   no limit), Count (copies evenly spaced in mean anomaly), Color ([r g b], 0-255).
%
%   sc = satflight.addSatellite(sc, "Imager", ...
%       satflight.orbitElements(Altitude=550, I="sso"), Mass=150, Area=1.2);
arguments
    sc (1,1) struct
    name (1,1) string
    orbit (1,1) struct
    opts.Mass (1,1) double
    opts.Area (1,1) double
    opts.Cd (1,1) double
    opts.DryMass (1,1) double
    opts.Count (1,1) double
    opts.Color (1,3) double
end
if ~isfield(orbit, "type")
    error("satflight:badOrbit", "Satellite %s: make the orbit with satflight.orbit* functions.", ...
        name);
end
sat = struct("name", char(name), "orbit", orbit);
sat = copyOptions(sat, opts, ["Color", "color"; "Mass", "mass"; "Area", "area"; ...
    "Cd", "cd"; "DryMass", "dry_mass"; "Count", "count"]);
sc = appendTo(sc, "satellites", sat);
end
