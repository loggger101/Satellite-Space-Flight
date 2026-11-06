function sc = addLaunch(sc, name, opts)
%ADDLAUNCH Add a rocket that lifts off from a pad during the run.
%   Vehicle: a preset name such as "Falcon 9 (approx.)" or "Electron (approx.)"
%   (the simulator lists them if you give an unknown one), or a struct with
%   name, stages {struct(name, thrust kN, isp_vac, isp_sl, propellant kg, dry kg)}
%   and the vehicle's other fields.
%
%   Where: Site (name), Lat, Lon (deg), Alt (km).
%   When (Timing): "now", "delay" (Delay s), "absolute" (T0 s after the epoch),
%   "raan" (into node Raan, deg) or "plane" (into the plane of satellite Target).
%   Guidance: "orbit" (to PerigeeAlt, ApogeeAlt km at Inclination deg, heading
%   Direction "north" or "south") or "open" (gravity turn on Azimuth deg with
%   VerticalTime s and Kick deg).
%   Payload: PayloadMass (kg), PayloadArea (m^2), PayloadCd, Circularize
%   (at first apogee), TrackStages (spent stages become objects), Color.
%
%   sc = satflight.addLaunch(sc, "Demo-1", Vehicle="Electron (approx.)", ...
%       Lat=-39.26, Lon=177.86, PerigeeAlt=520, ApogeeAlt=520, Inclination=97.5, ...
%       Direction="south", Timing="delay", Delay=600, PayloadMass=150);
arguments
    sc (1,1) struct
    name (1,1) string
    opts.Vehicle
    opts.Site (1,1) string
    opts.Lat (1,1) double
    opts.Lon (1,1) double
    opts.Alt (1,1) double
    opts.Timing (1,1) string
    opts.Delay (1,1) double
    opts.T0 (1,1) double
    opts.Guidance (1,1) string
    opts.PerigeeAlt (1,1) double
    opts.ApogeeAlt (1,1) double
    opts.Inclination (1,1) double
    opts.Direction (1,1) string
    opts.Raan (1,1) double
    opts.Target (1,1) string
    opts.Azimuth (1,1) double
    opts.VerticalTime (1,1) double
    opts.Kick (1,1) double
    opts.PayloadMass (1,1) double
    opts.PayloadArea (1,1) double
    opts.PayloadCd (1,1) double
    opts.Circularize (1,1) logical
    opts.TrackStages (1,1) logical
    opts.Color (1,3) double
end
if isfield(opts, "Vehicle") && ~(isStringScalar(opts.Vehicle) || ischar(opts.Vehicle) ...
        || (isstruct(opts.Vehicle) && isscalar(opts.Vehicle)))
    error("satflight:badLaunch", "Launch %s: Vehicle must be a preset name or a struct.", name);
end
s = struct("name", char(name));
s = copyOptions(s, opts, ["Vehicle", "vehicle"; "Site", "site"; "Lat", "lat"; "Lon", "lon"; ...
    "Alt", "alt"; "Timing", "timing"; "Delay", "delay"; "T0", "t0"; "Guidance", "guidance"; ...
    "PerigeeAlt", "perigee_alt"; "ApogeeAlt", "apogee_alt"; "Inclination", "inclination"; ...
    "Direction", "direction"; "Raan", "raan"; "Target", "target"; "Azimuth", "azimuth"; ...
    "VerticalTime", "vertical_time"; "Kick", "kick"; "PayloadMass", "payload_mass"; ...
    "PayloadArea", "payload_area"; "PayloadCd", "payload_cd"; "Circularize", "circularize"; ...
    "TrackStages", "track_stages"; "Color", "color"]);
if isfield(s, "vehicle") && isstruct(s.vehicle) && isfield(s.vehicle, "stages")
    s.vehicle.stages = asCellRow(s.vehicle.stages, "launch " + name + " stages");
end
sc = appendTo(sc, "launches", s);
end
