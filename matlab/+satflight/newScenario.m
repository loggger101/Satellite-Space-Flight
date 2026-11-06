function sc = newScenario(name, opts)
%NEWSCENARIO Start a scenario: name, epoch, force model and propagator.
%   sc = satflight.newScenario("My run") starts an empty scenario. Add to it
%   with satflight.addSatellite, addConstellation, addStation, addManeuver and
%   addLaunch, then write it with satflight.saveScenario.
%
%   Options (any left out take the simulator's default):
%     Description   text shown in the app
%     Epoch         start time: a datetime with a TimeZone, or an ISO string
%                   such as "2026-09-29T12:00:00Z" (default 2026-01-01T00:00:00Z)
%     Propagator    "cowell" (numerical, default), "kepler" or "j2mean"
%     J2, J3, J4, C22, Drag   force terms (logical; default J2 on, the rest off)
%     DensityScale  multiplies the atmosphere's density (default 1)
%     Integrator    "dopri5" (default), "rk4", "wh" or "leapfrog"
%     Rtol, Atol, HMax, HFixed   integrator tolerances and step limits
%     RecordDt      seconds between recorded samples (default 20)
%     Warp          initial time acceleration in the app (default 60)
%
%   Example:
%     sc = satflight.newScenario("Polar imager", Drag=true, ...
%         Epoch=datetime(2026,9,29,12,0,0,TimeZone="UTC"));
arguments
    name (1,1) string
    opts.Description (1,1) string
    opts.Epoch
    opts.Propagator (1,1) string
    opts.J2 (1,1) logical
    opts.J3 (1,1) logical
    opts.J4 (1,1) logical
    opts.C22 (1,1) logical
    opts.Drag (1,1) logical
    opts.DensityScale (1,1) double
    opts.Integrator (1,1) string
    opts.Rtol (1,1) double
    opts.Atol (1,1) double
    opts.HMax (1,1) double
    opts.HFixed (1,1) double
    opts.RecordDt (1,1) double
    opts.Warp (1,1) double
end
sc = struct("name", char(name));
sc = copyOptions(sc, opts, ["Description", "description"]);
if isfield(opts, "Epoch")
    sc.epoch = epochText(opts.Epoch);
end
sc = copyOptions(sc, opts, ["Propagator", "propagator"]);
integrator = copyOptions(struct(), opts, ["Integrator", "method"; "Rtol", "rtol"; ...
    "Atol", "atol"; "HMax", "h_max"; "HFixed", "h_fixed"]);
if ~isempty(fieldnames(integrator))
    sc.integrator = integrator;
end
forces = copyOptions(struct(), opts, ["J2", "j2"; "J3", "j3"; "J4", "j4"; "C22", "c22"; ...
    "Drag", "drag"; "DensityScale", "density_scale"]);
if ~isempty(fieldnames(forces))
    sc.forces = forces;
end
sc = copyOptions(sc, opts, ["RecordDt", "record_dt"; "Warp", "warp"]);
sc.satellites = {};
sc.constellations = {};
sc.stations = {};
sc.maneuvers = {};
sc.launches = {};
end

function text = epochText(epoch)
if isdatetime(epoch) && isscalar(epoch)
    if isempty(epoch.TimeZone)
        error("satflight:badEpoch", ...
            "Epoch %s has no time zone; give one, e.g. datetime(..., TimeZone=""UTC"").", ...
            string(epoch));
    end
    epoch.TimeZone = "UTC";
    if second(epoch) ~= floor(second(epoch))
        error("satflight:badEpoch", ...
            "Epoch %s has a fraction of a second; scenario epochs are whole seconds.", ...
            string(epoch));
    end
    epoch.Format = "yyyy-MM-dd'T'HH:mm:ss'Z'";
    text = char(string(epoch));
elseif (isstring(epoch) && isscalar(epoch)) || ischar(epoch)
    text = char(epoch);     % the simulator parses and checks it
else
    error("satflight:badEpoch", "Epoch must be a datetime or an ISO-8601 string, got a %s.", ...
        class(epoch));
end
end
