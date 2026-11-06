function [T, report] = runScenario(scenario, opts)
%RUNSCENARIO Propagate a scenario without graphics and read the ephemeris.
%   [T, report] = satflight.runScenario(scenario) runs the simulator's batch
%   mode (python -m satflight.batch) on a scenario struct or file and returns
%   the ephemeris as a table (see satflight.readRun) and the printed summary.
%
%   Options:
%     Duration    seconds, or text such as "90m", "12h", "3d" (default "1d")
%     Step        output interval in seconds or as text (default 60)
%     Out         CSV file to keep (default: a temporary file)
%     Propagator  "cowell", "kepler" or "j2mean" in place of the scenario's
%
%   [T, report] = satflight.runScenario("scenarios/hohmann_to_geo.json", Duration="8h");
arguments
    scenario
    opts.Duration = "1d"
    opts.Step = 60
    opts.Out (1,1) string = string(tempname) + ".csv"
    opts.Propagator (1,1) string
end
path = scenarioFile(scenario);
args = ["-m", "satflight.batch", path, "--duration", durationText(opts.Duration, "Duration"), ...
    "--step", durationText(opts.Step, "Step"), "--out", opts.Out];
if isfield(opts, "Propagator")
    args = [args, "--propagator", opts.Propagator];
end
report = string(runPython(args));
T = satflight.readRun(opts.Out);
end

function text = durationText(x, name)
if isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x)
    text = string(sprintf("%.17g", x));
elseif isStringScalar(x) || ischar(x)
    text = string(x);       % the simulator parses and checks it
else
    error("satflight:badDuration", "%s must be a number of seconds or text such as ""12h"".", name);
end
end
