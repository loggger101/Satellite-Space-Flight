function report = checkScenario(scenario)
%CHECKSCENARIO Have the simulator load a scenario and set it up; error if it refuses.
%   report = satflight.checkScenario(path) or satflight.checkScenario(sc)
%   reads the scenario with the simulator's strict loader and builds every
%   initial state (zero-length batch run). Anything it cannot use - an unknown
%   key or option, a value out of range, an orbit inside the Earth - raises an
%   error with the simulator's reason. On success it returns the summary table
%   of satellites at the epoch.
arguments
    scenario
end
path = scenarioFile(scenario);
report = string(runPython(["-m", "satflight.batch", path, "--duration", "0"]));
end
