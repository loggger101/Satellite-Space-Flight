function path = scenarioFile(scenario)
%SCENARIOFILE A file path for SCENARIO: the path itself, or a struct saved to a temp file.
%   The temp file is deleted when MATLAB's temp folder is cleaned, not here,
%   so the simulator's error messages can still be followed back to it.
if isstruct(scenario)
    path = satflight.saveScenario(scenario, string(tempname) + ".json");
elseif (isStringScalar(scenario) || ischar(scenario))
    path = string(scenario);
    if ~isfile(path)
        error("satflight:noFile", "No scenario file %s.", path);
    end
else
    error("satflight:badScenario", "Give a scenario struct or a file path, not a %s.", ...
        class(scenario));
end
end
