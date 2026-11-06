classdef BundledExportTest < ExportCase
    %BUNDLEDEXPORTTEST Every scenario in scenarios/, exported to a live script from its
    %   start and from part-way into the run (between whole seconds), runs in MATLAB
    %   and reproduces the simulator (held to it as ExportCase describes). A scenario
    %   added to scenarios/ is picked up here by itself. From the repository root:
    %     matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests/BundledExportTest.m'))"

    properties (TestParameter)
        scenario
        start = struct("fromTheStart", "", "anHourIn", "3600.5")
    end

    methods (TestParameterDefinition, Static)
        function scenario = bundledScenarios()
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            files = dir(fullfile(root, "scenarios", "*.json"));
            if isempty(files)
                error("BundledExportTest:none", "No scenarios in %s.", ...
                    fullfile(root, "scenarios"));
            end
            names = erase(string({files.name}), ".json");
            scenario = cell2struct(num2cell(names), matlab.lang.makeValidName(names), 2);
        end
    end

    methods (Test)
        function bundled(tc, scenario, start)
            args = ["--duration", "3h", "--step", "120"];
            name = scenario + "_start";
            if start ~= ""
                args = [args, "--at", start];
                name = scenario + "_later";
            end
            tc.check(tc.export(fullfile(tc.Root, "scenarios", scenario + ".json"), ...
                matlab.lang.makeValidName(name), args));
        end
    end
end
