classdef ExportTest < matlab.unittest.TestCase
    %EXPORTTEST Live scripts written by Scenarios > Export MATLAB run in MATLAB and
    %   reproduce the simulator: each case is exported with
    %   python -m satflight.matlab_export, run here, and its own cross-check section
    %   (the simulator run on the same scenario) must agree to within 10 m for every
    %   satellite still in orbit in both runs, and on which ones re-entered.
    %   From the repository root:
    %     matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests/ExportTest.m'))"

    properties
        Root
        Tmp
    end

    properties (Constant)
        Tolerance = 0.010      % km
    end

    methods (TestClassSetup)
        function setup(tc)
            addpath(fileparts(fileparts(mfilename("fullpath"))));
            tc.Root = string(satflight.repoRoot());
            fixture = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            tc.Tmp = string(fixture.Folder);
            set(groot, "defaultFigureVisible", "off");
            tc.addTeardown(@() set(groot, "defaultFigureVisible", "remove"));
        end
    end

    methods (Test)
        function hohmannFromTheStart(tc)
            % two burns (impulse, circularize) under numerical integration
            tc.check(tc.export(fullfile(tc.Root, "scenarios", "hohmann_to_geo.json"), ...
                "hohmann", ["--duration", "6h", "--step", "120"]));
        end

        function everyKindOfBurn(tc)
            % finite, plane change, RSW and ECI impulses, circularize and match velocity,
            % node / apsis timings, J3 + C22 + drag, under each of the three propagators
            for propagator = ["cowell", "kepler", "j2mean"]
                file = satflight.saveScenario(tc.everyBurn(propagator), ...
                    fullfile(tc.Tmp, "burns_" + propagator + ".json"));
                tc.check(tc.export(file, "burns_" + propagator, ...
                    ["--duration", "3h", "--step", "60"]));
            end
        end

        function reEntry(tc)
            % the balloon re-enters during the run in both
            D = tc.check(tc.export(fullfile(tc.Root, "scenarios", "drag_decay.json"), ...
                "decay", ["--duration", "12h", "--step", "300"]));
            tc.verifyEqual(D.status_matlab(D.name == "Balloon"), "re-entered");
        end

        function fromTheCurrentMoment(tc)
            % exported 3 h into the run, between whole seconds
            tc.check(tc.export(fullfile(tc.Root, "scenarios", "default.json"), "later", ...
                ["--at", "10817.25", "--duration", "2h", "--step", "120"]));
        end

        function midBurnSnapshot(tc)
            % exported while the finite burn is under way: the rest of it carries over
            file = satflight.saveScenario(tc.everyBurn("cowell"), fullfile(tc.Tmp, "mid.json"));
            tc.check(tc.export(file, "midburn", ["--at", "750", "--duration", "2h"]));
        end
    end

    methods
        function script = export(tc, scenario, name, args)
            script = fullfile(tc.Tmp, name + ".m");
            runPython(["-m", "satflight.matlab_export", scenario, "--out", script, args]);
            tc.assertTrue(matlab.desktop.editor.EditorUtils.isLiveCodeFile(char(script)), ...
                "MATLAB does not take the export for a Live Script");
        end

        function D = check(tc, script)
            close all
            out = runScript(script);
            D = out.differences;
            tc.verifyNotEmpty(D);
            tc.verifyEqual(D.status_matlab, D.status_simulator, script);
            bad = D(D.max_diff_km > tc.Tolerance, :);
            tc.verifyEmpty(bad, script + ": " + formattedDisplayText(bad));
            tc.verifyEqual(height(out.T), numel(out.sim.names) * numel(unique(out.T.t_s)));
        end
    end

    methods (Static)
        function sc = everyBurn(propagator)
            sc = satflight.newScenario("Every burn " + propagator, Propagator=propagator, ...
                J2=true, J3=true, C22=true, Drag=true, ...
                Epoch=datetime(2026, 9, 29, 12, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Burner", satflight.orbitElements(Altitude=500, ...
                I=40, Raan=10, Nu=5), Mass=800, Area=4);
            sc = satflight.addSatellite(sc, "Tilter", satflight.orbitElements(PerigeeAlt=400, ...
                ApogeeAlt=2000, I=60, Raan=50, Argp=30, Nu=100));
            sc = satflight.addSatellite(sc, "Chase", satflight.orbitElements(Altitude=480, ...
                I=40, Raan=10, Nu=3));
            sc = satflight.addStation(sc, "Kiruna", 67.86, 20.96);
            sc = satflight.addManeuver(sc, "Burner", "finite", Timing="delay", Delay=300, ...
                Dv=[1 0 0.2], Thrust=400, Isp=310, Duration=900);
            sc = satflight.addManeuver(sc, "Tilter", "plane_change", ...
                Timing="ascending_node", DeltaI=2);
            sc = satflight.addManeuver(sc, "Tilter", "impulse", Timing="apoapsis", ...
                Dv=[0.05 0.01 -0.02], Frame="RSW");
            sc = satflight.addManeuver(sc, "Chase", "circularize", Timing="periapsis");
            sc = satflight.addManeuver(sc, "Chase", "match_velocity", Timing="delay", ...
                Delay=5000, Target="Tilter");
            sc = satflight.addManeuver(sc, "Burner", "impulse", Timing="delay", Delay=9000, ...
                Dv=[0.001 0.002 0.003], Frame="ECI");
        end
    end
end

function out = runScript(file)
% runs the exported script here; its variables land in this workspace
run(file);
out = struct("T", T, "differences", differences, "sim", sim);
end

function runPython(args)
exe = string(getenv("SATFLIGHT_PYTHON"));
if exe == ""
    exe = "python";
end
old = getenv("PYTHONPATH");
restore = onCleanup(@() setenv("PYTHONPATH", old));
setenv("PYTHONPATH", satflight.repoRoot() + string(pathsep) + old);
[status, out] = system(strjoin(compose('"%s"', [exe, string(args)]), " "));
if status ~= 0
    error("satflight:python", "exit status %d:\n%s", status, out);
end
end
