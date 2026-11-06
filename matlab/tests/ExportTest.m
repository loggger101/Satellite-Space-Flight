classdef ExportTest < matlab.unittest.TestCase
    %EXPORTTEST Live scripts written by Scenarios > Export MATLAB run in MATLAB and
    %   reproduce the simulator: each case is exported with
    %   python -m satflight.matlab_export, run here, and its own cross-check section
    %   (the simulator run on the same scenario) must agree: positions to within 10 m
    %   while both runs have the object in orbit (50 m for one that comes down through
    %   the dense air, where the atmosphere table's small jumps at its layer boundaries
    %   fall at different points of each integrator's steps), the same objects at the
    %   same samples, the same final statuses, and the event logs line for line -
    %   launch and maneuver events to the millisecond, the rest within a step.
    %   From the repository root:
    %     matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests/ExportTest.m'))"

    properties
        Root
        Tmp
    end

    properties (Constant)
        Tolerance = 0.010      % km, objects in orbit
        DenseAir = 0.050       % km, objects that come down through the atmosphere
        Exact = 1e-3           % s, launch and maneuver events
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

        function launchDay(tc)
            % six launches: sounding rocket, Falcon 9, Electron (circularizing), Saturn V,
            % a GTO launch and one into the plane of the ISS; staging, separations, spent
            % stages re-entering
            [~, out] = tc.check(tc.export(fullfile(tc.Root, "scenarios", "launch_day.json"), ...
                "launchday", ["--duration", "2h", "--step", "60"]));
            tc.verifyEqual(numel(out.launches), 6);
            tc.verifyEqual([out.launches.outcome], [out.launches.plannedOutcome]);
            tc.verifyTrue(any(out.events.text == "Scheduled circularize at apogee for " + ...
                "Rocket Lab sat in 00:45:46.3"), "the Electron payload's circularization");
        end

        function launchDayMidAscent(tc)
            % exported with two vehicles in powered flight (one on its second stage under
            % closed-loop guidance) and three on their pads, between whole seconds
            [~, out] = tc.check(tc.export(fullfile(tc.Root, "scenarios", "launch_day.json"), ...
                "midascent", ["--at", "333.7", "--duration", "1h", "--step", "60"]));
            tc.verifyEqual(numel(out.sim.launches), 5);
            tc.verifyEqual(sum(arrayfun(@(L) ~isempty(L.state), out.sim.launches)), 2);
        end

        function launchesPlannedHere(tc)
            % with planLaunches = true MATLAB finds the launch window and searches the
            % pitch kick itself, and comes to the simulator's plan
            sc = satflight.newScenario("Planned here", J2=true, Drag=true, ...
                Epoch=datetime(2026, 9, 29, 12, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Target", satflight.orbitElements(Altitude=550, ...
                I=53, Raan=120));
            sc = satflight.addLaunch(sc, "Into the plane", Vehicle="Falcon 9 (approx.)", ...
                Lat=28.562, Lon=-80.577, Timing="plane", Target="Target", PerigeeAlt=400, ...
                ApogeeAlt=400);
            sc = satflight.addLaunch(sc, "To a node", Vehicle="Electron (approx.)", ...
                Lat=-39.262, Lon=177.865, Timing="raan", Raan=200, Inclination=97.5, ...
                Direction="south", PerigeeAlt=500, ApogeeAlt=520, PayloadMass=150);
            sc = satflight.addLaunch(sc, "Sounder", Vehicle="Sounding rocket", Lat=69.29, ...
                Lon=16.02, Guidance="open", Azimuth=0, Timing="delay", Delay=60);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "planned.json"));
            script = tc.export(file, "planned", ["--duration", "1h", "--step", "120"]);
            text = readlines(script);
            k = find(text == "planLaunches = false;   % true: plan the launches on their pads " + ...
                "here (window, pitch kick), e.g. after editing one");
            tc.assertNumElements(k, 1);
            text(k) = "planLaunches = true;";
            here = fullfile(tc.Tmp, "planned_here.m");
            writelines(text, here);
            [~, out] = tc.check(here);
            plan = [out.sim.launches.plan];
            tc.verifyEqual([out.launches.liftoff_s], [plan.t0], AbsTol=1e-6);
            tc.verifyEqual([out.launches.kick_deg], [plan.kick], AbsTol=1e-9);
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

        function [D, out] = check(tc, script)
            close all
            out = runScript(script);
            D = out.differences;
            tc.verifyNotEmpty(D);
            tc.verifyEqual(D.status_matlab, D.status_simulator, script);
            limit = repmat(tc.Tolerance, height(D), 1);
            limit(D.status_simulator == "re-entered") = tc.DenseAir;
            bad = D(D.max_diff_km > limit, :);
            tc.verifyEmpty(bad, script + ": " + formattedDisplayText(bad));
            % the same objects at the same samples
            rows = @(X) sortrows(table(round(X.t_s * 1000), string(X.name)));
            tc.verifyEqual(rows(out.T), rows(out.P), script);
            % the event logs, line for line
            ev = out.eventDifferences;
            % the simulator checks for events at the end of whole steps, and shows the
            % last stretch of a run without a check: an event within a step of the end
            % may be in one log only
            tEnd = max(out.T.t_s);
            lonely = ev(isnan(ev.dt_s) & ~(max(ev.t_matlab, ev.t_simulator) > ...
                tEnd - out.sim.integrator.hMax), :);
            tc.verifyEmpty(lonely, script + ": in one log only: " + formattedDisplayText(lonely));
            exact = ismember(ev.kind, ["launch", "maneuver", "info"]);
            off = ev(exact & ~(abs(ev.dt_s) <= tc.Exact & ev.same_text), :);
            tc.verifyEmpty(off, script + ": " + formattedDisplayText(off));
            late = ev(~exact & ~isnan(ev.dt_s) & ~(abs(ev.dt_s) <= out.sim.integrator.hMax), :);
            tc.verifyEmpty(late, script + ": " + formattedDisplayText(late));
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
% read them by name at run time: "events" alone would be MATLAB's events function here
out = struct();
for name = ["T", "P", "differences", "events", "eventDifferences", "launches", "sim"]
    out.(name) = eval(name);
end
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
