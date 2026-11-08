classdef ExportTest < ExportCase
    %EXPORTTEST Live scripts written by Scenarios > Export MATLAB run in MATLAB and
    %   reproduce the simulator (held to it as ExportCase describes), for cases built
    %   to reach every feature: each burn kind under each propagator, a burn under way,
    %   re-entry, launches from the pad and mid-ascent, launches planned in MATLAB, and
    %   a MATLAB-only export (no cross-check) giving the very same run.
    %   BundledExportTest runs every bundled scenario the same way. From the
    %   repository root:
    %     matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests/ExportTest.m'))"

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

        function matlabOnly(tc)
            % exported without the cross-check (--no-compare): it runs with the toolkit
            % off the path, runs no simulator, and gives exactly the run of the script
            % with the cross-check (held to the simulator here), launches in flight too
            file = fullfile(tc.Root, "scenarios", "launch_day.json");
            args = ["--at", "333.7", "--duration", "1h", "--step", "60"];
            [~, both] = tc.check(tc.export(file, "withcheck", args));
            alone = tc.runAlone(tc.export(file, "alone", [args, "--no-compare"]));
            tc.verifyFalse(isfield(alone, "P") || isfield(alone, "differences"), ...
                "the MATLAB-only script ran the simulator");
            tc.verifyEqual(alone.T, both.T);
            tc.verifyEqual(alone.events, both.events);
            tc.verifyEqual(alone.launches, both.launches);
            tc.verifyEqual(alone.passes, both.passes);
            tc.verifyEqual(alone.eclipses, both.eclipses);
        end

        function passesAndAnimationFile(tc)
            % the geostationary satellite is in view of two stations the whole hour: long
            % passes, whose highest sampled elevation must clear the station's mask; with
            % animationFile set the animation is also written as a GIF, a frame a sample
            script = tc.export(fullfile(tc.Root, "scenarios", "default.json"), "gif", ...
                ["--duration", "1h", "--step", "300"]);
            text = readlines(script);
            k = find(startsWith(text, "animationFile = """";"));
            tc.assertNumElements(k, 1);
            gif = fullfile(tc.Tmp, "run.gif");
            text(k) = "animationFile = """ + gif + """;";
            % and resultsFile: the run and its tables saved to a MAT-file
            k = find(startsWith(text, "resultsFile = """";"));
            tc.assertNumElements(k, 1);
            mat = fullfile(tc.Tmp, "run.mat");
            text(k) = "resultsFile = """ + mat + """;";
            here = fullfile(tc.Tmp, "gif_set.m");
            writelines(text, here);
            [~, out] = tc.check(here);
            saved = load(mat);
            tc.verifyEqual(sort(string(fieldnames(saved)))', sort(["sim", "T", "events", ...
                "keyEvents", "atEnd", "passes", "coverage", "eclipses", "betaAngles", ...
                "approaches", "drift"]));
            for name = ["T", "events", "passes", "coverage", "eclipses", "betaAngles", "drift"]
                tc.verifyEqual(saved.(name), out.(name), name);
            end
            long = out.passes(out.passes.minutes >= 10, :);
            tc.verifyGreaterThanOrEqual(height(long), 2);
            tc.verifyTrue(all(ismember(["Geostationary", "GPS MEO"], long.name)));
            frames = imfinfo(gif);
            tc.verifyNumElements(frames, numel(unique(out.T.t_s)));
            tc.verifyEqual(frames(1).DelayTime, round(100 * 20 / numel(frames)));
        end

        function midBurnSnapshot(tc)
            % exported while the finite burn is under way: the rest of it carries over
            file = satflight.saveScenario(tc.everyBurn("cowell"), fullfile(tc.Tmp, "mid.json"));
            tc.check(tc.export(file, "midburn", ["--at", "750", "--duration", "2h"]));
        end
    end
end
