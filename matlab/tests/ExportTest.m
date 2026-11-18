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
            % and each burn's state before and after it is the simulator's
            for propagator = ["cowell", "kepler", "j2mean"]
                file = satflight.saveScenario(tc.everyBurn(propagator), ...
                    fullfile(tc.Tmp, "burns_" + propagator + ".json"));
                [~, out] = tc.check(tc.export(file, "burns_" + propagator, ...
                    ["--duration", "3h", "--step", "60"]));
                tc.checkBurnsAgainst(out, file, 3 * 3600);
            end
        end

        function everyIntegrator(tc)
            % the fixed-step integrators run step for step as the simulator runs them,
            % truncation error and all: the same burns, re-entries and impacts, and the
            % events at the same moments
            steps = struct("rk4", 10, "wh", 30, "leapfrog", 2);
            for method = ["rk4", "wh", "leapfrog"]
                sc = tc.everyBurn("cowell");
                sc.name = "Every burn " + method;
                sc.integrator = struct("method", method, "h_fixed", steps.(method));
                file = satflight.saveScenario(sc, fullfile(tc.Tmp, "fixed_" + method + ".json"));
                [~, out] = tc.check(tc.export(file, "fixed_" + method, ...
                    ["--duration", "3h", "--step", "120"]));
                tc.verifyEqual(out.sim.integrator.method, method);
                tc.checkBurnsAgainst(out, file, 3 * 3600);
            end
        end

        function transferFiguresAndFiles(tc)
            % GTO to GEO: the apogee kick is the planner's 1837.6 m/s and leaves a
            % geostationary orbit, the parking-orbit route takes two burns; each object
            % gets its transfer figure. With figureFolder set every figure is saved as
            % PNG and PDF, and both animations as MP4 videos of every frame.
            file = fullfile(tc.Root, "scenarios", "gto_to_geo.json");
            script = tc.export(file, "gto", ["--duration", "8h", "--step", "300"]);
            text = readlines(script);
            k = find(startsWith(text, "figureFolder = """";"));
            tc.assertNumElements(k, 1);
            folder = fullfile(tc.Tmp, "figures");
            text(k) = "figureFolder = """ + folder + """;";
            here = fullfile(tc.Tmp, "gto_saved.m");
            writelines(text, here);
            [~, out] = tc.check(here);
            tc.checkBurnsAgainst(out, file, 8 * 3600);
            X = out.transfers;
            kick = X(X.name == "Apogee kick", :);
            tc.assertEqual(height(kick), 1);
            tc.verifyEqual(kick.dv_ms, 1837.6, AbsTol=0.1);
            tc.verifyEqual([kick.perigee1_km, kick.apogee1_km], [35786 35786], AbsTol=20);
            tc.verifyLessThan(kick.i1_deg, 0.1);
            tc.verifyEqual(kick.period1_min, 1436.07, AbsTol=1);
            tc.verifyEqual(nnz(X.name == "From parking"), 2);
            tc.verifyNumElements(findobj(groot, "Type", "figure", "-regexp", "Name", ...
                "^Transfer: "), 2);
            % the files: a PNG and a PDF per figure, and the two videos
            figs = findall(groot, "Type", "figure", "Tag", "satflight");
            tc.verifyNumElements(out.savedFiles, 2 * numel(figs));
            for f = reshape(out.savedFiles, 1, [])
                info = dir(f);
                tc.assertNotEmpty(info, f);
                tc.verifyGreaterThan(info.bytes, 1000, f);
            end
            pngs = out.savedFiles(endsWith(out.savedFiles, ".png"));
            tc.verifyNumElements(pngs, numel(figs));
            tc.verifyGreaterThanOrEqual(imfinfo(pngs(1)).Width, 1500, "300 dpi");
            % the map animation a frame a sample, the 3-D one its 240 frames (fewer only
            % where two fall on the same instant)
            frames = [numel(unique(out.T.t_s)), 200];
            names = ["animation_map.mp4", "animation_3d.mp4"];
            for k = 1:2
                video = VideoReader(fullfile(folder, names(k)));
                tc.verifyGreaterThanOrEqual(video.NumFrames, frames(k), names(k));
                tc.verifyEqual(mod([video.Width, video.Height], 2), [0 0], names(k));
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
                "Rocket Lab sat in 00:45:13.8"), "the Electron payload's circularization");
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
                "keyEvents", "atEnd", "burns", "objects", "passes", "coverage", "eclipses", ...
                "betaAngles", "approaches", "drift", "diagnostics", "conservation"]));
            for name = ["T", "events", "passes", "coverage", "eclipses", "betaAngles", "drift", ...
                    "burns", "objects", "diagnostics", "conservation"]
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
