classdef ExportTest < ExportCase
    %EXPORTTEST Live scripts written by Scenarios > Export MATLAB run in MATLAB and
    %   reproduce the simulator (held to it as ExportCase describes), for cases built
    %   to reach every feature: each burn kind under each propagator, a burn under way,
    %   re-entry, launches from the pad and mid-ascent (presets and vehicles of the
    %   user's own), launches planned in MATLAB, and
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

        function sunMoonAndSunlight(tc)
            % the Sun's and Moon's gravity and sunlight pressure (dimmed through the
            % Earth's penumbra) with each satellite's Cr: a light sail-like craft that
            % crosses the shadow, a dense one, and a finite burn (Cr A / m stays at
            % ignition, as the simulator keeps it)
            sc = satflight.newScenario("Sunlit", J2=true, Drag=true, Sun=true, Moon=true, ...
                Srp=true, Epoch=datetime(2026, 3, 20, 6, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Sail", satflight.orbitElements(Altitude=800, ...
                I=60, Raan=40, Nu=10), Mass=50, Area=20, Cr=1.9);
            sc = satflight.addSatellite(sc, "Dense", satflight.orbitElements( ...
                PerigeeAlt=600, ApogeeAlt=20000, I=30, Raan=120, Argp=45, Nu=200), Mass=2000, ...
                Area=5, Cr=1.1);
            sc = satflight.addSatellite(sc, "Burner", satflight.orbitElements(Altitude=700, ...
                I=98, Raan=0, Nu=90), Mass=500, Area=8);
            % (a small burn: it leaves e under 0.01, where the drift analysis does not
            % fit argp; at e 0.017 a quarter-day coast fits -2.40 deg/day against J2's
            % -2.96 with or without the Sun, Moon and sunlight - short-period J2)
            sc = satflight.addManeuver(sc, "Burner", "finite", Timing="delay", Delay=1200, ...
                Dv=[0.5 0 0], Thrust=50, Isp=220, Duration=200);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "sunlit.json"));
            [~, out] = tc.check(tc.export(file, "sunlit", ["--duration", "6h", "--step", ...
                "60"]));
            D = out.diagnostics;
            for c = ["acc_sun_ms2", "acc_moon_ms2"]
                tc.verifyTrue(all(D.(c) > 0), c);   % each term applied, and measured
            end
            % sunlight pushes in the light and not in the umbra, which the Sail crosses
            tc.verifyTrue(all(D.acc_srp_ms2 >= 0) && any(D.acc_srp_ms2 > 0));
            tc.verifyTrue(any(D.acc_srp_ms2(D.name == "Sail") == 0), "Sail never in umbra");
            tc.verifyTrue(all(startsWith(out.conservation.expected, "not kept (the Sun")));
        end

        function launchInSunlight(tc)
            % a launch under the Sun, Moon and sunlight pressure: the payload separates
            % with its own Cr, the spent stages keep the default, and every object flies
            % as the simulator flies it
            sc = satflight.newScenario("Sunlit launch", J2=true, Drag=true, Sun=true, ...
                Moon=true, Srp=true, Epoch=datetime(2026, 3, 20, 6, 0, 0, TimeZone="UTC"));
            sc = satflight.addLaunch(sc, "Sail", Vehicle="Falcon 9 (approx.)", ...
                Lat=28.562, Lon=-80.577, Timing="delay", Delay=300, PerigeeAlt=500, ...
                ApogeeAlt=500, Inclination=40, PayloadMass=800, PayloadArea=30, ...
                PayloadCr=1.7, TrackStages=true);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "sunlit_launch.json"));
            [~, out] = tc.check(tc.export(file, "sunlit_launch", ["--duration", "3h", ...
                "--step", "60"]));
            O = out.objects;
            tc.verifyEqual(O.cr(O.name == "Sail"), 1.7);
            stages = startsWith(O.name, "Sail ");
            tc.verifyTrue(any(stages), "no spent stage tracked");
            tc.verifyEqual(O.cr(stages), repmat(1.3, nnz(stages), 1));
        end

        function egm2008Field(tc)
            % the EGM2008 field (degree 20) in place of J2-J4 and C22, with the Sun and
            % Moon: a Sentinel-1-like orbit, a GPS-like one and a burn, held to the
            % simulator; then the field alone, where the Jacobi constant must hold
            sc = satflight.newScenario("Field", GravityDegree=20, Drag=true, Sun=true, ...
                Moon=true, Epoch=datetime(2026, 3, 20, 6, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30), Mass=2200, Area=10);
            sc = satflight.addSatellite(sc, "Mid", satflight.orbitElements(Altitude=20200, ...
                I=55, Raan=60, Nu=200));
            sc = satflight.addManeuver(sc, "Low", "impulse", Timing="delay", Delay=3000, ...
                Dv=[0.002 0 0.01]);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "field.json"));
            [diffs, out] = tc.check(tc.export(file, "field", ["--duration", "6h", "--step", ...
                "60"]));
            % held to 1 m, not the usual 10: the port agrees to under a millimetre, and a
            % field one degree short is 12 m out by the end (close to the usual limit)
            tc.verifyLessThan(diffs.max_diff_km, 0.001);
            D = out.diagnostics;
            tc.verifyTrue(all(D.acc_egm_ms2 > 0));
            tc.verifyTrue(all(isnan(D.acc_j2_ms2)), "J2 applied besides the field");
            sc = satflight.newScenario("Field alone", GravityDegree=20, J2=false, ...
                Epoch=datetime(2026, 3, 20, 6, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30));
            % a circularize aims with the field even with the j2 switch off (it used to
            % aim at a two-body orbit, here and in the simulator)
            sc = satflight.addManeuver(sc, "Low", "circularize", Timing="delay", Delay=1800);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "field_alone.json"));
            [diffs, out] = tc.check(tc.export(file, "field_alone", ["--duration", "3h", ...
                "--step", "60"]));
            tc.verifyLessThan(diffs.max_diff_km, 0.001);
            K = out.conservation;
            tc.verifyEqual(height(K), 2);                   % before and after the burn
            tc.verifyTrue(all(K.quantity == "Jacobi constant"));
            tc.verifyTrue(all(K.expected == "constant"));
            % the two routes through the coefficients (V/W for the motion, Legendre
            % for the potential) agree, or the constant would wander; the simulator's
            % own drift here is 4.7e-9, its integrator's tolerance (rtol 1e-9)
            tc.verifyLessThan(K.drift_rel, 1e-8);
        end

        function nrlmsisDrag(tc)
            % drag through NRLMSIS 2.1 (pymsis in MATLAB's Python) with the measured
            % space weather, through the storm of 12 November 2025 (ap 300): a
            % Sentinel-1-like orbit and a low, light one, held to the simulator
            tc.assumeTrue(hasSpaceWeather(), "the space-weather table is not here");
            % (a density scale other than 1, so a runtime that drops it disagrees)
            sc = satflight.newScenario("Storm", Drag=true, Atmosphere="msis", ...
                DensityScale=1.3, Epoch=datetime(2025, 11, 12, 0, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30), Mass=2164, Area=10);
            sc = satflight.addSatellite(sc, "Cube", satflight.orbitElements(Altitude=400, ...
                I=51.6, Raan=10, Nu=200), Mass=4, Area=0.03);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "storm.json"));
            script = tc.export(file, "storm", ["--duration", "6h", "--step", "60"]);
            tc.verifyTrue(any(contains(readlines(script), """atmosphere"", ""msis""")));
            [diffs, out] = tc.check(script);
            % the same model on the same inputs: under a metre (the table would be
            % kilometres off for the Cube, whose air is several times denser)
            tc.verifyLessThan(diffs.max_diff_km, 0.001);
            D = out.diagnostics;
            tc.verifyTrue(all(D.acc_drag_ms2 > 0));
            % the storm: the low one meets more drag than the table's mean air gives
            % (its 3.725e-12 kg/m^3 at 400 km; NRLMSIS gave about 1.4 times that here,
            % before the scale of 1.3)
            cube = D(D.name == "Cube", :);
            v = 7.67;                                       % km/s at 400 km, roughly
            table = 0.5 * 2.2 * 0.03 / 4 * 3.725e-12 * (v * 1e3) ^ 2;
            tc.verifyGreaterThan(median(cube.acc_drag_ms2), 1.2 * table);
        end

        function physicalCd(tc)
            % the drag coefficient of a sphere in the NRLMSIS air at each step (forces
            % "cd_model": "dria_sphere"), through the same storm: held to the simulator.
            % A runtime that kept Cd at 2.2 was 175 m off on the Cube in 6 h (12 m on
            % Low), checked 2026-11-25.
            tc.assumeTrue(hasSpaceWeather(), "the space-weather table is not here");
            sc = satflight.newScenario("Storm Cd", Drag=true, Atmosphere="msis", ...
                CdModel="dria_sphere", DensityScale=1.3, ...
                Epoch=datetime(2025, 11, 12, 0, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30), Mass=2164, Area=10);
            sc = satflight.addSatellite(sc, "Cube", satflight.orbitElements(Altitude=400, ...
                I=51.6, Raan=10, Nu=200), Mass=4, Area=0.03, Cd=2.5);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "storm_cd.json"));
            script = tc.export(file, "storm_cd", ["--duration", "6h", "--step", "60"]);
            tc.verifyTrue(any(contains(readlines(script), """cdModel"", ""dria_sphere""")));
            [diffs, out] = tc.check(script);
            tc.verifyLessThan(diffs.max_diff_km, 0.001);
            tc.verifyTrue(all(out.diagnostics.acc_drag_ms2 > 0));
        end

        function tidesAndRelativity(tc)
            % the solid-Earth tide and the Schwarzschild term (forces "tides",
            % "relativity") with the Sun and Moon, 12 h: held to the simulator under 1 mm
            sc = satflight.newScenario("Tidal", Sun=true, Moon=true, Tides=true, ...
                Relativity=true, Epoch=datetime(2025, 11, 12, 0, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30));
            sc = satflight.addSatellite(sc, "Eccentric", satflight.orbitElements( ...
                PerigeeAlt=500, ApogeeAlt=20000, I=63.4, Raan=40, Nu=10));
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "tidal.json"));
            script = tc.export(file, "tidal", ["--duration", "12h", "--step", "60"]);
            tc.verifyTrue(any(contains(readlines(script), '"tides", true, "relativity", true')));
            [diffs, out] = tc.check(script);
            tc.verifyLessThan(diffs.max_diff_km, 1e-6);
            tc.verifyTrue(all(out.diagnostics.acc_tides_ms2 > 0));
            tc.verifyTrue(all(out.diagnostics.acc_relativity_ms2 > 0));
        end

        function earthAlbedo(tc)
            % the Earth's reflected sunlight and infrared (forces "albedo") with the Sun's
            % pressure, 12 h: a Sentinel-like satellite and a light sail held to the
            % simulator under 1 mm
            sc = satflight.newScenario("Bright", Sun=true, Srp=true, Albedo=true, ...
                Epoch=datetime(2025, 11, 12, 0, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=693, ...
                I=98.18, Raan=278, Nu=30), Mass=2164, Area=10);
            sc = satflight.addSatellite(sc, "Sail", satflight.orbitElements(Altitude=800, ...
                I=51.6, Raan=10, Nu=200), Mass=5, Area=4, Cr=1.8);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "bright.json"));
            script = tc.export(file, "bright", ["--duration", "12h", "--step", "60"]);
            tc.verifyTrue(any(contains(readlines(script), '"albedo", true')));
            [diffs, out] = tc.check(script);
            tc.verifyLessThan(diffs.max_diff_km, 1e-6);
            tc.verifyTrue(all(out.diagnostics.acc_albedo_ms2 > 0));
        end

        function nrlmsisAscent(tc)
            % under NRLMSIS an ascent flies the NRLMSIS column over its pad at liftoff
            % (launch.msis_column), in the storm: two launches from far-apart pads, 15 and
            % 40 minutes in, held to the simulator, the max-Q lines word for word; then
            % exported again with the first one in powered flight
            tc.assumeTrue(hasSpaceWeather(), "the space-weather table is not here");
            sc = satflight.newScenario("Storm launches", Drag=true, Atmosphere="msis", ...
                Epoch=datetime(2025, 11, 12, 0, 0, 0, TimeZone="UTC"));
            sc = satflight.addLaunch(sc, "Up", Vehicle="Falcon 9 (approx.)", Lat=28.56, ...
                Lon=-80.58, PerigeeAlt=400, ApogeeAlt=400, Inclination=51.6, ...
                Direction="north", Timing="delay", Delay=900, PayloadMass=5000);
            sc = satflight.addLaunch(sc, "South", Vehicle="Electron (approx.)", ...
                Lat=-39.26, Lon=177.86, PerigeeAlt=200, ApogeeAlt=500, Inclination=97.4, ...
                Direction="south", Timing="delay", Delay=2400, PayloadMass=150, ...
                Circularize=true);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "storm_launches.json"));
            [~, out] = tc.check(tc.export(file, "stormlaunches", ["--duration", "1h", ...
                "--step", "60"]));
            tc.verifyEqual([out.launches.outcome], [out.launches.plannedOutcome]);
            tc.verifyEqual(nnz(contains(out.events.text, "max Q")), 2);
            [~, out] = tc.check(tc.export(file, "stormlaunchesflying", ["--at", "1000", ...
                "--duration", "1h", "--step", "60"]));
            tc.verifyEqual(sum(arrayfun(@(L) ~isempty(L.state), out.sim.launches)), 1);
        end

        function nrlmsisPastTheMeasuredDays(tc)
            % the 81-day mean F10.7 reaches predictions in the last 40 days of the
            % table: a run there stops with the reason, as the simulator does
            tc.assumeTrue(hasSpaceWeather(), "the space-weather table is not here");
            sc = satflight.newScenario("Too new", Drag=true, Atmosphere="msis", ...
                Epoch=datetime("today", TimeZone="UTC") - days(5));
            sc = satflight.addSatellite(sc, "Low", satflight.orbitElements(Altitude=500));
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "too_new.json"));
            script = tc.export(file, "too_new", ["--duration", "1h", "--step", "60", ...
                "--no-compare"]);
            try
                tc.runAlone(script);
                tc.verifyFail("a run past the measured space weather went ahead");
            catch err
                tc.verifyEqual(string(err.identifier), "satflight:spaceWeather", err.message);
                tc.verifySubstring(err.message, "81-day mean");
            end
            % and an unknown atmosphere is refused when the scenario is built
            tc.verifyError(@() satflight.checkScenario(satflight.newScenario("x", ...
                Atmosphere="jacchia")), "satflight:python");
        end

        function preciseOrbitReplay(tc)
            % Sentinel-1A on ESA's precise orbit files, read here as the simulator reads
            % them (polar motion taken out with the IERS table), beside its twin flown by
            % J2 and drag, with a station that sees both; the recording ends part-way
            % (3 July 24:00 UTC), where the truth stops and says so
            tc.assumeTrue(hasPreciseOrbits("S1A", datetime(2025, 7, 3)), ...
                "the Sentinel-1A files of 3 July 2025 or the IERS table are not here");
            sc = satflight.newScenario("Replay", J2=true, Drag=true, ...
                Epoch=datetime(2025, 7, 3, 18, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "S1A truth", satflight.orbitReplay("S1A", ...
                datetime(2025, 7, 3)));
            sc = satflight.addSatellite(sc, "S1A twin", satflight.orbitPrecise("S1A"), ...
                Mass=2164, Area=10, Cd=1.88);
            sc = satflight.addStation(sc, "Kiruna", 67.86, 20.96, MinEl=5);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "replay.json"));
            [diffs, out] = tc.check(tc.export(file, "replay", ["--duration", "8h", ...
                "--step", "60"]));
            % the truth is the recording itself: the same file, the same arithmetic
            tc.verifyLessThan(diffs.max_diff_km(diffs.name == "S1A truth"), 1e-6);
            tc.verifyLessThan(diffs.final_diff_km(diffs.name == "S1A truth"), 1e-6);
            T = out.T;
            truth = T(T.name == "S1A truth", :);
            tc.verifyEqual(truth.status(end), "end of data");
            tc.verifyEqual(nnz(out.events.text == "S1A truth: end of the recorded orbit " + ...
                "(S1A data until 2025-07-03 24:00 UTC)"), 1);
            tc.verifyTrue(any(startsWith(out.events.text, "AOS S1A truth")), "no pass of the truth");
            % the twin starts on the truth and is measured against it while both fly
            G = out.truthGap;
            tc.verifyEqual(unique(G.name), "S1A twin");
            tc.verifyLessThan(G.distance_km(1), 1e-9);
            tc.verifyEqual(G.distance_km, vecnorm([G.radial_km, G.along_km, G.cross_km], 2, 2), ...
                AbsTol=1e-12);
            ended = truth.t_s(find(truth.status ~= "active", 1));
            tc.verifyLessThan(max(G.t_s), ended);
            tc.verifyLessThan(max(G.distance_km), 5);   % J2 and drag alone, 6 h
            % the forces of the run do not fly the truth: no diagnostics for it
            tc.verifyFalse(any(out.diagnostics.name == "S1A truth"));
            tc.verifyTrue(any(out.diagnostics.name == "S1A twin"));
        end

        function replayWithoutItsFiles(tc)
            % the files are read when the script runs: with none in the data folder it
            % stops and names the first day missing, nothing bridged
            tc.assumeTrue(hasPreciseOrbits("S1A", datetime(2025, 7, 3)), ...
                "the Sentinel-1A files of 3 July 2025 or the IERS table are not here");
            sc = satflight.newScenario("Replay", J2=true, ...
                Epoch=datetime(2025, 7, 3, 18, 0, 0, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "S1A truth", satflight.orbitReplay("S1A", ...
                datetime(2025, 7, 3)));
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "replay_alone.json"));
            script = tc.export(file, "replay_alone", ["--duration", "1h", "--step", "60", ...
                "--no-compare"]);
            text = readlines(script);
            k = find(startsWith(text, "dataFolder = """";"));
            tc.assertNumElements(k, 1);
            empty = fullfile(tc.Tmp, "no_data");
            mkdir(empty);
            text(k) = "dataFolder = """ + empty + """;";
            here = fullfile(tc.Tmp, "replay_no_data.m");
            writelines(text, here);
            try
                tc.runAlone(here);
                tc.verifyFail("the script ran without its precise orbit files");
            catch err
                tc.verifyEqual(string(err.identifier), "satflight:noData", err.message);
                tc.verifySubstring(err.message, "2025-07-03");
            end
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
            tc.check(tc.export(fullfile(tc.Root, "scenarios", "orbit_tour.json"), "later", ...
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
                "Rocket Lab sat in 00:45:12.9"), "the Electron payload's circularization");
        end

        function launchDayMidAscent(tc)
            % exported with two vehicles in powered flight (one on its second stage under
            % closed-loop guidance) and three on their pads, between whole seconds
            [~, out] = tc.check(tc.export(fullfile(tc.Root, "scenarios", "launch_day.json"), ...
                "midascent", ["--at", "333.7", "--duration", "1h", "--step", "60"]));
            tc.verifyEqual(numel(out.sim.launches), 5);
            tc.verifyEqual(sum(arrayfun(@(L) ~isempty(L.state), out.sim.launches)), 2);
        end

        function customLaunches(tc)
            % vehicles of the user's own, as the Launch dialog's Vehicle... builds them:
            % four stages under a g limit with long coasts between them, a fairing and
            % vacuum upper engines; a single stage (given as a struct, not a cell); one
            % too weak to lift off; a sounding rocket whose first stage is a vacuum
            % engine (Isp sea level 0, from the pad up: no preset has that). Custom sites,
            % and names with quotes, a percent sign and an accent. Exported from the pad
            % and again with two of them in powered flight
            accent = "O'Brien ""Heavy"" 50% " + char(201) + "toile";
            stage = @(name, thrust, ispVac, ispSl, prop, dry) struct("name", name, ...
                "thrust", thrust, "isp_vac", ispVac, "isp_sl", ispSl, "propellant", prop, ...
                "dry", dry);
            vehicle = @(name, stages, fairing, fairingAlt, diameter, cd, maxG, coast) ...
                struct("name", name, "stages", {stages}, "fairing", fairing, ...
                "fairing_alt", fairingAlt, "diameter", diameter, "cd", cd, "max_g", maxG, ...
                "stage_coast", coast);
            quad = vehicle(accent, {stage("Core 'A'", 9000, 300, 270, 300000, 25000), ...
                stage("Sustainer", 1500, 340, 0, 60000, 4000), ...
                stage("Upper", 300, 350, 0, 12000, 1200), ...
                stage("Kick %d", 30, 320, 0, 800, 120)}, 1500, 95, 4.2, 0.45, 4, 12);
            mono = vehicle("SSTO", stage("Only", 2500, 450, 380, 120000, 6000), 0, 110, 3, ...
                0.3, 0, 0);
            weak = vehicle("Too weak", {stage("Lead", 50, 300, 280, 20000, 2000), ...
                stage("Top", 20, 320, 0, 2000, 300)}, 200, 110, 2, 0.5, 0, 5);
            lob = vehicle("Custom", {stage("Stage 1", 400, 280, 0, 15000, 1500), ...
                stage("Stage 2", 60, 310, 0, 2000, 300)}, 0, 110, 1.5, 0.5, 0, 2);
            sc = satflight.newScenario("Custom launches", J2=true, Drag=true, ...
                Epoch=datetime(2026, 9, 29, 12, 0, 0, TimeZone="UTC"));
            sc = satflight.addLaunch(sc, "Quad 'one' ""Q"" 100%", Vehicle=quad, Lat=-30.95, ...
                Lon=136.5, Alt=0.17, PerigeeAlt=600, ApogeeAlt=600, Inclination=45, ...
                Direction="south", Circularize=true, PayloadMass=2500, PayloadCr=1.6);
            sc = satflight.addLaunch(sc, "Single stage", Vehicle=mono, Lat=5.2, Lon=-52.8, ...
                Guidance="open", Azimuth=45, Kick=3, Timing="absolute", T0=900, ...
                PayloadMass=300, TrackStages=false);
            sc = satflight.addLaunch(sc, "Grounded", Vehicle=weak, Lat=45.9, Lon=63.3, ...
                Timing="delay", Delay=120);
            sc = satflight.addLaunch(sc, "Lob", Vehicle=lob, Lat=67.9, Lon=21.1, ...
                Guidance="open", Azimuth=0, Kick=1.5, PayloadMass=100);
            file = satflight.saveScenario(sc, fullfile(tc.Tmp, "custom_launches.json"));
            [~, out] = tc.check(tc.export(file, "customlaunches", ["--duration", "2h", ...
                "--step", "60"]));
            tc.verifyEqual(string({out.launches.vehicle}), [accent, "SSTO", "Too weak", ...
                "Custom"]);
            tc.verifyEqual([out.launches.outcome], ["orbit", "escape", "no_liftoff", ...
                "suborbital"]);
            tc.verifyEqual([out.launches.outcome], [out.launches.plannedOutcome]);
            % in flight: the four-stage vehicle on its sustainer, the sounding rocket on
            % its first stage
            [~, out] = tc.check(tc.export(file, "customascent", ["--at", "100", ...
                "--duration", "1h", "--step", "60"]));
            resumed = arrayfun(@(L) ~isempty(L.state), out.sim.launches);
            tc.verifyEqual(string({out.sim.launches(resumed).name}), ...
                ["Quad 'one' ""Q"" 100%", "Lob"]);
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
            script = tc.export(fullfile(tc.Root, "scenarios", "orbit_tour.json"), "gif", ...
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

function ok = hasSpaceWeather()
% whether this machine has CelesTrak's space-weather table where the simulator looks for it
root = string(getenv("SATFLIGHT_DATA"));
if strtrim(root) == ""
    home = getenv("HOME");
    if ispc
        home = getenv("USERPROFILE");
    end
    root = fullfile(string(home), "satflight-data");
end
ok = isfile(fullfile(root, "spaceweather", "SW-All.csv"));
end

function ok = hasPreciseOrbits(mission, day)
% whether this machine has the mission's POEORB file for day (UTC) and the IERS table, where
% the simulator looks for them ($SATFLIGHT_DATA, else ~/satflight-data)
root = string(getenv("SATFLIGHT_DATA"));
if strtrim(root) == ""
    home = getenv("HOME");
    if ispc
        home = getenv("USERPROFILE");
    end
    root = fullfile(string(home), "satflight-data");
end
before = string(datetime(day - days(1), Format="yyyyMMdd"));
files = dir(fullfile(root, "poeorb", mission, mission + "_OPER_AUX_POEORB_OPOD_*_V" + before ...
    + "T*.EOF.zip"));
ok = ~isempty(files) && isfile(fullfile(root, "eop", "finals2000A.all"));
end
