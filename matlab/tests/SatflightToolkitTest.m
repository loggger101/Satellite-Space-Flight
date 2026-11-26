classdef SatflightToolkitTest < matlab.unittest.TestCase
    %SATFLIGHTTOOLKITTEST Tests of the MATLAB toolkit against the real simulator.
    %   From the repository root:
    %     matlab -batch "addpath('matlab'); assertSuccess(runtests('matlab/tests'))"
    %   (tests/test_matlab.py runs this when MATLAB is on the PATH.)

    properties
        Root
        Tmp
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename("fullpath"));
            addpath(fileparts(here));
            tc.Root = string(satflight.repoRoot());
            fixture = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            tc.Tmp = string(fixture.Folder);
            set(groot, "defaultFigureVisible", "off");
            tc.addTeardown(@() set(groot, "defaultFigureVisible", "remove"));
        end
    end

    methods (Static)
        function sc = demoScenario()
            sc = satflight.newScenario("MATLAB toolkit test", Description="built in MATLAB", ...
                Epoch=datetime(2026, 9, 29, 12, 0, 0, TimeZone="UTC"), Drag=true, RecordDt=30);
            sc = satflight.addSatellite(sc, "Imager", ...
                satflight.orbitElements(Altitude=550, I="sso", Raan=30), Mass=150, Area=1.2, ...
                DryMass=120);
            sc = satflight.addSatellite(sc, "Molniya", satflight.orbitElements( ...
                PerigeeAlt=600, ApogeeAlt=39700, I=63.4, Argp=270), Color=[255 120 200]);
            sc = satflight.addSatellite(sc, "Probe", ...
                satflight.orbitState([7000 0 0], [0 7.546 0.5]));
            sc = satflight.addSatellite(sc, "Raw", satflight.orbitElements(Altitude=700, I=40, ...
                Osculating=true));
            sc = satflight.addSatellite(sc, "Relay", satflight.orbitGeo(-75));
            sc = satflight.addConstellation(sc, "Shell", Altitude=1200, Inclination=55, ...
                Total=6, Planes=3, Phasing=1);
            sc = satflight.addStation(sc, "Svalbard", 78.23, 15.39, MinEl=5);
            sc = satflight.addManeuver(sc, "Imager", "impulse", Timing="delay", Delay=1800, ...
                Dv=[0.005 0 0], Label="+5 m/s");
            sc = satflight.addLaunch(sc, "Demo-1", Vehicle="Electron (approx.)", ...
                Lat=-39.26, Lon=177.86, PerigeeAlt=520, ApogeeAlt=520, Inclination=97.5, ...
                Direction="south", Timing="delay", Delay=600, PayloadMass=150);
        end
    end

    methods (Test)
        function bundledScenariosRoundTrip(tc)
            % every bundled scenario, read and written back by MATLAB, is the same scenario
            files = dir(fullfile(tc.Root, "scenarios", "*.json"));
            tc.assertNotEmpty(files);
            args = strings(1, 0);
            for f = files'
                original = string(fullfile(f.folder, f.name));
                copy = satflight.saveScenario(satflight.loadScenario(original), ...
                    fullfile(tc.Tmp, f.name));
                args = [args, original, copy]; %#ok<AGROW>
            end
            out = runPython([fullfile(tc.Root, "matlab", "tests", "compare_scenarios.py"), args]);
            tc.verifyEqual(count(string(out), "same"), numel(files), out);
        end

        function builtScenarioIsAccepted(tc)
            path = satflight.saveScenario(tc.demoScenario(), fullfile(tc.Tmp, "demo.json"));
            report = satflight.checkScenario(path);
            for name = ["Imager", "Molniya", "Probe", "Raw", "Relay", "Shell"]
                tc.verifySubstring(report, name);
            end
            raw = jsondecode(fileread(path));
            tc.verifyEqual(raw.satellites{1}.orbit.i, 'sso');   % orbits differ: a cell array
            tc.verifyEqual(raw.epoch, '2026-09-29T12:00:00Z');
            tc.verifyFalse(isfield(raw.satellites{1}, "color"));     % unset: simulator default
            tc.verifyEqual(raw.satellites{2}.color', [255 120 200]);
            tc.verifyClass(raw.stations, "struct");                  % a one-entry list stays a list
            tc.verifyEqual(raw.satellites{1}.dry_mass, 120);
            tc.verifyTrue(raw.satellites{4}.orbit.osculating);
        end

        function preciseOrbitBuilders(tc)
            % a replay of a Sentinel-1 unit's precise orbit and its twin, written as the
            % simulator reads them; an unknown unit is refused here, a malformed day by
            % the simulator
            sc = satflight.newScenario("S1", Epoch=datetime(2025, 7, 3, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Truth", satflight.orbitReplay("S1A", ...
                datetime(2025, 7, 9)));
            sc = satflight.addSatellite(sc, "Twin", satflight.orbitPrecise("S1A"));
            path = satflight.saveScenario(sc, fullfile(tc.Tmp, "s1.json"));
            raw = jsondecode(fileread(path));
            tc.verifyEqual(raw.satellites(1).orbit, struct("type", 'replay', "mission", 'S1A', ...
                "until", '2025-07-09'));
            tc.verifyEqual(raw.satellites(2).orbit.type, 'precise');
            tc.verifyEqual(satflight.orbitReplay("S1C", "2025-07-09").until, '2025-07-09');
            tc.verifyError(@() satflight.orbitPrecise("S2A"), ...
                "MATLAB:validators:mustBeMember");
            sc = satflight.newScenario("S1", Epoch=datetime(2025, 7, 3, TimeZone="UTC"));
            sc = satflight.addSatellite(sc, "Truth", satflight.orbitReplay("S1A", "9 July"));
            path = satflight.saveScenario(sc, fullfile(tc.Tmp, "s1_bad.json"));
            try
                satflight.checkScenario(path);
                tc.verifyFail("the simulator took a malformed replay day");
            catch err
                tc.verifyEqual(string(err.identifier), "satflight:python", err.message);
                tc.verifySubstring(err.message, "expected a date as YYYY-MM-DD");
            end
        end

        function studyTablesMatchTheCsvs(tc)
            % the year study's CSVs as tables, every number as Python reads it
            folder = fullfile(tc.Root, "docs", "analysis", "sentinel1");
            S = satflight.readStudy();
            tc.verifyEqual(sort(string(fieldnames(S)))', ["arcs" "arcs_msis" "arcs_msis_dria" "arcs_msis_iers" "arcs_msis_iers_alb" "burns" "daily" "phasing"]);
            tc.verifyEqual(S.daily.date.TimeZone, 'UTC');
            tc.verifyEqual(S.burns.start_utc.TimeZone, 'UTC');
            tc.verifyEqual(string(S.daily.date(1), "yyyy-MM-dd"), "2025-07-01");
            tc.verifyEqual(unique(S.daily.mission)', ["S1A" "S1C"]);
            tc.verifyEqual(unique(S.burns.kind)', ["burn" "placebo"]);
            failed = S.arcs(S.arcs.error ~= "", :);
            tc.verifyNotEmpty(failed);
            tc.verifyTrue(all(isnan(failed.fit_rms_m)));
            tc.verifySubstring(failed.error(1), "(run tools/fetch_poeorb.py)");
            tc.verifyEqual(S.burns.note(1), "");
            tc.verifyEqual(height(S.arcs_msis), 6);      % the six arcs fitted again
            tc.verifyTrue(all(S.arcs_msis.error == ""));
            tc.verifyEqual(height(S.arcs_msis_dria), 6);  % and with the physical Cd
            tc.verifyTrue(all(S.arcs_msis_dria.error == ""));
            tc.verifyEqual(height(S.arcs_msis_iers), 6);  % and with tides and relativity
            tc.verifyTrue(all(S.arcs_msis_iers.error == ""));
            tc.verifyEqual(height(S.arcs_msis_iers_alb), 6);  % and with albedo too
            tc.verifyTrue(all(S.arcs_msis_iers_alb.error == ""));

            files = struct("daily_S1A", S.daily(S.daily.mission == "S1A", :), ...
                "daily_S1C", S.daily(S.daily.mission == "S1C", :), "phasing", S.phasing, ...
                "burns", S.burns, "arcs", S.arcs, "arcs_msis", S.arcs_msis, ...
                "arcs_msis_dria", S.arcs_msis_dria, "arcs_msis_iers", S.arcs_msis_iers, ...
                "arcs_msis_iers_alb", S.arcs_msis_iers_alb);
            lines = splitlines(strip(string(runPython([fullfile(tc.Root, "matlab", ...
                "tests", "study_sums.py"), folder]))));
            tc.assertGreaterThan(numel(lines), 60);
            for line = lines'
                f = split(line, ",");
                T = files.(erase(f(1), ".csv"));
                tc.verifyEqual(height(T), double(f(3)), line);
                x = T.(f(2));
                if isdatetime(x)
                    tc.verifyEqual(nnz(~isnat(x)), double(f(4)), line);
                elseif isstring(x)
                    tc.verifyEqual(nnz(x ~= ""), double(f(4)), line);
                else
                    tc.verifyEqual(nnz(~isnan(x)), double(f(4)), line);
                    tc.verifyEqual(sum(x(~isnan(x))), double(f(5)), "RelTol", 1e-12, ...
                        "AbsTol", 1e-9, line);
                end
            end
        end

        function studyFolderRefusals(tc)
            % a folder without the daily tables, or with a foreign column, is refused;
            % the arcs, burns and phasing may not have been run yet
            src = fullfile(tc.Root, "docs", "analysis", "sentinel1");
            empty = fullfile(tc.Tmp, "nostudy");
            mkdir(empty);
            tc.verifyError(@() satflight.readStudy(empty), "satflight:noFile");
            part = fullfile(tc.Tmp, "partstudy");
            mkdir(part);
            copyfile(fullfile(src, "daily_S1A.csv"), part);
            S = satflight.readStudy(part);
            tc.verifyEqual(unique(S.daily.mission), "S1A");
            tc.verifyEqual(height(S.arcs), 0);
            tc.verifyTrue(ismember("density_scale", S.arcs.Properties.VariableNames));
            tc.verifyEqual(height(S.phasing), 0);
            bad = fullfile(tc.Tmp, "badstudy");
            mkdir(bad);
            text = readlines(fullfile(src, "phasing.csv"));
            text(1) = replace(text(1), "max_deg", "biggest_deg");
            copyfile(fullfile(src, "daily_S1A.csv"), bad);
            writelines(text, fullfile(bad, "phasing.csv"));
            tc.verifyError(@() satflight.readStudy(bad), "satflight:badStudy");
        end

        function studyFigures(tc)
            S = satflight.readStudy();
            figs = satflight.plotStudy(S);
            tc.addTeardown(@() close(figs));
            tc.verifyEqual(numel(figs), 3);
            tc.verifyEqual(string({figs.Name}), ["Mean elements" "Track and phasing" ...
                "Prediction accuracy"]);
            one = satflight.plotStudy(S, Figures="accuracy");
            tc.addTeardown(@() close(one));
            tc.verifyEqual(string(one.Name), "Prediction accuracy");
            % the burn ticks are the burns, not the placebo windows
            ax = findobj(figs(1), "Type", "axes", "Tag", "a");
            ticks = findall(ax, "Tag", "burns");        % kept out of the legend
            tc.verifyEqual(sum(arrayfun(@(h) numel(h.XData), ticks)), ...
                nnz(S.burns.kind == "burn"));
            tc.verifyError(@() satflight.plotStudy(S, Figures="nope"), ...
                "MATLAB:validators:mustBeMember");
        end

        function sunMoonAndSunlightOptions(tc)
            % the third bodies and sunlight pressure switch on from MATLAB, with each
            % satellite's and each constellation's reflectivity, and the simulator
            % takes them
            sc = satflight.newScenario("Lit", Sun=true, Moon=true, Srp=true);
            sc = satflight.addSatellite(sc, "Sail", satflight.orbitElements(Altitude=700), ...
                Cr=1.6);
            sc = satflight.addConstellation(sc, "Shell", Altitude=1200, Inclination=55, ...
                Total=2, Planes=1, Phasing=0, Cr=1.1);
            path = satflight.saveScenario(sc, fullfile(tc.Tmp, "lit.json"));
            raw = jsondecode(fileread(path));
            tc.verifyTrue(raw.forces.sun && raw.forces.moon && raw.forces.srp);
            tc.verifyEqual(raw.satellites.cr, 1.6);
            tc.verifyEqual(raw.constellations.cr, 1.1);
            tc.verifySubstring(satflight.checkScenario(path), "SUN+MOON+SRP");
        end

        function loadedScenarioCanBeEdited(tc)
            sc = satflight.loadScenario(fullfile(tc.Root, "scenarios", "hohmann_to_geo.json"));
            tc.verifyClass(sc.satellites, "cell");
            tc.verifyClass(sc.stations, "cell");
            sc.satellites{1}.mass = 3500;
            sc = satflight.addStation(sc, "Goldstone", 35.43, -116.89);
            path = satflight.saveScenario(sc, fullfile(tc.Tmp, "edited.json"));
            raw = jsondecode(fileread(path));
            tc.verifyEqual(raw.satellites(1).mass, 3500);
            tc.verifyEqual(raw.stations.name, 'Goldstone');
            satflight.checkScenario(path);
        end

        function simulatorRefusalsReachMatlab(tc)
            % the checker is only worth something if it bites
            sc = satflight.newScenario("bad", Propagator="rk99");
            tc.verifyError(@() satflight.checkScenario(sc), "satflight:python");
            try
                satflight.checkScenario(sc);
            catch err
                tc.verifySubstring(err.message, "propagator");
            end
            sunk = satflight.addSatellite(satflight.newScenario("sunk"), "Mole", ...
                satflight.orbitElements(A=5000));
            tc.verifyError(@() satflight.checkScenario(sunk), "satflight:python");
            typo = satflight.addSatellite(satflight.newScenario("typo"), "X", ...
                satflight.orbitElements(Altitude=500));
            typo.satellites{1}.mas = 10;
            tc.verifyError(@() satflight.checkScenario(typo), "satflight:python");
        end

        function matlabSideRefusals(tc)
            nan = satflight.addManeuver(tc.demoScenario(), "Imager", "impulse", Dv=[NaN 0 0]);
            tc.verifyError(@() satflight.saveScenario(nan, fullfile(tc.Tmp, "nan.json")), ...
                "satflight:badScenario");
            tc.verifyFalse(isfile(fullfile(tc.Tmp, "nan.json")));
            tc.verifyError(@() satflight.newScenario("x", Epoch=datetime(2026, 1, 1)), ...
                "satflight:badEpoch");
            tc.verifyError(@() satflight.orbitElements(I=[1 2]), "satflight:badOrbit");
        end

        function runAndPlot(tc)
            out = fullfile(tc.Tmp, "run.csv");
            [T, report] = satflight.runScenario(tc.demoScenario(), Duration="2h", Step=600, ...
                Out=out);
            tc.verifyTrue(isfile(out));
            tc.verifySubstring(report, "Key events");
            tc.verifyClass(T.utc, "datetime");
            tc.verifyEqual(T.utc.TimeZone, 'UTC');
            names = unique(T.name, "stable");
            tc.verifyTrue(all(ismember(["Imager", "Molniya", "Probe", "Relay"], names)));
            imager = T(T.name == "Imager", :);
            tc.verifyEqual(imager.t_s', 0:600:7200);
            tc.verifyEqual(imager.alt_km(1), 550, AbsTol=30);
            relay = T(T.name == "Relay", :);
            tc.verifyEqual(relay.lon_deg, -75 * ones(height(relay), 1), AbsTol=0.05);

            figs = [satflight.plotGroundTrack(T, Scenario=tc.demoScenario()), ...
                satflight.plotAltitude(T, Names=["Imager", "Molniya"]), ...
                satflight.plotElements(T, Names="Molniya"), satflight.plotOrbit3D(T)];
            tc.addTeardown(@() close(figs));
            tc.verifyTrue(all(isgraphics(figs)));
            tc.verifyError(@() satflight.plotAltitude(T, Names="Nobody"), ...
                "satflight:unknownSatellite");
        end

        function readRunRefusesForeignFiles(tc)
            bad = fullfile(tc.Tmp, "other.csv");
            writelines(["time,x,y", "0,1,2"], bad);
            tc.verifyError(@() satflight.readRun(bad), "satflight:badEphemeris");
            tc.verifyError(@() satflight.readRun(fullfile(tc.Tmp, "missing.csv")), ...
                "satflight:noFile");
        end
    end
end

function out = runPython(args)
% the toolkit's own private helper is out of reach from here; same contract
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
