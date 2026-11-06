classdef (Abstract) ExportCase < matlab.unittest.TestCase
    %EXPORTCASE What the export suites share: each case exports a scenario with
    %   python -m satflight.matlab_export, runs the live script here, and holds its
    %   own cross-check section (the simulator run on the same scenario) to the
    %   simulator: positions to within 10 m while both runs have the object in orbit
    %   (50 m for one that comes down through the dense air, where the atmosphere
    %   table's small jumps at its layer boundaries fall at different points of each
    %   integrator's steps), the same objects at the same samples, the same final
    %   statuses, and the event logs line for line - launch and maneuver events to
    %   the millisecond, the rest within a step. Suites: ExportTest, BundledExportTest.

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
            hMax = out.sim.integrator.hMax;
            lonely = ev(isnan(ev.dt_s) & ~(max(ev.t_matlab, ev.t_simulator) > tEnd - hMax), :);
            lonely = lonely(~shortPasses(lonely, hMax), :);
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

function short = shortPasses(L, hMax)
% Rows of L (events in one log only) that pair up into a pass shorter than an
% integrator step - in sight of a station (AOS then LOS), out of it (LOS then AOS),
% in or out of the Earth's shadow - with both of its lines in the same log only. Both
% programs look for these at the ends of their steps, and each has its own steps: a
% pass that falls between two step ends of one program is not seen by it at all.
sim = isnan(L.t_matlab);
text = L.text;
text(sim) = L.text_simulator(sim);
t = L.t_matlab;
t(sim) = L.t_simulator(sim);
into = startsWith(text, "AOS ") | endsWith(text, " entered Earth's shadow");
outof = startsWith(text, "LOS ") | endsWith(text, " back in sunlight");
key = regexprep(text, {'^(AOS|LOS) ', ' entered Earth''s shadow$', ' back in sunlight$'}, ...
    {'', '', ''});
short = false(height(L), 1);
for a = reshape(find(into | outof), 1, [])
    if short(a)
        continue
    end
    b = find(~short & (into ~= into(a)) & (into | outof) & key == key(a) & ...
        sim == sim(a) & abs(t - t(a)) <= hMax, 1);
    if ~isempty(b)
        short([a, b]) = true;
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
