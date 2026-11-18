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
            % nothing for the Code Analyzer to mark in the editor: no unused function,
            % no needless brackets, no output shown by a missing semicolon
            issues = codeIssues(script).Issues;
            tc.verifyEmpty(issues, script + ": " + formattedDisplayText( ...
                issues(:, ["LineStart", "CheckID", "Description"])));
        end

        function out = runAlone(tc, script)
            % runs a MATLAB-only script with the matlab/ toolkit off the path: it
            % must need nothing outside itself
            toolkit = fullfile(tc.Root, "matlab");
            rmpath(toolkit);
            restore = onCleanup(@() addpath(toolkit));
            close all
            out = runScript(script);
        end

        function checkSections(tc, script, out)
            % the analysis sections: their tables agree with the event log, the
            % stations and J2 theory, and the sky plot and the animation were drawn
            ev = out.events;
            tc.verifyEqual(sort(out.eclipses.name), sort(unique(out.T.name)), script);
            tc.verifyEqual(sum(out.eclipses.eclipses), ...
                nnz(endsWith(ev.text, " entered Earth's shadow")), script);
            pct = out.eclipses.shadow_pct;
            tc.verifyTrue(all(pct >= 0 & pct <= 100 + 1e-9), script + ": shadow share");
            if isempty(out.sim.stations)
                tc.verifyFalse(isfield(out, "passes"), script);
            else
                P = out.passes;
                tc.verifyEqual(height(P), nnz(startsWith(ev.text, "AOS ")), script);
                closed = ~isnat(P.los);
                tc.verifyEqual(nnz(closed), nnz(startsWith(ev.text, "LOS ")), script);
                tc.verifyTrue(all(P.los(closed) >= P.aos(closed)) && all(P.minutes >= 0), ...
                    script + ": pass times");
                % a pass of 10 minutes or more has a sample between two of the
                % simulator's checks that saw it, so above the station's mask: the
                % elevation here is the one the simulator measures
                mask = arrayfun(@(nm) out.sim.stations([out.sim.stations.name] == nm).minEl, ...
                    P.station);
                long = P.minutes >= 10;
                low = P(long & ~(P.max_el_deg >= mask - 1e-6 & P.max_el_deg <= 90), :);
                tc.verifyEmpty(low, script + ": " + formattedDisplayText(low));
                tc.verifyTrue(issorted(P.aos), script + ": passes in the order they happen");
            end
            % the coverage: a row per station, from its passes
            if ~isempty(out.sim.stations)
                V = out.coverage;
                tc.verifyEqual(V.station, reshape([out.sim.stations.name], [], 1), script);
                runMin = minutes(max(out.T.utc) - min(out.T.utc));
                for k = 1:height(V)
                    mine = out.passes(out.passes.station == V.station(k), :);
                    tc.verifyEqual(V.passes(k), height(mine), script);
                    tc.verifyEqual(V.objects(k), numel(unique(mine.name)), script);
                    tc.verifyLessThanOrEqual(V.in_view_min(k), sum(mine.minutes) + 1e-9);
                    tc.verifyLessThanOrEqual(V.in_view_min(k) + V.longest_gap_min(k), ...
                        runMin + 1e-9, script + ": " + V.station(k));
                end
            end
            % the beta angles: an object enters the shadow only where its beta is inside
            % its limit (a degree's allowance for the penumbra and the samples between),
            % and one outside it all the run has no eclipse
            B = out.betaAngles;
            tc.verifyTrue(all(abs(B.beta_deg) <= 90), script);
            into = ev(endsWith(ev.text, " entered Earth's shadow"), :);
            for k = 1:height(into)
                nm = erase(into.text(k), " entered Earth's shadow");
                mine = B(B.name == nm, :);
                [~, j] = min(abs(mine.t_s - into.t_s(k)));
                if ~isempty(j) && ~isnan(mine.limit_deg(j))
                    tc.verifyLessThanOrEqual(abs(mine.beta_deg(j)), mine.limit_deg(j) + 1, ...
                        script + ": " + into.text(k) + " at " + into.t_s(k) + " s");
                end
            end
            for nm = unique(B.name)'
                mine = B(B.name == nm, :);
                if ~isempty(mine) && all(abs(mine.beta_deg) > mine.limit_deg + 1)
                    tc.verifyEqual(out.eclipses.eclipses(out.eclipses.name == nm), 0, ...
                        script + ": " + nm + " is in sunlight all round");
                end
            end
            % the sky plot: there when a pass of an object plotted has a sample to draw
            drawn = isfield(out, "passes") && any(ismember(out.passes.name, out.shown) & ...
                ~isnan(out.passes.max_el_deg));
            tc.verifyEqual(~isempty(findobj(groot, "Type", "figure", "Name", "Sky plot")), ...
                drawn, script + ": sky plot");
            % the orbit evolution: over two orbits or more, and the node and perigee
            % drifting as J2 theory has them, give or take what the other forces add
            % (within 10%, or 0.05 deg/day for a drift near zero)
            R = out.drift;
            tc.verifyTrue(all(ismember(R.name, unique(out.T.name))), script);
            tc.verifyTrue(all(R.days * 1440 >= 2 * R.period_min - 1e-6), script + ": span");
            tc.verifyTrue(all(R.perigee_km <= R.apogee_km) && all(isfinite(R.da_km_day)), script);
            for col = ["raan", "argp"]
                got = R.(col + "_deg_day");
                want = R.(col + "_j2_deg_day");
                tc.verifyEqual(isnan(got), isnan(want), script);
                off = R(abs(got - want) > 0.1 * abs(want) + 0.05, :);
                tc.verifyEmpty(off, script + ": " + col + " drift: " + formattedDisplayText(off));
            end
            % the maneuvers and close approaches: a row for each line that carries one
            names = unique(out.T.name);
            if isfield(out, "maneuvers")
                M = out.maneuvers;
                burns = ev(ev.kind == "maneuver" & contains(ev.text, ...
                    [": executed ", ": burnout - dV ", ": finite burn applied impulsively ("]), :);
                tc.verifyEqual(height(M), height(burns), script);
                tc.verifyEqual(M.utc, burns.utc, script);
                tc.verifyTrue(all(ismember(M.name, names)) && all(M.dv_ms >= 0), script);
                tc.verifyEqual(sum(out.deltaV.sum_dv_ms), sum(M.dv_ms), AbsTol=1e-9);
            end
            if isfield(out, "approaches")
                A = out.approaches;
                tc.verifyEqual(height(A), nnz(startsWith(ev.text, "Close approach: ")), script);
                tc.verifyTrue(all(ismember([A.name; A.other], names)), script);
                tc.verifyTrue(issorted(A.distance_km), script);
            end
            tc.verifyNotEmpty(findobj(groot, "Type", "figure", "Name", "Animation"), ...
                script + ": no animation");
            tc.checkBurns(script, out);
            tc.checkDiagnostics(script, out);
            tc.checkTheme(script);
        end

        function checkBurns(tc, script, out)
            % the burns table: a row for each burn the log reports done, with the
            % delta-v the log gives; an impulse leaves the position as it was; each
            % state just before a burn is where the run's last sample before it leads
            % (carried on along its two-body orbit, within a kilometre)
            B = out.burns;
            done = B(isfinite(B.t1_s), :);
            parts = hypot(hypot(done.dvV_ms, done.dvN_ms), done.dvB_ms);
            tc.verifyEqual(parts, done.dv_ms, script + ": V N B", AbsTol=1e-6, RelTol=1e-9);
            if isfield(out, "maneuvers")
                M = sortrows(out.maneuvers, ["name", "utc"]);
                ended = sortrows(table(done.name, out.sim.epoch + seconds(done.t1_s), ...
                    done.dv_ms, VariableNames=["name", "utc", "dv_ms"]), ["name", "utc"]);
                tc.verifyEqual(ended.name, M.name, script + ": burns");
                tc.verifyEqual(ended.dv_ms, M.dv_ms, script + ": burn delta-v", AbsTol=0.1);
            else
                tc.verifyEmpty(done, script + ": burns without a maneuvers section");
            end
            at = done.t1_s == done.t0_s;
            tc.verifyEqual(done.y1(at, 1:3), done.y0(at, 1:3), script + ": impulse moved it");
            C = struct("MU", 398600.4418);
            for k = 1:height(done)
                rows = out.T(out.T.name == done.name(k) & out.T.t_s < done.t0_s(k) - 1e-9 ...
                    & out.T.status == "active", :);
                if isempty(rows)
                    continue
                end
                y = [rows.x_km(end), rows.y_km(end), rows.z_km(end), rows.vx_kms(end), ...
                    rows.vy_kms(end), rows.vz_kms(end)];
                if dot(y(4:6), y(4:6)) / 2 - C.MU / norm(y(1:3)) >= 0
                    continue              % not on a closed orbit: no Kepler ellipse to follow
                end
                dt = done.t0_s(k) - rows.t_s(end);
                r = twoBody(y, dt, C.MU);
                tc.verifyLessThan(norm(r - done.y0(k, 1:3)), 1, sprintf( ...
                    "%s: burn %d of %s is %.3f km from where its last sample leads", script, ...
                    k, done.name(k), norm(r - done.y0(k, 1:3))));
            end
            if isfield(out, "transfers")
                X = out.transfers;
                tc.verifyEqual(height(X), height(B), script + ": transfers");
                tc.verifyEqual(X.dv_ms, B.dv_ms, script);
                fine = isfinite(X.perigee1_km) & isfinite(X.apogee1_km);
                tc.verifyTrue(all(X.perigee1_km(fine) <= X.apogee1_km(fine) + 1e-6), script);
                % the budget for any burn; the burn window when a plotted object burned
                has = @(name) ~isempty(findobj(groot, "Type", "figure", "Name", name));
                tc.verifyEqual(has("Delta-v budget"), ~isempty(done), script + ": budget");
                tc.verifyEqual(has("Burn window"), any(ismember(done.name, out.shown)), ...
                    script + ": burn window");
            end
        end

        function checkBurnsAgainst(tc, out, scenario, duration)
            % the states just before and after each burn applied at once, against the
            % simulator's own (matlab/tests/burn_states.py): to the cross-check's
            % tolerance in position, and in velocity to a centimetre a second
            file = fullfile(tc.Tmp, "burn_states.csv");
            runPython([fullfile(tc.Root, "matlab", "tests", "burn_states.py"), scenario, ...
                string(duration), file]);
            R = readtable(file, TextType="string");
            B = out.burns(out.burns.t1_s == out.burns.t0_s, :);
            tc.verifyEqual(height(B), height(R), scenario + ": burns applied at once");
            for k = 1:min(height(B), height(R))
                tc.verifyEqual(B.name(k), R.name(k), scenario);
                tc.verifyEqual(B.t0_s(k), R.t_s(k), scenario + ": burn time", AbsTol=1e-6);
                ref = [R{k, ["x0", "y0", "z0"]}; R{k, ["x1", "y1", "z1"]}];
                got = [B.y0(k, 1:3); B.y1(k, 1:3)];
                tc.verifyLessThan(max(vecnorm(got - ref, 2, 2)), tc.Tolerance, ...
                    sprintf("%s: burn %d position", scenario, k));
                ref = [R{k, ["vx0", "vy0", "vz0"]}; R{k, ["vx1", "vy1", "vz1"]}];
                got = [B.y0(k, 4:6); B.y1(k, 4:6)];
                tc.verifyLessThan(max(vecnorm(got - ref, 2, 2)), 1e-5, ...
                    sprintf("%s: burn %d velocity", scenario, k));
            end
        end

        function checkDiagnostics(tc, script, out)
            % over each coast of an object in orbit, what physics keeps constant there
            % (energy with the gravity terms' potential, or the Jacobi constant with
            % C22) held to the integrator's tolerance; with drag, energy never rises
            % by more than that
            K = out.conservation;
            tc.verifyTrue(all(K.drift_rel >= 0), script);
            kept = K(K.expected == "constant", :);
            bad = kept(kept.drift_rel > 1e-6, :);
            tc.verifyEmpty(bad, script + ": not conserved: " + formattedDisplayText(bad));
            D = out.diagnostics;
            tc.verifyTrue(all(ismember(D.name, unique(out.T.name))), script);
            tc.verifyTrue(all(D.acc_central_ms2 > 0), script);
            tc.verifyNotEmpty(findobj(groot, "Type", "figure", "Name", "Animation 3-D"), ...
                script + ": no 3-D animation");
        end

        function checkTheme(tc, script)
            % every figure, light and dark: each text against what it sits on (the
            % figure, its axes, a night side drawn there) at 4.5:1 or more, legend
            % text against its box, and a map's night side darker than its day side
            figs = findall(groot, "Type", "figure", "Tag", "satflight");
            tc.verifyNotEmpty(figs, script + ": no figure marked for the theme");
            for f = reshape(figs, 1, [])
                for style = ["light", "dark"]
                    theme(f, style);
                    where = sprintf("%s: %s, %s", script, f.Name, style);
                    for problem = themeProblems(f)
                        tc.verifyFail(where + ": " + problem);
                    end
                end
            end
        end

        function [D, out] = check(tc, script)
            close all
            out = runScript(script);
            tc.checkSections(script, out);
            tc.assertTrue(isfield(out, "differences"), script + ": no cross-check ran");
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

function problems = themeProblems(fig)
% what in figure fig is hard to read in its theme now (see checkTheme)
problems = strings(1, 0);
texts = [findall(fig, Type="text"); findall(fig, Type="textboxshape")];
for t = reshape(texts, 1, [])
    if t.Visible == "off" || all(strip(string(t.String)) == "") || ~isnumeric(t.Color)
        continue
    end
    back = fig.Color;
    ax = ancestor(t, ["axes", "polaraxes"]);
    if ~isempty(ax)
        if isnumeric(ax.Color) && ax.Visible == "on"
            back = [back; ax.Color]; %#ok<AGROW>
        end
        for night = reshape(findall(ax, Tag="satflight:night"), 1, [])
            back = [back; night.FaceColor]; %#ok<AGROW>
        end
    end
    ratio = min(arrayfun(@(k) contrast(t.Color, back(k, :)), 1:size(back, 1)));
    if ratio < 4.5
        problems(end + 1) = sprintf("text ""%s"" at contrast %.2f:1", ...
            strjoin(string(t.String), " "), ratio); %#ok<AGROW>
    end
end
for lg = reshape(findall(fig, Type="legend"), 1, [])
    if isnumeric(lg.Color) && contrast(lg.TextColor, lg.Color) < 4.5
        problems(end + 1) = sprintf("legend text at contrast %.2f:1", ...
            contrast(lg.TextColor, lg.Color)); %#ok<AGROW>
    end
end
for night = reshape(findall(fig, Tag="satflight:night"), 1, [])
    ax = ancestor(night, "axes");
    if luminance(night.FaceColor) >= luminance(ax.Color)
        problems(end + 1) = "the night side is not darker than the day side"; %#ok<AGROW>
    end
end
end

function r = contrast(a, b)
% WCAG contrast ratio of two RGB colours
la = luminance(a);
lb = luminance(b);
r = (max(la, lb) + 0.05) / (min(la, lb) + 0.05);
end

function l = luminance(c)
c = double(c);
lin = c / 12.92;
hi = c > 0.04045;
lin(hi) = ((c(hi) + 0.055) / 1.055) .^ 2.4;
l = [0.2126 0.7152 0.0722] * lin(:);
end

function r = twoBody(y, dt, mu)
% the position (km) dt seconds on along the two-body orbit through state y: the
% orbit's mean motion carried forward (Kepler's equation, Newton's method)
r0 = y(1:3);
v0 = y(4:6);
h = cross(r0, v0);
ev = cross(v0, h) / mu - r0 / norm(r0);
e = norm(ev);
a = 1 / (2 / norm(r0) - dot(v0, v0) / mu);
if e < 1e-9
    ev = r0 / norm(r0) * 1e-9;
    e = 1e-9;
end
P = ev / e;
Q = cross(h / norm(h), P);
nu0 = atan2(dot(r0, Q), dot(r0, P));
E0 = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(nu0 / 2));
M = E0 - e * sin(E0) + sqrt(mu / a ^ 3) * dt;
E = M;
for k = 1:50
    E = E - (E - e * sin(E) - M) / (1 - e * cos(E));
end
r = a * (cos(E) - e) * P + a * sqrt(1 - e ^ 2) * sin(E) * Q;
end

function out = runScript(file)
% runs the exported script here; its variables land in this workspace
run(file);
% read them by name at run time: "events" alone would be MATLAB's events function here.
% A MATLAB-only script has no P, differences or eventDifferences.
out = struct();
for name = ["T", "P", "differences", "events", "eventDifferences", "launches", "sim", ...
        "passes", "eclipses", "maneuvers", "deltaV", "approaches", "drift", "shown", ...
        "coverage", "betaAngles", "burns", "objects", "transfers", "diagnostics", ...
        "conservation", "savedFiles"]
    if exist(name, "var") == 1
        out.(name) = eval(name);
    end
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
