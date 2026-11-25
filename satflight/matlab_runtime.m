% ==========================================================================================
% Satellite Space Flight engine, ported to MATLAB.
% satflight/matlab_runtime.m: satflight.matlab_export appends this file to every exported
% script. It follows the simulator's Python code term by term (satflight/forces.py,
% atmosphere.py, frames.py, elements.py, analysis.py, maneuvers.py, launch.py, eclipse.py,
% ephemeris.py and the stepping and event rules of simulation.py), so a script runs the
% same physics, the same rocket ascents and logs the same events without Python.
% ==========================================================================================

function [T, events, launches, burns, objects] = satflightRun(sim, duration, step, opts)
%SATFLIGHTRUN Run the exported simulation as the simulator does.
%   [T, events, launches] = satflightRun(sim, duration, step) returns one table row
%   per object per sample - the columns of the simulator's ephemeris CSV, with a
%   launched payload from its pad and each spent stage from its separation - the
%   event log (the simulator's lines: launches, burns, eclipses, station passes,
%   close approaches, re-entries) and one entry per launch: outcome, insertion
%   orbit, delta-v budget and ascent profile.
%   [T, events, launches, burns, objects] = satflightRun(...) also returns the burns
%   (one row per burn: object, label, kind, start and end time, delta-v in total and
%   along the velocity, orbit normal and binormal (VNB) at ignition, mass, and the
%   exact state just before and just after it, which T samples only at its step) and
%   the objects (one row each: family, drag area and coefficient, mass at the end and
%   dry mass), for the transfer and diagnostic sections.
%   satflightRun(..., PlanLaunches=true) plans the launches still on their pads here
%   (launch window and pitch kick, as the simulator does when a scenario loads)
%   instead of flying the simulator's plan: needed after editing a launch.
arguments
    sim (1,1) struct
    duration (1,1) double {mustBeNonnegative, mustBeFinite}
    step (1,1) double {mustBePositive, mustBeFinite}
    opts.PlanLaunches (1,1) logical = false
end
C = satflightConstants();
if ~isfield(sim, "launches")
    sim.launches = satflightNoLaunches();
end
if isempty(sim.names) && isempty(sim.launches)
    error("satflight:empty", "There is nothing to run: no satellites and no launches.");
end
s = struct();
s.sim = sim;
s.C = C;
F = sim.forces;
s.env = struct("j2", F.j2, "j3", F.j3, "j4", F.j4, "densityScale", F.densityScale, ...
    "jd0", sim.jd0);
if msisDrag(F)
    % NRLMSIS drag: the measured space weather (read here) and pymsis (MATLAB's Python)
    s.sim.forces = msisSetup(F, sim.dataFolder, C);
    spaceWeatherIndices(s.sim.forces.sw, msisMillis(sim.jd0));  % a start it cannot serve
end
n = numel(sim.names);
s.names = reshape(string(sim.names), [], 1);
s.Y = reshape(sim.y0, n, 6);
s.mass = reshape(sim.mass, [], 1);
s.area = reshape(sim.area, [], 1);
s.cd = reshape(sim.cd, [], 1);
s.cr = reshape(sim.cr, [], 1);
s.dry = zeros(n, 1);                     % kg left with the propellant gone; 0 = no limit
if isfield(sim, "dryMass")
    s.dry = reshape(sim.dryMass, [], 1);
end
s.status = repmat("active", n, 1);
s.dvUsed = zeros(n, 1);
s.family = strings(n, 1);
s.shadow = ones(n, 1);
% a replayed object follows its precise orbit recording instead of being flown (read from
% the POEORB files here); mission: the Sentinel-1 unit of a replay or its twin
s.replay = cell(n, 1);
s.mission = strings(n, 1);
if isfield(sim, "mission")
    s.mission = reshape(string(sim.mission), [], 1);
    until = reshape(string(sim.replayUntil), [], 1);
    for i = reshape(find(until ~= ""), 1, [])
        s.replay{i} = replayLoad(s.mission(i), sim.epoch, until(i), sim.dataFolder, C);
    end
end
s.followed = ~cellfun(@isempty, s.replay);
s.replays = any(s.followed);
s = stationSetup(s, sim.stations);
s.man = sim.maneuvers;
s.pending = true(1, numel(s.man));
s.burns = struct("sat", {}, "man", {}, "t0", {}, "t1", {}, "m0", {}, "mdot", {});
s.burnLog = struct("sat", zeros(0, 1), "label", strings(0, 1), "kind", strings(0, 1), ...
    "t0", zeros(0, 1), "t1", zeros(0, 1), "y0", zeros(0, 6), "y1", zeros(0, 6), ...
    "m0", zeros(0, 1), "m1", zeros(0, 1), "dv", zeros(0, 3));
s.ascents = {};
s.reports = {};
s.evT = zeros(0, 1);
s.evKind = strings(0, 1);
s.evText = strings(0, 1);
s.t = 0;
s.closePairs = zeros(0, 2);
s.lastSample = [];
% the simulator's adaptive integrator (dopri5, ported below step for step) starts with a
% 10 s step and carries its step size from one stretch of the run to the next; events
% are looked for at the ends of its steps
s.hNext = min(10, sim.integrator.hMax);

% the sample times the batch runner writes: 0, step, 2 step, ... and the end
tout = 0;
while tout(end) < duration - 1e-9
    tout(end + 1) = tout(end) + min(step, duration - tout(end)); %#ok<AGROW>
end
s.tout = tout;
s.Yout = nan(n, 6, numel(tout));
s.statusOut = strings(n, numel(tout));

% loading, as the simulator loads a scenario: maneuvers, launches, then a first look
for j = 1:numel(s.man)
    s = logEvent(s, 0, sprintf("Scheduled %s for %s %s", s.man(j).label, s.man(j).sat, ...
        whenText(s.man(j).t, 0)), "maneuver");
end
for L = reshape(sim.launches, 1, [])
    s = launchVehicle(s, L, opts.PlanLaunches);
end
[s, s.Y] = followRecordings(s, s.Y, 0);
idx = freeIdx(s);
s = stepLeft(s, 0, idx, s.Y);
s = stepEvents(s, 0, idx, s.Y);
s = logEvent(s, 0, sprintf("Scenario '%s' loaded: %d satellites, %d stations, propagator %s", ...
    sim.name, numel(s.names), numel(s.st.name), sim.propagator), "info");
left = ignoredTerms(sim);
if ~isempty(left)
    s = logEvent(s, 0, sprintf("Propagator %s ignores %s: switch to cowell to integrate them", ...
        sim.propagator, strjoin(left, ", ")), "warn");
end
% the batch runner: a row at the start, then the run advanced to each sample time and
% shown there. The physics keeps its own steps (a sample never shortens one): what a
% sample shows between two step ends is integrated on the side from the last one
s = recordAt(s, 1, s.Y);
% the first advance has no physics state to go back to (the simulator's _resume): nothing
% joins the states as loaded to its first step, which logs what it finds at its end
s.lastSample = [];
for k = 2:numel(tout)
    s = advanceTo(s, tout(k));
    [s, Ynow] = presentState(s, tout(k));
    s = recordAt(s, k, Ynow);
end
missed = find(all(s.statusOut == "", 1));
if ~isempty(missed)
    error("satflight:internal", "No state was recorded for t = %g s (a bug in this script).", ...
        tout(missed(1)));
end
T = ephemerisTable(s, C);
events = table(s.evT, sim.epoch + seconds(s.evT), s.evKind, s.evText, ...
    VariableNames=["t_s", "utc", "kind", "text"]);
reports = s.reports;
for q = 1:numel(s.ascents)
    reports{end + 1} = launchSummary(s, s.ascents{q}); %#ok<AGROW>
end
launches = [reports{:}];
if isempty(launches)
    launches = struct([]);
else
    % in the order they were put on their pads (their payloads' rows)
    [~, order] = sort(arrayfun(@(R) find(s.names == R.name, 1), launches));
    launches = launches(order);
end
burns = burnTable(s);
objects = table(s.names, s.family, s.area, s.cd, s.mass, s.dry, s.cr, VariableNames=[ ...
    "name", "family", "area_m2", "cd", "mass_end_kg", "dry_kg", "cr"]);
end

function idx = freeIdx(s)
% the active objects the ensemble propagates: not on a pad or in powered flight, nor
% following a recording
act = s.status == "active" & ~s.followed;
for q = 1:numel(s.ascents)
    act(s.ascents{q}.sat) = false;
end
idx = find(act);
end

function s = recordAt(s, k, Ycur)
Ycur = presentAscents(s, Ycur, s.tout(k));
n = numel(s.names);
s.Yout(1:n, :, k) = Ycur;
s.statusOut(1:n, k) = s.status;
end

function s = advanceTo(s, tEnd)
% the simulator's advance: from boundary to boundary (liftoffs, ascent syncs, maneuvers,
% burnouts), the objects in orbit integrated in the run's own steps; a stretch that
% ends at tEnd, not at a boundary, stops at the last whole step before it
checkFinite(s, s.t, freeIdx(s));
s = executeDue(s);
stuck = 0;
tSeen = s.t;
while true
    if s.t == tSeen
        stuck = stuck + 1;
    else
        stuck = 0;
    end
    tSeen = s.t;
    if stuck > 1000
        error("satflight:stuck", "The run made no progress at t = %.3f s.", s.t);
    end
    tb = nextBoundary(s);
    land = tb <= tEnd;
    idx = freeIdx(s);
    if ~isempty(idx)
        tStop = inf;
        if ~land
            tStop = tEnd;
        end
        [s, left] = propagate(s, idx, tb, tStop);
        if left
            continue                % an object left the run: go on without it
        end
    elseif land
        s.t = tb;
    elseif ~any(cellfun(@(A) A.phase ~= "pad", s.ascents))
        s.t = tEnd;                 % nothing is integrated: there are no steps to keep
        s = steppedIdle(s, idx);
    end
    if ~land
        break
    end
    s = advanceAscents(s, s.t);
    if isempty(idx)
        s = steppedIdle(s, idx);
    end
    s = finishBurns(s);
    s = executeDue(s);
end
end

function [s, Y] = presentState(s, t)
% the states shown at t, which may lie up to a step past the physics (the simulator's
% _present): the objects in orbit integrated there on the side, the physics untouched,
% and the replayed ones where their recordings have them (one may end there)
Y = s.Y;
idx = freeIdx(s);
if t > s.t && ~isempty(idx)
    Y(idx, :) = peek(s, idx, t);
end
if t > s.t
    [s, Y] = followRecordings(s, Y, t);
end
end

function Y = peek(s, idx, t1)
% the states of objects idx at t1, less than a step ahead, with a scratch integrator
% that starts from the run's step size and leaves it as it is
Y0 = s.Y(idx, :);
if s.sim.propagator ~= "cowell"
    Y = analyticStep(Y0, t1 - s.t, s.sim.propagator, s.C);
    return
end
it = s.sim.integrator;
n = numel(idx);
f = cowellDerivative(s, idx, t1);
if it.method == "dopri5"
    y = dopri5(f, s.t, reshape(Y0, [], 1), t1, s.hNext, it, n, s.C);
else
    y = fixedSteps(f, s.t, reshape(Y0, [], 1), t1, it, n, s.C);
end
Y = reshape(y, n, 6);
end

function checkFinite(s, t, idx)
% the simulator's _check_finite: a state that is not finite is an error, never a result
bad = ~all(isfinite(s.Y(idx, :)), 2);
if any(bad)
    error("satflight:nonfinite", "Non-finite state at t = %.3f s for %s.", t, ...
        strjoin(s.names(idx(bad)), ", "));
end
end

function Y = presentAscents(s, Y, t)
for q = 1:numel(s.ascents)
    A = s.ascents{q};
    Y(A.sat, :) = ascentPeek(A, t, s.C);
end
end

function tb = nextBoundary(s)
% the next liftoff, ascent sync, maneuver or burnout, as the simulator finds it
tb = inf;
for q = 1:numel(s.ascents)
    A = s.ascents{q};
    if A.phase == "pad"
        if s.t < A.t0 && A.t0 < tb
            tb = A.t0;
        end
    else
        tb = min(tb, max(A.t + s.C.ASCENT_SYNC, s.t));
    end
end
for j = find(s.pending)
    tm = s.man(j).t;
    if s.t < tm && tm < tb
        tb = tm;
    end
end
for b = s.burns
    if s.t < b.t1 && b.t1 < tb
        tb = b.t1;
    end
end
end

function s = logEvent(s, t, text, kind)
s.evT(end + 1, 1) = t;
s.evKind(end + 1, 1) = kind;
s.evText(end + 1, 1) = text;
end

function text = whenText(t, now)
if t <= now + 1e-6
    text = "now";
else
    text = "in " + formatDuration(t - now);
end
end

function text = formatDuration(seconds)
% satflight.timeutil.format_duration: 3d 04:05:06, or 04:05:06.7 within a day
sgn = "";
if seconds < 0
    sgn = "-";
end
x = abs(seconds);
days = floor(x / 86400);
x = x - 86400 * days;
h = floor(x / 3600);
x = x - 3600 * h;
m = floor(x / 60);
sec = x - 60 * m;
if days > 0
    text = sgn + sprintf("%dd %02d:%02d:%02d", days, h, m, floor(sec));
else
    text = sgn + sprintf("%02d:%02d:%04.1f", h, m, sec);
end
end

function text = commaNum(x)
% Python's f"{x:,.0f}": thousands separated by commas
if ~isfinite(x)
    text = lower(string(x));
    return
end
text = string(sprintf("%.0f", x));
neg = startsWith(text, "-");
digits = char(erase(text, "-"));
k = numel(digits) - 3;
while k > 0
    digits = [digits(1:k), ',', digits(k + 1:end)];
    k = k - 3;
end
text = string(digits);
if neg
    text = "-" + text;
end
end

function left = ignoredTerms(sim)
% the enabled force terms the propagator cannot apply (the simulator's ignored_terms):
% kepler carries two-body motion only, j2mean two-body plus J2
left = strings(1, 0);
if sim.propagator == "cowell"
    return
end
F = sim.forces;
field = F.gravityDegree > 0;      % the field stands in for J2, J3, J4 and C22
on = [~field & [F.j2, F.j3, F.j4, F.c22], field, F.drag, F.sun, F.moon, F.srp];
drag = "DRAG";
if physicalCd(F)
    drag = "DRAG(MSIS,Cd)";
elseif msisDrag(F)
    drag = "DRAG(MSIS)";
end
terms = ["J2", "J3", "J4", "C22", "EGM" + F.gravityDegree, drag, "SUN", "MOON", "SRP"];
used = sim.propagator == "j2mean" & terms == "J2";
left = terms(on & ~used);
end

function [s, row] = addRow(s, name, y, mass, area, cd, family)
% a new object (a payload on its pad, a spent stage); its name made unique as the
% simulator makes it
row = numel(s.names) + 1;
s.names(row, 1) = uniqueName(s, name);
s.Y(row, :) = y;
s.mass(row, 1) = mass;
s.area(row, 1) = area;
s.cd(row, 1) = cd;
s.cr(row, 1) = s.C.CR_DEFAULT;   % launched objects: the simulator's default
s.dry(row, 1) = 0;
s.status(row, 1) = "active";
s.dvUsed(row, 1) = 0;
s.family(row, 1) = family;
s.shadow(row, 1) = 1;
s.replay{row, 1} = [];
s.mission(row, 1) = "";
s.followed(row, 1) = false;
s.visible(:, row) = false;
s.Yout(row, :, :) = NaN;
s.statusOut(row, :) = "";
end

function name = uniqueName(s, name)
if ~any(s.names == name)
    return
end
k = 2;
while any(s.names == name + " " + k)
    k = k + 1;
end
name = name + " " + k;
end

% --- maneuvers ------------------------------------------------------------------------------

function s = executeDue(s)
C = s.C;
due = false;
for j = find(s.pending)
    m = s.man(j);
    if m.t > s.t + 1e-6
        continue
    end
    due = true;
    s.pending(j) = false;
    i = find(s.names == m.sat, 1);
    if isempty(i)
        error("satflight:noSatellite", "Maneuver %s is for %s, which is not in the scenario.", ...
            m.label, m.sat);
    end
    if s.status(i) ~= "active"
        s = logEvent(s, s.t, sprintf("%s: %s skipped (%s)", m.sat, m.label, s.status(i)), "warn");
        continue
    end
    r = s.Y(i, 1:3);
    v = s.Y(i, 4:6);
    if m.kind == "finite"
        s = startFinite(s, j, i);
        continue
    end
    tv = [];
    if m.kind == "match_velocity"
        it = find(s.names == m.target, 1);
        if isempty(it)
            s = logEvent(s, s.t, sprintf("%s: %s skipped (%s no longer exists)", m.sat, ...
                m.label, m.target), "warn");
            continue
        end
        tv = s.Y(it, 4:6);
    end
    if m.kind == "circularize"
        try
            dv = circularVelocity(s, i) - v;
        catch err
            if err.identifier ~= "satflight:circular"
                rethrow(err)
            end
            s = logEvent(s, s.t, sprintf("%s: %s skipped (%s)", m.sat, m.label, err.message), ...
                "warn");
            continue
        end
    else
        dv = impulseEci(m, r, v, tv, C);
    end
    dvm = norm(dv);
    room = dvLeft(s, i, m.isp);
    if dvm > room
        s = logEvent(s, s.t, sprintf("%s: propellant ran out - %.2f of %.2f m/s delivered " + ...
            "(dry mass %g kg)", m.sat, room * 1000, dvm * 1000, s.dry(i)), "warn");
        dv = dv * (room / dvm);
        dvm = room;
    end
    m0 = s.mass(i);
    s.Y(i, 4:6) = v + dv;
    s.dvUsed(i) = s.dvUsed(i) + dvm;
    s.mass(i) = max(s.mass(i) * exp(-dvm / (m.isp * C.G0)), s.dry(i));
    s = logBurn(s, i, m, s.t, [r, v], m0, dv);
    s = logEvent(s, s.t, sprintf("%s: executed %s - dV %.2f m/s", m.sat, m.label, dvm * 1000), ...
        "maneuver");
end
if due && ~isempty(s.lastSample)
    % the next close-approach check starts after the burns
    s.lastSample.t = s.t;
    s.lastSample.Y = s.Y;
end
end

function s = startFinite(s, j, i)
C = s.C;
m = s.man(j);
mdot = 0;
if m.isp > 0
    mdot = m.thrust / (m.isp * C.G0 * 1000);      % kg/s
end
duration = m.duration;
if mdot > 0
    % the burn ends when the propellant does: at the dry mass, or with none set, at
    % 95 % of the mass
    if s.dry(i) > 0
        floorMass = s.dry(i);
        why = sprintf("dry mass %g kg", s.dry(i));
    else
        floorMass = 0.05 * s.mass(i);
        why = "95 % of the mass, with no dry mass set";
    end
    burnable = max(s.mass(i) - floorMass, 0) / mdot;
    if burnable < duration
        s = logEvent(s, s.t, sprintf("%s: propellant for only %s of %s s of burn (%s)", m.sat, ...
            commaNum(burnable), commaNum(duration), why), "warn");
        duration = burnable;
    end
end
if s.sim.propagator ~= "cowell"
    % the analytic propagators cannot integrate thrust: the same rocket-equation
    % delta-v is applied at once along the burn direction
    mf = s.mass(i) - mdot * duration;
    dvm = 0;
    if mf > 0
        dvm = m.isp * C.G0 * log(s.mass(i) / mf);
    end
    d = finiteDirection(m, s.Y(i, 1:3), s.Y(i, 4:6));
    before = s.Y(i, :);
    m0 = s.mass(i);
    s.Y(i, 4:6) = s.Y(i, 4:6) + dvm * d;
    s.dvUsed(i) = s.dvUsed(i) + dvm;
    s.mass(i) = mf;
    s = logBurn(s, i, m, s.t, before, m0, dvm * d);
    s = logEvent(s, s.t, sprintf("%s: finite burn applied impulsively (%.1f m/s) - switch " + ...
        "to Cowell to integrate thrust", m.sat, dvm * 1000), "maneuver");
    return
end
s.burns(end + 1) = struct("sat", i, "man", j, "t0", s.t, "t1", s.t + duration, ...
    "m0", s.mass(i), "mdot", mdot);
% opened here, closed at burnout with the state then (finishBurns)
s = logBurn(s, i, m, s.t, s.Y(i, :), s.mass(i), NaN(1, 3));
s = logEvent(s, s.t, sprintf("%s: ignition - %.0f N for %.0f s", m.sat, m.thrust, duration), ...
    "maneuver");
end

function s = finishBurns(s)
C = s.C;
keep = true(size(s.burns));
for k = 1:numel(s.burns)
    b = s.burns(k);
    if b.t1 > s.t + 1e-6
        continue
    end
    keep(k) = false;
    m = s.man(b.man);
    mf = b.m0 - b.mdot * (b.t1 - b.t0);
    dvm = 0;
    if mf > 0
        dvm = m.isp * C.G0 * log(b.m0 / mf);
    end
    s.mass(b.sat) = mf;
    s.dvUsed(b.sat) = s.dvUsed(b.sat) + dvm;
    open = find(s.burnLog.sat == b.sat & s.burnLog.t0 == b.t0 & isnan(s.burnLog.t1), 1);
    y0 = s.burnLog.y0(open, :);
    % the delta-v the thrust gave, along the direction it pointed at ignition
    s.burnLog.dv(open, :) = dvm * finiteDirection(m, y0(1:3), y0(4:6));
    s.burnLog.t1(open) = s.t;
    s.burnLog.y1(open, :) = s.Y(b.sat, :);
    s.burnLog.m1(open) = mf;
    s = logEvent(s, s.t, sprintf("%s: burnout - dV %.1f m/s, mass %.1f kg", m.sat, dvm * 1000, ...
        mf), "maneuver");
end
s.burns = s.burns(keep);
end

function s = logBurn(s, i, m, t, before, m0, dv)
% a burn for the burns table: object i's state and mass just before it (km, km/s, kg)
% and the delta-v it got (ECI, km/s). An impulse is done at once; a finite burn is
% opened with dv NaN and closed by finishBurns at burnout.
L = s.burnLog;
k = numel(L.t0) + 1;
L.sat(k, 1) = i;
L.label(k, 1) = string(m.label);
L.kind(k, 1) = string(m.kind);
L.t0(k, 1) = t;
L.y0(k, :) = before;
L.m0(k, 1) = m0;
L.dv(k, :) = dv;
if all(isfinite(dv))
    L.t1(k, 1) = t;
    L.y1(k, :) = s.Y(i, :);
    L.m1(k, 1) = s.mass(i);
else
    L.t1(k, 1) = NaN;
    L.y1(k, :) = NaN(1, 6);
    L.m1(k, 1) = NaN;
end
s.burnLog = L;
end

function B = burnTable(s)
% the burns table satflightRun returns: one row per burn, in the order they fired
L = s.burnLog;
n = numel(L.t0);
parts = NaN(n, 3);
for k = 1:n
    parts(k, :) = L.dv(k, :) * vnbBasis(L.y0(k, 1:3), L.y0(k, 4:6)).';
end
B = table(s.names(L.sat), L.label, L.kind, L.t0, L.t1, s.sim.epoch + seconds(L.t0), ...
    1000 * vecnorm(L.dv, 2, 2), 1000 * parts(:, 1), 1000 * parts(:, 2), 1000 * parts(:, 3), ...
    L.m0, L.m1, L.y0, L.y1, VariableNames=["name", "label", "kind", "t0_s", "t1_s", "utc", ...
    "dv_ms", "dvV_ms", "dvN_ms", "dvB_ms", "mass0_kg", "mass1_kg", "y0", "y1"]);
end

function s = scheduleAtApsis(s, i, m)
% queue maneuver m for object i at its next apoapsis (the circularization a launch
% asks for), as the simulator's schedule() resolves it: the two-body estimate, then,
% unless the run is two-body, the highest point of the trajectory the run will fly
el = rv2coe(s.Y(i, 1:3), s.Y(i, 4:6), s.C.MU);
if el.e >= 1
    error("satflight:maneuver", "open orbit has no apoapsis");
end
dt = timeToTrueAnomaly(s.Y(i, 1:3), s.Y(i, 4:6), pi, s.C.MU);
if ~isfinite(dt)
    error("satflight:maneuver", "orbit never reaches apoapsis");
end
m.t = s.t + dt;
if s.sim.propagator ~= "kepler"
    m.t = nextApsis(s, i, true);
end
s.man(end + 1) = m;
s.pending(end + 1) = true;
s = logEvent(s, s.t, sprintf("Scheduled %s for %s %s", m.label, m.sat, whenText(m.t, s.t)), ...
    "maneuver");
end

function t = nextApsis(s, i, highest)
% time of the highest (or lowest) point of object i's coming revolution: the extreme of
% 180 samples, then the root of r.v between its neighbours by bisection (the
% simulator's _next_apsis)
C = s.C;
el = rv2coe(s.Y(i, 1:3), s.Y(i, 4:6), C.MU);
times = s.t + linspace(0, 2 * pi / el.n, 181);
times = times(2:end);
track = coastTrack(s, i, s.Y(i, :), s.t, times, 0);
rad = vecnorm(reshape(track(1, 1:3, :), 3, []), 2, 1);
if highest
    [~, k] = max(rad);
else
    [~, k] = min(rad);
end
k = min(max(k, 2), numel(times) - 1);
a = times(k - 1);
b = times(k + 1);
% integration is a flow, so it may restart from the sample before; the analytic
% j2mean is not (it re-reads mean elements), so it starts from now
if s.sim.propagator == "cowell"
    t0 = a;
    y0 = reshape(track(1, :, k - 1), 1, 6);
else
    t0 = s.t;
    y0 = s.Y(i, :);
end
g = @(t) radialDot(coastTo(s, i, y0, t0, t, 0));
ga = g(a);
yb = reshape(track(1, :, k + 1), 1, 6);
gb = dot(yb(1:3), yb(4:6));
if ga * gb > 0
    t = times(k);
    return
end
while b - a > C.APSIS_TIME_TOL
    t = 0.5 * (a + b);
    gt = g(t);
    if (gt < 0) == (ga < 0)
        a = t;
        ga = gt;
    else
        b = t;
    end
end
t = 0.5 * (a + b);
end

function d = radialDot(y)
d = dot(y(1:3), y(4:6));
end

function [Y, h] = coastTo(s, rows, Y0, t0, t1, hMaxLong, h)
% the states Y0 (objects rows) carried from t0 to t1 under the run's own forces,
% coasting: no burns, no events, the run untouched (the simulator's coast_states). Its
% scratch integrator steps up to max(hMax, hMaxLong) s and starts with a step of h
% (left out: a fresh one's 10 s); h out is the step size it ends with
it = s.sim.integrator;
it.hMax = max(it.hMax, hMaxLong);
if nargin < 7
    h = min(10, it.hMax);
end
if t1 <= t0
    Y = Y0;
    return
end
if s.sim.propagator ~= "cowell"
    Y = analyticStep(Y0, t1 - t0, s.sim.propagator, s.C);
    return
end
n = numel(rows);
cdam = s.cd(rows) .* s.area(rows) ./ s.mass(rows);
cram = s.cr(rows) .* s.area(rows) ./ s.mass(rows);
f = @(t, y) derivative(t, y, n, cdam, cram, s.sim.forces, s.sim.jd0, s.man, zeros(0, 4), ...
    zeros(0, 1), s.C);
if it.method == "dopri5"
    [y, h] = dopri5(f, t0, reshape(Y0, [], 1), t1, h, it, n, s.C);
else
    y = fixedSteps(f, t0, reshape(Y0, [], 1), t1, it, n, s.C);
end
Y = reshape(y, n, 6);
end

function track = coastTrack(s, rows, Y0, t0, times, hMaxLong)
% the states (rows x 6 x times) of Y0 at the increasing times, coasting from t0, one
% scratch integrator carrying its step size from sample to sample (_coast_samples)
n = numel(rows);
track = zeros(n, 6, numel(times));
Y = Y0;
t = t0;
h = min(10, max(s.sim.integrator.hMax, hMaxLong));
for j = 1:numel(times)
    if s.sim.propagator == "cowell"
        [Y, h] = coastTo(s, rows, Y, t, times(j), hMaxLong, h);
        t = times(j);
    else
        Y = analyticStep(Y0, times(j) - t0, s.sim.propagator, s.C);
    end
    track(:, :, j) = Y;
end
end

function v = circularVelocity(s, i)
% velocity that puts object i on a circular orbit through where it is, in its plane,
% for this run's forces: no once-per-rev term in the radius over the coming
% revolution (J2 leaves a ripple at twice the orbital frequency). Newton on the radial
% and along-track speed (the simulator's circular_velocity)
C = s.C;
r = s.Y(i, 1:3);
rm = norm(r);
rh = r / rm;
th = unit3(cross3(cross3(r, s.Y(i, 4:6)), rh));
vc = sqrt(C.MU / rm);
F = s.sim.forces;
perturbed = s.sim.propagator == "j2mean" || (s.sim.propagator == "cowell" && ...
    (F.j2 || F.j3 || F.j4 || F.c22 || F.gravityDegree > 0 || F.drag || F.sun || ...
    F.moon || F.srp));      % every term the acceleration includes, as the simulator
if ~perturbed
    v = vc * th;
    return
end
period = 2 * pi * rm / vc;
times = s.t + linspace(0, period, C.CIRC_SAMPLES + 1);
times = times(2:end);
p = [0, vc];                                % radial, along-track km/s
step = [0 0; C.CIRC_STEP 0; 0 C.CIRC_STEP];
c = zeros(3, 2);
for pass = 1:C.CIRC_ITER
    trial = p + step;
    Y0 = [repmat(r, 3, 1), trial(:, 1) * rh + trial(:, 2) * th];
    track = coastTrack(s, [i; i; i], Y0, s.t, times, period / 360);
    for k = 1:3
        c(k, :) = oncePerRev(reshape(track(k, 1:3, :), 3, []).', rh, th, times - s.t);
    end
    if norm(c(1, :)) < C.CIRC_TOL
        v = p(1) * rh + p(2) * th;
        return
    end
    J = (c(2:3, :) - c(1, :)).' / C.CIRC_STEP;
    p = p - (J \ c(1, :).').';
end
error("satflight:circular", "no circular orbit found within %.0f mm (still %.3f m of " + ...
    "eccentricity)", C.CIRC_TOL * 1e6, norm(c(1, :)) * 1000);
end

function c = oncePerRev(P, rh, th, t)
% cos u and sin u amplitudes (km) of the radius along positions P (one per row), u the
% angle from rh toward th; the fit also takes a constant, a drift (drag) and the J2
% ripple at 2u, so only free eccentricity is left
u = atan2(P * th.', P * rh.');
t = t(:);
A = [ones(size(u)), t / t(end), cos(u), sin(u), cos(2 * u), sin(2 * u)];
coef = A \ vecnorm(P, 2, 2);
c = coef(3:4).';
end

function dv = dvLeft(s, i, isp)
% delta-v (km/s) the propellant still on board gives at isp seconds; inf with no dry mass
if s.dry(i) <= 0
    dv = inf;
elseif s.mass(i) > s.dry(i)
    dv = isp * s.C.G0 * log(s.mass(i) / s.dry(i));
else
    dv = 0;
end
end

function dt = timeToTrueAnomaly(r, v, nuTarget, mu)
% two-body time (s) until the orbit next reaches true anomaly nuTarget (inf if never)
el = rv2coe(r, v, mu);
e = el.e;
if e < 1
    dM = mod(trueToMean(nuTarget, e) - el.M, 2 * pi);
    if dM < 1e-9
        dM = dM + 2 * pi;
    end
    dt = dM / el.n;
    return
end
nuMax = acos(-1 / e);
nu0 = el.nu;
if nu0 > pi
    nu0 = nu0 - 2 * pi;
end
nut = nuTarget;
if nut > pi
    nut = nut - 2 * pi;
end
if ~(-nuMax < nut && nut < nuMax) || nut <= nu0
    dt = inf;
else
    dt = (trueToMean(nut, e) - trueToMean(nu0, e)) / el.n;
end
end

function dv = impulseEci(m, r, v, tv, C)
switch m.kind
    case "impulse"
        switch m.frame
            case "VNB"
                dv = m.dv * vnbBasis(r, v);
            case "RSW"
                dv = m.dv * rswBasis(r, v);
            case "ECI"
                dv = m.dv;
            otherwise
                error("satflight:frame", "Unknown maneuver frame %s.", m.frame);
        end
    case "circularize"
        rh = unitRows(r);
        h = unitRows(cross(r, v));
        dv = sqrt(C.MU / norm(r)) * cross(h, rh) - v;
    case "plane_change"
        k = unitRows(r);
        node = cross([0 0 1], cross(r, v));
        ang = deg2rad(m.deltaI);
        if dot(r, node) < 0
            ang = -ang;
        end
        c = cos(ang);
        sn = sin(ang);
        dv = v * c + cross(k, v) * sn + k * dot(k, v) * (1 - c) - v;
    case "match_velocity"
        dv = tv - v;
    otherwise
        error("satflight:kind", "%s is not an impulsive maneuver.", m.kind);
end
end

function d = finiteDirection(m, r, v)
d = m.dv / norm(m.dv);
switch m.frame
    case "VNB"
        d = d * vnbBasis(r, v);
    case "RSW"
        d = d * rswBasis(r, v);
    case "ECI"
    otherwise
        error("satflight:frame", "Unknown maneuver frame %s.", m.frame);
end
end

function B = vnbBasis(r, v)
vh = unitRows(v);
nh = unitRows(cross(r, v));
B = [vh; nh; cross(vh, nh)];
end

function B = rswBasis(r, v)
rh = unitRows(r);
wh = unitRows(cross(r, v));
B = [rh; cross(wh, rh); wh];
end

function u = unitRows(v)
n = vecnorm(v, 2, 2);
n(n == 0) = 1;
u = v ./ n;
end

% --- numerical propagation (propagator "cowell") ----------------------------------------------

function [s, left] = propagate(s, idx, t1, tStop)
% the simulator's _propagate: objects idx toward t1, landing on it, or with tStop < t1
% ending at the last whole step before tStop (a step is never shortened for it). After
% every step the run looks at them (stepped); left: it stopped early because an object
% left the run, and the integration starts again without it
it = s.sim.integrator;
n = numel(idx);
left = false;
if s.sim.propagator ~= "cowell"
    % kepler and j2mean: exact steps of hMax
    h = it.hMax;
    while s.t < t1 - 1e-9
        dt = min(h, t1 - s.t);
        if tStop < t1 && s.t + dt > tStop
            break
        end
        Ynew = analyticStep(s.Y(idx, :), dt, s.sim.propagator, s.C);
        [s, left] = stepped(s, s.t + dt, idx, Ynew);
        if left
            return
        end
    end
    return
end
f = cowellDerivative(s, idx, t1);
y = reshape(s.Y(idx, :), [], 1);
t = s.t;
if it.method ~= "dopri5"
    % rk4, wh, leapfrog: steps of hFixed from here, the last one cut short to land on t1
    while t < t1
        h = min(it.hFixed, t1 - t);
        if tStop < t1 && t + h > tStop
            break
        end
        y = fixedStep(it.method, f, t, y, h, n, s.C);
        if t1 - (t + h) < 1e-9
            t = t1;
        else
            t = t + h;
        end
        [s, left] = stepped(s, t, idx, reshape(y, n, 6));
        if left
            return
        end
    end
    return
end
% dopri5, its step size carried from one call to the next (s.hNext)
k1 = f(t, y);
h = min(s.hNext, it.hMax);
while t1 - t > 1e-9 * max(1, abs(t))
    last = h >= t1 - t;
    hTry = h;
    if last
        hTry = t1 - t;
    end
    if tStop < t1 && t + hTry > tStop
        break
    end
    [yNew, k7, err] = dopriStep(f, t, y, k1, hTry, it, n, s.C);
    if err <= 1
        if last
            t = t1;
        else
            t = t + hTry;
        end
        y = yNew;
        k1 = k7;
        [s, left] = stepped(s, t, idx, reshape(y, n, 6));
        h = dopriAccepted(h, hTry, err, last, it);
        if left
            break
        end
    else
        h = dopriRejected(hTry, err);
    end
end
s.hNext = h;
end

function [s, left] = stepped(s, t, idx, Yidx)
% the end of a step (the simulator's _post_step): the states of objects idx are Yidx;
% a state that is not finite is an error, then objects that have left go, then events
s.t = t;
s.Y(idx, :) = Yidx;
[s, s.Y] = followRecordings(s, s.Y, t);
checkFinite(s, t, idx);
[s, left] = stepLeft(s, t, idx, s.Y);
s = stepEvents(s, t, idx, s.Y);
end

function s = steppedIdle(s, idx)
% the end of a stretch with nothing integrated (the simulator's _post_step with no
% objects in orbit): replayed objects still move along their recordings and are looked at
if s.replays
    [s, s.Y] = followRecordings(s, s.Y, s.t);
    s = stepEvents(s, s.t, idx, s.Y);
end
end

function [y, h] = dopri5(f, t, y, t1, h, it, n, C)
% the simulator's dopri5 from state column y (n objects) at t to t1, starting with a step
% of h, on its own (no events): for a look ahead or a coast. h out: the next step size
k1 = f(t, y);
h = min(h, it.hMax);
while t1 - t > 1e-9 * max(1, abs(t))
    last = h >= t1 - t;
    hTry = h;
    if last
        hTry = t1 - t;
    end
    [yNew, k7, err] = dopriStep(f, t, y, k1, hTry, it, n, C);
    if err <= 1
        if last
            t = t1;
        else
            t = t + hTry;
        end
        y = yNew;
        k1 = k7;
        h = dopriAccepted(h, hTry, err, last, it);
    else
        h = dopriRejected(hTry, err);
    end
end
end

function [yNew, k7, err] = dopriStep(f, t, y, k1, h, it, n, C)
% one Dormand-Prince 5(4) step of h seconds (satflight/integrators.py): the 5th-order
% state, its derivative (the next step's first stage) and the scaled error, <= 1 when
% the step is accepted. An error the step cannot bring down above 1e-6 s is a fault
A = {[], 1 / 5, [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], ...
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], ...
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], ...
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]};
c = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];
E = B5 - B4;
K = zeros(numel(y), 7);
K(:, 1) = k1;
acc = y;
for st = 2:7
    acc = y;
    a = A{st};
    for j = 1:numel(a)
        if a(j) ~= 0
            acc = acc + h * a(j) * K(:, j);
        end
    end
    K(:, st) = f(t + c(st) * h, acc);
end
yNew = acc;                 % the last stage sits at the 5th-order solution (FSAL)
k7 = K(:, 7);
errVec = 0;
for j = 1:7
    if E(j) ~= 0
        errVec = errVec + E(j) * K(:, j);
    end
end
err = dopriError(h * errVec, y, yNew, it, n, C);
if ~(err <= 1) && h < 1e-6        % "not <=": a NaN error fails too
    error("satflight:step", "dopri5: step size fell below 1e-06 s at t = %.3f s with " + ...
        "the error still %.3g x the tolerance (a singular or non-finite state?)", t, err);
end
end

function err = dopriError(errVec, y0, y1, it, n, C)
% scaled RMS error of the worst object (the simulator's _error_norm): velocities are
% weighed in km by each orbit's own time scale sqrt(r^3/mu), so one atol means the same
% for km and km/s
if n == 0
    err = 0;
    return
end
Y0 = reshape(y0, n, 6);
Y1 = reshape(y1, n, 6);
tau = sqrt(sum(Y0(:, 1:3) .^ 2, 2) .^ 1.5 / C.MU);
w = [ones(n, 3), repmat(tau, 1, 3)];
sc = it.atol + it.rtol * max(abs(Y0 .* w), abs(Y1 .* w));
err = max(sqrt(mean((reshape(errVec, n, 6) .* w ./ sc) .^ 2, 2)));
end

function h = dopriAccepted(h, hTry, err, last, it)
% the next step size after an accepted step of hTry (h: the size it was asked to try)
if err == 0
    fac = 5;
else
    fac = min(5, max(0.2, 0.9 * err ^ -0.2));
end
if ~last
    h = min(it.hMax, hTry * fac);
else
    % a short final step says nothing about the natural step size
    h = min(it.hMax, max(h, hTry * fac));
end
end

function h = dopriRejected(hTry, err)
h = hTry * max(0.2, 0.9 * err ^ -0.2);
end

function y = fixedSteps(f, t, y, t1, it, n, C)
% the run's fixed-step method from t to t1 in steps of hFixed, the last cut short
while t < t1
    h = min(it.hFixed, t1 - t);
    y = fixedStep(it.method, f, t, y, h, n, C);
    if t1 - (t + h) < 1e-9
        t = t1;
    else
        t = t + h;
    end
end
end

function f = cowellDerivative(s, idx, te)
% the derivative of objects idx over a segment from now to te: the run's forces and the
% thrust of the finite burns under way all through it
cdam = s.cd(idx) .* s.area(idx) ./ s.mass(idx);
cram = s.cr(idx) .* s.area(idx) ./ s.mass(idx);   % (a burn's mass loss not passed on)
burnRows = zeros(0, 4);      % row in idx, maneuver, t0, m0 (mdot in a parallel list)
mdots = zeros(0, 1);
for b = s.burns
    if b.t0 <= s.t + 1e-9 && te <= b.t1 + 1e-9
        row = find(idx == b.sat, 1);
        if ~isempty(row)
            burnRows(end + 1, :) = [row, b.man, b.t0, b.m0]; %#ok<AGROW>
            mdots(end + 1, 1) = b.mdot; %#ok<AGROW>
        end
    end
end
nIdx = numel(idx);
F = s.sim.forces;
jd0 = s.sim.jd0;
man = s.man;
C = s.C;
f = @(t, y) derivative(t, y, nIdx, cdam, cram, F, jd0, man, burnRows, mdots, C);
end

function y = fixedStep(method, f, t, y, h, n, C)
% one step of h seconds of a fixed-step integrator from state column y (n objects) at t
switch method
    case "rk4"
        k1 = f(t, y);
        k2 = f(t + h / 2, y + h / 2 * k1);
        k3 = f(t + h / 2, y + h / 2 * k2);
        k4 = f(t + h, y + h * k3);
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    case "leapfrog"
        % kick-drift-kick with a straight-line drift
        Y = reshape(y, n, 6);
        a0 = reshape(f(t, y), n, 6);
        vHalf = Y(:, 4:6) + 0.5 * h * a0(:, 4:6);
        rNew = Y(:, 1:3) + h * vHalf;
        a1 = reshape(f(t + h, reshape([rNew, vHalf], [], 1)), n, 6);
        y = reshape([rNew, vHalf + 0.5 * h * a1(:, 4:6)], [], 1);
    case "wh"
        % Wisdom-Holman: half a kick of the perturbations, the exact two-body motion
        % over the step, the other half kick
        Y = reshape(y, n, 6);
        vHalf = Y(:, 4:6) + 0.5 * h * perturbation(f, t, y, n, C);
        [r1, v1] = keplerPropagate(Y(:, 1:3), vHalf, h, C.MU);
        v1 = v1 + 0.5 * h * perturbation(f, t + h, reshape([r1, v1], [], 1), n, C);
        y = reshape([r1, v1], [], 1);
    otherwise
        error("satflight:integrator", "Unknown integrator %s.", method);
end
end

function a = perturbation(f, t, y, n, C)
% the acceleration minus the Earth's point-mass term
Y = reshape(y, n, 6);
d = reshape(f(t, y), n, 6);
r = Y(:, 1:3);
a = d(:, 4:6) + C.MU * r ./ vecnorm(r, 2, 2) .^ 3;
end

function dy = derivative(t, y, n, cdam, cram, F, jd0, man, burnRows, mdots, C)
Y = reshape(y, n, 6);
r = Y(:, 1:3);
v = Y(:, 4:6);
theta = 0;
msis = msisDrag(F);
if F.c22 || F.gravityDegree > 0 || msis   % the terms that turn with the Earth (NRLMSIS:
    theta = gmst(jd0 + t / 86400);        % the ground point under each satellite)
end
masses = burnRows(:, 4) - mdots .* (t - burnRows(:, 3));
if F.drag && ~isempty(burnRows)
    % a burning craft gets lighter: its A/m grows (cdam holds Cd A / m at ignition)
    cdam(burnRows(:, 1)) = cdam(burnRows(:, 1)) .* burnRows(:, 4) ./ masses;
end
jd = [];
if F.sun || F.moon || F.srp || msis
    jd = jd0 + t / 86400;      % where the Sun and Moon are; NRLMSIS's time
end
a = acceleration(r, v, cdam, theta, F, C, jd, cram);
for k = 1:size(burnRows, 1)
    row = burnRows(k, 1);
    m = man(burnRows(k, 2));
    d = finiteDirection(m, r(row, :), v(row, :));
    a(row, :) = a(row, :) + d * (m.thrust / masses(k)) * 1e-3;
end
dy = reshape([v, a], [], 1);
end

function [s, gone] = stepLeft(s, t, idx, Y)
% the simulator's look at the end of every step (_post_step), Y the states then: an
% object below the re-entry height and descending (only a descending one: launches
% start low and climb), beyond the sphere of influence, or one that went below the
% re-entry height during the step and came back up (a dip) has left the run. Each is
% logged at the time it crossed, found on the step's Hermite curve, with its speed then
C = s.C;
idx = reshape(idx, [], 1);
gone = false;
if isempty(idx)
    return
end
r = Y(idx, 1:3);
v = Y(idx, 4:6);
rm = vecnorm(r, 2, 2);
alt = geodeticHeight(r, C);
out = (alt < C.REENTRY_ALT & sum(r .* v, 2) < 0) | rm > C.SOI;
dips = nan(numel(idx), 1);
if numel(idx) <= C.DIP_CHECK_LIMIT
    dips = dipTimes(s, t, idx, Y, alt);
end
for k = reshape(find(out | ~isnan(dips)), 1, [])
    i = idx(k);
    speed = norm(v(k, :));
    if rm(k) > C.SOI
        when = crossingTime(s, t, i, Y, @(pos, ~) vecnorm(pos, 2, 2) - C.SOI);
        s = leave(s, i, "escaped", t, when, speed);
    elseif alt(k) <= 0
        when = crossingTime(s, t, i, Y, @(pos, ~) geodeticHeight(pos, C));
        s = leave(s, i, "impacted", t, when, speedAt(s, t, i, Y, when, speed));
    else
        when = dips(k);
        if isnan(when)
            when = crossingTime(s, t, i, Y, @(pos, ~) geodeticHeight(pos, C) - C.REENTRY_ALT);
        end
        s = leave(s, i, "re-entered", t, when, speedAt(s, t, i, Y, when, speed));
    end
    gone = true;
end
end

function when = dipTimes(s, t, idx, Y, alt)
% the simulator's _dips: for each object of idx (heights alt now) that ends the step
% within DIP_MARGIN of the re-entry height, having started it at or above it, the time
% it first went below it during the step (NaN for the others and if it never did)
C = s.C;
when = nan(numel(idx), 1);
prev = previousStep(s, t, Y);
if isempty(prev)
    return
end
h = t - prev.t;
fr = linspace(0, 1, 17);
for k = reshape(find(alt < C.REENTRY_ALT + C.DIP_MARGIN), 1, [])
    i = idx(k);
    y0 = prev.Y(i, :);
    y1 = Y(i, :);
    if ~(geodeticHeight(y0(1:3), C) >= C.REENTRY_ALT)
        continue                          % already below (climbing out): not a dip
    end
    below = find(geodeticHeight(hermiteAt(y0, y1, h, fr), C) < C.REENTRY_ALT, 1);
    if ~isempty(below)
        low = @(f) geodeticHeight(hermiteAt(y0, y1, h, f), C) - C.REENTRY_ALT;
        when(k) = prev.t + h * crossingFraction(low, 0, fr(below));
    end
end
end

function prev = previousStep(s, t, Y)
% the states at the end of the step before the one that ended at t (states Y), or []
% when nothing continuous joins them (no step yet, objects added since)
prev = s.lastSample;
if isempty(prev) || ~(prev.t < t) || size(prev.Y, 1) ~= size(Y, 1)
    prev = [];
end
end

function when = crossingTime(s, t, i, Y, f)
% the simulator's _when: the time within the step that ended at t (states Y) at which
% f(position, time) changed sign for object i, on the step's cubic Hermite curve; t
% itself with no step to go back along or no change of sign on the curve
when = t;
prev = previousStep(s, t, Y);
if isempty(prev)
    return
end
h = t - prev.t;
y0 = prev.Y(i, :);
y1 = Y(i, :);
g = @(fr) f(hermiteAt(y0, y1, h, fr), prev.t + fr * h);
if (g(0) < 0) == (g(1) < 0)
    return
end
when = prev.t + h * crossingFraction(g, 0, 1);
end

function speed = speedAt(s, t, i, Y, when, now)
% the speed (km/s) of object i at time when within the step that ended at t, along the
% step's Hermite curve (now, its speed at the end, with no step to go back along)
prev = previousStep(s, t, Y);
if isempty(prev) || ~(when < t)
    speed = now;
    return
end
h = t - prev.t;
fr = (when - prev.t) / h;
y0 = prev.Y(i, :);
y1 = Y(i, :);
f2 = fr * fr;
vel = (6 * f2 - 6 * fr) * y0(1:3) / h + (3 * f2 - 4 * fr + 1) * y0(4:6) + ...
    (6 * fr - 6 * f2) * y1(1:3) / h + (3 * f2 - 2 * fr) * y1(4:6);
speed = norm(vel);
end

function p = hermiteAt(y0, y1, h, fr)
% positions (one row per fraction fr of the step) on the cubic Hermite curve through
% states y0 and y1 a step of h seconds apart
fr = fr(:);
f2 = fr .* fr;
f3 = f2 .* fr;
p = (2 * f3 - 3 * f2 + 1) .* y0(1:3) + (f3 - 2 * f2 + fr) .* h .* y0(4:6) + ...
    (3 * f2 - 2 * f3) .* y1(1:3) + (f3 - f2) .* h .* y1(4:6);
end

function fr = crossingFraction(g, lo, hi)
% the fraction of the step between lo and hi where g(fraction) changes sign (it must),
% by bisection to about 1e-7 of the step (the simulator's _crossing)
gLo = g(lo);
for k = 1:24
    mid = 0.5 * (lo + hi);
    gMid = g(mid);
    if (gMid < 0) == (gLo < 0)
        lo = mid;
        gLo = gMid;
    else
        hi = mid;
    end
end
fr = 0.5 * (lo + hi);
end

function s = leave(s, i, status, t, when, speed)
% object i has left the run at the end of the step that ended at t: logged at when,
% the time it crossed, with speed its speed then
C = s.C;
name = s.names(i);
s.status(i) = status;
switch status
    case "escaped"
        s = logEvent(s, when, sprintf("%s left Earth's sphere of influence", name), "alert");
    case "impacted"
        s = logEvent(s, when, sprintf("%s impacted the surface at %.2f km/s", name, speed), ...
            "alert");
    otherwise
        s = logEvent(s, when, sprintf("%s re-entered (below %.0f km, %.2f km/s)", name, ...
            C.REENTRY_ALT, speed), "alert");
end
mine = false(size(s.pending));
if ~isempty(s.man)
    mine = s.pending & reshape(string({s.man.sat}), 1, []) == name;
end
if any(mine)
    plural = "s";
    if nnz(mine) == 1
        plural = "";
    end
    s = logEvent(s, t, sprintf("%s: %d pending maneuver%s cancelled", name, nnz(mine), ...
        plural), "warn");
end
s.pending(mine) = false;
if ~isempty(s.burns)
    s.burns = s.burns([s.burns.sat] ~= i);
end
end

% --- analytic propagation (propagators "kepler" and "j2mean") --------------------------------

function Y = analyticStep(Y0, dt, propagator, C)
if propagator == "kepler"
    [r, v] = keplerPropagate(Y0(:, 1:3), Y0(:, 4:6), dt, C.MU);
else
    [r, v] = j2MeanPropagate(Y0(:, 1:3), Y0(:, 4:6), dt, C);
end
Y = [r, v];
end

function [r, v] = j2MeanPropagate(r, v, dt, C)
el = rv2coe(r, v, C.MU);
closed = el.e < 1;
[r, v] = keplerPropagate(r, v, dt, C.MU);
if any(closed)
    ec = el.e(closed);
    inc = el.i(closed);
    argp0 = el.argp(closed);
    a = osculatingToMeanA(el.a(closed), ec, inc, argp0, el.nu(closed), C);
    [rd, wd, md] = j2SecularRates(a, ec, inc, C);
    n = sqrt(C.MU ./ a .^ 3);
    M = el.M(closed) + (n + md) * dt;
    raan = el.raan(closed) + rd * dt;
    argp = argp0 + wd * dt;
    nu = meanToTrue(M, ec);
    aOsc = meanToOsculatingA(a, ec, inc, argp, nu, C);
    [rc, vc] = coe2rv(aOsc, ec, inc, raan, argp, nu, C.MU);
    r(closed, :) = rc;
    v(closed, :) = vc;
end
end

function [raanDot, argpDot, mDot] = j2SecularRates(a, e, i, C)
n = sqrt(C.MU ./ abs(a) .^ 3);
p = a .* (1 - e .* e);
k = n .* C.J2 .* (C.RE ./ p) .^ 2;
si2 = sin(i) .^ 2;
raanDot = -1.5 * k .* cos(i);
argpDot = 0.75 * k .* (4 - 5 * si2);
mDot = 0.75 * k .* sqrt(max(1 - e .* e, 0)) .* (2 - 3 * si2);
end

function shape = j2ShortPeriodShape(e, i, argp, nu)
q = 1 - e .* e;
ar3 = ((1 + e .* cos(nu)) ./ q) .^ 3;
s2 = sin(i) .^ 2;
shape = (2 / 3) * (1 - 1.5 * s2) .* (ar3 - q .^ -1.5) + s2 .* ar3 .* cos(2 * (argp + nu));
end

function a = meanToOsculatingA(aMean, e, i, argp, nu, C)
a = aMean + 1.5 * C.J2 * C.RE ^ 2 * j2ShortPeriodShape(e, i, argp, nu) ./ aMean;
end

function i = osculatingToMeanI(iOsc, a, e, argp, nu, C)
% Kozai's first-order J2 short-period term in the inclination, removed
p = abs(a) .* (1 - e .* e);
k = 0.375 * C.J2 * (C.RE ./ p) .^ 2 .* sin(2 * iOsc);
i = iOsc - k .* (cos(2 * (argp + nu)) + e .* cos(2 * argp + nu) + e / 3 .* cos(2 * argp + 3 * nu));
end

function a = osculatingToMeanA(aOsc, e, i, argp, nu, C)
k = 1.5 * C.J2 * C.RE ^ 2 * j2ShortPeriodShape(e, i, argp, nu);
a = 0.5 * (aOsc + sqrt(max(aOsc .* aOsc - 4 * k, 0)));
end

function [r, v] = keplerPropagate(r0, v0, dt, mu)
% universal-variable two-body propagation (Vallado alg. 8, Laguerre-Conway iteration)
n = size(r0, 1);
dt = dt .* ones(n, 1);
sqmu = sqrt(mu);
r0m = vecnorm(r0, 2, 2);
v0m = vecnorm(v0, 2, 2);
rv = sum(r0 .* v0, 2);
alpha = 2 ./ r0m - v0m .* v0m / mu;
ell = alpha > 1e-12;
period = inf(n, 1);
period(ell) = 2 * pi ./ sqrt(mu * abs(alpha(ell)) .^ 3);
dt(ell) = rem(dt(ell), period(ell));
chi = sqmu * dt ./ r0m;
chi(ell) = sqmu * dt(ell) .* alpha(ell);
hyp = alpha < -1e-12;
aH = -ones(n, 1);
aH(hyp) = 1 ./ alpha(hyp);
sgn = sign(dt);
arg = (-2 * mu * alpha .* dt) ./ (rv + sgn .* sqrt(-mu * aH) .* (1 - r0m .* alpha));
useH = hyp & arg > 0;
chi(useH) = sgn(useH) .* sqrt(-aH(useH)) .* log(arg(useH));
k = 5;
for iter = 1:60
    psi = chi .* chi .* alpha;
    [c2, c3] = stumpff(psi);
    F = rv / sqmu .* chi .* chi .* c2 + (1 - r0m .* alpha) .* chi .^ 3 .* c3 + r0m .* chi ...
        - sqmu * dt;
    dF = chi .* chi .* c2 + rv / sqmu .* chi .* (1 - psi .* c3) + r0m .* (1 - psi .* c2);
    ddF = rv / sqmu .* (1 - psi .* c2) + (1 - r0m .* alpha) .* chi .* (1 - psi .* c3);
    disc = sqrt(abs((k - 1) ^ 2 * dF .* dF - k * (k - 1) * F .* ddF));
    denom = dF + sign(dF) .* disc;
    denom(denom == 0) = 1;
    delta = k * F ./ denom;
    chi = chi - delta;
    if all(abs(delta) <= 1e-12 * max(1, abs(chi)))
        break
    end
end
psi = chi .* chi .* alpha;
[c2, c3] = stumpff(psi);
rm = chi .* chi .* c2 + rv / sqmu .* chi .* (1 - psi .* c3) + r0m .* (1 - psi .* c2);
f = 1 - chi .* chi ./ r0m .* c2;
g = dt - chi .^ 3 / sqmu .* c3;
gdot = 1 - chi .* chi ./ rm .* c2;
fdot = sqmu ./ (rm .* r0m) .* chi .* (psi .* c3 - 1);
r = f .* r0 + g .* v0;
v = fdot .* r0 + gdot .* v0;
end

function [c2, c3] = stumpff(psi)
c2 = zeros(size(psi));
c3 = zeros(size(psi));
small = abs(psi) < 1e-3;
pos = psi >= 1e-3;
neg = psi <= -1e-3;
ps = psi(small);
c2(small) = 0.5 - ps / 24 + ps .* ps / 720 - ps .^ 3 / 40320;
c3(small) = 1 / 6 - ps / 120 + ps .* ps / 5040 - ps .^ 3 / 362880;
sp = sqrt(psi(pos));
c2(pos) = (1 - cos(sp)) ./ psi(pos);
c3(pos) = (sp - sin(sp)) ./ sp .^ 3;
sn = sqrt(min(-psi(neg), 700 ^ 2));
c2(neg) = (cosh(sn) - 1) ./ (-psi(neg));
c3(neg) = (sinh(sn) - sn) ./ sn .^ 3;
end

% --- elements ------------------------------------------------------------------------------

function el = rv2coe(r, v, mu)
% osculating elements of each row (Vallado alg. 9, the simulator's conventions for
% equatorial and circular orbits)
rm = vecnorm(r, 2, 2);
vm = vecnorm(v, 2, 2);
hv = cross(r, v, 2);
hm = vecnorm(hv, 2, 2);
hmSafe = hm;
hmSafe(hm == 0) = 1;
what = hv ./ hmSafe;
nv = [-hv(:, 2), hv(:, 1), zeros(size(hm))];
nm = vecnorm(nv, 2, 2);
equatorial = nm < 1e-10 * hmSafe;
nmSafe = nm;
nmSafe(nm == 0) = 1;
nhat = nv ./ nmSafe;
nhat(equatorial, :) = repmat([1 0 0], nnz(equatorial), 1);
rdv = sum(r .* v, 2);
evec = ((vm .* vm - mu ./ rm) .* r - rdv .* v) / mu;
e = vecnorm(evec, 2, 2);
circular = e < 1e-10;
eSafe = e;
eSafe(e == 0) = 1;
ehat = evec ./ eSafe;
ehat(circular, :) = nhat(circular, :);
p = hm .* hm / mu;
a = inf(size(e));
open = abs(1 - e) > 1e-12;
a(open) = p(open) ./ (1 - e(open) .* e(open));
el.a = a;
el.e = e;
el.i = acos(min(max(hv(:, 3) ./ hmSafe, -1), 1));
el.raan = mod(atan2(nv(:, 2), nv(:, 1)), 2 * pi);
el.raan(equatorial) = 0;
el.argp = angleAbout(what, nhat, ehat);
el.argp(circular) = 0;
el.nu = angleAbout(what, ehat, r);
el.M = trueToMean(el.nu, e);
el.p = p;
el.rp = p ./ (1 + e);
el.ra = inf(size(e));
el.ra(e < 1) = p(e < 1) ./ (1 - e(e < 1));
el.n = sqrt(mu ./ abs(a) .^ 3);
end

function ang = angleAbout(axis, from, to)
ang = mod(atan2(sum(axis .* cross(from, to, 2), 2), sum(from .* to, 2)), 2 * pi);
end

function M = trueToMean(nu, e)
M = zeros(size(nu));
ell = e < 1 - 1e-9;
hyp = e > 1 + 1e-9;
par = ~(ell | hyp);
ee = e(ell);
E = atan2(sqrt(1 - ee .* ee) .* sin(nu(ell)), ee + cos(nu(ell)));
M(ell) = mod(E - ee .* sin(E), 2 * pi);
ee = e(hyp);
x = sqrt((ee - 1) ./ (ee + 1)) .* tan(nu(hyp) / 2);
Fh = 2 * atanh(min(max(x, -1 + 1e-15), 1 - 1e-15));
M(hyp) = ee .* sinh(Fh) - Fh;
D = tan(nu(par) / 2);
M(par) = D + D .^ 3 / 3;
end

function nu = meanToTrue(M, e)
% elliptic orbits (the only ones the J2-mean propagator hands over)
M = mod(M, 2 * pi);
E = M;
E(e >= 0.8) = pi;
for iter = 1:50
    dE = (E - e .* sin(E) - M) ./ (1 - e .* cos(E));
    E = E - dE;
    if all(abs(dE) < 1e-14)
        break
    end
end
nu = mod(atan2(sqrt(1 - e .* e) .* sin(E), cos(E) - e), 2 * pi);
end

function [r, v] = coe2rv(a, e, i, raan, argp, nu, mu)
p = a .* (1 - e .* e);
cn = cos(nu);
sn = sin(nu);
rr = p ./ (1 + e .* cn);
ci = cos(i); si = sin(i);
co = cos(raan); so = sin(raan);
cw = cos(argp); sw = sin(argp);
P = [co .* cw - so .* sw .* ci, so .* cw + co .* sw .* ci, sw .* si];
Q = [-co .* sw - so .* cw .* ci, -so .* sw + co .* cw .* ci, cw .* si];
sq = sqrt(mu ./ p);
r = (rr .* cn) .* P + (rr .* sn) .* Q;
v = (-sq .* sn) .* P + (sq .* (e + cn)) .* Q;
end

% --- forces --------------------------------------------------------------------------------

function a = acceleration(r, v, cdam, theta, F, C, jd, cram)
% jd: the Julian date (UTC, one for every row) the Sun and Moon are placed at; cram:
% Cr A / m (m^2/kg) of each row for sunlight pressure
x = r(:, 1); y = r(:, 2); z = r(:, 3);
r2 = x .* x + y .* y + z .* z;
rm = sqrt(r2);
a = -C.MU * r ./ rm .^ 3;
if F.gravityDegree > 0
    % the EGM2008 field to degree N stands in for J2, J3, J4 and C22
    a = a + fieldAcceleration(r, theta, F.field);
else
    if F.j2
        k = -1.5 * C.J2 * C.MU * C.RE ^ 2 ./ rm .^ 5;
        zz = 5 * z .* z ./ r2;
        a = a + [k .* x .* (1 - zz), k .* y .* (1 - zz), k .* z .* (3 - zz)];
    end
    if F.j3
        k = -2.5 * C.J3 * C.MU * C.RE ^ 3 ./ rm .^ 7;
        txy = 3 * z - 7 * z .^ 3 ./ r2;
        az = 6 * z .* z - 7 * z .^ 4 ./ r2 - 0.6 * r2;
        a = a + [k .* x .* txy, k .* y .* txy, k .* az];
    end
    if F.j4
        k = 1.875 * C.J4 * C.MU * C.RE ^ 4 ./ rm .^ 7;
        s = z .* z ./ r2;
        txy = 1 - 14 * s + 21 * s .* s;
        tz = 5 - 70 / 3 * s + 21 * s .* s;
        a = a + [k .* x .* txy, k .* y .* txy, k .* z .* tz];
    end
    if F.c22
        c = cos(theta); s = sin(theta);    % theta: one angle, or one per row of r
        xe = c .* x + s .* y;
        ye = -s .* x + c .* y;
        k = 3 * C.MU * C.RE ^ 2 ./ (r2 .* r2 .* rm);
        w = C.C22 * (xe .* xe - ye .* ye) + 2 * C.S22 * xe .* ye;
        q = 5 * w ./ r2;
        gx = k .* (2 * (C.C22 * xe + C.S22 * ye) - q .* xe);
        gy = k .* (2 * (C.S22 * xe - C.C22 * ye) - q .* ye);
        gz = -k .* q .* z;
        a = a + [c .* gx - s .* gy, s .* gx + c .* gy, gz];
    end
end
if F.drag
    vRel = v - [-C.OMEGA * y, C.OMEGA * x, zeros(size(x))];
    if msisDrag(F)
        % NRLMSIS 2.1 at each satellite's ground point and height, at the date
        c = cos(theta); s = sin(theta);
        [lat, lon, h] = ecefToGeodetic([c .* x + s .* y, -s .* x + c .* y, z], C);
        if physicalCd(F)
            % a sphere's Cd in that air; cdam holds Cd A / m at C.CD_REF, so the
            % satellite's own Cd is a correction factor on it (forces.accel_drag)
            [rho, nd, temp] = msisState(h, lat, lon, jd, F, C);
            air = rho > 0;
            if any(air)
                speed = vecnorm(vRel, 2, 2);
                factor = ones(size(rho));
                factor(air) = sphereCd(speed(air) * 1e3, nd(air, :), temp(air), C) / C.CD_REF;
                cdam = cdam .* factor;
            end
        else
            rho = msisDensity(h, lat, lon, jd, F, C);
        end
    else
        rho = density(geodeticHeight(r, C), F.densityScale, C);
    end
    speed = vecnorm(vRel, 2, 2);
    a = a + (-0.5e3 * rho .* cdam .* speed) .* vRel;
end
if F.sun || F.srp
    rs = sunPosition(jd, C);
    if F.sun
        a = a + thirdBody(r, rs, C.MU_SUN);
    end
    if F.srp
        % a cannonball pushed away from the Sun, dimmed in the Earth's shadow
        d = r - rs;
        dm = vecnorm(d, 2, 2);
        lit = shadowFraction(r, rs, C);
        mag = C.P_SRP * cram .* (C.AU ./ dm) .^ 2 * 1e-3 .* lit;
        a = a + mag .* d ./ dm;
    end
end
if F.moon
    a = a + thirdBody(r, moonPosition(jd, C), C.MU_MOON);
end
end

function a = thirdBody(r, rBody, mu)
% the tidal pull of a distant body at rBody (km, one row): its pull on each row of r
% less its pull on the Earth (the frame's origin)
d = rBody - r;
a = mu * (d ./ vecnorm(d, 2, 2) .^ 3 - rBody / norm(rBody) ^ 3);
end

function h = geodeticHeight(r, C)
% height (km) above the WGS-84 ellipsoid of positions r (one per row), exact; ECI and
% ECEF give the same (the ellipsoid is symmetric about the polar axis)
[~, h] = heikkinen(hypot(r(:, 1), r(:, 2)), r(:, 3), C);
end

function [lat, h] = heikkinen(p, z, C)
% geodetic latitude (rad) and height (km) at distance p from the polar axis and z along
% it: Heikkinen's exact closed form (1982; Zhu 1993). Deeper than half an Earth radius
% (an orbit's "perigee" far underground) its square roots fail, and the geocentric
% estimate stands in (the simulator's frames._heikkinen)
deep = p .* p + z .* z < (0.5 * C.RE) ^ 2;
pIn = p;
zIn = z;
p(deep) = C.RE;
z(deep) = 0;
a = C.RE;
e2 = C.E2;
b2 = a * a * (1 - e2);
ep2 = e2 / (1 - e2);
z2 = z .* z;
p2 = p .* p;
f = 54 * b2 * z2;
g = p2 + (1 - e2) * z2 - e2 * (a * a - b2);
c = e2 * e2 * f .* p2 ./ (g .* g .* g);
s = nthroot(1 + c + sqrt(c .* c + 2 * c), 3);
k = s + 1 + 1 ./ s;
pp = f ./ (3 * k .* k .* g .* g);
q = sqrt(1 + 2 * e2 * e2 * pp);
r0 = -pp * e2 .* p ./ (1 + q) + sqrt(max(0.5 * a * a * (1 + 1 ./ q) ...
    - pp * (1 - e2) .* z2 ./ (q .* (1 + q)) - 0.5 * pp .* p2, 0));
d2 = (p - e2 * r0) .^ 2;
u = sqrt(d2 + z2);
v = sqrt(d2 + (1 - e2) * z2);
z0 = b2 * z ./ (a * v);
lat = atan2(z + ep2 * z0, p);
h = u .* (1 - b2 ./ (a * v));
if any(deep)
    zd = zIn(deep);
    rm = hypot(pIn(deep), zd);
    s2 = zeros(size(rm));
    nz = rm > 0;
    s2(nz) = zd(nz) .^ 2 ./ rm(nz) .^ 2;
    lat(deep) = atan2(zd, pIn(deep) * (1 - e2));
    h(deep) = rm - C.RE * (1 - C.F * s2);
end
end

function tf = msisDrag(F)
% whether drag looks the air up in NRLMSIS (forces "atmosphere": "msis")
tf = F.drag && isfield(F, "atmosphere") && F.atmosphere == "msis";
end

function tf = physicalCd(F)
% whether drag takes a sphere's Cd in the NRLMSIS air (forces "cd_model": "dria_sphere");
% any other model is refused, not flown as fixed
tf = false;
if ~isfield(F, "cdModel") || F.cdModel == "fixed"
    return
end
if F.cdModel ~= "dria_sphere"
    error("satflight:cdModel", "Unknown drag coefficient model ""%s"" (fixed or " + ...
        "dria_sphere).", F.cdModel);
end
if F.drag && ~msisDrag(F)
    error("satflight:cdModel", "The physical drag coefficient needs the NRLMSIS atmosphere.");
end
tf = F.drag;
end

function F = msisSetup(F, folder, C)
% NRLMSIS's inputs, once per run: the measured space weather and pymsis
F.sw = spaceWeatherLoad(folder, C);
F.msis = pymsisModule();
end

function m = pymsisModule()
% pymsis in MATLAB's Python: the python.exe $SATFLIGHT_MSIS_PYTHON names if MATLAB has not
% loaded Python yet, else the one pyenv is set to
want = strtrim(string(getenv("SATFLIGHT_MSIS_PYTHON")));
pe = pyenv;
if want ~= "" && pe.Status == "NotLoaded"
    pe = pyenv(Version=want);
end
try
    m = py.importlib.import_module("pymsis");
catch err
    error("satflight:pymsis", "NRLMSIS drag runs pymsis in MATLAB's Python (%s), " + ...
        "which could not load it: %s\nInstall it there (python -m pip install pymsis), or " + ...
        "point MATLAB at a Python that has it: pyenv(Version=""C:\path\python.exe"") " + ...
        "before Python loads, or the SATFLIGHT_MSIS_PYTHON environment variable.", ...
        pe.Executable, err.message);
end
end

function ms = msisMillis(jd)
% a UTC Julian date as whole milliseconds since 1970 (as the simulator rounds it)
ms = round((jd - 2440587.5) * 86400e3);
end

function rho = msisDensity(h, lat, lon, jd, F, C)
% NRLMSIS 2.1 total mass density (kg/m^3) at geodetic heights (km), latitudes and
% longitudes (rad), all at the one UTC Julian date jd, times F.densityScale; zero above
% 2500 km like the table (atmosphere.msis_density)
rho = zeros(size(h));
air = h <= C.ATM_TOP;
if ~any(air)
    return
end
out = msisCall(h(air), lat(air), lon(air), jd, F);
rho(air) = out(:, 1) * F.densityScale;
end

function [rho, nd, temp] = msisState(h, lat, lon, jd, F, C)
% msisDensity's air with pymsis's nine species (m^-3, in C.SPECIES_KG's order; none
% where MSIS leaves one out) and its temperature (K); refused when the species do not
% add up to the total density within C.SPECIES_SUM_TOL (atmosphere.msis_state)
rho = zeros(size(h));
nd = zeros(numel(h), numel(C.SPECIES_KG));
temp = zeros(size(h));
air = h <= C.ATM_TOP;
if ~any(air)
    return
end
out = msisCall(h(air), lat(air), lon(air), jd, F);
if any(~isfinite(out(:, 1))) || any(~isfinite(out(:, 11)))
    error("satflight:msis", "NRLMSIS returned a non-finite density or temperature.");
end
species = out(:, 2:10);
species(isnan(species)) = 0;
total = species * C.SPECIES_KG(:);
off = abs(total ./ out(:, 1) - 1) > C.SPECIES_SUM_TOL;
if any(off)
    k = find(off, 1);
    error("satflight:msis", "NRLMSIS species add up to %.4g kg/m^3 but its density " + ...
        "is %.4g.", total(k), out(k, 1));
end
rho(air) = out(:, 1) * F.densityScale;
nd(air, :) = species;
temp(air) = out(:, 11);
end

function out = msisCall(h, lat, lon, jd, F)
% pymsis's output (one row per point: density, nine species, temperature) at heights
% (km), latitudes and longitudes (rad), all at the one UTC Julian date jd
if ~isscalar(jd)
    error("satflight:internal", "NRLMSIS takes one date per call, got %d.", numel(jd));
end
n = numel(h);
ms = msisMillis(jd);
[f107, f107a, ap] = spaceWeatherIndices(F.sw, ms);
whole = floor(ms / 86400000);
rest = ms - whole * 86400000;
d = datetime(1970, 1, 1) + days(whole);
utc = sprintf("%04d-%02d-%02dT%02d:%02d:%02d.%03d", d.Year, d.Month, d.Day, ...
    floor(rest / 3600000), floor(mod(rest, 3600000) / 60000), floor(mod(rest, 60000) / 1000), ...
    mod(rest, 1000));
aps = cell(1, n);
for k = 1:n
    aps{k} = py.list(num2cell(ap));
end
out = F.msis.calculate(py.list(repmat({char(utc)}, 1, n)), rad2deg(lon(:).'), ...
    rad2deg(lat(:).'), max(h(:), 0).', repmat(f107, 1, n), repmat(f107a, 1, n), ...
    py.list(aps), pyargs("geomagnetic_activity", int32(-1)));
out = double(out);
end

function cd = sphereCd(v, nd, temp, C)
% the drag coefficient of a sphere in free-molecular flow (drag_coefficient.sphere_cd):
% Sentman's closed form for each species, diffuse reemission from a C.WALL_K wall (rms
% form), SESAM accommodation from the atomic oxygen, weighted by mass density; v the
% speed through the air (m/s, N x 1), nd the species (m^-3, N x 9), temp the air's (K)
m = C.SPECIES_KG(:).';
rhoJ = nd .* m;
rho = sum(rhoJ, 2);
if any(rho <= 0)
    error("satflight:cd", "Drag coefficient: no air (every species density is zero).");
end
p = C.SESAM_K * nd(:, 3) .* temp;                   % column 3: atomic oxygen
alpha = p ./ (1 + p);
s = v ./ sqrt(2 * C.K_B * temp ./ m);
if any(s(:) < C.MIN_SPEED_RATIO)
    error("satflight:cd", "Drag coefficient: speed ratio below %g.", C.MIN_SPEED_RATIO);
end
vr = sqrt(1 + alpha .* (3 * (C.K_B ./ m) * C.WALL_K ./ v .^ 2 - 1));
s2 = s .^ 2;
cdJ = exp(-s2) .* (1 + 2 * s2) ./ (sqrt(pi) * s2 .* s) ...
    + (4 * s2 .^ 2 + 4 * s2 - 1) ./ (2 * s2 .^ 2) .* erf(s) + 2 * sqrt(pi) / 3 * vr;
cd = sum(rhoJ .* cdJ, 2) ./ rho;
end

function W = spaceWeatherLoad(folder, C)
% CelesTrak's SW-All.csv from the data folder, read strictly as the simulator reads it
% (spaceweather.read): the measured days (OBS, and INT: CelesTrak's fill for a day
% without a reading); predictions end it
file = fullfile(dataRoot(folder), "spaceweather", "SW-All.csv");
if ~isfile(file)
    error("satflight:noData", "No space-weather table at %s: run python " + ...
        "tools/fetch_space_weather.py (NRLMSIS drag needs it).", file);
end
lines = readlines(file, EmptyLineRule="skip");
if isempty(lines) || strtrim(lines(1)) ~= C.SW_HEADER
    error("satflight:spaceWeather", "%s: not CelesTrak's SW-All columns.", file);
end
cols = split(C.SW_HEADER, ",");
counts = count(lines(2:end), ",") + 1;
if any(counts ~= numel(cols))
    k = find(counts ~= numel(cols), 1);
    error("satflight:spaceWeather", "line %d: %d fields, expected %d.", k + 1, counts(k), ...
        numel(cols));
end
fields = split(strtrim(lines(2:end)), ",", 2);
kind = fields(:, cols == "F10.7_DATA_TYPE");
measured = kind == "OBS" | kind == "INT";
known = measured | kind == "PRD" | kind == "PRM";
if ~all(known)
    k = find(~known, 1);
    error("satflight:spaceWeather", "line %d: F10.7 type ""%s"", expected OBS, INT, PRD " + ...
        "or PRM.", k + 1, kind(k));
end
last = find(~measured, 1) - 1;
if isempty(last)
    last = numel(measured);
end
if any(measured(last + 1:end))
    k = last + find(measured(last + 1:end), 1);
    error("satflight:spaceWeather", "line %d: a measured day after a prediction (line %d).", ...
        k + 1, last + 2);
end
fields = fields(1:last, :);
num = @(name) str2double(fields(:, cols == name));
ap3 = str2double(fields(:, startsWith(cols, "AP") & strlength(cols) == 3));
W = struct("day", days(datetime(fields(:, 1), InputFormat="yyyy-MM-dd") ...
    - datetime(1970, 1, 1)), "ap3", ap3, "apd", num("AP_AVG"), "f107", num("F10.7_OBS"), ...
    "f107a", num("F10.7_OBS_CENTER81"));
bad = any(isnan([W.day, W.ap3, W.apd, W.f107, W.f107a]), 2) | any(W.ap3 < 0, 2);
if any(bad)
    k = find(bad, 1);
    error("satflight:spaceWeather", "line %d: not an SW-All row: %s", k + 1, lines(k + 1));
end
gap = find(diff(W.day) ~= 1, 1);
if ~isempty(gap)
    error("satflight:spaceWeather", "line %d: %s does not follow %s.", gap + 2, ...
        fields(gap + 1, 1), fields(gap, 1));
end
if numel(W.day) < 2
    error("satflight:spaceWeather", "%s: fewer than two measured days.", file);
end
end

function [f107, f107a, ap] = spaceWeatherIndices(W, ms)
% NRLMSIS's inputs at a time (ms since 1970, UTC), as spaceweather.Table.indices: F10.7 of
% the previous day, its 81-day mean centred on the day, and the ap array (the day's Ap; the
% 3-hour ap now and 3, 6, 9 h before; the means of the eight 12-33 and 36-57 h before)
minutes = floor(ms / 60000) - W.day(1) * 1440;
d = floor(minutes / 1440);                 % day index from the first, 0-based
k = floor(minutes / 180);                  % 3-hour slot index, 0-based
last = numel(W.day) - 1;
when = string(datetime(ms / 1000, ConvertFrom="posixtime", TimeZone="UTC", ...
    Format="yyyy-MM-dd'T'HH:mm"));
first = string(datetime(1970, 1, 1) + days(W.day(1)), "yyyy-MM-dd");
final = string(datetime(1970, 1, 1) + days(W.day(end)), "yyyy-MM-dd");
if d < 1 || d > last
    error("satflight:spaceWeather", "%s: outside the measured space-weather table " + ...
        "(%s..%s); fetch a newer table (python tools/fetch_space_weather.py)", when, ...
        first, final);
end
if k < 19
    error("satflight:spaceWeather", "%s: the ap history (57 h before it) starts before " + ...
        "the table's %s", when, first);
end
if d + 40 > last
    error("satflight:spaceWeather", "%s: its 81-day mean F10.7 reaches past the last " + ...
        "measured day %s into predictions; only measured space weather is used", when, final);
end
f107 = W.f107(d);                          % the previous day's (1-based d)
if f107 > 400 || f107 <= 0
    error("satflight:spaceWeather", "%s: it needs the F10.7 of %s, %.1f sfu: a solar " + ...
        "radio burst (over 400), not the quiet-Sun flux the model takes", when, ...
        string(datetime(1970, 1, 1) + days(W.day(d)), "yyyy-MM-dd"), f107);
end
f107a = W.f107a(d + 1);
s = reshape(W.ap3.', [], 1);               % every 3-hour ap in order
i = k + 1;                                 % the slot now, 1-based
ap = [W.apd(d + 1), s(i), s(i - 1), s(i - 2), s(i - 3), sum(s(i - 11:i - 4)) / 8, ...
    sum(s(i - 19:i - 12)) / 8];
end

function rho = density(h, scale, C)
% piecewise-exponential atmosphere (Vallado table 8-4), kg/m^3
hc = max(h, 0);
k = discretize(hc, [C.ATM(:, 1); inf]);
rho = C.ATM(k, 2) .* exp(-(hc - C.ATM(k, 1)) ./ C.ATM(k, 3)) * scale;
rho(h > C.ATM_TOP) = 0;
end

% --- time and frames -----------------------------------------------------------------------

function theta = gmst(jd)
% Greenwich mean sidereal time, rad (IAU-82, Vallado eq. 3-47)
t = (jd - 2451545.0) / 36525.0;
sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t .* t ...
    - 6.2e-6 * t .* t .* t;
theta = deg2rad(mod(sec, 86400) / 240);
end

function [lat, lon, h] = ecefToGeodetic(r, C)
% ECEF km -> geodetic latitude, longitude (rad) and height (km), exact
lon = atan2(r(:, 2), r(:, 1));
[lat, h] = heikkinen(hypot(r(:, 1), r(:, 2)), r(:, 3), C);
end

% --- precise orbits (replay.py, poeorb.py, eop.py) --------------------------------------------

function [s, Y] = followRecordings(s, Y, t)
% the simulator's _follow_recordings: each replayed object of states Y where its recording
% has it at t; one whose recording has ended stops at its end (its physics state too, so
% the steps after keep it there) and the log says so
for i = reshape(find(s.followed & s.status == "active"), 1, [])
    E = s.replay{i};
    if t > E.t(end) + 1e-9
        Y(i, :) = replayState(E, E.t(end));
        s.Y(i, :) = Y(i, :);
        s.status(i) = "end of data";
        s = logEvent(s, E.t(end), sprintf("%s: end of the recorded orbit (%s data until %s " + ...
            "24:00 UTC)", s.names(i), E.mission, E.until), "warn");
    else
        Y(i, :) = replayState(E, t);
    end
end
end

function y = replayState(E, t)
% the recorded state at t (s since the epoch): cubic Hermite on position and velocity
% between the 10 s samples around it (replay.Ephemeris.state)
if ~(E.t(1) - 1e-9 <= t && t <= E.t(end) + 1e-9)
    error("satflight:replay", "%s: t = %.1f s is outside the recording (%.0f..%.0f s)", ...
        E.mission, t, E.t(1), E.t(end));
end
k = min(max(sum(E.t <= t), 1), numel(E.t) - 1);
t0 = E.t(k);
h = E.t(k + 1) - t0;
u = (t - t0) / h;
r0 = E.r(k, :);
r1 = E.r(k + 1, :);
m0 = E.v(k, :) * h;
m1 = E.v(k + 1, :) * h;
u2 = u * u;
u3 = u * u * u;
r = (2 * u3 - 3 * u2 + 1) * r0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * r1 ...
    + (u3 - u2) * m1;
v = ((6 * u2 - 6 * u) * r0 + (3 * u2 - 4 * u + 1) * m0 + (-6 * u2 + 6 * u) * r1 ...
    + (3 * u2 - 2 * u) * m1) / h;
y = [r, v];
end

function E = replayLoad(mission, epoch, until, folder, C)
% the recording a replayed object follows (replay.for_orbit): the mission's precise orbit
% from the epoch to the end of day until (UTC), in the run's ECI frame (polar motion taken
% out with the IERS table, the Earth turned back by GMST at UT1), timed in s from the epoch
root = dataRoot(folder);
first = dateshift(epoch, "start", "day");
last = datetime(until, InputFormat="yyyy-MM-dd", TimeZone="UTC");
if last < first
    error("satflight:replay", "replay until %s ends before the scenario starts (%s)", ...
        until, string(first, "yyyy-MM-dd"));
end
e0 = posixtime(epoch);
A = poeorbArc(mission, first, last, root, C);
if A.unix(1) > e0
    % a day's own samples start at 00:00:02: the moments before come from the day before
    A = poeorbArc(mission, first - days(1), last, root, C);
end
P = eopTable(fullfile(root, "eop", "finals2000A.all"));
[r, v] = poeorbEci(A, P, C);
E = struct("mission", mission, "until", until, "t", A.unix - e0, "r", r, "v", v);
end

function root = dataRoot(folder)
% where the downloaded data lives (poeorb.data_root): the folder given, else
% $SATFLIGHT_DATA, else satflight-data in the home folder
root = string(folder);
if strtrim(root) == ""
    root = strtrim(string(getenv("SATFLIGHT_DATA")));
end
if root == ""
    if ispc
        home = getenv("USERPROFILE");
    else
        home = getenv("HOME");
    end
    root = fullfile(string(home), "satflight-data");
end
end

function [r, v] = poeorbEci(A, P, C)
% Earth-fixed states into the run's ECI frame (poeorb.to_eci): the ITRF onto the spin
% axis by polar motion (W = R2(xp) R1(yp)), then the Earth turned back by GMST at UT1
jd = 2440587.5 + A.unix / 86400;
[xp, yp] = eopPole(P, jd);
cx = cos(xp); sx = sin(xp); cy = cos(yp); sy = sin(yp);
% W = R2(xp) * R1(yp), row by row
W = {cx, sx .* sy, -sx .* cy; zeros(size(xp)), cy, sy; sx, -cx .* sy, cx .* cy};
r = zeros(size(A.r));
v = zeros(size(A.v));
for a = 1:3
    r(:, a) = W{a, 1} .* A.r(:, 1) + W{a, 2} .* A.r(:, 2) + W{a, 3} .* A.r(:, 3);
    v(:, a) = W{a, 1} .* A.v(:, 1) + W{a, 2} .* A.v(:, 2) + W{a, 3} .* A.v(:, 3);
end
theta = gmst(jd + A.dut1 / 86400);
c = cos(theta);
s = sin(theta);
r = [c .* r(:, 1) - s .* r(:, 2), s .* r(:, 1) + c .* r(:, 2), r(:, 3)];
v = [c .* v(:, 1) - s .* v(:, 2), s .* v(:, 1) + c .* v(:, 2), v(:, 3)];
v = v + [-C.OMEGA * r(:, 2), C.OMEGA * r(:, 1), zeros(size(theta))];
end

function A = poeorbArc(mission, first, last, root, C)
% days first..last of the mission's POEORB files as one arc (poeorb.load_arc): each day
% from its own file, 00:00 to 24:00 UTC, every 10 s; a day with no file, or neighbouring
% days that disagree by more than 10 m where they overlap, is an error (nothing bridged)
have = poeorbFiles(mission, root);
dayList = first:caldays(1):last;
want = string(dayList, "yyyy-MM-dd");
missing = want(~ismember(want, have.day));
if ~isempty(missing)
    error("satflight:noData", "%s: no downloaded file for %d day(s), first %s (run " + ...
        "tools/fetch_poeorb.py)", mission, numel(missing), missing(1));
end
A = struct("unix", zeros(0, 1), "dut1", zeros(0, 1), "r", zeros(0, 3), "v", zeros(0, 3));
prev = [];
for k = 1:numel(dayList)
    f = poeorbRead(have.path(have.day == want(k)), C);
    if ~isempty(prev)
        [~, ip, ic] = intersect(prev.unix, f.unix);
        keep = ~(prev.degraded(ip) | f.degraded(ic));
        if any(keep)
            gap = max(vecnorm(prev.r(ip(keep), :) - f.r(ic(keep), :), 2, 2)) * 1e3;
            if gap > C.SEAM_TOL_M
                error("satflight:poeorb", "%s: differs from the day before by %.3f m where " + ...
                    "they overlap (limit %g m)", f.name, gap, C.SEAM_TOL_M);
            end
        end
    end
    d0 = posixtime(dayList(k));
    in = f.unix >= d0 & f.unix < d0 + 86400;
    A.unix = [A.unix; f.unix(in)];
    A.dut1 = [A.dut1; f.dut1(in)];
    A.r = [A.r; f.r(in, :)];
    A.v = [A.v; f.v(in, :)];
    prev = f;
end
expect = numel(dayList) * 86400 / 10;
if numel(A.unix) ~= expect || any(diff(A.unix) ~= 10)
    error("satflight:poeorb", "%s %s..%s: %d state vectors, expected %d every 10 s", ...
        mission, want(1), want(end), numel(A.unix), expect);
end
end

function have = poeorbFiles(mission, root)
% the mission's downloaded files, the newest of each nominal day (poeorb.local_files)
files = dir(fullfile(root, "poeorb", mission, "*.EOF.zip"));
names = string({files.name});
have = struct("day", strings(0, 1), "path", strings(0, 1), "created", strings(0, 1));
for k = 1:numel(names)
    tok = regexp(names(k), "^(S1[A-D])_OPER_AUX_POEORB_OPOD_(\d{8}T\d{6})_V(\d{8}T\d{6})_" + ...
        "(\d{8}T\d{6})\.EOF\.zip$", "tokens", "once");
    if isempty(tok)
        error("satflight:poeorb", "%s is not a POEORB file name", names(k));
    end
    start = datetime(tok(3), InputFormat="yyyyMMdd'T'HHmmss", TimeZone="UTC");
    stop = datetime(tok(4), InputFormat="yyyyMMdd'T'HHmmss", TimeZone="UTC");
    if stop - start < hours(20) || stop - start > hours(30)
        error("satflight:poeorb", "%s: covers %s, not about 26 h", names(k), string(stop - start));
    end
    day = string(dateshift(start + (stop - start) / 2, "start", "day"), "yyyy-MM-dd");
    j = find(have.day == day, 1);
    if isempty(j)
        j = numel(have.day) + 1;
    elseif tok(2) <= have.created(j)
        continue
    end
    have.day(j, 1) = day;
    have.path(j, 1) = fullfile(files(k).folder, names(k));
    have.created(j, 1) = tok(2);
end
end

function f = poeorbRead(path, C)
% one POEORB file, checked as the simulator checks it (poeorb.read_file): its header, a
% state vector every 10 s over its validity window with the fields and units expected,
% known quality flags, TAI-UTC constant, UT1-UTC within 0.9 s, orbit numbers that do not go
% back, Sentinel-1's distance and speed, and velocities that are the positions' rate of
% change away from burns. Earth-fixed km and km/s; times as seconds since 1970 (UTC)
[~, stem] = fileparts(path);
name = stem + ".zip";
stem = extractBefore(stem, ".");
tmp = tempname;
mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, "s"));
inner = string(unzip(path, tmp));
if numel(inner) ~= 1
    error("satflight:poeorb", "%s: holds %d files, expected one .EOF", name, numel(inner));
end
txt = fileread(inner);
head = struct("File_Name", stem, "File_Type", "AUX_POEORB", ...
    "Mission", "Sentinel-1" + extractAfter(extractBefore(stem, 4), 2), ...
    "Ref_Frame", "EARTH_FIXED", "Time_Reference", "UTC");
tok = regexp(stem, "_V(\d{8}T\d{6})_(\d{8}T\d{6})$", "tokens", "once");
start = datetime(tok(1), InputFormat="yyyyMMdd'T'HHmmss", TimeZone="UTC");
stop = datetime(tok(2), InputFormat="yyyyMMdd'T'HHmmss", TimeZone="UTC");
head.Validity_Start = "UTC=" + string(start, "yyyy-MM-dd'T'HH:mm:ss");
head.Validity_Stop = "UTC=" + string(stop, "yyyy-MM-dd'T'HH:mm:ss");
if isempty(regexp(txt, "^(<\?xml[^>]*\?>)?\s*<Earth_Explorer_File>", "once"))
    error("satflight:poeorb", "%s: not an Earth Explorer file", name);
end
for field = string(fieldnames(head))'
    got = regexp(txt, "<" + field + ">([^<]*)</" + field + ">", "tokens", "once");
    if isempty(got) || strtrim(got) ~= head.(field)
        error("satflight:poeorb", "%s: %s is %s, expected %s", name, field, ...
            strjoin(string(got), ""), head.(field));
    end
end
count = regexp(txt, "<List_of_OSVs count=""(\d+)"">", "tokens", "once");
opened = regexp(txt, "<OSV>");
[osv, at] = regexp(txt, "<OSV>\s*<TAI>TAI=([^<]*)</TAI>\s*<UTC>UTC=([^<]*)</UTC>\s*" + ...
    "<UT1>UT1=([^<]*)</UT1>\s*<Absolute_Orbit>([^<]*)</Absolute_Orbit>\s*" + ...
    "<X unit=""m"">([^<]*)</X>\s*<Y unit=""m"">([^<]*)</Y>\s*<Z unit=""m"">([^<]*)</Z>\s*" + ...
    "<VX unit=""m/s"">([^<]*)</VX>\s*<VY unit=""m/s"">([^<]*)</VY>\s*" + ...
    "<VZ unit=""m/s"">([^<]*)</VZ>\s*<Quality>([^<]*)</Quality>\s*</OSV>", "tokens", "start");
if numel(at) ~= numel(opened)
    k = find(~ismember(opened, at), 1);
    error("satflight:poeorb", "%s, OSV %d: not the fields and units expected (TAI, UTC, " + ...
        "UT1, orbit, X Y Z in m, VX VY VZ in m/s, quality)", name, k - 1);
end
n = numel(opened);
if isempty(count) || str2double(count) ~= n
    error("satflight:poeorb", "%s: %d state vectors, but count says %s", name, n, ...
        strjoin(string(count), ""));
end
if n < 3
    error("satflight:poeorb", "%s: only %d state vectors", name, n);
end
F = string(vertcat(osv{:}));
quality = F(:, 11);
bad = find(~ismember(quality, C.QUALITIES), 1);
if ~isempty(bad)
    error("satflight:poeorb", "%s, OSV %d: unknown quality %s (known: %s)", name, bad - 1, ...
        quality(bad), strjoin(C.QUALITIES, ", "));
end
[utc, utcUs] = eeTimes(F(:, 2), name);
[tai, taiUs] = eeTimes(F(:, 1), name);
[ut1, ut1Us] = eeTimes(F(:, 3), name);
num = str2double(F(:, 5:10));
orbit = str2double(F(:, 4));
if any(isnan(num), "all") || any(isnan(orbit))
    error("satflight:poeorb", "%s: unreadable number", name);
end
stepOff = find(diff(utc) * 1e6 + diff(utcUs) ~= 10e6, 1);
if ~isempty(stepOff)
    error("satflight:poeorb", "%s, OSV %d: not 10 s after the one before", name, stepOff);
end
if utc(1) ~= posixtime(start) || utcUs(1) ~= 0 || utc(end) ~= posixtime(stop) || utcUs(end) ~= 0
    error("satflight:poeorb", "%s: the state vectors do not run over the validity window", name);
end
leap = (tai - utc) * 1e6 + (taiUs - utcUs);
if any(leap ~= leap(1))
    error("satflight:poeorb", "%s, OSV %d: TAI-UTC changes", name, find(leap ~= leap(1), 1) - 1);
end
dut1 = ((ut1 - utc) * 1e6 + (ut1Us - utcUs)) / 1e6;
if any(abs(dut1) >= 0.9)
    k = find(abs(dut1) >= 0.9, 1);
    error("satflight:poeorb", "%s, OSV %d: UT1-UTC = %g s, beyond 0.9 s", name, k - 1, dut1(k));
end
if any(diff(orbit) < 0)
    error("satflight:poeorb", "%s, OSV %d: the orbit number goes back", name, ...
        find(diff(orbit) < 0, 1));
end
r = num(:, 1:3) / 1000;
v = num(:, 4:6) / 1000;
for check = {{"|r|", r, C.POEORB_R, "km"}, {"|v|", v, C.POEORB_V, "km/s"}}
    [what, x, lim, unit] = check{1}{:};
    mag = vecnorm(x, 2, 2);
    k = find(~(mag >= lim(1) & mag <= lim(2)), 1);
    if ~isempty(k)
        error("satflight:poeorb", "%s, OSV %d: %s = %.3f %s, outside %g..%g", name, k - 1, ...
            what, mag(k), unit, lim(1), lim(2));
    end
end
% velocities against the positions' central differences, not across a burn
burn = quality == "DEGRADED-MANOEUVRE";
quiet = ~(burn(1:end - 2) | burn(2:end - 1) | burn(3:end));
err = vecnorm((r(3:end, :) - r(1:end - 2, :)) / 20 - v(2:end - 1, :), 2, 2);
err(~quiet) = 0;
[worst, k] = max(err);
if worst > C.POEORB_V_TOL
    error("satflight:poeorb", "%s, OSV %d: velocity disagrees with the positions by %.2f m/s", ...
        name, k, worst * 1e3);
end
f = struct("name", name, "unix", utc, "dut1", dut1, "r", r, "v", v, ...
    "degraded", quality ~= "NOMINAL");
end

function [sec, us] = eeTimes(text, name)
% "2025-06-30T22:59:42.000000" times: whole seconds since 1970 (UTC) and microseconds
v = sscanf(strjoin(text, " "), "%4d-%2d-%2dT%2d:%2d:%2d.%6d");
if numel(v) ~= 7 * numel(text)
    error("satflight:poeorb", "%s: unreadable time", name);
end
v = reshape(v, 7, []).';
days = datenum(v(:, 1), v(:, 2), v(:, 3)) - 719529;   %#ok<DATNM> % whole days since 1970
sec = days * 86400 + v(:, 4) * 3600 + v(:, 5) * 60 + v(:, 6);
us = v(:, 7);
end

function P = eopTable(path)
% the IERS finals2000A table (eop.read), strictly: every row up to the blank future must
% parse and the days follow one another. Pole x, y in rad; measured rows flagged
if ~isfile(path)
    error("satflight:noData", "no Earth-orientation table at %s: run python tools/fetch_eop.py", ...
        path);
end
lines = splitlines(string(fileread(path)));
if lines(end) == ""
    lines(end) = [];
end
blank = strlength(lines) <= 16 | extractBetween(pad(lines, 17), 17, 17) == " ";
firstBlank = find(blank, 1);
if ~isempty(firstBlank) && any(~blank(firstBlank:end))
    error("satflight:eop", "line %d: an empty row before line %d's values", firstBlank, ...
        firstBlank - 1 + find(~blank(firstBlank:end), 1));
end
rows = char(pad(lines(~blank), 68));
flag = rows(:, 17);
ut1Flag = rows(:, 58);
P.mjd = str2double(string(rows(:, 8:15)));
P.xp = str2double(string(rows(:, 19:27))) * (pi / (180.0 * 3600.0));
P.yp = str2double(string(rows(:, 38:46))) * (pi / (180.0 * 3600.0));
P.measured = flag == 'I' & ut1Flag == 'I';
bad = find(~ismember(flag, 'IP') | ~ismember(ut1Flag, 'IP') | isnan(P.mjd) | isnan(P.xp) ...
    | isnan(P.yp), 1);
if ~isempty(bad)
    error("satflight:eop", "line %d: not a finals2000A row", bad);
end
if numel(P.mjd) < 2
    error("satflight:eop", "%s: fewer than two days of values", path);
end
gap = find(diff(P.mjd) ~= 1, 1);
if ~isempty(gap)
    error("satflight:eop", "line %d: MJD %.0f does not follow %.0f", gap + 1, P.mjd(gap + 1), ...
        P.mjd(gap));
end
end

function [xp, yp] = eopPole(P, jd)
% polar motion (rad) at Julian dates jd (UTC), linear between the table's days
% (eop.Table.pole); only measured values, never the predictions
m = jd - 2400000.5;
i = floor(m - P.mjd(1)) + 1;      % the day at or before m: the days follow one another
out = i < 1 | (i >= numel(P.mjd) & m > P.mjd(end));
if any(out)
    error("satflight:eop", "MJD %.3f is outside the Earth-orientation table (MJD %.0f..%.0f)", ...
        m(find(out, 1)), P.mjd(1), P.mjd(end));
end
i = min(i, numel(P.mjd) - 1);
f = m - P.mjd(i);
ok = P.measured(i) & (P.measured(i + 1) | ~(f > 0));
if ~all(ok)
    error("satflight:eop", "MJD %.3f needs predicted Earth-orientation values; only " + ...
        "measured ones are used (fetch a newer table)", m(find(~ok, 1)));
end
xp = P.xp(i) + f .* (P.xp(i + 1) - P.xp(i));
yp = P.yp(i) + f .* (P.yp(i + 1) - P.yp(i));
end

function G = satflightTruthGap(T, sim)
%SATFLIGHTTRUTHGAP Each twin's offset from the replay of its mission (km), at every sample
%   while both fly: radial, along-track and cross-track in the replay's frame (its position,
%   the orbit normal, and the direction between), and the distance. The simulator shows the
%   same beside a selected twin (Simulation.gap_to_truth).
names = reshape(string(sim.names), [], 1);
mission = reshape(string(sim.mission), [], 1);
until = reshape(string(sim.replayUntil), [], 1);
G = table('Size', [0 8], 'VariableTypes', ["double", "datetime", "string", "string", ...
    "double", "double", "double", "double"], 'VariableNames', ["t_s", "utc", "name", ...
    "truth", "radial_km", "along_km", "cross_km", "distance_km"]);
G.utc.TimeZone = "UTC";
for k = reshape(find(mission ~= "" & until == ""), 1, [])
    truth = names(find(mission == mission(k) & until ~= "", 1));
    if isempty(truth)
        continue
    end
    a = T(T.name == names(k), :);
    b = T(T.name == truth, :);
    [~, ia, ib] = intersect(round(a.t_s * 1000), round(b.t_s * 1000));
    both = a.status(ia) == "active" & b.status(ib) == "active";
    ia = ia(both);
    ib = ib(both);
    r = [b.x_km(ib), b.y_km(ib), b.z_km(ib)];
    v = [b.vx_kms(ib), b.vy_kms(ib), b.vz_kms(ib)];
    d = [a.x_km(ia), a.y_km(ia), a.z_km(ia)] - r;
    rh = r ./ vecnorm(r, 2, 2);
    wh = cross(r, v, 2);
    wh = wh ./ vecnorm(wh, 2, 2);
    sh = cross(wh, rh, 2);
    gap = [sum(d .* rh, 2), sum(d .* sh, 2), sum(d .* wh, 2)];
    G = [G; table(a.t_s(ia), a.utc(ia), repmat(names(k), numel(ia), 1), ...
        repmat(truth, numel(ia), 1), gap(:, 1), gap(:, 2), gap(:, 3), vecnorm(gap, 2, 2), ...
        VariableNames=G.Properties.VariableNames)]; %#ok<AGROW>
end
end

function fig = satflightPlotTruthGap(G)
%SATFLIGHTPLOTTRUTHGAP Each twin's radial, along-track and cross-track gap to the replay of
%   its mission over the run (m).
fig = figure(Name="Gap to the precise orbit");
themeFigure(fig);
tl = tiledlayout(fig, 3, 1, TileSpacing="compact");
labels = ["radial_km", "Radial"; "along_km", "Along-track"; "cross_km", "Cross-track"];
twins = unique(G.name, "stable");
for j = 1:3
    ax = nexttile(tl);
    styleAxes(ax);
    hold(ax, "on");
    for k = 1:numel(twins)
        mine = G(G.name == twins(k), :);
        plot(ax, mine.t_s / 3600, mine.(labels(j, 1)) * 1000, Color=lineColor(k), ...
            DisplayName=twins(k));
    end
    grid(ax, "on");
    ylabel(ax, labels(j, 2) + " (m)");
    if j == 1
        title(ax, "Twin minus the precise orbit");
        legend(ax, Location="best");
    end
end
xlabel(ax, "Time since epoch (h)");
applyTheme(fig);
end

% --- events (simulation.py: eclipses, station passes, close approaches) ----------------------

function s = stepEvents(s, t, idx, Y)
% what the simulator looks for at the end of every step: eclipse entry and exit and
% station AOS/LOS (for each object while there are at most 40), close approaches
C = s.C;
w = idx(:);
if s.replays
    % a replayed object is the real one: it gets sunlight and stations, not approaches
    % (its twin flies metres from it on purpose)
    w = union(w, find(s.followed & s.status == "active"));
end
if numel(s.names) <= C.EVENT_DETAIL_LIMIT && ~isempty(w)
    w = sort(w);
    s = eclipseEvents(s, t, w, Y);
    if ~isempty(s.st.name)
        s = stationEvents(s, t, w, Y);
    end
end
s = approachEvents(s, t, idx, Y);
end

function s = eclipseEvents(s, t, w, Y)
% entering and leaving the shadow, seen at the end of the step and logged when half the
% Sun's disc showed (on the step's Hermite curve, the Sun where it is at the step's end)
C = s.C;
sun = sunPosition(s.sim.jd0 + t / 86400, C);
fr = shadowFraction(Y(w, 1:3), sun, C);
lit = @(pos, ~) shadowFraction(pos, sun, C) - 0.5;
for k = 1:numel(w)
    i = w(k);
    if s.shadow(i) >= 0.5 && 0.5 > fr(k)
        s = logEvent(s, crossingTime(s, t, i, Y, lit), sprintf("%s entered Earth's shadow", ...
            s.names(i)), "eclipse");
    elseif s.shadow(i) < 0.5 && 0.5 <= fr(k)
        s = logEvent(s, crossingTime(s, t, i, Y, lit), sprintf("%s back in sunlight", ...
            s.names(i)), "eclipse");
    end
    s.shadow(i) = fr(k);
end
end

function s = stationSetup(s, stations)
% ground stations: ECEF site, east-north-up axes, elevation mask
C = s.C;
nst = numel(stations);
s.st = struct("name", strings(nst, 1), "ecef", zeros(nst, 3), "enu", zeros(3, 3, nst), ...
    "minEl", zeros(nst, 1));
for k = 1:nst
    g = stations(k);
    lat = deg2rad(g.lat);
    lon = deg2rad(g.lon);
    s.st.name(k) = g.name;
    s.st.ecef(k, :) = geodeticToEcef(lat, lon, g.alt, C);
    s.st.enu(:, :, k) = [-sin(lon), cos(lon), 0; -sin(lat) * cos(lon), -sin(lat) * sin(lon), ...
        cos(lat); cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)];
    s.st.minEl(k) = g.minEl;
end
s.visible = false(nst, numel(s.names));
end

function s = stationEvents(s, t, w, Y)
% AOS and LOS, seen at the end of the step and logged when the object crossed the
% station's elevation mask (on the step's Hermite curve, the Earth turning under it)
for k = 1:numel(s.st.name)
    mask = deg2rad(s.st.minEl(k));
    sees = stationElevation(s, k, Y(w, 1:3), t) >= mask;
    above = @(pos, tt) stationElevation(s, k, pos, tt) - mask;
    for q = 1:numel(w)
        i = w(q);
        if sees(q) && ~s.visible(k, i)
            s = logEvent(s, crossingTime(s, t, i, Y, above), sprintf("AOS %s @ %s", ...
                s.names(i), s.st.name(k)), "station");
        elseif ~sees(q) && s.visible(k, i)
            s = logEvent(s, crossingTime(s, t, i, Y, above), sprintf("LOS %s @ %s", ...
                s.names(i), s.st.name(k)), "station");
        end
        s.visible(k, i) = sees(q);
    end
end
end

function el = stationElevation(s, k, r, t)
% elevation (rad) of ECI positions r (one per row) at time t above station k's horizon
th = gmst(s.sim.jd0 + t / 86400);
c = cos(th);
sn = sin(th);
ecef = [c * r(:, 1) + sn * r(:, 2), -sn * r(:, 1) + c * r(:, 2), r(:, 3)];
enu = (ecef - s.st.ecef(k, :)) * s.st.enu(:, :, k).';
rng = vecnorm(enu, 2, 2);
rng(rng == 0) = 1;
el = asin(min(max(enu(:, 3) ./ rng, -1), 1));
end

function rs = sunPosition(jd, C)
% geocentric Sun (km), the Astronomical Almanac series (satflight/ephemeris.py)
t = (jd - 2451545.0) / 36525.0;
lamM = 280.460 + 36000.771 * t;
m = (357.5291092 + 35999.05034 * t) * C.DEG;
lam = (lamM + 1.914666471 * sin(m) + 0.019994643 * sin(2 * m)) * C.DEG;
r = (1.000140612 - 0.016708617 * cos(m) - 0.000139589 * cos(2 * m)) * C.AU;
eps = (23.439291 - 0.0130042 * t) * C.DEG;
rs = [r * cos(lam), r * cos(eps) * sin(lam), r * sin(eps) * sin(lam)];
end

function a = fieldAcceleration(r, theta, G)
% the EGM2008 field's perturbation (km/s^2, degrees 2..N) at ECI positions r (one per
% row), the Earth turned by theta (one angle, or one per row): Cunningham's V/W
% recursion (Montenbruck & Gill 3.2.4-3.2.5), unnormalized, run degree by degree with
% every order and row at once, as the simulator's gravity.acceleration runs it
N = G.degree;
c = cos(theta);
s = sin(theta);
x = c .* r(:, 1) + s .* r(:, 2);
y = -s .* r(:, 1) + c .* r(:, 2);
z = r(:, 3);
r2 = x .* x + y .* y + z .* z;
rho = G.re * G.re ./ r2;
x0 = G.re * x ./ r2;
y0 = G.re * y ./ r2;
z0 = G.re * z ./ r2;
top = N + 2;                                  % V and W to degree N + 1
k = size(r, 1);
V = zeros(k, top, top);
W = zeros(k, top, top);
V(:, 1, 1) = G.re ./ sqrt(r2);
for n = 1:top - 1
    m = reshape(0:n - 1, 1, 1, []);           % the orders below the diagonal
    p = (2 * n - 1) ./ (n - m);
    q = (n + m - 1) ./ (n - m);
    V(:, n + 1, 1:n) = p .* z0 .* V(:, n, 1:n);
    W(:, n + 1, 1:n) = p .* z0 .* W(:, n, 1:n);
    if n >= 2
        V(:, n + 1, 1:n) = V(:, n + 1, 1:n) - q .* rho .* V(:, n - 1, 1:n);
        W(:, n + 1, 1:n) = W(:, n + 1, 1:n) - q .* rho .* W(:, n - 1, 1:n);
    end
    d = 2 * n - 1;                            % the diagonal from the one before
    V(:, n + 1, n + 1) = d * (x0 .* V(:, n, n) - y0 .* W(:, n, n));
    W(:, n + 1, n + 1) = d * (x0 .* W(:, n, n) + y0 .* V(:, n, n));
end
n0 = 2:N;                                     % the zonal terms (order 0)
[mm, nn] = meshgrid(1:N, 2:N);
keep = mm <= nn;
n1 = nn(keep)';                               % and the rest, degree n1, order m1
m1 = mm(keep)';
V2 = reshape(V, k, []);
W2 = reshape(W, k, []);
at = @(n, m) sub2ind([top top], n + 1, m + 1);   % the column of degree n, order m
C0 = G.c(n0 + 1, 1);
Cn = G.c(sub2ind(size(G.c), n1 + 1, m1 + 1));
Sn = G.s(sub2ind(size(G.s), n1 + 1, m1 + 1));
fac = 0.5 * (n1 - m1 + 1) .* (n1 - m1 + 2);
vp = V2(:, at(n1 + 1, m1 + 1));
wp = W2(:, at(n1 + 1, m1 + 1));
vm = V2(:, at(n1 + 1, m1 - 1));
wm = W2(:, at(n1 + 1, m1 - 1));
ax = -(V2(:, at(n0 + 1, 1)) * C0) + sum(0.5 * (-Cn .* vp - Sn .* wp) ...
    + fac .* (Cn .* vm + Sn .* wm), 2);
ay = -(W2(:, at(n0 + 1, 1)) * C0) + sum(0.5 * (-Cn .* wp + Sn .* vp) ...
    + fac .* (-Cn .* wm + Sn .* vm), 2);
az = sum(-(n0 + 1) .* C0' .* V2(:, at(n0 + 1, 0)), 2) ...
    + sum((n1 - m1 + 1) .* (-Cn .* V2(:, at(n1 + 1, m1)) - Sn .* W2(:, at(n1 + 1, m1))), 2);
kk = G.gm / (G.re * G.re);
gx = kk * ax;
gy = kk * ay;
gz = kk * az;
a = [c .* gx - s .* gy, s .* gx + c .* gy, gz];
end

function U = fieldPotential(r, theta, G)
% the EGM2008 field's perturbation potential (km^2/s^2, degrees 2..N; its gradient is
% fieldAcceleration) by fully normalized Legendre functions, as the simulator's
% gravity.potential sums it: a second route through the same coefficients
N = G.degree;
c = cos(theta);
s = sin(theta);
x = c .* r(:, 1) + s .* r(:, 2);
y = -s .* r(:, 1) + c .* r(:, 2);
z = r(:, 3);
rm = sqrt(x .* x + y .* y + z .* z);
lon = atan2(y, x);
t = z ./ rm;
u = sqrt(max(0, 1 - t .* t));
k = size(r, 1);
P = zeros(k, N + 1, N + 1);                   % P(:, n + 1, m + 1)
P(:, 1, 1) = 1;
P(:, 2, 2) = sqrt(3) * u;
for m = 2:N
    P(:, m + 1, m + 1) = sqrt((2 * m + 1) / (2 * m)) * u .* P(:, m, m);
end
for m = 0:N
    for n = m + 1:N
        a = sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)));
        P(:, n + 1, m + 1) = a * t .* P(:, n, m + 1);
        if n >= m + 2
            b = sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1) ...
                / ((n - m) * (n + m) * (2 * n - 3)));
            P(:, n + 1, m + 1) = P(:, n + 1, m + 1) - b * P(:, n - 1, m + 1);
        end
    end
end
% the fully normalized coefficients, back from the unnormalized ones the script holds
[mm, nn] = meshgrid(0:N, 0:N);
scale = sqrt((2 - (mm == 0)) .* (2 * nn + 1) ...
    .* exp(gammaln(max(nn - mm, 0) + 1) - gammaln(nn + mm + 1)));
scale(mm > nn) = 1;
cbar = reshape(G.c ./ scale, 1, N + 1, N + 1);
sbar = reshape(G.s ./ scale, 1, N + 1, N + 1);
mv = reshape(0:N, 1, 1, []);
inner = sum(P .* (cbar .* cos(lon .* mv) + sbar .* sin(lon .* mv)), 3);
U = G.gm ./ rm .* sum((G.re ./ rm) .^ (0:N) .* inner, 2);
end

function rm = moonPosition(jd, C)
% geocentric Moon (km), Vallado Alg. 31 (satflight/ephemeris.py), one date
t = (jd - 2451545.0) / 36525.0;
lam = (218.32 + 481267.8813 * t ...
    + 6.29 * sind(134.9 + 477198.85 * t) ...
    - 1.27 * sind(259.2 - 413335.38 * t) ...
    + 0.66 * sind(235.7 + 890534.23 * t) ...
    + 0.21 * sind(269.9 + 954397.70 * t) ...
    - 0.19 * sind(357.5 + 35999.05 * t) ...
    - 0.11 * sind(186.6 + 966404.05 * t)) * C.DEG;
beta = (5.13 * sind(93.3 + 483202.03 * t) ...
    + 0.28 * sind(228.2 + 960400.87 * t) ...
    - 0.28 * sind(318.3 + 6003.18 * t) ...
    - 0.17 * sind(217.6 - 407332.20 * t)) * C.DEG;
parallax = (0.9508 ...
    + 0.0518 * cosd(134.9 + 477198.85 * t) ...
    + 0.0095 * cosd(259.2 - 413335.38 * t) ...
    + 0.0078 * cosd(235.7 + 890534.23 * t) ...
    + 0.0028 * cosd(269.9 + 954397.70 * t)) * C.DEG;
d = C.RE / sin(parallax);
eps = (23.439291 - 0.0130042 * t) * C.DEG;
rm = [d * cos(beta) * cos(lam), ...
    d * (cos(eps) * cos(beta) * sin(lam) - sin(eps) * sin(beta)), ...
    d * (sin(eps) * cos(beta) * sin(lam) + cos(eps) * sin(beta))];
end

function nu = shadowFraction(r, rSun, C)
% fraction of the Sun's disc seen from each row of r (conical shadow, satflight/eclipse.py)
d = rSun - r;
rm = vecnorm(r, 2, 2);
dm = vecnorm(d, 2, 2);
a = asin(min(max(C.R_SUN ./ dm, -1), 1));
b = asin(min(max(C.RE ./ max(rm, C.RE), -1), 1));
c = acos(min(max(sum(-r .* d, 2) ./ (rm .* dm), -1), 1));
nu = ones(size(rm));
umbra = c < b - a;
annular = (c < a - b) & ~umbra;
partial = (c < a + b) & ~umbra & ~annular;
nu(umbra) = 0;
nu(annular) = 1 - (b(annular) ./ a(annular)) .^ 2;
if any(partial)
    ap = a(partial);
    bp = b(partial);
    cp = c(partial);
    x = (cp .* cp + ap .* ap - bp .* bp) ./ (2 * cp);
    y = sqrt(max(ap .* ap - x .* x, 0));
    area = ap .* ap .* acos(min(max(x ./ ap, -1), 1)) + bp .* bp .* acos(min(max((cp - x) ./ ...
        bp, -1), 1)) - cp .* y;
    nu(partial) = 1 - area ./ (pi * ap .* ap);
end
end

function s = approachEvents(s, t, idx, Y)
% pairs that came within the alert distance since the last step, found along each
% pair's relative motion over the step, and logged when they first do
C = s.C;
idx = idx(:);
orbiting = idx(geodeticHeight(Y(idx, 1:3), C) > 100);
mask = false(numel(s.names), 1);
mask(orbiting) = true;
if numel(orbiting) < 2
    s.lastSample = struct("t", t, "Y", Y, "mask", mask);   % event times still use it
    return
end
r = Y(orbiting, 1:3);
sweep = approachSweep(s, t, orbiting, Y);
[p, q] = candidatePairs(r, sweep, C);
d = vecnorm(r(p, :) - r(q, :), 2, 2);
fam = s.family(orbiting);
if any(fam ~= "")
    % a payload and the stages that carried it drift apart slowly
    none = fam == "";
    fam(none) = s.names(orbiting(none));
    d(fam(p) == fam(q)) = inf;
end
[a, b, dist, tca] = approachPairs(t, orbiting, p, q, d, sweep, C);
now = [a, b];
if ~isempty(now)
    if isempty(s.closePairs)
        isNew = true(size(a));
    else
        isNew = ~ismember(now, s.closePairs, "rows");
    end
    fresh = find(isNew);
    [~, order] = sortrows([tca(fresh), a(fresh), b(fresh)]);
    for k = reshape(fresh(order), 1, [])
        s = logEvent(s, tca(k), sprintf("Close approach: %s - %s %.2f km", s.names(a(k)), ...
            s.names(b(k)), dist(k)), "alert");
    end
end
s.closePairs = now;
s.lastSample = struct("t", t, "Y", Y, "mask", mask);
end

function sw = approachSweep(s, t, orbiting, Y)
% how far each object can have been, during the step that just ended, from where
% it is now (reach) and from the middle of its chord (mid, spread)
prev = s.lastSample;
sw = [];
if isempty(prev) || ~(prev.t < t) || ~isequal(size(prev.Y), size(Y))
    return
end
h = t - prev.t;
y0 = prev.Y(orbiting, :);
y1 = Y(orbiting, :);
moved = y1(:, 1:3) - y0(:, 1:3);
stepLen = vecnorm(moved, 2, 2);
bow = 4 / 27 * (vecnorm(h * y0(:, 4:6) - moved, 2, 2) + vecnorm(h * y1(:, 4:6) - moved, 2, 2));
reach = stepLen + bow;
mid = y1(:, 1:3) - 0.5 * moved;
spread = 0.5 * stepLen + bow;
fresh = ~prev.mask(orbiting);
reach(fresh) = -inf;
mid(fresh, :) = y1(fresh, 1:3);
spread(fresh) = 0;
sw = struct("t0", prev.t, "h", h, "y0", y0, "y1", y1, "reach", reach, "mid", mid, ...
    "spread", spread);
end

function [p, q] = candidatePairs(r, sw, C)
% every pair in small ensembles; pairs found through a spatial grid in large ones
n = size(r, 1);
if n <= C.ALL_PAIRS_LIMIT
    [q, p] = find(tril(true(n), -1));
    return
end
mid = r;
spread = zeros(n, 1);
if ~isempty(sw)
    mid = sw.mid;
    spread = sw.spread;
end
radius = C.CONJUNCTION_KM + 2 * max([spread; 0]);
[p, q] = nearPairs(mid, max(radius, 1));
end

function [p, q] = nearPairs(r, radius)
% index pairs (p < q) of rows of r closer than radius, binned in cubes radius wide
n = size(r, 1);
cellIdx = int64(min(max(floor(r / radius), -2 ^ 20), 2 ^ 20 - 2)) + 2 ^ 20;
key = cellIdx(:, 1) * 2 ^ 42 + cellIdx(:, 2) * 2 ^ 21 + cellIdx(:, 3);
[skey, order] = sort(key);
[ukey, first] = unique(skey, "first");
counts = diff([first; n + 1]);
P = cell(0, 1);
Q = cell(0, 1);
for dx = -1:1
    for dy = -1:1
        for dz = -1:1
            if ~(dx > 0 || (dx == 0 && (dy > 0 || (dy == 0 && dz >= 0))))
                continue
            end
            off = int64(dx) * 2 ^ 42 + int64(dy) * 2 ^ 21 + int64(dz);
            [tf, u] = ismember(key + off, ukey);
            a = find(tf);
            if isempty(a)
                continue
            end
            m = counts(u(tf));
            start = first(u(tf));
            A = repelem(a, m);
            within = (1:sum(m))' - repelem(cumsum(m) - m, m) - 1;
            B = order(repelem(start, m) + within);
            if off == 0
                keep = A < B;
                A = A(keep);
                B = B(keep);
            end
            P{end + 1, 1} = min(A, B); %#ok<AGROW>
            Q{end + 1, 1} = max(A, B); %#ok<AGROW>
        end
    end
end
p = vertcat(P{:}, zeros(0, 1));
q = vertcat(Q{:}, zeros(0, 1));
near = vecnorm(r(p, :) - r(q, :), 2, 2) < radius;
pq = sortrows([p(near), q(near)]);
p = pq(:, 1);
q = pq(:, 2);
end

function [a, b, dist, tca] = approachPairs(t, orbiting, p, q, d, sw, C)
% pairs within the alert distance at any time during the step, with the closest
% distance and its time (cubic Hermite relative motion between the step's ends)
lim = C.CONJUNCTION_KM;
dist = d;
tca = t * ones(size(d));
if ~isempty(sw)
    sel = find(d - sw.reach(p) - sw.reach(q) < lim);
    pp = p(sel);
    qq = q(sel);
    rel0 = sw.y0(pp, :) - sw.y0(qq, :);
    rel1 = sw.y1(pp, :) - sw.y1(qq, :);
    d0 = rel0(:, 1:3);
    m0 = sw.h * rel0(:, 4:6);
    c = rel1(:, 1:3) - d0;
    m1 = sw.h * rel1(:, 4:6);
    cc = sum(c .* c, 2);
    cc(cc <= 0) = 1;
    u = min(max(-sum(d0 .* c, 2) ./ cc, 0), 1);
    chord = vecnorm(d0 + u .* c, 2, 2);
    bow = 4 / 27 * (vecnorm(m0 - c, 2, 2) + vecnorm(m1 - c, 2, 2));
    near = chord - bow < lim;
    if any(near)
        [tau, dm] = hermiteMin(d0(near, :), c(near, :), m0(near, :), m1(near, :));
        k = sel(near);
        better = dm < dist(k);
        dist(k(better)) = dm(better);
        tca(k(better)) = sw.t0 + tau(better) * sw.h;
    end
end
k = find(dist < lim);
a = orbiting(p(k));
b = orbiting(q(k));
dist = dist(k);
tca = tca(k);
end

function [t, dist] = hermiteMin(d0, c, m0, m1)
% minimum distance from the origin of p(t) = d0 + c t + t (1 - t)^2 a - t^2 (1 - t) b
% (a = m0 - c, b = m1 - c), t in [0, 1]: a scan, then Newton's method kept between
% the best sample's neighbours
a = m0 - c;
b = m1 - c;
grid = linspace(0, 1, 17);
P2 = zeros(size(d0, 1), numel(grid));
for g = 1:numel(grid)
    P2(:, g) = sum(hermiteCurve(grid(g) * ones(size(d0, 1), 1), d0, c, a, b) .^ 2, 2);
end
[~, best] = min(P2, [], 2);
lo = grid(max(best - 1, 1))';
hi = grid(min(best + 1, numel(grid)))';
t = grid(best)';
for iter = 1:8
    [p, dp, ddp] = hermiteCurve(t, d0, c, a, b);
    g = sum(p .* dp, 2);
    gg = sum(dp .* dp, 2) + sum(p .* ddp, 2);
    gg(gg <= 0) = inf;
    t = min(max(t - g ./ gg, lo), hi);
end
dist = vecnorm(hermiteCurve(t, d0, c, a, b), 2, 2);
end

function [p, dp, ddp] = hermiteCurve(t, d0, c, a, b)
p = d0 + c .* t + (t - 2 * t .* t + t .^ 3) .* a - (t .* t - t .^ 3) .* b;
dp = c + (1 - 4 * t + 3 * t .* t) .* a - (2 * t - 3 * t .* t) .* b;
ddp = (6 * t - 4) .* a - (2 - 6 * t) .* b;
end

% --- frames ------------------------------------------------------------------------------------

function r = geodeticToEcef(lat, lon, alt, C)
% WGS-84 geodetic latitude, longitude (rad) and height (km) -> ECEF km
sl = sin(lat);
n = C.RE / sqrt(1 - C.E2 * sl * sl);
r = [(n + alt) * cos(lat) * cos(lon), (n + alt) * cos(lat) * sin(lon), (n * (1 - C.E2) + alt) * sl];
end

% --- launches: the run's side (satflight/simulation.py) ---------------------------------------

function s = launchVehicle(s, L, replan)
% put a vehicle on its pad, as the simulator's launch() does: with the plan the
% simulator made for it (liftoff time, plane, pitch kick), or, with replan, a plan made
% here; a vehicle exported in flight carries on from its saved state
C = s.C;
if strlength(L.name) == 0
    L.name = "Payload";
end
L.name = uniqueName(s, L.name);
try
    launchCheck(L, C);
    if replan && isempty(L.state)
        planeOf = @(name) planeOfSat(s, name);
        [res, kick, flight] = planLaunch(L, s.env, s.t, planeOf, C);
    else
        res = L.plan;
        kick = L.plan.kick;
        flight = ascentFly(L, res, s.env, kick, true, C);
    end
catch err
    if err.identifier ~= "satflight:launch"
        rethrow(err)
    end
    s = logEvent(s, s.t, sprintf("Launch of %s scrubbed: %s", L.name, err.message), "alert");
    return
end
A = ascentNew(L, res, kick, s.env, true, C);
if ~isempty(L.state)
    A = ascentRestore(A, L.state);
end
if A.phase ~= "pad"
    y = ascentPeek(A, s.t, C);
elseif A.t0 > s.t
    y = padState(L, ascGmst(s.env, s.t), C);
else
    y = A.y;
end
[s, row] = addRow(s, L.name, y, A.mass, A.area, A.veh.cd, L.name);
A.sat = row;
A.planned = flight;
outcome = flight.outcome;
if flight.outcome == "orbit" && L.guidance == "orbit"
    outcome = sprintf("orbit %s x %s km", commaNum(flight.insertion.hp), ...
        commaNum(flight.insertion.ha));
end
kind = "warn";
if flight.outcome == "orbit" || L.guidance == "open"
    kind = "launch";
end
site = L.site;
if strlength(site) == 0
    site = "the launch site";
end
s = logEvent(s, s.t, sprintf("%s: %s on the pad at %s (%.2f, %.2f), liftoff %s; planned %s", ...
    L.name, A.veh.name, site, L.lat, L.lon, whenText(A.t0, s.t), outcome), kind);
s.ascents{end + 1} = A;
s = advanceAscents(s, s.t);
end

function s = advanceAscents(s, t)
% fly every vehicle on a pad or in flight to t; spent stages join the run
if isempty(s.ascents)
    return
end
C = s.C;
spawned = struct("name", {}, "y", {}, "mass", {}, "area", {}, "cd", {}, "family", {});
keep = true(1, numel(s.ascents));
for q = 1:numel(s.ascents)
    A = s.ascents{q};
    if A.resumed && t < A.t + C.ASCENT_SYNC - 1e-9
        continue        % exported in flight: it steps again at its own next sync
    end
    A.resumed = false;
    A = ascentAdvanceTo(A, t, C);
    i = A.sat;
    s.Y(i, :) = A.y;
    s.mass(i) = A.mass;
    for k = 1:numel(A.evT)
        s = logEvent(s, A.evT(k), A.evText(k), A.evKind(k));
    end
    A.evT = zeros(0, 1);
    A.evText = strings(0, 1);
    A.evKind = strings(0, 1);
    for sp = A.spawns
        y = ascCoast(sp.y, sp.t, t, sp.mass, sp.cdA, A.env, C);
        spawned(end + 1) = struct("name", sp.name, "y", y, "mass", sp.mass, "area", A.area, ...
            "cd", A.veh.cd, "family", s.family(i)); %#ok<AGROW>
    end
    A.spawns = A.spawns([]);
    s.ascents{q} = A;
    if A.released
        s = finishAscent(s, A);
        keep(q) = false;
    end
end
s.ascents = s.ascents(keep);
for sp = spawned
    s = addRow(s, sp.name, sp.y, sp.mass, sp.area, sp.cd, sp.family);
end
end

function s = finishAscent(s, A)
% hand a separated payload over to the run
i = A.sat;
L = A.spec;
s.mass(i) = A.mass;
s.area(i) = L.payloadArea;
s.cd(i) = L.payloadCd;
s.cr(i) = L.payloadCr;
s.reports{end + 1} = launchSummary(s, A);
if A.outcome == "crash"
    s.status(i) = "impacted";
elseif A.outcome == "no_liftoff"
    s.status(i) = "scrubbed";
end
if s.status(i) ~= "active"
    if ~isempty(s.man)
        s.pending(s.pending & reshape(string({s.man.sat}), 1, []) == s.names(i)) = false;
    end
elseif L.circularize && A.outcome == "orbit"
    m = satflightBurn(s.names(i), "circularize", NaN, [0 0 0], "VNB", 0, "", 0, 300, 0, ...
        "circularize at apogee");
    try
        s = scheduleAtApsis(s, i, m);
    catch err
        if err.identifier ~= "satflight:maneuver"
            rethrow(err)
        end
        s = logEvent(s, s.t, sprintf("%s: circularization skipped (%s)", s.names(i), ...
            err.message), "warn");
    end
end
end

function [inc, normalAt] = planeOfSat(s, name)
% inclination (deg) and orbit normal over time of a satellite, with its J2 nodal
% regression: the plane a "plane" launch waits for
i = find(s.names == name, 1);
if isempty(i)
    error("satflight:launch", "no satellite named '%s'", name);
end
el = rv2coe(s.Y(i, 1:3), s.Y(i, 4:6), s.C.MU);
if el.e >= 1
    error("satflight:launch", "%s is not on a closed orbit", name);
end
F = s.sim.forces;
rate = 0;
if s.sim.propagator == "j2mean" || (s.sim.propagator == "cowell" && F.j2)
    % the mean orbit sets the drift
    a = osculatingToMeanA(el.a, el.e, el.i, el.argp, el.nu, s.C);
    iMean = osculatingToMeanI(el.i, a, el.e, el.argp, el.nu, s.C);
    rate = j2SecularRates(a, el.e, iMean, s.C);
end
tRef = s.t;
inc = rad2deg(el.i);
normalAt = @(t) planeNormal(el.i, el.raan + rate * (t - tRef));
end

function R = launchSummary(s, A)
% outcome, insertion orbit, delta-v budget and ascent profile of one launch
L = A.spec;
cols = ["met_s", "alt_km", "downrange_km", "speed_kms", "q_kPa", "accel_g"];
profile = array2table(A.samples, VariableNames=cols);
profile.lat_deg = A.track(:, 1);
profile.lon_deg = A.track(:, 2);
marks = array2table(A.marks, VariableNames=["met_s", "alt_km", "downrange_km"]);
marks.label = A.markLabels;
P = A.planned;
planned = array2table(P.samples, VariableNames=cols);
plannedMarks = array2table(P.marks, VariableNames=["met_s", "alt_km", "downrange_km"]);
plannedMarks.label = P.markLabels;
R = struct("name", s.names(A.sat), "vehicle", A.veh.name, "site", L.site, "lat", L.lat, ...
    "lon", L.lon, "liftoff_s", A.t0, "kick_deg", A.kickDeg, "outcome", A.outcome, ...
    "plannedOutcome", P.outcome, "insertion", A.insertion, "dvIdeal_kms", A.dvIdeal, ...
    "losses_kms", struct("gravity", A.losses(1), "drag", A.losses(2), "steering", A.losses(3)), ...
    "maxQ_kPa", A.maxQ(1) / 1000, "tMaxQ_s", A.maxQ(2), "profile", profile, "marks", marks, ...
    "planned", planned, "plannedMarks", plannedMarks);
end

% --- launches: the vehicle (satflight/launch.py) ----------------------------------------------

function A = ascentNew(L, res, kickDeg, env, record, C)
% one vehicle from the pad to payload separation (launch.Ascent)
V = L.vehicle;
A = struct();
A.spec = L;
A.veh = V;
A.res = res;
A.env = env;
A.kickDeg = kickDeg;
A.kick = deg2rad(kickDeg);
A.record = record;
A.t0 = res.t0;
A.t = res.t0;
A.y = padState(L, ascGmst(env, res.t0), C);
A.phase = "pad";
A.k = 1;                                   % current stage (1 = first)
A.prop = V.stages(1, 4);
A.mass = L.payloadMass + V.fairing + sum(V.stages(:, 4) + V.stages(:, 5));
A.fairingOn = V.fairing > 0;
A.burning = false;
A.coastUntil = -inf;
A.throttle = 1;
A.gWarned = zeros(1, 0);                   % stages whose g-limit overshoot is logged
A.thrustDir = unit3(A.y(1:3));
A.ctrl = [];                               % mdot, stage, mode, fixed of the last step
A.hold = [];
A.tgo = NaN;
A.kickStart = NaN;
A.sat = 0;
A.released = false;
A.outcome = "";
A.evT = zeros(0, 1);
A.evText = strings(0, 1);
A.evKind = strings(0, 1);
A.spawns = struct("t", {}, "name", {}, "y", {}, "mass", {}, "cdA", {});
A.q = 0;
A.maxQ = [0, NaN];
A.qLogged = false;
A.accel = 0;
A.dvIdeal = 0;
A.losses = [0 0 0];                        % gravity, drag, steering (km/s)
A.vStart = norm(A.y(4:6));
A.samples = zeros(0, 6);                   % MET, alt, downrange, speed, q kPa, accel g
A.track = zeros(0, 2);                     % geocentric lat, lon (deg)
A.marks = zeros(0, 3);                     % MET, alt, downrange
A.markLabels = strings(0, 1);
A.nextSample = -inf;
A.insertion = [];
A.planned = [];
A.resumed = false;
A.area = pi * V.diameter ^ 2 / 4;
A.cdA = V.cd * A.area;
r = geodeticToEcef(deg2rad(L.lat), deg2rad(L.lon), L.alt, C);
A.padHat = r / norm(r);
end

function A = ascentRestore(A, S)
% the flight state the simulator saved when the run was exported mid-ascent
A.t = S.t;
A.y = S.y;
A.phase = S.phase;
A.k = S.stage;
A.prop = S.prop;
A.mass = S.mass;
A.fairingOn = S.fairingOn;
A.burning = S.burning;
A.coastUntil = S.coastUntil;
A.throttle = S.throttle;
A.thrustDir = S.thrustDir;
A.ctrl = S.ctrl;
A.hold = S.hold;
A.tgo = S.tgo;
A.kickStart = S.kickStart;
A.q = S.q;
A.maxQ = [S.maxQ, S.tMaxQ];
A.qLogged = S.qLogged;
A.accel = S.accel;
A.dvIdeal = S.dvIdeal;
A.losses = S.losses;
A.vStart = S.vStart;
A.nextSample = S.nextSample;
A.resumed = true;
end

function A = ascentFly(L, res, env, kickDeg, record, C)
% a whole ascent to payload separation, on paper (launch.fly)
A = ascentNew(L, res, kickDeg, env, record, C);
A = ascentAdvanceTo(A, inf, C);
end

function y = padState(L, theta, C)
% ECI state of the pad (it turns with the Earth)
rE = geodeticToEcef(deg2rad(L.lat), deg2rad(L.lon), L.alt, C);
c = cos(-theta);
sn = sin(-theta);
r = [c * rE(1) + sn * rE(2), -sn * rE(1) + c * rE(2), rE(3)];
y = [r, -C.OMEGA * r(2), C.OMEGA * r(1), 0];
end

function th = ascGmst(env, t)
th = gmst(env.jd0 + t / 86400);
end

function A = ascentAdvanceTo(A, tEnd, C)
% fly (or wait on the pad) up to time tEnd
if A.phase == "pad"
    if tEnd < A.t0 - 1e-9
        A.y = padState(A.spec, ascGmst(A.env, tEnd), C);
        A.t = tEnd;
        return
    end
    A.t = A.t0;
    A.y = padState(A.spec, ascGmst(A.env, A.t0), C);
    A = ascLiftoff(A, C);
end
while ~A.released && A.t < tEnd - 1e-9
    t0 = A.t;
    A = ascStep(A, min(C.H_POWERED, tEnd - A.t), C);
    if ~A.released && A.t <= t0
        error("satflight:internal", "The ascent of %s made no progress at MET %.1f s.", ...
            A.spec.name, A.t - A.t0);
    end
end
if A.released && isfinite(tEnd) && A.t < tEnd - 1e-9
    A.y = ascCoast(A.y, A.t, tEnd, A.mass, A.spec.payloadCd * A.spec.payloadArea, A.env, C);
    A.t = tEnd;
end
end

function y = ascentPeek(A, t, C)
% the state at t, a fraction of a step ahead, as the simulator shows it between two
% steps of the ascent: the last step's thrust and steering, nothing changed
if A.phase == "pad"
    y = padState(A.spec, ascGmst(A.env, t), C);
    return
end
h = t - A.t;
if h <= 0 || A.released || isempty(A.ctrl)
    y = A.y;
    return
end
mdot = 0;
if A.burning
    mdot = A.ctrl.mdot;
end
y = ascRk4(A.y, h, A.mass, mdot, A.veh.stages(A.ctrl.stage, :), A.ctrl.mode, A.ctrl.fixed, ...
    A.cdA, A.env, C);
end

function A = ascEvent(A, text, kind)
A.evT(end + 1, 1) = A.t;
A.evText(end + 1, 1) = A.spec.name + ": " + text;
A.evKind(end + 1, 1) = kind;
end

function A = ascSample(A, force, C)
% a telemetry sample every SAMPLE_DT s (always when force)
if ~A.record || (~force && A.t < A.nextSample)
    return
end
A.nextSample = A.t + C.SAMPLE_DT;
[lat, lon, dr] = ascGeo(A, A.t, A.y, C);
A.samples(end + 1, :) = [A.t - A.t0, ascAltitude(A.y, C), dr, norm(A.y(4:6)), A.q / 1000, ...
    A.accel / C.G0M];
A.track(end + 1, :) = [lat, lon];
end

function A = ascMark(A, label, C)
% a labelled milestone (liftoff, staging ...) on the profile
if A.record
    [~, ~, dr] = ascGeo(A, A.t, A.y, C);
    A.marks(end + 1, :) = [A.t - A.t0, ascAltitude(A.y, C), dr];
    A.markLabels(end + 1, 1) = label;
end
end

function [lat, lon, dr] = ascGeo(A, t, y, C)
% geocentric latitude, longitude (deg) and downrange distance (km) at time t
th = ascGmst(A.env, t);
c = cos(th);
sn = sin(th);
u = unit3([c * y(1) + sn * y(2), -sn * y(1) + c * y(2), y(3)]);
ang = acos(max(-1, min(1, dot3(u, A.padHat))));
lat = rad2deg(asin(u(3)));
lon = rad2deg(atan2(u(2), u(1)));
dr = ang * C.RE;
end

function A = ascLiftoff(A, C)
% ignite the first stage, or scrub if it cannot lift the vehicle
st = A.veh.stages(1, :);
thrust = stageMdot(st, C) * C.G0M * stageIsp(st, ascPressure(A.spec.alt, C));
% weight on the turning Earth: gravity less the centrifugal relief, along the plumb line
g = ascGravity(A.y(1), A.y(2), A.y(3), A.env, C);
w2 = C.OMEGA * C.OMEGA;
weight = A.mass * sqrt((g(1) + w2 * A.y(1)) ^ 2 + (g(2) + w2 * A.y(2)) ^ 2 + g(3) ^ 2) * 1e3;
A.phase = "vertical";
if thrust <= weight
    A = ascRelease(A, "no_liftoff", sprintf("scrubbed - thrust/weight %.2f is below 1, the " + ...
        "vehicle cannot lift off", thrust / weight), "alert");
    return
end
A.burning = true;
site = A.spec.site;
if strlength(site) == 0
    site = "the pad";
end
A = ascEvent(A, sprintf("liftoff on %s from %s (T/W %.2f)", A.veh.name, site, ...
    thrust / weight), "launch");
A = ascMark(A, "Liftoff", C);
A = ascSample(A, true, C);
end

function [A, th] = ascThrottle(A, st, alt, C)
% throttle (0.4-1) that keeps the vehicle within its acceleration limit, as far as the
% engines go down: past that the limit is exceeded, and the log says so once per stage
if A.veh.maxG <= 0
    th = 1;
    return
end
a = stageMdot(st, C) * C.G0M * stageIsp(st, ascPressure(alt, C)) / A.mass;
want = A.veh.maxG * C.G0M / a;
if want < C.THROTTLE_MIN && ~any(A.gWarned == A.k)
    A.gWarned(end + 1) = A.k;
    A = ascEvent(A, sprintf("%s exceeds the %g g limit even at %.0f %% throttle (%.2f g)", ...
        A.veh.stageNames(min(A.k, size(A.veh.stages, 1))), A.veh.maxG, C.THROTTLE_MIN * 100, ...
        C.THROTTLE_MIN * a / C.G0M), "warn");
end
th = max(C.THROTTLE_MIN, min(1, want));
end

function p = ascPressure(h, C)
% ambient pressure over its sea-level value at geometric height h km: the US Standard
% Atmosphere 1976 (below 0 km as at 0 km; above 86 km isothermal decay)
hg = C.US76_R0 * max(h, 0) / (C.US76_R0 + max(h, 0));
if hg >= C.US76_TOP
    p = C.US76_P_TOP * exp(-C.US76_GMR * (hg - C.US76_TOP) / C.US76_T_TOP);
    return
end
k = find(C.US76(:, 1) <= hg, 1, "last");
p = C.US76_P(k) * us76Layer(C.US76(k, :), hg - C.US76(k, 1), C.US76_GMR);
end

function r = us76Layer(layer, dh, gmr)
% pressure ratio across dh km above the base of a US76 layer [base km, base K, lapse K/km]
if layer(3) == 0
    r = exp(-gmr * dh / layer(2));
else
    r = (layer(2) / (layer(2) + layer(3) * dh)) ^ (gmr / layer(3));
end
end

function A = ascStep(A, h, C)
% one step of at most h s, shortened to land exactly on burnout or on the cutoff energy
s0 = A.y;
t = A.t;
m0 = A.mass;
alt = ascAltitude(s0, C);
nst = size(A.veh.stages, 1);
if ~A.burning && A.k <= nst && A.coastUntil <= t + 1e-9
    A.burning = true;
    A = ascEvent(A, sprintf("%s ignition at %.0f km", A.veh.stageNames(A.k), alt), "launch");
    A = ascMark(A, A.veh.stageNames(A.k) + " ign.", C);
end
kk = min(A.k, nst);
st = A.veh.stages(kk, :);
mode = 0;
fixed = [0 0 0];
mdot = 0;
burnout = false;
cutoff = false;
if A.burning
    [A, mode, fixed] = ascSteer(A, s0, alt, C);
    [A, A.throttle] = ascThrottle(A, st, alt, C);
    mdot = stageMdot(st, C) * A.throttle;
    if mdot * h >= A.prop
        h = A.prop / mdot;
        burnout = true;
    end
elseif A.coastUntil > t
    h = min(h, A.coastUntil - t);
end
A.ctrl = struct("mdot", mdot, "stage", kk, "mode", mode, "fixed", fixed);
s1 = ascRk4(s0, h, m0, mdot, st, mode, fixed, A.cdA, A.env, C);
if A.burning && A.phase == "guided"
    e1 = ascEnergy(s1, C);
    if e1 >= A.res.eTarget
        e0 = ascEnergy(s0, C);
        f = 1;
        if e1 > e0
            f = min(1, max(0, (A.res.eTarget - e0) / (e1 - e0)));
        end
        h = h * f;
        s1 = ascRk4(s0, h, m0, mdot, st, mode, fixed, A.cdA, A.env, C);
        cutoff = true;
        burnout = false;
    end
end
A = ascAccount(A, s0, m0, mdot, st, mode, fixed, alt, h, C);
A.t = t + h;
A.y = s1;
A.mass = m0 - mdot * h;
A.prop = max(0, A.prop - mdot * h);
A = ascAfterStep(A, cutoff, burnout, C);
end

function A = ascAccount(A, s, m, mdot, st, mode, fixed, alt, h, C)
% telemetry and the delta-v budget, from the state at the start of a step
v = sqrt(s(4) ^ 2 + s(5) ^ 2 + s(6) ^ 2);
vh = [0 0 0];
if v > 0
    vh = s(4:6) / v;
end
w = [s(4) + C.OMEGA * s(2), s(5) - C.OMEGA * s(1), s(6)];
sp = sqrt(dot3(w, w));
rho = 0;
if alt < C.TOP_KM
    rho = ascDensity(alt, C) * A.env.densityScale;
end
A.q = 0.5 * rho * (sp * 1e3) ^ 2;
g = ascGravity(s(1), s(2), s(3), A.env, C);
A.losses(1) = A.losses(1) - dot3(g, vh) * h;
if m > 0 && sp > 0
    drag = 0.5e3 * rho * A.cdA / m * sp;
    A.losses(2) = A.losses(2) + drag * dot3(w, vh) * h;
end
if mdot > 0
    a = mdot * C.G0M * stageIsp(st, ascPressure(alt, C)) / m;
    A.accel = a;
    if mode == 1 && sp > 1e-6
        d = unit3(w);
    else
        d = fixed;
    end
    A.thrustDir = d;
    A.dvIdeal = A.dvIdeal + a * 1e-3 * h;
    A.losses(3) = A.losses(3) + a * 1e-3 * (1 - dot3(d, vh)) * h;
else
    A.accel = 0;
end
if A.q > A.maxQ(1)
    A.maxQ = [A.q, A.t - A.t0];
elseif ~A.qLogged && A.maxQ(1) > 1000 && A.q < 0.9 * A.maxQ(1)
    A.qLogged = true;
    A = ascEvent(A, sprintf("max Q %.1f kPa at T+%.0f s", A.maxQ(1) / 1000, A.maxQ(2)), "info");
end
end

function A = ascAfterStep(A, cutoff, burnout, C)
% fairing jettison, impact, cutoff and staging after a step
alt = ascAltitude(A.y, C);
if A.fairingOn && alt >= A.veh.fairingAlt
    A.fairingOn = false;
    A.mass = A.mass - A.veh.fairing;
    A = ascEvent(A, sprintf("fairing jettison at %.0f km", alt), "info");
    A = ascMark(A, "Fairing", C);
end
vr = dot3(A.y(1:3), A.y(4:6));
if alt < min(0, A.spec.alt) - 0.1 && vr < 0
    A = ascRelease(A, "crash", sprintf("vehicle impacted the ground at T+%.0f s", A.t - A.t0), ...
        "alert");
    return
end
if cutoff
    A = ascInsertion(A, "orbit", C);
elseif burnout
    A = ascBurnout(A, C);
end
A = ascSample(A, false, C);
end

function A = ascBurnout(A, C)
% a stage has run dry: separate it, or end the ascent if it was the last
name = A.veh.stageNames(A.k);
alt = ascAltitude(A.y, C);
A.burning = false;
if A.k == size(A.veh.stages, 1)
    if A.spec.guidance == "orbit"
        A = ascInsertion(A, "short", C);
    else
        A = ascInsertion(A, "orbit", C);
    end
    return
end
label = name + " burnout";
if A.k == 1
    label = "MECO";
end
A = ascEvent(A, sprintf("%s at %.0f km, %.2f km/s; %s separation", label, alt, ...
    norm(A.y(4:6)), name), "launch");
A = ascMark(A, name + " sep.", C);
A = ascDrop(A, name, A.veh.stages(A.k, 5));
A.k = A.k + 1;
A.prop = A.veh.stages(A.k, 4);
A.coastUntil = A.t + A.veh.stageCoast;
end

function A = ascDrop(A, name, mass)
% shed mass kg; a tracked stage becomes an object of its own
A.mass = A.mass - mass;
if A.spec.trackStages && mass > 0
    A.spawns(end + 1) = struct("t", A.t, "name", A.spec.name + " " + name, "y", A.y, ...
        "mass", mass, "cdA", A.cdA);
end
end

function A = ascInsertion(A, outcome, C)
% cutoff (or last burnout): the payload separates from the last stage
name = A.veh.stageNames(min(A.k, size(A.veh.stages, 1)));
el = rv2coe(A.y(1:3), A.y(4:6), C.MU);
hp = el.rp - C.RE;
ha = inf;
if el.e < 1
    ha = el.ra - C.RE;
end
met = A.t - A.t0;
A.insertion = struct("t", met, "hp", hp, "ha", ha, "i", rad2deg(el.i), ...
    "raan", rad2deg(el.raan), "e", el.e, "a", el.a, "propLeft", A.prop, ...
    "alt", ascAltitude(A.y, C));
closed = el.e < 1 && hp > C.MIN_PERIGEE;
if outcome == "orbit" && ~closed
    if el.e >= 1
        outcome = "escape";
    else
        outcome = "suborbital";
    end
end
if el.e < 1
    orbit = sprintf("%s x %s km, i %.2f deg", commaNum(hp), commaNum(ha), rad2deg(el.i));
else
    orbit = sprintf("escape, e %.2f", el.e);
end
kind = "launch";
if outcome == "orbit" && A.spec.guidance == "orbit"
    text = sprintf("SECO at T+%.0f s - orbit %s; payload separation", met, orbit);
elseif outcome == "orbit" || outcome == "escape"
    text = sprintf("final burnout at T+%.0f s - %s; payload separation", met, orbit);
elseif outcome == "short"
    text = sprintf("%s ran out of propellant before orbit (T+%.0f s, perigee %s km) - " + ...
        "payload separation", name, met, commaNum(hp));
    kind = "alert";
else
    text = sprintf("burnout at T+%.0f s on a sub-orbital path (apogee %s km)", met, commaNum(ha));
end
if outcome == "orbit"
    A = ascMark(A, "Cutoff", C);
else
    A = ascMark(A, "Burnout", C);
end
A = ascSample(A, true, C);
if A.fairingOn                % cut off below the jettison height: it goes now, alone
    A.fairingOn = false;
    A.mass = A.mass - A.veh.fairing;
    A = ascEvent(A, sprintf("fairing jettison at cutoff (%.0f km)", A.insertion.alt), "info");
end
A = ascDrop(A, name, A.mass - A.spec.payloadMass);
A.mass = A.spec.payloadMass;
A = ascRelease(A, outcome, text, kind);
end

function A = ascRelease(A, outcome, text, kind)
% end the ascent with outcome, logging text
A.released = true;
A.burning = false;
A.outcome = outcome;
A.phase = "released";
A = ascEvent(A, text, kind);
end

% --- launches: steering ---------------------------------------------------------------------

function [A, mode, fixed] = ascSteer(A, s, alt, C)
% advance the steering phase; thrust mode 1 = along the air-relative velocity,
% 2 = along the unit vector fixed
r = s(1:3);
up = plumbUp(r, A.env, C);
mode = 2;
fixed = up;
if A.phase == "vertical"
    if A.t - A.t0 < A.spec.verticalTime - 1e-9
        return
    end
    if A.kick > 1e-6
        A.phase = "kick";
        A.kickStart = A.t;
        A = ascEvent(A, sprintf("pitch kick %.2f deg toward azimuth %.1f deg", A.kickDeg, ...
            mod(rad2deg(A.res.azimuth), 360)), "info");
    else
        A.phase = "turn";
    end
end
if A.phase == "kick"
    w = [s(4) + C.OMEGA * s(2), s(5) - C.OMEGA * s(1), s(6)];
    sp = sqrt(dot3(w, w));
    lean = 0;
    if sp > 1e-6
        lean = acos(max(-1, min(1, dot3(w, up) / sp)));
    end
    if lean >= A.kick || A.t - A.kickStart > 40
        A.phase = "turn";
    else
        [east, north] = eastNorth(r);
        az = A.res.azimuth;
        hz = sin(az) * east + cos(az) * north;
        fixed = unit3(cos(A.kick) * up + sin(A.kick) * hz);
        return
    end
end
if A.phase == "turn"
    if A.spec.guidance == "orbit" && (A.k >= 2 || alt >= C.GUIDANCE_ALT)
        A.phase = "guided";
        A = ascEvent(A, sprintf("closed-loop guidance at %.0f km", alt), "info");
    else
        mode = 1;
        fixed = [0 0 0];
        return
    end
end
[A, fixed] = ascGuided(A, s, C);
end

function t = ascBurnTime(A, dv, C)
% time (s) the remaining stages need to deliver dv (km/s), coasts between stages
% included; all of it if they cannot
stages = A.veh.stages;
nst = size(stages, 1);
t = 0;
if ~A.burning
    t = max(0, A.coastUntil - A.t);
end
k = A.k;
prop = A.prop;
mass = A.mass;
while k <= nst
    st = stages(k, :);
    ve = st(2) * C.G0M / 1000;
    md = stageMdot(st, C);
    if prop > 0
        dvk = ve * log(mass / (mass - prop));
        if dv <= dvk
            t = t + mass * (1 - exp(-dv / ve)) / md;
            return
        end
        t = t + prop / md;
        dv = dv - dvk;
    end
    mass = mass - (prop + st(5));
    k = k + 1;
    if k <= nst
        t = t + A.veh.stageCoast;
        prop = stages(k, 4);
    end
end
end

function [A, d] = ascGuided(A, s, C)
% closed-loop thrust direction nulling the radial and out-of-plane errors at the
% predicted cutoff
r = s(1:3);
v = s(4:6);
rm = sqrt(dot3(r, r));
up = r / rm;
n = A.res.n;
eh = unit3(cross3(n, up));
vr = dot3(v, up);
vh = dot3(v, eh);
vn = dot3(v, n);
rn = dot3(r, n);
rT = A.res.rTarget;
vT = A.res.vTarget;
gEff = C.MU / rm ^ 2 - vh * vh / rm;
gT = C.MU / rT ^ 2 - vT * vT / rT;
dvh = vT - vh;
tgo = ascBurnTime(A, sqrt(dvh * dvh + vn * vn + vr * vr), C);
for iter = 1:2
    dvR = -vr + 0.5 * (gEff + gT) * tgo;
    tgo = ascBurnTime(A, sqrt(dvh * dvh + vn * vn + dvR * dvR), C);
end
tgo = max(tgo, 1);
A.tgo = tgo;
if tgo < C.FREEZE_TGO && ~isempty(A.hold)
    sR = A.hold(1);
    sN = A.hold(2);
else
    st = A.veh.stages(min(A.k, size(A.veh.stages, 1)), :);
    p = ascPressure(ascAltitude(r, C), C);
    aT = stageMdot(st, C) * A.throttle * C.G0M * stageIsp(st, p) / A.mass * 1e-3;
    b = -12 * (rT - rm - 0.5 * vr * tgo) / tgo ^ 3;
    a = (-vr - 0.5 * b * tgo * tgo) / tgo;
    bn = -12 * (-rn - 0.5 * vn * tgo) / tgo ^ 3;
    an = (-vn - 0.5 * bn * tgo * tgo) / tgo;
    sR = max(-0.7, min(0.95, (a + gEff) / aT));
    sN = max(-0.5, min(0.5, an / aT));
    if sR * sR + sN * sN > 0.995
        mag = sqrt(max(0, 0.995 - sR * sR));
        if sN < 0 || (sN == 0 && 1 / sN < 0)
            sN = -mag;
        else
            sN = mag;
        end
    end
    A.hold = [sR, sN];
end
sH = sqrt(max(0, 1 - sR * sR - sN * sN));
d = unit3(sR * up + sN * n + sH * eh);
end

% --- launches: dynamics -----------------------------------------------------------------------

function m = stageMdot(st, C)
% mass flow at full throttle (kg/s); st = [thrust kN, Isp vac, Isp sl, propellant, dry]
m = st(1) * 1e3 / (st(2) * C.G0M);
end

function isp = stageIsp(st, p)
% specific impulse (s) at ambient pressure ratio p (an Isp sl of 0: a vacuum engine)
sl = st(3);
if sl == 0
    sl = st(2);
end
isp = st(2) - (st(2) - sl) * p;
end

function h = ascAltitude(s, C)
rm = sqrt(s(1) * s(1) + s(2) * s(2) + s(3) * s(3));
z = s(3) / rm;
h = rm - C.RE * (1 - C.F * z * z);
end

function rho = ascDensity(h, C)
% the air an ascent flies through (kg/m^3) at geometric height h km, unscaled: the US
% Standard Atmosphere 1976, p / (R T), to 86 km, the orbit table above (launch.us76_density)
if h <= C.US76_TOP_GEOMETRIC
    hc = max(h, 0);
    hg = C.US76_R0 * hc / (C.US76_R0 + hc);
    k = find(C.US76(:, 1) <= hg, 1, "last");
    temperature = C.US76(k, 2) + C.US76(k, 3) * (hg - C.US76(k, 1));
    rho = ascPressure(h, C) * C.US76_P0 / (C.US76_R_AIR * temperature);
    return
end
if h > C.TOP_KM
    rho = 0;
    return
end
hc = max(h, 0);
k = find(C.ATM(:, 1) <= hc, 1, "last");
rho = C.ATM(k, 2) * exp(-(hc - C.ATM(k, 1)) / C.ATM(k, 3));
end

function g = ascGravity(x, y, z, env, C)
% point mass plus the enabled zonal terms (km/s^2)
r2 = x * x + y * y + z * z;
r = sqrt(r2);
k = -C.MU / (r2 * r);
ax = k * x;
ay = k * y;
az = k * z;
if env.j2
    f = -1.5 * C.J2 * C.MU * C.RE ^ 2 / (r2 * r2 * r);
    zz = 5 * z * z / r2;
    ax = ax + f * x * (1 - zz);
    ay = ay + f * y * (1 - zz);
    az = az + f * z * (3 - zz);
end
if env.j3 || env.j4
    r7 = r2 * r2 * r2 * r;
    if env.j3
        f = -2.5 * C.J3 * C.MU * C.RE ^ 3 / r7;
        txy = 3 * z - 7 * z ^ 3 / r2;
        ax = ax + f * x * txy;
        ay = ay + f * y * txy;
        az = az + f * (6 * z * z - 7 * z ^ 4 / r2 - 0.6 * r2);
    end
    if env.j4
        f = 1.875 * C.J4 * C.MU * C.RE ^ 4 / r7;
        sz = z * z / r2;
        txy = 1 - 14 * sz + 21 * sz * sz;
        ax = ax + f * x * txy;
        ay = ay + f * y * txy;
        az = az + f * z * (5 - 70 / 3 * sz + 21 * sz * sz);
    end
end
g = [ax, ay, az];
end

function ds = ascDeriv(s, tau, m0, mdot, st, mode, fixed, cdA, env, C)
% state derivative: gravity, drag in the turning atmosphere, thrust (mode 1 along the
% air-relative velocity, 2 along fixed; mdot 0 means none)
g = ascGravity(s(1), s(2), s(3), env, C);
w = [s(4) + C.OMEGA * s(2), s(5) - C.OMEGA * s(1), s(6)];
alt = ascAltitude(s, C);
m = m0 - mdot * tau;
sp = sqrt(w(1) * w(1) + w(2) * w(2) + w(3) * w(3));
if cdA > 0 && alt < C.TOP_KM
    rho = ascDensity(alt, C) * env.densityScale;
    g = g + (-0.5e3 * rho * cdA / m * sp) * w;
end
if mdot > 0
    a = mdot * C.G0M * stageIsp(st, ascPressure(alt, C)) / m * 1e-3;
    if mode == 1 && sp > 1e-6
        g = g + a * w / sp;
    else
        g = g + a * fixed;
    end
end
ds = [s(4:6), g];
end

function s = ascRk4(s, h, m0, mdot, st, mode, fixed, cdA, env, C)
k1 = ascDeriv(s, 0, m0, mdot, st, mode, fixed, cdA, env, C);
k2 = ascDeriv(s + 0.5 * h * k1, 0.5 * h, m0, mdot, st, mode, fixed, cdA, env, C);
k3 = ascDeriv(s + 0.5 * h * k2, 0.5 * h, m0, mdot, st, mode, fixed, cdA, env, C);
k4 = ascDeriv(s + h * k3, h, m0, mdot, st, mode, fixed, cdA, env, C);
s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
end

function s = ascCoast(s, t0, t1, mass, cdA, env, C)
% unpowered flight from t0 to t1 (a separated object catching up with the run)
t = t0;
while t < t1 - 1e-9
    h = min(C.H_COAST, t1 - t);
    s = ascRk4(s, h, mass, 0, [], 0, [0 0 0], cdA, env, C);
    t = t + h;
end
end

function e = ascEnergy(s, C)
e = 0.5 * (s(4) * s(4) + s(5) * s(5) + s(6) * s(6)) - C.MU / sqrt(s(1) * s(1) + s(2) * s(2) + ...
    s(3) * s(3));
end

function up = plumbUp(r, env, C)
% local vertical of the turning Earth: against gravity plus the centrifugal term
g = ascGravity(r(1), r(2), r(3), env, C);
w2 = C.OMEGA * C.OMEGA;
up = unit3([-(g(1) + w2 * r(1)), -(g(2) + w2 * r(2)), -g(3)]);
end

function [east, north] = eastNorth(r)
rho = hypot(r(1), r(2));
if rho > 1e-9
    east = [-r(2) / rho, r(1) / rho, 0];
else
    east = [0 1 0];
end
north = cross3(unit3(r), east);
end

function u = unit3(v)
n = sqrt(v(1) * v(1) + v(2) * v(2) + v(3) * v(3));
if n > 0
    u = v / n;
else
    u = [0 0 0];
end
end

function c = cross3(a, b)
c = [a(2) * b(3) - a(3) * b(2), a(3) * b(1) - a(1) * b(3), a(1) * b(2) - a(2) * b(1)];
end

function d = dot3(a, b)
d = a(1) * b(1) + a(2) * b(2) + a(3) * b(3);
end

% --- launches: planning (launch window and pitch kick) ----------------------------------------

function launchCheck(L, C)
% the simulator's checks on a launch (launch.validate), with its messages
V = L.vehicle;
S = V.stages;
if isempty(S)
    error("satflight:launch", "the vehicle has no stages");
end
for k = 1:size(S, 1)
    if min(S(k, [1 2 4])) <= 0 || S(k, 5) < 0 || S(k, 3) < 0
        error("satflight:launch", "%s: thrust, Isp and propellant must be positive", ...
            V.stageNames(k));
    end
end
if V.fairing < 0 || V.diameter < 0 || V.cd < 0
    error("satflight:launch", "masses, diameter and Cd cannot be negative");
end
if L.payloadMass < 0
    error("satflight:launch", "the payload mass cannot be negative");
end
m = L.payloadMass + V.fairing + sum(S(:, 4) + S(:, 5));
for k = 1:size(S, 1)
    if m - S(k, 4) <= 0
        error("satflight:launch", "%s would burn out at zero mass: give the last stage a " + ...
            "dry mass or carry a payload", V.stageNames(k));
    end
    m = m - S(k, 4) - S(k, 5) - V.fairing * (k == 1);
    burn = S(k, 4) / stageMdot(S(k, :), C);
    if burn > C.MAX_BURN
        error("satflight:launch", "%s would burn %.0f min; the ascent model allows %.0f " + ...
            "(use a finite-burn maneuver instead)", V.stageNames(k), burn / 60, C.MAX_BURN / 60);
    end
end
if L.inclination < 0 || L.inclination > 180
    error("satflight:launch", "target inclination (deg) must be between 0 and 180, got %g", ...
        L.inclination);
end
if abs(L.lat) > 90
    error("satflight:launch", "latitude must be within +/-90 deg");
end
if L.guidance == "orbit" && min(L.perigeeAlt, L.apogeeAlt) < C.MIN_PERIGEE
    error("satflight:launch", "target perigee must be at least %.0f km", C.MIN_PERIGEE);
end
if L.verticalTime < 0
    error("satflight:launch", "vertical rise time cannot be negative");
end
end

function n = planeNormal(inc, raan)
n = [sin(inc) * sin(raan), -sin(inc) * cos(raan), cos(inc)];
end

function text = unreachable(incDeg, latDeg)
lo = abs(latDeg);
text = sprintf("inclination %.1f deg cannot be reached directly from latitude %+.1f deg: " + ...
    "choose %.1f to %.1f deg (dogleg ascents are not modeled)", incDeg, latDeg, lo, 180 - lo);
end

function best = planeThrough(rHat, inc, direction)
% normal of the plane of inclination inc (rad) through the unit position rHat,
% crossing it northbound or southbound
a = rHat(1);
b = rHat(2);
c = rHat(3);
rho = hypot(a, b);
si = sin(inc);
ci = cos(inc);
lat = rad2deg(asin(max(-1, min(1, c))));
if si < 1e-9 || rho < 1e-9
    if abs(c) < 1e-6 && si < 1e-9
        best = [0 0 1];
        if ~(ci > 0)
            best = [0 0 -1];
        end
        return
    end
    error("satflight:launch", "%s", unreachable(rad2deg(inc), lat));
end
q = -c * ci / (rho * si);
if abs(q) > 1
    error("satflight:launch", "%s", unreachable(rad2deg(inc), lat));
end
lam = atan2(b, a);
best = [];
for psi = [lam + acos(q), lam - acos(q)]
    n = [si * cos(psi), si * sin(psi), ci];
    side = cross3(n, rHat);
    if (side(3) >= 0) == (direction == "north")
        best = n;
    end
end
if isempty(best)
    best = [si * cos(lam + acos(q)), si * sin(lam + acos(q)), ci];
end
end

function t = nextWindow(L, normalAt, tStart, env, C)
% first time >= tStart at which the pad lies in the plane normalAt(t), crossed in the
% launch direction
rE = geodeticToEcef(deg2rad(L.lat), deg2rad(L.lon), L.alt, C);
u = rE / norm(rE);
span = 2 * 86400;
step = 120;
pad = @(t) padUnit(u, ascGmst(env, t));
f = @(t) dot3(pad(t), normalAt(t));
n0 = normalAt(tStart);
inc = rad2deg(acos(max(-1, min(1, n0(3)))));
latGc = rad2deg(asin(u(3)));
if ~(abs(latGc) - 1e-6 <= inc && inc <= 180 - abs(latGc) + 1e-6)
    error("satflight:launch", "%s", unreachable(inc, latGc));
end
ta = tStart;
fa = f(ta);
while ta < tStart + span
    tb = ta + step;
    fb = f(tb);
    if fa == 0 || fa * fb < 0
        lo = ta;
        hi = tb;
        flo = fa;
        for iter = 1:50
            mid = 0.5 * (lo + hi);
            fm = f(mid);
            if flo * fm <= 0
                hi = mid;
            else
                lo = mid;
                flo = fm;
            end
        end
        t = 0.5 * (lo + hi);
        c = cross3(normalAt(t), pad(t));
        if (c(3) >= 0) == (L.direction == "north") || inc < 1e-3 || inc > 180 - 1e-3
            return
        end
    end
    ta = tb;
    fa = fb;
end
error("satflight:launch", "no launch window in the next two days");
end

function p = padUnit(u, th)
c = cos(th);
sn = sin(th);
p = [c * u(1) - sn * u(2), sn * u(1) + c * u(2), u(3)];
end

function res = resolveLaunch(L, env, tNow, planeOf, C)
% liftoff time, target plane and launch azimuth (launch.resolve)
normalAt = [];
inc = L.inclination;
if L.guidance == "orbit" && L.timing == "raan"
    nFix = planeNormal(deg2rad(inc), deg2rad(L.raan));
    normalAt = @(t) nFix;
elseif L.guidance == "orbit" && L.timing == "plane"
    if strlength(L.target) == 0
        error("satflight:launch", "choose the satellite whose plane to launch into");
    end
    [inc, normalAt] = planeOf(L.target);
end
if L.timing == "delay"
    t0 = tNow + max(0, L.delay);
elseif L.timing == "absolute"
    t0 = tNow;
    if ~isnan(L.t0)
        t0 = max(tNow, L.t0);
    end
elseif ~isempty(normalAt)
    t0 = nextWindow(L, normalAt, tNow + 1, env, C);
else
    t0 = tNow;
end
res = struct("t0", t0, "azimuth", deg2rad(L.azimuth), "n", [], "inclination", NaN, ...
    "rTarget", NaN, "vTarget", NaN, "eTarget", NaN, "kick", NaN);
if L.guidance == "open"
    return
end
hp = min(L.perigeeAlt, L.apogeeAlt);
ha = max(L.perigeeAlt, L.apogeeAlt);
rT = C.RE + hp;
aT = C.RE + 0.5 * (hp + ha);
vT = sqrt(C.MU * (2 / rT - 1 / aT));
y0 = padState(L, ascGmst(env, t0), C);
r0 = y0(1:3);
rHat = unit3(r0);
if isempty(normalAt)
    n = planeThrough(rHat, deg2rad(inc), L.direction);
else
    n = normalAt(t0);
end
d = unit3(cross3(n, rHat));
[east, north] = eastNorth(r0);
beta = atan2(dot3(d, east), dot3(d, north));
vRot = C.OMEGA * hypot(r0(1), r0(2));
res.azimuth = atan2(vT * sin(beta) - vRot, vT * cos(beta));
res.n = n;
res.inclination = rad2deg(acos(max(-1, min(1, n(3)))));
res.rTarget = rT;
res.vTarget = vT;
res.eTarget = -C.MU / (2 * aT);
end

function sc = launchScore(A, L, C)
% the planner's objective: reaching orbit first, then accuracy and propellant
ins = A.insertion;
if isempty(ins)
    sc = -1e12;
    return
end
if L.guidance == "open"
    if A.outcome == "orbit"
        sc = 1e7 + ins.hp;
    elseif isfinite(ins.ha)
        sc = ins.ha;
    else
        sc = 1e7;
    end
    return
end
if A.outcome == "orbit"
    miss = abs(ins.hp - min(L.perigeeAlt, L.apogeeAlt)) + abs(ins.ha - max(L.perigeeAlt, ...
        L.apogeeAlt));
    sc = 1e7 + ins.propLeft - 20 * miss;
    return
end
e = 0;
if ins.e < 1
    e = -C.MU / (2 * ins.a);
end
sc = -1e7 + e * 1e3;
end

function [res, kick, flight] = planLaunch(L, env, tNow, planeOf, C)
% resolve the launch and fly it; with no kick given, search the pitch kick that
% reaches the target with the most propellant left (launch.plan_launch)
res = resolveLaunch(L, env, tNow, planeOf, C);
if ~isnan(L.kick)
    kick = L.kick;
    flight = ascentFly(L, res, env, kick, true, C);
    res.kick = kick;
    return
end
tried = [zeros(0, 1), zeros(0, 1)];          % kick deg, score, in the order flown
grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0, 16.0];
scores = zeros(size(grid));
for j = 1:numel(grid)
    [scores(j), tried] = kickScore(grid(j), tried, L, res, env, C);
end
[~, j] = max(scores);
lo = grid(max(1, j - 1));
hi = grid(min(numel(grid), j + 1));
g = (sqrt(5) - 1) / 2;
a = hi - g * (hi - lo);
b = lo + g * (hi - lo);
for iter = 1:7
    [sa, tried] = kickScore(a, tried, L, res, env, C);
    [sb, tried] = kickScore(b, tried, L, res, env, C);
    if sa >= sb
        hi = b;
        b = a;
        a = hi - g * (hi - lo);
    else
        lo = a;
        a = b;
        b = lo + g * (hi - lo);
    end
end
[~, j] = max(tried(:, 2));
kick = tried(j, 1);
flight = ascentFly(L, res, env, kick, true, C);
res.kick = kick;
end

function [sc, tried] = kickScore(k, tried, L, res, env, C)
% the planner's score of pitch kick k (deg), each kick flown once
k = round(k, 3);
j = find(tried(:, 1) == k, 1);
if isempty(j)
    tried(end + 1, :) = [k, launchScore(ascentFly(L, res, env, k, false, C), L, C)];
    j = size(tried, 1);
end
sc = tried(j, 2);
end

% --- output --------------------------------------------------------------------------------

function T = ephemerisTable(s, C)
% one row per object per sample, from the sample it first exists at (as the batch
% runner writes them)
n = numel(s.names);
nt = numel(s.tout);
rows = n * nt;
t = reshape(repmat(s.tout, n, 1), rows, 1);
Y = reshape(permute(s.Yout, [1 3 2]), rows, 6);
status = reshape(s.statusOut, rows, 1);
names = repmat(s.names(:), nt, 1);
there = status ~= "";
t = t(there);
Y = Y(there, :);
status = status(there);
names = names(there);
theta = gmst(s.sim.jd0 + t / 86400);
c = cos(theta); sn = sin(theta);
ecef = [c .* Y(:, 1) + sn .* Y(:, 2), -sn .* Y(:, 1) + c .* Y(:, 2), Y(:, 3)];
[lat, lon, alt] = ecefToGeodetic(ecef, C);
el = rv2coe(Y(:, 1:3), Y(:, 4:6), C.MU);
T = table(t, s.sim.epoch + seconds(t), names, status, Y(:, 1), Y(:, 2), Y(:, 3), Y(:, 4), ...
    Y(:, 5), Y(:, 6), alt, rad2deg(lat), rad2deg(lon), el.a, el.e, rad2deg(el.i), ...
    rad2deg(el.raan), rad2deg(el.argp), rad2deg(el.nu), VariableNames=["t_s", "utc", "name", ...
    "status", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms", "alt_km", "lat_deg", ...
    "lon_deg", "a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]);
end

function C = satflightConstants()
% satflight/constants.py
C.MU = 398600.4418;
C.RE = 6378.137;
C.F = 1 / 298.257223563;
C.E2 = C.F * (2 - C.F);
C.OMEGA = 7.292115146706979e-5;
C.J2 = 1.08262668355e-3;
C.J3 = -2.53265648533e-6;
C.J4 = -1.61962159137e-6;
C.C22 = 1.57446037456e-6;
C.S22 = -9.03803806639e-7;
C.SOI = 924000.0;
C.R_SUN = 696000.0;
C.AU = 149597870.7;
C.MU_SUN = 1.32712440018e11;  % km^3/s^2
C.MU_MOON = 4902.800066;      % km^3/s^2
C.P_SRP = 4.56e-6;            % N/m^2, sunlight pressure at 1 AU
% satflight/scenario.py
C.CR_DEFAULT = 1.3;           % sunlight reflectivity of an object given none
% satflight/poeorb.py: the precise orbit files' checks
C.POEORB_R = [6900.0 7250.0];         % km: Sentinel-1 flies ~693 km up
C.POEORB_V = [6.5 8.0];               % km/s, Earth-fixed speed
C.POEORB_V_TOL = 0.0005;              % km/s: velocity vs the positions' central difference
C.SEAM_TOL_M = 10.0;                  % m: neighbouring days must agree where they overlap
C.QUALITIES = ["NOMINAL" "DEGRADED-MANOEUVRE" "DEGRADED-GAP"];
C.DEG = pi / 180;
C.G0 = 9.80665e-3;            % km/s^2
% satflight/simulation.py
C.REENTRY_ALT = 80.0;         % km
C.ASCENT_SYNC = 1.0;          % s
C.EVENT_DETAIL_LIMIT = 40;
C.ALL_PAIRS_LIMIT = 300;
C.DIP_MARGIN = 10.0;          % km
C.DIP_CHECK_LIMIT = 2000;
C.CONJUNCTION_KM = 10.0;
C.APSIS_TIME_TOL = 1e-6;      % s
C.CIRC_SAMPLES = 72;
C.CIRC_TOL = 1e-6;            % km
C.CIRC_ITER = 8;
C.CIRC_STEP = 1e-5;           % km/s
% the exported scripts' analysis (not the simulator's): below these the node and the
% perigee are not defined well enough to plot or to measure a drift of (J2 alone swings a
% near-circular orbit's osculating e by about 0.001, and its perigee all the way round)
C.I_DEFINED = 0.1;            % deg (or within it of 180)
C.E_DEFINED = 0.01;
% satflight/launch.py
C.G0M = 9.80665;              % m/s^2
C.THROTTLE_MIN = 0.4;
% US Standard Atmosphere 1976 below 86 km: geopotential base km, base K, lapse K/km
C.US76 = [0 288.15 -6.5; 11 216.65 0; 20 216.65 1.0; 32 228.65 2.8; 47 270.65 0;
    51 270.65 -2.8; 71 214.65 -2.0];
C.US76_TOP = 84.852;          % km geopotential (86 km geometric)
C.US76_TOP_GEOMETRIC = 86.0;  % km: above it ascents fly the orbit table
C.US76_P0 = 101325.0;         % Pa, sea-level pressure
C.US76_R_AIR = 287.053;       % J/(kg K)
C.US76_GMR = 34.163195;       % K/km
C.US76_R0 = 6356.766;         % km
C.US76_P = ones(size(C.US76, 1), 1);
for k = 2:size(C.US76, 1)
    C.US76_P(k) = C.US76_P(k - 1) * us76Layer(C.US76(k - 1, :), C.US76(k, 1) - C.US76(k - 1, 1), ...
        C.US76_GMR);
end
C.US76_P_TOP = C.US76_P(end) * us76Layer(C.US76(end, :), C.US76_TOP - C.US76(end, 1), C.US76_GMR);
C.US76_T_TOP = C.US76(end, 2) + C.US76(end, 3) * (C.US76_TOP - C.US76(end, 1));
C.GUIDANCE_ALT = 50.0;        % km
C.H_POWERED = 1.0;            % s
C.H_COAST = 5.0;              % s
C.SAMPLE_DT = 2.0;            % s
C.FREEZE_TGO = 4.0;           % s
C.MIN_PERIGEE = 100.0;        % km
C.MAX_BURN = 3600.0;          % s
% satflight/atmosphere.py: base altitude km, density kg/m^3, scale height km
C.TOP_KM = 2500.0;
C.ATM_TOP = 2500;              % km: no air above it, in the table and NRLMSIS
% satflight/drag_coefficient.py and atmosphere.py: the physical Cd
C.CD_REF = 2.2;                % the Cd a satellite's Cd A / m is taken at (forces.CD_REF)
C.K_B = 1.380649e-23;          % J/K
C.SESAM_K = 7.50e-17;          % m^3/K
C.WALL_K = 300.0;              % K
C.MIN_SPEED_RATIO = 0.1;
C.SPECIES_SUM_TOL = 0.01;      % species vs total density
C.SPECIES_KG = [28.0134 31.9988 15.9994 4.002602 1.00794 39.948 14.0067 15.9994 ...
    30.0061] * 1.66053907e-27;   % N2 O2 O He H Ar N anomalous-O NO
% CelesTrak's SW-All.csv columns (spaceweather.HEADER)
C.SW_HEADER = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5," + ...
    "AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE," + ...
    "F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81";
C.ATM = [0 1.225 7.249; 25 3.899e-2 6.349; 30 1.774e-2 6.682; 40 3.972e-3 7.554;
    50 1.057e-3 8.382; 60 3.206e-4 7.714; 70 8.770e-5 6.549; 80 1.905e-5 5.799;
    90 3.396e-6 5.382; 100 5.297e-7 5.877; 110 9.661e-8 7.263; 120 2.438e-8 9.473;
    130 8.484e-9 12.636; 140 3.845e-9 16.149; 150 2.070e-9 22.523; 180 5.464e-10 29.740;
    200 2.789e-10 37.105; 250 7.248e-11 45.546; 300 2.418e-11 53.628; 350 9.518e-12 53.298;
    400 3.725e-12 58.515; 450 1.585e-12 60.828; 500 6.967e-13 63.822; 600 1.454e-13 71.835;
    700 3.614e-14 88.667; 800 1.170e-14 124.64; 900 5.245e-15 181.05; 1000 3.019e-15 268.00];
end

function m = satflightBurn(sat, kind, t, dv, frame, deltaI, target, thrust, isp, duration, label)
% one scheduled maneuver; t is seconds after the epoch
m = struct("sat", string(sat), "kind", string(kind), "t", t, "dv", dv, ...
    "frame", string(frame), "deltaI", deltaI, "target", string(target), "thrust", thrust, ...
    "isp", isp, "duration", duration, "label", string(label));
end

function m = satflightNoBurns()
m = satflightBurn("", "", 0, [0 0 0], "", 0, "", 0, 1, 0, "");
m = m([]);
end

function V = satflightVehicle(name, stageNames, stages, fairing, fairingAlt, diameter, cd, ...
    maxG, stageCoast)
% a launcher; stages has one row per stage, first to burn first:
% thrust (kN, vacuum), Isp vacuum (s), Isp sea level (s, 0 for a vacuum engine),
% propellant (kg), dry mass (kg)
V = struct("name", string(name), "stageNames", reshape(string(stageNames), [], 1), ...
    "stages", stages, "fairing", fairing, "fairingAlt", fairingAlt, "diameter", diameter, ...
    "cd", cd, "maxG", maxG, "stageCoast", stageCoast);
end

function L = satflightLaunch(name, vehicle)
% a launch with the simulator's defaults (launch.LaunchSpec); set its fields after
L = struct("name", string(name), "vehicle", vehicle, "site", "", "lat", 28.562, ...
    "lon", -80.577, "alt", 0, "timing", "now", "delay", 0, "t0", NaN, "guidance", "orbit", ...
    "perigeeAlt", 400, "apogeeAlt", 400, "inclination", 51.6, "direction", "north", ...
    "raan", 0, "target", "", "azimuth", 90, "verticalTime", 10, "kick", NaN, ...
    "payloadMass", 1000, "payloadArea", 5, "payloadCd", 2.2, "payloadCr", 1.3, ...
    "circularize", false, ...
    "trackStages", true, "plan", [], "state", []);
end

function L = satflightNoLaunches()
L = satflightLaunch("", satflightVehicle("", "", zeros(0, 5), 0, 0, 0, 0, 0, 0));
L = L([]);
end

function P = satflightPlan(t0, azimuth, n, inclination, rTarget, vTarget, eTarget, kick)
% the simulator's plan of a launch: liftoff (s after the epoch), launch azimuth
% (rad), target plane normal, inclination (deg), insertion radius (km) and speed
% (km/s), target orbital energy (km^2/s^2), pitch kick (deg)
P = struct("t0", t0, "azimuth", azimuth, "n", n, "inclination", inclination, ...
    "rTarget", rTarget, "vTarget", vTarget, "eTarget", eTarget, "kick", kick);
end

% --- plots ---------------------------------------------------------------------------------

function names = satflightPlotNames(T)
%SATFLIGHTPLOTNAMES The objects to plot when none are named: all of them up to 12, and
%   past that the first 12 still in orbit at the end of the run, then the rest (in the
%   order they appear in T). Says so when it leaves some out.
present = unique(T.name, "stable")';
last = T(T.t_s == max(T.t_s), :);
inOrbit = ismember(present, last.name(last.status == "active"));
ranked = [present(inOrbit), present(~inOrbit)];
names = present(ismember(present, ranked(1:min(12, end))));
if numel(names) < numel(present)
    fprintf("Plotting %d of the %d objects (those still in orbit first); name them to choose.\n", ...
        numel(names), numel(present));
end
end

function styleAxes(ax)
% twelve distinct line colours: no two of the (at most 12) objects plotted share one
colororder(ax, orderedcolors("gem12"));
end

function P = satflightPalette(style)
%SATFLIGHTPALETTE The colours the plots draw with in a light or a dark theme (style
%   "light" or "dark", as a figure's Theme.BaseColorStyle): what the theme does not
%   colour itself. Map day and night sides, coastlines on a map and on the globe, the
%   Earth, stations, marker rims, muted lines, timeline bars, burn marks and labels.
if style == "dark"
    P = struct("text", [0.88 0.88 0.88], "back", [0.07 0.07 0.07], ...
        "day", [0.15 0.18 0.24], "night", [0.025 0.03 0.05], "coast", [0.62 0.66 0.72], ...
        "earth", [0.17 0.29 0.47], "land", [0.68 0.78 0.9], "station", [0.95 0.95 0.95], ...
        "rim", [0.92 0.92 0.92], "muted", [0.62 0.62 0.62], "life", [0.5 0.5 0.5], ...
        "shadow", [0.3 0.34 0.44], "burn", [1 0.55 0.15]);
else
    P = struct("text", [0.13 0.13 0.13], "back", [1 1 1], ...
        "day", [0.985 0.99 1], "night", [0.78 0.82 0.9], "coast", [0.33 0.36 0.4], ...
        "earth", [0.6 0.74 0.92], "land", [0.16 0.26 0.4], "station", [0.1 0.1 0.1], ...
        "rim", [0.1 0.1 0.1], "muted", [0.45 0.45 0.45], "life", [0.62 0.62 0.62], ...
        "shadow", [0.33 0.33 0.33], "burn", [0.85 0.33 0]);
end
end

function themed(h, role)
% marks graphics h as drawn in palette colour role; applyTheme colours them
set(h, Tag="satflight:" + role);
end

function themeFigure(fig)
% colours fig's marked graphics for its theme now and again whenever the theme
% changes (MATLAB settles a new figure's theme only after it is made: a figure that
% says "light" when created can be drawn dark)
fig.Tag = "satflight";
fig.ThemeChangedFcn = @(f, ~) applyTheme(f);
applyTheme(fig);
end

function applyTheme(fig)
P = satflightPalette(string(fig.Theme.BaseColorStyle));
paint = @(role, varargin) set(findall(fig, Tag="satflight:" + role), varargin{:});
paint("day", Color=P.day);
paint("night", FaceColor=P.night);
paint("coast", Color=P.coast);
paint("earth", FaceColor=P.earth);
paint("land", Color=P.land);
paint("station", MarkerFaceColor=P.station, MarkerEdgeColor=P.back);
paint("rim", MarkerEdgeColor=P.rim);
paint("muted", Color=P.muted);
paint("life", Color=P.life);
paint("shadow", Color=P.shadow);
paint("burn", Color=P.burn, MarkerFaceColor=P.burn);
paint("label", Color=P.text);
for h = reshape(findall(fig, Tag="satflight:fade"), 1, [])
    fadeTrail(h, P);
end
end

function fadeTrail(h, P)
% a trail (a patch line, its colour in UserData) fading from the background at its
% oldest point to its full colour at the newest, without transparency (slow to draw)
n = numel(h.XData);
if n < 2
    return
end
w = linspace(0.12, 1, n)';
h.FaceVertexCData = w * h.UserData + (1 - w) * P.back;
end

function c = lineColor(k)
% the k-th of styleAxes' colours, for a line and its marker drawn with the same one
% (a marker plotted without a colour would take the next colour in the order)
colors = orderedcolors("gem12");
c = colors(mod(k - 1, size(colors, 1)) + 1, :);
end

function names = plotNames(T, names)
present = unique(T.name, "stable")';
if isempty(names)
    names = satflightPlotNames(T);
else
    names = reshape(string(names), 1, []);
    unknown = setdiff(names, present);
    if ~isempty(unknown)
        error("satflight:unknownSatellite", "No satellite named %s.", strjoin(unknown, ", "));
    end
end
end

function fig = satflightPlotGroundTrack(T, stations, names)
arguments
    T table
    stations = struct("name", {}, "lat", {}, "lon", {})
    names = string.empty
end
names = plotNames(T, names);
fig = figure(Name="Ground track");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
box(ax, "on");
[night, sun] = drawNight(ax);       % as at the end markers
moveNight(night, sun, julianAt(T, max(T.t_s)), satflightConstants());
coast = satflightCoastline();
themed(plot(ax, coast(:, 1), coast(:, 2), LineWidth=0.5, HandleVisibility="off"), "coast");
for k = 1:numel(names)
    rows = T(T.name == names(k), :);
    lon = rows.lon_deg;
    lat = rows.lat_deg;
    for j = flip(find(abs(diff(lon)) > 180))'
        lon = [lon(1:j); NaN; lon(j + 1:end)];
        lat = [lat(1:j); NaN; lat(j + 1:end)];
    end
    plot(ax, lon, lat, Color=lineColor(k), LineWidth=1.2, DisplayName=names(k));
    themed(plot(ax, rows.lon_deg(end), rows.lat_deg(end), "o", Color=lineColor(k), ...
        MarkerFaceColor=lineColor(k), MarkerSize=7, HandleVisibility="off"), "rim");
end
for g = reshape(stations, 1, [])
    themed(plot(ax, g.lon, g.lat, "^", MarkerSize=8, HandleVisibility="off"), "station");
    themed(text(ax, g.lon + 2, g.lat, g.name, FontSize=8, Interpreter="none"), "label");
end
xlim(ax, [-180 180]); ylim(ax, [-90 90]); daspect(ax, [1 1 1]);
xticks(ax, -180:60:180); yticks(ax, -90:30:90); grid(ax, "on");
xlabel(ax, "Longitude (deg)"); ylabel(ax, "Latitude (deg)");
fmt = "yyyy-MM-dd HH:mm";
title(ax, sprintf("Ground track, %s to %s UTC", string(T.utc(1), fmt), string(T.utc(end), fmt)));
subtitle(ax, "o where each object is at the end; the night side as then");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function fig = satflightPlotAltitude(T, names)
arguments
    T table
    names = string.empty
end
names = plotNames(T, names);
fig = figure(Name="Altitude");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
for nm = names
    rows = T(T.name == nm, :);
    plot(ax, rows.t_s / 3600, rows.alt_km, LineWidth=1.2, DisplayName=nm);
end
grid(ax, "on");
ax.YLim(1) = 0;          % the ground; nothing is plotted below it
xlabel(ax, "Time since epoch (h)"); ylabel(ax, "Altitude (km)"); title(ax, "Altitude");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function fig = satflightPlotElements(T, names)
arguments
    T table
    names = string.empty
end
names = plotNames(T, names);
C = satflightConstants();
panels = ["a_km", "Semi-major axis (km)"; "e", "Eccentricity"; "i_deg", "Inclination (deg)"; ...
    "raan_deg", sprintf("RAAN (deg)|where i \\geq %g\\circ", C.I_DEFINED); ...
    "argp_deg", sprintf("Arg. of perigee (deg)|where e \\geq %g", C.E_DEFINED); ...
    "alt_km", "Altitude (km)"];
fig = figure(Name="Orbital elements");
layout = tiledlayout(fig, 2, 3, TileSpacing="compact");
title(layout, "Osculating elements");
subtitle(layout, "in orbit only: a gap where an object is on its way up or on a path " + ...
    "that meets the Earth");
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    styleAxes(ax);
    hold(ax, "on");
    for nm = names
        rows = T(T.name == nm, :);
        y = rows.(panels(p, 1));
        if panels(p, 1) ~= "alt_km"
            y(~inOrbit(rows, C)) = NaN;
        end
        % the node of an orbit in the equator, and the perigee of a circular one, are
        % not defined: they wander and would swamp the panel
        if panels(p, 1) == "raan_deg"
            y(~nodeDefined(rows, C)) = NaN;
        elseif panels(p, 1) == "argp_deg"
            % (an orbit in the equator measures its perigee from the x axis: not this)
            y(rows.e < C.E_DEFINED | ~nodeDefined(rows, C)) = NaN;
        end
        if ismember(panels(p, 1), ["raan_deg", "argp_deg"])
            ok = ~isnan(y);
            y(ok) = rad2deg(unwrap(deg2rad(y(ok))));
        end
        plot(ax, rows.t_s / 3600, y, LineWidth=1.1, DisplayName=nm);
    end
    grid(ax, "on");
    if panels(p, 1) == "alt_km"
        ax.YLim(1) = 0;      % the ground
    end
    xlabel(ax, "Time (h)"); ylabel(ax, split(panels(p, 2), "|"));
end
lg = legend(ax, Interpreter="none");
lg.Layout.Tile = "east";     % beside the panels, not over one of them
themeFigure(fig);
end

function fig = satflightPlotOrbits(T, names)
arguments
    T table
    names = string.empty
end
C = satflightConstants();
names = plotNames(T, names);
fig = figure(Name="Orbits (ECI)");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
earth = drawEarth(ax, C);
tEnd = max(T.t_s);
earth.Matrix = makehgtform("zrotate", gmst(julianAt(T, tEnd)));   % as at the end markers
for k = 1:numel(names)
    rows = T(T.name == names(k), :);
    plot3(ax, rows.x_km, rows.y_km, rows.z_km, Color=lineColor(k), LineWidth=1.2, ...
        DisplayName=names(k));
    themed(plot3(ax, rows.x_km(end), rows.y_km(end), rows.z_km(end), "o", ...
        Color=lineColor(k), MarkerFaceColor=lineColor(k), HandleVisibility="off"), "rim");
end
axis(ax, "equal"); grid(ax, "on"); view(ax, 35, 25);
xlabel(ax, "x (km)"); ylabel(ax, "y (km)"); zlabel(ax, "z (km)");
title(ax, "Trajectories, ECI", "the Earth turned as at " + string(T.utc(find(T.t_s == tEnd, ...
    1)), "yyyy-MM-dd HH:mm") + " UTC");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function fig = satflightPlotAscent(launches, names)
%SATFLIGHTPLOTASCENT Ascent profiles of the launches: altitude against downrange
%   distance, speed, dynamic pressure and acceleration against mission time, with
%   the staging events marked and the simulator's plan dashed.
arguments
    launches struct
    names = string.empty
end
fig = gobjects(0);
if isempty(launches)
    return
end
if ~isempty(names)
    launches = launches(ismember([launches.name], string(names)));
end
for L = reshape(launches, 1, [])
    if isempty(L.profile)
        continue
    end
    fig(end + 1) = figure(Name="Ascent: " + L.name); %#ok<AGROW>
    layout = tiledlayout(fig(end), 2, 2, TileSpacing="compact");
    title(layout, sprintf("%s - %s, %s", L.name, L.vehicle, L.outcome), Interpreter="none");
    panels = ["downrange_km", "alt_km", "Downrange (km)", "Altitude (km)";
        "met_s", "speed_kms", "Mission time (s)", "Inertial speed (km/s)";
        "met_s", "q_kPa", "Mission time (s)", "Dynamic pressure (kPa)";
        "met_s", "accel_g", "Mission time (s)", "Thrust acceleration (g)"];
    for k = 1:size(panels, 1)
        ax = nexttile(layout);
        hold(ax, "on");
        grid(ax, "on");
        themed(plot(ax, L.planned.(panels(k, 1)), L.planned.(panels(k, 2)), "--", ...
            DisplayName="plan"), "muted");
        plot(ax, L.profile.(panels(k, 1)), L.profile.(panels(k, 2)), LineWidth=1.3, ...
            DisplayName="flown");
        xlabel(ax, panels(k, 3));
        ylabel(ax, panels(k, 4));
        if k == 1
            themed(plot(ax, L.marks.downrange_km, L.marks.alt_km, "o", ...
                HandleVisibility="off"), "station");
            [x, y, labels] = mergeMarks(L.marks.downrange_km, L.marks.alt_km, ...
                string(L.marks.label), L.profile.downrange_km, L.profile.alt_km);
            themed(text(ax, x, y, "  " + labels, FontSize=7, Interpreter="none"), "label");
            legend(ax, Location="southeast");
        else
            for j = 1:height(L.marks)
                themed(xline(ax, L.marks.met_s(j), ":", HandleVisibility="off"), "muted");
            end
        end
    end
    themeFigure(fig(end));
end
end

function [x, y, labels] = mergeMarks(x, y, labels, xs, ys)
% marks (x, y) labelled with one text each, those that would print on top of one
% another (within 4% of the profile's extent xs, ys of the first of a run of them)
% joined into one text at that first mark: "S-II sep., S-IVB ign."
spanX = max(max(xs) - min(xs), eps);
spanY = max(max(ys) - min(ys), eps);
keep = true(size(x));
first = 1;
for j = 2:numel(x)
    if hypot((x(j) - x(first)) / spanX, (y(j) - y(first)) / spanY) < 0.04
        labels(first) = labels(first) + ", " + labels(j);
        keep(j) = false;
    else
        first = j;
    end
end
x = x(keep);
y = y(keep);
labels = labels(keep);
end

function earth = drawEarth(ax, C)
% the Earth to scale (the WGS-84 ellipsoid) with its coastlines, in an hgtransform:
% earth.Matrix = makehgtform("zrotate", gmst(jd)) turns it to where it is at jd
earth = hgtransform(ax);
[x, y, z] = sphere(48);
% opaque: it hides what is behind it (and a see-through one is twice as slow to draw)
themed(surf(C.RE * x, C.RE * y, C.RE * (1 - C.F) * z, Parent=earth, FaceColor=[0.55 0.7 0.9], ...
    EdgeColor="none", HandleVisibility="off"), "earth");
coast = deg2rad(satflightCoastline());
lon = coast(:, 1);
lat = coast(:, 2);
up = C.RE * 1.002;       % just above the surface, so that its facets do not hide them
themed(plot3(up * cos(lat) .* cos(lon), up * cos(lat) .* sin(lon), up * (1 - C.F) * sin(lat), ...
    Parent=earth, LineWidth=0.6, HandleVisibility="off"), "land");
end

function [night, sun] = drawNight(ax)
% the night side of a longitude-latitude map and the point where the Sun is overhead;
% moveNight places them. Call it first: the rest of the map is drawn over the night
% side. The theme colours the day side (the axes) and the night side darker, opaque:
% transparency would make each animation frame slow to draw.
themed(ax, "day");
night = patch(ax, NaN, NaN, [0.5 0.5 0.5], EdgeColor="none", DisplayName="night side");
themed(night, "night");
sun = plot(ax, NaN, NaN, "p", Color=[0.55 0.35 0], MarkerSize=13, ...
    MarkerFaceColor=[1 0.8 0.1], MarkerEdgeColor=[0.55 0.35 0], DisplayName="Sun overhead");
end

function moveNight(night, sun, jd, C)
% the night side and the subsolar point at Julian date jd, in the map's longitudes (as
% T's lon_deg: the Earth-fixed frame turned by gmst)
rs = sunPosition(jd, C);
theta = gmst(jd);
x = cos(theta) * rs(1) + sin(theta) * rs(2);
y = -sin(theta) * rs(1) + cos(theta) * rs(2);
sunLon = atan2d(y, x);
sunLat = atan2d(rs(3), hypot(x, y));
% the terminator, where the Sun is on the horizon: tan(lat) = -cos(lon - sunLon) / tan(sunLat);
% the night side holds the pole away from the Sun (the south pole when sunLat is 0)
lat0 = sunLat + (sunLat == 0) * 1e-9;
lon = (-180:2:180)';
lat = atand(-cosd(lon - sunLon) / tand(lat0));
pole = -90 * sign(lat0);
set(night, XData=[lon; 180; -180], YData=[lat; pole; pole]);
set(sun, XData=sunLon, YData=sunLat);
end

function jd = julianAt(T, t)
% the Julian date t seconds into the run of T (the export's epoch is T's t = 0)
k = find(T.t_s == T.t_s(1), 1);
jd = juliandate(T.utc(k)) + (t - T.t_s(k)) / 86400;
end

function S = eventSpans(T, events, kind)
% the spans between the opening and closing lines of the event log, per object (and
% station): passes over the stations (kind "station": AOS to LOS) or spells in the
% Earth's shadow ("eclipse": entered the shadow to back in sunlight). The simulator
% opens none at the start of a run: it first looks at the end of the first step. A
% span still open when the object stops (comes down, leaves, or the run ends) runs to
% its last sample in orbit and has stop NaT.
E = events(events.kind == kind, :);
n = height(E);
if kind == "station"
    % (cellstr: a cell of tokens per line, also for a single line)
    tok = regexp(cellstr(E.text), "^(AOS|LOS) (.+) @ (.+)$", "tokens", "once");
    bad = cellfun(@isempty, tok);
    tok = string(vertcat(tok{~bad}, cell(0, 3)));
    opens = tok(:, 1) == "AOS";
    name = tok(:, 2);
    place = tok(:, 3);
else
    opens = endsWith(E.text, " entered Earth's shadow");
    bad = ~opens & ~endsWith(E.text, " back in sunlight");
    name = erase(E.text, [" entered Earth's shadow", " back in sunlight"]);
    place = strings(n, 1);
end
if any(bad)
    error("satflight:events", "Not a %s line of the event log: %s", kind, ...
        strjoin(E.text(bad), "; "));
end
epoch = T.utc(1) - seconds(T.t_s(1));
S = table(strings(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), true(0, 1), ...
    VariableNames=["name", "place", "t0", "t1", "closed"]);
key = name + " @ " + place;
for k = unique(key, "stable")'
    j = find(key == k);
    t0 = NaN;
    for q = j'
        if opens(q) && isnan(t0)
            t0 = E.t_s(q);
        elseif ~opens(q) && ~isnan(t0)
            S(end + 1, :) = {name(q), place(q), t0, E.t_s(q), true}; %#ok<AGROW>
            t0 = NaN;
        elseif opens(q)
            error("satflight:events", "%s at %.3f s opens a span already open.", ...
                E.text(q), E.t_s(q));
        else
            error("satflight:events", "%s at %.3f s closes a span that is not open.", ...
                E.text(q), E.t_s(q));
        end
    end
    if ~isnan(t0)
        rows = T(T.name == name(j(1)) & T.status == "active", :);
        S(end + 1, :) = {name(j(1)), place(j(1)), t0, max([rows.t_s; t0]), false}; %#ok<AGROW>
    end
end
S.start = epoch + seconds(S.t0);
S.stop = epoch + seconds(S.t1);
S.stop(~S.closed) = NaT;
S.minutes = (S.t1 - S.t0) / 60;
end

function fewerEvents(T)
% says so when the run had more objects than the simulator logs eclipses and passes for
C = satflightConstants();
n = numel(unique(T.name));
if n > C.EVENT_DETAIL_LIMIT
    fprintf("This run has %d objects. The simulator logs eclipses and station passes " + ...
        "only while there are at most %d, so none after that are in the event log or " + ...
        "here.\n", n, C.EVENT_DETAIL_LIMIT);
end
end

function P = satflightPasses(T, events, stations)
%SATFLIGHTPASSES The station passes of the run, from the AOS and LOS lines of the
%   event log: the object, the station, aos and los (UTC), the minutes in view and the
%   highest elevation at the samples in between (NaN for a pass between two samples).
%   A pass under way when the object stops (re-entry, the end of the run) has los NaT
%   and counts its minutes to the object's last sample in orbit.
C = satflightConstants();
fewerEvents(T);
S = eventSpans(T, events, "station");
P = table(S.name, S.place, S.start, S.stop, S.minutes, nan(height(S), 1), ...
    VariableNames=["name", "station", "aos", "los", "minutes", "max_el_deg"]);
for k = 1:height(S)
    g = stations(find([stations.name] == S.place(k), 1));
    rows = T(T.name == S.name(k) & T.t_s >= S.t0(k) & T.t_s <= S.t1(k), :);
    if ~isempty(rows)
        P.max_el_deg(k) = max(elevationDeg(rows, g, C));
    end
end
P = sortrows(P, ["aos", "station", "name"]);     % in the order they happen
end

function V = satflightCoverage(T, passes, stations)
%SATFLIGHTCOVERAGE Each station's coverage in the run, from its passes: the number
%   of passes and of objects seen, the minutes with at least one object in view (the
%   passes of all objects merged) and their share of the run, and the longest wait
%   with none in view (minutes; from the start of the run and to its end too).
stations = reshape([stations.name], [], 1);
n = numel(stations);
t0 = min(T.utc);
t1 = max(T.utc);
total = minutes(t1 - t0);
V = table(stations, zeros(n, 1), zeros(n, 1), zeros(n, 1), zeros(n, 1), zeros(n, 1), ...
    VariableNames=["station", "passes", "objects", "in_view_min", "in_view_pct", ...
    "longest_gap_min"]);
for k = 1:n
    P = passes(passes.station == stations(k), :);
    V.passes(k) = height(P);
    V.objects(k) = numel(unique(P.name));
    % the spans in view, merged where they overlap (an open pass runs its minutes)
    span = sortrows([minutes(P.aos - t0), minutes(P.aos - t0) + P.minutes]);
    merged = zeros(0, 2);
    for j = 1:size(span, 1)
        if ~isempty(merged) && span(j, 1) <= merged(end, 2)
            merged(end, 2) = max(merged(end, 2), span(j, 2));
        else
            merged(end + 1, :) = span(j, :); %#ok<AGROW>
        end
    end
    V.in_view_min(k) = sum(merged(:, 2) - merged(:, 1));
    gaps = diff([0; reshape(merged', [], 1); total]);
    V.longest_gap_min(k) = max(gaps(1:2:end));
end
if total > 0
    V.in_view_pct = 100 * V.in_view_min / total;
end
end

function B = satflightBeta(T)
%SATFLIGHTBETA The beta angle of each object at its samples in orbit: the angle (deg)
%   between its orbit's plane and the direction of the Sun, positive when the Sun is on
%   the side the orbit turns about (its angular momentum). limit_deg is asin(RE / r)
%   for a near-circular orbit (e < 0.05; NaN otherwise): with |beta| above it the
%   object never passes through the Earth's shadow. Beta drifts as the node moves and
%   the Earth goes round the Sun.
C = satflightConstants();
rows = T(inOrbit(T, C), :);
r = [rows.x_km, rows.y_km, rows.z_km];
h = cross(r, [rows.vx_kms, rows.vy_kms, rows.vz_kms], 2);
[when, ~, at] = unique(rows.t_s);     % the Sun once per sample time
jd = juliandate(rows.utc(1)) + (when - rows.t_s(1)) / 86400;
sun = cell2mat(arrayfun(@(j) sunPosition(j, C), jd, UniformOutput=false));
sun = sun(at, :);
beta = asind(sum(h .* sun, 2) ./ (vecnorm(h, 2, 2) .* vecnorm(sun, 2, 2)));
limit = asind(C.RE ./ vecnorm(r, 2, 2));
limit(rows.e >= 0.05) = NaN;          % it holds for a circular orbit only
B = table(rows.name, rows.t_s, rows.utc, beta, limit, ...
    VariableNames=["name", "t_s", "utc", "beta_deg", "limit_deg"]);
end

function fig = satflightPlotBeta(B, names)
%SATFLIGHTPLOTBETA The beta angles of satflightBeta for the objects in names against
%   time, each with its limit (dashed, both signs): beyond it a circular orbit is in
%   sunlight all the way round.
fig = figure(Name="Beta angle");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
for k = 1:numel(names)
    rows = B(B.name == names(k), :);
    if isempty(rows)
        continue
    end
    t = rows.t_s / 3600;
    plot(ax, t, rows.beta_deg, Color=lineColor(k), LineWidth=1.2, DisplayName=names(k));
    plot(ax, [t; NaN; t], [rows.limit_deg; NaN; -rows.limit_deg], "--", ...
        Color=lineColor(k), LineWidth=0.6, HandleVisibility="off");
end
themed(yline(ax, 0, HandleVisibility="off"), "muted");
ylim(ax, [-90 90]);
yticks(ax, -90:30:90);
grid(ax, "on");
xlabel(ax, "Time since epoch (h)");
ylabel(ax, "Beta angle (deg)");
title(ax, "Sun angle to each orbit's plane", "dashed: beyond it a near-circular " + ...
    "orbit never enters the Earth's shadow");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function [el, az] = elevationDeg(rows, g, C)
% elevation (deg) of the samples rows of T above station g, as the simulator measures
% it for AOS and LOS: over the plane at right angles to the station's geodetic up; and
% azimuth (deg, clockwise from north)
theta = gmst(juliandate(rows.utc));
c = cos(theta);
sn = sin(theta);
ecef = [c .* rows.x_km + sn .* rows.y_km, -sn .* rows.x_km + c .* rows.y_km, rows.z_km];
lat = deg2rad(g.lat);
lon = deg2rad(g.lon);
d = ecef - geodeticToEcef(lat, lon, g.alt, C);
up = [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)];
el = asind(min(max((d * up.') ./ vecnorm(d, 2, 2), -1), 1));
east = [-sin(lon), cos(lon), 0];
north = [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)];
az = mod(atan2d(d * east.', d * north.'), 360);
end

function fig = satflightPlotSky(T, events, stations, names)
%SATFLIGHTPLOTSKY Each station's sky: the path of every pass of the objects in names
%   across it, azimuth clockwise from north and elevation from the horizon (the rim)
%   to overhead (the centre), at the run's samples; o marks the first sample of a pass,
%   the dashed circle the station's elevation mask. A pass between two samples has no
%   path. No figure when no pass has one.
arguments
    T table
    events table
    stations struct
    names = string.empty
end
C = satflightConstants();
names = plotNames(T, names);
S = eventSpans(T, events, "station");
S = S(ismember(S.name, names), :);
paths = {};
for k = 1:height(S)
    rows = T(T.name == S.name(k) & T.t_s >= S.t0(k) & T.t_s <= S.t1(k), :);
    if ~isempty(rows)
        g = stations(find([stations.name] == S.place(k), 1));
        [el, az] = elevationDeg(rows, g, C);
        paths(end + 1, :) = {S.place(k), S.name(k), az, el}; %#ok<AGROW>
    end
end
fig = gobjects(0);
if isempty(paths)
    fprintf("No pass of the plotted objects spans a sample: no sky plot.\n");
    return
end
places = [stations.name];
places = places(ismember(places, [paths{:, 1}]));
seen = names(ismember(names, [paths{:, 2}]));
fig = figure(Name="Sky plot");
layout = tiledlayout(fig, "flow", TileSpacing="compact");
title(layout, "Passes across each station's sky");
subtitle(layout, "elevation 0 at the rim, 90 overhead; o first sample of a pass; " + ...
    "dashed the elevation mask");
around = linspace(0, 2 * pi, 181);
for j = 1:numel(places)
    pax = polaraxes(layout);
    pax.Layout.Tile = j;
    hold(pax, "on");
    set(pax, ThetaZeroLocation="top", ThetaDir="clockwise", RLim=[0 90], ...
        RTick=30:30:90, RTickLabel=["60", "30", "0"], RAxisLocation=112);
    g = stations(find([stations.name] == places(j), 1));
    themed(polarplot(pax, around, repmat(90 - g.minEl, size(around)), "--", ...
        HandleVisibility="off"), "muted");
    for q = reshape(find([paths{:, 1}] == places(j)), 1, [])
        color = lineColor(find(names == paths{q, 2}, 1));
        theta = deg2rad(paths{q, 3});
        r = 90 - paths{q, 4};
        polarplot(pax, theta, r, Color=color, LineWidth=1.2, HandleVisibility="off");
        polarplot(pax, theta(1), r(1), "o", Color=color, MarkerFaceColor=color, ...
            MarkerSize=4, HandleVisibility="off");
    end
    title(pax, places(j), Interpreter="none");
end
% the legend: one entry per object with a path, in the last station's axes
for nm = seen
    polarplot(pax, NaN, NaN, Color=lineColor(find(names == nm, 1)), LineWidth=1.2, ...
        DisplayName=nm);
end
themed(polarplot(pax, NaN, NaN, "--", DisplayName="elevation mask"), "muted");
lg = legend(pax, Interpreter="none");
lg.Layout.Tile = "east";
themeFigure(fig);
end

function ok = inOrbit(rows, C)
% the samples of rows of T on an orbit that clears the Earth: bound, with its perigee
% above the equator's radius. Not on a pad, on the way up, or on a path down.
ok = rows.status == "active" & rows.e < 1 & rows.a_km .* (1 - rows.e) > C.RE;
end

function ok = nodeDefined(rows, C)
% the samples of rows of T on an orbit out of the equator, whose node is defined
ok = rows.i_deg >= C.I_DEFINED & rows.i_deg <= 180 - C.I_DEFINED;
end

function D = satflightDrift(T, events, sim)
%SATFLIGHTDRIFT How each orbit changes in the run after the object's last burn, from
%   the samples while it is in orbit (an orbit that clears the Earth), fitted with a
%   straight line over at least two orbits: from (UTC) and over how many days, the
%   period and the mean perigee and apogee altitudes; the change of the mean semi-major
%   axis per day (drag lowers it; J2's short-period swing taken out when J2 is on), and
%   the drift of the node and of the argument of perigee per day measured beside what
%   the J2 secular theory gives for that orbit (0 without J2). A drift is NaN where it
%   is not defined: the node of an orbit in the equator, the perigee of a circular one.
%   J2 counts only where it moves the objects: under cowell with J2 on, and always under
%   j2mean; the kepler propagator ignores it even when the scenario turns it on
C = satflightConstants();
j2 = sim.propagator == "j2mean" || (sim.propagator == "cowell" && sim.forces.j2);
M = satflightManeuvers(events);
epoch = T.utc(1) - seconds(T.t_s(1));
names = unique(T.name, "stable");
n = numel(names);
D = table(names, NaT(n, 1, TimeZone="UTC"), nan(n, 1), nan(n, 1), nan(n, 1), nan(n, 1), ...
    nan(n, 1), nan(n, 1), nan(n, 1), nan(n, 1), nan(n, 1), VariableNames=["name", "from", ...
    "days", "period_min", "perigee_km", "apogee_km", "da_km_day", "raan_deg_day", ...
    "raan_j2_deg_day", "argp_deg_day", "argp_j2_deg_day"]);
fitted = false(n, 1);
day = 86400;
for k = 1:n
    rows = T(T.name == names(k), :);
    burns = M.utc(M.name == names(k));
    if ~isempty(burns)
        rows = rows(rows.utc > max(burns), :);
    end
    % a finite burn still going at the end of the run: the stretch before it
    lit = events.utc(events.kind == "maneuver" & startsWith(events.text, names(k) + ...
        ": ignition - "));
    if ~isempty(lit) && (isempty(burns) || max(lit) > max(burns))
        rows = rows(rows.utc < max(lit), :);
    end
    rows = rows(inOrbit(rows, C), :);
    if height(rows) < 5
        continue
    end
    t = rows.t_s;
    a = rows.a_km;
    e = rows.e;
    i = deg2rad(rows.i_deg);
    if j2
        a = osculatingToMeanA(a, e, i, deg2rad(rows.argp_deg), deg2rad(rows.nu_deg), C);
    end
    period = 2 * pi * sqrt(mean(a) ^ 3 / C.MU);
    if t(end) - t(1) < 2 * period
        continue
    end
    fitted(k) = true;
    slope = @(y) day * ([t - t(1), ones(size(t))] \ y)';
    s = slope(a);
    D.da_km_day(k) = s(1);
    D.from(k) = epoch + seconds(t(1));
    D.days(k) = (t(end) - t(1)) / day;
    D.period_min(k) = period / 60;
    D.perigee_km(k) = mean(a .* (1 - e)) - C.RE;
    D.apogee_km(k) = mean(a .* (1 + e)) - C.RE;
    [raanJ2, argpJ2] = j2SecularRates(mean(a), mean(e), mean(i), C);
    if ~j2
        [raanJ2, argpJ2] = deal(0);
    end
    if all(nodeDefined(rows, C))
        s = slope(rad2deg(unwrap(deg2rad(rows.raan_deg))));
        D.raan_deg_day(k) = s(1);
        D.raan_j2_deg_day(k) = rad2deg(raanJ2) * day;
    end
    if all(e >= C.E_DEFINED & nodeDefined(rows, C))
        s = slope(rad2deg(unwrap(deg2rad(rows.argp_deg))));
        D.argp_deg_day(k) = s(1);
        D.argp_j2_deg_day(k) = rad2deg(argpJ2) * day;
    end
end
if ~all(fitted)
    fprintf("Not in orbit for two orbits after its last burn, so left out: %s.\n", ...
        strjoin(names(~fitted), ", "));
end
D = D(fitted, :);
end

function S = satflightEclipses(T, events)
%SATFLIGHTECLIPSES Each object's time in the Earth's shadow, from the event log: the
%   number of eclipses, the minutes in shadow in all and the longest, and their share
%   of its time in the run: from its first sample to its last in orbit, or from the
%   start of its first eclipse and to the end of its last when those lie outside (a
%   spent stage appears at the first sample after its separation, and comes down
%   between two).
fewerEvents(T);
spans = eventSpans(T, events, "eclipse");
names = unique(T.name, "stable");
S = table(names, zeros(size(names)), zeros(size(names)), zeros(size(names)), ...
    zeros(size(names)), VariableNames=["name", "eclipses", "shadow_min", "longest_min", ...
    "shadow_pct"]);
for k = 1:numel(names)
    mine = spans.minutes(spans.name == names(k));
    first = min([T.t_s(T.name == names(k)); spans.t0(spans.name == names(k))]);
    last = max([T.t_s(T.name == names(k) & T.status == "active"); ...
        spans.t1(spans.name == names(k))]);
    S.eclipses(k) = numel(mine);
    S.shadow_min(k) = sum(mine);
    S.longest_min(k) = max([mine; 0]);
    if last > first
        S.shadow_pct(k) = 100 * S.shadow_min(k) * 60 / (last - first);
    end
end
end

function M = satflightManeuvers(events)
%SATFLIGHTMANEUVERS The maneuvers carried out in the run, from the event log: the
%   object, when (UTC), what (the maneuver's label, or "finite burn" at a finite
%   burn's burnout) and the delta-v (m/s). Skipped and cancelled ones are not in it
%   (they are in the event log, as warnings).
E = events(events.kind == "maneuver", :);
% the lines that carry a delta-v: name, label (the first) and delta-v; the others
% (scheduled, ignition) carry none
executed = "^(.+): executed (.+) - dV ([\d.]+) m/s$";
finite = ["^(.+): burnout - dV ([\d.]+) m/s, mass [\d.]+ kg$", "finite burn"
    "^(.+): finite burn applied impulsively \(([\d.]+) m/s\) - switch to Cowell to " + ...
    "integrate thrust$", "finite burn, applied at once"];
other = "^Scheduled .+ for .+|^.+: ignition - [\d.]+ N for [\d.]+ s$";
n = height(E);
[name, what] = deal(strings(n, 1));
dv = nan(n, 1);
for k = 1:n
    entry = E.text(k);
    tok = regexp(entry, executed, "tokens", "once");
    if ~isempty(tok)
        [name(k), what(k), dv(k)] = deal(tok(1), tok(2), double(tok(3)));
        continue
    end
    for j = 1:size(finite, 1)
        tok = regexp(entry, finite(j, 1), "tokens", "once");
        if ~isempty(tok)
            [name(k), what(k), dv(k)] = deal(tok(1), finite(j, 2), double(tok(2)));
            break
        end
    end
    if isnan(dv(k)) && isempty(regexp(entry, other, "once"))
        error("satflight:events", "Not a maneuver line of the event log: %s", entry);
    end
end
carried = ~isnan(dv);
% (row, column) indexing: a column of one row indexed by a scalar false would give 0x0,
% which groupsummary rejects as a grouping variable
M = table(name(carried, 1), E.utc(carried, 1), what(carried, 1), dv(carried, 1), ...
    VariableNames=["name", "utc", "maneuver", "dv_ms"]);
end

function A = satflightApproaches(events)
%SATFLIGHTAPPROACHES The close approaches of the run, from the event log: the two
%   objects, the time of closest approach (UTC) and the distance then (km), closest
%   first. The simulator logs a pair when it first comes within the alert distance.
E = events(events.kind == "alert" & startsWith(events.text, "Close approach: "), :);
tok = regexp(cellstr(E.text), "^Close approach: (.+) - (.+) ([\d.]+) km$", "tokens", "once");
bad = cellfun(@isempty, tok);
if any(bad)
    error("satflight:events", "Not a close approach line of the event log: %s", ...
        strjoin(E.text(bad), "; "));
end
tok = string(vertcat(tok{:}, cell(0, 3)));
A = table(tok(:, 1), tok(:, 2), E.utc, double(tok(:, 3)), ...
    VariableNames=["name", "other", "tca", "distance_km"]);
A = sortrows(A, ["distance_km", "tca"]);
if isempty(A)
    C = satflightConstants();
    fprintf("No two objects came within %g km of each other (a payload and the stages " + ...
        "that carried it are not compared).\n", C.CONJUNCTION_KM);
end
end

function fig = satflightPlotTimeline(T, events, names)
%SATFLIGHTPLOTTIMELINE When each object is in the Earth's shadow (grey) and in view of
%   each station (a colour per station), against time.
arguments
    T table
    events table
    names = string.empty
end
names = plotNames(T, names);
shadow = eventSpans(T, events, "eclipse");
shadow = shadow(ismember(shadow.name, names), :);
passes = eventSpans(T, events, "station");
passes = passes(ismember(passes.name, names), :);
stations = unique(passes.place, "stable")';
fig = figure(Name="Timeline");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
box(ax, "on");
% each object's time in the run (in orbit or flying), under its shadow and passes
life = table(names', zeros(numel(names), 1), zeros(numel(names), 1), ...
    VariableNames=["name", "t0", "t1"]);
for k = 1:numel(names)
    t = T.t_s(T.name == names(k) & T.status == "active");
    life{k, ["t0", "t1"]} = [min([t; NaN]), max([t; NaN])];
end
themed(bars(ax, life, names, 0, [0.6 0.6 0.6], 1.5, "in the run"), "life");
themed(bars(ax, shadow, names, 0, [0.35 0.35 0.35], 9, "Earth's shadow"), "shadow");
% passes as thinner bars over the shadow, the stations side by side in each row
offsets = zeros(size(stations));
if numel(stations) > 1
    offsets = linspace(-0.2, 0.2, numel(stations));
end
for j = 1:numel(stations)
    bars(ax, passes(passes.place == stations(j), :), names, offsets(j), lineColor(j), 4, ...
        "in view of " + stations(j));
end
if max(T.t_s) > min(T.t_s)
    xlim(ax, [min(T.t_s), max(T.t_s)] / 3600);
end
ylim(ax, [0.4, numel(names) + 0.6]);
yticks(ax, 1:numel(names));
yticklabels(ax, names);
ax.TickLabelInterpreter = "none";
set(ax, YDir="reverse", XGrid="on");     % no row lines: they would look like time in the run
xlabel(ax, "Time since epoch (h)");
title(ax, "Eclipses and station passes");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function h = bars(ax, S, names, offset, color, width, label)
% the spans S as horizontal bars, one row per object of names; one legend entry
% (none when there are no spans)
row = arrayfun(@(nm) find(names == nm), S.name) + offset;
x = [S.t0, S.t1, nan(height(S), 1)]' / 3600;
y = [row, row, nan(height(S), 1)]';
shown = "off";
if height(S) > 0
    shown = "on";
end
h = plot(ax, [NaN; x(:)], [NaN; y(:)], Color=color, LineWidth=width, DisplayName=label, ...
    HandleVisibility=shown);
end

function fig = satflightAnimate(T, events, stations, names, opts)
%SATFLIGHTANIMATE Plays the run back: the ground track and the orbits about the
%   turning Earth over each object's last orbit (one period back, all of an escape or
%   suborbital path) up to each moment, each object where it is then (o in orbit,
%   x come down or gone), the night side of the Earth shaded on the map, and the
%   latest lines of the event log underneath.
%   Options: Frames - at most this many, spread evenly over the samples (240);
%   Seconds - how long it plays in a figure window (20). In the Live Editor every
%   frame is drawn and the output becomes an animation with its own playback
%   controls. A batch run (matlab -batch) draws only the last frame. File - also
%   write it, every frame, to this animated GIF (.gif) or video (.mp4), to play in
%   Seconds.
arguments
    T table
    events table
    stations = struct("name", {}, "lat", {}, "lon", {})
    names = string.empty
    opts.Frames (1,1) double {mustBeInteger, mustBePositive} = 240
    opts.Seconds (1,1) double {mustBeNonnegative, mustBeFinite} = 20
    opts.File (1,1) string = ""
end
C = satflightConstants();
names = plotNames(T, names);
times = unique(T.t_s);
frames = times(unique(round(linspace(1, numel(times), min(opts.Frames, numel(times))))));
epoch = T.utc(1) - seconds(T.t_s(1));
logged = events(events.kind ~= "info", :);

fig = figure(Name="Animation", Position=[100 100 1200 500]);
layout = tiledlayout(fig, 1, 5, TileSpacing="compact", Padding="compact");
map = nexttile(layout, [1 3]);
styleAxes(map);
hold(map, "on");
box(map, "on");
[night, sun] = drawNight(map);
coast = satflightCoastline();
themed(plot(map, coast(:, 1), coast(:, 2), LineWidth=0.5, HandleVisibility="off"), "coast");
for g = reshape(stations, 1, [])
    themed(plot(map, g.lon, g.lat, "^", MarkerSize=7, HandleVisibility="off"), "station");
end
xlim(map, [-180 180]); ylim(map, [-90 90]); daspect(map, [1 1 1]);
xticks(map, -180:60:180); yticks(map, -90:30:90); grid(map, "on");
xlabel(map, "Longitude (deg)"); ylabel(map, "Latitude (deg)");

globe = nexttile(layout, [1 2]);
hold(globe, "on");
earth = drawEarth(globe, C);
shownRows = T(ismember(T.name, names), :);
reach = max([vecnorm([shownRows.x_km, shownRows.y_km, shownRows.z_km], 2, 2); 1.2 * C.RE]);
lim = 1.05 * [-reach, reach];
axis(globe, "equal");
xlim(globe, lim); ylim(globe, lim); zlim(globe, lim);
axis(globe, "vis3d");
axis(globe, "off");
view(globe, 35, 25);

n = numel(names);
track = struct("t", cell(1, n), "lon", [], "lat", [], "rows", []);
[trail, marker, orbit, marker3] = deal(gobjects(1, n));
for k = 1:n
    rows = T(T.name == names(k), :);
    lon = rows.lon_deg;
    lat = rows.lat_deg;
    t = rows.t_s;
    for j = flip(find(abs(diff(lon)) > 180))'
        lon = [lon(1:j); NaN; lon(j + 1:end)];
        lat = [lat(1:j); NaN; lat(j + 1:end)];
        t = [t(1:j); t(j); t(j + 1:end)];
    end
    track(k) = struct("t", t, "lon", lon, "lat", lat, "rows", rows);
    trail(k) = plot(map, NaN, NaN, Color=lineColor(k), LineWidth=1.1, DisplayName=names(k));
    marker(k) = plot(map, NaN, NaN, "o", MarkerSize=7, Color=lineColor(k), ...
        MarkerFaceColor=lineColor(k), HandleVisibility="off");
    orbit(k) = plot3(globe, NaN, NaN, NaN, Color=lineColor(k), LineWidth=1.1, ...
        HandleVisibility="off");
    marker3(k) = plot3(globe, NaN, NaN, NaN, "o", MarkerSize=6, Color=lineColor(k), ...
        MarkerFaceColor=lineColor(k), HandleVisibility="off");
end
themed([marker, marker3], "rim");
lg = legend(map, Orientation="horizontal", NumColumns=min(n + 2, 7), Interpreter="none");
lg.Layout.Tile = "north";
themeFigure(fig);

delay = opts.Seconds / numel(frames);
W = frameWriter(opts.File, numel(frames) / max(opts.Seconds, eps));
% a figure window shows each frame as it is drawn (paced to play in Seconds); the Live
% Editor's figures are not on screen but it records each frame drawn, as an animation;
% a batch run (whose figures report Visible "on" all the same) has no one to show the
% frames to: only the last is drawn, by whatever draws or saves the figure next
batch = batchStartupOptionUsed;
onScreen = fig.Visible == "on" && ~batch;
everyFrame = ~batch || opts.File ~= "";
started = tic;
for f = 1:numel(frames)
    t = frames(f);
    for k = 1:n
        rows = track(k).rows;
        i = find(rows.t_s <= t, 1, "last");
        if isempty(i)
            set([trail(k), marker(k), orbit(k), marker3(k)], Visible="off");
            continue
        end
        % the trail: the last orbit (one period back), all of it on an open path
        since = -Inf;
        if rows.a_km(i) > 0
            since = rows.t_s(i) - 2 * pi * sqrt(rows.a_km(i) ^ 3 / C.MU);
        end
        m0 = find(track(k).t >= since, 1);
        m = find(track(k).t <= t, 1, "last");
        i0 = find(rows.t_s >= since, 1);
        mark = "o";
        if rows.status(i) ~= "active"
            mark = "x";
        end
        set(trail(k), XData=track(k).lon(m0:m), YData=track(k).lat(m0:m), Visible="on");
        set(marker(k), XData=rows.lon_deg(i), YData=rows.lat_deg(i), Marker=mark, Visible="on");
        set(orbit(k), XData=rows.x_km(i0:i), YData=rows.y_km(i0:i), ZData=rows.z_km(i0:i), ...
            Visible="on");
        set(marker3(k), XData=rows.x_km(i), YData=rows.y_km(i), ZData=rows.z_km(i), ...
            Marker=mark, Visible="on");
    end
    jd = julianAt(T, t);
    earth.Matrix = makehgtform("zrotate", gmst(jd));
    moveNight(night, sun, jd, C);
    title(layout, sprintf("%s UTC   (T+%s)", string(epoch + seconds(t), ...
        "yyyy-MM-dd HH:mm:ss"), formatDuration(t)));
    recent = logged(logged.t_s <= t, :);
    recent = recent(max(1, end - 2):end, :);
    caption = string(recent.utc, "HH:mm:ss") + "  " + recent.text;
    xlabel(layout, [caption; strings(3 - numel(caption), 1)], Interpreter="none", FontSize=9);
    if everyFrame
        drawnow
    end
    W = writeFrame(W, fig);
    if onScreen
        pause(max(0, f * delay - toc(started)));
    end
end
closeWriter(W, numel(frames));
end

% --- orbit transfers ------------------------------------------------------------------------

function X = satflightTransfers(burns)
%SATFLIGHTTRANSFERS One row per burn of satflightRun's burns: when, which object, the
%   delta-v in total and along the velocity (V), the orbit normal (N) and the binormal
%   (B) at ignition (m/s), the orbit just before and just after it (perigee and apogee
%   altitude, km; inclination, deg; period, min) and the propellant it used (kg). The
%   orbits come from the exact states at the burn, not from the run's samples.
C = satflightConstants();
before = orbitSummary(burns.y0, C);
after = orbitSummary(burns.y1, C);
X = table(burns.name, burns.label, burns.utc, burns.t1_s - burns.t0_s, burns.dv_ms, ...
    burns.dvV_ms, burns.dvN_ms, burns.dvB_ms, before.perigee_km, after.perigee_km, ...
    before.apogee_km, after.apogee_km, before.i_deg, after.i_deg, before.period_min, ...
    after.period_min, burns.mass0_kg - burns.mass1_kg, VariableNames=["name", "label", ...
    "utc", "burn_s", "dv_ms", "dvV_ms", "dvN_ms", "dvB_ms", "perigee0_km", "perigee1_km", ...
    "apogee0_km", "apogee1_km", "i0_deg", "i1_deg", "period0_min", "period1_min", ...
    "propellant_kg"]);
if isempty(X)
    fprintf("No burns were made in this run.\n");
end
end

function S = orbitSummary(Y, C)
% perigee and apogee altitude (km, above the equator's radius), inclination (deg) and
% period (min) of the states Y (one per row); NaN for the apogee and period of an
% open orbit
n = size(Y, 1);
S = table(NaN(n, 1), NaN(n, 1), NaN(n, 1), NaN(n, 1), VariableNames=["perigee_km", ...
    "apogee_km", "i_deg", "period_min"]);
ok = all(isfinite(Y), 2);
if ~any(ok)
    return
end
el = rv2coe(Y(ok, 1:3), Y(ok, 4:6), C.MU);
p = el.a .* (1 - el.e .* el.e);
S.perigee_km(ok) = p ./ (1 + el.e) - C.RE;
bound = el.e < 1;
apo = NaN(size(p));
apo(bound) = el.a(bound) .* (1 + el.e(bound)) - C.RE;
S.apogee_km(ok) = apo;
S.i_deg(ok) = rad2deg(el.i);
period = NaN(size(p));
period(bound) = 2 * pi * sqrt(el.a(bound) .^ 3 / C.MU) / 60;
S.period_min(ok) = period;
end

function P = orbitCurve(y, C, n)
% n points (n x 3, km) around the two-body orbit through state y: all of it when it is
% closed, the part out to 4 times the radius at y when it is open
r = y(1:3);
v = y(4:6);
h = cross(r, v);
ev = cross(v, h) / C.MU - r / norm(r);
e = norm(ev);
p = dot(h, h) / C.MU;
w = h / norm(h);
if e > 1e-9
    P1 = ev / e;
else
    P1 = r / norm(r);
end
Q1 = cross(w, P1);
if e < 1
    nu = linspace(0, 2 * pi, n)';
else
    reach = acos(min(max((p / (4 * norm(r)) - 1) / e, -1), 1));
    nu = linspace(-reach, reach, n)';
end
rad = p ./ (1 + e * cos(nu));
P = rad .* (cos(nu) * P1 + sin(nu) * Q1);
end

function fig = satflightPlotTransfer(T, burns, names)
%SATFLIGHTPLOTTRANSFER For each object in names that burns: its orbit before the first
%   burn (dashed), the path it flies between burns, its orbit after the last one, and
%   each burn where it fires (numbered, with its delta-v and an arrow along it); in 3-D
%   about the Earth, and seen along the line where the first and last orbits' planes
%   cross, so that a change of plane shows as the angle between them. Burns are
%   numbered by their row in burns (and in satflightTransfers' table).
arguments
    T table
    burns table
    names = string.empty
end
C = satflightConstants();
names = plotNames(T, names);
fig = gobjects(0);
for nm = names
    ids = find(burns.name == nm & isfinite(burns.t1_s));
    B = burns(ids, :);
    if isempty(B)
        continue
    end
    rows = T(T.name == nm, :);
    start = orbitCurve(B.y0(1, :), C, 361);
    final = orbitCurve(B.y1(end, :), C, 361);
    legs = cell(height(B) - 1, 1);
    for k = 1:height(B) - 1
        between = rows(rows.t_s > B.t1_s(k) & rows.t_s < B.t0_s(k + 1), :);
        legs{k} = [B.y1(k, 1:3); [between.x_km, between.y_km, between.z_km]; B.y0(k + 1, 1:3)];
    end
    fired = cell(height(B), 1);       % the path flown under thrust (finite burns)
    for k = 1:height(B)
        during = rows(rows.t_s > B.t0_s(k) & rows.t_s < B.t1_s(k), :);
        fired{k} = [B.y0(k, 1:3); [during.x_km, during.y_km, during.z_km]; B.y1(k, 1:3)];
    end
    reach = max(vecnorm([start; final; vertcat(legs{:})], 2, 2));
    before = orbitSummary(B.y0(1, :), C);
    after = orbitSummary(B.y1(end, :), C);

    fig(end + 1) = figure(Name="Transfer: " + nm, Position=[80 80 1250 560]); %#ok<AGROW>
    layout = tiledlayout(fig(end), 1, 3, TileSpacing="compact", Padding="compact");
    title(layout, sprintf("%s: %d burn%s, %.1f m/s in all", nm, height(B), ...
        plural(height(B)), sum(B.dv_ms)), Interpreter="none");
    subtitle(layout, sprintf("from %s to %s", orbitText(before), orbitText(after)), ...
        Interpreter="none");

    ax = nexttile(layout, [1 2]);
    hold(ax, "on");
    earth = drawEarth(ax, C);
    earth.Matrix = makehgtform("zrotate", gmst(julianAt(T, B.t0_s(1))));
    themed(plot3(ax, start(:, 1), start(:, 2), start(:, 3), "--", LineWidth=1.1, ...
        DisplayName="orbit before"), "muted");
    plot3(ax, final(:, 1), final(:, 2), final(:, 3), Color=lineColor(2), LineWidth=1.8, ...
        DisplayName="orbit after");
    for k = 1:numel(legs)
        plot3(ax, legs{k}(:, 1), legs{k}(:, 2), legs{k}(:, 3), Color=lineColor(1), ...
            LineWidth=1.6, DisplayName="transfer", HandleVisibility=onlyFirst(k));
    end
    burnMarks(ax, B, ids, fired, reach);
    axis(ax, "equal"); grid(ax, "on"); view(ax, 35, 25);
    xlabel(ax, "x (km)"); ylabel(ax, "y (km)"); zlabel(ax, "z (km)");
    title(ax, "In 3-D (ECI)");
    legend(ax, Location="southoutside", Orientation="horizontal", Interpreter="none");

    % the second view: along the line where the planes before and after cross
    n0 = cross(B.y0(1, 1:3), B.y0(1, 4:6));
    n1 = cross(B.y1(end, 1:3), B.y1(end, 4:6));
    n0 = n0 / norm(n0);
    n1 = n1 / norm(n1);
    nodes = cross(n0, n1);
    if norm(nodes) < 1e-6        % one plane: look along the first burn's position
        nodes = B.y0(1, 1:3);
    end
    ex = nodes / norm(nodes);
    ez = n1;
    ey = cross(ez, ex);
    side = @(P) [P * ey.', P * ez.'];
    ax2 = nexttile(layout);
    hold(ax2, "on");
    around = linspace(0, 2 * pi, 121)';
    themed(fill(ax2, C.RE * cos(around), C.RE * sin(around), [0.5 0.5 0.5], ...
        EdgeColor="none", HandleVisibility="off"), "earth");
    q = side(start);
    themed(plot(ax2, q(:, 1), q(:, 2), "--", LineWidth=1.1, HandleVisibility="off"), "muted");
    q = side(final);
    plot(ax2, q(:, 1), q(:, 2), Color=lineColor(2), LineWidth=1.8, HandleVisibility="off");
    for k = 1:numel(legs)
        q = side(legs{k});
        plot(ax2, q(:, 1), q(:, 2), Color=lineColor(1), LineWidth=1.6, HandleVisibility="off");
    end
    q = side(B.y0(:, 1:3));
    themed(plot(ax2, q(:, 1), q(:, 2), "p", MarkerSize=13, HandleVisibility="off"), "burn");
    axis(ax2, "equal"); grid(ax2, "on"); box(ax2, "on");
    xlabel(ax2, "across the line of nodes (km)"); ylabel(ax2, "out of the final plane (km)");
    tilt = acosd(min(max(dot(n0, n1), -1), 1));
    title(ax2, sprintf("Edge-on: the planes %.2f%s apart", tilt, char(176)));
    themeFigure(fig(end));
end
if isempty(fig)
    fprintf("None of the plotted objects burned: no transfer plot.\n");
end
end

function burnMarks(ax, B, ids, fired, reach)
% the burns of table B in 3-D axes ax: the path flown under thrust (finite burns), a
% star where each fires, an arrow along its delta-v and its number (ids: its row in
% the burns table) and size
scale = 0.18 * reach;
for k = 1:height(B)
    if size(fired{k}, 1) > 2
        themed(plot3(ax, fired{k}(:, 1), fired{k}(:, 2), fired{k}(:, 3), LineWidth=4, ...
            HandleVisibility="off"), "burn");
    end
    p = B.y0(k, 1:3);
    d = B.y1(k, 4:6) - B.y0(k, 4:6);
    if B.t1_s(k) > B.t0_s(k) || norm(d) == 0
        d = [B.dvV_ms(k), B.dvN_ms(k), B.dvB_ms(k)] * vnbBasis(p, B.y0(k, 4:6));
    end
    d = scale * d / max(norm(d), eps);
    themed(plot3(ax, p(1), p(2), p(3), "p", MarkerSize=15, DisplayName="burn", ...
        HandleVisibility=onlyFirst(k)), "burn");
    themed(quiver3(ax, p(1), p(2), p(3), d(1), d(2), d(3), 0, LineWidth=2, ...
        MaxHeadSize=0.6, HandleVisibility="off"), "burn");
    themed(text(ax, p(1) + d(1), p(2) + d(2), p(3) + d(3), sprintf("  #%d  %.1f m/s", ...
        ids(k), B.dv_ms(k)), FontSize=11, FontWeight="bold", Interpreter="none"), "label");
end
end

function v = onlyFirst(k)
% HandleVisibility for the k-th of a set of lines sharing one legend entry
v = "off";
if k == 1
    v = "on";
end
end

function s = plural(n)
s = "";
if n ~= 1
    s = "s";
end
end

function text = orbitText(S)
% "185 x 35,786 km, 28.5 deg" for one row of orbitSummary
if isnan(S.apogee_km)
    text = sprintf("an open path (perigee %s km), %.2f deg", commaNum(S.perigee_km), S.i_deg);
else
    text = sprintf("%s x %s km, %.2f deg", commaNum(S.perigee_km), commaNum(S.apogee_km), ...
        S.i_deg);
end
end

function fig = satflightPlotDeltaV(burns, tEnd)
%SATFLIGHTPLOTDELTAV The delta-v budget: each burn's delta-v along the velocity (V),
%   the orbit normal (N) and the binormal (B), with its total (numbered by its row in
%   burns); each object's delta-v added up over the run (to tEnd, s); and its mass,
%   stepping down as the burns use propellant.
arguments
    burns table
    tEnd (1,1) double = NaN
end
ids = find(isfinite(burns.t1_s));
B = burns(ids, :);
fig = gobjects(0);
if isempty(B)
    fprintf("No burns were made: no delta-v budget plot.\n");
    return
end
fig = figure(Name="Delta-v budget", Position=[90 90 1250 500]);
layout = tiledlayout(fig, 1, 3, TileSpacing="compact", Padding="compact");
title(layout, sprintf("Delta-v budget: %.1f m/s in %d burn%s", sum(B.dv_ms), height(B), ...
    plural(height(B))));
ax = nexttile(layout);
hold(ax, "on");
parts = [B.dvV_ms, B.dvN_ms, B.dvB_ms];
bar(ax, 1:height(B), parts, "grouped");
themed(plot(ax, 1:height(B), B.dv_ms, "_", MarkerSize=22, LineWidth=3, ...
    DisplayName="total"), "label");
legend(ax, ["V (along the velocity)", "N (orbit normal)", "B (binormal)", "total"], ...
    Location="southoutside");
xticks(ax, 1:height(B));
xticklabels(ax, compose("#%d", ids));
ax.TickLabelInterpreter = "none";
grid(ax, "on");
xlabel(ax, "Burn (row of transfers)");
ylabel(ax, "Delta-v (m/s)");
title(ax, "Each burn, by direction");
ax.YLim(2) = 1.2 * max([B.dv_ms; parts(:)]);
lift = 0.03 * diff(ax.YLim);
for k = 1:height(B)
    themed(text(ax, k, B.dv_ms(k) + lift, sprintf("%.1f\n%s", B.dv_ms(k), B.name(k)), ...
        HorizontalAlignment="center", VerticalAlignment="bottom", FontSize=8, ...
        Interpreter="none"), "label");
end

ax = nexttile(layout);
hold(ax, "on");
ax2 = nexttile(layout);
hold(ax2, "on");
who = unique(B.name, "stable")';
if ~isfinite(tEnd)
    tEnd = 1.05 * max(B.t1_s) + 1;
end
for k = 1:numel(who)
    b = B(B.name == who(k), :);
    t = [0; reshape([b.t0_s, b.t1_s]', [], 1); tEnd] / 3600;
    used = [0; reshape([zeros(height(b), 1), b.dv_ms]', [], 1)];
    used = cumsum(used);
    used(end + 1) = used(end); %#ok<AGROW>
    plot(ax, t, used, Color=lineColor(k), LineWidth=1.6, DisplayName=who(k));
    mass = [b.mass0_kg(1); reshape([b.mass0_kg, b.mass1_kg]', [], 1); b.mass1_kg(end)];
    plot(ax2, t, mass, Color=lineColor(k), LineWidth=1.6, DisplayName=who(k));
    themed(text(ax2, t(end), mass(end), sprintf(" %.0f kg used", b.mass0_kg(1) - ...
        b.mass1_kg(end)), FontSize=8, VerticalAlignment="bottom", HorizontalAlignment="right"), ...
        "label");
end
grid(ax, "on");
xlabel(ax, "Time since epoch (h)");
ylabel(ax, "Delta-v so far (m/s)");
title(ax, "Added up");
legend(ax, Location="southoutside", Interpreter="none");
grid(ax2, "on");
xlabel(ax2, "Time since epoch (h)");
ylabel(ax2, "Mass (kg)");
title(ax2, "Mass (rocket equation)");
themeFigure(fig);
end

function fig = satflightPlotBurnWindow(T, burns, names)
%SATFLIGHTPLOTBURNWINDOW The orbit of each object in names that burns, over the part of
%   the run that holds the burns: perigee and apogee altitude, inclination and period,
%   from the run's samples and the exact states just before and after each burn (so
%   each impulse is a clean step), the burns marked.
arguments
    T table
    burns table
    names = string.empty
end
C = satflightConstants();
names = plotNames(T, names);
ids = find(isfinite(burns.t1_s) & ismember(burns.name, names));
B = burns(ids, :);
fig = gobjects(0);
if isempty(B)
    fprintf("None of the plotted objects burned: no burn window plot.\n");
    return
end
span = max(B.t1_s) - min(B.t0_s);
pad = max(0.15 * span, 1800);
t0 = max(min(B.t0_s) - pad, min(T.t_s));
t1 = min(max(B.t1_s) + pad, max(T.t_s));
fig = figure(Name="Burn window", Position=[100 100 1100 640]);
layout = tiledlayout(fig, 2, 2, TileSpacing="compact", Padding="compact");
title(layout, sprintf("The orbit around the burns, %s to %s UTC", ...
    string(julianUtc(T, t0), "yyyy-MM-dd HH:mm"), string(julianUtc(T, t1), "yyyy-MM-dd HH:mm")));
subtitle(layout, "osculating elements; dotted lines: the burns, numbered by their row " + ...
    "of transfers");
panels = ["perigee_km", "Perigee altitude (km)"; "apogee_km", "Apogee altitude (km)"; ...
    "i_deg", "Inclination (deg)"; "period_min", "Period (min)"];
axs = gobjects(1, 4);
for p = 1:4
    axs(p) = nexttile(layout);
    hold(axs(p), "on");
    grid(axs(p), "on");
    ylabel(axs(p), panels(p, 2));
    xlabel(axs(p), "Time since epoch (h)");
end
who = unique(B.name, "stable")';
for k = 1:numel(who)
    rows = T(T.name == who(k) & T.t_s >= t0 & T.t_s <= t1, :);
    rows = rows(inOrbit(rows, C), :);
    b = B(B.name == who(k), :);
    t = [rows.t_s; b.t0_s; b.t1_s];   % (rows of burns, before and after each)
    Y = [rows.x_km, rows.y_km, rows.z_km, rows.vx_kms, rows.vy_kms, rows.vz_kms; b.y0; b.y1];
    % at a burn's instant, the state before comes first, then the state after
    order = [zeros(height(rows), 1); -ones(height(b), 1); ones(height(b), 1)];
    [~, sorted] = sortrows([t, order]);
    S = orbitSummary(Y(sorted, :), C);
    tt = t(sorted) / 3600;
    c = find(names == who(k), 1);
    for p = 1:4
        plot(axs(p), tt, S.(panels(p, 1)), Color=lineColor(c), LineWidth=1.5, ...
            DisplayName=who(k));
    end
end
for p = 1:4
    for k = 1:height(B)
        themed(xline(axs(p), B.t0_s(k) / 3600, ":", HandleVisibility="off"), "muted");
    end
    xlim(axs(p), [t0, t1] / 3600);
end
top = axs(1).YLim(2);
for k = 1:height(B)
    themed(text(axs(1), B.t0_s(k) / 3600, top, sprintf(" #%d", ids(k)), VerticalAlignment="top", ...
        FontSize=9, FontWeight="bold"), "label");
end
lg = legend(axs(1), Interpreter="none");
lg.Layout.Tile = "east";
themeFigure(fig);
end

function u = julianUtc(T, t)
% the UTC time t seconds into the run of T
k = find(T.t_s == T.t_s(1), 1);
u = T.utc(k) + seconds(t - T.t_s(k));
end

% --- physics diagnostics --------------------------------------------------------------------

function [D, K] = satflightDiagnostics(T, sim, burns, objects)
%SATFLIGHTDIAGNOSTICS The physics of the run, per object and sample while it is in
%   orbit: specific orbital energy (two-body, and with the potential of the gravity
%   terms the run applies), angular momentum (its size and its z part), speed,
%   distance from the centre and period, the Jacobi constant (energy less the Earth's
%   spin times h_z), and the size of the acceleration from each force the run applies
%   (D). And for each coast, the stretch between burns in orbit, how far the quantity
%   that physics keeps constant there drifted (K): energy for gravity alone, the Jacobi
%   constant with C22 or the EGM2008 field (which turn with the Earth); with drag on, energy
%   falls. A conserved quantity that drifts by more than the integrator's tolerance
%   would point to a fault.
C = satflightConstants();
F = sim.forces;
if msisDrag(F)
    F = msisSetup(F, sim.dataFolder, C);
end
terms = ["j2", "j3", "j4", "c22", "drag", "sun", "moon", "srp", "egm"];
applied = false(1, 9);
if sim.propagator == "cowell"
    field = F.gravityDegree > 0;      % the field stands in for J2, J3, J4 and C22
    applied = [~field & [F.j2, F.j3, F.j4, F.c22], F.drag, F.sun, F.moon, F.srp, field];
end
present = unique(T.name, "stable")';
if isfield(sim, "replayUntil")
    % a replayed object follows its recording: the forces of the run do not fly it
    names = reshape(string(sim.names), [], 1);
    present = present(~ismember(present, names(reshape(string(sim.replayUntil), [], 1) ~= "")));
end
D = cell(numel(present), 1);
for q = 1:numel(present)
    rows = T(T.name == present(q), :);
    rows = rows(inOrbit(rows, C), :);
    r = [rows.x_km, rows.y_km, rows.z_km];
    v = [rows.vx_kms, rows.vy_kms, rows.vz_kms];
    rm = vecnorm(r, 2, 2);
    speed = vecnorm(v, 2, 2);
    h = cross(r, v, 2);
    theta = gmst(juliandate(rows.utc));
    kepler = speed .^ 2 / 2 - C.MU ./ rm;
    total = speed .^ 2 / 2 + potential(r, theta, applied, C, F.field);
    period = NaN(size(rm));
    bound = kepler < 0;
    period(bound) = 2 * pi * sqrt((-C.MU ./ (2 * kepler(bound))) .^ 3 / C.MU) / 60;
    acc = NaN(height(rows), 10);
    acc(:, 1) = 1000 * C.MU ./ rm .^ 2;
    props = objects(objects.name == present(q), :);
    for j = find(applied)
        G = struct("j2", false, "j3", false, "j4", false, "c22", false, "drag", false, ...
            "densityScale", F.densityScale, "sun", false, "moon", false, "srp", false, ...
            "gravityDegree", 0, "field", F.field);
        if terms(j) == "egm"
            G.gravityDegree = F.gravityDegree;
        else
            G.(terms(j)) = true;
        end
        if terms(j) == "drag" && msisDrag(F)
            G.atmosphere = F.atmosphere;
            if isfield(F, "cdModel")
                G.cdModel = F.cdModel;
            end
            G.sw = F.sw;
            G.msis = F.msis;
        end
        cdam = zeros(height(rows), 1);
        cram = zeros(height(rows), 1);
        if ~isempty(props)
            m = massAt(burns, props, rows.t_s);
            if terms(j) == "drag"
                cdam = props.cd(1) * props.area_m2(1) ./ m;
            elseif terms(j) == "srp"
                cram = props.cr(1) * props.area_m2(1) ./ m;
            end
        end
        if any(terms(j) == ["sun", "moon", "srp"]) || msisDrag(G)
            % the Sun and Moon at each sample's own date (NRLMSIS: its air then)
            jd = juliandate(rows.utc);
            extra = zeros(height(rows), 3);
            for k = 1:height(rows)
                extra(k, :) = acceleration(r(k, :), v(k, :), cdam(k), theta(k), G, C, ...
                    jd(k), cram(k)) + C.MU * r(k, :) / rm(k) ^ 3;
            end
        else
            % (theta: the Earth's turn at each sample, for C22 and the EGM2008 field)
            extra = acceleration(r, v, cdam, theta, G, C, [], []) + C.MU * r ./ rm .^ 3;
        end
        acc(:, j + 1) = 1000 * vecnorm(extra, 2, 2);
    end
    D{q} = table(repmat(present(q), height(rows), 1), rows.t_s, kepler, total, ...
        total - C.OMEGA * h(:, 3), vecnorm(h, 2, 2), h(:, 3), speed, rm, period, acc(:, 1), ...
        acc(:, 2), acc(:, 3), acc(:, 4), acc(:, 5), acc(:, 6), acc(:, 7), acc(:, 8), ...
        acc(:, 9), acc(:, 10), VariableNames=["name", "t_s", ...
        "energy_MJkg", "energy_total_MJkg", "jacobi_MJkg", "h_km2s", "hz_km2s", "speed_kms", ...
        "r_km", "period_min", "acc_central_ms2", "acc_j2_ms2", "acc_j3_ms2", "acc_j4_ms2", ...
        "acc_c22_ms2", "acc_drag_ms2", "acc_sun_ms2", "acc_moon_ms2", "acc_srp_ms2", ...
        "acc_egm_ms2"]);
end
D = vertcat(D{:});
if isempty(D)
    D = table(strings(0, 1), zeros(0, 1), VariableNames=["name", "t_s"]);
end

% the coasts, and the quantity physics keeps there
if applied(4) || applied(9)
    kept = "jacobi_MJkg";
    what = "Jacobi constant";
else
    kept = "energy_total_MJkg";
    what = "energy";
end
K = table('Size', [0 8], 'VariableTypes', ["string", "double", "double", "double", "string", ...
    "double", "double", "string"], 'VariableNames', ["name", "from_s", "to_s", "samples", ...
    "quantity", "drift_rel", "hz_drift_rel", "expected"]);
expected = "constant";
outside = ["the Sun", "the Moon", "sunlight pressure"];
outside = outside(applied(6:8));
if ~isempty(outside)
    % bodies and light from outside the Earth's field work on the orbit
    expected = "not kept (" + strjoin(outside, ", ") + ")";
elseif applied(5)
    expected = "falls (drag)";
elseif sim.propagator == "j2mean"
    % the J2-mean propagator moves mean elements: the osculating energy of its states
    % swings with J2's short-period terms, by about 1e-3
    expected = "not kept (J2 mean elements)";
end
for q = 1:numel(present)
    d = D(D.name == present(q), :);
    if height(d) < 3
        continue
    end
    b = burns(burns.name == present(q), :);
    % a new coast after each burn and wherever the samples skip (not in orbit)
    step = median(diff(d.t_s));
    cut = [1; find(diff(d.t_s) > 1.5 * step) + 1];
    for k = 1:height(b)
        cut = [cut; find(d.t_s > b.t0_s(k), 1)]; %#ok<AGROW>
    end
    cut = unique([cut; height(d) + 1]);
    for k = 1:numel(cut) - 1
        seg = d(cut(k):cut(k + 1) - 1, :);
        if ~isempty(b)
            firing = any(seg.t_s' >= b.t0_s & seg.t_s' <= b.t1_s, 1)';
            seg = seg(~firing, :);
        end
        if height(seg) < 3
            continue
        end
        x = seg.(kept);
        hz = seg.hz_km2s;
        K(end + 1, :) = {present(q), seg.t_s(1), seg.t_s(end), height(seg), what, ...
            (max(x) - min(x)) / abs(mean(x)), (max(hz) - min(hz)) / max(abs(mean(hz)), ...
            eps), expected}; %#ok<AGROW>
    end
end
end

function U = potential(r, theta, applied, C, field)
% the specific potential energy (km^2/s^2) at positions r (one per row) of point-mass
% gravity and the terms applied (J2, J3, J4, C22, the EGM2008 field; theta the
% Earth's turn): the
% potential whose gradient is minus acceleration()'s gravity
rm = vecnorm(r, 2, 2);
s = r(:, 3) ./ rm;
U = -C.MU ./ rm;
if applied(1)
    U = U + C.MU * C.J2 * C.RE ^ 2 ./ rm .^ 3 .* (3 * s .^ 2 - 1) / 2;
end
if applied(2)
    U = U + C.MU * C.J3 * C.RE ^ 3 ./ rm .^ 4 .* (5 * s .^ 3 - 3 * s) / 2;
end
if applied(3)
    U = U + C.MU * C.J4 * C.RE ^ 4 ./ rm .^ 5 .* (35 * s .^ 4 - 30 * s .^ 2 + 3) / 8;
end
if applied(4)
    c = cos(theta);
    sn = sin(theta);
    xe = c .* r(:, 1) + sn .* r(:, 2);
    ye = -sn .* r(:, 1) + c .* r(:, 2);
    U = U - 3 * C.MU * C.RE ^ 2 * (C.C22 * (xe .^ 2 - ye .^ 2) + 2 * C.S22 * xe .* ye) ./ rm .^ 5;
end
if applied(9)
    U = U - fieldPotential(r, theta, field);
end
end

function m = massAt(burns, props, t)
% the mass (kg) of the object of props at times t: what it ends with, plus the
% propellant of each burn still to come
m = props.mass_end_kg(1) * ones(size(t));
b = burns(burns.name == props.name(1) & isfinite(burns.t1_s), :);
for k = 1:height(b)
    later = t < b.t0_s(k);
    m(later) = m(later) + b.mass0_kg(k) - b.mass1_kg(k);
end
end

function fig = satflightPlotDiagnostics(D, burns, names)
%SATFLIGHTPLOTDIAGNOSTICS The physics of satflightDiagnostics for the objects in names:
%   energy, angular momentum, speed and period against time (burns dotted), the size
%   of each force on the first of them that burns (or the first), and how far each
%   object's energy (with the gravity terms' potential) has moved from its value at
%   the start of the coast it is in.
fig = gobjects(0);
names = names(ismember(names, D.name));
if isempty(names)
    fprintf("No object was in orbit at a sample: no diagnostics plot.\n");
    return
end
fig = figure(Name="Physics diagnostics", Position=[70 70 1250 700]);
layout = tiledlayout(fig, 2, 3, TileSpacing="compact", Padding="compact");
title(layout, "Physics diagnostics");
subtitle(layout, "in orbit only; dotted lines: burns");
panels = ["energy_MJkg", "Specific energy, two-body (MJ/kg)"; ...
    "h_km2s", "Angular momentum |h| (km^2/s)"; "speed_kms", "Speed (km/s)"; ...
    "period_min", "Period (min)"];
B = burns(isfinite(burns.t1_s) & ismember(burns.name, names), :);
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    hold(ax, "on");
    for k = 1:numel(names)
        d = D(D.name == names(k), :);
        plot(ax, d.t_s / 3600, gaps(d.t_s, d.(panels(p, 1))), Color=lineColor(k), ...
            LineWidth=1.2, DisplayName=names(k));
    end
    burnLines(ax, B);
    grid(ax, "on");
    xlabel(ax, "Time since epoch (h)");
    ylabel(ax, panels(p, 2));
end
lg = legend(ax, Interpreter="none");
lg.Layout.Tile = "east";

% the forces on one object
focus = names(1);
if ~isempty(B)
    focus = B.name(1);
end
ax = nexttile(layout);
hold(ax, "on");
d = D(D.name == focus, :);
forces = ["acc_central_ms2", "central gravity"; "acc_j2_ms2", "J2"; "acc_j3_ms2", "J3"; ...
    "acc_j4_ms2", "J4"; "acc_c22_ms2", "C22"; "acc_drag_ms2", "drag"; ...
    "acc_sun_ms2", "the Sun"; "acc_moon_ms2", "the Moon"; "acc_srp_ms2", "sunlight"; ...
    "acc_egm_ms2", "EGM2008 field"];
for j = 1:size(forces, 1)
    y = d.(forces(j, 1));
    if any(isfinite(y))
        plot(ax, d.t_s / 3600, gaps(d.t_s, y), LineWidth=1.3, DisplayName=forces(j, 2));
    end
end
ax.YScale = "log";
burnLines(ax, B(B.name == focus, :));
grid(ax, "on");
xlabel(ax, "Time since epoch (h)");
ylabel(ax, "Acceleration (m/s^2)");
title(ax, "Forces on " + focus, Interpreter="none");
legend(ax, Location="best", FontSize=8);

% the drift of energy within each coast
ax = nexttile(layout);
hold(ax, "on");
for k = 1:numel(names)
    d = D(D.name == names(k), :);
    if height(d) < 2
        continue
    end
    b = B(B.name == names(k), :);
    coast = cumsum([1; diff(d.t_s) > 1.5 * median(diff(d.t_s))]);
    for j = 1:height(b)
        coast = coast + (d.t_s > b.t0_s(j));
    end
    drift = NaN(height(d), 1);
    for c = unique(coast)'
        in = coast == c;
        e = d.energy_total_MJkg(in);
        drift(in) = abs(e - e(1)) / abs(e(1));
    end
    drift(drift == 0) = NaN;
    plot(ax, d.t_s / 3600, gaps(d.t_s, drift), Color=lineColor(k), LineWidth=1.1, ...
        DisplayName=names(k));
end
ax.YScale = "log";
burnLines(ax, B);
grid(ax, "on");
xlabel(ax, "Time since epoch (h)");
ylabel(ax, "|\DeltaE / E| since the coast began");
title(ax, "Energy drift (with the gravity terms)");
themeFigure(fig);
end

function y = gaps(t, y)
% y with NaN where the samples t skip (the object was not in orbit), so the line breaks
skip = [false; diff(t) > 1.5 * median(diff(t))];
y(skip) = NaN;
end

function burnLines(ax, B)
for k = 1:height(B)
    themed(xline(ax, B.t0_s(k) / 3600, ":", HandleVisibility="off"), "muted");
end
end

% --- the 3-D animation ----------------------------------------------------------------------

function fig = satflightAnimate3D(T, events, burns, names, opts)
%SATFLIGHTANIMATE3D Plays the run back in 3-D: the orbits about the turning Earth, each
%   object with a trail over its last orbit that fades with age and its current orbit
%   dotted, a flash and an arrow where a burn fires, the latest lines of the event log
%   underneath, and the camera turning slowly. Time runs at its own pace: slower about
%   the burns and launches, faster over the coasts between them (the title gives the
%   speed). Options as satflightAnimate's: Frames (240), Seconds (20) and File (an
%   animated .gif or an .mp4 video).
arguments
    T table
    events table
    burns table
    names = string.empty
    opts.Frames (1,1) double {mustBeInteger, mustBePositive} = 240
    opts.Seconds (1,1) double {mustBeNonnegative, mustBeFinite} = 20
    opts.File (1,1) string = ""
end
C = satflightConstants();
names = plotNames(T, names);
B = burns(isfinite(burns.t1_s) & ismember(burns.name, names), :);
logged = events(events.kind ~= "info", :);
moments = [B.t0_s; B.t1_s; logged.t_s(logged.kind == "launch" | logged.kind == "maneuver")];
frames = frameTimes(min(T.t_s), max(T.t_s), opts.Frames, moments);
epoch = T.utc(1) - seconds(T.t_s(1));
pace = opts.Seconds / numel(frames);

fig = figure(Name="Animation 3-D", Position=[120 80 1000 760]);
ax = axes(fig, Position=[0.02 0.1 0.96 0.82]);
hold(ax, "on");
earth = drawEarth(ax, C);
shownRows = T(ismember(T.name, names), :);
reach = max([vecnorm([shownRows.x_km, shownRows.y_km, shownRows.z_km], 2, 2); 1.3 * C.RE]);
reach = min(reach, 12 * max([shownRows.a_km(shownRows.a_km > 0); C.RE]));
lim = 1.05 * [-reach, reach];
axis(ax, "equal");
xlim(ax, lim); ylim(ax, lim); zlim(ax, lim);
axis(ax, "vis3d");
axis(ax, "off");
n = numel(names);
[trail, marker, ghost] = deal(gobjects(1, n));
for k = 1:n
    trail(k) = patch(ax, XData=NaN, YData=NaN, ZData=NaN, FaceVertexCData=[0 0 0], ...
        FaceColor="none", EdgeColor="interp", LineWidth=1.8, UserData=lineColor(k), ...
        HandleVisibility="off");
    themed(trail(k), "fade");
    ghost(k) = plot3(ax, NaN, NaN, NaN, ":", Color=lineColor(k), LineWidth=0.8, ...
        HandleVisibility="off");
    marker(k) = plot3(ax, NaN, NaN, NaN, "o", MarkerSize=8, Color=lineColor(k), ...
        MarkerFaceColor=lineColor(k), DisplayName=names(k));
end
themed(marker, "rim");
flash = plot3(ax, NaN, NaN, NaN, "p", MarkerSize=26, DisplayName="burn");
themed(flash, "burn");
arrow = quiver3(ax, NaN, NaN, NaN, NaN, NaN, NaN, 0, LineWidth=2.5, MaxHeadSize=0.7, ...
    HandleVisibility="off");
themed(arrow, "burn");
note = text(ax, NaN, NaN, NaN, "", FontSize=11, FontWeight="bold", Interpreter="none");
themed(note, "label");
legend(ax, [marker, flash], Location="northeast", Interpreter="none");
heading = annotation(fig, "textbox", [0.02 0.93 0.96 0.06], String="", EdgeColor="none", ...
    FontSize=12, FontWeight="bold", HorizontalAlignment="center", Interpreter="none");
ticker = annotation(fig, "textbox", [0.02 0.005 0.96 0.09], String="", EdgeColor="none", ...
    FontSize=9, Interpreter="none", VerticalAlignment="bottom");
themed([heading, ticker], "label");
themeFigure(fig);

tracks = cell(1, n);
for k = 1:n
    tracks{k} = T(T.name == names(k), :);
end
W = frameWriter(opts.File, numel(frames) / max(opts.Seconds, eps));
batch = batchStartupOptionUsed;
onScreen = fig.Visible == "on" && ~batch;
everyFrame = ~batch || opts.File ~= "";
started = tic;
P = satflightPalette(string(fig.Theme.BaseColorStyle));
for f = 1:numel(frames)
    t = frames(f);
    for k = 1:n
        [y, active, rows, i] = stateAt(tracks{k}, B(B.name == names(k), :), t, C);
        if isempty(i)
            set([trail(k), marker(k), ghost(k)], Visible="off");
            continue
        end
        a = rows.a_km(i);
        since = -Inf;
        if a > 0
            since = t - 2 * pi * sqrt(a ^ 3 / C.MU);
        end
        past = rows(rows.t_s >= since & rows.t_s <= t, :);
        pts = [past.x_km, past.y_km, past.z_km; y(1:3)];
        set(trail(k), XData=[pts(:, 1); NaN], YData=[pts(:, 2); NaN], ZData=[pts(:, 3); NaN], ...
            FaceVertexCData=zeros(size(pts, 1) + 1, 3), Visible="on");
        fadeTrail(trail(k), P);
        mark = "o";
        if ~active
            mark = "x";
        end
        set(marker(k), XData=y(1), YData=y(2), ZData=y(3), Marker=mark, Visible="on");
        if active && norm(y(1:3)) > C.RE
            o = orbitCurve(y, C, 181);
            set(ghost(k), XData=o(:, 1), YData=o(:, 2), ZData=o(:, 3), Visible="on");
        else
            set(ghost(k), Visible="off");
        end
    end
    % a burn firing (or just fired) now
    near = 0.012 * (frames(end) - frames(1));
    firing = find(t >= B.t0_s - near & t <= B.t1_s + near, 1);
    if isempty(firing)
        set([flash, arrow], Visible="off");
        note.String = "";
    else
        p = B.y0(firing, 1:3);
        d = [B.dvV_ms(firing), B.dvN_ms(firing), B.dvB_ms(firing)] * ...
            vnbBasis(p, B.y0(firing, 4:6));
        d = 0.2 * reach * d / max(norm(d), eps);
        set(flash, XData=p(1), YData=p(2), ZData=p(3), Visible="on");
        set(arrow, XData=p(1), YData=p(2), ZData=p(3), UData=d(1), VData=d(2), WData=d(3), ...
            Visible="on");
        set(note, Position=p + 1.15 * d, String=sprintf("%s\n%s: %.1f m/s", B.name(firing), ...
            B.label(firing), B.dv_ms(firing)));
    end
    earth.Matrix = makehgtform("zrotate", gmst(julianAt(T, t)));
    view(ax, 35 + 50 * (f - 1) / max(numel(frames) - 1, 1), 22);
    speed = 0;
    if f > 1
        speed = (frames(f) - frames(f - 1)) / max(pace, eps);
    end
    heading.String = sprintf("%s UTC   (T+%s)   time x%s", string(epoch + seconds(t), ...
        "yyyy-MM-dd HH:mm:ss"), formatDuration(t), commaNum(speed));
    recent = logged(logged.t_s <= t, :);
    recent = recent(max(1, end - 2):end, :);
    ticker.String = string(recent.utc, "HH:mm:ss") + "  " + recent.text;
    if everyFrame
        drawnow
    end
    W = writeFrame(W, fig);
    if onScreen
        pause(max(0, f * pace - toc(started)));
    end
end
closeWriter(W, numel(frames));
end

function t = frameTimes(t0, t1, n, moments)
% n frame times from t0 to t1, closer together about the moments (burns, launches):
% each moment draws frames to it as if the run spent 6 times longer there. The
% moments themselves are among the frames.
if t1 <= t0 || n < 2
    t = t0;
    return
end
tg = linspace(t0, t1, 4000)';
width = max(0.015 * (t1 - t0), 60);
density = ones(size(tg));
for m = reshape(moments(moments >= t0 & moments <= t1), 1, [])
    density = density + 6 * exp(-0.5 * ((tg - m) / width) .^ 2);
end
total = cumtrapz(tg, density);
[total, keep] = unique(total);
t = interp1(total, tg(keep), linspace(0, total(end), n)');
for m = reshape(moments(moments >= t0 & moments <= t1), 1, [])
    [~, j] = min(abs(t - m));
    t(j) = m;
end
t = unique(t);
end

function [y, active, rows, i] = stateAt(rows, b, t, C)
% the state of an object at time t: its last sample at or before t (or the end of a
% burn since then), carried on along its two-body orbit to t; active false when it
% has come down or gone
i = find(rows.t_s <= t, 1, "last");
y = NaN(1, 6);
active = false;
if isempty(i)
    return
end
y = [rows.x_km(i), rows.y_km(i), rows.z_km(i), rows.vx_kms(i), rows.vy_kms(i), rows.vz_kms(i)];
tb = rows.t_s(i);
active = rows.status(i) == "active";
done = find(b.t1_s > tb & b.t1_s <= t, 1, "last");
if ~isempty(done)
    y = b.y1(done, :);
    tb = b.t1_s(done);
end
firing = find(b.t0_s > tb & b.t0_s <= t, 1, "last");
if ~isempty(firing) && b.t1_s(firing) > t      % a finite burn under way: from its start
    y = b.y0(firing, :);
    tb = b.t0_s(firing);
end
if active && t > tb && norm(y(1:3)) > C.RE
    [r, v] = keplerPropagate(y(1:3), y(4:6), t - tb, C.MU);
    y = [r, v];
end
end

% --- writing animations and figures ---------------------------------------------------------

function W = frameWriter(file, fps)
% an animation file to write frames to: an animated GIF (.gif) or an MPEG-4 video
% (.mp4) at fps frames a second; none for file ""
W = struct("kind", "", "file", file, "video", [], "size", [], "delay", 1 / max(fps, 1e-3), ...
    "count", 0);
if file == ""
    return
end
[~, ~, ext] = fileparts(file);
switch lower(ext)
    case ".gif"
        W.kind = "gif";
    case ".mp4"
        W.kind = "mp4";
        W.video = VideoWriter(file, "MPEG-4");
        W.video.FrameRate = min(max(fps, 1), 60);
        W.video.Quality = 95;
        open(W.video);
    otherwise
        error("satflight:animationFile", "An animation is written as .gif or .mp4, not %s.", ...
            file);
end
end

function W = writeFrame(W, fig)
if W.kind == ""
    return
end
img = frame2im(getframe(fig));
if isempty(W.size)
    W.size = 2 * floor([size(img, 1), size(img, 2)] / 2);    % even, for the video codec
end
% every frame the size of the first (a window can be resized while it plays)
img = img(1:min(end, W.size(1)), 1:min(end, W.size(2)), :);
img(end + 1:W.size(1), :, :) = 255;
img(:, end + 1:W.size(2), :) = 255;
W.count = W.count + 1;
if W.kind == "mp4"
    writeVideo(W.video, img);
    return
end
[ind, cmap] = rgb2ind(img, 256);
if W.count == 1
    imwrite(ind, cmap, W.file, "gif", LoopCount=Inf, DelayTime=max(W.delay, 0.02));
else
    imwrite(ind, cmap, W.file, "gif", WriteMode="append", DelayTime=max(W.delay, 0.02));
end
end

function closeWriter(W, frames)
if W.kind == ""
    return
end
if W.kind == "mp4"
    close(W.video);
end
fprintf("Wrote the animation (%d frames) to %s\n", frames, W.file);
end

function files = satflightSaveFigures(folder, figs)
%SATFLIGHTSAVEFIGURES Saves figures (all this script's figures still open when none are
%   given) to folder, each as a PNG at 300 dpi and a PDF, named after the figure:
%   "Ground track" -> ground_track.png and .pdf. The PDF is vector graphics, sharp at
%   any zoom, except for a figure with a 3-D Earth (a surface, which a PDF holds as an
%   image at 300 dpi). Returns the files written.
arguments
    folder (1,1) string
    figs = findall(groot, Type="figure", Tag="satflight")
end
if ~isfolder(folder)
    mkdir(folder);
end
figs = flip(figs(:));                 % findall lists the newest first
files = strings(0, 1);
taken = strings(0, 1);
for k = 1:numel(figs)
    fig = figs(k);
    base = lower(strip(regexprep(string(fig.Name), "[^A-Za-z0-9]+", "_"), "_"));
    if base == ""
        base = "figure";
    end
    stem = base;
    j = 2;
    while ismember(stem, taken)
        stem = base + "_" + j;
        j = j + 1;
    end
    taken(end + 1) = stem; %#ok<AGROW>
    png = fullfile(folder, stem + ".png");
    pdf = fullfile(folder, stem + ".pdf");
    exportgraphics(fig, png, Resolution=300);
    if isempty(findall(fig, Type="surface"))
        exportgraphics(fig, pdf, ContentType="vector");
    else
        exportgraphics(fig, pdf, ContentType="image", Resolution=300);
    end
    files = [files; png; pdf]; %#ok<AGROW>
end
fprintf("Saved %d figures to %s (PNG and PDF).\n", numel(figs), folder);
end

% --- cross-check against the simulator ------------------------------------------------------

function [P, E] = satflightRunSimulator(folder, scenarioText, duration, step)
%SATFLIGHTRUNSIMULATOR Run the Python simulator on the exported scenario (needs the
%   Satellite Space Flight folder, with its matlab/ toolkit, and Python with numpy).
%   Returns its ephemeris and its event log.
toolkit = fullfile(folder, "matlab");
if ~isfolder(fullfile(toolkit, "+satflight"))
    error("satflight:noSimulator", ...
        "The simulator is not at %s (no matlab/+satflight there). Set simulatorFolder to " + ...
        "your Satellite Space Flight folder, or crossCheck = false to skip this section.", folder);
end
addpath(toolkit);
file = string(tempname) + ".json";
writelines(scenarioText, file, Encoding="UTF-8");
[P, ~, E] = satflight.runScenario(file, Duration=duration, Step=step);
end

function D = satflightCompare(T, P)
%SATFLIGHTCOMPARE Position difference (km) between this script's run T and the simulator's P.
names = intersect(unique(T.name, "stable"), unique(P.name, "stable"), "stable");
onlySim = setdiff(unique(P.name, "stable"), names, "stable");
if ~isempty(onlySim)
    fprintf("Only in the simulator's run: %s\n", strjoin(onlySim, ", "));
end
onlyHere = setdiff(unique(T.name, "stable"), names, "stable");
if ~isempty(onlyHere)
    fprintf("Only in this script's run: %s\n", strjoin(onlyHere, ", "));
end
% max_diff_km is over the samples where both runs still have the object in flight. One
% that re-enters stops in both at the end of the integrator step in which it went below
% 80 km descending; this script takes the simulator's own steps, so the two frozen
% positions (final_diff_km) agree as closely as the rest of the run.
D = table('Size', [numel(names) 6], ...
    'VariableTypes', ["string", "double", "double", "double", "string", "string"], ...
    'VariableNames', ["name", "max_diff_km", "final_diff_km", "final_alt_diff_km", ...
    "status_matlab", "status_simulator"]);
fig = figure(Name="MATLAB vs simulator");
ax = axes(fig);
styleAxes(ax);
hold(ax, "on");
for k = 1:numel(names)
    a = T(T.name == names(k), :);
    b = P(P.name == names(k), :);
    [~, ia, ib] = intersect(round(a.t_s * 1000), round(b.t_s * 1000));
    if isempty(ia)
        error("satflight:noOverlap", "No common sample times for %s.", names(k));
    end
    dr = vecnorm([a.x_km(ia) - b.x_km(ib), a.y_km(ia) - b.y_km(ib), a.z_km(ia) - b.z_km(ib)], 2, 2);
    both = a.status(ia) == "active" & b.status(ib) == "active";
    D(k, :) = {names(k), max([dr(both); 0]), dr(end), a.alt_km(ia(end)) - b.alt_km(ib(end)), ...
        a.status(ia(end)), b.status(ib(end))};
    if k <= 12 && any(both)
        plot(ax, a.t_s(ia(both)) / 3600, max(dr(both), 1e-9) * 1000, DisplayName=names(k));
    end
end
ax.YScale = "log";
grid(ax, "on");
xlabel(ax, "Time since epoch (h)"); ylabel(ax, "Position difference (m)");
title(ax, "This script vs the simulator");
legend(ax, Location="eastoutside", Interpreter="none");
themeFigure(fig);
end

function M = satflightCompareEvents(events, E)
%SATFLIGHTCOMPAREEVENTS This script's event log against the simulator's, line by line.
%   Events pair up by kind and by their text with the numbers taken out (the n-th such
%   line here with the n-th there); dt_s is the time here minus the time there and
%   same_text whether the numbers agree to the digits printed. A line found in one log
%   only has NaN for the other's time. Launch events come out of the same fixed-step
%   ascent and agree to the millisecond; eclipses, station passes and re-entries are
%   seen at the end of the same integrator steps and logged at the moment they happened
%   within one (on the step's Hermite curve), so they agree to milliseconds. Neither
%   program looks for events in the last stretch of a run, shorter than a step.
kindA = string(events.kind);
kindB = string(E.kind);
textA = string(events.text);
textB = string(E.text);
match = zeros(height(events), 1);           % row of E paired with each row here
usedB = false(height(E), 1);
% first the lines whose text is the same, then the rest by their text without numbers
for pass = 1:2
    if pass == 1
        keyA = kindA + "|" + textA;
        keyB = kindB + "|" + textB;
    else
        keyA = eventKeys(kindA, textA);
        keyB = eventKeys(kindB, textB);
    end
    for i = reshape(find(match == 0), 1, [])
        j = find(~usedB & keyB == keyA(i), 1);
        if ~isempty(j)
            match(i) = j;
            usedB(j) = true;
        end
    end
end
paired = match > 0;
tSim = NaN(height(events), 1);
tSim(paired) = E.t_s(match(paired));
textSim = strings(height(events), 1);
textSim(paired) = textB(match(paired));
M = table(kindA, textA, events.t_s, tSim, events.t_s - tSim, paired & textA == textSim, ...
    textSim, VariableNames=["kind", "text", "t_matlab", "t_simulator", "dt_s", "same_text", ...
    "text_simulator"]);
rest = find(~usedB);
M = [M; table(kindB(rest), strings(numel(rest), 1), NaN(numel(rest), 1), E.t_s(rest), ...
    NaN(numel(rest), 1), false(numel(rest), 1), textB(rest), VariableNames=M.Properties.VariableNames)];
missing = sum(isnan(M.t_matlab)) + sum(isnan(M.t_simulator));
fprintf("Events: %d paired, %d with the same text, %d in one log only; largest time " + ...
    "difference %.3g s.\n", sum(~isnan(M.dt_s)), sum(M.same_text), missing, max([abs(M.dt_s); 0]));
end

function key = eventKeys(kind, text)
key = kind + "|" + regexprep(text, "[-+]?\d[\d,]*(\.\d+)?", "#");
end
