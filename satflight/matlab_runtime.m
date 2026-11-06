% ==========================================================================================
% Satellite Space Flight engine, ported to MATLAB.
% satflight/matlab_runtime.m: satflight.matlab_export appends this file to every exported
% script. It follows the simulator's Python code term by term (satflight/forces.py,
% atmosphere.py, frames.py, elements.py, analysis.py, maneuvers.py, launch.py, eclipse.py,
% ephemeris.py and the stepping and event rules of simulation.py), so a script runs the
% same physics, the same rocket ascents and logs the same events without Python.
% ==========================================================================================

function [T, events, launches] = satflightRun(sim, duration, step, opts)
%SATFLIGHTRUN Run the exported simulation as the simulator does.
%   [T, events, launches] = satflightRun(sim, duration, step) returns one table row
%   per object per sample - the columns of the simulator's ephemeris CSV, with a
%   launched payload from its pad and each spent stage from its separation - the
%   event log (the simulator's lines: launches, burns, eclipses, station passes,
%   close approaches, re-entries) and one entry per launch: outcome, insertion
%   orbit, delta-v budget and ascent profile.
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
n = numel(sim.names);
s.names = reshape(string(sim.names), [], 1);
s.Y = reshape(sim.y0, n, 6);
s.mass = reshape(sim.mass, [], 1);
s.area = reshape(sim.area, [], 1);
s.cd = reshape(sim.cd, [], 1);
s.status = repmat("active", n, 1);
s.dvUsed = zeros(n, 1);
s.family = strings(n, 1);
s.shadow = ones(n, 1);
s = stationSetup(s, sim.stations);
s.man = sim.maneuvers;
s.pending = true(1, numel(s.man));
s.burns = struct("sat", {}, "man", {}, "t0", {}, "t1", {}, "m0", {}, "mdot", {});
s.ascents = {};
s.reports = {};
s.evT = zeros(0, 1);
s.evKind = strings(0, 1);
s.evText = strings(0, 1);
s.t = 0;
s.closePairs = zeros(0, 2);
s.lastSample = [];

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
idx = freeIdx(s);
s = checkLeft(s, idx);
s = stepEvents(s, 0, idx, s.Y);
s = logEvent(s, 0, sprintf("Scenario '%s' loaded: %d satellites, %d stations, propagator %s", ...
    sim.name, numel(s.names), numel(s.st.name), sim.propagator), "info");
s = recordAt(s, 1, s.Y);                 % the first row is written before anything executes
s = executeDue(s);

tEnd = tout(end);
while s.t < tEnd - 1e-9
    tb = nextBoundary(s);
    te = min(tb, tEnd);
    idx = freeIdx(s);
    if isempty(idx)
        s = recordBetween(s, te, idx, []);
        s.t = te;
        s.lastSample = [];
    elseif s.sim.propagator == "cowell"
        s = segmentCowell(s, idx, te);
    else
        s = segmentAnalytic(s, idx, te);
    end
    if abs(s.t - tb) <= 1e-9
        s = advanceAscents(s, tb);
        s = finishBurns(s);
        s = executeDue(s);
    end
    k = find(abs(tout - s.t) <= 1e-9, 1);
    if ~isempty(k)
        s = recordAt(s, k, s.Y);
    end
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
end

function idx = freeIdx(s)
% the active objects the ensemble propagates: not on a pad or in powered flight
act = s.status == "active";
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

function s = recordBetween(s, te, idx, sol)
% samples strictly between now and te, from the dense output of an ode45 segment
% (or as they are, with nothing integrated); vehicles on pads and in flight as shown
ks = find(s.tout > s.t + 1e-9 & s.tout < te - 1e-9);
if isempty(ks)
    return
end
if ~isempty(sol)
    vals = deval(sol, s.tout(ks));
end
for q = 1:numel(ks)
    Ycur = s.Y;
    if ~isempty(sol)
        Ycur(idx, :) = reshape(vals(:, q), numel(idx), 6);
    end
    s = recordAt(s, ks(q), Ycur);
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

function [s, row] = addRow(s, name, y, mass, area, cd, family)
% a new object (a payload on its pad, a spent stage); its name made unique as the
% simulator makes it
row = numel(s.names) + 1;
s.names(row, 1) = uniqueName(s, name);
s.Y(row, :) = y;
s.mass(row, 1) = mass;
s.area(row, 1) = area;
s.cd(row, 1) = cd;
s.status(row, 1) = "active";
s.dvUsed(row, 1) = 0;
s.family(row, 1) = family;
s.shadow(row, 1) = 1;
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
    dv = impulseEci(m, r, v, tv, C);
    dvm = norm(dv);
    s.Y(i, 4:6) = v + dv;
    s.dvUsed(i) = s.dvUsed(i) + dvm;
    s.mass(i) = s.mass(i) * exp(-dvm / (m.isp * C.G0));
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
    duration = min(duration, 0.95 * s.mass(i) / mdot);
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
    s.Y(i, 4:6) = s.Y(i, 4:6) + dvm * d;
    s.dvUsed(i) = s.dvUsed(i) + dvm;
    s.mass(i) = mf;
    s = logEvent(s, s.t, sprintf("%s: finite burn applied impulsively (%.1f m/s) - switch " + ...
        "to Cowell to integrate thrust", m.sat, dvm * 1000), "maneuver");
    return
end
s.burns(end + 1) = struct("sat", i, "man", j, "t0", s.t, "t1", s.t + duration, ...
    "m0", s.mass(i), "mdot", mdot);
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
    s = logEvent(s, s.t, sprintf("%s: burnout - dV %.1f m/s, mass %.1f kg", m.sat, dvm * 1000, ...
        mf), "maneuver");
end
s.burns = s.burns(keep);
end

function s = scheduleAtApsis(s, i, m)
% queue maneuver m for object i at its next apoapsis (the circularization a launch
% asks for), as the simulator's schedule() resolves it
el = rv2coe(s.Y(i, 1:3), s.Y(i, 4:6), s.C.MU);
if el.e >= 1
    error("satflight:maneuver", "open orbit has no apoapsis");
end
dt = timeToTrueAnomaly(s.Y(i, 1:3), s.Y(i, 4:6), pi, s.C.MU);
if ~isfinite(dt)
    error("satflight:maneuver", "orbit never reaches apoapsis");
end
m.t = s.t + dt;
s.man(end + 1) = m;
s.pending(end + 1) = true;
s = logEvent(s, s.t, sprintf("Scheduled %s for %s %s", m.label, m.sat, whenText(m.t, s.t)), ...
    "maneuver");
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

function s = segmentCowell(s, idx, te)
C = s.C;
[s, gone] = checkLeft(s, idx);
if gone
    return
end
cdam = s.cd(idx) .* s.area(idx) ./ s.mass(idx);
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
it = s.sim.integrator;
opts = odeset(RelTol=it.rtol, AbsTol=it.atol, MaxStep=it.hMax, ...
    Events=@(t, y) leaveEvents(y, nIdx, C));
F = s.sim.forces;
f = @(t, y) derivative(t, y, nIdx, cdam, F, s.sim.jd0, s.man, burnRows, mdots, C);
sol = ode45(f, [s.t, te], reshape(s.Y(idx, :), [], 1), opts);
tReach = sol.x(end);
s = recordBetween(s, tReach, idx, sol);
% the simulator looks for events at the end of every integrator step
for j = 2:numel(sol.x) - 1
    Ystep = s.Y;
    Ystep(idx, :) = reshape(sol.y(:, j), nIdx, 6);
    s = stepEvents(s, sol.x(j), idx, Ystep);
end
s.Y(idx, :) = reshape(sol.y(:, end), nIdx, 6);
s.t = tReach;
if tReach < te - 1e-9 && isfield(sol, "ie") && ~isempty(sol.ie)
    for e = reshape(sol.ie, 1, [])
        row = mod(e - 1, nIdx) + 1;
        s = leave(s, idx(row), e > nIdx);
    end
end
if numel(sol.x) > 1
    s = stepEvents(s, s.t, idx, s.Y);
end
end

function dy = derivative(t, y, n, cdam, F, jd0, man, burnRows, mdots, C)
Y = reshape(y, n, 6);
r = Y(:, 1:3);
v = Y(:, 4:6);
theta = 0;
if F.c22
    theta = gmst(jd0 + t / 86400);
end
a = acceleration(r, v, cdam, theta, F, C);
for k = 1:size(burnRows, 1)
    row = burnRows(k, 1);
    m = man(burnRows(k, 2));
    mass = burnRows(k, 4) - mdots(k) * (t - burnRows(k, 3));
    d = finiteDirection(m, r(row, :), v(row, :));
    a(row, :) = a(row, :) + d * (m.thrust / mass) * 1e-3;
end
dy = reshape([v, a], [], 1);
end

function [value, terminal, direction] = leaveEvents(y, n, C)
% a satellite is lost once it is below the re-entry altitude AND descending (as the
% simulator decides it): the first value turns negative exactly then, whether it
% sinks through 80 km or turns over below it; the second marks escape
Y = reshape(y, n, 6);
r = Y(:, 1:3);
radialSpeed = sum(r .* Y(:, 4:6), 2) ./ vecnorm(r, 2, 2);
value = [max(approxAltitude(r, C) - C.REENTRY_ALT, radialSpeed); vecnorm(r, 2, 2) - C.SOI];
terminal = ones(2 * n, 1);
direction = [-ones(n, 1); ones(n, 1)];
end

function [s, gone] = checkLeft(s, idx)
% a satellite already below the re-entry altitude and descending (or beyond the
% sphere of influence) leaves before anything is integrated
C = s.C;
r = s.Y(idx, 1:3);
v = s.Y(idx, 4:6);
low = approxAltitude(r, C) < C.REENTRY_ALT & sum(r .* v, 2) < 0;
far = vecnorm(r, 2, 2) > C.SOI;
gone = any(low | far);
for k = reshape(find(low | far), 1, [])
    s = leave(s, idx(k), far(k));
end
end

function s = leave(s, i, escaped)
C = s.C;
name = s.names(i);
r = s.Y(i, 1:3);
speed = norm(s.Y(i, 4:6));
if escaped
    s.status(i) = "escaped";
    s = logEvent(s, s.t, sprintf("%s left Earth's sphere of influence", name), "alert");
elseif approxAltitude(r, C) <= 0
    s.status(i) = "impacted";
    s = logEvent(s, s.t, sprintf("%s impacted the surface at %.2f km/s", name, speed), "alert");
else
    s.status(i) = "re-entered";
    s = logEvent(s, s.t, sprintf("%s re-entered (below %.0f km, %.2f km/s)", name, ...
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
    s = logEvent(s, s.t, sprintf("%s: %d pending maneuver%s cancelled", name, nnz(mine), ...
        plural), "warn");
end
s.pending(mine) = false;
if ~isempty(s.burns)
    s.burns = s.burns([s.burns.sat] ~= i);
end
end

% --- analytic propagation (propagators "kepler" and "j2mean") --------------------------------

function s = segmentAnalytic(s, idx, te)
C = s.C;
h = s.sim.integrator.hMax;
while s.t < te - 1e-9
    dt = min(h, te - s.t);
    Y0 = s.Y(idx, :);
    tn = s.t + dt;
    ks = find(s.tout > s.t + 1e-9 & s.tout < tn - 1e-9);
    for q = 1:numel(ks)
        Ycur = s.Y;
        Ycur(idx, :) = analyticStep(Y0, s.tout(ks(q)) - s.t, s.sim.propagator, C);
        s = recordAt(s, ks(q), Ycur);
    end
    s.Y(idx, :) = analyticStep(Y0, dt, s.sim.propagator, C);
    s.t = tn;
    [s, gone] = checkLeft(s, idx);
    s = stepEvents(s, tn, idx, s.Y);
    if tn < te - 1e-9
        k = find(abs(s.tout - tn) <= 1e-9, 1);
        if ~isempty(k)
            s = recordAt(s, k, s.Y);
        end
    end
    if gone
        return
    end
end
end

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

function a = acceleration(r, v, cdam, theta, F, C)
x = r(:, 1); y = r(:, 2); z = r(:, 3);
r2 = x .* x + y .* y + z .* z;
rm = sqrt(r2);
a = -C.MU * r ./ rm .^ 3;
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
    c = cos(theta); s = sin(theta);
    xe = c * x + s * y;
    ye = -s * x + c * y;
    k = 3 * C.MU * C.RE ^ 2 ./ (r2 .* r2 .* rm);
    w = C.C22 * (xe .* xe - ye .* ye) + 2 * C.S22 * xe .* ye;
    q = 5 * w ./ r2;
    gx = k .* (2 * (C.C22 * xe + C.S22 * ye) - q .* xe);
    gy = k .* (2 * (C.S22 * xe - C.C22 * ye) - q .* ye);
    gz = -k .* q .* z;
    a = a + [c * gx - s * gy, s * gx + c * gy, gz];
end
if F.drag
    vRel = v - [-C.OMEGA * y, C.OMEGA * x, zeros(size(x))];
    rho = density(approxAltitude(r, C), F.densityScale, C);
    speed = vecnorm(vRel, 2, 2);
    a = a + (-0.5e3 * rho .* cdam .* speed) .* vRel;
end
end

function h = approxAltitude(r, C)
rm = vecnorm(r, 2, 2);
h = rm - C.RE * (1 - C.F * (r(:, 3) ./ rm) .^ 2);
end

function rho = density(h, scale, C)
% piecewise-exponential atmosphere (Vallado table 8-4), kg/m^3
hc = max(h, 0);
k = discretize(hc, [C.ATM(:, 1); inf]);
rho = C.ATM(k, 2) .* exp(-(hc - C.ATM(k, 1)) ./ C.ATM(k, 3)) * scale;
rho(h > 2500) = 0;
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
x = r(:, 1); y = r(:, 2); z = r(:, 3);
lon = atan2(y, x);
p = hypot(x, y);
lat = atan2(z, p * (1 - C.E2));
for k = 1:6
    sl = sin(lat);
    n = C.RE ./ sqrt(1 - C.E2 * sl .* sl);
    lat = atan2(z + C.E2 * n .* sl, p);
end
sl = sin(lat);
n = C.RE ./ sqrt(1 - C.E2 * sl .* sl);
h = p .* cos(lat) + (z + C.E2 * n .* sl) .* sl - n;
end

% --- events (simulation.py: eclipses, station passes, close approaches) ----------------------

function s = stepEvents(s, t, idx, Y)
% what the simulator looks for at the end of every step: eclipse entry and exit and
% station AOS/LOS (for each object while there are at most 40), close approaches
C = s.C;
if numel(s.names) <= C.EVENT_DETAIL_LIMIT && ~isempty(idx)
    w = sort(idx(:));
    s = eclipseEvents(s, t, w, Y);
    if ~isempty(s.st.name)
        s = stationEvents(s, t, w, Y);
    end
end
s = approachEvents(s, t, idx, Y);
end

function s = eclipseEvents(s, t, w, Y)
fr = shadowFraction(Y(w, 1:3), sunPosition(s.sim.jd0 + t / 86400, s.C), s.C);
for k = 1:numel(w)
    i = w(k);
    if s.shadow(i) >= 0.5 && 0.5 > fr(k)
        s = logEvent(s, t, sprintf("%s entered Earth's shadow", s.names(i)), "eclipse");
    elseif s.shadow(i) < 0.5 && 0.5 <= fr(k)
        s = logEvent(s, t, sprintf("%s back in sunlight", s.names(i)), "eclipse");
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
th = gmst(s.sim.jd0 + t / 86400);
c = cos(th);
sn = sin(th);
r = Y(w, 1:3);
ecef = [c * r(:, 1) + sn * r(:, 2), -sn * r(:, 1) + c * r(:, 2), r(:, 3)];
for k = 1:numel(s.st.name)
    enu = (ecef - s.st.ecef(k, :)) * s.st.enu(:, :, k).';
    rng = vecnorm(enu, 2, 2);
    rng(rng == 0) = 1;
    sees = asin(min(max(enu(:, 3) ./ rng, -1), 1)) >= deg2rad(s.st.minEl(k));
    for q = 1:numel(w)
        i = w(q);
        if sees(q) && ~s.visible(k, i)
            s = logEvent(s, t, sprintf("AOS %s @ %s", s.names(i), s.st.name(k)), "station");
        elseif ~sees(q) && s.visible(k, i)
            s = logEvent(s, t, sprintf("LOS %s @ %s", s.names(i), s.st.name(k)), "station");
        end
        s.visible(k, i) = sees(q);
    end
end
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
orbiting = idx(approxAltitude(Y(idx, 1:3), C) > 100);
if numel(orbiting) < 2
    s.lastSample = [];
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
mask = false(numel(s.names), 1);
mask(orbiting) = true;
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
    rate = j2SecularRates(el.a, el.e, el.i, s.C);
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
p = exp(-max(A.spec.alt, 0) / C.P_SCALE_H);
thrust = stageMdot(st, C) * C.G0M * stageIsp(st, p);
r = sqrt(A.y(1) ^ 2 + A.y(2) ^ 2 + A.y(3) ^ 2);
weight = A.mass * C.MU / (r * r) * 1e3;
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

function th = ascThrottle(A, st, alt, C)
% throttle (0.4-1) that keeps the vehicle within its acceleration limit
if A.veh.maxG <= 0
    th = 1;
    return
end
p = exp(-max(alt, 0) / C.P_SCALE_H);
a = stageMdot(st, C) * C.G0M * stageIsp(st, p) / A.mass;
th = max(0.4, min(1, A.veh.maxG * C.G0M / a));
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
    A.throttle = ascThrottle(A, st, alt, C);
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
    p = exp(-max(alt, 0) / C.P_SCALE_H);
    a = mdot * C.G0M * stageIsp(st, p) / m;
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
    p = exp(-max(ascAltitude(r, C), 0) / C.P_SCALE_H);
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
% the atmosphere's density (kg/m^3) at h km, unscaled
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
    p = exp(-max(alt, 0) / C.P_SCALE_H);
    a = mdot * C.G0M * stageIsp(st, p) / m * 1e-3;
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
C.DEG = pi / 180;
C.G0 = 9.80665e-3;            % km/s^2
% satflight/simulation.py
C.REENTRY_ALT = 80.0;         % km
C.ASCENT_SYNC = 1.0;          % s
C.EVENT_DETAIL_LIMIT = 40;
C.ALL_PAIRS_LIMIT = 300;
C.CONJUNCTION_KM = 10.0;
% satflight/launch.py
C.G0M = 9.80665;              % m/s^2
C.P_SCALE_H = 7.0;            % km
C.GUIDANCE_ALT = 50.0;        % km
C.H_POWERED = 1.0;            % s
C.H_COAST = 5.0;              % s
C.SAMPLE_DT = 2.0;            % s
C.FREEZE_TGO = 4.0;           % s
C.MIN_PERIGEE = 100.0;        % km
C.MAX_BURN = 3600.0;          % s
% satflight/atmosphere.py: base altitude km, density kg/m^3, scale height km
C.TOP_KM = 2500.0;
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
    "payloadMass", 1000, "payloadArea", 5, "payloadCd", 2.2, "circularize", false, ...
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

function names = plotNames(T, names)
present = unique(T.name, "stable")';
if isempty(names)
    names = present(1:min(12, end));
    if numel(present) > 12
        fprintf("Plotting the first 12 of %d satellites; pass names to choose.\n", numel(present));
    end
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
hold(ax, "on");
box(ax, "on");
coast = satflightCoastline();
plot(ax, coast(:, 1), coast(:, 2), Color=[0.55 0.55 0.55], LineWidth=0.5, ...
    HandleVisibility="off");
for nm = names
    rows = T(T.name == nm, :);
    lon = rows.lon_deg;
    lat = rows.lat_deg;
    for j = flip(find(abs(diff(lon)) > 180))'
        lon = [lon(1:j); NaN; lon(j + 1:end)];
        lat = [lat(1:j); NaN; lat(j + 1:end)];
    end
    h = plot(ax, lon, lat, LineWidth=1.2, DisplayName=nm);
    plot(ax, rows.lon_deg(end), rows.lat_deg(end), "o", MarkerFaceColor=h.Color, ...
        MarkerEdgeColor="k", HandleVisibility="off");
end
for g = reshape(stations, 1, [])
    plot(ax, g.lon, g.lat, "^", MarkerSize=8, MarkerFaceColor="k", MarkerEdgeColor="w", ...
        HandleVisibility="off");
    text(ax, g.lon + 2, g.lat, g.name, FontSize=8);
end
xlim(ax, [-180 180]); ylim(ax, [-90 90]); daspect(ax, [1 1 1]);
xticks(ax, -180:60:180); yticks(ax, -90:30:90); grid(ax, "on");
xlabel(ax, "Longitude (deg)"); ylabel(ax, "Latitude (deg)");
fmt = "yyyy-MM-dd HH:mm";
title(ax, sprintf("Ground track, %s to %s UTC", string(T.utc(1), fmt), string(T.utc(end), fmt)));
legend(ax, Location="southoutside", Orientation="horizontal", Interpreter="none");
end

function fig = satflightPlotAltitude(T, names)
arguments
    T table
    names = string.empty
end
names = plotNames(T, names);
fig = figure(Name="Altitude");
ax = axes(fig);
hold(ax, "on");
for nm = names
    rows = T(T.name == nm, :);
    plot(ax, rows.t_s / 3600, rows.alt_km, LineWidth=1.2, DisplayName=nm);
end
grid(ax, "on");
xlabel(ax, "Time since epoch (h)"); ylabel(ax, "Altitude (km)"); title(ax, "Altitude");
legend(ax, Location="best", Interpreter="none");
end

function fig = satflightPlotElements(T, names)
arguments
    T table
    names = string.empty
end
names = plotNames(T, names);
panels = ["a_km", "Semi-major axis (km)"; "e", "Eccentricity"; "i_deg", "Inclination (deg)"; ...
    "raan_deg", "RAAN, unwrapped (deg)"; "argp_deg", "Arg. of perigee, unwrapped (deg)"; ...
    "alt_km", "Altitude (km)"];
fig = figure(Name="Orbital elements");
layout = tiledlayout(fig, 2, 3, TileSpacing="compact");
title(layout, "Osculating elements");
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    hold(ax, "on");
    for nm = names
        rows = T(T.name == nm, :);
        y = rows.(panels(p, 1));
        if ismember(panels(p, 1), ["raan_deg", "argp_deg"])
            y = rad2deg(unwrap(deg2rad(y)));
        end
        plot(ax, rows.t_s / 3600, y, LineWidth=1.1, DisplayName=nm);
    end
    grid(ax, "on");
    xlabel(ax, "Time (h)"); ylabel(ax, panels(p, 2));
end
legend(ax, Location="best", Interpreter="none");
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
hold(ax, "on");
[x, y, z] = sphere(48);
surf(ax, C.RE * x, C.RE * y, C.RE * (1 - C.F) * z, FaceColor=[0.35 0.55 0.85], ...
    EdgeColor=[0.25 0.4 0.65], EdgeAlpha=0.3, FaceAlpha=0.6, HandleVisibility="off");
for nm = names
    rows = T(T.name == nm, :);
    h = plot3(ax, rows.x_km, rows.y_km, rows.z_km, LineWidth=1.2, DisplayName=nm);
    plot3(ax, rows.x_km(end), rows.y_km(end), rows.z_km(end), "o", MarkerFaceColor=h.Color, ...
        MarkerEdgeColor="k", HandleVisibility="off");
end
axis(ax, "equal"); grid(ax, "on"); view(ax, 35, 25);
xlabel(ax, "x (km)"); ylabel(ax, "y (km)"); zlabel(ax, "z (km)");
title(ax, "Trajectories, ECI");
legend(ax, Location="best", Interpreter="none");
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
        plot(ax, L.planned.(panels(k, 1)), L.planned.(panels(k, 2)), "--", ...
            Color=[0.6 0.6 0.6], DisplayName="plan");
        plot(ax, L.profile.(panels(k, 1)), L.profile.(panels(k, 2)), LineWidth=1.3, ...
            DisplayName="flown");
        xlabel(ax, panels(k, 3));
        ylabel(ax, panels(k, 4));
        if k == 1
            plot(ax, L.marks.downrange_km, L.marks.alt_km, "o", MarkerFaceColor="k", ...
                MarkerEdgeColor="w", HandleVisibility="off");
            text(ax, L.marks.downrange_km, L.marks.alt_km, "  " + L.marks.label, FontSize=7);
            legend(ax, Location="southeast");
        else
            for j = 1:height(L.marks)
                xline(ax, L.marks.met_s(j), ":", Color=[0.4 0.4 0.4], HandleVisibility="off");
            end
        end
    end
end
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
% that re-enters stops here at the moment it is below 80 km and descending, and in the
% simulator at the end of the integrator step in which that happened, so the two frozen
% positions (final_diff_km) can differ by up to one step's travel.
D = table('Size', [numel(names) 6], ...
    'VariableTypes', ["string", "double", "double", "double", "string", "string"], ...
    'VariableNames', ["name", "max_diff_km", "final_diff_km", "final_alt_diff_km", ...
    "status_matlab", "status_simulator"]);
fig = figure(Name="MATLAB vs simulator");
ax = axes(fig);
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
legend(ax, Location="best", Interpreter="none");
end

function M = satflightCompareEvents(events, E)
%SATFLIGHTCOMPAREEVENTS This script's event log against the simulator's, line by line.
%   Events pair up by kind and by their text with the numbers taken out (the n-th such
%   line here with the n-th there); dt_s is the time here minus the time there and
%   same_text whether the numbers agree to the digits printed. A line found in one log
%   only has NaN for the other's time. Launch events come out of the same fixed-step
%   ascent and agree to the millisecond; eclipses, station passes and re-entries are
%   seen at the end of an integrator step, so they agree to within a step. The
%   simulator shows the last stretch of a run (less than a step) without looking for
%   events in it, so an event that close to the end may be in this log only.
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
