% ==========================================================================================
% Satellite Space Flight engine, ported to MATLAB.
% satflight/matlab_runtime.m: satflight.matlab_export appends this file to every exported
% script. It follows the simulator's Python code term by term (satflight/forces.py,
% atmosphere.py, frames.py, elements.py, analysis.py, maneuvers.py and the stepping
% rules of simulation.py), so a script runs the same physics without Python.
% ==========================================================================================

function [T, events] = satflightRun(sim, duration, step)
%SATFLIGHTRUN Propagate the exported satellites as the simulator does.
%   [T, events] = satflightRun(sim, duration, step) returns one table row per
%   satellite per sample - the columns of the simulator's ephemeris CSV - and
%   the event log (burns, re-entries).
arguments
    sim (1,1) struct
    duration (1,1) double {mustBeNonnegative, mustBeFinite}
    step (1,1) double {mustBePositive, mustBeFinite}
end
C = satflightConstants();
n = numel(sim.names);
if n == 0
    error("satflight:empty", "There are no satellites to propagate in MATLAB.");
end
s = struct();
s.sim = sim;
s.Y = sim.y0;
s.mass = sim.mass(:);
s.status = repmat("active", n, 1);
s.dvUsed = zeros(n, 1);
s.pending = true(1, numel(sim.maneuvers));
s.burns = struct("sat", {}, "man", {}, "t0", {}, "t1", {}, "m0", {}, "mdot", {});
s.events = strings(0, 1);
s.t = 0;

% the sample times the batch runner writes: 0, step, 2 step, ... and the end
tout = 0;
while tout(end) < duration - 1e-9
    tout(end + 1) = tout(end) + min(step, duration - tout(end)); %#ok<AGROW>
end
s.tout = tout;
s.Yout = nan(n, 6, numel(tout));
s.statusOut = strings(n, numel(tout));
s = recordAt(s, 1, s.Y);                 % the first row is written before anything executes
s = executeDue(s, C);

while s.t < tout(end) - 1e-9
    tb = nextBoundary(s);
    te = min(tb, tout(end));
    if s.sim.propagator == "cowell"
        s = segmentCowell(s, te, C);
    else
        s = segmentAnalytic(s, te, C);
    end
    if abs(s.t - tb) <= 1e-9
        s = finishBurns(s, C);
        s = executeDue(s, C);
    end
    k = find(abs(tout - s.t) <= 1e-9, 1);
    if ~isempty(k)
        s = recordAt(s, k, s.Y);
    end
end
missed = find(any(s.statusOut == "", 1));
if ~isempty(missed)
    error("satflight:internal", "No state was recorded for t = %g s (a bug in this script).", ...
        tout(missed(1)));
end
T = ephemerisTable(s, C);
events = s.events;
end

function s = recordAt(s, k, Ycur)
s.Yout(:, :, k) = Ycur;
s.statusOut(:, k) = s.status;
end

function tb = nextBoundary(s)
tb = inf;
for j = find(s.pending)
    tm = s.sim.maneuvers(j).t;
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

function s = logEvent(s, t, text)
s.events(end + 1, 1) = sprintf("T+%s  %s", durationText(t), text);
end

function text = durationText(t)
d = floor(t / 86400);
rest = t - 86400 * d;
h = floor(rest / 3600);
m = floor((rest - 3600 * h) / 60);
sec = rest - 3600 * h - 60 * m;
text = sprintf("%dd %02d:%02d:%04.1f", d, h, m, sec);
end

% --- maneuvers ------------------------------------------------------------------------------

function s = executeDue(s, C)
for j = find(s.pending)
    m = s.sim.maneuvers(j);
    if m.t > s.t + 1e-6
        continue
    end
    s.pending(j) = false;
    i = find(s.sim.names == m.sat, 1);
    if isempty(i)
        error("satflight:noSatellite", "Maneuver %s is for %s, which is not in the scenario.", ...
            m.label, m.sat);
    end
    if s.status(i) ~= "active"
        s = logEvent(s, s.t, sprintf("%s: %s skipped (%s)", m.sat, m.label, s.status(i)));
        continue
    end
    r = s.Y(i, 1:3);
    v = s.Y(i, 4:6);
    if m.kind == "finite"
        s = startFinite(s, j, i, C);
        continue
    end
    tv = [];
    if m.kind == "match_velocity"
        it = find(s.sim.names == m.target, 1);
        if isempty(it)
            s = logEvent(s, s.t, sprintf("%s: %s skipped (%s no longer exists)", m.sat, ...
                m.label, m.target));
            continue
        end
        tv = s.Y(it, 4:6);
    end
    dv = impulseEci(m, r, v, tv, C);
    dvm = norm(dv);
    s.Y(i, 4:6) = v + dv;
    s.dvUsed(i) = s.dvUsed(i) + dvm;
    s.mass(i) = s.mass(i) * exp(-dvm / (m.isp * C.G0));
    s = logEvent(s, s.t, sprintf("%s: executed %s - dV %.2f m/s", m.sat, m.label, dvm * 1000));
end
end

function s = startFinite(s, j, i, C)
m = s.sim.maneuvers(j);
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
    s = logEvent(s, s.t, sprintf("%s: finite burn applied impulsively (%.1f m/s)", m.sat, ...
        dvm * 1000));
    return
end
s.burns(end + 1) = struct("sat", i, "man", j, "t0", s.t, "t1", s.t + duration, ...
    "m0", s.mass(i), "mdot", mdot);
s = logEvent(s, s.t, sprintf("%s: ignition - %.0f N for %.0f s", m.sat, m.thrust, duration));
end

function s = finishBurns(s, C)
keep = true(size(s.burns));
for k = 1:numel(s.burns)
    b = s.burns(k);
    if b.t1 > s.t + 1e-6
        continue
    end
    keep(k) = false;
    m = s.sim.maneuvers(b.man);
    mf = b.m0 - b.mdot * (b.t1 - b.t0);
    dvm = 0;
    if mf > 0
        dvm = m.isp * C.G0 * log(b.m0 / mf);
    end
    s.mass(b.sat) = mf;
    s.dvUsed(b.sat) = s.dvUsed(b.sat) + dvm;
    s = logEvent(s, s.t, sprintf("%s: burnout - dV %.1f m/s, mass %.1f kg", m.sat, dvm * 1000, mf));
end
s.burns = s.burns(keep);
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

function s = segmentCowell(s, te, C)
idx = find(s.status == "active");
if isempty(idx)
    s.t = te;
    return
end
[s, gone] = checkLeft(s, idx, C);
if gone
    return
end
cdam = s.sim.cd(idx) .* s.sim.area(idx) ./ s.mass(idx);
burnRows = zeros(0, 4);      % row in idx, maneuver, t0, m0 ... mdot kept in a parallel list
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
f = @(t, y) derivative(t, y, nIdx, cdam, s.sim, burnRows, mdots, C);
sol = ode45(f, [s.t, te], reshape(s.Y(idx, :), [], 1), opts);
tReach = sol.x(end);
ks = find(s.tout > s.t + 1e-9 & s.tout < tReach - 1e-9);
if ~isempty(ks)
    vals = deval(sol, s.tout(ks));
    for q = 1:numel(ks)
        Ycur = s.Y;
        Ycur(idx, :) = reshape(vals(:, q), nIdx, 6);
        s = recordAt(s, ks(q), Ycur);
    end
end
s.Y(idx, :) = reshape(sol.y(:, end), nIdx, 6);
s.t = tReach;
if tReach < te - 1e-9 && isfield(sol, "ie") && ~isempty(sol.ie)
    for e = reshape(sol.ie, 1, [])
        row = mod(e - 1, nIdx) + 1;
        s = leave(s, idx(row), e > nIdx, C);
    end
end
end

function dy = derivative(t, y, n, cdam, sim, burnRows, mdots, C)
Y = reshape(y, n, 6);
r = Y(:, 1:3);
v = Y(:, 4:6);
F = sim.forces;
theta = 0;
if F.c22
    theta = gmst(sim.jd0 + t / 86400);
end
a = acceleration(r, v, cdam, theta, F, C);
for k = 1:size(burnRows, 1)
    row = burnRows(k, 1);
    m = sim.maneuvers(burnRows(k, 2));
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

function [s, gone] = checkLeft(s, idx, C)
% a satellite already below the re-entry altitude and descending (or beyond the
% sphere of influence) leaves before anything is integrated
r = s.Y(idx, 1:3);
v = s.Y(idx, 4:6);
low = approxAltitude(r, C) < C.REENTRY_ALT & sum(r .* v, 2) < 0;
far = vecnorm(r, 2, 2) > C.SOI;
gone = any(low | far);
for k = reshape(find(low | far), 1, [])
    s = leave(s, idx(k), far(k), C);
end
end

function s = leave(s, i, escaped, C)
name = s.sim.names(i);
r = s.Y(i, 1:3);
speed = norm(s.Y(i, 4:6));
if escaped
    s.status(i) = "escaped";
    s = logEvent(s, s.t, sprintf("%s left Earth's sphere of influence", name));
elseif approxAltitude(r, C) <= 0
    s.status(i) = "impacted";
    s = logEvent(s, s.t, sprintf("%s impacted the surface at %.2f km/s", name, speed));
else
    s.status(i) = "re-entered";
    s = logEvent(s, s.t, sprintf("%s re-entered (below %.0f km, %.2f km/s)", name, ...
        C.REENTRY_ALT, speed));
end
mine = s.pending & reshape(string({s.sim.maneuvers.sat}), 1, []) == name;
if any(mine)
    s = logEvent(s, s.t, sprintf("%s: %d pending maneuver(s) cancelled", name, nnz(mine)));
end
s.pending(mine) = false;
s.burns = s.burns([s.burns.sat] ~= i);
end

% --- analytic propagation (propagators "kepler" and "j2mean") --------------------------------

function s = segmentAnalytic(s, te, C)
idx = find(s.status == "active");
if isempty(idx)
    s.t = te;
    return
end
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
    if tn < te - 1e-9
        k = find(abs(s.tout - tn) <= 1e-9, 1);
        if ~isempty(k)
            s = recordAt(s, k, s.Y);
        end
    end
    [s, gone] = checkLeft(s, idx, C);
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

% --- output --------------------------------------------------------------------------------

function T = ephemerisTable(s, C)
n = numel(s.sim.names);
nt = numel(s.tout);
rows = n * nt;
t = reshape(repmat(s.tout, n, 1), rows, 1);
Y = reshape(permute(s.Yout, [1 3 2]), rows, 6);
theta = gmst(s.sim.jd0 + t / 86400);
c = cos(theta); sn = sin(theta);
ecef = [c .* Y(:, 1) + sn .* Y(:, 2), -sn .* Y(:, 1) + c .* Y(:, 2), Y(:, 3)];
[lat, lon, alt] = ecefToGeodetic(ecef, C);
el = rv2coe(Y(:, 1:3), Y(:, 4:6), C.MU);
T = table(t, s.sim.epoch + seconds(t), repmat(s.sim.names(:), nt, 1), ...
    reshape(s.statusOut, rows, 1), Y(:, 1), Y(:, 2), Y(:, 3), Y(:, 4), Y(:, 5), Y(:, 6), ...
    alt, rad2deg(lat), rad2deg(lon), el.a, el.e, rad2deg(el.i), rad2deg(el.raan), ...
    rad2deg(el.argp), rad2deg(el.nu), VariableNames=["t_s", "utc", "name", "status", ...
    "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms", "alt_km", "lat_deg", "lon_deg", ...
    "a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]);
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
C.G0 = 9.80665e-3;            % km/s^2
C.REENTRY_ALT = 80.0;         % km
% satflight/atmosphere.py: base altitude km, density kg/m^3, scale height km
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

% --- cross-check against the simulator ------------------------------------------------------

function P = satflightRunSimulator(folder, scenarioText, duration, step)
%SATFLIGHTRUNSIMULATOR Run the Python simulator on the exported scenario (needs the
%   Satellite Space Flight folder, with its matlab/ toolkit, and Python with numpy).
toolkit = fullfile(folder, "matlab");
if ~isfolder(fullfile(toolkit, "+satflight"))
    error("satflight:noSimulator", ...
        "The simulator is not at %s (no matlab/+satflight there). Set simulatorFolder to " + ...
        "your Satellite Space Flight folder, or crossCheck = false to skip this section.", folder);
end
addpath(toolkit);
file = string(tempname) + ".json";
writelines(scenarioText, file, Encoding="UTF-8");
P = satflight.runScenario(file, Duration=duration, Step=step);
end

function D = satflightCompare(T, P)
%SATFLIGHTCOMPARE Position difference (km) between this script's run T and the simulator's P.
names = intersect(unique(T.name, "stable"), unique(P.name, "stable"), "stable");
onlySim = setdiff(unique(P.name, "stable"), names, "stable");
if ~isempty(onlySim)
    fprintf("Only in the simulator's run (not reproduced in MATLAB): %s\n", strjoin(onlySim, ", "));
end
% max_diff_km is over the samples where both runs still have the satellite in orbit. One
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
    if k <= 12
        plot(ax, a.t_s(ia(both)) / 3600, max(dr(both), 1e-9) * 1000, DisplayName=names(k));
    end
end
ax.YScale = "log";
grid(ax, "on");
xlabel(ax, "Time since epoch (h)"); ylabel(ax, "Position difference (m)");
title(ax, "This script vs the simulator");
legend(ax, Location="best", Interpreter="none");
end
