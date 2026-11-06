%% Analyse a simulator run: J2 nodal regression, measured against theory
% Runs the bundled j2_precession scenario for three days, then fits the
% drift of each satellite's ascending node (RAAN) and compares it with the
% secular J2 rate
%
%   dRAAN/dt = -3/2 n J2 (Re/p)^2 cos(i)
%
% A CSV exported from the app (Ctrl+O > Export CSV) works the same way:
% T = satflight.readRun("path/to/export.csv").

addpath(fullfile(fileparts(mfilename("fullpath")), ".."));

scenario = fullfile(satflight.repoRoot, "scenarios", "j2_precession.json");
T = satflight.runScenario(scenario, Duration="3d", Step=600);

MU = 398600.4418;       % km^3/s^2   (satflight/constants.py)
RE = 6378.137;          % km
J2 = 1.08262668355e-3;

results = table();
for name = unique(T.name, "stable")'
    rows = T(T.name == name, :);
    i = mean(rows.i_deg);
    if i < 1
        continue        % an equatorial orbit has no ascending node to follow
    end
    days = rows.t_s / 86400;
    raan = rad2deg(unwrap(deg2rad(rows.raan_deg)));
    fit = polyfit(days, raan, 1);
    a = mean(rows.a_km);        % orbit-averaged: close to the mean element
    e = mean(rows.e);
    n = sqrt(MU / a^3);                                    % rad/s
    theory = rad2deg(-1.5 * n * J2 * (RE / (a * (1 - e^2)))^2 * cosd(i)) * 86400;
    results = [results; table(name, i, fit(1), theory, fit(1) - theory, ...
        VariableNames=["name", "i_deg", "measured_deg_day", "theory_deg_day", ...
        "difference_deg_day"])]; %#ok<AGROW>
end
disp(results);

satflight.plotElements(T);
