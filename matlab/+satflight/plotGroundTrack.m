function fig = plotGroundTrack(T, opts)
%PLOTGROUNDTRACK Ground tracks of a run on a longitude-latitude map with coastlines.
%   fig = satflight.plotGroundTrack(T) plots every satellite in the table
%   from satflight.readRun / runScenario.
%   Options: Names (which satellites), Scenario (a scenario struct or file:
%   its ground stations are marked).
arguments
    T table
    opts.Names = string.empty
    opts.Scenario = []
end
names = pickNames(T, opts.Names);
fig = figure(Name="Ground track");
ax = axes(fig);
hold(ax, "on");
box(ax, "on");

coastFile = fullfile(satflight.repoRoot, "assets", "coastlines.json");
coast = jsondecode(fileread(coastFile));
if ~iscell(coast)       % polylines of unequal length decode to a cell of n-by-2 arrays
    error("satflight:badAsset", "%s is not a list of [lon, lat] polylines.", coastFile);
end
lines = cellfun(@(c) [c; NaN NaN], coast, UniformOutput=false);
lines = vertcat(lines{:});
plot(ax, lines(:, 1), lines(:, 2), Color=[0.55 0.55 0.55], LineWidth=0.5, ...
    HandleVisibility="off");

for n = names
    rows = T(T.name == n, :);
    [lon, lat] = breakWraps(rows.lon_deg, rows.lat_deg);
    h = plot(ax, lon, lat, LineWidth=1.2, DisplayName=n);
    plot(ax, rows.lon_deg(end), rows.lat_deg(end), "o", MarkerFaceColor=h.Color, ...
        MarkerEdgeColor="k", HandleVisibility="off");
end

if ~isempty(opts.Scenario)
    sc = opts.Scenario;
    if ~isstruct(sc)
        sc = satflight.loadScenario(sc);
    end
    stations = {};
    if isfield(sc, "stations")
        stations = asCellRow(sc.stations, "stations");
    end
    for k = 1:numel(stations)
        g = stations{k};
        plot(ax, g.lon, g.lat, "^", MarkerSize=8, MarkerFaceColor="k", ...
            MarkerEdgeColor="w", HandleVisibility="off");
        text(ax, g.lon + 2, g.lat, string(g.name), FontSize=8);
    end
end

xlim(ax, [-180 180]);
ylim(ax, [-90 90]);
daspect(ax, [1 1 1]);
xticks(ax, -180:60:180);
yticks(ax, -90:30:90);
grid(ax, "on");
xlabel(ax, "Longitude (deg)");
ylabel(ax, "Latitude (deg)");
fmt = "yyyy-MM-dd HH:mm";
title(ax, sprintf("Ground track, %s to %s UTC", string(T.utc(1), fmt), string(T.utc(end), fmt)));
legend(ax, Location="southoutside", Orientation="horizontal", Interpreter="none");
end
