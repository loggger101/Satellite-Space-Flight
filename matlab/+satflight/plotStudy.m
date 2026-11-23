function figs = plotStudy(S, opts)
%PLOTSTUDY The Sentinel-1 year study's figures from readStudy's tables.
%   figs = satflight.plotStudy(satflight.readStudy())
%   figs = satflight.plotStudy(S, Figures="accuracy")
%
%   The same three figures as the study's report (docs/analysis/sentinel1):
%     "elements"  revolution-mean a, inclination, LTAN and eccentricity per
%                 day, with the burns ticked on the a axis
%     "track"     the ground track's daily RMS offset (days within 1 km of
%                 the usual track) and the 1A-1C phasing
%     "accuracy"  the weekly arcs' prediction error after a one-day fit and
%                 the drag the fit needed
%   A line breaks over a gap in the data instead of bridging it.
arguments
    S (1,1) struct
    opts.Figures (1,:) string {mustBeMember(opts.Figures, ["elements", "track", ...
        "accuracy"])} = ["elements", "track", "accuracy"]
end
colours = dictionary(["S1A", "S1C"], {[42 111 219] / 255, [217 98 43] / 255});
missions = unique(S.daily.mission)';
figs = gobjects(1, 0);
for which = opts.Figures
    switch which
        case "elements"
            figs(end + 1) = elements(S, missions, colours); %#ok<AGROW>
        case "track"
            figs(end + 1) = track(S, missions, colours); %#ok<AGROW>
        case "accuracy"
            figs(end + 1) = accuracy(S, missions, colours); %#ok<AGROW>
    end
end
end

function fig = elements(S, missions, colours)
fig = figure(Name="Mean elements");
tl = tiledlayout(fig, 4, 1, TileSpacing="compact");
labels = ["mean a - 7000 km", "mean inclination (deg)", "LTAN - 18:00 (min)", ...
    "eccentricity (x1e-3)"];
ax = gobjects(1, 4);
for k = 1:4
    ax(k) = nexttile(tl);
    hold(ax(k), "on");
    grid(ax(k), "on");
    ylabel(ax(k), labels(k));
end
ax(1).Tag = "a";
for m = missions
    D = S.daily(S.daily.mission == m, :);
    style = {"Color", colours{m}, "LineWidth", 1};
    gapped(ax(1), D.date, D.a_km - 7000, [style, {"DisplayName", "Sentinel-1" + extractAfter(m, 2)}]);
    gapped(ax(2), D.date, D.i_deg, style);
    gapped(ax(3), D.date, (D.ltan_h - 18) * 60, style);
    gapped(ax(4), D.date, D.e * 1e3, style);
end
% the station-kept band; a satellite moved off it (1C, June 2026) runs off scale
a = S.daily.a_km - 7000;
band = prctile(a(~isnan(a)), [1 99]);
pad = 0.6 * diff(band);
ylim(ax(1), band + [-pad pad]);
if max(a) > band(2) + pad
    text(ax(1), 0.01, 0.92, sprintf("off scale: up to %+.2f", max(a)), ...
        Units="normalized", FontSize=8);
end
burns = S.burns(S.burns.kind == "burn", :);
for m = missions
    t = dateshift(burns.start_utc(burns.mission == m), "start", "day");
    plot(ax(1), t, repmat(ax(1).YLim(1), size(t)), "|", Color=colours{m}, MarkerSize=8, ...
        Tag="burns", HandleVisibility="off");
end
legend(ax(1), Location="northeast");
title(ax(1), "Revolution-mean elements, daily (burns marked on the a axis)");
linkaxes(ax, "x");
end

function fig = track(S, missions, colours)
fig = figure(Name="Track and phasing");
tl = tiledlayout(fig, 2, 1, TileSpacing="compact");
ax = nexttile(tl);
hold(ax, "on");
grid(ax, "on");
for m = missions
    D = S.daily(S.daily.mission == m, :);
    y = D.track_rms_m;
    y(D.track_max_m > 1000) = NaN;          % moved off the usual track that day
    gapped(ax, D.date, y, {"Color", colours{m}, "LineWidth", 1, ...
        "DisplayName", "Sentinel-1" + extractAfter(m, 2)});
end
ylabel(ax, "node offset RMS (m)");
title(ax, "Ground track: daily RMS east-west offset of the equator crossings " + ...
    "(days within 1 km)", FontSize=10);
legend(ax, Location="northeast");
ax2 = nexttile(tl);
grid(ax2, "on");
hold(ax2, "on");
if height(S.phasing)
    gapped(ax2, S.phasing.date, S.phasing.mean_deg, {"LineWidth", 1});  % theme's colour
end
ylabel(ax2, "1C behind 1A (deg)");
title(ax2, "Phasing");
linkaxes([ax ax2], "x");
end

function fig = accuracy(S, missions, colours)
fig = figure(Name="Prediction accuracy");
tl = tiledlayout(fig, 3, 1, TileSpacing="compact");
ax = gobjects(1, 3);
for k = 1:3
    ax(k) = nexttile(tl);
    hold(ax(k), "on");
    grid(ax(k), "on");
end
set(ax(1), YScale="log");
ok = S.arcs(S.arcs.error == "", :);
for m = missions
    A = ok(ok.mission == m, :);
    if isempty(A)
        continue
    end
    c = colours{m};
    unit = "1" + extractAfter(m, 2);
    plot(ax(1), A.start, A.day1_s_m, "o", Color=c, MarkerSize=4, DisplayName=unit + " day 1");
    plot(ax(1), A.start, A.day3_s_m, "^", Color=c, MarkerSize=4, DisplayName=unit + " day 3");
    plot(ax(2), A.start, A.day1_r_m, "o", Color=c, MarkerSize=4, DisplayName=unit + " radial");
    plot(ax(2), A.start, A.day1_w_m, "x", Color=c, MarkerSize=4, ...
        DisplayName=unit + " cross-track");
    plot(ax(3), A.start, A.density_scale, "o-", Color=c, MarkerSize=3, LineWidth=0.8, ...
        DisplayName="Sentinel-1" + extractAfter(m, 2));
end
ylabel(ax(1), "along-track RMS (m)");
title(ax(1), "Prediction error after a one-day fit");
ylabel(ax(2), "day-1 RMS (m)");
ylabel(ax(3), "fitted density scale");
title(ax(3), "Drag the fit needs, as a scale on the static table (Cd 2.2)");
for k = 1:3
    legend(ax(k), Location="best", NumColumns=2, FontSize=8);
end
linkaxes(ax, "x");
end

function gapped(ax, t, y, style)
% a line that breaks wherever consecutive days are more than a day apart
k = find(diff(t) > days(1)) + 1;
for j = flip(k')
    t = [t(1:j - 1); t(j - 1) + days(1); t(j:end)];
    y = [y(1:j - 1); NaN; y(j:end)];
end
plot(ax, t, y, style{:});
end
