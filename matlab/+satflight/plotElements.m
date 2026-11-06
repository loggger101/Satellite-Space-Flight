function fig = plotElements(T, opts)
%PLOTELEMENTS Osculating orbital elements against time, one panel per element.
%   fig = satflight.plotElements(T) shows semi-major axis, eccentricity,
%   inclination, RAAN, argument of perigee and altitude. RAAN and argument of
%   perigee are unwrapped (no jump at 360 deg), so J2 drift shows as a slope.
%   Options: Names (which satellites).
arguments
    T table
    opts.Names = string.empty
end
names = pickNames(T, opts.Names);
panels = ["a_km", "Semi-major axis (km)"; "e", "Eccentricity"; "i_deg", "Inclination (deg)"; ...
    "raan_deg", "RAAN, unwrapped (deg)"; "argp_deg", "Arg. of perigee, unwrapped (deg)"; ...
    "alt_km", "Altitude (km)"];
fig = figure(Name="Orbital elements");
layout = tiledlayout(fig, 2, 3, TileSpacing="compact");
title(layout, "Osculating elements");
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    hold(ax, "on");
    for n = names
        rows = T(T.name == n, :);
        y = rows.(panels(p, 1));
        if ismember(panels(p, 1), ["raan_deg", "argp_deg"])
            y = rad2deg(unwrap(deg2rad(y)));
        end
        plot(ax, rows.t_s / 3600, y, LineWidth=1.1, DisplayName=n);
    end
    grid(ax, "on");
    xlabel(ax, "Time (h)");
    ylabel(ax, panels(p, 2));
end
legend(ax, Location="best", Interpreter="none");
end
