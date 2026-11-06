function fig = plotAltitude(T, opts)
%PLOTALTITUDE Altitude above the ellipsoid against time for each satellite.
%   fig = satflight.plotAltitude(T, Names=["ISS" "Hubble"])
arguments
    T table
    opts.Names = string.empty
end
names = pickNames(T, opts.Names);
fig = figure(Name="Altitude");
ax = axes(fig);
hold(ax, "on");
for n = names
    rows = T(T.name == n, :);
    plot(ax, rows.t_s / 3600, rows.alt_km, LineWidth=1.2, DisplayName=n);
end
grid(ax, "on");
xlabel(ax, "Time since epoch (h)");
ylabel(ax, "Altitude (km)");
title(ax, "Altitude");
legend(ax, Location="best", Interpreter="none");
end
