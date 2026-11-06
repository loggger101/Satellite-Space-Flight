function fig = plotOrbit3D(T, opts)
%PLOTORBIT3D Trajectories in the Earth-centred inertial frame around a to-scale Earth.
%   fig = satflight.plotOrbit3D(T, Names="Transfer")
arguments
    T table
    opts.Names = string.empty
end
R_EARTH = 6378.137;             % km, WGS-84 equatorial radius (satflight.constants)
F_EARTH = 1 / 298.257223563;    % flattening
names = pickNames(T, opts.Names);
fig = figure(Name="Orbits (ECI)");
ax = axes(fig);
hold(ax, "on");
[x, y, z] = sphere(48);
surf(ax, R_EARTH * x, R_EARTH * y, R_EARTH * (1 - F_EARTH) * z, FaceColor=[0.35 0.55 0.85], ...
    EdgeColor=[0.25 0.4 0.65], EdgeAlpha=0.3, FaceAlpha=0.6, HandleVisibility="off");
for n = names
    rows = T(T.name == n, :);
    h = plot3(ax, rows.x_km, rows.y_km, rows.z_km, LineWidth=1.2, DisplayName=n);
    plot3(ax, rows.x_km(end), rows.y_km(end), rows.z_km(end), "o", MarkerFaceColor=h.Color, ...
        MarkerEdgeColor="k", HandleVisibility="off");
end
axis(ax, "equal");
grid(ax, "on");
view(ax, 35, 25);
xlabel(ax, "x (km)");
ylabel(ax, "y (km)");
zlabel(ax, "z (km)");
title(ax, "Trajectories, ECI");
legend(ax, Location="best", Interpreter="none");
end
