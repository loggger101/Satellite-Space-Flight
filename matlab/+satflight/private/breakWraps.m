function [lon, lat] = breakWraps(lon, lat)
%BREAKWRAPS Insert NaN where a track crosses the date line, so it is not drawn across the map.
lon = reshape(lon, [], 1);
lat = reshape(lat, [], 1);
jumps = find(abs(diff(lon)) > 180);
for k = numel(jumps):-1:1
    j = jumps(k);
    lon = [lon(1:j); NaN; lon(j + 1:end)];
    lat = [lat(1:j); NaN; lat(j + 1:end)];
end
end
