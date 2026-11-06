function T = readRun(file)
%READRUN Read an ephemeris CSV written by the simulator into a table.
%   T = satflight.readRun(file) reads the CSV that the batch runner
%   (python -m satflight.batch ... --out file, or satflight.runScenario) and
%   the app's Scenarios > Export CSV write. One row per satellite per sample:
%
%     t_s                 seconds since the scenario epoch
%     utc                 datetime (UTC)
%     name, status        satellite name and state ("active", "decayed", ...)
%     x_km .. vz_kms      ECI position (km) and velocity (km/s)
%     alt_km, lat_deg, lon_deg   geodetic altitude and position
%     a_km, e, i_deg, raan_deg, argp_deg, nu_deg   osculating elements
%
%   A file whose columns differ from these is refused, not guessed at.
arguments
    file (1,1) string
end
expected = ["t_s", "utc", "name", "status", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", ...
    "vz_kms", "alt_km", "lat_deg", "lon_deg", "a_km", "e", "i_deg", "raan_deg", "argp_deg", ...
    "nu_deg"];
if ~isfile(file)
    error("satflight:noFile", "No ephemeris file %s.", file);
end
opts = delimitedTextImportOptions("NumVariables", numel(expected), "Delimiter", ",", ...
    "DataLines", [2 Inf], "VariableNamesLine", 1, "Encoding", "UTF-8");
opts.VariableNames = expected;
opts.VariableTypes = repmat("double", 1, numel(expected));
opts = setvartype(opts, ["utc", "name", "status"], "string");
opts.ExtraColumnsRule = "error";
opts.ImportErrorRule = "error";
opts.MissingRule = "error";

fid = fopen(file, "r", "n", "UTF-8");
header = string(fgetl(fid));
fclose(fid);
found = split(strip(header), ",")';
if ~isequal(found, expected)
    error("satflight:badEphemeris", ...
        "%s does not have the simulator's ephemeris columns.\nExpected: %s\nFound:    %s", ...
        file, strjoin(expected, ","), strjoin(found, ","));
end
T = readtable(file, opts);
T.utc = datetime(T.utc, InputFormat="yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone="UTC");
if any(isnat(T.utc))
    error("satflight:badEphemeris", "%s: unreadable utc time on row %d.", file, ...
        find(isnat(T.utc), 1) + 1);
end
T.Properties.VariableUnits = ["s", "", "", "", "km", "km", "km", "km/s", "km/s", "km/s", ...
    "km", "deg", "deg", "km", "", "deg", "deg", "deg", "deg"];
end
