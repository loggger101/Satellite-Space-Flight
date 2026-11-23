function S = readStudy(folder)
%READSTUDY Read the Sentinel-1 year study's CSVs into tables.
%   S = satflight.readStudy() reads docs/analysis/sentinel1 in this checkout;
%   S = satflight.readStudy(folder) reads another folder that
%   tools/sentinel1_study.py wrote. S has four tables:
%
%     daily     one row per satellite per day: mission, date, revs,
%               burn_revs, gap_revs, revolution-mean a_km, e, argp_deg,
%               i_deg, ltan_h, and the ground track's track_rms_m and
%               track_max_m (east-west offset of the equator crossings)
%     phasing   one row per day: how far 1C runs behind 1A, mean_deg,
%               min_deg, max_deg
%     burns     every manoeuvre window (kind "burn") and every control window
%               midway between them where nothing burned (kind "placebo"),
%               sized from the jump in the mean elements; note says why a
%               window could not be sized
%     arcs      one row per weekly accuracy arc: the one-day fit, the fitted
%               drag, and the prediction error for three days (R/S/W, m);
%               error says why an arc could not be run
%
%   Dates and times are datetimes in UTC. A blank number is NaN and a blank
%   note or error is "". The daily tables are required; phasing, burns and
%   arcs are empty tables when the study has not written them yet. A file
%   whose columns differ from the study's is refused, not guessed at.
%
%   save("study.mat", "-struct", "S") keeps the tables as a MAT file.
arguments
    folder (1,1) string = fullfile(satflight.repoRoot(), "docs", "analysis", "sentinel1")
end
daily = dir(fullfile(folder, "daily_*.csv"));
if isempty(daily)
    error("satflight:noFile", "No daily_*.csv in %s: run tools/sentinel1_study.py first.", ...
        folder);
end
dailyCols = ["date", "revs", "burn_revs", "gap_revs", "a_km", "e", "argp_deg", "i_deg", ...
    "ltan_h", "track_rms_m", "track_max_m"];
parts = cell(numel(daily), 1);
for k = 1:numel(daily)
    T = readOne(fullfile(folder, daily(k).name), dailyCols, "date", "date", strings(1, 0));
    mission = extractBetween(string(daily(k).name), "daily_", ".csv");
    T = addvars(T, repmat(mission, height(T), 1), Before=1, NewVariableNames="mission");
    parts{k} = T;
end
S.daily = vertcat(parts{:});
S.phasing = readOne(fullfile(folder, "phasing.csv"), ["date", "mean_deg", "min_deg", ...
    "max_deg"], "date", "date", strings(1, 0));
S.burns = readOne(fullfile(folder, "burns.csv"), ["mission", "kind", "start_utc", ...
    "end_utc", "windows", "da_m", "di_mdeg", "de_1e6", "dv_along_mm_s", "dv_cross_mm_s", ...
    "revs_before", "revs_after", "note"], ["start_utc", "end_utc"], "time", ...
    ["mission", "kind", "note"]);
S.arcs = readOne(fullfile(folder, "arcs.csv"), ["mission", "start", "fit_rms_m", ...
    "fit_r_m", "fit_s_m", "fit_w_m", "cd_am", "density_scale", "burns_fit", ...
    "burns_check", "iterations", "fit_hours", "check_hours", "day1_r_m", "day1_s_m", ...
    "day1_w_m", "day2_r_m", "day2_s_m", "day2_w_m", "day3_r_m", "day3_s_m", "day3_w_m", ...
    "at24h_s_m", "at48h_s_m", "at72h_s_m", "seconds", "error"], "start", "date", ...
    ["mission", "error"]);
end

function T = readOne(file, expected, times, timeKind, texts)
% one study CSV with exactly the columns expected; times are datetime columns
% ("date": yyyy-MM-dd, "time": yyyy-MM-ddTHH:mm:ss), texts string columns, the
% rest numbers
types = repmat("double", 1, numel(expected));
types(ismember(expected, [times, texts])) = "string";
if ~isfile(file)
    T = table(Size=[0 numel(expected)], VariableTypes=types, VariableNames=expected);
    for c = times
        T.(c) = NaT(0, 1, TimeZone="UTC");
    end
    return
end
fid = fopen(file, "r", "n", "UTF-8");
header = string(fgetl(fid));
fclose(fid);
found = split(strip(header), ",")';
if ~isequal(found, expected)
    error("satflight:badStudy", ...
        "%s does not have the study's columns.\nExpected: %s\nFound:    %s", ...
        file, strjoin(expected, ","), strjoin(found, ","));
end
opts = delimitedTextImportOptions("NumVariables", numel(expected), "Delimiter", ",", ...
    "DataLines", [2 Inf], "VariableNamesLine", 1, "Encoding", "UTF-8");
opts.VariableNames = expected;
opts.VariableTypes = types;
opts = setvaropts(opts, [times, texts], "QuoteRule", "remove", "FillValue", "");
opts.ExtraColumnsRule = "error";
opts.ImportErrorRule = "error";
opts.MissingRule = "fill";
T = readtable(file, opts);
format = "yyyy-MM-dd";
if timeKind == "time"
    format = "yyyy-MM-dd'T'HH:mm:ss";
end
for c = times
    t = datetime(T.(c), InputFormat=format, TimeZone="UTC");
    if any(isnat(t))
        error("satflight:badStudy", "%s: unreadable %s on row %d.", file, c, ...
            find(isnat(t), 1) + 1);
    end
    T.(c) = t;
end
end
