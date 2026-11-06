function names = pickNames(T, requested)
%PICKNAMES Satellite names to plot: REQUESTED (all must be in T), or every one in T.
names = unique(T.name, "stable")';
if isempty(names)
    error("satflight:emptyRun", "The ephemeris has no rows.");
end
if ~isempty(requested)
    requested = reshape(string(requested), 1, []);
    unknown = setdiff(requested, names, "stable");
    if ~isempty(unknown)
        error("satflight:unknownSatellite", "No satellite named %s in this run (it has %s).", ...
            strjoin(unknown, ", "), strjoin(names, ", "));
    end
    names = requested;
end
end
