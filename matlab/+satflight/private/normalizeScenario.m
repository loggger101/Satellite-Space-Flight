function sc = normalizeScenario(sc)
%NORMALIZESCENARIO Lists as cell rows, nulls removed, every number finite.
%   A field holding [] is a JSON null (jsondecode's reading of one, or an
%   unset value); it is removed so the simulator's default (None) applies.
%   A NaN or Inf anywhere is an error naming where it is.
if ~isstruct(sc) || ~isscalar(sc)
    error("satflight:badScenario", "Expected a scenario struct, got a %s.", class(sc));
end
if ~isfield(sc, "name")
    error("satflight:badScenario", "The scenario has no name (start it with satflight.newScenario).");
end
for list = ["satellites", "constellations", "stations", "maneuvers", "launches"]
    if isfield(sc, list)
        sc.(list) = asCellRow(sc.(list), list);
    end
end
if isfield(sc, "launches")
    for k = 1:numel(sc.launches)
        if isfield(sc.launches{k}, "vehicle") && isstruct(sc.launches{k}.vehicle) ...
                && isfield(sc.launches{k}.vehicle, "stages")
            sc.launches{k}.vehicle.stages = asCellRow(sc.launches{k}.vehicle.stages, ...
                sprintf("launches{%d} stages", k));
        end
    end
end
sc = clean(sc, "scenario");
end

function x = clean(x, where)
if isstruct(x)
    if ~isscalar(x)
        error("satflight:badScenario", "%s: a list of objects must be a cell array.", where);
    end
    for f = string(fieldnames(x))'
        v = x.(f);
        if isnumeric(v) && isempty(v)
            x = rmfield(x, f);
        else
            x.(f) = clean(v, where + "." + f);
        end
    end
elseif iscell(x)
    for k = 1:numel(x)
        x{k} = clean(x{k}, sprintf("%s{%d}", where, k));
    end
elseif isnumeric(x)
    if ~isreal(x) || any(~isfinite(x), "all")
        error("satflight:badScenario", "%s is %s; every number must be finite and real.", ...
            where, mat2str(x));
    end
elseif isstring(x) && isscalar(x)
    x = char(x);
end
end
