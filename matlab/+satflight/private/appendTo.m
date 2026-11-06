function sc = appendTo(sc, list, item)
%APPENDTO Add ITEM to the scenario list LIST (a cell row), creating it if absent.
if ~isstruct(sc) || ~isscalar(sc)
    error("satflight:badScenario", ...
        "Expected a scenario struct from satflight.newScenario or satflight.loadScenario.");
end
if ~isfield(sc, list)
    sc.(list) = {};
end
sc.(list) = asCellRow(sc.(list), list);
sc.(list){end + 1} = item;
end
