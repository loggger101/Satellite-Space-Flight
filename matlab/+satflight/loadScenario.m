function sc = loadScenario(path)
%LOADSCENARIO Read a scenario JSON file into a struct you can change and save.
%   sc = satflight.loadScenario(fullfile(satflight.repoRoot, "scenarios", "default.json"));
%   sc.satellites{1}.mass = 600;
%   sc = satflight.addStation(sc, "Goldstone", 35.43, -116.89);
%   satflight.saveScenario(sc, fullfile(satflight.repoRoot, "scenarios", "user", "mine.json"));
%
%   Lists (satellites, constellations, stations, maneuvers, launches) are cell
%   arrays of structs, whatever their length. JSON nulls are left out.
arguments
    path (1,1) string
end
sc = jsondecode(fileread(path));
sc = normalizeScenario(sc);
end
