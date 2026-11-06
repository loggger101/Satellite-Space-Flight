function path = saveScenario(sc, path)
%SAVESCENARIO Write a scenario struct as a JSON file the simulator opens.
%   path = satflight.saveScenario(sc, path) creates the folder if needed and
%   returns the full path. Save to fullfile(satflight.repoRoot, "scenarios",
%   "user", ...) to find the file in the app's Open list, and run
%   satflight.checkScenario on it: the simulator is what decides whether every
%   value is usable. A NaN or Inf is refused here, before anything is written.
arguments
    sc (1,1) struct
    path (1,1) string
end
sc = normalizeScenario(sc);
text = jsonencode(sc, PrettyPrint=true);
folder = fileparts(path);
if folder ~= "" && ~isfolder(folder)
    mkdir(folder);
end
[fid, msg] = fopen(path, "w", "n", "UTF-8");
if fid < 0
    error("satflight:write", "Cannot write %s: %s", path, msg);
end
closer = onCleanup(@() fclose(fid));
fwrite(fid, text, "char");
fwrite(fid, newline, "char");
clear closer
path = string(fullpathOf(path));
end

function p = fullpathOf(p)
info = dir(p);
p = fullfile(info.folder, info.name);
end
