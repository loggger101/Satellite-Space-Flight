function root = repoRoot()
%REPOROOT Folder of the Satellite Space Flight checkout this toolkit belongs to.
%   root = satflight.repoRoot() is the folder holding satflight/, scenarios/
%   and matlab/. Bundled scenarios are in fullfile(root, "scenarios"); save
%   your own to fullfile(root, "scenarios", "user") to find them in the app's
%   Open list.
root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
if ~isfolder(fullfile(root, "satflight")) || ~isfolder(fullfile(root, "scenarios"))
    error("satflight:noRepo", ...
        "%s is not a Satellite Space Flight checkout (no satflight/ or scenarios/ folder). " + ...
        "Keep the matlab/ folder inside the repository.", root);
end
end
