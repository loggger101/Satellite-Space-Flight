function out = runPython(args)
%RUNPYTHON Run the simulator's Python with ARGS (a string array) and return its output.
%   The interpreter is the SATFLIGHT_PYTHON environment variable, or "python"
%   on the PATH. This repository's checkout goes first on PYTHONPATH, so an
%   installed copy of satflight cannot shadow it. A non-zero exit status is an
%   error carrying everything the program printed.
exe = string(getenv("SATFLIGHT_PYTHON"));
if exe == ""
    exe = "python";
end
root = string(satflight.repoRoot());
args = string(args);
cmd = strjoin(compose('"%s"', [exe, reshape(args, 1, [])]), " ");

oldPath = getenv("PYTHONPATH");
oldEnc = getenv("PYTHONIOENCODING");
restore = onCleanup(@() setenvs(oldPath, oldEnc));
if oldPath == ""
    setenv("PYTHONPATH", root);
else
    setenv("PYTHONPATH", root + pathsep + oldPath);
end
setenv("PYTHONIOENCODING", "utf-8");
[status, out] = system(cmd);
if status ~= 0
    error("satflight:python", ...
        "The simulator refused or failed (exit status %d):\n%s\n\nCommand: %s\n" + ...
        "(Set the SATFLIGHT_PYTHON environment variable if this is not the Python " + ...
        "that has numpy and the simulator's requirements.)", status, strtrim(out), cmd);
end
end

function setenvs(oldPath, oldEnc)
setenv("PYTHONPATH", oldPath);
setenv("PYTHONIOENCODING", oldEnc);
end
