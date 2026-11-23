function orbit = orbitReplay(mission, until)
%ORBITREPLAY A Sentinel-1 unit following ESA's precise orbit (POEORB) files.
%   The satellite is not flown: it is where the files put it, from the scenario's
%   epoch to the end of day UNTIL (UTC), and then stops. It cannot burn. The files
%   and the IERS table come from tools/fetch_poeorb.py and tools/fetch_eop.py.
%   satflight.orbitReplay("S1A", datetime(2025, 7, 9))
%   satflight.orbitReplay("S1C", "2025-07-09")
arguments
    mission (1,1) string {mustBeMember(mission, ["S1A", "S1B", "S1C", "S1D"])}
    until (1,1) {mustBeA(until, ["datetime", "string", "char"])}
end
if isdatetime(until)
    until = string(datetime(until, Format="yyyy-MM-dd"));
end
orbit = struct("type", "replay", "mission", char(mission), "until", char(until));
end
