function orbit = orbitPrecise(mission)
%ORBITPRECISE A Sentinel-1 unit's twin: started on its precise orbit, then flown.
%   The satellite starts where ESA's precise orbit (POEORB) file puts the unit at
%   the scenario's epoch and is flown by the scenario's forces like any other.
%   Beside an orbitReplay of the same unit, its gap to the truth is measured.
%   satflight.orbitPrecise("S1A")
arguments
    mission (1,1) string {mustBeMember(mission, ["S1A", "S1B", "S1C", "S1D"])}
end
orbit = struct("type", "precise", "mission", char(mission));
end
