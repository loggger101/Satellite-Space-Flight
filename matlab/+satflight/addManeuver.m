function sc = addManeuver(sc, sat, kind, opts)
%ADDMANEUVER Schedule a burn for satellite SAT.
%   KIND: "impulse", "circularize", "plane_change", "match_velocity" or "finite".
%
%   When (Timing): "now", "delay" (Delay seconds after the start),
%   "periapsis", "apoapsis", "ascending_node", "descending_node", or
%   "absolute" (T seconds after the epoch).
%
%   What: Dv ([3] km/s in Frame "VNB", "RSW" or "ECI"; a direction for
%   "finite"), DeltaI (deg, plane_change), Target (satellite to match for
%   match_velocity), Thrust (N), Isp (s) and Duration (s) for a finite burn,
%   Label (shown in the event log).
%
%   sc = satflight.addManeuver(sc, "Transfer", "impulse", Timing="absolute", ...
%       T=600, Dv=[2.425 0 0], Label="Hohmann burn 1");
arguments
    sc (1,1) struct
    sat (1,1) string
    kind (1,1) string
    opts.Timing (1,1) string
    opts.T (1,1) double
    opts.Delay (1,1) double
    opts.Dv (1,3) double
    opts.Frame (1,1) string
    opts.DeltaI (1,1) double
    opts.Target (1,1) string
    opts.Thrust (1,1) double
    opts.Isp (1,1) double
    opts.Duration (1,1) double
    opts.Label (1,1) string
end
m = struct("sat", char(sat), "kind", char(kind));
m = copyOptions(m, opts, ["Timing", "timing"; "T", "t"; "Delay", "delay"; "Dv", "dv"; ...
    "Frame", "frame"; "DeltaI", "delta_i"; "Target", "target"; "Thrust", "thrust"; ...
    "Isp", "isp"; "Duration", "duration"; "Label", "label"]);
sc = appendTo(sc, "maneuvers", m);
end
