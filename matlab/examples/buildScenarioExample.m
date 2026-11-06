%% Build a scenario in MATLAB, check it, run it and plot the result
% Run from anywhere: this adds the toolkit to the path. The scenario is saved
% to scenarios/user/, so it also appears in the app's Open list (Ctrl+O).

addpath(fullfile(fileparts(mfilename("fullpath")), ".."));

%% Describe the run
sc = satflight.newScenario("MATLAB demo", ...
    Description="Built with the MATLAB toolkit: a sun-synchronous imager, a Molniya " + ...
        "relay, a small Walker shell and a Svalbard ground station.", ...
    Epoch=datetime(2026, 9, 29, 12, 0, 0, TimeZone="UTC"), Drag=true, Warp=120);

sc = satflight.addSatellite(sc, "Imager", ...
    satflight.orbitElements(Altitude=550, I="sso", Raan=30), Mass=150, Area=1.2);
sc = satflight.addSatellite(sc, "Molniya", ...
    satflight.orbitElements(PerigeeAlt=600, ApogeeAlt=39700, I=63.4, Argp=270, Raan=60));
sc = satflight.addConstellation(sc, "Shell", Altitude=1200, Inclination=55, ...
    Total=12, Planes=3, Phasing=1);
sc = satflight.addStation(sc, "Svalbard", 78.23, 15.39, MinEl=5);

% raise the imager by 10 m/s one hour in, along the velocity (VNB frame)
sc = satflight.addManeuver(sc, "Imager", "impulse", Timing="delay", Delay=3600, ...
    Dv=[0.010 0 0], Label="+10 m/s prograde");

%% Save it and let the simulator check it
path = satflight.saveScenario(sc, fullfile(satflight.repoRoot, "scenarios", "user", ...
    "matlab_demo.json"));
fprintf("Saved %s\n\n", path);
disp(satflight.checkScenario(path));

%% Propagate 12 hours and plot
[T, report] = satflight.runScenario(path, Duration="12h", Step=60);
disp(report);

satflight.plotGroundTrack(T, Names=["Imager", "Molniya"], Scenario=sc);
satflight.plotAltitude(T, Names="Imager");
satflight.plotElements(T, Names="Molniya");
satflight.plotOrbit3D(T);
