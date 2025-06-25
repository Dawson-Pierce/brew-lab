clear; clc; close all 

% --- Constants ---
mu = 3.986004418e14; % Earth's gravitational parameter, m^3/s^2
Re = 6371e3;         % Earth's radius, m
h = 600e3;           % Altitude above surface, m
r0 = Re + h;         % Initial orbital radius, m
v0 = sqrt(mu/r0);    % Circular orbit velocity, m/s

% --- Initial State (ECI) ---
% Allows for arbitrary angle in xy-plane
state0 = [r0; 0; 0; 0; v0; 0]; % [x; y; z; vx; vy; vz]

% --- Define continuous dynamics ---
f = @(t, x) [x(4:6); -mu/norm(x(1:3))^3 * x(1:3)];

custom_dyn = BREW.dynamics.ContinuousDynamics('f', f);

% --- Simulation Parameters ---
dt = 1;               % timestep in seconds
num_steps = 400;     % number of steps (~1 orbital period for LEO)
states = zeros(6, num_steps);
states(:,1) = state0;

% --- Propagate Orbit Using ode45 ---
for k = 2:num_steps
    states(:,k) = custom_dyn.propagateState(k, dt, states(:,k-1));
end

x = states(1,:); 
y = states(2,:); 
z = states(3,:);

% --- Convert ECI to Geodetic Coordinates (treat ECI ~ ECEF for short duration) ---
lla = ecef2lla([x', y', z']);
lat = lla(:,1); lon = lla(:,2); alt = lla(:,3);

% % --- Plot ---
f = uifigure;
g = geoglobe(f);
% geoplot3(g, lat, lon, alt, 'r', 'HeightReference', 'ellipsoid', 'LineStyle','none','Marker','o', 'LineWidth', 2); 

target_motion = BREW.dynamics.common.Integrator_3D();

alphas = {10,10};
betas = {1,1}; 
means = {[0; 10; 1000; 0; 0.1; 0],[0; 20; 1000; 0; 0.05; 0]}; % in lla, so it's trippy. Arbitrarily set. 
covariances = {eye(3), eye(3)}; 
IWdofs = {1500, 1500}; 
IWshapes = {[0.1 0 0; 0 0.5 0; 0 0 0.1],[0.1 0 0; 0 0.5 0; 0 0 0.1]}; 
weights = [1,1];

mix = BREW.distributions.GGIWMixture( ...
    'alphas', alphas, ...
    'betas', betas, ...
    'means', means, ...
    'covariances', covariances, ...
    'IWdofs', IWdofs, ...
    'IWshapes', IWshapes, ...
    'weights', weights);

meas_points = geoplot3(g,0,0,0,'r','Marker','o','LineStyle','none');

gifFilename = 'SatelliteDetections.gif';

% --- Animate camera following the satellite ---
for k = 1:5:num_steps

    new_points = mix.sample_measurements();

    for kk = 1:length(mix.distributions)
        mean_temp = mix.means{kk};
        mix.means{kk} = target_motion.propagateState([],dt,mean_temp);
    end

    meas_points.LatitudeData = new_points(1,:);
    meas_points.LongitudeData = new_points(2,:);
    meas_points.HeightData = new_points(3,:);

    % Camera position at current satellite position
    campos(g, lat(k), lon(k), alt(k));  
    
    % Compute Earth-relative pitch and heading (optional)
    % Vector to Earth center in ECEF is just -[x;y;z]
    r = [x(k); y(k); z(k)];
    v = [states(4,k); states(5,k); states(6,k)];

    % Rough direction pointing to velocity vector (flattened to horizontal plane for heading)
    east = cross([0; 0; 1], r); % east direction at that point
    north = cross(r, east);    % north direction (completes tangent frame)

    % Normalize
    east = east / norm(east);
    north = north / norm(north);

    % Project velocity into tangent plane to get local heading
    v_proj = [dot(v, east); dot(v, north)];
    heading = atan2d(v_proj(1), v_proj(2)); % angle from north

    % Pitch is angle between velocity vector and horizontal plane
    pitch = asind(dot(v, r) / (norm(v) * norm(r)));

    % Update camera orientation
    % camheading(g, heading);
    % campitch(g, -pitch); % negative to look down from behind 

    % campitch(g,-70)
    camheading(g,50)

    drawnow;

    % % capture the frame as an RGB image
    % frame = getframe(f);
    % img   = frame2im(frame);
    % 
    % % convert to an indexed image with a fixed 256‐color map
    % [A, map] = rgb2ind(img, 256);
    % 
    % % write to GIF: first frame creates the file, subsequent frames append
    % if k == 1
    %     imwrite(A, map, gifFilename, 'gif', ...
    %             'LoopCount', Inf, ...      % make it loop forever
    %             'DelayTime', 0.1);         % seconds between frames
    % else
    %     imwrite(A, map, gifFilename, 'gif', ...
    %             'WriteMode', 'append', ...
    %             'DelayTime', 0.1);
    % end
    % 
    % pause(0.05);
end
