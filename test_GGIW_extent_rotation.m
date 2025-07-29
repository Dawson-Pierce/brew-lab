% Test script for 3D GGIW extent rotation using Integrator_3D_euler 

clear; close all; clc;

% Create a 3D GGIW object
alpha = 1000;
beta = 1;
meanVal = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0.2; 0.1; 0.3]; 
covVal = 0.1 * eye(length(meanVal));
IWdof = 10;
IWshape = [4 1 0.5; 1 4 0.3; 0.5 0.3 0.1]; % nontrivial, non-axis-aligned
GGIWobj = BREW.distributions.GGIW(alpha, beta, meanVal, covVal, IWdof, IWshape);

dt = 0.2; % time step
num_steps = 100;

% Create the motion model
motion = BREW.dynamics.Integrator_3D_euler();

% Set up figure
figure; 
ax = axes;
axis equal;
grid on;
hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Animated GGIW Extent Rotation');
view(3);

% Plot initial extent
GGIWobj.plot_distribution(ax, 1:3, 0.95, 'w');

axis equal
axis(ax, [-5 5 -5 5 -5 5]);

% Animation loop
for k = 1:num_steps
    cla(ax);
    % Plot the current extent
    GGIWobj.IWshape = motion.propagate_extent(GGIWobj.mean,  GGIWobj.IWshape, 'dt', dt);
    if k == round(num_steps / 2)
        GGIWobj.mean = motion.propagateState([], dt, GGIWobj.mean, 1*[0 0 0 rand(1,3)]');
    else
        GGIWobj.mean = motion.propagateState([], dt, GGIWobj.mean);
    end

    GGIWobj.plot_distribution(ax, 1:3, 0.95, 'r');
    meas_temp = GGIWobj.sample_measurements([1 2 3]);

    scatter3(meas_temp(1,:),meas_temp(2,:),meas_temp(3,:),'w*'); hold on

    % Optionally plot the mean point
    plot3(ax, meanVal(1), meanVal(2), meanVal(3), 'wo', 'MarkerFaceColor', 'w');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('Animated GGIW Extent Rotation (Step %d/%d)', k, num_steps)); 
    drawnow;
    pause(0.05);
end 