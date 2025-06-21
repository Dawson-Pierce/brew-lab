% Test script for EKF tracking a GGIW truth model in 3D

clear; close all; clc;

% --- Truth Model Initialization (same as test_GGIW_extent_rotation) ---
alpha = 100;
beta = 1;
meanVal = [0; 0; 0; 0; 0; 0];
covVal = 0.1 * eye(length(meanVal));
IWdof = 10;
IWshape = [4 1 0.5; 1 4 0.3; 0.5 0.3 0.2];
GGIW_truth = BREW.distributions.GGIW(alpha, beta, meanVal, covVal, IWdof, IWshape);

% --- Motion Model ---
dt = 0.2;
num_steps = 100;
motion = BREW.dynamics.common.Integrator_3D();

% --- Measurement Model ---
H = [eye(3), zeros(3, length(meanVal)-3)]; % Extract x, y, z
R = 0.05 * eye(3); % Measurement noise

% --- EKF Initialization ---
process_noise = 0.01 * eye(length(meanVal));
GGIW_init = BREW.distributions.GGIW(alpha, beta, meanVal, covVal, IWdof, IWshape); % Same as truth for simplicity
EKF = BREW.filters.ExtendedKalmanFilter('dyn_obj', motion, 'H', H);
EKF.setProcessNoise(process_noise);
EKF.setMeasurementNoise(R);

% --- Plot Setup ---
figure;
ax = axes;
axis equal;
grid on;
hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('EKF Tracking GGIW Truth Model');
view(3);
axis(ax, [-5 5 -5 5 -5 5]);

% --- Initial Plots ---
GGIW_truth.plot_distribution(ax, 1:3, 0.95, 'w');

gifFilename = 'Tracker.gif';

% --- Main Loop ---
GGIW_est = GGIW_init;
for k = 1:num_steps
    cla(ax);
    % --- Propagate Truth ---
    GGIW_truth.IWshape = motion.propagate_extent(dt, GGIW_truth.mean, GGIW_truth.IWshape);
    if k == round(num_steps / 2)
        GGIW_truth.mean = motion.propagateState([], dt, GGIW_truth.mean, rand(3,1));
    else
        GGIW_truth.mean = motion.propagateState([], dt, GGIW_truth.mean);
    end
    
    % --- Generate Measurement ---
    meas = GGIW_truth.sample_measurements([1 2 3]);
    
    % --- EKF Predict ---
    GGIW_pred = EKF.predict([], dt, GGIW_est, []);
    % --- EKF Correct ---
    GGIW_est = EKF.correct(dt, meas, GGIW_pred);
    
    % --- Plot Truth Extent ---
    % GGIW_truth.plot_distribution(ax, 1:3, 0.95, 'w');
    % --- Plot EKF Estimate Extent ---
    GGIW_est.plot_distribution(ax, 1:3, 0.95, 'r');
    % --- Plot Measurements ---
    scatter3(meas(1,:), meas(2,:), meas(3,:), 'w*'); hold on
    % --- Plot EKF Mean ---
    plot3(ax, GGIW_est.mean(1), GGIW_est.mean(2), GGIW_est.mean(3), 'ro', 'MarkerFaceColor', 'r');
    % --- Plot Truth Mean ---
    plot3(ax, GGIW_truth.mean(1), GGIW_truth.mean(2), GGIW_truth.mean(3), 'wo', 'MarkerFaceColor', 'w');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('EKF Tracking GGIW Truth (Step %d/%d)', k, num_steps));
    
    axis(ax, [GGIW_truth.mean(1)-5  GGIW_truth.mean(1)+5 GGIW_truth.mean(2)-5  GGIW_truth.mean(2)+5 GGIW_truth.mean(3)-5  GGIW_truth.mean(3)+5]);

    drawnow;

    % capture the frame as an RGB image
    frame = getframe(gcf);
    img   = frame2im(frame);

    % convert to an indexed image with a fixed 256‐color map
    [A, map] = rgb2ind(img, 256);

    % write to GIF: first frame creates the file, subsequent frames append
    if k == 1
        imwrite(A, map, gifFilename, 'gif', ...
                'LoopCount', Inf, ...      % make it loop forever
                'DelayTime', 0.1);         % seconds between frames
    else
        imwrite(A, map, gifFilename, 'gif', ...
                'WriteMode', 'append', ...
                'DelayTime', 0.1);
    end

    pause(0.05);
end 