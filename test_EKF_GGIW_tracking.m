% Test script for EKF tracking a GGIW truth model in 3D

clear; close all; clc;

% --- Truth Model Initialization ---
truth = BREW.models.Mixture();
truth.dist_type = "GGIW";
truth.means = {[0; 0; 0; 0; 0; 0]};
truth.covariances = {0.5 * eye(6)};
truth.weights = 1;
truth.alphas = 100;
truth.betas = 1;
truth.vs = 100;
truth.Vs = {[100 1 0.5; 1 100 0.3; 0.5 0.3 5]};

% --- Motion Model ---
dt = 0.2;
num_steps = 20;
motion = BREW.dynamics.Integrator_3D();

% --- Measurement Model ---
H = [eye(3), zeros(3, 3)];
R = 0.2 * eye(3);

% --- EKF Initialization ---
process_noise = 0.01 * eye(6);
EKF = BREW.filters.GGIWEKF('dyn_obj', motion, 'H', H, ...
    'process_noise', process_noise, 'measurement_noise', R);

GGIW_est = BREW.models.Mixture();
GGIW_est.dist_type = "GGIW";
GGIW_est.means = truth.means;
GGIW_est.covariances = truth.covariances;
GGIW_est.weights = 1;
GGIW_est.alphas = truth.alphas;
GGIW_est.betas = truth.betas;
GGIW_est.vs = truth.vs;
GGIW_est.Vs = truth.Vs;

% --- Plot Setup ---
figure;
ax = axes;
axis equal;
grid on;
hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('EKF Tracking GGIW Truth Model');
view(3);

truth.plot_distributions(1:3, 'ax', ax, 'colors', [1 1 1]);

gifFilename = 'Tracker.gif';

% --- Main Loop ---
for k = 1:num_steps
    cla(ax);

    % --- Propagate Truth ---
    d = size(truth.Vs{1}, 1);
    truth.Vs{1} = motion.propagate_extent(dt, truth.means{1}, truth.Vs{1});
    if k == round(num_steps / 2)
        truth.means{1} = motion.propagateState(dt, truth.means{1}, 'u', rand(3,1));
    else
        truth.means{1} = motion.propagateState(dt, truth.means{1});
    end

    % --- Generate Measurement ---
    meas = truth.sample_measurements([1 2 3]);

    % --- EKF Predict ---
    GGIW_pred = EKF.predict(dt, GGIW_est, 'tau', 1);

    % --- EKF Correct ---
    [GGIW_est, lik] = EKF.correct(dt, meas, GGIW_pred);

    disp(lik)

    % --- Plot EKF Estimate Extent ---
    GGIW_est.plot_distributions(1:3, 'ax', ax, 'colors', [1 0 0]);

    % --- Plot Measurements ---
    scatter3(meas(1,:), meas(2,:), meas(3,:), 0.5, 'k*'); hold on

    % --- Plot EKF Mean ---
    plot3(ax, GGIW_est.means{1}(1), GGIW_est.means{1}(2), GGIW_est.means{1}(3), ...
        'ro', 'MarkerFaceColor', 'r');

    % --- Plot Truth Mean ---
    plot3(ax, truth.means{1}(1), truth.means{1}(2), truth.means{1}(3), ...
        'wo', 'MarkerFaceColor', 'w');

    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('EKF Tracking GGIW Truth (Step %d/%d)', k, num_steps));

    ax_extender = 20;
    mu = truth.means{1};
    axis(ax, [mu(1)-ax_extender mu(1)+ax_extender ...
              mu(2)-ax_extender mu(2)+ax_extender ...
              mu(3)-ax_extender mu(3)+ax_extender]);

    drawnow;

    % Capture GIF frame
    frame = getframe(gcf);
    img = frame2im(frame);
    [A, map] = rgb2ind(img, 256);

    if k == 1
        imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end

    pause(0.05);
end
