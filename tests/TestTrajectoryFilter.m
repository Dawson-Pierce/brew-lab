% Test script for single-target Trajectory Gaussian EKF
% NOTE: Uses inline EKF predict/correct since BREW.filters.TrajectoryGaussianEKF
%       predict/correct are stubs (override in a subclass for reuse).

% clear; clc; 
close all

%% Target setup

mean_val = [0; 0; 1; 1/2];
cov_val = diag([0.001, 0.001, 0.01, 0.01]);

target = BREW.models.Gaussian(mean_val, cov_val);

%% Filter setup

Q = 0.1 * eye(4);
H = [1 0 0 0; 0 1 0 0];
R = 5 * eye(2);
sd = 4;  % state_dim

% Initialize estimate
init_mean = mvnrnd(target.mean', target.covariance, 1)';
est = BREW.models.TrajectoryGaussian(init_mean, cov_val, sd, init_mean);

dt = 1;
t = 0:dt:100;

%% Running the loop

meas_hst = [];

for k = 1:length(t)
    % Propagate truth
    target.mean = propagate_si(dt, target.mean, 2);
    meas = mvnrnd((H * target.mean)', 0.25*R, 1)';
    meas_hst = [meas_hst, meas]; %#ok<AGROW>

    % --- Predict (inline trajectory Gaussian EKF) ---
    F = si_state_mat(dt, 2);
    pred_mean = F * est.mean;
    pred_cov = F * est.covariance * F' + Q;
    % For trajectory: stack onto history
    new_hist = [est.mean_history, pred_mean(end-sd+1:end)];
    est = BREW.models.TrajectoryGaussian(pred_mean, pred_cov, sd, new_hist);

    % --- Correct (inline EKF) ---
    % Build H for stacked state (observe last block)
    n_stacked = length(est.mean);
    H_stack = zeros(size(H,1), n_stacked);
    H_stack(:, end-sd+1:end) = H;

    innovation = meas - H_stack * est.mean;
    S = H_stack * est.covariance * H_stack' + R;
    K = est.covariance * H_stack' / S;
    cor_mean = est.mean + K * innovation;
    cor_cov = (eye(n_stacked) - K * H_stack) * est.covariance;
    cor_cov = (cor_cov + cor_cov') / 2;
    new_hist(:, end) = cor_mean(end-sd+1:end);
    est = BREW.models.TrajectoryGaussian(cor_mean, cor_cov, sd, new_hist);

    % --- Plot ---
    scatter(meas_hst(1,:), meas_hst(2,:), 'k*'); grid on; hold on
    xlim([-5 105]); ylim([-5 105]);

    est_mix = BREW.models.Mixture();
    est_mix.dist_type = "TrajectoryGaussian";
    est_mix.components = {est};
    est_mix.weights = 1;
    utils.plot_distributions(est_mix, [1 2], 'c', 'r', 'window_color', 'c', 'LineWidth', 2);
    hold off
    drawnow;
end

%% Local helpers
function x = propagate_si(dt, x, dims)
    F = si_state_mat(dt, dims);
    x = F * x;
end

function F = si_state_mat(dt, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
end
