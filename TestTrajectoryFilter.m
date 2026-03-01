clear; clc; close all

%% Target setup

mean_val = [0; 0; 1; 1/2];
cov_val = diag([0.001, 0.001, 0.01, 0.01]);

target = BREW.models.Mixture();
target.dist_type = "Gaussian";
target.means = {mean_val};
target.covariances = {cov_val};
target.weights = 1;

%% Filter setup

dyn = BREW.dynamics.Integrator_2D();

% Initialize estimate as a trajectory Mixture
init_mean = target.sample_measurements(1:4);  % sample full state as initial estimate
est = BREW.models.Mixture();
est.dist_type = "TrajectoryGaussian";
est.state_dim = 4;
est.means = {init_mean};
est.covariances = {cov_val};
est.weights = 1;
est.init_indices = 1;
est.mean_histories = {reshape(init_mean, [4, 1])};

ekf = BREW.filters.TrajectoryGaussianEKF('dyn_obj', dyn, ...
    'process_noise', 0.1 * eye(4), ...
    'H', [1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 5 * eye(2));

dt = 1;
t = 0:dt:100;

%% Running the loop

meas_hst = [];

for k = 1:length(t)
    % Propagate truth
    target.means{1} = dyn.propagateState(dt, target.means{1});
    meas = target.sample_measurements([1 2]);

    meas_hst = [meas_hst, meas];

    est = ekf.predict(dt, est);
    [est, q] = ekf.correct(dt, meas, est);

    % Plotting
    scatter(meas_hst(1,:), meas_hst(2,:), 'w*'); grid on; hold on
    xlim([-5 105])
    ylim([-5 105])

    est.plot_distributions([1 2], 'c', 'r', 'window_color', 'c', 'LineWidth', 2);
    hold off

    drawnow;
end
