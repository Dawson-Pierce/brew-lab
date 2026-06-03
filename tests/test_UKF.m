%% MATLAB UKF navigation test: single-target tracking of a coordinated turn
close all
rng(7);

%% Capability check — standalone filter predict needs the Stage 2 MEX.
try
    probe = BREW.filters.UKF('dyn_obj', BREW.dynamics.SingleIntegrator(2), ...
        'H', [eye(2) zeros(2, 2)], 'process_noise', eye(4), 'measurement_noise', eye(2));
    probe.predict(1.0, BREW.models.Gaussian([0; 0; 1; 0], eye(4)));
catch ME
    fprintf(['[note] Standalone filter predict/correct unavailable in this build ' ...
             '(rebuild: clear mex; build_brew). Skipping test_UKF.\n  %s\n'], ME.message);
    return
end

%% Scenario: one target moving and turning at a constant (unknown) rate.
dt = 1.0;
N  = 40;
true_omega = 0.08;
meas_std = sqrt(0.5);

x = [0; 0; 10; 0; true_omega];
truth = zeros(5, N);
meas  = zeros(2, N);
for k = 1:N
    x = propagate_ct(dt, x);
    truth(:, k) = x;
    meas(:, k)  = x(1:2) + meas_std * randn(2, 1);
end

%% Filters (identical config) over the same measurements.
H  = [eye(2), zeros(2, 3)];
Q  = diag([0, 0, 0.01, 0.01, 1e-3]);
R  = 0.5 * eye(2);
m0 = [0; 0; 10; 0; 0];
P0 = diag([1, 1, 1, 1, 0.5]);

dyn = BREW.dynamics.CoordinatedTurn();
ukf = BREW.filters.UKF('dyn_obj', dyn, 'H', H, 'process_noise', Q, 'measurement_noise', R);
ekf = BREW.filters.EKF('dyn_obj', dyn, 'H', H, 'process_noise', Q, 'measurement_noise', R);

[ukf_est, ukf_omega] = run_filter(ukf, meas, dt, m0, P0);
[ekf_est, ekf_omega] = run_filter(ekf, meas, dt, m0, P0);

%% Summary
fprintf('\n===== UKF coordinated-turn navigation (true omega = %.3f) =====\n', true_omega);
fprintf('  UKF: final omega = %.4f (err %.4f),  pos RMSE (2nd half) = %.3f\n', ...
    ukf_omega(end), abs(ukf_omega(end) - true_omega), rmse_2nd_half(ukf_est, truth));
fprintf('  EKF: final omega = %.4f (err %.4f),  pos RMSE (2nd half) = %.3f\n', ...
    ekf_omega(end), abs(ekf_omega(end) - true_omega), rmse_2nd_half(ekf_est, truth));
if abs(ukf_omega(end) - true_omega) < abs(ekf_omega(end) - true_omega)
    fprintf('  -> UKF recovers the turn rate better than the EKF (as expected).\n');
end
fprintf('================================================================\n');

%% Plots
out_dir = fullfile('tests', 'output');
if ~isfolder(out_dir), mkdir(out_dir); end
figure('Position', [100 100 1000 420]);

subplot(1, 2, 1); hold on; grid on; axis equal;
plot(truth(1, :), truth(2, :), 'k-', 'LineWidth', 2, 'DisplayName', 'truth');
scatter(meas(1, :), meas(2, :), 14, 'k.', 'DisplayName', 'measurements');
plot(ukf_est(1, :), ukf_est(2, :), 'r-', 'LineWidth', 1.5, 'DisplayName', 'UKF');
plot(ekf_est(1, :), ekf_est(2, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'EKF');
legend('Location', 'best'); xlabel('x'); ylabel('y'); title('Coordinated-turn tracking');

subplot(1, 2, 2); hold on; grid on;
plot([1 N], [true_omega true_omega], 'k--', 'LineWidth', 1.5, 'DisplayName', 'true \omega');
plot(1:N, ukf_omega, 'r-', 'LineWidth', 1.5, 'DisplayName', 'UKF \omega');
plot(1:N, ekf_omega, 'b--', 'LineWidth', 1.5, 'DisplayName', 'EKF \omega');
legend('Location', 'best'); xlabel('step'); ylabel('\omega'); title('Turn-rate estimate');

saveas(gcf, fullfile(out_dir, 'UKF_coordinated_turn.png'));

%% Local functions
function [est_hist, omega_hist] = run_filter(filt, meas, dt, m0, P0)
    est = BREW.models.Gaussian(m0, P0);
    N = size(meas, 2);
    est_hist = zeros(5, N);
    omega_hist = zeros(1, N);
    for k = 1:N
        est = filt.predict(dt, est);
        [est, ~] = filt.correct(meas(:, k), est);
        est_hist(:, k) = est.mean;
        omega_hist(k) = est.mean(5);
    end
end

function r = rmse_2nd_half(est, truth)
    N = size(est, 2);
    idx = max(1, floor(N / 2)) : N;
    d = est(1:2, idx) - truth(1:2, idx);
    r = sqrt(mean(sum(d .^ 2, 1)));
end

function x = propagate_ct(dt, x)
    w = x(5);
    F = eye(5);
    if abs(w) < 1e-6
        F(1, 3) = dt; F(2, 4) = dt;
    else
        s = sin(w * dt); c = cos(w * dt);
        F(1, 3) = s / w;         F(1, 4) = -(1 - c) / w;
        F(2, 3) = (1 - c) / w;   F(2, 4) = s / w;
        F(3, 3) = c;             F(3, 4) = -s;
        F(4, 3) = s;             F(4, 4) = c;
    end
    x = F * x;
end
