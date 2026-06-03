%% Single-target IGGIW EKF tracking on a 2D gridded intensity field

close all

%% Truth: 2D moving extended region with a Gaussian intensity profile

truth_mean      = [-15; -10;  2; 1.5];
truth_extent    = [9 1.5; 1.5 4];
peak_intensity  = 55;
intensity_noise = 1.5;
detect_thresh   = 35;

hot_spot_offset = [2.5; -1.0];
hot_spot_extent = [1.5 0; 0 1.0];
hot_spot_peak   = 75;

[xg, yg] = meshgrid(-30:0.5:30, -30:0.5:30);
grid_xy  = [xg(:)'; yg(:)'];

H = [eye(2), zeros(2,2)];
R = 0.05 * eye(2);
Q = blkdiag(1e-3*eye(2), 1e-2*eye(2));

dt = 0.5;
t  = 0:dt:20;

%% Filter hyperparameters (mirror brew C++ defaults; tuned for intensity weights)

eta              = 4.0;
lambda           = 4.0;
omega            = 5.0;
delta_gamma      = 0.9;
gamma_growth     = 1.0;
delta_extent     = 0.9;

d = size(truth_extent, 1);
dof_floor = 2*d + 2;

est = BREW.models.IGGIW( ...
    10.0, ...
    0.25, ...
    truth_mean + [3; 3; 0; 0], ...
    diag([4 4 1 1]), ...
    dof_floor + 5, ...
    eye(d) * (dof_floor + 5 - dof_floor));

filt_handle = BREW.filters.IGGIWEKF( ...
    'dyn_obj', BREW.dynamics.SingleIntegrator(2), ...
    'H', H, 'process_noise', Q, 'measurement_noise', R, ...
    'eta', eta, 'lambda', lambda, 'omega', omega, ...
    'intensity_forgetting_factor', delta_gamma, ...
    'intensity_growth', gamma_growth, ...
    'extent_forgetting_factor', delta_extent);
assert(filt_handle.handle_ ~= 0, 'IGGIWEKF C++ handle was not created');
assert(filt_handle.dist_type_ == "IGGIW", 'IGGIWEKF dist_type mismatch');

%% Plot setup (intensity field on left, tracker overlay on right)

fig = figure('Position', [100 100 1000 480]);
ax_field = subplot(1, 2, 1); hold(ax_field, 'on'); axis(ax_field, 'equal');
xlim(ax_field, [-30 30]); ylim(ax_field, [-30 30]);
title(ax_field, 'Reflectivity field');
xlabel(ax_field, 'X'); ylabel(ax_field, 'Y');

ax_track = subplot(1, 2, 2); hold(ax_track, 'on'); axis(ax_track, 'equal');
xlim(ax_track, [-30 30]); ylim(ax_track, [-30 30]);
title(ax_track, 'IGGIW estimate vs truth');
xlabel(ax_track, 'X'); ylabel(ax_track, 'Y'); grid(ax_track, 'on');

%% Tracking loop
for k = 1:numel(t)
    cla(ax_field); cla(ax_track);

    %% Propagate truth (single-integrator 2D)
    truth_mean = propagate_si(dt, truth_mean, 2);

    %% Sample gridded intensity field over the truth extent
    dx_main = grid_xy - truth_mean(1:2);
    quad_main = sum(dx_main .* (truth_extent \ dx_main), 1);

    hot_centre = truth_mean(1:2) + hot_spot_offset;
    dx_hot = grid_xy - hot_centre;
    quad_hot = sum(dx_hot .* (hot_spot_extent \ dx_hot), 1);

    field_intensity = peak_intensity * exp(-0.5 * quad_main) ...
                   + hot_spot_peak * exp(-0.5 * quad_hot) ...
                   + intensity_noise * randn(1, size(grid_xy, 2));

    imagesc(ax_field, xg(1,:), yg(:,1), reshape(field_intensity, size(xg)));
    set(ax_field, 'YDir', 'normal');
    colorbar(ax_field);

    keep = field_intensity > detect_thresh;
    meas_xy   = grid_xy(:, keep);
    meas_int  = field_intensity(keep);
    N = numel(meas_int);

    %% IGGIW EKF predict
    est = iggiw_predict(est, dt, Q, ...
        delta_gamma, gamma_growth, delta_extent);

    %% IGGIW EKF correct — weight each cell by its intensity
    if N > 0
        est = iggiw_correct(est, meas_xy, meas_int(:), H, R, eta, lambda, omega);
    end

    %% Overlay: detected cells, truth, hot spot, IGGIW estimate
    scatter(ax_track, meas_xy(1,:), meas_xy(2,:), 12, meas_int, 'filled', ...
        'DisplayName', 'detections');
    plot_extent_ellipse(ax_track, truth_mean(1:2), truth_extent, ...
        'k--', 'truth centroid');
    scatter(ax_track, hot_centre(1), hot_centre(2), 80, 'm', 'p', ...
        'filled', 'DisplayName', 'hot spot');
    plot_iggiw_estimate(ax_track, est);
    legend(ax_track, 'Location', 'northeastoutside');
    drawnow;
end

fprintf('IGGIW single-target tracking finished.\n');
fprintf('  final alpha/beta = %.2f / %.2f\n', est.alpha, est.beta);
fprintf('  InverseGamma mean intensity = beta/(alpha-1) = %.2f\n', ...
    est.beta / max(est.alpha - 1, eps));
fprintf('  final v = %.2f (extent dof)\n', est.v);
fprintf('  estimated extent (V/(v-dof_floor)):\n');
disp(est.V / max(est.v - (2*size(est.V,1)+2), eps));

%% --- IGGIW EKF math (ported from brew/core/filters/iggiw_ekf.hpp) ---

function next = iggiw_predict(prev, dt, Q, ...
        delta_gamma, gamma_growth, delta_extent)
    n = numel(prev.mean) / 2;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    next_mean = F * prev.mean;
    next_cov  = F * prev.covariance * F' + Q;
    next_cov  = 0.5 * (next_cov + next_cov');

    eps_d = eps;

    alpha_prev = max(prev.alpha, 1 + eps_d);
    beta_prev  = max(prev.beta,  eps_d);
    gamma_bar  = gamma_growth * beta_prev / (alpha_prev - 1);

    next_alpha = max(1 + delta_gamma * (alpha_prev - 1), 1 + eps_d);
    next_beta  = max(gamma_bar * (next_alpha - 1), eps_d);

    d = size(prev.V, 1);
    dof_floor = 2*d + 2;
    v_prev = max(prev.v, dof_floor + eps_d);

    X_bar  = prev.V / (v_prev - dof_floor);
    X_bar  = 0.5 * (X_bar + X_bar');

    next_v = max(dof_floor + delta_extent * (v_prev - dof_floor), ...
                 dof_floor + eps_d);
    next_V = X_bar * (next_v - dof_floor);
    next_V = 0.5 * (next_V + next_V');

    next = BREW.models.IGGIW(next_alpha, next_beta, ...
        next_mean, next_cov, next_v, next_V);
end

function next = iggiw_correct(prev, positions, intensities, H, R, ...
        eta, lambda, omega)
    d = size(prev.V, 1);
    weights = intensities(:);

    eps_d = eps;
    sum_w = max(sum(weights), eps_d);
    r     = max(mean(weights), eps_d);

    z_bar  = (positions * weights) / sum_w;
    diffs  = positions - z_bar;
    Z_meas = (diffs .* weights') * diffs' / sum_w;
    Z_meas = 0.5 * (Z_meas + Z_meas');

    dof_floor = 2*d + 2;
    v_safe = max(prev.v, dof_floor + eps_d);

    X_hat = prev.V / (v_safe - dof_floor);
    X_hat = 0.5 * (X_hat + X_hat');

    z_hat      = H * prev.mean;
    innovation = z_bar - z_hat;

    R_hat = X_hat / lambda + R;
    R_hat = 0.5 * (R_hat + R_hat');

    S = H * prev.covariance * H' + R_hat;
    S = 0.5 * (S + S');

    K = prev.covariance * H' / S;

    next_alpha = prev.alpha + eta;
    next_beta  = prev.beta  + eta * r;

    next_mean = prev.mean + K * innovation;
    next_cov  = prev.covariance - K * S * K';
    next_cov  = 0.5 * (next_cov + next_cov');

    next_v = prev.v + omega;
    next_V = prev.V + omega * Z_meas;
    next_V = 0.5 * (next_V + next_V');

    next = BREW.models.IGGIW(next_alpha, next_beta, ...
        next_mean, next_cov, next_v, next_V);
end

%% --- Plot helpers ---

function plot_iggiw_estimate(ax, est)
    d = size(est.V, 1);
    dof_floor = 2*d + 2;
    X = est.V / max(est.v - dof_floor, eps);
    X = 0.5 * (X + X');
    plot_extent_ellipse(ax, est.mean(1:2), X, 'r-', ...
        sprintf('IGGIW (mean intensity\\approx%.1f)', ...
            est.beta / max(est.alpha - 1, eps)));
    scatter(ax, est.mean(1), est.mean(2), 60, 'r', 'x', 'LineWidth', 2, ...
        'HandleVisibility', 'off');
end

function plot_extent_ellipse(ax, centre, X, style, label)
    theta = linspace(0, 2*pi, 64);
    circle = [cos(theta); sin(theta)];
    [U, S, ~] = svd(X);
    radii = sqrt(max(diag(S), 0));
    pts = U * diag(radii) * circle * 2;
    pts = pts + centre(:);
    plot(ax, pts(1,:), pts(2,:), style, 'LineWidth', 1.5, 'DisplayName', label);
end

%% --- Local helpers ---

function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    x = F * x;
end
