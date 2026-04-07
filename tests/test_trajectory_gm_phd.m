% clear; clc; 
close all

%% Target setup (truth model)

truth = BREW.models.Mixture();
truth.dist_type = "Gaussian";
truth.components = {
    BREW.models.Gaussian([0; -20; 1; 1/2], zeros(4))
    BREW.models.Gaussian([0; 20; 1; -1/2], zeros(4))
    BREW.models.Gaussian([0; 0; 1; 0], zeros(4))
};
truth.weights = [1, 1, 1];

% Track which targets are active
truth_init = [1, 1, 5];

%% Birth model (C++ handle — no idx param in new API)

birth = BREW.models.TrajectoryGaussianMixture( ...
    'means', {[0; 0; 1; 0]}, ...
    'covariances', {diag([1, 50, 1.5, 1])}, ...
    'weights', [0.1]);

%% Inner filter setup

dyn = BREW.dynamics.SingleIntegrator(2);

ekf = BREW.filters.TrajectoryGaussianEKF( ...
    'dyn_obj', dyn, ...
    'process_noise', diag([0.1 0.1 0.001 0.001]), ...
    'H', [1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 0.05 * [1 0; 0 1], ...
    'window_size', 20);

dt = 1;
t = 0:dt:80;

%% PHD setup

phd = BREW.multi_target.PHD('filter', ekf, 'birth_model', birth, ...
    'prob_detection', 0.8, 'prob_survive', 0.98, 'max_components', 25, ...
    'extract_threshold', 0.5, 'merge_threshold', 4, ...
    'clutter_rate', 0.05, 'clutter_density', 0.01);

%% Running the loop

meas_hst = [];
meas_cov = diag([0.05, 0.05]);
max_meas = 60;
x_bound = [-20 120];
y_bound = [-70 70];

gifFile = fullfile('tests','output', 'gm-trajectory.gif');

for k = 1:length(t)

    % Propagate truth (only active targets)
    for ii = 1:truth.length()
        if truth_init(ii) <= k
            c = truth.components{ii};
            c.mean = propagate_si(dt, c.mean, 2);
            truth.components{ii} = c;
        end
    end

    % Clutter
    num_noise = round(max_meas * rand(1));
    noisy_meas = [rand(1, num_noise) * diff(x_bound) + x_bound(1); ...
                  rand(1, num_noise) * diff(y_bound) + y_bound(1)];

    % Real measurements from active targets
    meas = [];
    for ii = 1:truth.length()
        if truth_init(ii) <= k
            mu_i = truth.components{ii}.mean;
            mi = mvnrnd(mu_i([1, 2])', meas_cov, 1)';
            meas = [meas, mi]; %#ok<AGROW>
        end
    end
    meas = [meas, noisy_meas];
    meas_hst = [meas_hst, meas]; %#ok<AGROW>

    phd.predict(dt);
    phd.correct(dt, meas);
    est_mix = phd.cleanup();

    % Plotting
    scatter(meas(1,:), meas(2,:), 8, 'k*'); grid on; hold on
    utils.plot_distributions(est_mix, [1 2], 'LineWidth', 2, 'window_color', 'r', 'window_width', 2.5);
    hold off

    xlim(x_bound)
    ylim(y_bound)
    drawnow;

    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if k == 1
        imwrite(imind, cm, gifFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end

%% Local helpers
function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    x = F * x;
end
