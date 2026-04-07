% clear; clc; 
close all

%% Target setup (truth model)

truth = BREW.models.Mixture();
truth.dist_type = "Gaussian";
truth.components = {
    BREW.models.Gaussian([0; -20; 1; 1/2], zeros(4))
    BREW.models.Gaussian([0; 20; 1; -1/2], zeros(4))
    BREW.models.Gaussian([0; 0; 3/2; 0], zeros(4))
};
truth.weights = [1, 1, 1];

%% Birth model (C++ handle)

birth = BREW.models.GaussianMixture( ...
    'means', {[20; 0; 1; 0]}, ...
    'covariances', {diag([10, 5, 1, 1])}, ...
    'weights', [1]);

%% Inner filter setup

dyn = BREW.dynamics.SingleIntegrator(2);

ekf = BREW.filters.EKF( ...
    'dyn_obj', dyn, ...
    'process_noise', diag([0.1 0.1 0.1 0.05]), ...
    'H', [1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 0.1 * diag([1; 1]));

dt = 1;
t = 0:dt:80;

%% PHD setup

phd = BREW.multi_target.PHD('filter', ekf, 'birth_model', birth, ...
    'prob_detection', 0.9, 'prob_survive', 0.98, 'max_components', 50, ...
    'extract_threshold', 0.5);

%% Running the loop

meas_hst = [];
meas_cov = diag([0.05, 0.05]);

gifFile = fullfile('tests','output', 'gm.gif');

for k = 1:length(t)

    % Propagate truth
    for ii = 1:truth.length()
        c = truth.components{ii};
        c.mean = propagate_si(dt, c.mean, 2);
        truth.components{ii} = c;
    end

    meas = utils.sample_measurements(truth, [1, 2], meas_cov);
    meas_hst = [meas_hst, meas]; %#ok<AGROW>

    phd.predict(dt);
    phd.correct(dt, meas);
    est_mix = phd.cleanup();

    % Plotting
    scatter(meas_hst(1,:), meas_hst(2,:), 'k*'); grid on; hold on
    utils.plot_distributions(est_mix, [1 2]);
    hold off

    xlim([-20 120])
    ylim([-70 70])
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
