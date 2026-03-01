clear; clc; close all

%% Target setup (truth model using Mixture)

truth = BREW.models.Mixture();
truth.dist_type = "Gaussian";
truth.means = {[0; -20; 1; 1/2], [0; 20; 1; -1/2], [0; 0; 3/2; 0]};
truth.covariances = repmat({zeros(4)}, 1, 3);
truth.weights = [1, 1, 1];

%% Birth model setup (C++ handle for PHD)

birth = BREW.models.GaussianMixture( ...
    'means', {[20; 0; 1; 0]}, ...
    'covariances', {diag([10, 5, 1, 1])}, ...
    'weights', [1]);

%% Inner filter setup

dyn = BREW.dynamics.Integrator_2D();

ekf = BREW.filters.EKF( ...
    'dyn_obj', dyn, ...
    'process_noise', diag([0.1 0.1 0.1 0.05]), ...
    'H', [1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 0.1 * diag([1; 1]));

dt = 1;
t = 0:dt:80;

%% PHD setup

phd = BREW.multi_target.PHD('filter', ekf, 'birth_model', birth, ...
    'prob_detection', 0.9, 'prob_survive', 0.98, 'max_terms', 50, ...
    'extract_threshold', 0.5);

%% Running the loop

meas_hst = [];
meas_cov = diag([0.05, 0.05]);

for k = 1:length(t)

    % Propagate truth
    for ii = 1:length(truth)
        truth.means{ii} = dyn.propagateState(dt, truth.means{ii});
    end

    meas = truth.sample_measurements([1, 2], meas_cov);

    meas_hst = [meas_hst, meas];

    phd.predict(dt);

    phd.correct(dt, meas);

    est_mix = phd.cleanup()

    % Plotting
    scatter(meas_hst(1,:), meas_hst(2,:), 'w*'); grid on; hold on

    est_mix.plot_distributions([1 2]);

    hold off

    xlim([-20 120])
    ylim([-70 70])

    drawnow;

    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    if k == 1
        imwrite(imind, cm, 'gm.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, 'gm.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
