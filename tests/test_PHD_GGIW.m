%% Testing PHD script (GGIW)

% clear; clc; 
close all

%% Truth model setup

truth = BREW.models.Mixture();
truth.dist_type = "GGIW";
truth.components = {
    BREW.models.GGIW(70, 1, [0; 10; 10; -0.75; -0.75; 0], eye(6), 10, [5 0 0; 0 5 0; 0 0 2])
    BREW.models.GGIW(70, 1, [5; 20; 10; 0; -1; 0],        eye(6), 10, [5 0 0; 0 5 0; 0 0 2])
    BREW.models.GGIW(70, 1, [0; 12.5; 0; -0.5; 0; 1],     eye(6), 10, [5 0 0; 0 5 0; 0 0 2])
};
truth.weights = [1, 1, 1];

% Dynamics
dyn = BREW.dynamics.SingleIntegrator(3);

dt = 0.2;
tf = 10;
t = 0:dt:tf;
measurements = {};

%% Filter setup

H = [eye(3) zeros(3, 3)];
R = 0.2 * eye(3);
process_noise = 0.1 * eye(6);

inner_filter = BREW.filters.GGIWEKF('dyn_obj', dyn, ...
    'H', H, 'process_noise', process_noise, 'measurement_noise', R);

%% Birth model (C++ handle)

birth_model = BREW.models.GGIWMixture( ...
    'means', {[0; 15; 10; 0; 0; 0]}, ...
    'covariances', {diag([15, 15, 15, 2, 2, 2])}, ...
    'alphas', {10}, 'betas', {1}, 'vs', {10}, ...
    'Vs', {eye(3)}, ...
    'weights', [0.001]);

%% PHD setup

phd = BREW.multi_target.PHD('filter', inner_filter, ...
    'birth_model', birth_model, ...
    'prob_detection', 0.7, ...
    'prob_survive', 0.95, ...
    'max_components', 50, ...
    'extract_threshold', 0.5, ...
    'clutter_rate', 0);

f = figure; ax = axes;
axis equal; grid on; hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z'); view(3);
axis(ax, [-10 10 0 20 0 20]);

gifFile = fullfile('tests','output', 'GGIW_PHD_3D.gif');

for k = 1:length(t)
    cla(ax);

    % Propagate truth
    for kk = 1:truth.length()
        c = truth.components{kk};
        c.mean = propagate_si(dt, c.mean, 3);
        truth.components{kk} = c;
    end

    % Generate measurements
    measurements{k} = utils.sample_measurements(truth); %#ok<SAGROW>
    scatter3(measurements{k}(1,:), measurements{k}(2,:), measurements{k}(3,:), 'k*', 'SizeData', 0.5)

    % PHD filter
    phd.predict(dt);
    phd.correct(dt, measurements{k});
    est_mix = phd.cleanup();

    % fprintf("timestep: %f \n", t(k))
    % disp(est_mix.weights)

    utils.plot_distributions(est_mix, [1, 2, 3], 'ax', ax);
    drawnow;

    % GIF
    frame = getframe(f);
    img = frame2im(frame);
    [A, map] = rgb2ind(img, 256);
    if k == 1
        imwrite(A, map, gifFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
    pause(0.3)
end

%% Local helpers
function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    x = F * x;
end
