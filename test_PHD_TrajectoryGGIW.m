%% Testing PHD script (Trajectory GGIW)

clear; clc; close all

%% Truth model setup (using Mixture as GGIW for propagation)

truth = BREW.models.Mixture();
truth.dist_type = "GGIW";
truth.means = {[0; 10; 10; -0.75; -0.75; 0], [5; 20; 10; 0; -1; 0], [0; 12.5; 0; -0.5; 0; 1]};
truth.covariances = {eye(6), eye(6), eye(6)};
truth.weights = [1, 1, 1];
truth.alphas = [50, 50, 50];
truth.betas = [1, 1, 1];
truth.vs = [10, 10, 10];
truth.Vs = {[5 0 0; 0 5 0; 0 0 2], [5 0 0; 0 5 0; 0 0 2], [5 0 0; 0 5 0; 0 0 2]};

% Define truth target motion
target_motion = BREW.dynamics.Integrator_3D();

% Time setup
dt = 0.2;
tf = 10;
t = 0:dt:tf;

measurements = {};

%% Filter setup

H = [eye(3) zeros(3, length(truth.means{1}) - 3)];
R = 0.2 * eye(3);
process_noise = 0.01 * eye(length(truth.means{1}));

inner_filter = BREW.filters.TrajectoryGGIWEKF('dyn_obj', target_motion, ...
    'H', H, 'process_noise', process_noise, 'measurement_noise', R, 'L', 20);

%% Birth model (C++ handle)

birth_model = BREW.models.TrajectoryGGIWMixture( ...
    'idx', {1}, ...
    'means', {[0; 15; 10; 0; 0; 0]}, ...
    'covariances', {diag([10, 10, 10, 2, 2, 2])}, ...
    'alphas', {10}, ...
    'betas', {1}, ...
    'vs', {10}, ...
    'Vs', {[1 0 0; 0 1 0; 0 0 1]}, ...
    'weights', [1]);

%% PHD setup

phd = BREW.multi_target.PHD('filter', inner_filter, 'birth_model', birth_model, ...
    'prob_detection', 0.8, 'prob_survive', 0.8, 'max_terms', 50, ...
    'extract_threshold', 0.5);

f = figure;
ax = axes;
axis equal;
grid on;
hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);

axis(ax, [-10 10 0 20 0 20]);

gifFilename = 'TrajectoryGGIW_PHD_3D.gif';

for k = 1:length(t)
    cla(ax);

    % Propagate truth
    for kk = 1:length(truth)
        truth.means{kk} = target_motion.propagateState(dt, truth.means{kk});
    end

    % Generate measurements from truth
    measurements{k} = truth.sample_measurements();

    scatter3(measurements{k}(1,:), measurements{k}(2,:), measurements{k}(3,:), 'w*', 'SizeData', 0.5)

    % PHD filter
    phd.predict(dt);
    phd.correct(dt, measurements{k});
    est_mix = phd.cleanup();

    fprintf("timestep: %f \n", t(k))

    est_mix.plot_distributions([1, 2, 3], 'ax', ax, 'window_color', 'r', 'window_width', 1.6);

    drawnow;

    % Capture GIF frame
    frame = getframe(f);
    img = frame2im(frame);
    [A, map] = rgb2ind(img, 256);

    if k == 1
        imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end

    pause(0.3)
end
