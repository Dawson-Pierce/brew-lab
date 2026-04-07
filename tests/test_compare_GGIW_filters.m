%% Comparison: GGIW-PHD vs Trajectory-GGIW-PHD vs GGIW-GLMB (2D)
% Runs all three filters on the same scenario and compares runtimes.

close all

%% Scenario setup (2D extended targets)

state_dim = 4; % [x, y, vx, vy]
ext_dim = 2;   % 2D extent

truth = BREW.models.Mixture();
truth.dist_type = "GGIW";
truth.components = {
    BREW.models.GGIW(20, 1, [0; 0; 1; 0.5],   0.01*eye(state_dim), 10, [4 0; 0 3])
    BREW.models.GGIW(20, 1, [10; 10; -0.5; 0], 0.01*eye(state_dim), 10, [3 0; 0 5])
    BREW.models.GGIW(20, 1, [5; -5; 0; 0.8],   0.01*eye(state_dim), 10, [2 0; 0 2])
};
truth.weights = [1, 1, 1];

dt = 0.5;
tf = 15;
t = 0:dt:tf;
N = length(t);

%% Shared parameters

H = [eye(ext_dim), zeros(ext_dim, state_dim - ext_dim)];
R = 0.1 * eye(ext_dim);
Q = 0.01 * eye(state_dim);

prob_detection = 0.98;
prob_survive   = 0.99;
extract_thresh = 0.5;

%% Build dynamics + filters

dyn = BREW.dynamics.SingleIntegrator(2);

ekf_phd = BREW.filters.GGIWEKF('dyn_obj', dyn, 'H', H, ...
    'process_noise', Q, 'measurement_noise', R);
ekf_traj = BREW.filters.TrajectoryGGIWEKF('dyn_obj', dyn, 'H', H, ...
    'process_noise', Q, 'measurement_noise', R, 'window_size', 5);
ekf_glmb = BREW.filters.GGIWEKF('dyn_obj', dyn, 'H', H, ...
    'process_noise', Q, 'measurement_noise', R);

%% Birth models (parameters roughly match truth)

birth_mean = [5; 2; 0; 0];
birth_cov  = diag([50, 50, 2, 2]);
birth_alpha = 15;
birth_v = 8;
birth_V = 2*eye(ext_dim);

birth_phd = BREW.models.GGIWMixture( ...
    'alphas', {birth_alpha}, 'betas', {1}, ...
    'means', {birth_mean}, 'covariances', {birth_cov}, ...
    'vs', {birth_v}, 'Vs', {birth_V}, ...
    'weights', [0.05]);

birth_traj = BREW.models.TrajectoryGGIWMixture( ...
    'alphas', {birth_alpha}, 'betas', {1}, ...
    'means', {birth_mean}, 'covariances', {birth_cov}, ...
    'vs', {birth_v}, 'Vs', {birth_V}, ...
    'weights', [0.05]);

birth_glmb = BREW.models.GGIWMixture( ...
    'alphas', {birth_alpha}, 'betas', {1}, ...
    'means', {birth_mean}, 'covariances', {birth_cov}, ...
    'vs', {birth_v}, 'Vs', {birth_V}, ...
    'weights', [0.1]);  % higher for Bernoulli existence probability

%% Create RFS filters

phd = BREW.multi_target.PHD('filter', ekf_phd, 'birth_model', birth_phd, ...
    'prob_detection', prob_detection, 'prob_survive', prob_survive, ...
    'clutter_rate', 0, 'clutter_density', 0, ...
    'max_components', 80, 'extract_threshold', extract_thresh, ...
    'merge_threshold', 5.0, 'gate_threshold', 15.0);

traj_phd = BREW.multi_target.PHD('filter', ekf_traj, 'birth_model', birth_traj, ...
    'prob_detection', prob_detection, 'prob_survive', prob_survive, ...
    'clutter_rate', 0, 'clutter_density', 0, ...
    'max_components', 80, 'extract_threshold', extract_thresh, ...
    'merge_threshold', 5.0, 'gate_threshold', 15.0);

glmb = BREW.multi_target.GLMB('filter', ekf_glmb, 'birth_model', birth_glmb, ...
    'prob_detection', prob_detection, 'prob_survive', prob_survive, ...
    'clutter_rate', 0.5, 'clutter_density', 1e-2, ...
    'max_hypotheses', 200, 'k_best', 10, ...
    'extract_threshold', 0.3, 'gate_threshold', 15.0);

%% Pre-generate measurements (shared across all filters)

fprintf('Generating measurements...\n');
measurements = cell(1, N);
for k = 1:N
    for ii = 1:truth.length()
        c = truth.components{ii};
        c.mean = propagate_si(dt, c.mean, 2);
        truth.components{ii} = c;
    end
    measurements{k} = utils.sample_measurements(truth, [1, 2]);
end

% Reset truth for cardinality plot reference
truth.components = {
    BREW.models.GGIW(20, 1, [0; 0; 1; 0.5],   0.01*eye(state_dim), 10, [4 0; 0 3])
    BREW.models.GGIW(20, 1, [10; 10; -0.5; 0], 0.01*eye(state_dim), 10, [3 0; 0 5])
    BREW.models.GGIW(20, 1, [5; -5; 0; 0.8],   0.01*eye(state_dim), 10, [2 0; 0 2])
};

%% Run filters and time them

% --- GGIW-PHD ---
fprintf('Running GGIW-PHD...\n');
est_phd = cell(1, N);
tic;
for k = 1:N
    phd.predict(dt);
    phd.correct(dt, measurements{k});
    est_phd{k} = phd.cleanup();
end
time_phd = toc;

% --- Trajectory-GGIW-PHD ---
fprintf('Running Trajectory-GGIW-PHD...\n');
est_traj = cell(1, N);
tic;
for k = 1:N
    traj_phd.predict(dt);
    traj_phd.correct(dt, measurements{k});
    est_traj{k} = traj_phd.cleanup();
end
time_traj = toc;

% --- GGIW-GLMB ---
fprintf('Running GGIW-GLMB...\n');
est_glmb = cell(1, N);
tic;
for k = 1:N
    glmb.predict(dt);
    glmb.correct(dt, measurements{k});
    est_glmb{k} = glmb.cleanup();
end
time_glmb = toc;
tracks_glmb = glmb.track_histories();

%% Print runtime comparison

fprintf('\n===== Runtime Comparison =====\n');
fprintf('  GGIW-PHD:            %.4f s  (%.2f ms/step)\n', time_phd,  1000*time_phd/N);
fprintf('  Trajectory-GGIW-PHD: %.4f s  (%.2f ms/step)\n', time_traj, 1000*time_traj/N);
fprintf('  GGIW-GLMB:           %.4f s  (%.2f ms/step)\n', time_glmb, 1000*time_glmb/N);
fprintf('==============================\n');

%% Plot estimates with extent history

plot_every = 1;  % plot extent snapshot every N steps (tune this)
plot_dims = [1, 2];
d1 = plot_dims(1); d2 = plot_dims(2);

figure('Position', [100 100 1500 450]);

titles = {'GGIW-PHD', 'Trajectory-GGIW-PHD', 'GGIW-GLMB'};
times = [time_phd, time_traj, time_glmb];
axes_h = gobjects(1, 3);

for p = 1:3
    axes_h(p) = subplot(1, 3, p);
    hold(axes_h(p), 'on'); grid(axes_h(p), 'on'); axis(axes_h(p), 'equal');
    title(axes_h(p), sprintf('%s (%.1f ms/step)', titles{p}, 1000*times(p)/N));
    xlabel(axes_h(p), 'X'); ylabel(axes_h(p), 'Y');

    % Measurements (light gray background)
    for k = 1:N
        scatter(axes_h(p), measurements{k}(1,:), measurements{k}(2,:), 1, [0.85 0.85 0.85], '.');
    end
end

% --- Panel 1: GGIW-PHD (extent snapshots every plot_every steps) ---
ax1 = axes_h(1);
snap_steps = plot_every:plot_every:N;
if snap_steps(end) ~= N, snap_steps(end+1) = N; end
n_snaps = numel(snap_steps);
for si = 1:n_snaps
    k = snap_steps(si);
    mix = est_phd{k};
    if isempty(mix) || mix.isempty(), continue; end
    alpha_fade = 0.3 + 0.7 * (si / n_snaps);  % fade: older = lighter
    lw = 1 + 1.5 * (si / n_snaps);
    for ci = 1:mix.length()
        comp = mix.components{ci};
        clr = lines(1);  % single color, fade with time
        plot_ggiw_extent_(ax1, comp, plot_dims, clr, alpha_fade, lw);
    end
end

% --- Panel 2: Trajectory-GGIW-PHD (trajectory paths + extent snapshots along history) ---
ax2 = axes_h(2);
mix = est_traj{end};
if ~isempty(mix) && ~mix.isempty()
    traj_colors = lines(mix.length());
    for ci = 1:mix.length()
        comp = mix.components{ci};
        clr = traj_colors(ci,:);
        if isempty(comp.mean_history), continue; end
        hist = comp.mean_history;
        sd = comp.state_dim;
        T = size(hist, 2);

        % Full trajectory path
        plot(ax2, hist(d1,:), hist(d2,:), '-', 'Color', clr, 'LineWidth', 1.5);

        % Extent snapshots along the history at regular intervals
        snap_idx = plot_every:plot_every:T;
        if isempty(snap_idx) || snap_idx(end) ~= T, snap_idx(end+1) = T; end
        for si = 1:numel(snap_idx)
            ti = snap_idx(si);
            alpha_fade = 0.3 + 0.7 * (si / numel(snap_idx));
            lw = 1 + 1.5 * (si / numel(snap_idx));
            % Use the terminal extent (V) centered at this history point
            mu_ti = hist(plot_dims, ti);
            plot_ggiw_extent_at_(ax2, mu_ti, comp.V, comp.v, plot_dims, clr, alpha_fade, lw);
        end

        % Marker at current position
        plot(ax2, hist(d1,end), hist(d2,end), 'o', 'Color', clr, ...
            'MarkerFaceColor', clr, 'MarkerSize', 6);
    end
end

% --- Panel 3: GGIW-GLMB (labeled tracks + extent snapshots along track history) ---
ax3 = axes_h(3);
if ~isempty(tracks_glmb)
    n_tracks = numel(tracks_glmb);
    track_colors = lines(n_tracks);
    for tr = 1:n_tracks
        states = tracks_glmb(tr).states;
        if numel(states) < 2, continue; end
        xy = cell2mat(states);  % state_dim x T_track
        clr = track_colors(tr,:);
        T_tr = size(xy, 2);

        % Full track path
        plot(ax3, xy(d1,:), xy(d2,:), '-', 'Color', clr, 'LineWidth', 1.5);

        % Extent snapshots along track at regular intervals
        snap_idx = plot_every:plot_every:T_tr;
        if isempty(snap_idx) || snap_idx(end) ~= T_tr, snap_idx(end+1) = T_tr; end
        % Find the matching GLMB component for extent (use final estimate)
        glmb_final = est_glmb{end};
        V_track = []; v_track = [];
        if ~isempty(glmb_final) && ~glmb_final.isempty()
            % Use first available component's extent as proxy
            for ci = 1:glmb_final.length()
                V_track = glmb_final.components{ci}.V;
                v_track = glmb_final.components{ci}.v;
                break;
            end
        end

        if ~isempty(V_track)
            for si = 1:numel(snap_idx)
                ti = snap_idx(si);
                alpha_fade = 0.3 + 0.7 * (si / numel(snap_idx));
                lw = 1 + 1.5 * (si / numel(snap_idx));
                mu_ti = xy(plot_dims, ti);
                plot_ggiw_extent_at_(ax3, mu_ti, V_track, v_track, plot_dims, clr, alpha_fade, lw);
            end
        end

        % Label at endpoint
        plot(ax3, xy(d1,end), xy(d2,end), 'o', 'Color', clr, ...
            'MarkerFaceColor', clr, 'MarkerSize', 6);
        text(ax3, xy(d1,end)+0.3, xy(d2,end)+0.3, ...
            sprintf('ID %d', tracks_glmb(tr).id), 'Color', clr, 'FontSize', 8);
    end
end

saveas(gcf, fullfile('tests', 'output', 'filter_comparison.png'));
fprintf('Saved comparison figure.\n');

%% Cardinality over time

results_all = {est_phd, est_traj, est_glmb};

figure;
hold on; grid on;
for p = 1:3
    card = zeros(1, N);
    for k = 1:N
        if ~isempty(results_all{p}{k})
            card(k) = results_all{p}{k}.length();
        end
    end
    plot(t, card, 'LineWidth', 2, 'DisplayName', titles{p});
end
yline(3, '--k', 'Truth', 'LineWidth', 1);
xlabel('Time'); ylabel('Estimated Cardinality');
title('Cardinality Comparison');
legend('Location', 'best');
saveas(gcf, fullfile('tests', 'output', 'cardinality_comparison.png'));

%% Local helpers

function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    x = F * x;
end

function plot_ggiw_extent_(ax, comp, dims, clr, alpha_val, lw)
    %PLOT_GGIW_EXTENT_ Draw mean extent ellipse for a GGIW component.
    d = size(comp.V, 1);
    mu = comp.mean(dims);
    V2 = comp.V(dims, dims);
    me = V2 / (comp.v - d - 1);
    draw_extent_ellipse_(ax, mu, me, clr, alpha_val, lw);
end

function plot_ggiw_extent_at_(ax, mu, V, v, dims, clr, alpha_val, lw)
    %PLOT_GGIW_EXTENT_AT_ Draw mean extent ellipse at a given position.
    d = size(V, 1);
    V2 = V(dims, dims);
    me = V2 / (v - d - 1);
    draw_extent_ellipse_(ax, mu, me, clr, alpha_val, lw);
end

function draw_extent_ellipse_(ax, mu, me, clr, alpha_val, lw)
    [Ev, Ed] = eig(me);
    theta = linspace(0, 2*pi, 80);
    a = sqrt(max(Ed(1,1), 0));
    b = sqrt(max(Ed(2,2), 0));
    ell = Ev * [a*cos(theta); b*sin(theta)] + mu;
    plot(ax, ell(1,:), ell(2,:), '-', 'Color', [clr, alpha_val], 'LineWidth', lw);
end
