%% JGLMB MATLAB test (Gaussian point targets): Murty vs Gibbs joint update
close all
rng(7);

%% Scenario
dt = 1;
t  = 0:dt:60;
xb = [-10 80];
yb = [-45 45];
meas_cov = diag([0.25, 0.25]);
p_detect = 0.95;
clutter_lambda = 3;

truth0 = { [0; -25; 1;  0.6]
           [0;  25; 1; -0.6]
           [0;   0; 1.2; 0  ] };
n_truth = numel(truth0);

%% Pre-generate truth trajectories + measurements (shared by both updates)
truth_hist = cell(numel(t), 1);
meas       = cell(numel(t), 1);
state = truth0;
for k = 1:numel(t)
    for ii = 1:n_truth
        state{ii} = propagate_si(dt, state{ii}, 2);
    end
    truth_hist{k} = state;

    z = [];
    for ii = 1:n_truth
        if rand < p_detect
            z = [z, mvnrnd(state{ii}(1:2)', meas_cov)'];
        end
    end
    n_clut = poissrnd(clutter_lambda);
    if n_clut > 0
        z = [z, [rand(1, n_clut) * diff(xb) + xb(1); ...
                 rand(1, n_clut) * diff(yb) + yb(1)]];
    end
    meas{k} = z;
end

%% Run the JGLMB with each joint-update strategy over identical measurements
[estM, cardM, timeM, tracksM] = run_jglmb(meas, t, dt, false, xb, yb);

gibbs_ok = true;
estG = {}; cardG = []; timeG = NaN; tracksG = [];
try
    [estG, cardG, timeG, tracksG] = run_jglmb(meas, t, dt, true, xb, yb);
catch ME
    gibbs_ok = false;
    fprintf(['[note] Gibbs update unavailable in this build ' ...
             '(rebuild: clear mex; build_brew). Running Murty only.\n  %s\n'], ...
            ME.message);
end

%% Summary
back = max(1, numel(t) - 20) : numel(t);
fprintf('\n===== JGLMB tracking (truth cardinality = %d) =====\n', n_truth);
fprintf('  Murty update: %.3f s (%.2f ms/step), final card=%d, mean card(last %d)=%.2f\n', ...
    timeM, 1000*timeM/numel(t), cardM(end), numel(back), mean(cardM(back)));
if gibbs_ok
    fprintf('  Gibbs update: %.3f s (%.2f ms/step), final card=%d, mean card(last %d)=%.2f\n', ...
        timeG, 1000*timeG/numel(t), cardG(end), numel(back), mean(cardG(back)));
end
fprintf('==================================================\n');

%% Animated overlay (2 panels if Gibbs ran, else 1)
out_dir = fullfile('tests', 'output');
if ~isfolder(out_dir), mkdir(out_dir); end
gifFile = fullfile(out_dir, 'JGLMB.gif');

fig = figure('Position', [100 100 1000 460]);
n_panel = 1 + gibbs_ok;
for k = 1:numel(t)
    clf(fig);
    draw_panel(subplot(1, n_panel, 1), 'JGLMB (Murty)', ...
        truth_hist{k}, meas{k}, estM{k}, xb, yb);
    if gibbs_ok
        draw_panel(subplot(1, n_panel, 2), 'JGLMB (Gibbs)', ...
            truth_hist{k}, meas{k}, estG{k}, xb, yb);
    end
    sgtitle(sprintf('t = %.0f', t(k)));
    drawnow;

    frame = getframe(fig);
    [imind, cm] = rgb2ind(frame2im(frame), 256);
    if k == 1
        imwrite(imind, cm, gifFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end

%% Cardinality vs time
cfig = figure;
plot(t, n_truth * ones(size(t)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'truth'); hold on
plot(t, cardM, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Murty');
if gibbs_ok
    plot(t, cardG, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Gibbs');
end
grid on; xlabel('t'); ylabel('estimated cardinality');
title('JGLMB estimated cardinality'); legend('Location', 'best');
saveas(cfig, fullfile(out_dir, 'JGLMB_cardinality.png'));

%% Labeled track trajectories (JGLMB keeps labeled histories, unlike a PHD)
tfig = figure('Position', [100 100 1000 460]);
draw_traj_panel(subplot(1, n_panel, 1), 'JGLMB (Murty)', truth_hist, tracksM, xb, yb);
if gibbs_ok
    draw_traj_panel(subplot(1, n_panel, 2), 'JGLMB (Gibbs)', truth_hist, tracksG, xb, yb);
end
sgtitle('JGLMB labeled track trajectories (truth = black)');
saveas(tfig, fullfile(out_dir, 'JGLMB_trajectories.png'));

%% ---- Local functions ----
function [ests, card, elapsed, tracks] = run_jglmb(meas, t, dt, use_gibbs, xb, yb)
    ekf = BREW.filters.EKF( ...
        'dyn_obj', BREW.dynamics.SingleIntegrator(2), ...
        'process_noise', diag([0.1 0.1 0.1 0.05]), ...
        'H', [1 0 0 0; 0 1 0 0], ...
        'measurement_noise', 0.25 * eye(2));

    birth = BREW.models.GaussianMixture( ...
        'means', {[0; -25; 1; 0]; [0; 25; 1; 0]; [0; 0; 1; 0]}, ...
        'covariances', {diag([9 9 1 1]), diag([9 9 1 1]), diag([9 9 1 1])}, ...
        'weights', [0.1, 0.1, 0.1]);

    args = {'filter', ekf, 'birth_model', birth, ...
            'prob_detection', 0.95, 'prob_survive', 0.99, ...
            'clutter_rate', 3, 'clutter_density', 1 / (diff(xb) * diff(yb)), ...
            'req_surv', 50, 'req_upd', 50, 'max_hypotheses', 100, ...
            'extract_threshold', 0.5, 'gate_threshold', 25};
    if use_gibbs
        args = [args, {'use_gibbs', true}];
    end
    jglmb = BREW.multi_target.JGLMB(args{:});

    ests = cell(numel(t), 1);
    card = zeros(numel(t), 1);
    tstart = tic;
    for k = 1:numel(t)
        jglmb.predict(dt);
        jglmb.correct(dt, meas{k});
        ests{k} = jglmb.cleanup();
        card(k) = ests{k}.length();
    end
    elapsed = toc(tstart);
    tracks = jglmb.track_histories();
end

function draw_panel(ax, ttl, truth_state, z, est_mix, xb, yb)
    cla(ax); hold(ax, 'on'); grid(ax, 'on');
    for ii = 1:numel(truth_state)
        plot(ax, truth_state{ii}(1), truth_state{ii}(2), 'k^', ...
            'MarkerFaceColor', 'k', 'MarkerSize', 6);
    end
    if ~isempty(z)
        scatter(ax, z(1, :), z(2, :), 12, 'k*');
    end
    if est_mix.length() > 0
        utils.plot_distributions(est_mix, [1 2], 'ax', ax, 'c', [0.85 0.1 0.1]);
    end
    xlim(ax, xb); ylim(ax, yb);
    title(ax, sprintf('%s  (tracks=%d)', ttl, est_mix.length()));
end

function draw_traj_panel(ax, ttl, truth_hist, tracks, xb, yb)
    cla(ax); hold(ax, 'on'); grid(ax, 'on');
    nt = numel(truth_hist{1});
    for ii = 1:nt
        tx = cellfun(@(s) s{ii}(1), truth_hist);
        ty = cellfun(@(s) s{ii}(2), truth_hist);
        plot(ax, tx, ty, 'k-', 'LineWidth', 1.2);
    end
    utils.plot_track_histories(tracks, [1 2], 'ax', ax, 'LineWidth', 2, 'min_len', 2);
    xlim(ax, xb); ylim(ax, yb);
    title(ax, sprintf('%s  (%d tracks)', ttl, numel(tracks)));
end

function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2 * n); F(1:n, n + 1:2 * n) = dt * eye(n);
    x = F * x;
end
