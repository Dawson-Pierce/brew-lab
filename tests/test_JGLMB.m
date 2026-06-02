%% JGLMB MATLAB test (Gaussian point targets): Murty vs Gibbs joint update
% Tracks three crossing 2D targets in clutter with the JGLMB filter, exercising
% both the exact Murty ranked-assignment update (default) and the Gibbs-sampler
% update (set via 'use_gibbs', true). Both filters are run over the SAME
% pre-generated measurements so the two updates can be compared directly.
%
% NOTE: the 'use_gibbs' option requires a MEX built from the commit that added it
% (run: clear mex; build_brew). If the loaded MEX predates it, the script runs
% the Murty update only and prints a note instead of failing.
%
% Run from the project root (the folder containing +BREW and +utils).
close all
rng(7);   % reproducible detections + clutter

%% Scenario
dt = 1;
t  = 0:dt:60;
xb = [-10 80];          % field x-bounds (also the clutter region)
yb = [-45 45];          % field y-bounds
meas_cov = diag([0.25, 0.25]);
p_detect = 0.95;
clutter_lambda = 3;     % expected clutter points per scan

% Three targets [x; y; vx; vy], all present for the whole run.
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
            z = [z, mvnrnd(state{ii}(1:2)', meas_cov)']; %#ok<AGROW>
        end
    end
    n_clut = poissrnd(clutter_lambda);
    if n_clut > 0
        z = [z, [rand(1, n_clut) * diff(xb) + xb(1); ...
                 rand(1, n_clut) * diff(yb) + yb(1)]]; %#ok<AGROW>
    end
    meas{k} = z;
end

%% Run the JGLMB with each joint-update strategy over identical measurements
[estM, cardM, timeM] = run_jglmb(meas, t, dt, false, xb, yb);

gibbs_ok = true;
estG = {}; cardG = []; timeG = NaN;
try
    [estG, cardG, timeG] = run_jglmb(meas, t, dt, true, xb, yb);
catch ME
    gibbs_ok = false;
    fprintf(['[note] Gibbs update unavailable in this build ' ...
             '(rebuild: clear mex; build_brew). Running Murty only.\n  %s\n'], ...
            ME.message);
end

%% Summary
back = max(1, numel(t) - 20) : numel(t);   % steady-state window
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

%% ---- Local functions ----
function [ests, card, elapsed] = run_jglmb(meas, t, dt, use_gibbs, xb, yb)
    % Fresh inner filter + birth model per run (each JGLMB copies/clones them).
    ekf = BREW.filters.EKF( ...
        'dyn_obj', BREW.dynamics.SingleIntegrator(2), ...
        'process_noise', diag([0.1 0.1 0.1 0.05]), ...
        'H', [1 0 0 0; 0 1 0 0], ...
        'measurement_noise', 0.25 * eye(2));

    % Birth covers each target's appearance region.
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
end

function draw_panel(ax, ttl, truth_state, z, est_mix, xb, yb)
    cla(ax); hold(ax, 'on'); grid(ax, 'on');
    % truth
    for ii = 1:numel(truth_state)
        plot(ax, truth_state{ii}(1), truth_state{ii}(2), 'k^', ...
            'MarkerFaceColor', 'k', 'MarkerSize', 6);
    end
    % measurements
    if ~isempty(z)
        scatter(ax, z(1, :), z(2, :), 12, 'k*');
    end
    % estimates
    if est_mix.length() > 0
        utils.plot_distributions(est_mix, [1 2], 'ax', ax, 'c', [0.85 0.1 0.1]);
    end
    xlim(ax, xb); ylim(ax, yb);
    title(ax, sprintf('%s  (tracks=%d)', ttl, est_mix.length()));
end

function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2 * n); F(1:n, n + 1:2 * n) = dt * eye(n);
    x = F * x;
end
