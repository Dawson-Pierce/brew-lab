%% PHD-IGGIW tracking on a gridded intensity field (storm-cell scenario)
% Pipeline per timestep:
%   1. Sample a 2D intensity field (background + Gaussian storm cells +
%      offset hot spots inside each cell + noise).
%   2. Pre-threshold the image to select supra-threshold pixels.
%   3. Pack pixels as [x; y; intensity] columns and pass to the PHD.
%   4. PHD's cluster_object (DBSCAN) partitions the columns into per-cell
%      measurement groups, which the inner IGGIW EKF batch-updates against
%      each track.
%
% Notes:
%   - IGGIW's measurement format is packed (pos_d + weight_1) per column,
%     so [x; y; intensity] columns are exactly what the filter expects.
%   - DBSCAN distance is Euclidean across all rows; with intensity in row 3
%     the per-cell intensity variation contributes to the gate. We scale
%     intensity into the cluster matrix so spatial separation dominates.
%   - Set rng to a fixed seed for reproducibility.

% clear; clc;
close all
rng(7);

%% Truth: two moving storm cells with offset hot spots.
% Velocities chosen so both cells stay inside the [-30,30]^2 grid for the
% full simulation window.
truth(1).mean    = [-15; -10;  1.0;  0.8];
truth(1).extent  = [9 1.5; 1.5 4];
truth(1).peak    = 55;
truth(1).hot_off = [2.5; -1.0];
truth(1).hot_ext = [1.5 0; 0 1.0];
truth(1).hot_peak = 75;

truth(2).mean    = [12;  8; -0.8; 0.5];
truth(2).extent  = [5 -1; -1 6];
truth(2).peak    = 50;
truth(2).hot_off = [-1.5; 1.5];
truth(2).hot_ext = [1 0; 0 1.2];
truth(2).hot_peak = 70;

intensity_noise = 1.5;
detect_thresh   = 35;
% Scale intensity into the measurement matrix so DBSCAN distance is
% spatially dominated. Pixels in the same cell vary by ~peak-thresh in
% intensity (≈40); scaling by 1/20 keeps that contribution ≈2 spatial units.
intensity_scale = 1/20;

%% Detector grid
[xg, yg] = meshgrid(-30:0.5:30, -30:0.5:30);
grid_xy  = [xg(:)'; yg(:)'];

%% Filter setup
dyn = BREW.dynamics.SingleIntegrator(2);
H   = [eye(2), zeros(2,2)];
R   = 0.1 * eye(2);
Q   = blkdiag(1e-3*eye(2), 5e-3*eye(2));

% centroid_power dials how aggressively hot spots pull the centroid:
%   1.0 (default) = standard intensity-weighted mean (z_bar = sum(w*x)/sum(w))
%   p > 1         = z_bar = sum(w^p * x) / sum(w^p), biased toward peak cells
%   p < 1         = relaxed toward geometric (unweighted) mean
% For a hot-spot-aware storm tracker, p in [2, 4] sharpens the centroid
% onto the high-reflectivity core.
centroid_power = 3.0;

inner = BREW.filters.IGGIWEKF( ...
    'dyn_obj', dyn, 'H', H, ...
    'process_noise', Q, 'measurement_noise', R, ...
    'eta', 4.0, 'lambda', 4.0, 'omega', 5.0, ...
    'intensity_forgetting_factor', 0.9, ...
    'intensity_growth', 1.0, ...
    'extent_forgetting_factor', 0.9, ...
    'centroid_power', centroid_power);

%% Birth model — broad IGGIW prior at the field origin
birth = BREW.models.IGGIWMixture( ...
    'alphas',      {10.0}, ...
    'betas',       {0.5}, ...
    'means',       {[0; 0; 0; 0]}, ...
    'covariances', {diag([300, 300, 4, 4])}, ...
    'vs',          {6 + 5}, ...
    'Vs',          {3 * eye(2)}, ...
    'weights',     0.05);

%% PHD
phd = BREW.multi_target.PHD( ...
    'filter', inner, 'birth_model', birth, ...
    'prob_detection', 0.95, ...
    'prob_survive',   0.99, ...
    'clutter_rate',   0, ...
    'max_components', 50, ...
    'extract_threshold', 0.5, ...
    'merge_threshold',   4.0, ...
    'prune_threshold',   1e-4, ...
    'gate_threshold',    16.0);

% DBSCAN: epsilon in the scaled measurement space. Pixel grid spacing 0.5
% and intensity-scaled bandwidth ~2 means epsilon ≈ 2 keeps intra-cell
% pixels together and separates cells (which are tens of units apart).
phd.set_cluster_object(BREW.clustering.DBSCAN(2.0, 4));

%% Animation
animate  = 1;  % 1 = write an animated GIF next to this script in tests/output
out_dir  = fullfile(fileparts(mfilename('fullpath')), 'output');
gif_file = fullfile(out_dir, 'PHD_IGGIW.gif');
if animate && ~isfolder(out_dir)
    mkdir(out_dir);
end

%% Plot setup — single axes: estimates overlaid on the reflectivity image
fig = figure('Position', [100 100 700 600]);
ax = axes(fig); hold(ax,'on'); axis(ax,'equal');
xlim(ax,[-30 30]); ylim(ax,[-30 30]);
xlabel(ax,'X'); ylabel(ax,'Y');

dt = 0.5;
t  = 0:dt:15;

%% Tracking loop
for k = 1:numel(t)
    cla(ax);

    % --- Propagate truth ---
    for ti = 1:numel(truth)
        truth(ti).mean = propagate_si(dt, truth(ti).mean, 2);
    end

    % --- Sample gridded intensity field ---
    field_intensity = intensity_noise * randn(1, size(grid_xy, 2));
    for ti = 1:numel(truth)
        dx = grid_xy - truth(ti).mean(1:2);
        q1 = sum(dx .* (truth(ti).extent \ dx), 1);
        hot_centre = truth(ti).mean(1:2) + truth(ti).hot_off;
        dh = grid_xy - hot_centre;
        q2 = sum(dh .* (truth(ti).hot_ext \ dh), 1);
        field_intensity = field_intensity ...
            + truth(ti).peak     * exp(-0.5 * q1) ...
            + truth(ti).hot_peak * exp(-0.5 * q2);
    end

    imagesc(ax, xg(1,:), yg(:,1), reshape(field_intensity, size(xg)));
    set(ax,'YDir','normal'); colorbar(ax);

    % --- Threshold to get supra-threshold pixels ---
    keep = field_intensity > detect_thresh;
    meas_xy  = grid_xy(:, keep);
    meas_int = field_intensity(keep);

    % --- Build measurement matrix and run PHD ---
    if ~isempty(meas_int)
        measurements = [meas_xy; intensity_scale * meas_int(:)'];
    else
        measurements = zeros(3, 0);
    end

    phd.predict(dt);
    phd.correct(measurements);
    est_mix = phd.cleanup();

    % --- Overlay IGGIW estimates on the reflectivity image ---
    for ci = 1:est_mix.length()
        comp = est_mix.components{ci};
        d = size(comp.V, 1); dof_floor = 2*d + 2;
        X = comp.V / max(comp.v - dof_floor, eps);
        X = 0.5*(X + X');
        plot_extent_ellipse(ax, comp.mean(1:2), X, 'r-', ...
            sprintf('IGGIW #%d w=%.2f', ci, est_mix.weights(ci)));
        scatter(ax, comp.mean(1), comp.mean(2), 60, 'r', 'x', ...
            'LineWidth', 2, 'HandleVisibility','off');
    end
    if est_mix.length() > 0
        legend(ax, 'Location','northeastoutside');
    end

    title(ax, sprintf('t=%.1f  tracks=%d  pixels=%d  p=%.1f', ...
        t(k), est_mix.length(), nnz(keep), centroid_power));
    drawnow;

    % --- Animation frame ---
    if animate
        [A, map] = rgb2ind(frame2im(getframe(fig)), 256);
        if k == 1
            imwrite(A, map, gif_file, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
        else
            imwrite(A, map, gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
end

fprintf('PHD-IGGIW finished (centroid_power = %.2f). Final tracks: %d\n', ...
    centroid_power, est_mix.length());
for ci = 1:est_mix.length()
    comp = est_mix.components{ci};
    % Compare estimated centroid to nearest truth centroid + hot spot
    truth_centres = arrayfun(@(s) s.mean(1:2), truth, 'UniformOutput', false);
    truth_hots    = arrayfun(@(s) s.mean(1:2) + s.hot_off, truth, ...
                             'UniformOutput', false);
    dists = cellfun(@(c) norm(comp.mean(1:2) - c), truth_centres);
    [~, near] = min(dists);
    fprintf(['  track %d: pos=[%.2f, %.2f] | nearest truth centroid ' ...
             '[%.2f, %.2f]  hot spot [%.2f, %.2f]\n'], ...
        ci, comp.mean(1), comp.mean(2), ...
        truth_centres{near}(1), truth_centres{near}(2), ...
        truth_hots{near}(1), truth_hots{near}(2));
    fprintf('           mean intensity = %.2f, v = %.2f\n', ...
        comp.beta / max(comp.alpha - 1, eps) / intensity_scale, comp.v);
end

%% Plot helpers

function plot_extent_ellipse(ax, centre, X, style, label)
    theta = linspace(0, 2*pi, 64);
    circle = [cos(theta); sin(theta)];
    [U, S, ~] = svd(X);
    radii = sqrt(max(diag(S), 0));
    pts = U * diag(radii) * circle * 2;       % 2-sigma
    pts = pts + centre(:);
    plot(ax, pts(1,:), pts(2,:), style, 'LineWidth', 1.5, 'DisplayName', label);
end

function x = propagate_si(dt, x, dims)
    n = dims;
    F = eye(2*n); F(1:n, n+1:2*n) = dt * eye(n);
    x = F * x;
end
