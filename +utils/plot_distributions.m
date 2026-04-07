function plot_distributions(mix, plt_inds, varargin)
%PLOT_DISTRIBUTIONS Plot mixture components.
%   utils.plot_distributions(mix, plt_inds, 'Name', Value, ...)
%
%   Dispatches on mix.dist_type to plot Gaussian ellipses, GGIW extents,
%   trajectory paths, etc.
%
%   Name-Value pairs:
%     'ax'           - axes handle (default gca)
%     'colors'/'c'   - Nx3 color matrix or single color
%     'LineWidth'    - default 2
%     'MarkerSize'   - default 4
%     'num_std'      - std devs for Gaussian ellipse (default 2)
%     'h'            - confidence for GGIW chi2 (default 0.95)
%     'window_color' - trajectory window highlight color
%     'window_width' - trajectory window line width
%     'window_style' - trajectory window line style (default '-')
%     'LineStyle'    - default '-'

    if nargin < 2, plt_inds = 1; end

    n = mix.length();
    if n == 0, return; end

    ip = inputParser; ip.KeepUnmatched = true;
    addParameter(ip, 'ax', gca);
    addParameter(ip, 'colors', []);
    addParameter(ip, 'c', []);
    addParameter(ip, 'LineWidth', 2);
    addParameter(ip, 'MarkerSize', 4);
    addParameter(ip, 'num_std', 2);
    addParameter(ip, 'h', 0.95);
    addParameter(ip, 'window_color', []);
    addParameter(ip, 'window_width', []);
    addParameter(ip, 'window_style', '-');
    addParameter(ip, 'LineStyle', '-');
    parse(ip, varargin{:});

    ax = ip.Results.ax; lw = ip.Results.LineWidth; ms = ip.Results.MarkerSize;
    ls = ip.Results.LineStyle; ns = ip.Results.num_std; hc = ip.Results.h;
    wc = ip.Results.window_color; ww = ip.Results.window_width; ws = ip.Results.window_style;
    if isempty(ww), ww = lw; end

    colors = ip.Results.colors;
    if isempty(colors), colors = ip.Results.c; end
    if isempty(colors), colors = lines(n); end
    if size(colors,1) == 1, colors = repmat(colors, n, 1); end

    prev_hold = ishold(ax); hold(ax, 'on');

    for i = 1:n
        comp = mix.components{i};
        clr = colors(i,:);

        switch mix.dist_type
            case "Gaussian"
                plot_gaussian_(comp, plt_inds, ax, clr, ns, lw, ms, ls);
            case "GGIW"
                plot_ggiw_(comp, plt_inds, ax, clr, hc, lw, ls);
            case {"GGIWOrientation"}
                plot_ggiw_(comp, plt_inds, ax, clr, hc, lw, ls);
            case "TrajectoryGaussian"
                plot_traj_gaussian_(comp, plt_inds, ax, clr, ns, lw, ms, ls, wc, ww, ws);
            case {"TrajectoryGGIW", "TrajectoryGGIWOrientation"}
                plot_traj_ggiw_(comp, plt_inds, ax, clr, hc, lw, ms, ls, wc, ww, ws);
            otherwise
                mu = comp.mean;
                if length(plt_inds) >= 3
                    plot3(ax, mu(plt_inds(1)), mu(plt_inds(2)), mu(plt_inds(3)), ...
                        'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
                elseif length(plt_inds) == 2
                    plot(ax, mu(plt_inds(1)), mu(plt_inds(2)), ...
                        'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
                end
        end
    end

    if ~prev_hold, hold(ax, 'off'); end
end

%% ---- Gaussian ----
function plot_gaussian_(comp, dims, ax, clr, ns, lw, ms, ls)
    mu = comp.mean(dims); S = comp.covariance(dims, dims);
    if length(dims) == 1
        x = linspace(mu-ns*sqrt(S), mu+ns*sqrt(S), 200);
        plot(ax, x, normpdf(x,mu,sqrt(S)), 'Color', clr, 'LineWidth', lw, 'LineStyle', ls);
    elseif length(dims) == 2
        plot(ax, mu(1), mu(2), '*', 'Color', clr, 'MarkerSize', ms);
        draw_ellipse_(ax, mu, S, ns, clr, lw);
    elseif length(dims) == 3
        draw_ellipsoid_(ax, mu, S, ns, clr, lw, ms);
    end
end

%% ---- GGIW ----
function plot_ggiw_(comp, dims, ax, clr, hc, lw, ls)
    mu = comp.mean(dims); V2 = comp.V(dims,dims);
    d = size(comp.V, 1);
    me = V2 / (comp.v - d - 1);
    if length(dims) == 2
        [Ev, Ed] = eig(me);
        t = linspace(0,2*pi,100);
        a = sqrt(Ed(1,1)); b = sqrt(Ed(2,2));
        ell = Ev*[a*cos(t); b*sin(t)] + mu;
        plot(ax, ell(1,:), ell(2,:), ls, 'Color', clr, 'LineWidth', lw);
        sc = sqrt(chi2inv(hc, 2));
        ell_c = Ev*[sc*a*cos(t); sc*b*sin(t)] + mu;
        plot(ax, ell_c(1,:), ell_c(2,:), '--', 'Color', clr, 'LineWidth', lw);
    elseif length(dims) == 3
        draw_extent_3d_(ax, mu, me, hc, clr, lw);
    end
end

%% ---- Trajectory Gaussian ----
function plot_traj_gaussian_(comp, dims, ax, clr, ns, lw, ms, ls, wc, ww, ws)
    if isempty(comp.mean_history), return; end
    hist = comp.mean_history; sd = comp.state_dim;
    if sd == 0, sd = size(hist,1); end
    T = size(hist, 2);
    win_len = T;
    if sd > 0 && ~isempty(comp.mean)
        win_len = min(T, numel(comp.mean) / sd);
    end
    fl = T - win_len;

    if length(dims) == 2
        d1 = dims(1); d2 = dims(2);
        if fl > 0, plot(ax, hist(d1,1:fl+1), hist(d2,1:fl+1), ls, 'Color', clr, 'LineWidth', lw); end
        wi = max(1,fl+1):T;
        if ~isempty(wc), plot(ax, hist(d1,wi), hist(d2,wi), ws, 'Color', wc, 'LineWidth', ww);
        else, plot(ax, hist(d1,wi), hist(d2,wi), ls, 'Color', clr, 'LineWidth', lw); end
        plot(ax, hist(d1,end), hist(d2,end), 'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
        % Terminal covariance ellipse
        if ~isempty(comp.covariance)
            C = comp.covariance; nT = size(C,1);
            if nT >= sd
                Ct = C(nT-sd+1:nT, nT-sd+1:nT);
                draw_ellipse_(ax, hist(dims,end), Ct(dims,dims), ns, clr, lw);
            end
        end
    elseif length(dims) == 3
        d1=dims(1); d2=dims(2); d3=dims(3);
        if fl > 0, plot3(ax, hist(d1,1:fl+1), hist(d2,1:fl+1), hist(d3,1:fl+1), ls, 'Color', clr, 'LineWidth', lw); end
        wi = max(1,fl+1):T;
        if ~isempty(wc), plot3(ax, hist(d1,wi), hist(d2,wi), hist(d3,wi), ws, 'Color', wc, 'LineWidth', ww);
        else, plot3(ax, hist(d1,wi), hist(d2,wi), hist(d3,wi), ls, 'Color', clr, 'LineWidth', lw); end
        plot3(ax, hist(d1,end), hist(d2,end), hist(d3,end), 'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
        view(ax, 3);
    end
end

%% ---- Trajectory GGIW ----
function plot_traj_ggiw_(comp, dims, ax, clr, hc, lw, ms, ls, wc, ww, ws)
    if isempty(comp.mean_history), return; end
    hist = comp.mean_history; sd = comp.state_dim;
    if sd == 0, sd = size(hist,1); end
    T = size(hist, 2);
    win_len = T;
    if sd > 0 && ~isempty(comp.mean)
        win_len = min(T, numel(comp.mean) / sd);
    end
    fl = T - win_len;
    mu2 = hist(dims, end);
    d = size(comp.V, 1);
    me = comp.V(dims,dims) / (comp.v - d - 1);

    if length(dims) == 2
        d1 = dims(1); d2 = dims(2);
        if fl > 0, plot(ax, hist(d1,1:fl+1), hist(d2,1:fl+1), ls, 'Color', clr, 'LineWidth', lw); end
        wi = max(1,fl+1):T;
        if ~isempty(wc), plot(ax, hist(d1,wi), hist(d2,wi), ws, 'Color', wc, 'LineWidth', ww);
        else, plot(ax, hist(d1,wi), hist(d2,wi), ls, 'Color', clr, 'LineWidth', lw); end
        [Ev, Ed] = eig(me); t = linspace(0,2*pi,100);
        a = sqrt(Ed(1,1)); b = sqrt(Ed(2,2));
        ell = Ev*[a*cos(t); b*sin(t)] + mu2;
        plot(ax, ell(1,:), ell(2,:), ls, 'Color', clr, 'LineWidth', lw);
        sc = sqrt(chi2inv(hc, 2));
        ell_c = Ev*[sc*a*cos(t); sc*b*sin(t)] + mu2;
        plot(ax, ell_c(1,:), ell_c(2,:), '--', 'Color', clr, 'LineWidth', lw);
    elseif length(dims) == 3
        d1=dims(1); d2=dims(2); d3=dims(3);
        if fl > 0, plot3(ax, hist(d1,1:fl+1), hist(d2,1:fl+1), hist(d3,1:fl+1), ls, 'Color', clr, 'LineWidth', lw); end
        wi = max(1,fl+1):T;
        if ~isempty(wc), plot3(ax, hist(d1,wi), hist(d2,wi), hist(d3,wi), ws, 'Color', wc, 'LineWidth', ww);
        else, plot3(ax, hist(d1,wi), hist(d2,wi), hist(d3,wi), ls, 'Color', clr, 'LineWidth', lw); end
        draw_extent_3d_(ax, mu2, me, hc, clr, lw);
        plot3(ax, mu2(1), mu2(2), mu2(3), 'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
        view(ax, 3);
    end
end

%% ---- Shared helpers ----
function draw_ellipse_(ax, mu, S, ns, clr, lw)
    [V, D] = eig(S); t = linspace(0,2*pi,100);
    a = ns*sqrt(D(1,1)); b = ns*sqrt(D(2,2));
    ell = V*[a*cos(t); b*sin(t)] + mu;
    patch(ax, ell(1,:), ell(2,:), '--', 'FaceColor', clr, 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
end

function draw_ellipsoid_(ax, mu, S, ns, clr, lw, ms)
    [V, D] = eig(S); [x,y,z] = sphere(30); xyz = [x(:) y(:) z(:)]';
    sc = ns*sqrt(diag(D));
    pts = V*diag(sc)*xyz + mu;
    X = reshape(pts(1,:),size(x)); Y = reshape(pts(2,:),size(y)); Z = reshape(pts(3,:),size(z));
    surf(ax, X, Y, Z, 'FaceAlpha', 0.3, 'EdgeColor', clr, 'LineStyle', '--', 'FaceColor', clr);
    plot3(ax, mu(1), mu(2), mu(3), '*', 'Color', clr, 'MarkerSize', ms);
    view(ax, 3);
end

function draw_extent_3d_(ax, mu, me, hc, clr, lw)
    [Vm, Dm] = eig(me); [x,y,z] = sphere(30); xyz = [x(:) y(:) z(:)]';
    pts = Vm*diag(sqrt(diag(Dm)))*xyz + mu;
    X = reshape(pts(1,:),size(x)); Y = reshape(pts(2,:),size(y)); Z = reshape(pts(3,:),size(z));
    surf(ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, 'EdgeColor', clr, ...
        'EdgeAlpha', 0.3, 'FaceColor', clr);
    sc = sqrt(chi2inv(hc, 3));
    pts_c = Vm*diag(sc*sqrt(diag(Dm)))*xyz + mu;
    Xc = reshape(pts_c(1,:),size(x)); Yc = reshape(pts_c(2,:),size(y)); Zc = reshape(pts_c(3,:),size(z));
    surf(ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, 'EdgeColor', clr, ...
        'EdgeAlpha', 0.3, 'LineStyle', '--', 'FaceColor', clr);
    view(ax, 3);
end
