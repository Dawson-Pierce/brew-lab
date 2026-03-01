classdef Mixture < handle
    %MIXTURE Generic mixture data class returned from RFS filter extraction.
    %   Holds component means, covariances, weights, and distribution-specific
    %   fields (trajectory histories, GGIW parameters, etc.).
    %
    %   Constructed automatically by RFS filter cleanup/extract methods,
    %   or manually for truth model simulation.

    properties (SetAccess = public)
        means           cell   = {}      % {mean1, mean2, ...} column vectors
        covariances     cell   = {}      % {cov1, cov2, ...} matrices
        weights         double = []      % [w1, w2, ...] row vector
        dist_type       string = ""      % "Gaussian", "TrajectoryGaussian", "GGIW", "TrajectoryGGIW"

        % TrajectoryGaussian / TrajectoryGGIW fields
        state_dim       double = 0
        init_indices    double = []
        mean_histories  cell   = {}      % {state_dim x T matrices}

        % GGIW / TrajectoryGGIW fields
        alphas          double = []
        betas           double = []
        vs              double = []
        Vs              cell   = {}
    end

    methods
        function obj = Mixture(raw, dist_type_override)
            %MIXTURE Construct from a raw struct (returned by brew_mex) or manually.
            if nargin == 0, return; end
            if nargin >= 2 && ~isempty(dist_type_override)
                obj.dist_type = string(dist_type_override);
            end

            if isstruct(raw)
                if isfield(raw, 'means'),       obj.means       = raw.means;       end
                if isfield(raw, 'covariances'), obj.covariances = raw.covariances; end
                if isfield(raw, 'weights'),     obj.weights     = raw.weights(:)'; end
                if isfield(raw, 'dist_type') && isempty(dist_type_override)
                    obj.dist_type = string(raw.dist_type);
                end
                if isfield(raw, 'state_dim'),      obj.state_dim      = raw.state_dim;      end
                if isfield(raw, 'init_indices'),    obj.init_indices    = raw.init_indices;    end
                if isfield(raw, 'mean_histories'),  obj.mean_histories  = raw.mean_histories;  end
                if isfield(raw, 'alphas'), obj.alphas = raw.alphas; end
                if isfield(raw, 'betas'),  obj.betas  = raw.betas;  end
                if isfield(raw, 'vs'),     obj.vs     = raw.vs;     end
                if isfield(raw, 'Vs'),     obj.Vs     = raw.Vs;     end
            end
        end

        function n = length(obj)
            n = numel(obj.weights);
        end

        function n = size(obj)
            n = numel(obj.weights);
        end

        function tf = isempty(obj)
            tf = isempty(obj.weights);
        end

        %% ---- Sampling ----

        function measurements = sample_measurements(obj, xy_inds, varargin)
            %SAMPLE_MEASUREMENTS Draw measurements from each mixture component.
            %   meas = obj.sample_measurements(xy_inds)
            %   meas = obj.sample_measurements(xy_inds, meas_cov)  % for Gaussian
            %
            %   For Gaussian/TrajectoryGaussian: one measurement per component
            %     drawn from mvnrnd using the component mean and meas_cov.
            %   For GGIW/TrajectoryGGIW: Poisson-distributed number of
            %     measurements per component drawn from the extent.

            if nargin < 2 || isempty(xy_inds)
                xy_inds = 1:min(3, numel(obj.means{1}));
            end

            all_meas = [];

            if contains(obj.dist_type, "GGIW")
                % GGIW sampling: Poisson count, mvnrnd from extent
                for i = 1:numel(obj.weights)
                    mu = obj.means{i};
                    alpha_i = obj.alphas(i);
                    beta_i  = obj.betas(i);
                    V_i     = obj.Vs{i};
                    v_i     = obj.vs(i);
                    d       = size(V_i, 1);

                    % For trajectory types, get the last state
                    if contains(obj.dist_type, "Trajectory") && ~isempty(obj.mean_histories)
                        hist = obj.mean_histories{i};
                        sd = obj.state_dim;
                        if sd > 0
                            mu = hist(:, end);
                        end
                    end

                    center = mu(xy_inds);
                    lam = alpha_i / beta_i;
                    N = poissrnd(lam);
                    extent = V_i / (v_i - d - 1);

                    if N > 0
                        mi = mvnrnd(center(:)', extent, N)';
                    else
                        mi = zeros(length(xy_inds), 0);
                    end
                    all_meas = [all_meas, mi]; %#ok<AGROW>
                end
            else
                % Gaussian sampling: one measurement per component
                meas_cov = [];
                if nargin >= 3, meas_cov = varargin{1}; end

                for i = 1:numel(obj.weights)
                    mu = obj.means{i};

                    % For trajectory types, get the last state
                    if contains(obj.dist_type, "Trajectory") && ~isempty(obj.mean_histories)
                        hist = obj.mean_histories{i};
                        sd = obj.state_dim;
                        if sd > 0
                            mu = hist(:, end);
                        end
                    end

                    center = mu(xy_inds);
                    if ~isempty(meas_cov)
                        C = meas_cov;
                    else
                        C = obj.covariances{i}(xy_inds, xy_inds);
                    end
                    mi = mvnrnd(center(:)', C, 1)';
                    all_meas = [all_meas, mi]; %#ok<AGROW>
                end
            end

            measurements = all_meas;
        end

        %% ---- Plotting ----

        function plot_distributions(obj, plt_inds, varargin)
            %PLOT_DISTRIBUTIONS Plot extracted mixture components with visualization.
            %   plot_distributions(plt_inds, 'Name', Value, ...)
            %
            %   plt_inds    - indices of state dimensions to plot (e.g., [1 2])
            %
            %   Name-Value pairs:
            %     'ax'           - axes handle (default gca)
            %     'colors'       - Nx3 color matrix or single color (default: lines(n))
            %     'c'            - alias for 'colors'
            %     'LineWidth'    - line width (default 2)
            %     'MarkerSize'   - marker size (default 4)
            %     'num_std'      - number of std devs for Gaussian ellipse (default 2)
            %     'h'            - confidence level for GGIW chi2 (default 0.95)
            %     'window_color' - trajectory window highlight color (default [])
            %     'window_width' - trajectory window line width (default LineWidth)
            %     'window_style' - trajectory window line style (default '-')
            %     'LineStyle'    - line style (default '-')

            if nargin < 2, plt_inds = 1; end

            n = numel(obj.weights);
            if n == 0, return; end

            ip = inputParser;
            ip.KeepUnmatched = true;
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

            ax  = ip.Results.ax;
            lw  = ip.Results.LineWidth;
            ms  = ip.Results.MarkerSize;
            ls  = ip.Results.LineStyle;
            ns  = ip.Results.num_std;
            h   = ip.Results.h;
            wc  = ip.Results.window_color;
            ww  = ip.Results.window_width;
            ws  = ip.Results.window_style;
            if isempty(ww), ww = lw; end

            % Resolve colors
            colors = ip.Results.colors;
            if isempty(colors), colors = ip.Results.c; end
            if isempty(colors), colors = lines(n); end
            if (ischar(colors) && size(colors,1)==1) || (isnumeric(colors) && isequal(size(colors),[1,3]))
                colors = repmat(colors, n, 1);
            end

            hold_state = ishold(ax);
            hold(ax, 'on');

            % Dispatch by distribution type
            switch obj.dist_type
                case "Gaussian"
                    for i = 1:n
                        clr = colors(i,:);
                        obj.plot_gaussian_(i, plt_inds, ax, clr, ns, lw, ms, ls);
                    end
                case "GGIW"
                    for i = 1:n
                        clr = colors(i,:);
                        obj.plot_ggiw_(i, plt_inds, ax, clr, h, lw, ls);
                    end
                case "TrajectoryGaussian"
                    for i = 1:n
                        clr = colors(i,:);
                        obj.plot_traj_gaussian_(i, plt_inds, ax, clr, ns, lw, ms, ls, wc, ww, ws);
                    end
                case "TrajectoryGGIW"
                    for i = 1:n
                        clr = colors(i,:);
                        obj.plot_traj_ggiw_(i, plt_inds, ax, clr, h, lw, ms, ls, wc, ww, ws);
                    end
                otherwise
                    % Fallback: just plot point estimates
                    for i = 1:n
                        m = obj.means{i};
                        clr = colors(i,:);
                        if length(plt_inds) >= 3
                            plot3(ax, m(plt_inds(1)), m(plt_inds(2)), m(plt_inds(3)), ...
                                'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
                        elseif length(plt_inds) == 2
                            plot(ax, m(plt_inds(1)), m(plt_inds(2)), ...
                                'o', 'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
                        end
                    end
            end

            if ~hold_state, hold(ax, 'off'); end
        end
    end

    methods (Access = private)

        function plot_gaussian_(obj, i, plt_inds, ax, clr, num_std, lw, ms, ls)
            mu = obj.means{i}(:);
            Sigma = obj.covariances{i};
            mu2 = mu(plt_inds);
            Sigma2 = Sigma(plt_inds, plt_inds);

            if length(plt_inds) == 1
                % 1D: PDF curve
                x = linspace(mu2 - num_std*sqrt(Sigma2), mu2 + num_std*sqrt(Sigma2), 200);
                y = normpdf(x, mu2, sqrt(Sigma2));
                plot(ax, x, y, 'Color', clr, 'LineWidth', lw, 'LineStyle', ls);
                plot(ax, mu2, normpdf(mu2, mu2, sqrt(Sigma2)), '*', 'Color', clr);
            elseif length(plt_inds) == 2
                % 2D: covariance ellipse patch + mean marker
                plot(ax, mu2(1), mu2(2), '*', 'Color', clr, 'MarkerSize', ms);
                [V, D] = eig(Sigma2);
                t = linspace(0, 2*pi, 100);
                a = num_std * sqrt(D(1,1));
                b = num_std * sqrt(D(2,2));
                ellipse = V * [a*cos(t); b*sin(t)] + mu2;
                patch(ax, ellipse(1,:), ellipse(2,:), '--', 'FaceColor', clr, ...
                    'FaceAlpha', 0.3, 'EdgeAlpha', 0);
            elseif length(plt_inds) == 3
                % 3D: covariance ellipsoid
                [V, D] = eig(Sigma2);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                scale = num_std * sqrt(diag(D));
                pts = V * diag(scale) * xyz + mu2;
                X = reshape(pts(1,:), size(x));
                Y = reshape(pts(2,:), size(y));
                Z = reshape(pts(3,:), size(z));
                surf(ax, X, Y, Z, 'FaceAlpha', 0.3, 'EdgeColor', clr, ...
                    'LineStyle', '--', 'FaceColor', clr);
                plot3(ax, mu2(1), mu2(2), mu2(3), '*', 'Color', clr, 'MarkerSize', ms);
                view(ax, 3);
            end
        end

        function plot_ggiw_(obj, i, plt_inds, ax, clr, h_conf, lw, ls)
            mu = obj.means{i}(:);
            V_i = obj.Vs{i};
            v_i = obj.vs(i);
            d = size(V_i, 1);

            mu2 = mu(plt_inds);
            V2 = V_i(plt_inds, plt_inds);
            mean_extent = V2 / (v_i - d - 1);

            if length(plt_inds) == 2
                % 2D: mean extent ellipse + confidence ellipse
                [Evec, Eval] = eig(mean_extent);
                t = linspace(0, 2*pi, 100);

                % Mean extent ellipse (solid)
                a = sqrt(Eval(1,1));
                b = sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2;
                plot(ax, ellipse(1,:), ellipse(2,:), ls, 'Color', clr, ...
                    'LineWidth', lw, 'DisplayName', 'mean extent');

                % Confidence ellipse (dashed)
                scale = sqrt(chi2inv(h_conf, 2));
                a_c = scale * sqrt(Eval(1,1));
                b_c = scale * sqrt(Eval(2,2));
                ellipse_c = Evec*[a_c*cos(t); b_c*sin(t)] + mu2;
                plot(ax, ellipse_c(1,:), ellipse_c(2,:), '--', 'Color', clr, ...
                    'LineWidth', lw, 'DisplayName', sprintf('%3.0f%% confidence', h_conf*100));

            elseif length(plt_inds) == 3
                % 3D: mean extent ellipsoid + confidence ellipsoid
                [Vmean, Dmean] = eig(mean_extent);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';

                % Mean extent ellipsoid
                pts = Vmean * diag(sqrt(diag(Dmean))) * xyz + mu2;
                X = reshape(pts(1,:), size(x));
                Y = reshape(pts(2,:), size(y));
                Z = reshape(pts(3,:), size(z));
                surf(ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, ...
                    'EdgeColor', clr, 'EdgeAlpha', 0.3, 'LineStyle', ls, ...
                    'FaceColor', clr, 'DisplayName', 'mean extent');

                % Confidence ellipsoid
                scale = sqrt(chi2inv(h_conf, 3));
                conf_extent = mean_extent * scale^2;
                [Vconf, Dconf] = eig(conf_extent);
                pts_c = Vconf * diag(sqrt(diag(Dconf))) * xyz + mu2;
                Xc = reshape(pts_c(1,:), size(x));
                Yc = reshape(pts_c(2,:), size(y));
                Zc = reshape(pts_c(3,:), size(z));
                surf(ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, ...
                    'EdgeColor', clr, 'EdgeAlpha', 0.3, 'LineStyle', '--', ...
                    'FaceColor', clr, 'DisplayName', sprintf('%3.0f%% confidence', h_conf*100));
                view(ax, 3);
            end
        end

        function plot_traj_gaussian_(obj, i, plt_inds, ax, clr, num_std, lw, ms, ls, wc, ww, ws)
            if isempty(obj.mean_histories) || i > numel(obj.mean_histories)
                return;
            end
            hist = obj.mean_histories{i};
            if isempty(hist), return; end
            sd = obj.state_dim;
            if sd == 0, sd = size(hist, 1); end

            T = size(hist, 2);
            idx0 = obj.init_indices(i);
            x_ax = idx0 : (idx0 + T - 1);

            % Determine window length from stacked mean
            win_len = T;
            if i <= numel(obj.means) && ~isempty(obj.means{i}) && sd > 0
                win_len = min(T, numel(obj.means{i}) / sd);
            end
            frozen_len = T - win_len;

            % Plot trajectory path
            if length(plt_inds) == 1
                dim = plt_inds(1);
                if dim > sd, return; end
                y_ax = hist(dim, :);
                % Frozen portion
                if frozen_len > 0
                    fi = 1:frozen_len+1;
                    plot(ax, x_ax(fi), y_ax(fi), ls, 'Color', clr, 'LineWidth', lw);
                end
                % Window portion
                if win_len > 0 && ~isempty(wc)
                    wi = max(1, frozen_len+1):T;
                    plot(ax, x_ax(wi), y_ax(wi), ws, 'Color', wc, 'LineWidth', ww);
                elseif win_len > 0
                    wi = max(1, frozen_len+1):T;
                    plot(ax, x_ax(wi), y_ax(wi), ls, 'Color', clr, 'LineWidth', lw);
                end
                plot(ax, x_ax(end), y_ax(end), 'o', 'Color', clr, ...
                    'MarkerFaceColor', clr, 'MarkerSize', ms);
            elseif length(plt_inds) == 2
                d1 = plt_inds(1); d2 = plt_inds(2);
                if d1 > sd || d2 > sd, return; end
                % Frozen portion
                if frozen_len > 0
                    fi = 1:frozen_len+1;
                    plot(ax, hist(d1, fi), hist(d2, fi), ls, 'Color', clr, 'LineWidth', lw);
                end
                % Window portion
                if win_len > 0 && ~isempty(wc)
                    wi = max(1, frozen_len+1):T;
                    plot(ax, hist(d1, wi), hist(d2, wi), ws, 'Color', wc, 'LineWidth', ww);
                elseif win_len > 0
                    wi = max(1, frozen_len+1):T;
                    plot(ax, hist(d1, wi), hist(d2, wi), ls, 'Color', clr, 'LineWidth', lw);
                end
                plot(ax, hist(d1, end), hist(d2, end), 'o', 'Color', clr, ...
                    'MarkerFaceColor', clr, 'MarkerSize', ms);

                % Terminal covariance ellipse
                if i <= numel(obj.covariances) && ~isempty(obj.covariances{i})
                    C = obj.covariances{i};
                    nTot = size(C, 1);
                    if nTot >= sd
                        idx_end = (nTot - sd + 1):nTot;
                        Cterm = C(idx_end, idx_end);
                        Cterm2 = Cterm(plt_inds, plt_inds);
                        mu_end = hist(plt_inds, end);
                        [V, D] = eig(Cterm2);
                        t = linspace(0, 2*pi, 100);
                        a = num_std * sqrt(D(1,1));
                        b = num_std * sqrt(D(2,2));
                        ell = V * [a*cos(t); b*sin(t)] + mu_end;
                        patch(ax, ell(1,:), ell(2,:), '--', 'FaceColor', clr, ...
                            'FaceAlpha', 0.2, 'EdgeAlpha', 0);
                    end
                end
            elseif length(plt_inds) == 3
                d1 = plt_inds(1); d2 = plt_inds(2); d3 = plt_inds(3);
                if d1 > sd || d2 > sd || d3 > sd, return; end
                % Frozen portion
                if frozen_len > 0
                    fi = 1:frozen_len+1;
                    plot3(ax, hist(d1, fi), hist(d2, fi), hist(d3, fi), ...
                        ls, 'Color', clr, 'LineWidth', lw);
                end
                % Window portion
                if win_len > 0 && ~isempty(wc)
                    wi = max(1, frozen_len+1):T;
                    plot3(ax, hist(d1, wi), hist(d2, wi), hist(d3, wi), ...
                        ws, 'Color', wc, 'LineWidth', ww);
                elseif win_len > 0
                    wi = max(1, frozen_len+1):T;
                    plot3(ax, hist(d1, wi), hist(d2, wi), hist(d3, wi), ...
                        ls, 'Color', clr, 'LineWidth', lw);
                end
                plot3(ax, hist(d1, end), hist(d2, end), hist(d3, end), 'o', ...
                    'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
                view(ax, 3);
            end
        end

        function plot_traj_ggiw_(obj, i, plt_inds, ax, clr, h_conf, lw, ms, ls, wc, ww, ws)
            if isempty(obj.mean_histories) || i > numel(obj.mean_histories)
                return;
            end
            hist = obj.mean_histories{i};
            if isempty(hist), return; end
            sd = obj.state_dim;
            if sd == 0, sd = size(hist, 1); end

            T = size(hist, 2);
            idx0 = obj.init_indices(i);
            x_ax = idx0 : (idx0 + T - 1);

            % Determine window length
            win_len = T;
            if i <= numel(obj.means) && ~isempty(obj.means{i}) && sd > 0
                win_len = min(T, numel(obj.means{i}) / sd);
            end
            frozen_len = T - win_len;

            % GGIW extent at terminal state
            V_i = obj.Vs{i};
            v_i = obj.vs(i);
            d = size(V_i, 1);
            mu_end = hist(:, end);
            mu2 = mu_end(plt_inds);
            V2 = V_i(plt_inds, plt_inds);
            mean_extent = V2 / (v_i - d - 1);

            if length(plt_inds) == 2
                d1 = plt_inds(1); d2 = plt_inds(2);
                % Frozen portion
                if frozen_len > 0
                    fi = 1:frozen_len+1;
                    plot(ax, hist(d1, fi), hist(d2, fi), ls, 'Color', clr, 'LineWidth', lw);
                end
                % Window portion
                if win_len > 0 && ~isempty(wc)
                    wi = max(1, frozen_len+1):T;
                    plot(ax, hist(d1, wi), hist(d2, wi), ws, 'Color', wc, 'LineWidth', ww);
                elseif win_len > 0
                    wi = max(1, frozen_len+1):T;
                    plot(ax, hist(d1, wi), hist(d2, wi), ls, 'Color', clr, 'LineWidth', lw);
                end

                % Mean extent ellipse
                [Evec, Eval] = eig(mean_extent);
                t = linspace(0, 2*pi, 100);
                a = sqrt(Eval(1,1));
                b = sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2;
                plot(ax, ellipse(1,:), ellipse(2,:), ls, 'Color', clr, ...
                    'LineWidth', lw, 'DisplayName', 'mean extent');

                % Confidence ellipse
                scale = sqrt(chi2inv(h_conf, 2));
                a_c = scale * sqrt(Eval(1,1));
                b_c = scale * sqrt(Eval(2,2));
                ell_c = Evec*[a_c*cos(t); b_c*sin(t)] + mu2;
                plot(ax, ell_c(1,:), ell_c(2,:), '--', 'Color', clr, ...
                    'LineWidth', lw, 'DisplayName', sprintf('%3.0f%% confidence', h_conf*100));

            elseif length(plt_inds) == 3
                d1 = plt_inds(1); d2 = plt_inds(2); d3 = plt_inds(3);
                % Frozen portion
                if frozen_len > 0
                    fi = 1:frozen_len+1;
                    plot3(ax, hist(d1, fi), hist(d2, fi), hist(d3, fi), ...
                        ls, 'Color', clr, 'LineWidth', lw);
                end
                % Window portion
                if win_len > 0 && ~isempty(wc)
                    wi = max(1, frozen_len+1):T;
                    plot3(ax, hist(d1, wi), hist(d2, wi), hist(d3, wi), ...
                        ws, 'Color', wc, 'LineWidth', ww);
                elseif win_len > 0
                    wi = max(1, frozen_len+1):T;
                    plot3(ax, hist(d1, wi), hist(d2, wi), hist(d3, wi), ...
                        ls, 'Color', clr, 'LineWidth', lw);
                end

                % Mean extent ellipsoid
                [Vmean, Dmean] = eig(mean_extent);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                pts = Vmean * diag(sqrt(diag(Dmean))) * xyz + mu2;
                X = reshape(pts(1,:), size(x));
                Y = reshape(pts(2,:), size(y));
                Z = reshape(pts(3,:), size(z));
                surf(ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, ...
                    'EdgeColor', clr, 'EdgeAlpha', 0.3, 'LineStyle', '--', ...
                    'FaceColor', clr, 'DisplayName', 'mean extent');

                % Confidence ellipsoid
                scale = sqrt(chi2inv(h_conf, 3));
                conf_extent = mean_extent * scale^2;
                [Vconf, Dconf] = eig(conf_extent);
                pts_c = Vconf * diag(sqrt(diag(Dconf))) * xyz + mu2;
                Xc = reshape(pts_c(1,:), size(x));
                Yc = reshape(pts_c(2,:), size(y));
                Zc = reshape(pts_c(3,:), size(z));
                surf(ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, ...
                    'EdgeColor', clr, 'EdgeAlpha', 0.3, 'LineStyle', '--', ...
                    'FaceColor', clr, 'DisplayName', sprintf('%3.0f%% confidence', h_conf*100));

                plot3(ax, mu2(1), mu2(2), mu2(3), 'o', 'Color', clr, ...
                    'MarkerFaceColor', clr, 'MarkerSize', ms);
                view(ax, 3);
            end
        end
    end
end
