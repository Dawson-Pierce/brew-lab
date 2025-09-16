classdef TrajectoryGGIW < BREW.distributions.TrajectoryBaseModel
    % Represents a Gaussian distribution object.
    
    properties 
        means        % trajectory history based on L-scan window
        covariances  % Matrix of covariances based on L-scan window
        mean_history % trajectory history (includes outside L-scan)
        cov_history  % 3D matrix of covariances per mean, doesn't have cross terms
        alphas       % history of alphas (1D)
        betas        % history of betas (1D)
        IWdofs       % history of dof for IW (1D)
        IWshapes     % 3D matrix for history of IW matrices 
        d            % dimension of IW 
    end
    
    methods
        function obj = TrajectoryGGIW(idx, alpha, beta, meanVal, covVal, IWdof, IWshape, varargin)

            p = inputParser;
            addParameter(p,'state_dim',length(meanVal));

            parse(p, varargin{:});
            
            obj@BREW.distributions.TrajectoryBaseModel(idx, p.Results.state_dim);
            
            obj.means = meanVal;
            obj.mean_history = meanVal;
            obj.covariances = covVal;
            obj.cov_history = covVal;

            obj.alphas = alpha;
            obj.betas = beta;
            obj.IWdofs = IWdof;
            obj.IWshapes = IWshape;
            if ~isempty(IWshape)
                obj.d = size(IWshape,1);
            end

        end 
        
        function state = getLastState(obj)
            % Gets most recent state
            state = obj.means((length(obj.means) - obj.state_dim + 1):end,:);
        end

        function cov = getLastCov(obj)
            cov = obj.covariances((length(obj.means) - obj.state_dim + 1):end,(length(obj.means) - obj.state_dim + 1):end);
        end

        function states = RearrangeStates(obj)
            % Gets states from column to N X T matrix 
            % Useful for plotting
            states = reshape(obj.means, [obj.state_dim length(obj.means) / obj.state_dim]);
        end

        function plot(obj, plt_inds, varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'c','w');
            addParameter(p,'ax',gca)
            addParameter(p,'h',0.95)
            addParameter(p,'LineWidth',0.8);
            addParameter(p,'LineStyle','-'); 
            parse(p, varargin{:});


            addParameter(p,'window_style',p.Results.LineStyle); 
            addParameter(p,'window_color',p.Results.c); 
            addParameter(p,'window_width',p.Results.LineWidth); 
            parse(p, varargin{:})

            ax = p.Results.ax;
            h = p.Results.h;

            if all(p.Results.LineStyle == p.Results.window_style) || all(p.Results.c == p.Results.window_color) || all(p.Results.LineWidth == p.Results.window_width) 
                plot_window = true;
            else
                plot_window = false;
            end

            states = obj.mean_history;

            mu = obj.getLastState();
            V = obj.IWshapes(:,:,end);
            v = obj.IWdofs(end);

            mu2 = mu(plt_inds); 
            V2 = V(plt_inds, plt_inds);

            if length(plt_inds) == 3 
                plot3(states(plt_inds(1),:), states(plt_inds(2),:), states(plt_inds(3),:), ...
                    'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth); 
                if plot_window
                    states_window = obj.RearrangeStates();
                    plot3(states_window(plt_inds(1),:), states_window(plt_inds(2),:), states_window(plt_inds(3),:), ...
                        'Color', p.Results.window_color,'LineStyle', p.Results.window_style, 'LineWidth', p.Results.window_width);
                end

                mean_extent = V2 / (v - obj.d - 1);
                [Vmean, Dmean] = eig(mean_extent);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                ellipsoid_pts = Vmean * diag(sqrt(diag(Dmean))) * xyz + mu2;
                X = reshape(ellipsoid_pts(1,:), size(x));
                Y = reshape(ellipsoid_pts(2,:), size(y));
                Z = reshape(ellipsoid_pts(3,:), size(z));
                surf(ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, 'EdgeColor', p.Results.c,'EdgeAlpha',0.3, 'LineStyle', '--', 'FaceColor', p.Results.c, 'DisplayName', 'mean extent');
                scale = sqrt(chi2inv(h, 3));
                conf_extent = mean_extent * scale^2;
                [Vconf, Dconf] = eig(conf_extent);
                ellipsoid_pts_conf = Vconf * diag(sqrt(diag(Dconf))) * xyz + mu2;
                Xc = reshape(ellipsoid_pts_conf(1,:), size(x));
                Yc = reshape(ellipsoid_pts_conf(2,:), size(y));
                Zc = reshape(ellipsoid_pts_conf(3,:), size(z));
                surf(ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, 'EdgeColor', p.Results.c,'EdgeAlpha',0.3, 'LineStyle', '--', 'FaceColor', p.Results.c, 'DisplayName', sprintf('%3.2f%% confidence interval',h*100));
                
                xlabel('X-axis');
                ylabel('Y-axis');
                zlabel('Z-axis');

            elseif length(plt_inds) == 2 
                plot(p.Results.ax, states(plt_inds(1),:), states(plt_inds(2),:), ...
                    'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth);

                if plot_window
                    states_window = obj.RearrangeStates();
                    plot(p.Results.ax, states_window(plt_inds(1),:), states_window(plt_inds(2),:), ...
                        'Color', p.Results.window_color,'LineStyle', p.Results.window_style, 'LineWidth', p.Results.window_width);
                end 

                mean_extent = V2 / (v - obj.d - 1);
                [Evec, Eval] = eig(mean_extent);
                t = linspace(0, 2*pi, 100);
                scale = sqrt(chi2inv(h, 2));
                a = scale * sqrt(Eval(1,1)); 
                b = scale * sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2; 
                plot(ax, ellipse(1,:), ellipse(2,:), '--', 'Color', p.Results.c, 'DisplayName', sprintf("%3.2f%% confidence interval",h*100));
                a = sqrt(Eval(1,1)); 
                b = sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2; 
                plot(ax, ellipse(1,:), ellipse(2,:), '-', 'Color', p.Results.c, 'DisplayName', "mean extent");
                
                xlabel('X-axis');
                ylabel('Y-axis'); 
                
            else
                disp("error plotting - please have plt_inds be a length of 2 or 3")
            end
        end
        

    end
end 