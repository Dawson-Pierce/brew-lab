classdef TrajectoryGGIW < BREW.distributions.TrajectoryBaseModel
    % Represents a Gaussian distribution object.
    
    properties
        alpha 
        beta 
        means
        covariances
        IWdofs 
        IWshapes 
        d 
    end 
    
    methods
        function obj = TrajectoryGGIW(idx, alpha, beta, meanVals, covVals, IWdofs, IWshapes,varargin) 
            if nargin < 1, alpha = []; end
            if nargin < 2, beta = []; end
            if nargin < 3, meanVals = []; end
            if nargin < 4, covVals = []; end
            if nargin < 5, IWdofs = []; end
            if nargin < 6, IWshapes = []; end

            p = inputParser;
            addParameter(p,'state_dim',length(meanVals));

            parse(p, varargin{:});
            
            obj@BREW.distributions.TrajectoryBaseModel(idx, p.Results.state_dim); 
            obj.alpha = alpha;
            obj.beta = beta;
            obj.means = meanVals;
            obj.covariances = covVals;
            obj.IWdofs = IWdofs;
            obj.IWshapes = IWshapes;
            if ~isempty(IWshapes)
                obj.d = size(IWshapes,1);
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
            states = reshape(obj.means, [obj.state_dim obj.window_size]);
        end

        function plot(obj, plt_inds, varargin)
            p = inputParser;
            addParameter(p,'c','w');
            addParameter(p,'ax',gca)
            addParameter(p,'LineWidth',0.8);
            addParameter(p,'LineStyle','-'); 
            addParameter(p,'h',0);
            addParameter(p,'delta',[]);
            parse(p, varargin{:});

            states = obj.RearrangeStates();

            V = obj.IWshapes(:,:,end); 
            v = obj.IWdofs(end);   
            V = V(plt_inds, plt_inds);

            d = length(plt_inds);
            h = p.Results.h;

            if v <= d+1
                    error('Degrees of freedom must exceed d+1 for valid IW mean.');
            end

            mean_extent = V / (v - d - 1);

            if length(plt_inds) == 3 
                plot3(p.Results.ax,states(plt_inds(1),:), states(plt_inds(2),:), states(plt_inds(3),:), 'Color', p.Results.c, 'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth); hold on
                xlabel('X-axis');
                ylabel('Y-axis');
                zlabel('Z-axis');

                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                ellipsoid_pts = Vmean * diag(sqrt(diag(Dmean))) * xyz + states(plt_inds,end);
                X = reshape(ellipsoid_pts(1,:), size(x));
                Y = reshape(ellipsoid_pts(2,:), size(y));
                Z = reshape(ellipsoid_pts(3,:), size(z));
                surf(p.Results.ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, 'EdgeColor', p.Results.c,'EdgeAlpha',0.3, 'LineStyle', '--', 'FaceColor', p.Results.c, 'DisplayName', 'mean extent');
                scale = sqrt(chi2inv(h, 3));
                conf_extent = mean_extent * scale^2;
                [Vconf, Dconf] = eig(conf_extent);
                ellipsoid_pts_conf = Vconf * diag(sqrt(diag(Dconf))) * xyz + states(plt_inds,end);
                Xc = reshape(ellipsoid_pts_conf(1,:), size(x));
                Yc = reshape(ellipsoid_pts_conf(2,:), size(y));
                Zc = reshape(ellipsoid_pts_conf(3,:), size(z));
                surf(p.Results.ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, 'EdgeColor', p.Results.c,'EdgeAlpha',0.3, 'LineStyle', '--', 'FaceColor', p.Results.c, 'DisplayName', sprintf('%3.2f%% confidence interval',h*100));
            else
                plot(p.Results.ax, states(plt_inds(1),:), states(plt_inds(2),:), 'Color', p.Results.c, 'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth); hold on
                xlabel('X-axis');
                ylabel('Y-axis'); 

                [Evec, Eval] = eig(mean_extent);
                t = linspace(0, 2*pi, 100);
                scale = sqrt(chi2inv(h, 2));
                a = scale * sqrt(Eval(1,1)); 
                b = scale * sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + states(plt_inds,end); 
                plot(p.Results.ax, ellipse(1,:), ellipse(2,:), '--', 'Color', p.Results.c, 'DisplayName', sprintf("%3.2f%% confidence interval",h*100));
                a = sqrt(Eval(1,1)); 
                b = sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + states(plt_inds,end); 
                plot(p.Results.ax, ellipse(1,:), ellipse(2,:), '-', 'Color', p.Results.c, 'DisplayName', "mean extent");
            end

            if ~isempty(p.Results.delta)
                for kk = 1:p.Results.delta:size(states,2)
                    % plot ellipses here
                end
            end

        end 

    end
end 