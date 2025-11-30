classdef GGIW < BREW.distributions.BaseSingleModel
    % Represents a Gamma Gaussian Inverse Wishart Distribution.
    
    properties
        alpha 
        beta 
        mean
        covariance
        IWdof 
        IWshape 
        d 
        eigs
        basis
    end 
    
    methods
        function obj = GGIW(alpha, beta, meanVal, covVal, IWdof, IWshape) 
            if nargin < 1, alpha = []; end
            if nargin < 2, beta = []; end
            if nargin < 3, meanVal = []; end
            if nargin < 4, covVal = []; end
            if nargin < 5, IWdof = []; end
            if nargin < 6, IWshape = []; end
            obj@BREW.distributions.BaseSingleModel();
            obj.alpha = alpha;
            obj.beta = beta;
            obj.mean = meanVal;
            obj.covariance = covVal;
            obj.IWdof = IWdof;
            obj.IWshape = IWshape;
            if ~isempty(IWshape)
                obj.d = size(IWshape,1);
            end
        end
                
        function measurements = sample_measurements(obj, xy_inds, random_extent) 
            if nargin < 2 || isempty(xy_inds)
                if obj.d == 2
                    xy_inds = [1 2]; 
                elseif obj.d == 3
                    xy_inds = [1 2 3]; 
                else
                    error("Invalid IW Shape.")
                end
            end
            if nargin < 3, random_extent = false; end
            lam = obj.alpha / obj.beta;
            N = poissrnd(lam);
            center = obj.mean(xy_inds);
            if random_extent
                extent = iwishrnd(obj.IWshape, obj.IWdof);
            else
                extent = obj.IWshape / (obj.IWdof + obj.d + 1);
            end
            if N > 0
                measurements = mvnrnd(center(:)', extent, N)';
            else
                measurements = zeros(obj.d, 0);
            end
        end
        
        function disp_params(obj) 
            fprintf('Gamma: alpha = %g, beta = %g\n', obj.alpha, obj.beta);
            fprintf('Gaussian mean = \n');
            disp(obj.mean);
            fprintf('Gaussian covariance = \n');
            disp(obj.covariance);
            fprintf('IW dof = %g\n', obj.IWdof);
            fprintf('IW shape = \n');
            disp(obj.IWshape);
        end
        
        function plot(obj, plt_inds, varargin)
            p = inputParser;
            p.KeepUnmatched = true; 
            addParameter(p,'color','k');
            addParameter(p,'ax',gca)
            addParameter(p,'h',0.95)
            addParameter(p,'LineWidth',0.8);
            addParameter(p,'LineStyle','-'); 
            parse(p, varargin{:});

            q = inputParser;
            addParameter(q,'h_LineStyle',p.Results.LineStyle)

            parse(q, p.Unmatched)

            ax = p.Results.ax;
            h = p.Results.h;

            mu = obj.mean;
            V = obj.IWshape;
            v = obj.IWdof;

            mu2 = mu(plt_inds); 
            V2 = V(plt_inds, plt_inds);

            if length(plt_inds) == 3 

                mean_extent = V2 / (v - obj.d - 1);
                [Vmean, Dmean] = eig(mean_extent);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                ellipsoid_pts = Vmean * diag(sqrt(diag(Dmean))) * xyz + mu2;
                X = reshape(ellipsoid_pts(1,:), size(x));
                Y = reshape(ellipsoid_pts(2,:), size(y));
                Z = reshape(ellipsoid_pts(3,:), size(z));
                surf(ax, real(X), real(Y), real(Z), 'FaceAlpha', 0.3, 'EdgeColor', p.Results.color,'EdgeAlpha',0.3, 'LineStyle', p.Results.LineStyle, 'FaceColor', p.Results.color, 'DisplayName', 'mean extent');
                scale = sqrt(chi2inv(h, 3));
                conf_extent = mean_extent * scale^2;
                [Vconf, Dconf] = eig(conf_extent);
                ellipsoid_pts_conf = Vconf * diag(sqrt(diag(Dconf))) * xyz + mu2;
                Xc = reshape(ellipsoid_pts_conf(1,:), size(x));
                Yc = reshape(ellipsoid_pts_conf(2,:), size(y));
                Zc = reshape(ellipsoid_pts_conf(3,:), size(z));
                surf(ax, real(Xc), real(Yc), real(Zc), 'FaceAlpha', 0.15, 'EdgeColor', p.Results.color,'EdgeAlpha',0.3, 'LineStyle', q.Results.h_LineStyle, 'FaceColor', p.Results.color, 'DisplayName', sprintf('%3.2f%% confidence interval',h*100));
                
                % xlabel('X-axis');
                % ylabel('Y-axis');
                % zlabel('Z-axis');

            elseif length(plt_inds) == 2 

                mean_extent = V2 / (v - obj.d - 1);
                [Evec, Eval] = eig(mean_extent);
                t = linspace(0, 2*pi, 100);
                scale = sqrt(chi2inv(h, 2));
                a = scale * sqrt(Eval(1,1)); 
                b = scale * sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2; 
                plot(ax, ellipse(1,:), ellipse(2,:), q.Results.h_LineStyle, 'Color', p.Results.color, 'LineWidth',p.Results.LineWidth, 'DisplayName', sprintf("%3.2f%% confidence interval",h*100));
                a = sqrt(Eval(1,1)); 
                b = sqrt(Eval(2,2));
                ellipse = Evec*[a*cos(t); b*sin(t)] + mu2; 
                plot(ax, ellipse(1,:), ellipse(2,:), p.Results.LineStyle, 'Color', p.Results.color, 'LineWidth',p.Results.LineWidth, 'DisplayName', "mean extent");
                
                % xlabel('X-axis');
                % ylabel('Y-axis'); 
                
            else
                disp("error plotting - please have plt_inds be a length of 2 or 3")
            end
        end
    end
end 