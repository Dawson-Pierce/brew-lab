classdef Gaussian < BREW.distributions.BaseSingleModel
    % Represents a Gaussian distribution object.
    
    properties 
        mean 
        covariance 
    end
    
    methods
        function obj = Gaussian(meanVal, covVal) 
            if nargin < 1, meanVal = []; end
            if nargin < 2, covVal = []; end
            obj@BREW.distributions.BaseSingleModel();

            obj.mean = meanVal;
            obj.covariance = covVal;

        end 
        
        function s = sample(obj, numSamples) 
            if nargin < 2, numSamples = 1; end
            s = mvnrnd(obj.mean(:)', obj.covariance, numSamples)';
        end

        function s = sample_measurements(obj,xy_indices)
            mean_temp = obj.mean(xy_indices);
            cov_temp = obj.covariance(xy_indices,xy_indices);
            s = mvnrnd(mean_temp(:)', cov_temp, 1)';
        end
        
        function p = pdf(obj, x) 
            p = mvnpdf(x(:)', obj.mean(:)', obj.covariance);
        end
        
        function disp(obj) 
            fprintf('Mean = \n');
            disp(obj.mean);
            fprintf('Covariance = \n');
            disp(obj.covariance);
        end
        function plot_distribution(obj, ax, plt_inds, num_std, color) 
            if nargin < 2 || isempty(ax), ax = gca; end
            if nargin < 3 || isempty(plt_inds)
                error('plot_distribution requires plt_inds for which states to plot.');
            end
            if nargin < 4 || isempty(num_std), num_std = 2; end 
            if nargin < 5, color = 'b'; end 
    
            mu = obj.mean(:); 
            Sigma = obj.covariance;
            mu2 = mu(plt_inds); 
            Sigma2 = Sigma(plt_inds, plt_inds);

            if length(mu2) == 1
                x = linspace(mu2 - num_std*sqrt(Sigma2), mu2 + num_std*sqrt(Sigma2), 200);
                y = normpdf(x, mu2, sqrt(Sigma2));
                plot(ax, x, y, 'Color', color, 'DisplayName', sprintf('%d-sigma PDF', num_std));
                hold(ax, 'on');
                plot(ax, mu2, normpdf(mu2, mu2, sqrt(Sigma2)), '*', 'Color', color, 'DisplayName', 'mean');
            elseif length(mu2) == 2
                plot(ax, mu2(1), mu2(2), '*', 'Color', color, 'DisplayName', 'mean'); 
                hold(ax, 'on');
                [V, D] = eig(Sigma2);
                t = linspace(0, 2*pi, 100);
                a = num_std * sqrt(D(1,1)); 
                b = num_std * sqrt(D(2,2));
                ellipse = V * [a*cos(t); b*sin(t)] + mu2;
                patch(ax, ellipse(1,:), ellipse(2,:), '--', 'FaceColor', color, ...
                     'FaceAlpha',0.3,'EdgeAlpha',0, ...
                     'DisplayName', sprintf('%d-sigma ellipse', num_std));
            elseif length(mu2) == 3
                view(ax,3)
                [V, D] = eig(Sigma2);
                [x, y, z] = sphere(30);
                xyz = [x(:) y(:) z(:)]';
                scale = num_std * sqrt(diag(D));
                ellipsoid_pts = V * diag(scale) * xyz + mu2;
                X = reshape(ellipsoid_pts(1,:), size(x));
                Y = reshape(ellipsoid_pts(2,:), size(y));
                Z = reshape(ellipsoid_pts(3,:), size(z));
                surf(ax, X, Y, Z, 'FaceAlpha', 0.3, 'EdgeColor', color, 'LineStyle', '--', 'FaceColor', color, 'DisplayName', sprintf('%d-sigma ellipsoid', num_std));
                hold(ax, 'on');
                plot3(ax, mu2(1), mu2(2), mu2(3), '*', 'Color', color, 'DisplayName', 'mean');
                view(ax, 3);
            else
                error('plot_distribution only supports 1D, 2D, or 3D.');
            end
        end

    end
end 