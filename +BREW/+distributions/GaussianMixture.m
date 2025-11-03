classdef GaussianMixture < BREW.distributions.BaseMixtureModel
    % Gaussian Mixture object.
    
    properties (Dependent)
        mean       % Cell array of means for each component
        covariance % Cell array of covariances for each component
    end

    properties 
        isExtendedTarget = 0;
    end
    
    methods
        function obj = GaussianMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', []);
            addParameter(p, 'means', []);
            addParameter(p, 'covariances', []);
            addParameter(p,'weights',[]);

            % Parse known arguments
            parse(p, varargin{:});

            if ~isempty(p.Results.dist_list)
                dists = p.Results.dist_list;
            end

            if ~isempty(p.Results.means) && ~isempty(p.Results.covariances) 
                nDists = size(p.Results.means,3); 
                for i = 1:nDists 
                    dists(i) = BREW.distributions.Gaussian(p.Results.means(:,:,i), p.Results.covariances(:,:,i));
                end 
            end
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);
        end

        function val = get.mean(obj) 
            val = cat(3, obj.distributions.mean);
        end

        function set.mean(obj, val)
            for k = 1:numel(obj.distributions)
                obj.distributions(k).mean = val(:, :, k);
            end
        end
        function val = get.covariance(obj) 
            val = cat(3, obj.distributions.covariance);
        end
        function set.covariance(obj, val)
            for k = 1:numel(obj.distributions)
                obj.distributions(k).covariance = val(:, :, k);
            end
        end 
        
        function p = pdf(obj, x)
            % Evaluate the PDF at point x
            p = 0;
            for i = 1:numel(obj.distributions)
                p = p + obj.weights(i) * obj.distributions(i).pdf(x);
            end
        end

         function measurements = sample_measurements(obj, xy_inds)

            all_meas = []; 
            for i = 1:numel(obj.distributions)
                mi = obj.distributions(i).sample_measurements(xy_inds);
                all_meas = [all_meas, mi];  
            end
        
            measurements = all_meas;  
        end
        
        function disp(obj)
            % Display method for the Gaussian Mixture
            for i = 1:numel(obj.distributions)
                if isempty(obj.weights)
                    fprintf('Component %d:\n', i);
                else
                    fprintf('Component %d (weight = %g):\n', i, obj.weights(i));
                end
                disp(obj.distributions(i))
            end
        end

        function obj = merge(obj, threshold)
            if nargin < 2
                threshold = 4; % default Mahalanobis^2 threshold
            end
        
            means = obj.means;          % n×1×M
            covs = obj.covariances;     % n×n×M
            weights = obj.weights(:);   % M×1
            nComp = numel(weights);
        
            keepMerging = true;
        
            while keepMerging && nComp > 1
                keepMerging = false;
        
                % compute pairwise Mahalanobis distances
                minDist = inf;
                pair = [];
        
                for i = 1:nComp
                    mi = means(:, :, i);
                    Pi = covs(:, :, i);
        
                    for j = i+1:nComp
                        mj = means(:, :, j);
                        Pj = covs(:, :, j);
        
                        diff = mi - mj;
                        C = 0.5 * (Pi + Pj);
                        d2 = diff' * (C \ diff);
        
                        if d2 < minDist
                            minDist = d2;
                            pair = [i j];
                        end
                    end
                end
        
                % merge if below threshold
                if minDist < threshold
                    i = pair(1); j = pair(2);
        
                    wi = weights(i);
                    wj = weights(j);
                    mi = means(:, :, i);
                    mj = means(:, :, j);
                    Pi = covs(:, :, i);
                    Pj = covs(:, :, j);
        
                    w = wi + wj;
                    m = (wi * mi + wj * mj) / w;
                    P = (wi * (Pi + (mi - m) * (mi - m)') + ...
                         wj * (Pj + (mj - m) * (mj - m)')) / w;
        
                    % replace i with merged, delete j
                    means(:, :, i) = m;
                    covs(:, :, i) = P;
                    weights(i) = w;
        
                    means(:, :, j) = [];
                    covs(:, :, j) = [];
                    weights(j) = [];
        
                    nComp = nComp - 1;
                    keepMerging = true;
                end
            end
        
            obj = BREW.distributions.GaussianMixture( ...
                'means', means, ...
                'covariances', covs, ...
                'weights', weights);
        end


        function plot_distributions(obj, plt_inds, varargin) 

            n = numel(obj.distributions);

            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'colors',lines(n)) 
            addParameter(p,'ax',gca)
            addParameter(p,'num_std',2)
            parse(p, varargin{:});

            colors = p.Results.colors; 
            ax = p.Results.ax;
            num_std = p.Results.num_std;
        
            % if user passed a single color (char or 1×3), replicate it
            if (ischar(colors) && size(colors,1)==1) || (isnumeric(colors) && isequal(size(colors),[1,3]))
                colors = repmat(colors, n, 1);
            end 
        
            % if no colors given, pull an n‐by‐3 list from lines
            if nargin < 5 || isempty(colors)
                colors = lines(n);
            end
        
            % if user passed a single color (char or 1×3), replicate it
            if (ischar(colors) && size(colors,1)==1) || (isnumeric(colors) && isequal(size(colors),[1,3]))
                colors = repmat(colors, n, 1);
            end
        
            for i = 1:n
                % pick the i-th row of colors as an RGB triplet
                c = colors(i,:);
                obj.distributions(i).plot_distribution(plt_inds, 'ax', ax, 'num_std', num_std, 'color',c);
            end
        end
        
    end
end 