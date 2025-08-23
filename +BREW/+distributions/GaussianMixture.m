classdef GaussianMixture < BREW.distributions.BaseMixtureModel
    % Gaussian Mixture object.
    
    properties (Dependent)
        means       % Cell array of means for each component
        covariances % Cell array of covariances for each component
    end

    properties 
        isExtendedTarget = 0;
    end
    
    methods
        function obj = GaussianMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', {});
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p,'weights',[]);

            % Parse known arguments
            parse(p, varargin{:});

            dists = {};

            if ~isempty(p.Results.dist_list)
                dists = p.Results.dist_list;
            end

            if ~isempty(p.Results.means) && ~isempty(p.Results.covariances) 
                for i = 1:numel(p.Results.means)
                    dists{end+1} = BREW.distributions.Gaussian(p.Results.means{i}, p.Results.covariances{i});
                end 
            end
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);
        end

        function val = get.means(obj)
            val = cellfun(@(d) d.mean, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.means(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.mean = val{i};
            end
        end
        function val = get.covariances(obj)
            val = cellfun(@(d) d.covariance, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.covariances(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.covariance = val{i};
            end
        end
        
        function newObj = copy(obj)
            % Deep copy of the GaussianMixture object
            new_means = obj.means;
            new_covariances = obj.covariances;
            new_weights = obj.weights;
            newObj = BREW.distributions.GaussianMixture(new_means, new_covariances, new_weights);
        end
        
        function s = sample(obj, numSamples)
            % Draw samples from the mixture model
            if nargin < 2, numSamples = 1; end
            w = obj.weights(:) / sum(obj.weights);
            idx = randsample(1:numel(obj.distributions), numSamples, true, w);
            d = length(obj.distributions{1}.mean);
            s = zeros(d, numSamples);
            for i = 1:numSamples
                s(:,i) = obj.distributions{idx(i)}.sample(1);
            end
        end
        
        function p = pdf(obj, x)
            % Evaluate the PDF at point x
            p = 0;
            for i = 1:numel(obj.distributions)
                p = p + obj.weights(i) * obj.distributions{i}.pdf(x);
            end
        end

         function measurements = sample_measurements(obj, xy_inds)

            all_meas = []; 
            for i = 1:numel(obj.distributions)
                mi = obj.distributions{i}.sample_measurements(xy_inds);
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
                disp(obj.distributions{i});
            end
        end

        function obj = merge(obj, threshold)
            if nargin < 2
                threshold = 4; % default Mahalanobis^2 threshold
            end
        
            means_copy = obj.means;
            covs = obj.covariances;
            weights = obj.weights;
        
            keepMerging = true;
            while keepMerging && numel(means_copy) > 1
                keepMerging = false;
                N = numel(means_copy);
        
                % find closest pair
                minDist = inf;
                pair = [];
                for i = 1:N
                    for j = i+1:N
                        diff = means_copy{i} - means_copy{j};
                        % symmetric Mahalanobis distance using avg covariance
                        C = (covs{i} + covs{j})/2;
                        d2 = diff' * (C \ diff);
                        if d2 < minDist
                            minDist = d2;
                            pair = [i j];
                        end
                    end
                end
        
                if minDist < threshold
                    i = pair(1); j = pair(2);
        
                    wi = weights(i); wj = weights(j);
                    mi = means_copy{i}; mj = means_copy{j};
                    Pi = covs{i}; Pj = covs{j};
                    w = wi + wj;
                    m = (wi*mi + wj*mj) / w;
                    P = (wi*(Pi + (mi-m)*(mi-m)') + ...
                         wj*(Pj + (mj-m)*(mj-m)')) / w;
        
                    % replace i with merged, delete j
                    means_copy{i} = m;
                    covs{i} = P;
                    weights(i) = w;
        
                    means_copy(j) = [];
                    covs(j) = [];
                    weights(j) = [];
        
                    keepMerging = true;
                end
            end
        
            obj = BREW.distributions.GaussianMixture( ...
                'means', means_copy, ...
                'covariances', covs, ...
                'weights', weights );
        end

        function plot_distributions(obj, ax, plt_inds, num_std, colors)
            % Plot all GGIW components in 2D (or 3D if plt_inds has 3 elements)
            if nargin < 2 || isempty(ax),        ax = gca;       end
            if nargin < 3 || isempty(plt_inds),   plt_inds = [1 2]; end
            if nargin < 4 || isempty(num_std),          num_std = 1;      end
        
            n = numel(obj.distributions);
        
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
                obj.distributions{i}.plot_distribution(ax, plt_inds, num_std, c);
            end
        end
        
    end
end 