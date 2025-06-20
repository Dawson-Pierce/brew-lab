classdef GGIWMixture < BREW.distributions.BaseMixtureModel
    % Gamma Gaussian Inverse Wishart Mixture object.
    
    properties (Dependent)
        alphas
        betas
        means
        covariances
        IWdofs
        IWshapes
    end
    
    methods
        function obj = GGIWMixture(alphas, betas, means, covariances, IWdofs, IWshapes, weights)
            % Initialize a GGIWMixture object
            if nargin < 1, alphas = {}; end
            if nargin < 2, betas = {}; end
            if nargin < 3, means = {}; end
            if nargin < 4, covariances = {}; end
            if nargin < 5, IWdofs = {}; end
            if nargin < 6, IWshapes = {}; end
            if nargin < 7, weights = []; end
            dists = {};
            for i = 1:numel(alphas)
                dists{end+1} = BREW.distributions.GGIW(alphas{i}, betas{i}, means{i}, covariances{i}, IWdofs{i}, IWshapes{i});
            end
            obj@BREW.distributions.BaseMixtureModel(dists, weights);
        end
        
        function val = get.alphas(obj)
            val = cellfun(@(d) d.alpha, obj.distributions, 'UniformOutput', false);
        end
        function set.alphas(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.alpha = val{i};
            end
        end
        function val = get.betas(obj)
            val = cellfun(@(d) d.beta, obj.distributions, 'UniformOutput', false);
        end
        function set.betas(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.beta = val{i};
            end
        end
        function val = get.means(obj)
            val = cellfun(@(d) d.mean, obj.distributions, 'UniformOutput', false);
        end
        function set.means(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.mean = val{i};
            end
        end
        function val = get.covariances(obj)
            val = cellfun(@(d) d.covariance, obj.distributions, 'UniformOutput', false);
        end
        function set.covariances(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.covariance = val{i};
            end
        end
        function val = get.IWdofs(obj)
            val = cellfun(@(d) d.IWdof, obj.distributions, 'UniformOutput', false);
        end
        function set.IWdofs(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.IWdof = val{i};
            end
        end
        function val = get.IWshapes(obj)
            val = cellfun(@(d) d.IWshape, obj.distributions, 'UniformOutput', false);
        end
        function set.IWshapes(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.IWshape = val{i};
            end
        end
        
        function s = sample(obj, numSamples)
            % Draw samples from the mixture model (returns cell array of measurements)
            if nargin < 2, numSamples = 1; end
            w = obj.weights(:) / sum(obj.weights);
            idx = randsample(1:numel(obj.distributions), numSamples, true, w);
            s = cell(numSamples,1);
            for i = 1:numSamples
                s{i} = obj.distributions{idx(i)}.sample_measurements();
            end
        end
        
        function disp(obj)
            % Display method for the GGIW Mixture
            for i = 1:numel(obj.distributions)
                fprintf('Component %d (weight = %g):\n', i, obj.weights(i));
                disp(obj.distributions{i});
            end
        end
        
        function plot_distributions(obj, ax, plt_inds, h, color)
            % Plot all GGIW components in 2D
            if nargin < 2 || isempty(ax), ax = gca; end
            if nargin < 3 || isempty(plt_inds), plt_inds = [1 2]; end
            if nargin < 4, h = 0.95; end
            if nargin < 5, color = 'r'; end
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.plot_distribution(ax, plt_inds, h, color);
            end
        end
        
        function addComponents(obj, new_alphas, new_betas, new_means, new_covariances, new_IWdofs, new_IWshapes, new_weights)
            % Add new GGIW components to the mixture
            % Each argument is a cell array (except new_weights, which is numeric)
            for i = 1:numel(new_alphas)
                obj.distributions{end+1} = BREW.distributions.GGIW(new_alphas{i}, new_betas{i}, new_means{i}, new_covariances{i}, new_IWdofs{i}, new_IWshapes{i});
                obj.weights(end+1) = new_weights(i);
            end
        end
    end
end 