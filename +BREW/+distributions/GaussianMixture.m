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
            obj@BREW.distributions.BaseMixtureModel(dists, weights);
        end
        
        function newObj = copy(obj)
            % Deep copy of the GaussianMixture object
            new_means = obj.means;
            new_covariances = obj.covariances;
            new_weights = obj.weights;
            newObj = BREW.distributions.GaussianMixture(new_means, new_covariances, new_weights);
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
        
    end
end 