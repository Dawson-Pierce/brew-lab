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
        
        function measurements = sample_measurements(obj, xy_inds, random_extent)
            % SAMPLE_MEASUREMENTS  draw from each GGIW in the mixture and concatenate
            %
            %   meas = obj.sample_measurements(xy_inds,random_extent)
            %     returns an N_total×d array of measurements
            %     (each row is one [x y (z)] vector).
            %
            %   Default xy_inds = [1 2] for d=2, [1 2 3] for d=3.
            %   Default random_extent = false.
            
            %--- handle defaults (inspect first component for dimensionality) ---
            if nargin < 2 || isempty(xy_inds)
                d = obj.distributions{1}.d;
                if d==2
                    xy_inds = [1 2];
                elseif d==3
                    xy_inds = [1 2 3];
                else
                    error('Invalid dimension for IW shape.');
                end
            end
            if nargin < 3 || isempty(random_extent)
                random_extent = false;
            end
        
            %--- draw & concatenate ---
            all_meas = [];         % will be d × N_total
            for i = 1:numel(obj.distributions)
                mi = obj.distributions{i}.sample_measurements(xy_inds, random_extent);
                all_meas = [all_meas, mi];   %#ok<AGROW>
            end
        
            %--- return as an N_total-by-d list of [x y (z)] rows ---
            measurements = all_meas.';  
        end

        function disp(obj)
            % Display method for the GGIW Mixture
            for i = 1:numel(obj.distributions)
                fprintf('Component %d (weight = %g):\n', i, obj.weights(i));
                disp(obj.distributions{i});
            end
        end
        
        function plot_distributions(obj, ax, plt_inds, h, colors)
            % Plot all GGIW components in 2D (or 3D if plt_inds has 3 elements)
            if nargin < 2 || isempty(ax),        ax = gca;       end
            if nargin < 3 || isempty(plt_inds),   plt_inds = [1 2]; end
            if nargin < 4 || isempty(h),          h = 0.95;      end
        
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
                obj.distributions{i}.plot_distribution(ax, plt_inds, h, c);
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