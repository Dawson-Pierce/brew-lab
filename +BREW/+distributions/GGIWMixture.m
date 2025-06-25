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

    properties 
        isExtendedTarget = 1;
    end

    methods
        function obj = GGIWMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', {});
            addParameter(p, 'alphas', {});
            addParameter(p, 'betas', {});
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'IWdofs', {});
            addParameter(p, 'IWshapes', {});
            addParameter(p, 'weights', []);
        
            parse(p, varargin{:});
        
            dists = {};
        
            % Option 1: Use given GGIW objects
            if ~isempty(p.Results.dist_list)
                dists = p.Results.dist_list;
            end
        
            % Option 2: Construct GGIW objects from components
            if ~isempty(p.Results.alphas) && ...
               ~isempty(p.Results.betas) && ...
               ~isempty(p.Results.means) && ...
               ~isempty(p.Results.covariances) && ...
               ~isempty(p.Results.IWdofs) && ...
               ~isempty(p.Results.IWshapes)
           
                N = numel(p.Results.alphas);  % Assume all parameter lists are the same length
                for i = 1:N
                    dists{end+1} = BREW.distributions.GGIW( ...
                        p.Results.alphas{i}, ...
                        p.Results.betas{i}, ...
                        p.Results.means{i}, ...
                        p.Results.covariances{i}, ...
                        p.Results.IWdofs{i}, ...
                        p.Results.IWshapes{i});
                end
            end
        
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);
        end
        
        function val = get.isExtendedTarget(obj)
            val = 1;
        end

        function val = get.alphas(obj)
            val = cellfun(@(d) d.alpha, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.alphas(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.alpha = val{i};
            end
        end
        function val = get.betas(obj)
            val = cellfun(@(d) d.beta, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.betas(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.beta = val{i};
            end
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
        function val = get.IWdofs(obj)
            val = cellfun(@(d) d.IWdof, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.IWdofs(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.IWdof = val{i};
            end
        end
        function val = get.IWshapes(obj)
            val = cellfun(@(d) d.IWshape, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.IWshapes(obj, val)
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
        
            measurements = all_meas;  
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
    end
end 