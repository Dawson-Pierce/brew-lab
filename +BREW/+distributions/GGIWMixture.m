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
        
        function plot_distributions(obj, plt_inds, varargin) 

            n = numel(obj.distributions);

            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'colors',lines(n))
            addParameter(p,'h',0.95)
            addParameter(p,'ax',gca)
            parse(p, varargin{:});

            colors = p.Results.colors;
            h = p.Results.h;
            ax = p.Results.ax;
        
            % if user passed a single color (char or 1×3), replicate it
            if (ischar(colors) && size(colors,1)==1) || (isnumeric(colors) && isequal(size(colors),[1,3]))
                colors = repmat(colors, n, 1);
            end
        
            for i = 1:n
                % pick the i-th row of colors as an RGB triplet
                c = colors(i,:);
                obj.distributions{i}.plot(plt_inds, 'h',ax, 'h', h, 'c', c);
            end
        end

        function obj = merge(obj, threshold)
            % MERGE  Average-merge nearby GGIW components 
                if nargin < 2 || isempty(threshold), threshold = 4; end
            
                if isempty(obj.distributions) || numel(obj.distributions) < 2
                    return;
                end
            
                % pull current params as cell arrays / vectors
                means  = obj.means;         % cell{N} of column vectors
                covs   = obj.covariances;   % cell{N} of matrices
                alphas = obj.alphas;        % cell{N} of scalars
                betas  = obj.betas;         % cell{N} of scalars
                vIW    = obj.IWdofs;        % cell{N} of scalars
                VIW    = obj.IWshapes;      % cell{N} of matrices
            
                w = obj.weights(:).';
                if isempty(w), w = ones(1, numel(means)); end
            
                N = numel(means);
                remaining = true(1, N);
            
                % accumulators for merged mixture
                w_lst = [];               % 1×K
                m_lst = {};               % 1×K cells
                P_lst = {};               % 1×K cells
                a_lst = {}; b_lst = {};   % 1×K cells
                v_lst = {}; V_lst = {};   % 1×K cells
            
                while any(remaining)
                    inds = find(remaining);
                    % pick the heaviest remaining component as reference (jj)
                    [~, k] = max(w(inds));
                    jj = inds(k);
            
                    % Cholesky of reference covariance for gating
                    Cjj = covs{jj};
                    Cjj = (Cjj + Cjj.')/2;
                    [R, p] = chol(Cjj);
                    if p ~= 0
                        epsj = 1e-9 * max(1, trace(Cjj)/size(Cjj,1));
                        Cjj = Cjj + epsj * eye(size(Cjj));
                        R = chol(Cjj); 
                    end
            
                    % collect indices that fall within gate of jj
                    comp_mask = false(1, N);
                    for ii = inds
                        d = means{ii} - means{jj};
                        y = R \ d;                 % Mahalanobis via chol
                        val = y' * y;              % d' * inv(Cjj) * d
                        if val <= threshold
                            comp_mask(ii) = true;
                        end
                    end
                    grp = find(comp_mask);
                    w_new = sum(w(grp));
            
                    % if degenerate weights, just drop them from consideration
                    if w_new <= 0
                        remaining(grp) = false;
                        continue;
                    end
            
                    % weighted averages
                    % mean
                    m_new = 0;
                    for ii = grp
                        m_new = m_new + w(ii) * means{ii};
                    end
                    m_new = m_new / w_new;
            
                    % covariance (simple weighted average; symmetrize)
                    P_sum = 0;
                    for ii = grp
                        P_sum = P_sum + w(ii) * covs{ii};
                    end
                    P_new = P_sum / w_new;
                    P_new = 0.5 * (P_new + P_new.');
            
                    % gamma params
                    a_new = 0; b_new = 0;
                    for ii = grp
                        a_new = a_new + w(ii) * alphas{ii};
                        b_new = b_new + w(ii) * betas{ii};
                    end
                    a_new = a_new / w_new;
                    b_new = b_new / w_new;
            
                    % IW params (dof scalar, shape matrix; symmetrize shape)
                    v_new = 0; V_sum = 0;
                    for ii = grp
                        v_new = v_new + w(ii) * vIW{ii};
                        V_sum = V_sum + w(ii) * VIW{ii};
                    end
                    v_new = v_new / w_new;
                    V_new = V_sum / w_new;
                    V_new = 0.5 * (V_new + V_new.');
            
                    % append merged component
                    w_lst(end+1) = w_new; 
                    m_lst{end+1} = m_new; 
                    P_lst{end+1} = P_new; 
                    a_lst{end+1} = a_new; 
                    b_lst{end+1} = b_new; 
                    v_lst{end+1} = v_new; 
                    V_lst{end+1} = V_new; 
            
                    % remove grouped comps from remaining set
                    remaining(grp) = false;
                end
            
                % rebuild mixture from averaged groups 
                obj = BREW.distributions.GGIWMixture( ...
                    'alphas',      a_lst, ...
                    'betas',       b_lst, ...
                    'means',       m_lst, ...
                    'covariances', P_lst, ...
                    'IWdofs',      v_lst, ...
                    'IWshapes',    V_lst, ...
                    'weights',     w_lst ); 
            end


    end
end 