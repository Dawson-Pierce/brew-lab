classdef GGIWMixture < BREW.distributions.BaseMixtureModel
    % Gamma Gaussian Inverse Wishart Mixture object.
    
    properties (Dependent)
        alpha
        beta
        mean
        covariance
        IWdof
        IWshape
    end 

    properties 
        isExtendedTarget = 1;
    end

    methods
        function obj = GGIWMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', []);
            addParameter(p, 'alphas', []);
            addParameter(p, 'betas', []);
            addParameter(p, 'means', []);
            addParameter(p, 'covariances', []);
            addParameter(p, 'IWdofs', []);
            addParameter(p, 'IWshapes', []);
            addParameter(p, 'weights', []);
        
            parse(p, varargin{:}); 
        
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

                nDists = size(p.Results.means,3);
                dists(1, nDists) = BREW.distributions.GGIW();
                for i = 1:nDists
                    dists(i) = BREW.distributions.GGIW( ...
                        p.Results.alphas(:,:,i), ...
                        p.Results.betas(:,:,i), ...
                        p.Results.means(:,:,i), ...
                        p.Results.covariances(:,:,i), ...
                        p.Results.IWdofs(:,:,i), ...
                        p.Results.IWshapes(:,:,i));
                end
            end
        
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);
        end

        function val = get.alpha(obj)
            val = cat(3, obj.distributions.alpha);
        end
        function set.alpha(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions(k).alpha = val(:, :, k);
            end
        end
        function val = get.beta(obj)
            val = cat(3, obj.distributions.beta);
        end
        function set.beta(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions(k).beta = val(:, :, k);
            end
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
        function val = get.IWdof(obj)
            val = cat(3, obj.distributions.IWdof);
        end
        function set.IWdof(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions(k).IWdof = val(:, :, k);
            end
        end
        function val = get.IWshape(obj)
            val = cat(3, obj.distributions.IWshape);
        end
        function set.IWshape(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions(k).IWshape = val(:, :, k);
            end
        end

        function measurements = sample_measurements(obj, xy_inds)
            if nargin < 2 || isempty(xy_inds)
                d = obj.distributions(1).d;
                if d==2
                    xy_inds = [1 2];
                elseif d==3
                    xy_inds = [1 2 3];
                else
                    error('Invalid dimension for IW shape.');
                end
            end

            all_meas = []; 
            for i = 1:numel(obj.distributions)
                mi = obj.distributions(i).sample_measurements(xy_inds);
                all_meas = [all_meas, mi]; 
            end
        
            measurements = all_meas;  
        end

        function disp(obj)
            % Display method for the GGIW Mixture
            for i = 1:numel(obj.distributions)
                fprintf('Component %d (weight = %g):\n', i, obj.weights(i));
                disp(obj.distributions(i));
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
                obj.distributions{i}.plot_distribution(ax, plt_inds, h, c);
            end
        end

        function obj = merge(obj, threshold)
            if nargin < 2 || isempty(threshold), threshold = 4; end
        
            % Early exit if only one or none
            if isempty(obj.distributions) || numel(obj.distributions) < 2
                return;
            end
        
            means  = obj.mean;          % [nx1xM]
            covs   = obj.covariance;    % [nxnxM]
            alphas = obj.alpha;         % [1x1xM]
            betas  = obj.beta;          % [1x1xM]
            vIW    = obj.IWdof;         % [1x1xM]
            VIW    = obj.IWshape;       % [nxnxM]
            w      = obj.weight(:)';    % [1xM]
            if isempty(w), w = ones(1, size(means,3)); end
        
            M = size(means,3);
            keep = true(1, M);
        
            newW = [];
            newM = [];
            newP = [];
            newA = [];
            newB = [];
            newV = [];
            newS = [];
        
            while any(keep)
                idx = find(keep);
                [~, iMax] = max(w(idx));
                j = idx(iMax);
        
                Cj = covs(:,:,j);
                Cj = (Cj + Cj')/2;
                [R, p] = chol(Cj);
                if p ~= 0
                    epsj = 1e-9 * max(1, trace(Cj)/size(Cj,1));
                    Cj = Cj + epsj * eye(size(Cj));
                    R = chol(Cj);
                end
        
                diffs = means(:,:,idx) - means(:,:,j);
                d2 = sum((R \ diffs).^2, 1);
                grp = idx(d2 <= threshold);
        
                w_grp = w(grp);
                w_new = sum(w_grp);
                if w_new <= 0
                    keep(grp) = false;
                    continue;
                end
        
                % Weighted averages
                w_norm = w_grp / w_new;
                m_new = sum(means(:,:,grp) .* reshape(w_norm, 1,1,[]), 3);
                P_new = sum(covs(:,:,grp) .* reshape(w_norm, 1,1,[]), 3);
                P_new = 0.5 * (P_new + P_new');
                a_new = sum(alphas(:,:,grp) .* w_norm, 3);
                b_new = sum(betas(:,:,grp) .* w_norm, 3);
                v_new = sum(vIW(:,:,grp) .* w_norm, 3);
                S_new = sum(VIW(:,:,grp) .* reshape(w_norm, 1,1,[]), 3);
                S_new = 0.5 * (S_new + S_new');
        
                newW(end+1) = w_new;
                newM(:,:,end+1) = m_new;
                newP(:,:,end+1) = P_new;
                newA(:,:,end+1) = a_new;
                newB(:,:,end+1) = b_new;
                newV(:,:,end+1) = v_new;
                newS(:,:,end+1) = S_new;
        
                keep(grp) = false;
            end
        
            % Normalize weights
            newW = newW / sum(newW);
        
            obj = BREW.distributions.GGIWMixture( ...
                'mean', newM, ...
                'covariance', newP, ...
                'alpha', newA, ...
                'beta', newB, ...
                'IWdof', newV, ...
                'IWshape', newS, ...
                'weight', newW );
        end



    end
end 