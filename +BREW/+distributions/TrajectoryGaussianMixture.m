classdef TrajectoryGaussianMixture < BREW.distributions.BaseMixtureModel
    % Trajectory Gaussian Mixture object.
    
    properties (Dependent)
        mean % Property to get matrix of means for speedy propagation
        idx % list of indices (important for RFS since birth model needs to change idx)
        L % list of lengths for each distribution 
        state_dim 
    end

    properties 
        isExtendedTarget = 0;
        isTrajectory = 1;
        L_max
        means 
    end
    
    methods
        function obj = TrajectoryGaussianMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', []);
            addParameter(p, 'means', []);
            addParameter(p, 'covariances', []);
            addParameter(p, 'idx', []);
            addParameter(p,'weights',[]);
            addParameter(p,'L_max',50);

            % Parse known arguments
            parse(p, varargin{:}); 

            if ~isempty(p.Results.dist_list)
                dists = p.Results.dist_list;
            end

            if ~isempty(p.Results.means) && ~isempty(p.Results.covariances) 
                nDists = size(p.Results.means,3);
                for i = 1:nDists
                    if isempty(p.Results.idx)
                        idx_temp = 1;
                    else
                        idx_temp = p.Results.idx(i);
                    end
                    dists(i) = BREW.distributions.TrajectoryGaussian(idx_temp,p.Results.means(:,:,i), p.Results.covariances(:,:,i),'L_max',p.Results.L_max);
                end 
            end
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);

            obj.L_max = p.Results.L_max;

        end

        function val = get.idx(obj) 
            val = cat(1, obj.distributions.init_idx);
        end

        function set.idx(obj, val)
            for k = 1:numel(obj.distributions)
                obj.distributions(k).init_idx = val(k); 
            end
        end 

        function val = get.state_dim(obj) 
            arr = cat(1, obj.distributions.state_dim);
            val = round(sum(arr)/length(arr));
        end

        function val = get.L(obj) 
            val = cat(1, obj.distributions.L);
        end

        function obj = append_history(obj, val)
            for k = 1:numel(obj.distributions)
                obj.distributions(k).mean_history = [obj.distributions(k).mean_history, val(:,:,k)];
            end
        end

        function set.L(obj, val)
            for k = 1:numel(obj.distributions)
                obj.distributions(k).L = val(k); 
            end
        end

        function val = get.mean(obj) 
            val = cat(3, obj.distributions.mean);
        end  

        function val = GetCovs(obj,idx) 
            val = cat(3, obj.distributions(idx).covariances);
        end
        function obj = SetCovs(obj, val, idx) 
            arr = find(idx); 
            for i = 1:numel(arr)
                obj.distributions(arr(i)).covariances = val(:,:,i);
                nx = obj.distributions(arr(i)).state_dim;
                obj.distributions(arr(i)).covariance = val(end-nx+1:end,end-nx+1:end,i);
            end
        end

        function val = GetMeans(obj,idx)
            val = cat(3, obj.distributions(idx).means);
        end
        function obj = SetMeans(obj, val, idx) 
            arr = find(idx); 
            for i = 1:numel(arr)
                obj.distributions(arr(i)).means = val(:,:,i); 

                % Automatically set the history (less function calls in the filter, and less loops)
                nx = obj.distributions(arr(i)).state_dim;
                L = size(val(:,:,i),1) / nx;
                obj.distributions(arr(i)).mean_history(:,end-L+1:end) = reshape(val(:,:,i),nx,[]); 

                obj.distributions(arr(i)).mean = obj.distributions(arr(i)).mean_history(:,end);
            end
        end

        function obj = AppendMeanHistory(obj,val) 
            for i = 1:numel(obj.distributions)
                obj.distributions(i).mean_history(:,end+1) = val(:,:,i); 
            end
        end

        % function obj = SetMeanHistory(obj, val, idx) 
        %     arr = 1:numel(obj.distributions); 
        %     for i = 1:numel(arr(idx))
        %         nx = obj.distributions(arr(i)).state_dim;
        %         L = size(val(:,:,i),1) / nx;
        %         obj.distributions(arr(i)).mean_history(:,end-L+1:end) = val(:,:,i); 
        %     end
        % end

        function plot_distributions(obj, plt_inds, varargin)

            n = numel(obj.distributions);

            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'colors',lines(n))
            parse(p, varargin{:}); 

            colors = p.Results.colors;

            % if user passed a single color (char or 1×3), replicate it
            if (ischar(colors) && size(colors,1)==1) || (isnumeric(colors) && isequal(size(colors),[1,3]))
                colors = repmat(colors, n, 1);
            end

            colors = flip(colors);

            hold on;

            for k = 1:length(obj.distributions)
                obj.distributions(k).plot(plt_inds,'c',colors(k,:),p.Unmatched);
            end 
        end
        
        function measurements = sample_measurements(obj, xy_inds, idx, meas_cov)

            if isempty(idx)
                idx = 0;
                disp("for proper sampling of trajectory object, insert index for measurement timestep")
            end

            all_meas = []; 
            for i = 1:numel(obj.distributions)
                if obj.distributions(i).init_idx <= idx
                    mi = obj.distributions(i).sample_measurements(xy_inds, meas_cov);
                    all_meas = [all_meas, mi];  
                end
            end
        
            measurements = all_meas;  
        end

        function obj = merge(obj, threshold) 
            if nargin < 2 || isempty(threshold), threshold = 4; end
        
            dists   = obj.distributions;
            weights = obj.weights(:).';
            if isempty(weights)
                weights = ones(1, numel(dists));
            end
        
            keepMerging = true;
        
            while keepMerging && numel(dists) > 1
                keepMerging = false;
                N = numel(dists);
        
                best_d2 = inf;
                best_i = 0;
                best_j = 0;
        
                % ---- Find closest pair ----
                for i = 1:N
                    mi = dists(i).mean;
                    Pi = dists(i).getLastCov();
                    Pi = 0.5*(Pi + Pi'); % ensure symmetric
        
                    for j = i+1:N
                        mj = dists(j).mean;
                        Pj = dists(j).getLastCov();
                        Pj = 0.5*(Pj + Pj');
        
                        % only merge if trajectory lengths are equal
                        if numel(mi) == numel(mj)
                            C = 0.5 * (Pi + Pj);
                            C = 0.5 * (C + C');
        
                            % compute Mahalanobis distance safely
                            [R,p] = chol(C);
                            if p ~= 0
                                epsj = 1e-9 * max(1, trace(C)/size(C,1));
                                [R,p] = chol(C + epsj*eye(size(C)));
                                if p ~= 0
                                    invC = pinv(C);
                                    d2 = (mi - mj)' * invC * (mi - mj);
                                else
                                    y = R \ (mi - mj);
                                    d2 = y' * y;
                                end
                            else
                                y = R \ (mi - mj);
                                d2 = y' * y;
                            end
                        else
                            d2 = inf;
                        end
        
                        if d2 < best_d2
                            best_d2 = d2;
                            best_i = i;
                            best_j = j;
                        end
                    end
                end
        
                % ---- Absorb if within threshold ----
                if best_d2 < threshold
                    i = best_i;
                    j = best_j;
        
                    wi = weights(i);
                    wj = weights(j);
        
                    if wi >= wj
                        keep = i; drop = j;
                    else
                        keep = j; drop = i;
                    end
        
                    % absorb weights
                    weights(keep) = weights(keep) + weights(drop);
        
                    % keep distribution mean/covariance unchanged
                    dists(drop) = [];
                    weights(drop) = [];
        
                    keepMerging = true;
                end
            end
        
            obj.distributions = dists;
            obj.weights = weights;
        end

        function disp(obj)
            % Display method for the Gaussian Mixture
            for i = 1:numel(obj.distributions)
                if isempty(obj.weights)
                    fprintf('Component %d:\n', i);
                else
                    fprintf('Component %d (weight = %g):\n', i, obj.weights(i));
                end
                disp(obj.distributions(i));
            end
        end
        
    end
end 