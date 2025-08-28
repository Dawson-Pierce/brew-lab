classdef TrajectoryGaussianMixture < BREW.distributions.BaseMixtureModel
    % Trajectory Gaussian Mixture object.
    
    properties (Dependent)
        mean_trajectories       % Cell array of means for each trajectory
        covariance_trajectories % Cell array of covariances for each trajectory 
    end

    properties 
        isExtendedTarget = 0;
    end
    
    methods
        function obj = TrajectoryGaussianMixture(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dist_list', {});
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'idx', {});
            addParameter(p,'weights',[]);

            % Parse known arguments
            parse(p, varargin{:});

            dists = {};

            if ~isempty(p.Results.dist_list)
                dists = p.Results.dist_list;
            end

            if ~isempty(p.Results.means) && ~isempty(p.Results.covariances) 
                for i = 1:numel(p.Results.means)
                    dists{end+1} = BREW.distributions.TrajectoryGaussian(p.Results.idx{i},p.Results.means{i}(:), p.Results.covariances{i});
                end 
            end
            obj@BREW.distributions.BaseMixtureModel(dists, p.Results.weights);
        end

        function val = get.mean_trajectories(obj)
            val = cellfun(@(d) d.means, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.mean_trajectories(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.means = val{i};
            end
        end
        function val = get.covariance_trajectories(obj)
            val = cellfun(@(d) d.covariances, obj.distributions, 'UniformOutput', false);
        end
        function obj = set.covariance_trajectories(obj, val)
            for i = 1:numel(obj.distributions)
                obj.distributions{i}.covariances = val{i};
            end
        end

        function plot(obj, plt_inds, varargin)

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

            for k = 1:length(obj.distributions)
                obj.distributions{k}.plot(plt_inds,'c',colors(k,:),p.Unmatched);
            end 
        end
        
        function measurements = sample_measurements(obj, xy_inds, idx, meas_cov)

            if isempty(idx)
                idx = 0;
                disp("for proper sampling of trajectory object, insert index for measurement timestep")
            end

            all_meas = []; 
            for i = 1:numel(obj.distributions)
                if obj.distributions{i}.init_idx <= idx
                    mi = obj.distributions{i}.sample_measurements(xy_inds, meas_cov);
                    all_meas = [all_meas, mi];  
                end
            end
        
            measurements = all_meas;  
        end

        function obj = merge(obj, threshold) 
            if nargin < 2 || isempty(threshold), threshold = 4; end
        
            dists   = obj.distributions;
            weights = obj.weights(:).'; 
        
            keepMerging = true;

            while keepMerging && numel(dists) > 1
                keepMerging = false;
                N = numel(dists);
        
                % ---- find closest pair by last-state Mahalanobis^2 ----
                best_d2 = inf; best_i = 0; best_j = 0; best_R = [];
                for i = 1:N
                    mi = dists{i}.getLastState();
                    Pi = dists{i}.getLastCov(); Pi = 0.5*(Pi+Pi');
                    for j = i+1:N
                        mj = dists{j}.getLastState();
                        Pj = dists{j}.getLastCov(); Pj = 0.5*(Pj+Pj');
        
                        C = 0.5*(Pi + Pj); C = 0.5*(C + C');
                        [R,p] = chol(C);
                        if p ~= 0
                            epsj = 1e-9 * max(1, trace(C)/size(C,1));
                            [R,p] = chol(C + epsj*eye(size(C)));
                            if p ~= 0
                                % fallback to pinv
                                invC = pinv(C);
                                d2 = (mi-mj)' * invC * (mi-mj);
                            else
                                y  = R \ (mi - mj);
                                d2 = y' * y;
                            end
                        else
                            y  = R \ (mi - mj);
                            d2 = y' * y;
                        end
        
                        if d2 < best_d2
                            best_d2 = d2; best_i = i; best_j = j; best_R = R; 
                        end
                    end
                end
        
                if best_d2 < threshold 
                    i = best_i; j = best_j;
                    
                    if isempty(weights)
                        weights = ones(1, numel(dists));
                    end
                    wi = weights(i); wj = weights(j); W = wi + wj;

                    mi = dists{i}.getLastState();
                    mj = dists{j}.getLastState();
                    Pi = 0.5*(dists{i}.getLastCov() + dists{i}.getLastCov()'); 
                    Pj = 0.5*(dists{j}.getLastCov() + dists{j}.getLastCov()');
                    
                    if wi >= wj
                        keep = i; drop = j;
                        m_keep = mi; P_keep = Pi; m_other = mj; P_other = Pj;
                    else
                        keep = j; drop = i;
                        m_keep = mj; P_keep = Pj; m_other = mi; P_other = Pi;
                    end
                    
                    gamma_max = 0.2;
                    gamma = min(min(wi,wj)/W, gamma_max);
                    
                    if gamma > 0
                        dm = (m_other - m_keep);
                        m_new = m_keep + gamma * dm;
                    
                        P_new = (1-gamma)*P_keep + gamma*P_other + gamma*(1-gamma)*(dm*dm');
                        P_new = 0.5*(P_new + P_new');
                    else
                        m_new = m_keep;
                        P_new = P_keep;
                    end
                    
                    base  = dists{keep};
                    d     = base.state_dim;
                    nTot  = numel(base.means);
                    idxk  = (nTot - d + 1) : nTot;
                    
                    base.means(idxk) = m_new;
                    Cfull = base.covariances;
                    Cfull(idxk, idxk) = P_new;
                    base.covariances = 0.5*(Cfull + Cfull');
                    
                    dists{keep}   = base;
                    weights(keep) = W;
                    dists(drop)   = [];
                    weights(drop) = [];
                    
                    keepMerging = true; 

                end
            end
        
            obj.distributions = dists;
            obj.weights       = weights;
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