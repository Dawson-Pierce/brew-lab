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
            p = inputParser;
            p.KeepUnmatched = true;
            parse(p, varargin{:});
            
            nv = namedargs2cell(p.Unmatched); 

            for k = 1:length(obj.distributions)
                obj.distributions{k}.plot(plt_inds,nv{:});
            end
        end
        
        
        function newObj = copy(obj)
            % Deep copy of the object
            new_means = obj.mean_trajectories;
            new_covariances = obj.covariance_trajectories;
            new_weights = obj.weights;
            new_idx = obj.idx;
            newObj = BREW.distributions.TrajectoryGaussianMixture(new_idx,new_means, new_covariances, new_weights);
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