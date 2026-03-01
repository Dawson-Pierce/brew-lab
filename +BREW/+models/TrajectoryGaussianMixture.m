classdef TrajectoryGaussianMixture < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "TrajectoryGaussian"
    end
    methods
        function obj = TrajectoryGaussianMixture(varargin)
            p = inputParser;
            addParameter(p, 'idx', {});
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'weights', []);
            parse(p, varargin{:});
            obj.handle_ = brew_mex('create_mixture', 'TrajectoryGaussian', ...
                p.Results.idx, p.Results.means, p.Results.covariances, ...
                p.Results.weights);
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
