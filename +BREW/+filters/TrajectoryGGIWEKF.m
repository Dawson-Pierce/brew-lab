classdef TrajectoryGGIWEKF < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "TrajectoryGGIW"
    end
    methods
        function obj = TrajectoryGGIWEKF(varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'process_noise', []);
            addParameter(p, 'H', []);
            addParameter(p, 'measurement_noise', []);
            addParameter(p, 'L', 50);
            addParameter(p, 'temporal_decay', 1.0);
            addParameter(p, 'forgetting_factor', 1.0);
            addParameter(p, 'scaling_parameter', 1.0);
            parse(p, varargin{:});
            R = p.Results.measurement_noise;
            if isscalar(R), R = R; end  % scalar R is fine, MEX handles 1x1
            obj.handle_ = brew_mex('create_filter', 'TrajectoryGGIWEKF', ...
                p.Results.dyn_obj.handle_, p.Results.process_noise, ...
                p.Results.H, R, ...
                p.Results.L, ...
                p.Results.temporal_decay, ...
                p.Results.forgetting_factor, ...
                p.Results.scaling_parameter);
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
