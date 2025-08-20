classdef TrajectoryBaseModel
    % Trajectory Base Model for trajectory filters
    % Useful as an alternative to labeled RFS

    properties
        state_dim     % length of state vector
        window_size   % length of trajectory 
        init_idx      % index of trajectory being created
    end

    methods
        function obj = TrajectoryBaseModel(idx, state_dim)
            obj.state_dim = state_dim;
            obj.window_size = 1;
            obj.init_idx = idx;
        end
    end
end