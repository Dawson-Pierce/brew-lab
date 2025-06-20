classdef ExtendedKalmanFilter < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class

    % 

    properties
        process_noise
        measurement_noise
    end

    methods
        function obj = ExtendedKalmanFilter(varargin)
            p = inputParser; 
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'F', []);
            addParameter(p, 'G', []);

            % Parse known arguments
            parse(p, varargin{:});
            
            if ~isempty(p.Results.dyn_obj) || (~isempty(p.Results.F)&&(~isempty(p.Results.G)))
                obj.setStateModel('dyn_obj',p.Results.dyn_obj,'F',p.Results.F,'G',p.Results.G)
            end
        end 

        function setStateModel(varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'F', []);
            addParameter(p, 'G', []);
            parse(p, varargin{:});
            
            if ~isempty(p.Results.dyn_obj)
                obj.dyn_obj_ = p.Results.dyn_obj; 
            elseif ~isempty(p.Results.F) && ~isempty(p.Results.G)
                obj.dyn_obj_ = BREW.dynamics.discrete.LinearModel(p.Results.F,p.Results.G);
            end
        end
    end

    methods (Abstract)
        nextDist = predict(obj, timestep, dt, prevDist, u)
        [nextDist, likelihood] = correct(obj, timestep, dt, prevDist, u)
    end

end