classdef (Abstract) FiltersBase < handle
    % FiltersBase class
    %
    % Defines all common functions and properties for each filter

    properties
        dyn_obj_ 
        h
        H
        process_noise
        measurement_noise
    end

    methods
        function obj = FiltersBase(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'dyn_obj', []);  % plug and play dynamic object
            addParameter(p, 'F', []); % F for linear kinematics
            addParameter(p, 'G', []); % G for linear kinematics
            addParameter(p, 'f', []); % function handle for kinematics
            addParameter(p, 'g', []); % function handle for input 
            addParameter(p, 'M', []); % function/matrix for extent rotation

            addParameter(p, 'h', []); % function handle for measurement function
            addParameter(p, 'H', []); % function handle for measurement matrix

            addParameter(p, 'process_noise', []); 
            addParameter(p, 'measurement_noise', []);
            
            % Parse known arguments
            parse(p, varargin{:});

            obj.setStateModel('dyn_obj',p.Results.dyn_obj,'F',p.Results.F,...
                'G',p.Results.G,'f',p.Results.f,'g',p.Results.g,'M',p.Results.M) 

            obj.setMeasurementModel('h',p.Results.h,'H',p.Results.H) 

            obj.process_noise = p.Results.process_noise;
            obj.measurement_noise = p.Results.measurement_noise;
            
        end 

        function setProcessNoise(obj,Q)
            obj.process_noise = Q;
        end

        function setMeasurementNoise(obj,R)
            obj.measurement_noise = R;
        end

        function setStateModel(obj,varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'F', []);
            addParameter(p, 'G', []);
            addParameter(p, 'f', []); 
            addParameter(p, 'g', []); 
            addParameter(p, 'M', []); 
            parse(p, varargin{:});
            
            if ~isempty(p.Results.dyn_obj)
                obj.dyn_obj_ = p.Results.dyn_obj; 
                if ~isempty(p.Results.M)
                    obj.dyn_obj_.setRotationModel(M);
                end
            elseif (~isempty(p.Results.F) && isempty(p.Results.f)) && ~isempty(p.Results.G)
                obj.dyn_obj_ = BREW.dynamics.LinearModel(p.Results.F,p.Results.G);
                if ~isempty(p.Results.M)
                    obj.dyn_obj_.setRotationModel(M);
                end
            elseif ~isempty(p.Results.f)
                obj.dyn_obj_ = ...
                    BREW.dynamics.FunctionHandleDynamics( ...
                    'f',p.Results.f, ...
                    'g',p.Results.g, ...
                    'F',p.Results.F, ...
                    'G',p.Results.G, ...
                    'M',p.Results.M);
            end
        end

        function setMeasurementModel(obj,varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'h', []);
            addParameter(p, 'H', []); 

            parse(p, varargin{:});

            obj.h = p.Results.h;
            obj.H = p.Results.H;
        end

        function measurement = estimate_measurement(obj,state)
            if ~isempty(obj.h)
                measurement = obj.h(state);
            elseif ~isempty(obj.H)
                measurement = obj.getMeasurementMatrix(state) * state;
            end
        end

        function measurementMatrix = getMeasurementMatrix(obj,state)
            if isa(obj.H,'function_handle')
                measurementMatrix = obj.H(state); 
            else
                measurementMatrix = obj.H;
            end
        end
    end

    methods (Abstract)
        nextDist = predict(obj, timestep, dt, prevDist, varargin)
        [nextDist, likelihood] = correct(obj, dt, meas, prevDist,varargin)
    end

end