classdef (Abstract) FiltersBase < handle
    % FiltersBase class
    %
    % Defines all common functions and properties for each filter

    properties
        dyn_obj_
        f
        h
        H
        process_noise
        measurement_noise
    end

    methods
        function obj = FiltersBase(varargin)
            p = inputParser; 
            addParameter(p, 'dyn_obj', []);  % plug and play dynamic object
            addParameter(p, 'F', []); % F for linear kinematics
            addParameter(p, 'G', []); % G for linear kinematics
            addParameter(p, 'f', []); % function handle for kinematics
            addParameter(p, 'm', []); % function handle for extent dynamics
            addParameter(p, 'M', []); % function handle for extent rotation

            addParameter(p, 'h', []); % function handle for extent dynamics
            addParameter(p, 'H', []); % function handle for extent rotation

            % Parse known arguments
            parse(p, varargin{:});

            obj.setStateModel('dyn_obj',p.Results.dyn_obj,'F',p.Results.F,...
                'G',p.Results.G,'f',p.Results.f,'m',p.Results.m,'M',p.Results.M) 
        end 

        function setStateModel(obj,varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'F', []);
            addParameter(p, 'G', []);
            addParameter(p, 'f', []);
            addParameter(p, 'm', []);
            addParameter(p, 'M', []);
            
            parse(p, varargin{:});
            
            if ~isempty(p.Results.dyn_obj)
                obj.dyn_obj_ = p.Results.dyn_obj; 
            elseif ~isempty(p.Results.F) && ~isempty(p.Results.G)
                obj.dyn_obj_ = BREW.dynamics.discrete.LinearModel(p.Results.F,p.Results.G);
            end
        end

        function setMeasurementModel(obj,varargin)
        end
    end

    methods (Abstract)
        nextDist = predict(obj, timestep, dt, prevDist, u)
        [nextDist, likelihood] = correct(obj, timestep, dt, prevDist, u)
    end

end