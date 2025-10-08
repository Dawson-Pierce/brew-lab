classdef (Abstract) RFSBase < handle & matlab.mixin.Copyable
    % FiltersBase class
    %
    % Defines all common functions and properties for each filter

    properties
        filter_
        birth_model
        prob_detection
        prob_survive
        clutter_rate 
        clutter_density
    end

    methods
        function obj = RFSBase(varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'filter', []); 
            addParameter(p, 'birth_model', []); 
            addParameter(p, 'prob_detection', 1);
            addParameter(p, 'prob_survive', 1);
            addParameter(p, 'clutter_rate', 0);
            addParameter(p, 'clutter_density', 0);

            % Parse known arguments
            parse(p, varargin{:});

            obj.filter_ = p.Results.filter; 
            obj.birth_model = p.Results.birth_model;
            obj.prob_detection = p.Results.prob_detection;
            obj.prob_survive = p.Results.prob_survive;
            obj.clutter_rate = p.Results.clutter_rate;
            obj.clutter_density = p.Results.clutter_density;
  
        end 

        function setProcessNoise(obj,Q)
            if isempty(obj.filter_)
                error("Please choose a filter first. ")
            end
            obj.filter_.process_noise = Q;
        end

        function setMeasurementNoise(obj,R)
            if isempty(obj.filter_)
                error("Please choose a filter first. ")
            end
            obj.filter_.measurement_noise = R;
        end

        function setStateModel(obj,varargin)
            if isempty(obj.filter_)
                error("Please choose a filter first. ")
            end
            obj.filter_.setStateModel(varargin);
        end

        function setMeasurementModel(obj,varargin)
            if isempty(obj.filter_)
                error("Please choose a filter first. ")
            end
            obj.filter_.setMeasurementModel(varargin);
        end
    end

    methods (Abstract)
        predict(obj, timestep, dt, varargin)
        correct(obj, dt, meas) 
        cleanup(obj)
    end

end