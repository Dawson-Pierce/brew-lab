classdef PHD < handle
    %PHD Probability Hypothesis Density filter wrapper.
    %   Wraps brew::multi_target::PHD<T> via the brew_mex gateway.
    %
    %   Usage:
    %       phd = BREW.multi_target.PHD('filter', ekf, 'birth_model', birth, ...
    %           'prob_detection', 0.9, 'prob_survive', 0.99, ...
    %           'clutter_rate', 1e-4, 'clutter_density', 1e-3, ...
    %           'prune_threshold', 1e-4, 'merge_threshold', 4, ...
    %           'max_components', 100, 'extract_threshold', 0.5, ...
    %           'gate_threshold', 9);
    %
    %       phd.predict(dt);
    %       phd.correct(measurements);
    %       est = phd.cleanup();

    properties (SetAccess = private)
        handle_    uint64
        dist_type_ string
    end

    methods
        function obj = PHD(varargin)
            p = inputParser;
            addParameter(p, 'filter', []);
            addParameter(p, 'birth_model', []);
            addParameter(p, 'prob_detection', 0.9);
            addParameter(p, 'prob_survive', 0.99);
            addParameter(p, 'clutter_rate', 0.0);
            addParameter(p, 'clutter_density', 0.0);
            addParameter(p, 'prune_threshold', 1e-4);
            addParameter(p, 'merge_threshold', 4.0);
            addParameter(p, 'max_components', 100);
            % Accept 'max_terms' as alias for 'max_components'
            addParameter(p, 'max_terms', []);
            addParameter(p, 'extract_threshold', 0.5);
            addParameter(p, 'gate_threshold', 9.0);
            parse(p, varargin{:});

            filt  = p.Results.filter;
            birth = p.Results.birth_model;
            obj.dist_type_ = filt.dist_type_;

            max_comp = p.Results.max_components;
            if ~isempty(p.Results.max_terms)
                max_comp = p.Results.max_terms;
            end

            params = struct( ...
                'prob_detection',  p.Results.prob_detection, ...
                'prob_survive',    p.Results.prob_survive, ...
                'clutter_rate',    p.Results.clutter_rate, ...
                'clutter_density', p.Results.clutter_density, ...
                'prune_threshold', p.Results.prune_threshold, ...
                'merge_threshold', p.Results.merge_threshold, ...
                'max_components',  max_comp, ...
                'extract_threshold', p.Results.extract_threshold, ...
                'gate_threshold',  p.Results.gate_threshold);

            obj.handle_ = brew_mex('create_rfs', 'PHD', obj.dist_type_, ...
                filt.handle_, birth.handle_, params);
        end

        function predict(obj, dt)
            %PREDICT Propagate the PHD intensity forward by dt.
            brew_mex('rfs_predict', obj.handle_, dt);
        end

        function correct(obj, varargin)
            %CORRECT Update the PHD with measurements.
            %   correct(measurements)
            %   correct(dt, measurements)  % dt ignored, for backward compat
            if nargin == 3
                measurements = varargin{2};
            else
                measurements = varargin{1};
            end
            if isempty(measurements)
                measurements = zeros(1, 0);
            end
            brew_mex('rfs_correct', obj.handle_, measurements);
        end

        function est = cleanup(obj)
            %CLEANUP Prune, merge, cap, and extract state estimates.
            %   Returns a BREW.models.Mixture object.
            raw = brew_mex('rfs_cleanup_and_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end

        function set_birth_weights(obj, weights)
            %SET_BIRTH_WEIGHTS Modify the birth model weights.
            %   set_birth_weights(weights) sets the birth model component
            %   weights. Pass a scalar to set all components to the same
            %   value, or a vector matching the number of birth components.
            brew_mex('rfs_set_birth_weights', obj.handle_, weights);
        end

        function est = extract(obj)
            %EXTRACT Get current state estimates without cleanup.
            raw = brew_mex('rfs_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
