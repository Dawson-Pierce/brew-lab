classdef GLMB < handle
    %GLMB Generalized Labeled Multi-Bernoulli filter wrapper.

    properties (SetAccess = private)
        handle_    uint64
        dist_type_ string
    end

    methods
        function obj = GLMB(varargin)
            p = inputParser;
            addParameter(p, 'filter', []);
            addParameter(p, 'birth_model', []);
            addParameter(p, 'prob_detection', 0.9);
            addParameter(p, 'prob_survive', 0.99);
            addParameter(p, 'clutter_rate', 0.0);
            addParameter(p, 'clutter_density', 0.0);
            addParameter(p, 'prune_threshold_hypothesis', 1e-3);
            addParameter(p, 'prune_threshold_bernoulli', 1e-3);
            addParameter(p, 'max_hypotheses', 100);
            addParameter(p, 'extract_threshold', 0.5);
            addParameter(p, 'gate_threshold', 9.0);
            addParameter(p, 'k_best', 5);
            parse(p, varargin{:});

            filt  = p.Results.filter;
            birth = p.Results.birth_model;
            obj.dist_type_ = filt.dist_type_;

            params = struct( ...
                'prob_detection',  p.Results.prob_detection, ...
                'prob_survive',    p.Results.prob_survive, ...
                'clutter_rate',    p.Results.clutter_rate, ...
                'clutter_density', p.Results.clutter_density, ...
                'prune_threshold_hypothesis', p.Results.prune_threshold_hypothesis, ...
                'prune_threshold_bernoulli', p.Results.prune_threshold_bernoulli, ...
                'max_hypotheses',  p.Results.max_hypotheses, ...
                'extract_threshold', p.Results.extract_threshold, ...
                'gate_threshold',  p.Results.gate_threshold, ...
                'k_best',          p.Results.k_best);

            obj.handle_ = brew_mex('create_rfs', 'GLMB', obj.dist_type_, ...
                filt.handle_, birth.handle_, params);
        end

        function predict(obj, dt)
            brew_mex('rfs_predict', obj.handle_, dt);
        end

        function correct(obj, varargin)
            if nargin == 3, measurements = varargin{2};
            else, measurements = varargin{1}; end
            if isempty(measurements), measurements = zeros(1, 0); end
            brew_mex('rfs_correct', obj.handle_, measurements);
        end

        function est = cleanup(obj)
            raw = brew_mex('rfs_cleanup_and_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end

        function est = extract(obj)
            raw = brew_mex('rfs_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end

        function card = cardinality(obj)
            card = brew_mex('rfs_get_cardinality', obj.handle_);
        end

        function h = track_histories(obj)
            h = brew_mex('rfs_get_track_histories', obj.handle_);
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
