classdef PHD < BREW.multi_target.RFSBase
    % Probability Hypothesis Density class

    properties 
        prune_threshold
        max_terms
        merge_threshold
        extract_threshold
        extracted_mix
        cluster_obj
        spawn_cov = []; 
        spawn_weight = [];
        enable_spawn = false;
        Mix
    end 

    methods
        function obj = PHD(varargin)
            p = inputParser;
            p.CaseSensitive = true;

            % These are for base class
            addParameter(p, 'filter', []); 
            addParameter(p, 'birth_model', []); 
            addParameter(p, 'prob_detection', 1);
            addParameter(p, 'prob_survive', 1);
            addParameter(p, 'clutter_rate', 0);
            addParameter(p, 'clutter_density', 0);

            % These are PHD specific
            addParameter(p,'prune_threshold',0.0001)
            addParameter(p,'max_terms',100)
            addParameter(p,'extract_threshold',0.5)
            % addParameter(p,'merge_threshold',4)
            addParameter(p,'cluster_obj',[])

            parse(p, varargin{:});

            obj@BREW.multi_target.RFSBase( ...
                'filter', p.Results.filter, ...
                'birth_model', p.Results.birth_model, ...
                'prob_detection', p.Results.prob_detection, ...
                'prob_survive', p.Results.prob_survive, ...
                'clutter_rate', p.Results.clutter_rate, ...
                'clutter_density', p.Results.clutter_density ...
            );

            obj.prune_threshold = p.Results.prune_threshold;
            obj.max_terms = p.Results.max_terms;
            obj.extract_threshold = p.Results.extract_threshold;
            % obj.merge_threshold = p.Results.merge_threshold;
            obj.cluster_obj = p.Results.cluster_obj;

            obj.Mix = obj.birth_model;
            obj.extracted_mix = {};

        end

        function obj = predict(obj, dt, varargin)
            if obj.enable_spawn
                spawn_mix = obj.gen_spawned_targets(obj.Mix);
            end

            obj.Mix = obj.predict_prob_density(dt,obj.Mix,varargin);

            if obj.enable_spawn
                obj.Mix = obj.Mix.addComponents(spawn_mix);
            end 

            obj.Mix = obj.Mix.addComponents(obj.birth_model);
        end

        function obj = correct(obj, dt, meas, varargin)
            % Input is expected to be matrix of points, but added logic if
            % it's a cell of measurements

            % Matrix format: M x T where T is each point, and M is
            % dimension of measurement

            if ~isa(meas,'cell')
                if obj.Mix.isExtendedTarget
                    % Clustering needs to happen
                    meas_new = obj.cluster_obj.cluster(meas);
                else
                    meas_new = mat2cell(meas, size(meas,1), ones(1, size(meas,2)));
                end
            else
                meas_new = meas;
            end
            % meas_new is a cell of all measurements

            % in the case that the target is not detected
            undetected_mix = obj.Mix;
            undetected_mix.weights = (1 - obj.prob_detection) * undetected_mix.weights;

            % Correct the mixture for each measurement
            obj.Mix = obj.correct_prob_density(dt, meas_new, obj.Mix, varargin);

            obj.Mix.addComponents(undetected_mix);

        end

        function extracted_dist = extract(obj)
            extracted_dist = obj.Mix.extract_mix(obj.extract_threshold);
            obj.extracted_mix{end+1} = extracted_dist;
        end

        function extracted_dist = cleanup(obj)
            obj.prune();
            obj.cap();
            extracted_dist = obj.extract();
        end 
    end

    methods (Access = protected)

        function obj = prune(obj) 
            idx = find(obj.Mix.weights < obj.prune_threshold); 
            obj.Mix.removeComponents(idx);
        end

        function obj = cap(obj)
            if length(obj.Mix) > obj.max_terms
                w = sum(obj.Mix.weights);
                [~, idx] = sort(obj.Mix.weights, 'descend');

                dist = obj.Mix.distributions(idx(1:obj.max_terms));
                weights = obj.Mix.weights(idx(1:obj.max_terms)) * w / (sum(obj.Mix.weights));

                obj.Mix.distributions = dist;
                obj.Mix.weights = weights;
            end
        end


        function new_mix = correct_prob_density(obj, dt, meas, mix, varargin)
            mix.weights = obj.prob_detection * mix.weights;
            dist = {};
            weights = [];
            for z = 1:length(meas)
                w_lst = []; 
                for k = 1:length(mix)
                    [dist{end+1},qz] = obj.filter_.correct(dt,meas{z},mix.distributions{k});
                    w_lst(end+1) = qz * mix.weights(k);
                end
                weights = [weights, w_lst ./ (obj.clutter_rate + obj.clutter_density + sum(w_lst))];
            end

            new_mix = mix;
            new_mix.distributions = dist;
            new_mix.weights = weights;
        end

        function new_mix = predict_prob_density(obj,dt,mix,varargin)
            % Predict the mixture probabilities

            new_mix = mix;

            for k = 1:length(mix)
                new_mix.weights(k) = new_mix.weights(k) * obj.prob_survive; 
            end

            for k = 1:length(mix) 
                new_mix.distributions{k} = obj.filter_.predict(dt,mix.distributions{k});
            end

        end

        function spawn_mix = gen_spawned_targets(obj,mixture)
            spawn_mix = mixture; % copy mixture, should be generalized for all mixtures
            for k = 1:length(spawn_mix)
                spawn_mix.covariances{k} = obj.spawn_cov; % all mixtures should have a covariance
                spawn_mix.weights(k) = obj.spawn_weight; 
            end
        end
    end

end