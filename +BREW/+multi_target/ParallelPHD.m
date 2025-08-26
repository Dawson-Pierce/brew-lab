classdef ParallelPHD < BREW.multi_target.RFSBase
    % Probability Hypothesis Density class

    properties 
        prune_threshold
        max_terms
        merge_threshold
        extract_threshold
        extracted_mix
        cluster_obj 
        Mix
    end 

    methods
        function obj = ParallelPHD(varargin)
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
            addParameter(p,'merge_threshold',4)
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
            obj.merge_threshold = p.Results.merge_threshold;
            obj.cluster_obj = p.Results.cluster_obj;

            obj.Mix = obj.birth_model.copy();
            obj.extracted_mix = {};

        end

        function obj = predict(obj, dt, varargin) 
            
            obj.predict_prob_density(dt,varargin);

            obj.Mix.addComponents(obj.birth_model.copy());

            for kk = 1:length(obj.birth_model)
                if isa(obj.birth_model.distributions{kk},"BREW.distributions.TrajectoryBaseModel")
                    prev = obj.birth_model.distributions{kk}.init_idx;
                    obj.birth_model.distributions{kk}.init_idx = prev + 1;
                end
            end

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
            undetected_mix = obj.Mix.copy(); 
            undetected_mix.weights = (1 - obj.prob_detection) * undetected_mix.weights;

            % Correct the mixture for each measurement
            obj.correct_prob_density(dt, meas_new, varargin);

            obj.Mix.addComponents(undetected_mix);

        end

        function extracted_dist = extract(obj)
            extracted_dist = obj.Mix.extract_mix(obj.extract_threshold);
            obj.extracted_mix{end+1} = extracted_dist;
        end

        function extracted_dist = cleanup(obj)
            obj.prune();
            obj.merge();
            obj.cap();
            extracted_dist = obj.extract();
        end 
    end

    methods (Access = protected)

        function obj = prune(obj) 
            idx = find(obj.Mix.weights < obj.prune_threshold); 
            obj.Mix.removeComponents(idx);
        end

        function obj = merge(obj)
            obj.Mix.merge(obj.merge_threshold);
        end

        function obj = cap(obj)
            if length(obj.Mix) > obj.max_terms
                w = sum(obj.Mix.weights);
                [~, idx] = sort(obj.Mix.weights, 'descend');

                dist = obj.Mix.distributions(idx(1:obj.max_terms));
                weights = obj.Mix.weights(idx(1:obj.max_terms)); % * w / (sum(obj.Mix.weights(idx(1:obj.max_terms))));

                obj.Mix.distributions = dist;
                obj.Mix.weights = weights;
            end
        end


        function obj = correct_prob_density(obj, dt, meas, varargin)
            Z  = numel(meas);
            d_in = obj.Mix.distributions; 
            K  = numel(d_in);
            w0 = (obj.prob_detection * obj.Mix.weights); 
            C  = obj.clutter_rate * obj.clutter_density; 

            dist_grid = cell(Z, K);
            w_grid    = zeros(Z, K);
            
            parfor z = 1:Z
                y   = meas{z}; 
                dist_z = cell(1, K);
                w_lst  = zeros(1, K); 
                for k = 1:K
                    [dist_z{k}, qz] = obj.filter_.correct(dt, y, d_in{k});
                    w_lst(k) = qz * w0(k);
                end 
                w_lst = w_lst ./ (C + sum(w_lst)); 
                dist_grid(z,:) = dist_z;
                w_grid(z,:) = w_lst;
            end
            
            obj.Mix.distributions = reshape(dist_grid.', 1, []); 
            obj.Mix.weights = reshape(w_grid.',    1, []); 
        end

        function obj = predict_prob_density(obj,dt,varargin)
            % Predict the mixture probabilities
            new_weights = zeros(size(obj.Mix.weights));
            new_distributions = cell(size(obj.Mix.distributions));
        
            parfor k = 1:length(obj.Mix)
                new_weights(k) = obj.Mix.weights(k) * obj.prob_survive;
                new_distributions{k} = obj.filter_.predict(dt, obj.Mix.distributions{k});
            end
        
            % Now assign outside of parfor
            obj.Mix.weights = new_weights;
            obj.Mix.distributions = new_distributions;

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