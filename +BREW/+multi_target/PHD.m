classdef PHD < BREW.multi_target.RFSBase
    % Probability Hypothesis Density class

    properties 
        prune_threshold
        max_terms
        merge_threshold
        extract_threshold
        gate_threshold
        extracted_mix
        cluster_obj 
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
            addParameter(p, 'clutter_rate', 0.00001);
            addParameter(p, 'clutter_density', 0.00001);

            % These are PHD specific
            addParameter(p,'prune_threshold',0.0001)
            addParameter(p,'max_terms',100)
            addParameter(p,'extract_threshold',0.5)
            addParameter(p,'merge_threshold',4)
            addParameter(p,'gate_threshold',9)
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
            obj.gate_threshold = p.Results.gate_threshold;
            obj.merge_threshold = p.Results.merge_threshold;
            obj.cluster_obj = p.Results.cluster_obj;

            obj.Mix = obj.birth_model.copy();
            obj.extracted_mix = {};

        end

        function obj = predict(obj, dt, varargin) 
            
            obj.predict_prob_density(dt,varargin);

            obj.Mix.addComponents(obj.birth_model.copy());

            if obj.birth_model.isTrajectory
                obj.birth_model.idx = obj.birth_model.idx + 1;
            end 

            % for kk = 1:length(obj.birth_model)
            %     if isa(obj.birth_model.distributions{kk},"BREW.distributions.TrajectoryBaseModel")
            %         prev = obj.birth_model.distributions{kk}.init_idx;
            %         obj.birth_model.distributions{kk}.init_idx = prev + 1;
            %     end
            % end

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
            extracted_dist = obj.Mix.copy();
            extracted_dist.extract_mix(obj.extract_threshold);
            obj.extracted_mix{end+1} = extracted_dist.copy();
        end

        function [extracted_dist,obj] = cleanup(obj)
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
                weights = obj.Mix.weights(idx(1:obj.max_terms)); 

                obj.Mix.distributions = dist;
                obj.Mix.weights = weights * w/sum(weights);
            end
        end

        function obj = correct_prob_density(obj, dt, meas, varargin)
            det_weights = obj.prob_detection * obj.Mix.weights;

            new_mix = obj.Mix.copy();
            new_mix.distributions = obj.Mix.distributions([]);  % empty of same type
            new_mix.weights = zeros(0, 1, 'like', obj.Mix.weights);

            for z = 1:numel(meas)
                [temp_mix, qz] = obj.filter_.correct(dt, meas{z}, obj.Mix.copy()); 
                w_temp = det_weights .* qz;
                w_temp = w_temp ./ (obj.clutter_density + obj.clutter_rate + sum(w_temp));

                temp_mix.weights = w_temp;
                new_mix.addComponents(temp_mix); 
            end

            obj.Mix = new_mix; 
        
        end


        function obj = predict_prob_density(obj,dt,varargin)
            % Predict the mixture probabilities
            obj.Mix.weights = obj.Mix.weights * obj.prob_survive;  
            obj.Mix = obj.filter_.predict(dt,obj.Mix);
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