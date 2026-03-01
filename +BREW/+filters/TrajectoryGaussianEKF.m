classdef TrajectoryGaussianEKF < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "TrajectoryGaussian"
    end
    properties
        dyn_obj_
        H_
        process_noise_
        measurement_noise_
        L_window
    end
    methods
        function obj = TrajectoryGaussianEKF(varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'process_noise', []);
            addParameter(p, 'H', []);
            addParameter(p, 'measurement_noise', []);
            addParameter(p, 'L', 50);
            parse(p, varargin{:});
            R = p.Results.measurement_noise;
            obj.dyn_obj_ = p.Results.dyn_obj;
            obj.H_ = p.Results.H;
            obj.process_noise_ = p.Results.process_noise;
            obj.measurement_noise_ = R;
            obj.L_window = p.Results.L;
            obj.handle_ = brew_mex('create_filter', 'TrajectoryGaussianEKF', ...
                p.Results.dyn_obj.handle_, p.Results.process_noise, ...
                p.Results.H, R, ...
                p.Results.L);
        end

        function nextDist = predict(obj, dt, prevDist, varargin)
            %PREDICT Trajectory Gaussian EKF predict on a single-component Mixture.
            sd = prevDist.state_dim;
            prevMean = prevDist.means{1};
            window_size = numel(prevMean) / sd;

            if window_size < obj.L_window
                start_idx = 1;
            else
                start_idx = sd + 1;
            end

            prevState = prevMean(end-sd+1:end);
            prevCov = prevDist.covariances{1}(start_idx:end, start_idx:end);

            nextState = obj.dyn_obj_.propagateState(dt, prevState);
            F = obj.dyn_obj_.getStateMat(dt, prevState);

            if window_size < obj.L_window
                F_dot = kron([zeros(1, window_size-1), 1], F);
            else
                F_dot = kron([zeros(1, obj.L_window-2), 1], F);
            end

            proc_noise = obj.process_noise_;
            newCov = [prevCov, prevCov * F_dot'; ...
                      F_dot * prevCov, F_dot * prevCov * F_dot' + proc_noise];
            newMean = [prevMean(start_idx:end); nextState];

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "TrajectoryGaussian";
            nextDist.state_dim = sd;
            nextDist.means = {newMean};
            nextDist.covariances = {newCov};
            nextDist.weights = prevDist.weights;
            nextDist.init_indices = prevDist.init_indices;

            % Update mean_histories
            new_win_size = numel(newMean) / sd;
            states_matrix = reshape(newMean, [sd, new_win_size]);
            if ~isempty(prevDist.mean_histories) && ~isempty(prevDist.mean_histories{1})
                prev_hist = prevDist.mean_histories{1};
                if window_size < obj.L_window
                    nextDist.mean_histories = {states_matrix};
                else
                    nextDist.mean_histories = {[prev_hist(:, 1:end-obj.L_window+1), states_matrix]};
                end
            else
                nextDist.mean_histories = {states_matrix};
            end
        end

        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist, varargin)
            %CORRECT Trajectory Gaussian EKF correct on a single-component Mixture.
            sd = prevDist.state_dim;
            prevMean = prevDist.means{1};
            prevCov = prevDist.covariances{1};
            window_size = numel(prevMean) / sd;

            prevState = prevMean(end-sd+1:end);

            H = obj.H_;
            if isa(H, 'function_handle')
                H = H(prevState);
            end
            est_meas = H * prevState;

            if (window_size - obj.L_window) < 0
                H_dot = kron([zeros(1, window_size-1), 1], H);
            else
                H_dot = kron([zeros(1, obj.L_window-1), 1], H);
            end

            epsilon = meas - est_meas;

            S = H_dot * prevCov * H_dot' + obj.measurement_noise_;
            K = prevCov * H_dot' / S;

            newMean = prevMean + K * epsilon;
            newCov = prevCov - K * H_dot * prevCov;

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "TrajectoryGaussian";
            nextDist.state_dim = sd;
            nextDist.means = {newMean};
            nextDist.covariances = {newCov};
            nextDist.weights = prevDist.weights;
            nextDist.init_indices = prevDist.init_indices;

            % Update mean_histories from the corrected mean
            states_matrix = reshape(newMean, [sd, window_size]);
            if ~isempty(prevDist.mean_histories) && ~isempty(prevDist.mean_histories{1})
                prev_hist = prevDist.mean_histories{1};
                T_prev = size(prev_hist, 2);
                T_win = size(states_matrix, 2);
                frozen_len = T_prev - T_win;
                if frozen_len > 0
                    nextDist.mean_histories = {[prev_hist(:, 1:frozen_len), states_matrix]};
                else
                    nextDist.mean_histories = {states_matrix};
                end
            else
                nextDist.mean_histories = {states_matrix};
            end

            likelihood = mvnpdf(meas, est_meas, S);
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
