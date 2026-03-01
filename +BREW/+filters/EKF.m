classdef EKF < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "Gaussian"
    end
    properties
        dyn_obj_
        H_
        process_noise_
        measurement_noise_
    end
    methods
        function obj = EKF(varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'process_noise', []);
            addParameter(p, 'H', []);
            addParameter(p, 'measurement_noise', []);
            parse(p, varargin{:});
            R = p.Results.measurement_noise;
            obj.dyn_obj_ = p.Results.dyn_obj;
            obj.H_ = p.Results.H;
            obj.process_noise_ = p.Results.process_noise;
            obj.measurement_noise_ = R;
            obj.handle_ = brew_mex('create_filter', 'EKF', ...
                p.Results.dyn_obj.handle_, p.Results.process_noise, ...
                p.Results.H, R);
        end

        function setProcessNoise(obj, Q)
            obj.process_noise_ = Q;
        end

        function setMeasurementNoise(obj, R)
            obj.measurement_noise_ = R;
        end

        function nextDist = predict(obj, dt, prevDist, varargin)
            %PREDICT EKF predict step on a single-component Mixture.
            prevState = prevDist.means{1};
            prevCov = prevDist.covariances{1};

            nextState = obj.dyn_obj_.propagateState(dt, prevState);
            F = obj.dyn_obj_.getStateMat(dt, prevState);
            nextCov = F * prevCov * F' + obj.process_noise_;

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "Gaussian";
            nextDist.means = {nextState};
            nextDist.covariances = {nextCov};
            nextDist.weights = prevDist.weights;
        end

        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist, varargin)
            %CORRECT EKF correct step on a single-component Mixture.
            prevState = prevDist.means{1};
            prevCov = prevDist.covariances{1};

            H = obj.H_;
            if isa(H, 'function_handle')
                H = H(prevState);
            end
            est_meas = H * prevState;
            epsilon = meas - est_meas;

            R = obj.measurement_noise_;
            S = H * prevCov * H' + R;
            S = 0.5 * (S + S');
            K = (prevCov * H') / S;

            nextState = prevState + K * epsilon;
            I = eye(length(prevState));
            nextCov = (I - K*H) * prevCov * (I - K*H)' + K * R * K';

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "Gaussian";
            nextDist.means = {nextState};
            nextDist.covariances = {nextCov};
            nextDist.weights = prevDist.weights;

            likelihood = mvnpdf(meas, est_meas, S);
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
