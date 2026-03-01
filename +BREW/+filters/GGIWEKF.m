classdef GGIWEKF < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "GGIW"
    end
    properties
        dyn_obj_
        H_
        process_noise_
        measurement_noise_
        temporal_decay_
        forgetting_factor_
        scaling_parameter_
    end
    methods
        function obj = GGIWEKF(varargin)
            p = inputParser;
            addParameter(p, 'dyn_obj', []);
            addParameter(p, 'process_noise', []);
            addParameter(p, 'H', []);
            addParameter(p, 'measurement_noise', []);
            addParameter(p, 'temporal_decay', 1.0);
            addParameter(p, 'forgetting_factor', 1.0);
            addParameter(p, 'scaling_parameter', 1.0);
            parse(p, varargin{:});
            R = p.Results.measurement_noise;
            obj.dyn_obj_ = p.Results.dyn_obj;
            obj.H_ = p.Results.H;
            obj.process_noise_ = p.Results.process_noise;
            obj.measurement_noise_ = R;
            obj.temporal_decay_ = p.Results.temporal_decay;
            obj.forgetting_factor_ = p.Results.forgetting_factor;
            obj.scaling_parameter_ = p.Results.scaling_parameter;
            obj.handle_ = brew_mex('create_filter', 'GGIWEKF', ...
                p.Results.dyn_obj.handle_, p.Results.process_noise, ...
                p.Results.H, R, ...
                p.Results.temporal_decay, ...
                p.Results.forgetting_factor, ...
                p.Results.scaling_parameter);
        end

        function setProcessNoise(obj, Q)
            obj.process_noise_ = Q;
        end

        function setMeasurementNoise(obj, R)
            obj.measurement_noise_ = R;
        end

        function nextDist = predict(obj, dt, prevDist, varargin)
            %PREDICT GGIW-EKF predict on a single-component GGIW Mixture.
            ip = inputParser;
            ip.CaseSensitive = true;
            ip.KeepUnmatched = true;
            addParameter(ip, 'forgetting_factor', obj.forgetting_factor_);
            addParameter(ip, 'tau', obj.temporal_decay_);
            parse(ip, varargin{:});

            prevState = prevDist.means{1};
            prevCov = prevDist.covariances{1};
            prevAlpha = prevDist.alphas(1);
            prevBeta = prevDist.betas(1);
            prevV = prevDist.Vs{1};
            prevv = prevDist.vs(1);
            d = size(prevV, 1);

            nextState = obj.dyn_obj_.propagateState(dt, prevState, ip.Unmatched);
            F = obj.dyn_obj_.getStateMat(dt, prevState);
            nextCov = F * prevCov * F' + obj.process_noise_;

            nextAlpha = prevAlpha / ip.Results.forgetting_factor;
            nextBeta = prevBeta / ip.Results.forgetting_factor;
            nextv = 2*d + 2 + exp(-dt / ip.Results.tau) * (prevv - 2*d - 2) + 1e-3;
            nextV = (nextv - 2*d - 2) * ...
                (prevv - 2*d - 2)^-1 * ...
                obj.dyn_obj_.propagate_extent(dt, prevState, prevV);

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "GGIW";
            nextDist.means = {nextState};
            nextDist.covariances = {nextCov};
            nextDist.weights = prevDist.weights;
            nextDist.alphas = nextAlpha;
            nextDist.betas = nextBeta;
            nextDist.vs = nextv;
            nextDist.Vs = {nextV};
        end

        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist, varargin)
            %CORRECT GGIW-EKF correct on a single-component GGIW Mixture.
            ip = inputParser;
            ip.CaseSensitive = true;
            addParameter(ip, 'scaling_parameter', obj.scaling_parameter_);
            parse(ip, varargin{:});

            rho = ip.Results.scaling_parameter;

            prevState = prevDist.means{1};
            prevCov = prevDist.covariances{1};
            prevAlpha = prevDist.alphas(1);
            prevBeta = prevDist.betas(1);
            prevV = prevDist.Vs{1};
            prevv = prevDist.vs(1);
            d = size(prevV, 1);

            H = obj.H_;
            if isa(H, 'function_handle')
                H = H(prevState);
            end
            est_meas = H * prevState;

            W = size(meas, 2);
            mean_meas = mean(meas, 2);
            diff_Z = meas - mean_meas;
            scatter_meas = diff_Z * diff_Z';

            X_hat = prevV / (prevv - 2*d - 2);
            sqrtXhat = BREW.filters.GGIWEKF.sqrtmDiag_(X_hat);

            epsilon = mean_meas - est_meas;
            N = epsilon * epsilon';

            R_hat = rho * X_hat + obj.measurement_noise_;
            sqrtRhat = BREW.filters.GGIWEKF.sqrtmDiag_(R_hat);
            sqrtRhat = 0.5 * (sqrtRhat' + sqrtRhat);

            S = H * prevCov * H' + R_hat / W;
            K = prevCov * H' / S;

            sqrtXinvR = sqrtXhat / sqrtRhat;
            sqrtXinvR = 0.5 * (sqrtXinvR' + sqrtXinvR);

            N_hat = sqrtXinvR * N / (sqrtRhat') * sqrtXhat';

            nextAlpha = prevAlpha + W;
            nextBeta = prevBeta + 1;
            nextState = prevState + K * epsilon;
            nextCov = prevCov - K * H * prevCov;
            nextCov = (nextCov + nextCov') / 2;
            nextv = prevv + W;
            nextV = prevV + N_hat + scatter_meas;
            nextV = 0.5 * (nextV' + nextV);

            nextDist = BREW.models.Mixture();
            nextDist.dist_type = "GGIW";
            nextDist.means = {nextState};
            nextDist.covariances = {nextCov};
            nextDist.weights = prevDist.weights;
            nextDist.alphas = nextAlpha;
            nextDist.betas = nextBeta;
            nextDist.vs = nextv;
            nextDist.Vs = {nextV};

            % Log-likelihood
            v0 = prevv; V0 = prevV;
            v1 = nextv; V1 = nextV;
            a0 = prevAlpha; b0 = prevBeta;
            a1 = nextAlpha; b1 = nextBeta;

            log_likelihood = ...
                (v0 - d - 1)/2 * log(det(V0)) ...
              - (v1 - d - 1)/2 * log(det(V1)) ...
              + sum(gammaln((v1 - d - 1)/2 + (1 - (1:d))/2)) ...
              - sum(gammaln((v0 - d - 1)/2 + (1 - (1:d))/2)) ...
              + 0.5 * log(det(X_hat)) ...
              - 0.5 * log(det(S)) ...
              + gammaln(a1) - gammaln(a0) ...
              + a0 * log(b0) - a1 * log(b1) ...
              - (W * log(pi) + log(W)) * d / 2;

            likelihood = exp(log_likelihood);
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end

    methods (Static, Access = private)
        function sqrtP = sqrtmDiag_(P)
            [Q, D] = eig((P+P')/2);
            dd = diag(D);
            dd = max(real(dd), eps(class(P)));
            D = diag(dd);
            sqrtP = real(Q * sqrt(D) * Q');
        end
    end
end
