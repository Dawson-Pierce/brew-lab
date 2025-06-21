classdef ExtendedKalmanFilter < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class

    % most of the stuff for dynamics model and measurement model is 
    % inherited, just update the predict and correct methodologies 

    methods  
        function nextDist = predict(obj, timestep, dt, prevDist, u, varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'forgetting_factor', 1);
            addParameter(p, 'tau', 1);

            % Parse known arguments
            parse(p, varargin{:});

            if isa(prevDist,'BREW.distributions.Gaussian')
                prevState = prevDist.mean;
                prevCov = prevDist.covariance; 

                nextState = obj.dyn_obj_.propagateState(timestep, dt, prevState, u);

                F =  obj.dyn_obj_.getStateMat(timestep, dt, prevState);
                nextCov = F * prevCov * F' + obj.process_noise;

                nextDist = BREW.distributions.Gaussian(nextState,nextCov);

            elseif isa(prevDist,'BREW.distributions.GGIW')
                prevAlpha = prevDist.alpha;
                prevBeta = prevDist.beta;
                prevState = prevDist.mean;
                prevCov = prevDist.covariance; 
                prevIWdof = prevDist.IWdof;
                prevIWshape = prevDist.IWshape; 

                nextState = obj.dyn_obj_.propagateState(timestep, dt, prevState, u);

                F =  obj.dyn_obj_.getStateMat(timestep, dt, prevState);
                nextCov = F * prevCov * F' + obj.process_noise;

                nextAlpha = prevAlpha / p.Results.forgetting_factor; 
                nextBeta = prevBeta / p.Results.forgetting_factor;
                nextIWdof = 2*prevDist.d + 2 + exp(-dt / p.Results.tau) * (prevIWdof - 2*prevDist.d - 2); 
                nextIWshape = (nextIWdof - 2*prevDist.d - 2) * ...
                    (prevIWdof - 2*prevDist.d - 2)^-1 * ...
                    obj.dyn_obj_.propagate_extent(prevState, prevIWshape); % this last function computes M*V*M'

                % Create the next distribution for GGIW
                nextDist = BREW.distributions.GGIW(nextAlpha, nextBeta, nextState, nextCov, nextIWdof, nextIWshape);
            end
        
        end
        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist) 

            if isa(prevDist,'BREW.distributions.Gaussian')
                prevState = prevDist.mean;
                prevCov = prevDist.covariance;

                est_meas = obj.estimate_measurement(prevState); % Defined in base class

                epsilon = meas - est_meas;

                H = obj.getMeasurementMatrix(prevState);

                S = H * prevCov * H' + obj.measurement_noise;
                S = 0.5 * (S + S');

                K = prevCov * H' * inv(S);

                nextState = prevState + K * epsilon;

                nextCov = (eye(size(prevCov,1))-K*H)*prevCov*(eye(size(prevCov,1))-K*H)' + K*obj.measurement_noise*K';

                nextDist = BREW.distributions.Gaussian(nextState, nextCov);
                likelihood = mvnpdf(meas, est_meas, S);

            elseif isa(prevDist,'BREW.distributions.GGIW')
                prevAlpha = prevDist.alpha;
                prevBeta = prevDist.beta;
                prevState = prevDist.mean;
                prevCov = prevDist.covariance; 
                prevIWdof = prevDist.IWdof;
                prevIWshape = prevDist.IWshape; 
                d = prevDist.d;

                est_meas = obj.estimate_measurement(prevState); % Defined in base class

                W = size(meas,2);

                mean_meas = mean(meas,2);

                diff_Z = meas - mean_meas;

                scatter_meas = diff_Z * diff_Z';

                X_hat = prevIWshape / (prevIWdof - 2*d - 2);

                epsilon = mean_meas - est_meas;

                N = epsilon * epsilon';

                H = obj.getMeasurementMatrix(prevState);

                S = H * prevCov * H' + X_hat / W + obj.measurement_noise;

                K = prevCov * H' * S^-1;

                N_hat = X_hat^(1/2) * S^(-1/2) * N * S^(-dt/2) * X_hat^(dt/2);

                nextAlpha = prevAlpha + W;
                nextBeta = prevBeta + 1;
                nextState = prevState + K * epsilon;
                nextCov = prevCov - K * H * prevCov; 
                nextIWdof = prevIWdof + W;
                nextIWshape = prevIWshape + N_hat + scatter_meas;

                nextDist = BREW.distributions.GGIW(nextAlpha,nextBeta,nextState,nextCov,nextIWdof,nextIWshape); 
                
                v0 = prevIWdof;   V0 = prevIWshape;
                v1 = nextIWdof;   V1 = nextIWshape;
                a0 = prevAlpha;   b0 = prevBeta;
                a1 = nextAlpha;   b1 = nextBeta;
                
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
        
        end
    end 

end