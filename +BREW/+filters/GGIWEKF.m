classdef GGIWEKF < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class for GGIW class 

    % most of the stuff for dynamics model and measurement model is 
    % inherited, just update the predict and correct methodologies 

    % dynamics model also has capability to be an input into the filter so
    % we can change it mid tracking application

    methods  
        function nextDist = predict(obj, dt, prevDist, varargin)
            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p, 'forgetting_factor', 1);
            addParameter(p, 'tau', 1);
            addParameter(p,'u',0);
            addParameter(p,'dyn_obj',[]); % adds capability to change dynamics model
            addParameter(p,'process_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:});

            u = p.Results.u;

            dyn_obj = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise;
           
            prevAlpha = prevDist.alpha;
            prevBeta = prevDist.beta;
            prevState = prevDist.mean;
            prevCov = prevDist.covariance; 
            prevIWdof = prevDist.IWdof;
            prevIWshape = prevDist.IWshape; 

            if ~isempty(dyn_obj)
                nextState = dyn_obj.propagateState(dt, prevState, u);
                F =  dyn_obj.getStateMat(dt, prevState);
            elseif ~isempty(obj.dyn_obj_)
                nextState = obj.dyn_obj_.propagateState(dt, prevState, u);
                F =  obj.dyn_obj_.getStateMat(dt, prevState);
            end
            if ~isempty(proc_noise)
                nextCov = F * prevCov * F' + proc_noise;
            else
                nextCov = F * prevCov * F' + obj.process_noise;
            end

            nextAlpha = prevAlpha / p.Results.forgetting_factor; 
            nextBeta = prevBeta / p.Results.forgetting_factor;
            nextIWdof = 2*prevDist.d + 2 + exp(-dt / p.Results.tau) * (prevIWdof - 2*prevDist.d - 2); 
            nextIWshape = (nextIWdof - 2*prevDist.d - 2) * ...
                (prevIWdof - 2*prevDist.d - 2)^-1 * ...
                obj.dyn_obj_.propagate_extent(prevState, prevIWshape); % this last function computes M*V*M'

            % Create the next distribution for GGIW
            nextDist = BREW.distributions.GGIW(nextAlpha, nextBeta, nextState, nextCov, nextIWdof, nextIWshape);
        
        end
        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist, varargin) 

            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p,'H',[]); % could be constant, could be function handle ie H = @(x) ...
            addParameter(p,'h',[]); % expected to be function handle ie h = @(x) ...
            addParameter(p,'meas_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:});

            h_input = p.Results.h;
            H_input = p.Results.H;
            meas_noise = p.Results.meas_noise;

            prevAlpha = prevDist.alpha;
            prevBeta = prevDist.beta;
            prevState = prevDist.mean;
            prevCov = prevDist.covariance; 
            prevIWdof = prevDist.IWdof;
            prevIWshape = prevDist.IWshape; 
            d = prevDist.d;

            if ~isempty(h_input)
                est_meas = h_input(prevState);
            else
                est_meas = obj.estimate_measurement(prevState); 
            end 

            if ~isempty(H_input)
                if isa(H_input,'function_handle')
                    H = H_input(prevState);
                else
                    H = H_input;
                end
            else
                H = obj.getMeasurementMatrix(prevState);
            end

            W = size(meas,2);

            mean_meas = mean(meas,2);

            diff_Z = meas - mean_meas;

            scatter_meas = diff_Z * diff_Z';

            X_hat = prevIWshape / (prevIWdof - 2*d - 2);

            epsilon = mean_meas - est_meas;

            N = epsilon * epsilon';

            if ~isempty(meas_noise)
                S = H * prevCov * H' + X_hat / W + meas_noise;
            else
                S = H * prevCov * H' + X_hat / W + obj.measurement_noise;
            end

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