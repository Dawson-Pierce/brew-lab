classdef TrajectoryGGIWEKF < BREW.filters.TrajectoryGaussianEKF 

    methods 
        function nextDist = predict(obj, dt, prevDist, varargin)
            p = inputParser; 
            p.CaseSensitive = true; 
            p.KeepUnmatched   = true; 
            addParameter(p, 'forgetting_factor', 1);
            addParameter(p, 'tau', 1);
            addParameter(p,'u',0);
            addParameter(p,'dyn_obj',[]); % adds capability to change dynamics model
            addParameter(p,'process_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:}); 

            dyn_obj = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise; 

            if (prevDist.window_size < obj.L_window)
                start_idx = 1;
            else
                start_idx = prevDist.state_dim+1;
            end

            prevState = prevDist.getLastState();
            prevCov = prevDist.covariances(start_idx:end,start_idx:end); 

            prevAlpha = prevDist.alphas(end);
            prevBeta = prevDist.betas(end); 
            prevIWdof = prevDist.IWdofs(end);
            prevIWshape = prevDist.IWshapes(:,:,end); 

            if ~isempty(dyn_obj)
                nextState = dyn_obj.propagateState(dt, prevState, p.Unmatched);
                F =  dyn_obj.getStateMat(dt, prevState);
            elseif ~isempty(obj.dyn_obj_)
                nextState = obj.dyn_obj_.propagateState(dt, prevState, p.Unmatched);
                F =  obj.dyn_obj_.getStateMat(dt, prevState);
            end

            if (prevDist.window_size < obj.L_window)
                F_dot = kron([zeros(1,prevDist.window_size-1), 1],F);
            else
                F_dot = kron([zeros(1,obj.L_window-2), 1],F);
            end

            if isempty(proc_noise)
                proc_noise = obj.process_noise;
            end 

            newCov = [prevCov, prevCov * F_dot'; F_dot * prevCov, F_dot * prevCov * F_dot' + proc_noise]; 

            newMean = [prevDist.means(start_idx:end); nextState];

            nextAlpha = prevAlpha / p.Results.forgetting_factor; 
            nextBeta = prevBeta / p.Results.forgetting_factor;
            nextIWdof = 2*prevDist.d + 2 + exp(-dt / p.Results.tau) * (prevIWdof - 2*prevDist.d - 2); 
            nextIWshape = (nextIWdof - 2*prevDist.d - 2) * ...
                (prevIWdof - 2*prevDist.d - 2)^-1 * ...
                obj.dyn_obj_.propagate_extent(prevState, prevIWshape); % this last function computes M*V*M'

            nextDist = prevDist;
            nextDist.means = newMean;
            nextDist.covariances = newCov;
            nextDist.cov_history(:,:,end+1) = newCov((end-prevDist.state_dim+1):end,(end-prevDist.state_dim+1):end);
            nextDist.alphas(end+1) = nextAlpha;
            nextDist.betas(end+1) = nextBeta;
            nextDist.IWdofs(end+1) = nextIWdof;
            nextDist.IWshapes(:,:,end+1) = nextIWshape;

            if (prevDist.window_size < obj.L_window)
                nextDist.mean_history = nextDist.RearrangeStates();
            else
                nextDist.mean_history(:,end-obj.L_window+2:end+1) = nextDist.RearrangeStates();
            end
            nextDist.window_size = nextDist.window_size+1;
        
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

            prevState = prevDist.getLastState();
            prevCov = prevDist.covariances;  

            prevAlpha = prevDist.alphas(end);
            prevBeta = prevDist.betas(end); 
            prevIWdof = prevDist.IWdofs(end);
            prevIWshape = prevDist.IWshapes(:,:,end); 
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

            if (prevDist.window_size - obj.L_window) < 0
                H_dot = kron([zeros(1,prevDist.window_size-1), 1],H);
            else
                H_dot = kron([zeros(1,obj.L_window-1), 1],H);
            end

            W = size(meas,2);

            mean_meas = mean(meas,2);

            diff_Z = meas - mean_meas;

            scatter_meas = diff_Z * diff_Z';

            X_hat = prevIWshape / (prevIWdof - 2*d - 2);

            epsilon = mean_meas - est_meas;

            N = epsilon * epsilon'; 

            if ~isempty(meas_noise)
                S = H_dot * prevCov * H_dot' + X_hat / W + meas_noise; 
            else
                S = H_dot * prevCov * H_dot' + X_hat / W + obj.measurement_noise; 
            end

            K = prevCov * H_dot' * S^-1;

            N_hat = X_hat^(1/2) * S^(-1/2) * N * S^(-dt/2) * X_hat^(dt/2);

            newCov = prevDist.covariances - K * H_dot * prevDist.covariances;

            nextAlpha = prevAlpha + W;
            nextBeta = prevBeta + 1; 
            nextIWdof = prevIWdof + W;
            nextIWshape = prevIWshape + N_hat + scatter_meas;

            nextDist = prevDist; 
            nextDist.means = prevDist.means + K * epsilon;
            nextDist.covariances = newCov;
            nextDist.cov_history(:,:,end) = newCov((end-prevDist.state_dim+1):end,(end-prevDist.state_dim+1):end);

            nextDist.alphas(end) = nextAlpha;
            nextDist.betas(end) = nextBeta;
            nextDist.IWdofs(end) = nextIWdof;
            nextDist.IWshapes(:,:,end) = nextIWshape;
            
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

        function val = gate_meas(obj, pred_dist, z, gamma)
            state = pred_dist.getLastState();
            P = pred_dist.getLastCov();
            
            est_z = obj.estimate_measurement(state);

            H = obj.getMeasurementMatrix();

            R = obj.measurement_noise;

            Sj = H * P * H' + R; 
            Sj= (Sj+ Sj')/2; 

            Vs = chol(Sj); 
            inv_sqrt_Sj= inv(Vs);

            iSj= inv_sqrt_Sj*inv_sqrt_Sj'; 

            nu = z - est_z;

            dist= nu' * iSj * nu;

            val = dist < gamma;
        end
    end 

end