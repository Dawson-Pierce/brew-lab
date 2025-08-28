classdef TrajectoryGaussianEKF < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class for gaussian
    % trajectories

    properties
        L_window % For efficient computing of trajectory matrices
    end

    methods
        function obj = TrajectoryGaussianEKF(varargin)
            p = inputParser;
            p.CaseSensitive = true;
            p.KeepUnmatched = true; 

            addParameter(p,'L',50)

            parse(p,varargin{:})
            
            obj@BREW.filters.FiltersBase(p.Unmatched);

            obj.L_window = p.Results.L;

        end
        function nextDist = predict(obj, dt, prevDist, varargin)
            p = inputParser; 
            p.CaseSensitive = true; 
            p.KeepUnmatched   = true; 
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

            nextDist = prevDist;
            nextDist.means = newMean;
            nextDist.covariances = newCov;
            nextDist.cov_history(:,:,end+1) = newCov((end-prevDist.state_dim+1):end,(end-prevDist.state_dim+1):end);

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

            % H_dot = kron([zeros(1,prevDist.window_size-1), 1],H);

            % est_meas = H_dot * prevDist.means;

            epsilon = meas - est_meas; 

            if ~isempty(meas_noise)
                S = H_dot * prevCov * H_dot' + meas_noise; 
            else
                S = H_dot * prevCov * H_dot' + obj.measurement_noise; 
            end

            K = prevCov * H_dot' * S^-1;

            newCov = prevDist.covariances - K * H_dot * prevDist.covariances;

            nextDist = prevDist; 
            nextDist.means = prevDist.means + K * epsilon;
            nextDist.covariances = newCov;
            nextDist.cov_history(:,:,end) = newCov((end-prevDist.state_dim+1):end,(end-prevDist.state_dim+1):end);
            
            likelihood = mvnpdf(meas, est_meas, S);
        
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