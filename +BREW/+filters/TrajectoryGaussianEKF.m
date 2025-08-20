classdef TrajectoryGaussianEKF < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class for gaussian
    % trajectories

    methods  
        function nextDist = predict(obj, dt, prevDist, varargin)
            p = inputParser; 
            p.CaseSensitive = true; 
            p.KeepUnmatched   = true; 
            addParameter(p,'dyn_obj',[]); % adds capability to change dynamics model
            addParameter(p,'process_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:});

            fn  = fieldnames(p.Unmatched);
            val = struct2cell(p.Unmatched);
            unmatched = reshape([fn.'; val.'], 1, []);

            dyn_obj = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise; 

            prevState = prevDist.getLastState();
            prevCov = prevDist.covariances; 

            if ~isempty(dyn_obj)
                nextState = dyn_obj.propagateState(dt, prevState, unmatched{:});
                F =  dyn_obj.getStateMat(dt, prevState);
            elseif ~isempty(obj.dyn_obj_)
                nextState = obj.dyn_obj_.propagateState(dt, prevState, unmatched{:});
                F =  obj.dyn_obj_.getStateMat(dt, prevState);
            end

            F_dot = kron([zeros(1,prevDist.window_size-1), 1],F);

            if ~isempty(proc_noise)
                newCov = [prevCov, prevCov * F_dot'; F_dot * prevCov, F_dot * prevCov * F_dot' + proc_noise]; % F * prevCov * F' + proc_noise;
            else
                newCov = [prevCov, prevCov * F_dot'; F_dot * prevCov, F_dot * prevCov * F_dot' + obj.process_noise]; %newCov = F * prevCov * F' + obj.process_noise;
            end 

            % nextState = F_dot * prevDist.means;

            newMean = [prevDist.means; nextState];

            nextDist = prevDist;
            nextDist.means = newMean;
            nextDist.covariances = newCov;
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

            H_dot = kron([zeros(1,prevDist.window_size-1), 1],H);

            % est_meas = H_dot * prevDist.means;

            epsilon = meas - est_meas; 

            if ~isempty(meas_noise)
                S = H_dot * prevCov * H_dot' + meas_noise; 
            else
                S = H_dot * prevCov * H_dot' + obj.measurement_noise; 
            end

            K = prevCov * H_dot' * S^-1;

            nextDist = prevDist; 
            nextDist.means = prevDist.means + K * epsilon;
            nextDist.covariances = prevDist.covariances - K * H_dot * prevDist.covariances;
            
            likelihood = mvnpdf(meas, est_meas, S);
        
        end
    end 

end