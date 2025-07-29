classdef EKF < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class for gaussians

    % most of the stuff for dynamics model and measurement model is 
    % inherited, just update the predict and correct methodologies 

    % dynamics model also has capability to be an input into the filter so
    % we can change it mid tracking application

    methods  
        function nextDist = predict(obj, dt, prevDist, varargin)
            p = inputParser; 
            p.CaseSensitive = true; 
            addParameter(p,'u',0);
            addParameter(p,'dyn_obj',[]); % adds capability to change dynamics model
            addParameter(p,'process_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:});

            u = p.Results.u;

            dyn_obj = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise; 

            prevState = prevDist.mean;
            prevCov = prevDist.covariance; 

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

            nextDist = BREW.distributions.Gaussian(nextState, nextCov);
        
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

            prevState = prevDist.mean;
            prevCov = prevDist.covariance; 

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

            epsilon = meas - est_meas; 

            if ~isempty(meas_noise)
                S = H * prevCov * H' + meas_noise; 
            else
                S = H * prevCov * H' + obj.measurement_noise; 
            end

            K = prevCov * H' * S^-1;

            nextState = prevState + K * epsilon;
            nextCov = (eye(length(prevState)) - K * H) * prevCov; % prevCov - K * S * K'; 

            nextDist = BREW.distributions.Gaussian(nextState,nextCov); 
            
            likelihood = mvnpdf(meas, est_meas, S);
        
        end
    end 

end