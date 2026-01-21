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
            p.KeepUnmatched = true;
            addParameter(p, 'forgetting_factor', 1);
            addParameter(p, 'tau', 1); 
            addParameter(p,'dyn_obj',[]); % adds capability to change dynamics model
            addParameter(p,'process_noise',[]); 

            % Parse known arguments
            parse(p, varargin{:}); 

            dyn_obj = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise;
           
            prevAlpha = prevDist.alpha;
            prevBeta = prevDist.beta;
            prevState = prevDist.mean;
            prevCov = prevDist.covariance; 
            prevIWdof = prevDist.IWdof;
            prevIWshape = prevDist.IWshape; 

            if ~isempty(dyn_obj)
                nextState = dyn_obj.propagateState(dt, prevState, p.Unmatched);
                F =  dyn_obj.getStateMat(dt, prevState);
            elseif ~isempty(obj.dyn_obj_)
                nextState = obj.dyn_obj_.propagateState(dt, prevState, p.Unmatched);
                F =  obj.dyn_obj_.getStateMat(dt, prevState);
            end
            if ~isempty(proc_noise)
                nextCov = F * prevCov * F' + proc_noise;
            else
                nextCov = F * prevCov * F' + obj.process_noise;
            end

            nextAlpha = prevAlpha / p.Results.forgetting_factor; 
            nextBeta = prevBeta / p.Results.forgetting_factor;
            nextIWdof = 2*prevDist.d + 2 + exp(-dt / p.Results.tau) * (prevIWdof - 2*prevDist.d - 2) + 1e-3; 
            nextIWshape = (nextIWdof - 2*prevDist.d - 2) * ...
                (prevIWdof - 2*prevDist.d - 2)^-1 * ...
                obj.dyn_obj_.propagate_extent(dt,prevState, prevIWshape); % this last function computes M*V*M'

            % Create the next distribution for GGIW
            nextDist = prevDist;
            nextDist.alpha = nextAlpha;
            nextDist.beta = nextBeta;
            nextDist.mean = nextState;
            nextDist.covariance = nextCov;
            nextDist.IWdof = nextIWdof;
            nextDist.IWshape = nextIWshape;
            % nextDist = BREW.distributions.GGIW(nextAlpha, nextBeta, nextState, nextCov, nextIWdof, nextIWshape);
        
        end
        function [nextDist, likelihood] = correct(obj, dt, meas, prevDist, varargin) 

            p = inputParser; 
            p.CaseSensitive = true;
            addParameter(p,'H',[]); % could be constant, could be function handle ie H = @(x) ...
            addParameter(p,'h',[]); % expected to be function handle ie h = @(x) ...
            addParameter(p,'scaling_parameter',1)
            addParameter(p,'meas_noise',obj.measurement_noise); 
            addParameter(p,'basis_tracking',0)
            addParameter(p,'beta',0)

            % Parse known arguments
            parse(p, varargin{:});

            h_input = p.Results.h;
            H_input = p.Results.H;
            meas_noise = p.Results.meas_noise;
            rho = p.Results.scaling_parameter;

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

            sqrtXhat = obj.sqrtmDiag(X_hat);

            epsilon = mean_meas - est_meas;

            N = epsilon * epsilon';

            R_hat = rho*X_hat + meas_noise;

            sqrtRhat = obj.sqrtmDiag(R_hat);
            sqrtRhat = 0.5 * (sqrtRhat' + sqrtRhat);
 
            S = H * prevCov * H' + R_hat / W;

            % sqrtS = obj.sqrtmDiag(S); 

            K =  prevCov * H' / S; 

            sqrtXinvR = sqrtXhat/sqrtRhat; 
            sqrtXinvR = 0.5 * (sqrtXinvR' + sqrtXinvR);
            
            N_hat = sqrtXinvR * N / (sqrtRhat') * sqrtXhat';
            % Z_hat = sqrtS \ scatter_meas / (sqrtS'); 

            nextAlpha = prevAlpha + W;
            nextBeta = prevBeta + 1; 
            nextState = prevState + K * epsilon;
            nextCov = prevCov - K * H * prevCov; 
            nextCov = (nextCov + nextCov')/2; 
            nextIWdof = prevIWdof + W;
            nextIWshape = prevIWshape + N_hat + scatter_meas;

            nextIWshape = 0.5 * (nextIWshape' + nextIWshape);

            nextDist = prevDist.copy();
            nextDist.alpha = nextAlpha;
            nextDist.beta = nextBeta;
            nextDist.mean = nextState;
            nextDist.covariance = nextCov;
            nextDist.IWdof = nextIWdof;
            nextDist.IWshape = nextIWshape;
            % nextDist = BREW.distributions.GGIW(nextAlpha,nextBeta,nextState,nextCov,nextIWdof,nextIWshape); 
            
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

            if p.Results.basis_tracking 
                % Currently only works for 2D 
                [u, s, ~] = svd(nextDist.IWshape);

                if isempty(prevDist.basis) 
                    nextDist.basis = u;
                    nextDist.eigs = s;
                else
                    prev_u = prevDist.basis; 
                    beta = p.Results.beta;

                    A = u'*prev_u; 

                    [~, I] = max(abs(A), [], 2);
                    u = u(:, I);
                    A = A(:, I);
                
                    % fix sign flips
                    for k = 1:size(u,2)
                        if A(k,k) < 0
                            u(:,k) = -u(:,k);
                        end
                    end

                    u = (1-beta) * u + beta*prev_u;

                    nextDist.basis = u;
                    nextDist.eigs = s;

                    nextDist.IWshape = u * s * u';
                end
            end
        
        end

        function val = gate_meas(obj, pred_dist, z, gamma)
            state = pred_dist.mean;
            P = pred_dist.covariance;
            
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

    methods (Access = private)
        function sqrtP = sqrtmDiag(obj,P)
            [Q, D] = eig((P+P')/2);
            
            d = diag(D);
            d = max(real(d),eps(class(P)));
            D = diag(d);
            
            sqrtP = real(Q*sqrt(D)*Q');
            end
    end

end