classdef TrajectoryGaussianEKF < BREW.filters.FiltersBase
    % Extended Kalman Filter / Kalman Filter class for gaussian
    % trajectories

    properties 
    end

    methods
        function obj = TrajectoryGaussianEKF(varargin)
            p = inputParser;
            p.CaseSensitive = true;
            p.KeepUnmatched = true; 

            addParameter(p,'L',50)

            parse(p,varargin{:})
            
            obj@BREW.filters.FiltersBase(p.Unmatched);

        end
        function dist = predict(obj, dt, dist, varargin) 
            p = inputParser;
            p.CaseSensitive = true;
            p.KeepUnmatched = true;
            addParameter(p,'dyn_obj',obj.dyn_obj_);
            addParameter(p,'process_noise',obj.process_noise);
            parse(p,varargin{:});
        
            dyn_obj   = p.Results.dyn_obj;
            proc_noise = p.Results.process_noise; 
            prevMean = dist.mean; 

            nx = size(prevMean,1); 
        
            L = dist.L;

            [uniqueL,~,ic] = unique(L); 

            nextState = dyn_obj.propagateState(dt, prevMean, p.Unmatched);
            dist.AppendMeanHistory(nextState);
            F_all     = dyn_obj.getStateMat(dt, prevMean);
        
            for j = 1:numel(uniqueL)

                curr_idx = ic == j; % should produce logical array we can use for each unique length
                L_temp = uniqueL(j);

                P = dist.GetCovs(curr_idx);
                means = dist.GetMeans(curr_idx);
                F = F_all(:,:,curr_idx); 

                F_dot = zeros(size(F,1), size(F,2)*L_temp, size(F,3), 'like', F);
                F_dot(:, end-size(F,1)+1:end, :) = F;
        
                PFt  = pagemtimes(P, permute(F_dot,[2 1 3])); 
                FPFt = pagemtimes(F_dot, PFt); 
        
                if L_temp >= dist.L_max 
                    newCovBlock = zeros(size(P), 'like', P); 
                    newCovBlock(1:end-nx, 1:end-nx, :) = P(nx+1:end, nx+1:end, :); 
                    newCovBlock(end-nx+1:end, end-nx+1:end, :) = FPFt + proc_noise;
                    PFt = PFt(1:end-nx,:,:);
                    newCovBlock(1:end-nx, end-nx+1:end, :) = PFt;
                    newCovBlock(end-nx+1:end, 1:end-nx, :) = permute(PFt, [2 1 3]);

                    newMeans = zeros(size(means), 'like', means); 
                    newMeans(1:end-nx,1,:) = means(nx+1:end,:);
                    newMeans(end-nx+1:end,1,:) = nextState(:,:,curr_idx);
                else
                    newCovBlock = zeros(size(P,1)+nx, size(P,2)+nx, size(P,3), 'like', P);
                    newCovBlock(1:size(P,1), 1:size(P,2), :) = P;
                    newCovBlock(1:size(P,1), size(P,2)+1:end, :) = PFt;
                    newCovBlock(size(P,1)+1:end, 1:size(P,2), :) = permute(PFt,[2 1 3]);
                    newCovBlock(size(P,1)+1:end, size(P,2)+1:end, :) = FPFt + proc_noise;

                    newMeans = zeros(size(means,1)+nx,1,size(means,3), 'like', means); 
                    newMeans(1:end-nx,1,:) = means;
                    newMeans(end-nx+1:end,1,:) = nextState(:,:,curr_idx);
                end
        
                dist.SetCovs(newCovBlock,curr_idx); 
                dist.SetMeans(newMeans,curr_idx);
                dist.L(curr_idx) = min(dist.L_max, dist.L(curr_idx) + 1); 
            end 
        end


        function [dist, likelihood] = correct(obj, dt, meas, dist, varargin)

            p = inputParser; 
            p.CaseSensitive = true;
            p.KeepUnmatched = true;
            addParameter(p,'H',[]); 
            addParameter(p,'h',[]); 
            addParameter(p,'meas_noise',obj.measurement_noise);
            addParameter(p,'gate',4); 
            parse(p, varargin{:});
        
            h_input = p.Results.h;
            H_input = p.Results.H;
            meas_noise = p.Results.meas_noise;
            gate = p.Results.gate;
        
            prevMean = dist.mean; 
            nx = size(prevMean,1);            
        
            if ~isempty(h_input)
                est_meas = h_input(prevMean);
            else
                est_meas = obj.estimate_measurement(prevMean);
            end
        
            if ~isempty(H_input)
                if isa(H_input,'function_handle')
                    H = H_input(prevMean);
                else
                    H = H_input;
                end
            else
                H = obj.getMeasurementMatrix(prevMean);
            end

            L = dist.L;
            uniqueL = unique(L);

            likelihood = nan(1,size(prevMean,3),'like',prevMean);
        
            for j = 1:numel(uniqueL)

                L_temp = uniqueL(j);
                curr_idx = L == L_temp; % should produce logical array we can use for each unique length 
        
                P = dist.GetCovs(curr_idx);
                means = dist.GetMeans(curr_idx);

                H_dot = zeros(size(H,1), size(H,2)*L_temp, size(means,3), 'like', means);
                H_dot(:, end-size(H,2)+1:end,:) = H(:,:,curr_idx); 

                PHt = pagemtimes(P, permute(H_dot,[2 1 3]));
                S   = pagemtimes(H_dot, PHt) + meas_noise;
                S_inv = pageinv(S);

                est_measurements = est_meas(:,:,curr_idx);

                eps = meas - est_measurements;

                K   = pagemtimes(PHt, S_inv);

                Xnew = means + pagemtimes(K, eps);

                KH = pagemtimes(K, H_dot);
                Pnew = P - pagemtimes(KH, P);

                dist.SetMeans(Xnew,curr_idx);
                dist.SetCovs(Pnew,curr_idx); 

                counter = 1;
                likelihood_temp = zeros(1,size(S,3));
                for k = 1:size(S,3)
                    likelihood_temp(counter) = mvnpdf(meas', est_measurements(:,:,k)', S(:,:,k));
                    counter = counter + 1;
                end

                likelihood(curr_idx) = likelihood_temp;
            end
        
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