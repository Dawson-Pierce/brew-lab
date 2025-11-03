clear; clc; close all

%% Target setup

mean = [0; 0; 1; 1/2];
cov = diag([0.001, 0.001, 0.01, 0.01]); % since generating measurements from gaussian, this dictates measurement resolution

target = BREW.distributions.Gaussian(mean,cov); 


%% Filter setup

dyn = BREW.dynamics.Integrator_2D();

multi_mean = gpuArray(cat(3,[0; 0; 1; 1/2],[0; -20; 1; 1/2],[0; 20; 1; 1/2]));
multi_cov = gpuArray(cat(3,diag([0.001, 0.1, 0.01, 0.01]),diag([0.001, 0.1, 0.01, 0.01]),diag([0.001, 0.1, 0.01, 0.01])));

est = BREW.distributions.TrajectoryGaussianMixture('means',multi_mean,'covariances',multi_cov,'L_max',5); 

ekf = BREW.filters.TrajectoryGaussianEKF('dyn_obj',dyn,'process_noise',0.1 * eye(4),'H',[1 0 0 0; 0 1 0 0], 'measurement_noise', 1 * [1; 1]);

dt = 1;

t = 0:dt:100;


%% Running the loop

meas_hst = [];

q_hist = zeros(3,length(t));

for k = 1:length(t) 
    target.mean = dyn.propagateState(dt,target.mean);
    meas = target.sample_measurements([1 2]);

    meas_hst = [meas_hst, meas];

    est = ekf.predict(dt,est); % Predict the next state
    [est,q] = ekf.correct(dt, meas, est); % Update the estimate with the new measurements 

    q_hist(:,k) = q / sum(q);  

    cla
    
    % Plotting
    subplot(2,1,1)
    scatter(meas_hst(1,:),meas_hst(2,:),'w*'); grid on; 
    xlim([-5 105])
    ylim([-5 105])

    est.plot_distributions([1 2],'window_style','--');  

    subplot(2,1,2)
    plot(q_hist(:,1:k)'); grid on; hold off

    drawnow; 
end