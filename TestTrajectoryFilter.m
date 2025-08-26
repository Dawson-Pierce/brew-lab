clear; clc; close all

%% Target setup

mean = [0; 0; 1; 1/2];
cov = diag([0.001, 0.001, 0.01, 0.01]); % since generating measurements from gaussian, this dictates measurement resolution

target = BREW.distributions.Gaussian(mean,cov); 


%% Filter setup

dyn = BREW.dynamics.Integrator_2D();

est = BREW.distributions.TrajectoryGaussian(1,target.sample(),cov); % Randomize mean initialization

ekf = BREW.filters.TrajectoryGaussianEKF('dyn_obj',dyn,'process_noise',0.1 * eye(4),'H',[1 0 0 0; 0 1 0 0], 'measurement_noise', 5 * [1; 1]);

dt = 1;

t = 0:dt:100;


%% Running the loop

meas_hst = [];

for k = 1:length(t)
    target.mean = dyn.propagateState(dt,target.mean);
    meas = target.sample_measurements([1 2]);

    meas_hst = [meas_hst, meas];

    est = ekf.predict(dt,est); % Predict the next state
    [est,q] = ekf.correct(dt, meas, est); % Update the estimate with the new measurements
    
    % Plotting
    scatter(meas_hst(1,:),meas_hst(2,:),'w*'); grid on; hold on
    xlim([-5 105])
    ylim([-5 105])

    est.plot([1 2],'c','r','lineWidth',2); hold off

    drawnow; 
end