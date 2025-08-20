clear; clc; close all

%% Target setup

means = {[0; -20; 1; 1/2], [0; 20; 1; -1/2], [0; 0; 2; 0]};
covariances = repmat({zeros(4)},1,3); 
idx = {1, 30, 65};
weights = [1, 1, 1];

truth = BREW.distributions.TrajectoryGaussianMixture( ...
    'idx',idx, ...
    'means',means, ...
    'covariances',covariances, ...
    'weights',weights); 


%% inner filter setup

birth = BREW.distributions.TrajectoryGaussianMixture( ...
    'idx',{1}, ...
    'means',{[0; 0; 1; 0]}, ...
    'covariances',{diag([10, 10, 0.1, 5])}, ...
    'weights',[1]);

dyn = BREW.dynamics.Integrator_2D();

ekf = BREW.filters.TrajectoryGaussianEKF( ...
    'dyn_obj',dyn, ...
    'process_noise',0.1 * eye(4), ...
    'H',[1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 10 * [1; 1]);

dt = 1;

t = 0:dt:100;

%% PHD setup

phd = BREW.multi_target.PHD('filter',ekf, 'birth_model', birth,...
    'prob_detection', 0.9, 'prob_survive', 0.8, 'max_terms',50);

%% Running the loop

meas_hst = [];

meas_cov = diag([0.25, 0.25]);

for k = 1:length(t)
    
    for ii = 1:length(truth)
        if truth.distributions{ii}.init_idx <= k
            truth.distributions{ii} = ekf.predict(dt,truth.distributions{ii}); 
        end
    end

    meas = truth.sample_measurements([1,2], k, meas_cov);

    meas_hst = [meas_hst, meas];

    phd.predict(dt,{}); 

    for kk = 1:length(phd.birth_model)
        phd.birth_model.distributions{kk}.init_idx = k; % need for trajectory filter
    end

    phd.correct(dt, meas);  

    est_mix = phd.cleanup();

    % Plotting
    scatter(meas_hst(1,:),meas_hst(2,:),'w*'); grid on; hold on 

    est_mix.plot([1 2],'c','r-','lineWidth',2); hold off

    xlim([-20 120])
    ylim([-70 70])

    drawnow; 

    frame = getframe(gcf);              % capture figure
    im = frame2im(frame);               % convert to image
    [imind,cm] = rgb2ind(im,256);       % indexed color
    
    if k == 1
        imwrite(imind,cm,'gm-trajectory.gif','gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(imind,cm,'gm-trajectory.gif','gif','WriteMode','append','DelayTime',0.1);
    end
end