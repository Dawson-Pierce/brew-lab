clear; clc; close all

%% Target setup

means = {[0; -20; 1; 1/2], [0; 20; 1; -1/2], [0; 0; 1; 0]};
covariances = repmat({zeros(4)},1,3); 
idx = {1, 1, 5};
weights = [1, 1, 1];

truth = BREW.distributions.TrajectoryGaussianMixture( ...
    'idx',idx, ...
    'means',means, ...
    'covariances',covariances, ...
    'weights',weights); 


%% birth model setup

birth = BREW.distributions.TrajectoryGaussianMixture( ...
    'idx',{1}, ...
    'means',{[0; 0; 1; 0]}, ...
    'covariances',{diag([1, 50, 1.5, 1])}, ...
    'weights',[0.1]);

%% inner filter setup

dyn = BREW.dynamics.Integrator_2D();

ekf = BREW.filters.TrajectoryGaussianEKF( ...
    'dyn_obj',dyn, ...
    'process_noise',diag([0.1 0.1 0.001 0.001]), ...
    'H',[1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 0.05 * [1 0; 0 1], ...
    'L',20);

dt = 1;

t = 0:dt:80;

%% PHD setup

phd = BREW.multi_target.PHD('filter',ekf, 'birth_model', birth,...
    'prob_detection', 0.8, 'prob_survive', 0.98, 'max_terms',25, ...
    'extract_threshold',0.5, 'merge_threshold',4, ...
    'clutter_rate',0.05,'clutter_density',0.01);

%% Running the loop

meas_hst = [];

meas_cov = diag([0.05, 0.05]);

max_meas = 60;

x_bound = [-20 120];
y_bound = [-70 70];

for k = 1:length(t)
    
    for ii = 1:length(truth)
        if truth.distributions{ii}.init_idx <= k
            truth.distributions{ii} = ekf.predict(dt,truth.distributions{ii}); 
        end
    end


    num_noise_meas = round(max_meas * rand(1));
    x_noise = rand(1,num_noise_meas) * (x_bound(2) - x_bound(1)) + x_bound(1);
    y_noise = rand(1,num_noise_meas) * (y_bound(2) - y_bound(1)) + y_bound(1);

    noisy_meas = [x_noise; y_noise];

    meas = truth.sample_measurements([1,2], k, meas_cov);

    meas = [meas, noisy_meas];

    meas_hst = [meas_hst, meas];

    phd.predict(dt,{}); 

    phd.correct(dt, meas); 

    est_mix = phd.cleanup();

    % Plotting 
    scatter(meas(1,:),meas(2,:), 8,'w*'); grid on; hold on

    est_mix.plot_distributions([1 2],'LineWidth',2,'window_color','r','window_width',2.5); hold off

    xlim(x_bound)
    ylim(y_bound)

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