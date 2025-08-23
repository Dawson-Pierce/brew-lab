clear; clc; close all

%% Target setup

means = {[0; -20; 1; 1/2], [0; 20; 1; -1/2], [0; 0; 3/2; 0]};
covariances = repmat({zeros(4)},1,3);  
weights = [1, 1, 1];

truth = BREW.distributions.GaussianMixture( ... 
    'means',means, ...
    'covariances',covariances, ...
    'weights',weights); 


%% birth model setup

birth = BREW.distributions.GaussianMixture( ... 
    'means',{[20; 0; 1; 0]}, ...
    'covariances',{diag([10, 5, 1, 1])}, ...
    'weights',[1]);

%% inner filter setup

dyn = BREW.dynamics.Integrator_2D();

ekf = BREW.filters.EKF( ...
    'dyn_obj',dyn, ...
    'process_noise',diag([0.1 0.1 0.1 0.05]), ...
    'H',[1 0 0 0; 0 1 0 0], ...
    'measurement_noise', 0.1 * diag([1; 1]));

dt = 1;

t = 0:dt:80;

%% PHD setup

phd = BREW.multi_target.PHD('filter',ekf, 'birth_model', birth,...
    'prob_detection', 0.9, 'prob_survive', 0.98, 'max_terms',50,'extract_threshold',0.5);

%% Running the loop

meas_hst = [];

meas_cov = diag([0.05, 0.05]);

for k = 1:length(t)
    
    for ii = 1:length(truth) 
        truth.distributions{ii}.mean = dyn.propagateState(dt,truth.distributions{ii}.mean);  
    end

    meas = truth.sample_measurements([1,2]);

    meas_hst = [meas_hst, meas];

    phd.predict(dt,{}); 

    % for kk = 1:length(phd.birth_model)
    %     phd.birth_model.distributions{kk}.init_idx = k; % need for trajectory filter
    % end

    phd.correct(dt, meas);

    est_mix = phd.cleanup()

    % Plotting
    scatter(meas_hst(1,:),meas_hst(2,:),'w*'); grid on; hold on 

    % for kk = 1:length(est_mix)
    %     mu = est_mix.distributions{kk}.mean(1:2);
    %     C = est_mix.distributions{kk}.covariance(1:2,1:2);
    %     scatter(mu(1),mu(2),50,'r*'); 
    % 
    %     % [Evec, Eval] = eig(C);
    %     % t = linspace(0, 2*pi, 100);
    %     % scale = 1;
    %     % a = scale * sqrt(Eval(1,1)); 
    %     % b = scale * sqrt(Eval(2,2));
    %     % ellipse = Evec*[a*cos(t); b*sin(t)] + mu; 
    %     % plot(ellipse(1,:), ellipse(2,:), 'r--');
    % end

    est_mix.plot_distributions(gca, [1 2]);

    hold off

    xlim([-20 120])
    ylim([-70 70])

    drawnow; 

    frame = getframe(gcf);              % capture figure
    im = frame2im(frame);               % convert to image
    [imind,cm] = rgb2ind(im,256);       % indexed color
    
    if k == 1
        imwrite(imind,cm,'gm.gif','gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(imind,cm,'gm.gif','gif','WriteMode','append','DelayTime',0.1);
    end
end