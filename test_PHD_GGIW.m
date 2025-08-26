%% Testing PHD script

clear; clc; close all

alphas = {50,50,50};
betas = {1,1,1}; 
means = {[0; 10; 10; -0.75; -0.75; 0],[5; 20; 10; 0; -1; 0],[0; 12.5; 0; -0.5; 0; 1]}; 
covariances = {eye(6), eye(6), eye(6)}; 
IWdofs = {10, 10, 10}; 
IWshapes = {[5 0 0; 0 5 0; 0 0 2],[5 0 0; 0 5 0; 0 0 2],[5 0 0; 0 5 0; 0 0 2]}; 
weights = [1,1,1];

truth = BREW.distributions.GGIWMixture( ...
    'alphas', alphas, ...
    'betas', betas, ...
    'means', means, ...
    'covariances', covariances, ...
    'IWdofs', IWdofs, ...
    'IWshapes', IWshapes, ...
    'weights', weights);

% define truth target motion
target_motion = BREW.dynamics.Integrator_3D();

% generate measurements 

dt = 0.2;
tf = 10; 

t = 0:dt:tf;

measurements = {};

% initialize filter(s)
H = [eye(3) zeros(3, length(means{1})-3)];
R = 0.2 * eye(3); % Measurement noise
process_noise = 0.01 * eye(length(means{1})); 
inner_filter = BREW.filters.GGIWEKF('dyn_obj',target_motion, ...
    'H',H,'process_noise',process_noise,'measurement_noise', R);

alpha = {10};
beta = {1}; 
mean = {[0; 15; 10; 0; 0; 0]}; 
covariance = {diag([5,5,5,2,2,2])}; 
IWdof = {10}; 
IWshape = {[1 0 0; 0 1 0; 0 0 1]}; 
weight = [1];

birth_model = BREW.distributions.GGIWMixture( ...
    'alphas', alpha, ...
    'betas', beta, ...
    'means', mean, ...
    'covariances', covariance, ...
    'IWdofs', IWdof, ...
    'IWshapes', IWshape, ...
    'weights', weight);

phd = BREW.multi_target.ParallelPHD('filter',inner_filter, 'birth_model', birth_model,...
    'prob_detection', 0.8, 'prob_survive', 0.8, 'max_terms',50, ...
    'cluster_obj',BREW.clustering.DBSCAN_obj(3,5),'extract_threshold',0.5);

f = figure; 
ax = axes;
axis equal;
grid on;
hold(ax, 'on');
xlabel('X'); ylabel('Y'); zlabel('Z'); 
view(3); 

axis(ax, [-10 10 0 20 0 20]);

gifFilename = 'GGIW_PHD_3D.gif';

for k = 1:length(t)
    cla(ax);
    timestep = t(k);
    for kk = 1:length(truth.distributions) 
        truth.means{kk} = target_motion.propagateState(dt,truth.means{kk});
    end
    measurements{k} = truth.sample_measurements(); % Generate measurements from the truth distribution

    scatter3(measurements{k}(1,:),measurements{k}(2,:),measurements{k}(3,:),'w*','SizeData',0.5)

    % PHD stuff
    phd.predict(dt,{}); % Predict the state of the PHD filter 

    phd.correct(dt,measurements{k}); % Update the PHD filter with the new measurements

    est_mix = phd.cleanup();

    fprintf("timestep: %f \n",t(k)) 

    est_mix.plot_distributions(ax,[1,2,3],0.95);

    drawnow;

    % capture the frame as an RGB image
    frame = getframe(f);
    img   = frame2im(frame);

    % convert to an indexed image with a fixed 256‐color map
    [A, map] = rgb2ind(img, 256);

    % write to GIF: first frame creates the file, subsequent frames append
    if k == 1
        imwrite(A, map, gifFilename, 'gif', ...
                'LoopCount', Inf, ...      % make it loop forever
                'DelayTime', 0.1);         % seconds between frames
    else
        imwrite(A, map, gifFilename, 'gif', ...
                'WriteMode', 'append', ...
                'DelayTime', 0.1);
    end

    pause(0.3)
end 