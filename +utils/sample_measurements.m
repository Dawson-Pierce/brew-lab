function measurements = sample_measurements(mix, xy_inds, meas_cov)
%SAMPLE_MEASUREMENTS Draw measurements from each mixture component.
%   meas = utils.sample_measurements(mix, xy_inds)
%   meas = utils.sample_measurements(mix, xy_inds, meas_cov)
%
%   mix      - BREW.models.Mixture
%   xy_inds  - indices of state dimensions (e.g., [1 2])
%   meas_cov - (optional) measurement covariance for Gaussian types

    if nargin < 2 || isempty(xy_inds)
        xy_inds = 1:min(3, numel(mix.components{1}.mean));
    end
    if nargin < 3, meas_cov = []; end

    all_meas = [];

    for i = 1:mix.length()
        comp = mix.components{i};

        % Get the current-time mean (trajectory types use mean_history)
        mu = comp.mean;
        if isprop(comp, 'mean_history') && ~isempty(comp.mean_history)
            mu = comp.mean_history(:, end);
        end
        center = mu(xy_inds);

        if isprop(comp, 'alpha') && ~isempty(comp.alpha)
            % GGIW: Poisson number of measurements from extent
            V_i = comp.V;
            v_i = comp.v;
            d = size(V_i, 1);
            lam = comp.alpha / comp.beta;
            N = poissrnd(lam);
            extent = V_i / (v_i - d - 1);
            if N > 0
                mi = mvnrnd(center(:)', extent, N)';
            else
                mi = zeros(length(xy_inds), 0);
            end
        else
            % Gaussian: one measurement per component
            if ~isempty(meas_cov)
                C = meas_cov;
            else
                C = comp.covariance(xy_inds, xy_inds);
            end
            mi = mvnrnd(center(:)', C, 1)';
        end

        all_meas = [all_meas, mi]; %#ok<AGROW>
    end

    measurements = all_meas;
end
