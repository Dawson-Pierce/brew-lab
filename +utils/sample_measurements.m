function measurements = sample_measurements(mix, xy_inds, meas_cov)
%SAMPLE_MEASUREMENTS Draw measurements from each mixture component.

    if nargin < 2 || isempty(xy_inds)
        xy_inds = 1:min(3, numel(mix.components{1}.mean));
    end
    if nargin < 3, meas_cov = []; end

    all_meas = [];

    for i = 1:mix.length()
        comp = mix.components{i};

        mu = comp.mean;
        if isprop(comp, 'mean_history') && ~isempty(comp.mean_history)
            mu = comp.mean_history(:, end);
        end
        center = mu(xy_inds);

        if isprop(comp, 'alpha') && ~isempty(comp.alpha)
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
            if ~isempty(meas_cov)
                C = meas_cov;
            else
                C = comp.covariance(xy_inds, xy_inds);
            end
            mi = mvnrnd(center(:)', C, 1)';
        end

        all_meas = [all_meas, mi];
    end

    measurements = all_meas;
end
