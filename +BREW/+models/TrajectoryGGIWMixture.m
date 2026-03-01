classdef TrajectoryGGIWMixture < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "TrajectoryGGIW"
    end
    methods
        function obj = TrajectoryGGIWMixture(varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'idx', {});
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'alphas', {});
            addParameter(p, 'betas', {});
            addParameter(p, 'vs', {});
            addParameter(p, 'Vs', {});
            addParameter(p, 'weights', []);
            parse(p, varargin{:});
            % MEX expects alphas/betas/vs as double arrays, not cell arrays
            a = p.Results.alphas; if iscell(a), a = cell2mat(a); end
            b = p.Results.betas;  if iscell(b), b = cell2mat(b); end
            v = p.Results.vs;     if iscell(v), v = cell2mat(v); end
            obj.handle_ = brew_mex('create_mixture', 'TrajectoryGGIW', ...
                p.Results.idx, p.Results.means, p.Results.covariances, ...
                a, b, v, p.Results.Vs, ...
                p.Results.weights);
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
