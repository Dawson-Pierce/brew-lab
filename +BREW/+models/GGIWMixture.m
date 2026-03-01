classdef GGIWMixture < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "GGIW"
    end
    methods
        function obj = GGIWMixture(varargin)
            p = inputParser;
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'alphas', {});
            addParameter(p, 'betas', {});
            addParameter(p, 'vs', {});
            addParameter(p, 'Vs', {});
            addParameter(p, 'weights', []);
            parse(p, varargin{:});
            obj.handle_ = brew_mex('create_mixture', 'GGIW', ...
                p.Results.means, p.Results.covariances, ...
                p.Results.alphas, p.Results.betas, ...
                p.Results.vs, p.Results.Vs, ...
                p.Results.weights);
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
