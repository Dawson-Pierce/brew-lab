classdef GaussianMixture < handle
    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "Gaussian"
    end
    methods
        function obj = GaussianMixture(varargin)
            p = inputParser;
            addParameter(p, 'means', {});
            addParameter(p, 'covariances', {});
            addParameter(p, 'weights', []);
            parse(p, varargin{:});
            obj.handle_ = brew_mex('create_mixture', 'Gaussian', ...
                p.Results.means, p.Results.covariances, p.Results.weights);
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
