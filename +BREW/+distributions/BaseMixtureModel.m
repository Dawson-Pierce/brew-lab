classdef BaseMixtureModel < handle
    % Generic base class for mixture distribution models.
    % Defines required functions and base attributes for the mixture model.
    
    properties
        distributions = {}  % Cell array of BaseSingleModel objects
        weights = []        % Array of weights
        monteCarloSize = 1e4;
    end
    
    methods
        function obj = BaseMixtureModel(distributions, weights, monteCarloSize)
            if nargin > 0 && ~isempty(distributions)
                obj.distributions = distributions;
            end
            if nargin > 1 && ~isempty(weights)
                obj.weights = weights;
            end
            if nargin > 2
                obj.monteCarloSize = monteCarloSize;
            end
        end
        
        function s = sample(obj, varargin)
            % Draw a sample from the mixture model (to be implemented by child class)
            warning('sample not implemented by class %s', class(obj));
            s = NaN;
        end
        
        function p = pdf(obj, x)
            % Calculate the PDF value at the given point (to be implemented by child class)
            warning('pdf not implemented by class %s', class(obj));
            p = NaN;
        end
        
        function addComponents(obj, varargin)
            % Add component distributions to the mixture (to be implemented by child class)
            warning('addComponents not implemented by class %s', class(obj));
        end
        
        function removeComponents(obj, indices)
            % Remove component distributions from the mixture by index
            if ~iscell(indices)
                indices = num2cell(indices);
            end
            for i = sort([indices{:}], 'descend')
                obj.distributions(i) = [];
                obj.weights(i) = [];
            end
        end
        
        function n = numComponents(obj)
            n = numel(obj.distributions);
        end
    end
end 