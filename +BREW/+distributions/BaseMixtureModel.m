classdef BaseMixtureModel < handle & matlab.mixin.Copyable
    % Generic base class for mixture distribution models.
    % Defines required functions and base attributes for the mixture model.
    
    properties
        distributions = []  % Array of BaseSingleModel objects
        weights = []        % Array of weights 
    end
    
    methods
        function obj = BaseMixtureModel(distributions, weights)
            if nargin > 0 && ~isempty(distributions)
                obj.distributions = distributions;
            end
            if nargin > 1 && ~isempty(weights)
                obj.weights = weights;
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
        
        function obj = removeComponents(obj, indices)
            % Remove component distributions from the mixture by index 
            if isempty(indices)
                return;
            end
        
            % Ensure indices are within bounds
            numComponents = numel(obj.distributions);
            if any(indices < 1) || any(indices > numComponents)
                error('removeComponents:IndexOutOfBounds', ...
                      'One or more indices are out of range.');
            end
        
            % Convert to logical mask of components to keep
            mask = true(1, numComponents);
            mask(indices) = false;
        
            % Apply mask
            obj.distributions = obj.distributions(mask);
            obj.weights = obj.weights(mask);
        end
        
        function n = length(obj)
            n = numel(obj.distributions);
        end

        function new_mix = extract_mix(obj,threshold) 
            new_mix = obj;
            idx = obj.weights >= threshold; 

            new_mix.distributions = obj.distributions(idx);
            new_mix.weights = obj.weights(idx);
        end

        function obj = addComponents(obj, mix)
            % Add new components to the mixture
            % Each argument is a cell array (except new_weights, which is numeric)
            for i = 1:length(mix)
                obj.distributions(end+1) = mix.distributions(i);
                obj.weights(end+1) = mix.weights(i);
            end
        end

        function obj = merge(obj,threshold)
            % Placeholder for merging components in mixture
            % This function should be overwritten
        end
    end

end 