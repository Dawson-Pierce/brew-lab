classdef BaseSingleModel 
    % Generic base class for distribution models.
    % Defines required functions and base attributes for the distribution.
    
    properties 
        monteCarloSize = 1e4; % Number of samples for MC operations
    end
    
    methods
        function obj = BaseSingleModel(monteCarloSize) 
            if nargin > 0
                obj.monteCarloSize = monteCarloSize;
            end
        end
        
        function s = sample(obj, varargin)
            % Draw a sample from the distribution (to be implemented by child class)
            warning('sample not implemented by class %s', class(obj));
            s = NaN;
        end
        
        function p = pdf(obj, x)
            % Calculate the PDF value at the given point (to be implemented by child class)
            warning('pdf not implemented by class %s', class(obj));
            p = NaN;
        end
        
        function disp(obj)
            % Display method for the distribution
            fprintf('No information to show for this class '); 
        end
    end
end 