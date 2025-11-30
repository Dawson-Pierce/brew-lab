classdef BaseSingleModel < matlab.mixin.Copyable
    % Generic base class for distribution models.
    % Defines required functions and base attributes for the distribution.
    
    properties 

    end
    
    methods
        function obj = BaseSingleModel() 
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
        
        % function disp(obj)
        %     % Display method for the distribution
        %     fprintf('No information to show for this class \n'); 
        % end
    end
end 