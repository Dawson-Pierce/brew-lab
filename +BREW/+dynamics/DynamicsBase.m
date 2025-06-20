classdef (Abstract) DynamicsBase < handle
    % Defines common attributes for all dynamics models.
    % This is an abstract base class for linear and nonlinear dynamics.
    %
    % Properties:
    %   stateNames: cell array of state names
    %
    % Methods (Abstract):
    %   propagateState: Propagate the state forward in time
    %   getStateMat: Get the state matrix
    %   getInputMat: Get the input matrix

    properties (Abstract, Constant)
        stateNames
    end

    methods (Abstract)
        nextState = propagateState(obj, timestep, dt, state, u)
        stateMat = getStateMat(obj, timestep, dt, varargin)
        inputMat = getInputMat(obj, timestep, dt, varargin)
    end

    methods
        function new_extent = propagate_extent(obj, dt, state, extent)
            new_extent = extent; 
        end
        function rotated_extent = rotate_extent(obj, dt, state, extent)
            rotated_extent = extent; 
        end
    end
end 