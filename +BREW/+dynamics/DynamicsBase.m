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

    properties 
        M = []
    end

    methods (Abstract)
        nextState = propagateState(obj, dt, state, varargin)
        stateMat = getStateMat(obj, dt, varargin)
        inputMat = getInputMat(obj, dt, varargin)
    end

    methods
        function setRotationModel(obj,M)
            obj.M = M;
        end
        function new_extent = propagate_extent(obj, state, extent, varargin)
            if ~isempty(obj.M)
                if isa(obj.M,'function_handle')
                    new_extent = obj.M(state) * extent * obj.M(state)';
                else
                    new_extent = obj.M * extent * obj.M';
                end
            else
                new_extent = extent;
            end
        end 
    end
end 