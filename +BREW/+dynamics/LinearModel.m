classdef LinearModel < BREW.dynamics.DynamicsBase
    properties 
        stateNames = {'unknown'}
        F
        G
    end
    methods
        function obj = LinearModel(F,G)
            obj.F = F;
            obj.G = G;
        end
        function nextState = propagateState(obj, timestep, dt, state, u) 
            nextState = obj.F*state + obj.G*u;
        end
        function stateMat = getStateMat(obj, timestep, dt, varargin)
            stateMat = obj.F;
        end
        function inputMat = getInputMat(obj, timestep, dt, varargin)
            inputMat = obj.G;
        end
    end
end 