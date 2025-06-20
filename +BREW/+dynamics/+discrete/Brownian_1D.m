classdef Brownian_1D < BREW.dynamics.DynamicsBase
    % Defines Brownian motion in one dimension, aka random walk on position

    properties (Constant)
        stateNames = {'x'}
    end

    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt);
            nextState = F*state + G*u;
        end

        function stateMat = getStateMat(obj, timestep, dt, varargin)
            stateMat = 1;
        end

        function inputMat = getInputMat(obj, timestep, dt, varargin)
            inputMat = 1;
        end
    end
end 