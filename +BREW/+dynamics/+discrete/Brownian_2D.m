classdef Brownian_2D < BREW.dynamics.DynamicsBase
    % Defines Brownian motion in one dimension, aka random walk on position

    properties (Constant)
        stateNames = {'x','y'}
    end

    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt);
            nextState = F*state + G*u;
        end

        function stateMat = getStateMat(obj, timestep, dt, varargin)
            stateMat = eye(2);
        end

        function inputMat = getInputMat(obj, timestep, dt, varargin)
            inputMat = eye(2);
        end
    end
end 