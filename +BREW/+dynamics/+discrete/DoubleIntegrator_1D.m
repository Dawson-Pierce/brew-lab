classdef DoubleIntegrator_1D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','vx','ax'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt,state);
            nextState = F*state + G*u;
        end
        function stateMat = getStateMat(obj, timestep, dt, varargin)
            stateMat = [1 dt 0.5*dt^2; 0 1 dt; 0 0 1];
        end
        function inputMat = getInputMat(obj, timestep, dt, state, varargin)
            inputMat = [0.5*dt^2; dt; 1];
        end
    end
end 