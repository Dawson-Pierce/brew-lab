classdef Integrator_2D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','vx','vy'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt,state);
            nextState = F*state + G*u;
        end
        function stateMat = getStateMat(obj, timestep, dt, state, varargin)
            stateMat = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
        end
        function inputMat = getInputMat(obj, timestep, dt, state, varargin)
            inputMat = [dt 0; 0 dt; 1 0; 0 1];
        end
    end
end 