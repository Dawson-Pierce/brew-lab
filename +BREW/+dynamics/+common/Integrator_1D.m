classdef Integrator_1D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','vx'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt);
            nextState = F*state + G*u;
        end
        function stateMat = getStateMat(obj, timestep, dt, varargin)
            stateMat = [1 dt; 0 1];
        end
        function inputMat = getInputMat(obj, timestep, dt, varargin)
            inputMat = [dt; 1];
        end
    end
end 