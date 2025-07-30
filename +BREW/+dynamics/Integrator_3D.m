classdef Integrator_3D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz'}
    end
    methods
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;

            addParameter(p, 'u', 0); 

            parse(p, varargin{:});

            F = obj.getStateMat(dt,state);
            G = obj.getInputMat(dt,state);
            nextState = F*state + G*p.Results.u;
        end
        function stateMat = getStateMat(obj, dt, varargin)
            stateMat = [1 0 0 dt 0 0; 
                0 1 0 0 dt 0; 
                0 0 1 0 0 dt; 
                0 0 0 1 0 0; 
                0 0 0 0 1 0; 
                0 0 0 0 0 1];
        end
        function inputMat = getInputMat(obj, dt, state, varargin)
            inputMat = [dt 0 0; 0 dt 0; 0 0 dt; 
                1 0 0; 0 1 0; 0 0 1];
        end
    end
end 