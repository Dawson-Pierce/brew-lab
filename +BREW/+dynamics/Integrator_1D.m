classdef Integrator_1D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','vx'}
    end
    methods
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;

            addParameter(p, 'u', zeros(1,1,size(state,3))); 

            parse(p, varargin{:});

            F = obj.getStateMat(dt, state); 
            G = obj.getInputMat(dt, state); 
            nextState = pagemtimes(F, state) + pagemtimes(G, p.Results.u);
        end
        function stateMat = getStateMat(obj, dt, state, varargin)
            F = [1 dt; 0 1];
            stateMat = repmat(F,1,1,size(state,3));
        end
        function inputMat = getInputMat(obj, dt, state, varargin)
            G = [dt; 1];
            inputMat = repmat(G,1,1,size(state,3));
        end
    end
end 