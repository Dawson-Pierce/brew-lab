classdef DoubleIntegrator_3D < handle
    properties (SetAccess = private)
        handle_ uint64
    end
    properties
        M = []
    end
    methods
        function obj = DoubleIntegrator_3D()
            obj.handle_ = brew_mex('create_dynamics', 'DoubleIntegrator3D');
        end
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'u', zeros(3,1));
            parse(p, varargin{:});
            F = obj.getStateMat(dt, state);
            G = obj.getInputMat(dt, state);
            nextState = F*state + G*p.Results.u;
        end
        function F = getStateMat(~, dt, varargin)
            F = [1 0 0 dt 0 0 0.5*dt^2 0 0;
                 0 1 0 0 dt 0 0 0.5*dt^2 0;
                 0 0 1 0 0 dt 0 0 0.5*dt^2;
                 0 0 0 1 0 0 dt 0 0;
                 0 0 0 0 1 0 0 dt 0;
                 0 0 0 0 0 1 0 0 dt;
                 0 0 0 0 0 0 1 0 0;
                 0 0 0 0 0 0 0 1 0;
                 0 0 0 0 0 0 0 0 1];
        end
        function G = getInputMat(~, dt, varargin)
            G = [0.5*dt^2 0 0; 0 0.5*dt^2 0; 0 0 0.5*dt^2;
                 dt 0 0; 0 dt 0; 0 0 dt;
                 1 0 0; 0 1 0; 0 0 1];
        end
        function new_extent = propagate_extent(obj, dt, state, extent, varargin)
            if ~isempty(obj.M)
                if isa(obj.M,'function_handle')
                    new_extent = obj.M(state,dt) * extent * obj.M(state,dt)';
                else
                    new_extent = obj.M * extent * obj.M';
                end
            else
                new_extent = extent;
            end
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
