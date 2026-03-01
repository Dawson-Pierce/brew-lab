classdef Integrator_1D < handle
    properties (SetAccess = private)
        handle_ uint64
    end
    properties
        M = []
    end
    methods
        function obj = Integrator_1D()
            obj.handle_ = brew_mex('create_dynamics', 'Integrator1D');
        end
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'u', 0);
            parse(p, varargin{:});
            F = obj.getStateMat(dt, state);
            G = obj.getInputMat(dt, state);
            nextState = F*state + G*p.Results.u;
        end
        function F = getStateMat(~, dt, varargin)
            F = [1 dt; 0 0];
        end
        function G = getInputMat(~, dt, varargin)
            G = [dt; 1];
        end
        function new_extent = propagate_extent(obj, ~, ~, extent, varargin)
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
