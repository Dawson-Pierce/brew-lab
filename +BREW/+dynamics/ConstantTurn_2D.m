classdef ConstantTurn_2D < handle
    properties (SetAccess = private)
        handle_ uint64
    end
    properties
        M = []
    end
    methods
        function obj = ConstantTurn_2D()
            obj.handle_ = brew_mex('create_dynamics', 'ConstantTurn2D');
        end
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'u', zeros(2,1));
            parse(p, varargin{:});
            G = obj.getInputMat(dt, state);
            nextState = [state(1) + dt*state(3)*cos(state(4));
                         state(2) + dt*state(3)*sin(state(4));
                         state(3);
                         state(4) + dt * state(5);
                         state(5)] + G * p.Results.u;
        end
        function F = getStateMat(~, dt, state, varargin)
            v  = state(3);
            th = state(4);
            F = eye(5);
            F(1,3) = dt*cos(th);
            F(1,4) = -dt*v*sin(th);
            F(2,3) = dt*sin(th);
            F(2,4) =  dt*v*cos(th);
            F(4,5) = dt;
        end
        function G = getInputMat(~, dt, state, varargin)
            th = state(4);
            G = zeros(5,2);
            G(1,1) = dt*cos(th);
            G(2,1) = dt*sin(th);
            G(3,1) = 1;
            G(5,2) = 1;
        end
        function new_extent = propagate_extent(~, dt, state, extent, varargin)
            omega = state(5);
            dtheta = omega * dt;
            c = cos(dtheta);
            s = sin(dtheta);
            R = [c -s; s c];
            new_extent = R * extent * R';
        end
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
