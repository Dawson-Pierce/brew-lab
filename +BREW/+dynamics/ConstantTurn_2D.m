classdef ConstantTurn_2D < BREW.dynamics.DynamicsBase
    % inputs defined as velocity and omega

    properties (Constant)
        stateNames = {'x','y','v','theta','omega'}
    end
    methods
        function nextState = propagateState(obj, dt, state, varargin)
            p = inputParser;
            p.CaseSensitive = true;

            addParameter(p, 'u', zeros(2,1)); 

            parse(p, varargin{:});

            % F = obj.getStateMat(dt,state);
            G = obj.getInputMat(dt,state);

            nextState = [state(1) + dt*state(3)*cos(state(4));
                state(2) + dt*state(3)*sin(state(4));
                state(3); 
                state(4) + dt * state(5); 
                state(5)] + G * p.Results.u;

            % nextState = F*state + G*p.Results.u;
        end
        function F = getStateMat(obj, dt, state, varargin)
            x  = state(1);
            y  = state(2);
            v  = state(3);
            th = state(4);
            w  = state(5);
        
            F = eye(5);
        
            F(1,3) = dt*cos(th);
            F(1,4) = -dt*v*sin(th);
        
            F(2,3) = dt*sin(th);
            F(2,4) =  dt*v*cos(th);

            F(4,5) = dt;
        end
        function G = getInputMat(obj, dt, state, varargin)
            th = state(4);

            G = zeros(5,2);
            G(1,1) = dt*cos(th);
            G(2,1) = dt*sin(th);
            G(3,1) = 1; 
            G(5,2) = 1;
        end

        function new_extent = propagate_extent(obj, dt, state, extent, varargin)
            theta = state(4);
            omega = state(5);
            
            dtheta = omega * dt;
        
            c = cos(dtheta);
            s = sin(dtheta);
            R = [c -s; s c];
        
            new_extent = R * extent * R';
        end
    end
end 