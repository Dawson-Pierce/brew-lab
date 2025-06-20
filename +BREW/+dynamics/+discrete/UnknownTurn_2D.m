classdef UnknownTurn_2D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','vx','vy','omega'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt,state);
            nextState = F*state + G*u;
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            x = state(1); y = state(2); vx = state(3); vy = state(4); omega = state(5);
            if abs(omega) < 1e-6
                F = eye(5);
                F(1,3) = dt; F(2,4) = dt;
            else
                F = eye(5);
                F(1,3) = (sin(omega*dt)/omega);
                F(1,4) = -(1-cos(omega*dt))/omega;
                F(2,3) = (1-cos(omega*dt))/omega;
                F(2,4) = (sin(omega*dt)/omega);
                F(3,3) = cos(omega*dt);
                F(3,4) = -sin(omega*dt);
                F(4,3) = sin(omega*dt);
                F(4,4) = cos(omega*dt);
            end
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = eye(5);
        end
        function new_extent = propagate_extent(obj, dt, state, extent)
            % Propagate the extent by rotating it according to the angular rate in the state
            % extent: 2x2 matrix (e.g., for ellipse)
            % state: [x, y, vx, vy, omega]
            theta = state(5) * dt;
            R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
            new_extent = R * extent * R';
        end
        function rotated_extent = rotate_extent(obj, dt, state, extent)
            % Rotate the extent by the angle implied by the angular rate and dt
            theta = state(5) * dt;
            R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
            rotated_extent = R * extent * R';
        end
    end
end 