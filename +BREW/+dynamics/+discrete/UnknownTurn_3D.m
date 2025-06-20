classdef UnknownTurn_3D < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz','omega'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            F = obj.getStateMat(timestep,dt,state);
            G = obj.getInputMat(timestep,dt,state);
            nextState = F*state + G*u;
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            x = state(1); y = state(2); z = state(3); vx = state(4); vy = state(5); vz = state(6); omega = state(7);
            if abs(omega) < 1e-6
                F = eye(7);
                F(1,4) = dt; F(2,5) = dt; F(3,6) = dt;
            else
                F = eye(7);
                F(1,4) = (sin(omega*dt)/omega);
                F(1,5) = -(1-cos(omega*dt))/omega;
                F(2,4) = (1-cos(omega*dt))/omega;
                F(2,5) = (sin(omega*dt)/omega);
                F(3,6) = dt;
                F(4,4) = cos(omega*dt);
                F(4,5) = -sin(omega*dt);
                F(5,4) = sin(omega*dt);
                F(5,5) = cos(omega*dt);
            end
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = eye(7);
        end
        function new_extent = propagate_extent(obj, dt, state, extent)
            % Propagate the extent by rotating it about the z-axis according to the angular rate in the state
            % extent: 3x3 matrix (e.g., for ellipsoid)
            % state: [x, y, z, vx, vy, vz, omega]
            theta = state(7) * dt;
            Rz = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
            new_extent = Rz * extent * Rz';
        end
        function rotated_extent = rotate_extent(obj, dt, state, extent)
            % Rotate the extent by the angle implied by the angular rate and dt about the z-axis
            theta = state(7) * dt;
            Rz = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
            rotated_extent = Rz * extent * Rz';
        end
    end
end 