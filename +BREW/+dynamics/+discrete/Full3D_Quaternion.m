classdef Full3D_Quaternion < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz','q0','q1','q2','q3','wx','wy','wz'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            if nargin < 5 || isempty(u)
                u = zeros(6,1);
            end
            pos = state(1:3);
            vel = state(4:6);
            q = state(7:10);
            omega = state(11:13);
            pos_next = pos + vel * dt;
            vel_next = vel;
            Omega = [0 -omega(1) -omega(2) -omega(3);
                     omega(1) 0 omega(3) -omega(2);
                     omega(2) -omega(3) 0 omega(1);
                     omega(3) omega(2) -omega(1) 0];
            q_next = q + 0.5 * Omega * q * dt;
            q_next = q_next / norm(q_next);
            omega_next = omega;
            nextState = [pos_next; vel_next; q_next; omega_next] + obj.getInputMat(timestep, dt, state) * u;
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            F = eye(13);
            F(1,4) = dt; F(2,5) = dt; F(3,6) = dt;
            q = state(7:10);
            wx = state(11); wy = state(12); wz = state(13);
            Omega = [0 -wx -wy -wz;
                     wx 0 wz -wy;
                     wy -wz 0 wx;
                     wz wy -wx 0];
            F(7:10,7:10) = eye(4);
            F(7:10,11) = 0.5 * [ -q(2);  q(1);  q(4); -q(3)] * dt;
            F(7:10,12) = 0.5 * [ -q(3); -q(4);  q(1);  q(2)] * dt;
            F(7:10,13) = 0.5 * [ -q(4);  q(3); -q(2);  q(1)] * dt;
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = zeros(13,6);
            q = state(7:10);
            G(1:3,1:3) = dt*eye(3); % position
            G(4:6,1:3) = eye(3);    % velocity
            % Quaternion update block
            G(7:10,4:6) = 0.5*dt*[
                -q(2) -q(3) -q(4);
                 q(1) -q(4)  q(3);
                 q(4)  q(1) -q(2);
                -q(3)  q(2)  q(1)];
            G(11:13,4:6) = eye(3);  % angular rates
        end
        function new_extent = propagate_extent(obj, dt, state, extent)
            % Propagate the extent by rotating it according to the angular rates (wx, wy, wz) over dt using quaternion math
            % state: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
            omega = state(11:13);
            omega_norm = norm(omega);
            if omega_norm < 1e-12
                R = eye(3);
            else
                theta = omega_norm * dt;
                k = omega / omega_norm;
                K = [0, -k(3), k(2); k(3), 0, -k(1); -k(2), k(1), 0];
                R = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
            end
            new_extent = R * extent * R';
        end
        function rotated_extent = rotate_extent(obj, dt, state, extent)
            % Rotate the extent by the quaternion update implied by the angular rates (wx, wy, wz) over dt
            omega = state(11:13);
            omega_norm = norm(omega);
            if omega_norm < 1e-12
                R = eye(3);
            else
                theta = omega_norm * dt;
                k = omega / omega_norm;
                K = [0, -k(3), k(2); k(3), 0, -k(1); -k(2), k(1), 0];
                R = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
            end
            rotated_extent = R * extent * R';
        end
    end
end 