classdef DoubleIntegrator_3D_quaternion < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz','ax','ay','az','q0','q1','q2','q3','wx','wy','wz','alpha_x','alpha_y','alpha_z'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            pos = state(1:3);
            vel = state(4:6);
            acc = state(7:9);
            q = state(10:13);
            omega = state(14:16);
            alpha = state(17:19);
            pos_next = pos + vel*dt + 0.5*acc*dt^2;
            vel_next = vel + acc*dt;
            acc_next = acc;
            Omega = [0 -omega(1) -omega(2) -omega(3);
                     omega(1) 0 omega(3) -omega(2);
                     omega(2) -omega(3) 0 omega(1);
                     omega(3) omega(2) -omega(1) 0];
            Omega_alpha = [0 -alpha(1) -alpha(2) -alpha(3);
                           alpha(1) 0 alpha(3) -alpha(2);
                           alpha(2) -alpha(3) 0 alpha(1);
                           alpha(3) alpha(2) -alpha(1) 0];
            q_next = q + 0.5*Omega*q*dt + 0.25*Omega_alpha*q*dt^2;
            q_next = q_next / norm(q_next);
            omega_next = omega + alpha*dt;
            alpha_next = alpha;
            nextState = [pos_next; vel_next; acc_next; q_next; omega_next; alpha_next];
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            F = eye(19);
            F(1,4) = dt; F(2,5) = dt; F(3,6) = dt;
            F(1,7) = 0.5*dt^2; F(2,8) = 0.5*dt^2; F(3,9) = 0.5*dt^2;
            F(4,7) = dt; F(5,8) = dt; F(6,9) = dt;
            q = state(10:13);
            wx = state(14); wy = state(15); wz = state(16);
            Omega = [0 -wx -wy -wz;
                     wx 0 wz -wy;
                     wy -wz 0 wx;
                     wz wy -wx 0];
            F(10:13,10:13) = eye(4);
            F(10:13,14) = 0.5 * [ -q(2);  q(1);  q(4); -q(3)] * dt;
            F(10:13,15) = 0.5 * [ -q(3); -q(4);  q(1);  q(2)] * dt;
            F(10:13,16) = 0.5 * [ -q(4);  q(3); -q(2);  q(1)] * dt;
            F(10:13,17:19) = 0.25 * [
                -q(2) -q(3) -q(4);
                 q(1) -q(4)  q(3);
                 q(4)  q(1) -q(2);
                -q(3)  q(2)  q(1)] * dt^2;
            F(14,17) = dt; F(15,18) = dt; F(16,19) = dt;
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = zeros(19,6);
            q = state(10:13);
            G(1:3,1:3) = 0.5*dt^2*eye(3); % position
            G(4:6,1:3) = dt*eye(3);       % velocity
            G(7:9,1:3) = eye(3);          % acceleration
            G(10:13,4:6) = 0.25*dt^2*[
                -q(2) -q(3) -q(4);
                 q(1) -q(4)  q(3);
                 q(4)  q(1) -q(2);
                -q(3)  q(2)  q(1)];
            G(14:16,4:6) = dt*eye(3);     % angular rates
            G(17:19,4:6) = eye(3);        % angular acceleration
        end
    end
end 