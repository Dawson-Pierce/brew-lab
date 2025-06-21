classdef DoubleIntegrator_3D_euler < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz','ax','ay','az','phi','theta','psi','p','q','r','alpha_x','alpha_y','alpha_z'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            pos = state(1:3);
            vel = state(4:6);
            acc = state(7:9);
            eul = state(10:12);
            rates = state(13:15);
            alpha = state(16:18);
            pos_next = pos + vel*dt + 0.5*acc*dt^2;
            vel_next = vel + acc*dt;
            acc_next = acc;
            phi = eul(1); theta = eul(2);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            eul_next = eul + T * rates * dt + 0.5*T*alpha*dt^2;
            rates_next = rates + alpha*dt;
            alpha_next = alpha;

            G = obj.getInputMat(timestep, dt, state);

            nextState = [pos_next; vel_next; acc_next; eul_next; rates_next; alpha_next] + G*u;
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            F = eye(18);
            F(1,4) = dt; F(2,5) = dt; F(3,6) = dt;
            F(1,7) = 0.5*dt^2; F(2,8) = 0.5*dt^2; F(3,9) = 0.5*dt^2;
            F(4,7) = dt; F(5,8) = dt; F(6,9) = dt;
            phi = state(10); theta = state(11);
            p = state(13); q = state(14); r = state(15);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            dT_dphi = [0 cos(phi)*tan(theta) -sin(phi)*tan(theta);
                       0 -sin(phi)           -cos(phi);
                       0 cos(phi)/cos(theta) -sin(phi)/cos(theta)];
            dT_dtheta = [0 sin(phi)*sec(theta)^2 cos(phi)*sec(theta)^2;
                         0 0 0;
                         0 sin(phi)*tan(theta)/cos(theta) cos(phi)*tan(theta)/cos(theta)];
            F(10:12,10) = dT_dphi * [p; q; r] * dt;
            F(10:12,11) = dT_dtheta * [p; q; r] * dt;
            F(10:12,13:15) = T * dt;
            F(10:12,16:18) = 0.5*T*dt^2;
            F(13,16) = dt; F(14,17) = dt; F(15,18) = dt;
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = zeros(18,6);
            phi = state(10); theta = state(11);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            G(1:3,1:3) = 0.5*dt^2*eye(3); % position
            G(4:6,1:3) = dt*eye(3);       % velocity
            G(7:9,1:3) = eye(3);          % acceleration
            G(10:12,4:6) = 0.5*dt^2*T;    % Euler angles
            G(13:15,4:6) = dt*eye(3);     % angular rates
            G(16:18,4:6) = eye(3);        % angular acceleration
        end
    end
end 