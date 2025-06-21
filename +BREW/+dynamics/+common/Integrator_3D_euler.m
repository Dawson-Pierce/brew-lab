classdef Integrator_3D_euler < BREW.dynamics.DynamicsBase
    properties (Constant)
        stateNames = {'x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'}
    end
    methods
        function nextState = propagateState(obj, timestep, dt, state, u)
            if nargin < 5 || isempty(u)
                u = zeros(6,1);
            end
            pos = state(1:3);
            vel = state(4:6);
            eul = state(7:9);
            rates = state(10:12);
            pos_next = pos + vel * dt;
            vel_next = vel;
            phi = eul(1); theta = eul(2);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            eul_next = eul + T * rates * dt;
            rates_next = rates;
            
            G = obj.getInputMat(timestep, dt, state);

            nextState = [pos_next; vel_next; eul_next; rates_next] + G * u;
        end
        function F = getStateMat(obj, timestep, dt, state, varargin)
            F = eye(12);
            F(1,4) = dt; F(2,5) = dt; F(3,6) = dt;
            phi = state(7); theta = state(8);
            p = state(10); q = state(11); r = state(12);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            dT_dphi = [0 cos(phi)*tan(theta) -sin(phi)*tan(theta);
                       0 -sin(phi)           -cos(phi);
                       0 cos(phi)/cos(theta) -sin(phi)/cos(theta)];
            dT_dtheta = [0 sin(phi)*sec(theta)^2 cos(phi)*sec(theta)^2;
                         0 0 0;
                         0 sin(phi)*tan(theta)/cos(theta) cos(phi)*tan(theta)/cos(theta)];
            F(7:9,7) = dT_dphi * [p; q; r] * dt;
            F(7:9,8) = dT_dtheta * [p; q; r] * dt;
            F(7:9,10:12) = T * dt;
        end
        function G = getInputMat(obj, timestep, dt, state, varargin)
            G = zeros(12,6);
            phi = state(7); theta = state(8);
            T = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
                 0 cos(phi)           -sin(phi);
                 0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
            G(1:3,1:3) = dt*eye(3); % position
            G(4:6,1:3) = eye(3);    % velocity
            G(7:9,4:6) = dt*T;      % Euler angles
            G(10:12,4:6) = eye(3);  % angular rates
        end
        function new_extent = propagate_extent(obj, state, extent, varargin)
            % Propagate the extent by rotating it according to the angular rates (p, q, r) over dt
            % state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'dt', []); 
            parse(p, varargin{:});

            dt = p.Results.dt;
            
            dphi = state(10) * dt;
            dtheta = state(11) * dt;
            dpsi = state(12) * dt;
            Rx = [1 0 0; 0 cos(dphi) -sin(dphi); 0 sin(dphi) cos(dphi)];
            Ry = [cos(dtheta) 0 sin(dtheta); 0 1 0; -sin(dtheta) 0 cos(dtheta)];
            Rz = [cos(dpsi) -sin(dpsi) 0; sin(dpsi) cos(dpsi) 0; 0 0 1];
            R = Rz * Ry * Rx;
            new_extent = R * extent * R';
        end
    end
end 