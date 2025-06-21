classdef ContinuousDynamics < BREW.dynamics.DynamicsBase
    % ContinuousDynamics assumes ẋ = f(t, x) + g(t, x) * u
    % Uses ode45 for integration.
    % Extent: extent = M(x) * prev_extent * M(x)'

    properties (Constant)
        stateNames = {'Unknown'}
    end

    properties
        f      % function handle for continuous-time dynamics: f(t, x)
        F = [] % linearized state transition function handle or matrix
        g      % function handle for control input effect: g(t, x)
        G = [] % linearized input matrix or function 
    end

    methods

        function obj = ContinuousDynamics(varargin)
            p = inputParser;
            p.CaseSensitive = true;
            addParameter(p, 'F', []);
            addParameter(p, 'G', []);
            addParameter(p, 'f', []);
            addParameter(p, 'g', []);
            addParameter(p, 'M', []);
            parse(p, varargin{:});

            obj.F = p.Results.F;
            obj.G = p.Results.G;
            obj.f = p.Results.f;
            obj.g = p.Results.g;
            obj.M = p.Results.M;
        end

        function nextState = propagateState(obj, timestep, dt, state, u)
            if nargin < 5, u = []; end
            tspan = [0, dt]; % relative time interval for ode45

            ode_rhs = @(t, x) obj.compute_rhs(timestep + t, x, u);

            opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
            [~, X] = ode45(ode_rhs, tspan, state, opts);
            nextState = X(end, :)';
        end

        function dx = compute_rhs(obj, t, x, u)
            % Compute RHS: dx/dt = f(t, x) + g(t, x) * u
            if isempty(obj.f)
                dx = zeros(size(x));
            else
                dx = obj.f(t, x);
            end

            if ~isempty(obj.g) && ~isempty(u)
                if isa(obj.g, 'function_handle')
                    dx = dx + obj.g(t, x) * u;
                else
                    dx = dx + obj.g * u;
                end
            end
        end

        function stateMat = getStateMat(obj, timestep, ~, state, varargin)
            if isa(obj.F, 'function_handle')
                stateMat = obj.F(timestep, state);
            else
                stateMat = obj.F;
            end
        end

        function inputMat = getInputMat(obj, timestep, ~, state, varargin)
            if isa(obj.G, 'function_handle')
                inputMat = obj.G(timestep, state);
            else
                inputMat = obj.G;
            end
        end

        function new_extent = propagate_extent(obj, state, extent, varargin)
            if isa(obj.M, 'function_handle')
                Mx = obj.M(state);
                new_extent = Mx * extent * Mx';
            elseif isempty(obj.M)
                new_extent = extent;
            else
                new_extent = obj.M * extent * obj.M';
            end
        end

    end
end
