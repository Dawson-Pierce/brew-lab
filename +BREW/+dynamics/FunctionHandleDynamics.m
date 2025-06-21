classdef FunctionHandleDynamics < BREW.dynamics.DynamicsBase
    % NOTE: Assumes x dot = f(timestep, state, dt) + g(timestep, state, dt) * u
    % NOTE: Assumes extent = M(timestep, state, dt) * prev_extent * M(timestep, state, dt)'

    properties (Constant)
        stateNames = {'Unknown'}
    end

    properties
        f      % function handle for dynamics 
        F = [] % function handle for linearized state transition
        g      % function handle for g function
        G = [] % function handle for G matrix 
    end

    methods 

        function obj = FunctionHandleDynamics(varargin)
            p = inputParser;  
            p.CaseSensitive = true;
            addParameter(p, 'F', []); % F for linear kinematics
            addParameter(p, 'G', []); % G for linear kinematics
            addParameter(p, 'f', []); % function handle for kinematics
            addParameter(p, 'g', []); % function handle for input 
            addParameter(p, 'M', []); % function/matrix for extent rotation 

            % Parse known arguments
            parse(p, varargin{:});

            obj.F = p.Results.F;
            obj.G = p.Results.G;
            obj.f = p.Results.f;
            obj.g = p.Results.g;
            obj.M = p.Results.M;
        end

        function nextState = propagateState(obj, timestep, dt, state, u)
            if ~isempty(obj.g) && ~isempty(u)
                if isa(obj.g,'function_handle')
                    nextState = obj.f(timestep, state, dt) + obj.g(timestep, state, dt) * u;
                else
                    nextState = obj.f(timestep, state, dt) + obj.g * u;
                end
            else
                nextState = obj.f(timestep, state, dt);
            end
        end

        function stateMat = getStateMat(obj, timestep, dt, state, varargin)
            if isa(obj.F,'function_handle')
                stateMat = obj.F(timestep, state, dt);
            else
                stateMat = obj.F;
            end
        end
        
        function inputMat = getInputMat(obj, timestep, dt, state, varargin) 
            if isa(obj.G,'function_handle')
                inputMat = obj.G(timestep, state, dt);
            else
                inputMat = obj.G;
            end
        end

        function new_extent = propagate_extent(obj, dt, state, extent)
            if isa(obj.M,'function_handle')
                new_extent = obj.M(timestep, state, dt) * extent * obj.M(timestep, state, dt)';
            elseif isempty(obj.M)
                new_extent = extent;
            else
                new_extent = obj.M * extent * obj.M';
            end
        end 
    end
end 