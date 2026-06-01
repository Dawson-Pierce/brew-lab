function trail = trajectory_trail(comp)
%TRAJECTORY_TRAIL Full estimated path of a trajectory-model component.
%   trail = utils.trajectory_trail(comp) returns a (state_dim x T) matrix of
%   per-step state means describing the component's full estimated trajectory,
%   suitable for plotting. It is the single source of truth for "the trajectory
%   trail" shared by utils.plot_distributions and the test scripts.
%
%   Construction:
%     - uses the full state_history (not the bounded L-scan window), so the
%       whole lifetime is drawn;
%     - drops state_history(:,1), the birth prior (the seed/birth location, not
%       an estimated state), so the trail starts at the first real estimate;
%     - appends the current window estimate, because state_history lags the live
%       estimate by one step (the current state is not pushed to it until the
%       next predict) — without this the trail would end one step behind the
%       measurements.
%
%   Falls back to the L-scan window (mean_history) when no full state_history
%   was recorded (e.g. a manually-assembled component).
    trail = comp.state_history;
    if ~isempty(trail)
        trail = trail(:, 2:end);                         % drop birth prior
        if ~isempty(comp.mean_history)
            trail = [trail, comp.mean_history(:, end)];  % append current estimate
        end
    else
        trail = comp.mean_history;
    end
end
