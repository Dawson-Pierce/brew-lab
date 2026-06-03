function trail = trajectory_trail(comp)
%TRAJECTORY_TRAIL Full estimated path of a trajectory-model component.
    trail = comp.state_history;
    if ~isempty(trail)
        trail = trail(:, 2:end);
        if ~isempty(comp.mean_history)
            trail = [trail, comp.mean_history(:, end)];
        end
    else
        trail = comp.mean_history;
    end
end
