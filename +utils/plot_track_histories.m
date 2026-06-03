function plot_track_histories(tracks, dims, varargin)
%PLOT_TRACK_HISTORIES Plot labeled track trajectories from a label-aware RFS.

    if nargin < 2 || isempty(dims), dims = [1 2]; end
    ip = inputParser; ip.KeepUnmatched = true;
    addParameter(ip, 'ax', gca);
    addParameter(ip, 'colors', []);
    addParameter(ip, 'LineWidth', 1.5);
    addParameter(ip, 'MarkerSize', 6);
    addParameter(ip, 'show_id', true);
    addParameter(ip, 'min_len', 1);
    parse(ip, varargin{:});

    ax = ip.Results.ax; lw = ip.Results.LineWidth; ms = ip.Results.MarkerSize;
    if isempty(tracks), return; end

    n = numel(tracks);
    colors = ip.Results.colors;
    if isempty(colors), colors = lines(n); end
    if size(colors, 1) < n, colors = repmat(colors, ceil(n / size(colors, 1)), 1); end

    prev_hold = ishold(ax); hold(ax, 'on');
    d1 = dims(1); d2 = dims(2);
    is3d = numel(dims) >= 3;
    if is3d, d3 = dims(3); end

    for tr = 1:n
        states = tracks(tr).states;
        if isempty(states) || numel(states) < ip.Results.min_len, continue; end
        xy = cell2mat(states);
        clr = colors(tr, :);
        if is3d
            plot3(ax, xy(d1, :), xy(d2, :), xy(d3, :), '-', 'Color', clr, 'LineWidth', lw);
            plot3(ax, xy(d1, end), xy(d2, end), xy(d3, end), 'o', ...
                'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
            if ip.Results.show_id
                text(ax, xy(d1, end), xy(d2, end), xy(d3, end), ...
                    sprintf(' ID %d', tracks(tr).id), 'Color', clr, 'FontSize', 8);
            end
        else
            plot(ax, xy(d1, :), xy(d2, :), '-', 'Color', clr, 'LineWidth', lw);
            plot(ax, xy(d1, end), xy(d2, end), 'o', ...
                'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', ms);
            if ip.Results.show_id
                text(ax, xy(d1, end) + 0.3, xy(d2, end) + 0.3, ...
                    sprintf('ID %d', tracks(tr).id), 'Color', clr, 'FontSize', 8);
            end
        end
    end

    if ~prev_hold, hold(ax, 'off'); end
end
