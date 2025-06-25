classdef DBSCAN_obj < handle
    properties
        epsilon     % Neighborhood radius
        minpts      % Minimum number of points
    end

    methods
        function obj = DBSCAN_obj(epsilon, minpts)
            if nargin < 1, epsilon = 1.0; end
            if nargin < 2, minpts = 3; end
            obj.epsilon = epsilon;
            obj.minpts = minpts;
        end

        function cluster_cells = cluster(obj, Z)
            % Z is N x T (each column is a measurement)
            [N, T] = size(Z);
            if T == 0
                cluster_cells = {};
                return;
            end

            % Transpose for dbscan: it expects M x N (points as rows)
            labels = dbscan(Z', obj.epsilon, obj.minpts);

            % Get valid clusters (positive integers only)
            unique_labels = unique(labels(labels > 0));

            cluster_cells = cell(1, numel(unique_labels));
            for i = 1:numel(unique_labels)
                idx = labels == unique_labels(i);
                cluster_cells{i} = Z(:, idx);
            end
        end

        function unclustered = getUnclustered(obj, Z)
            % Returns unclustered points (noise) as individual N×1 cells
            [~, T] = size(Z);
            if T == 0
                unclustered = {};
                return;
            end

            labels = dbscan(Z', obj.epsilon, obj.minpts);
            idx_noise = labels == -1;
            unclustered = mat2cell(Z(:, idx_noise), size(Z,1), ones(1, sum(idx_noise)));
        end
    end
end
