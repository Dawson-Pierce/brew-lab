classdef TrajectoryGaussian < BREW.distributions.TrajectoryBaseModel
    % Represents a Gaussian distribution object.
    
    properties 
        means
        covariances 
    end
    
    methods
        function obj = TrajectoryGaussian(idx, meanVal, covVal, varargin)

            p = inputParser;
            addParameter(p,'state_dim',length(meanVal));

            parse(p, varargin{:});
            
            obj@BREW.distributions.TrajectoryBaseModel(idx, p.Results.state_dim);
            
            obj.means = meanVal;
            obj.covariances = covVal;

        end 
        
        function state = getLastState(obj)
            % Gets most recent state
            state = obj.means((length(obj.means) - obj.state_dim + 1):end,:);
        end

        function cov = getLastCov(obj)
            cov = obj.covariances((length(obj.means) - obj.state_dim + 1):end,(length(obj.means) - obj.state_dim + 1):end);
        end

        function states = RearrangeStates(obj)
            % Gets states from column to N X T matrix 
            % Useful for plotting
            states = reshape(obj.means, [obj.state_dim obj.window_size]);
        end

        function s = sample_measurements(obj,xy_indices,meas_cov)
            mean = obj.getLastState(); 
            mean_temp = mean(xy_indices); 
            s = mvnrnd(mean_temp(:)', meas_cov, 1)';
        end

        function plot(obj, plt_inds, varargin)
            p = inputParser;
            addParameter(p,'c','w');
            addParameter(p,'ax',gca)
            addParameter(p,'LineWidth',0.8);
            addParameter(p,'LineStyle','-'); 
            parse(p, varargin{:});

            states = obj.RearrangeStates();
            if length(plt_inds) == 3
                plot3(states(plt_inds(1),:), states(plt_inds(2),:), states(plt_inds(3),:), 'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth);
                xlabel('X-axis');
                ylabel('Y-axis');
                zlabel('Z-axis');
            else
                plot(p.Results.ax, states(plt_inds(1),:), states(plt_inds(2),:), 'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth);
                xlabel('X-axis');
                ylabel('Y-axis'); 
            end
        end
        

    end
end 