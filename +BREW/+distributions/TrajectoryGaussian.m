classdef TrajectoryGaussian < BREW.distributions.TrajectoryBaseModel
    % Represents a Gaussian distribution object.
    
    properties 
        means        % trajectory history based on L-scan window
        covariances  % Matrix of covariances based on L-scan window
        mean_history % trajectory history (includes outside L-scan)
        cov_history  % 3D Matrix of covariances per mean, doesn't have cross terms
    end
    
    methods
        function obj = TrajectoryGaussian(idx, meanVal, covVal, varargin)

            p = inputParser;
            addParameter(p,'state_dim',length(meanVal));

            parse(p, varargin{:});
            
            obj@BREW.distributions.TrajectoryBaseModel(idx, p.Results.state_dim);
            
            obj.means = meanVal;
            obj.mean_history = meanVal;
            obj.covariances = covVal;
            obj.cov_history = covVal;

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
            states = reshape(obj.means, [obj.state_dim length(obj.means) / obj.state_dim]);
        end

        function s = sample_measurements(obj,xy_indices,meas_cov)
            mean = obj.getLastState(); 
            mean_temp = mean(xy_indices); 
            s = mvnrnd(mean_temp(:)', meas_cov, 1)';
        end

        function plot(obj, plt_inds, varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'c','w');
            addParameter(p,'ax',gca)
            addParameter(p,'LineWidth',0.8);
            addParameter(p,'LineStyle','-'); 
            parse(p, varargin{:});


            addParameter(p,'window_style',p.Results.LineStyle); 
            addParameter(p,'window_color',p.Results.c); 
            addParameter(p,'window_width',p.Results.LineWidth); 
            parse(p, varargin{:})

            if all(p.Results.LineStyle == p.Results.window_style) || all(p.Results.c == p.Results.window_color) || all(p.Results.LineWidth == p.Results.window_width) 
                plot_window = true;
            else
                plot_window = false;
            end

            states = obj.mean_history;

            if length(plt_inds) == 3 
                plot3(states(plt_inds(1),:), states(plt_inds(2),:), states(plt_inds(3),:), ...
                    'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth); 
                if plot_window
                    states_window = obj.RearrangeStates();
                    plot3(states_window(plt_inds(1),:), states_window(plt_inds(2),:), states_window(plt_inds(3),:), ...
                        'Color', p.Results.window_color,'LineStyle', p.Results.window_style, 'LineWidth', p.Results.window_width);
                end
                
                xlabel('X-axis');
                ylabel('Y-axis');
                zlabel('Z-axis');

            elseif length(plt_inds) == 2 
                plot(p.Results.ax, states(plt_inds(1),:), states(plt_inds(2),:), ...
                    'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth);

                if plot_window
                    states_window = obj.RearrangeStates();
                    plot(p.Results.ax, states_window(plt_inds(1),:), states_window(plt_inds(2),:), ...
                        'Color', p.Results.window_color,'LineStyle', p.Results.window_style, 'LineWidth', p.Results.window_width);
                end 
                
                xlabel('X-axis');
                ylabel('Y-axis'); 
                
            elseif isscalar(plt_inds) == 1 
                ind_valid = obj.init_idx+1:(obj.init_idx+obj.window_size);
                plot(p.Results.ax, ind_valid, states(plt_inds,:), ...
                    'Color', p.Results.c,'LineStyle', p.Results.LineStyle, 'LineWidth', p.Results.LineWidth); 

                if  plot_window
                    states_window = obj.RearrangeStates();
                    if (size(obj.mean_history,2) == size(states_window,2))
                        ind_valid = obj.init_idx+1:(obj.init_idx+size(obj.mean_history,2));
                    else
                        ind_valid = (obj.init_idx+1)+(size(obj.mean_history,2) - size(states_window,2)):(obj.init_idx+size(obj.mean_history,2));
                    end
                    plot(p.Results.ax, ind_valid, states_window(plt_inds,:), ...
                        'Color', p.Results.window_color,'LineStyle', p.Results.window_style, 'LineWidth', p.Results.window_width); 
                end
                
            else
                disp("error plotting - please have plt_inds be a length in range of 1-3")
            end
        end
        

    end
end 