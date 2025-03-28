function generate_divergence_video(divergence_tensor, tensor, output_video_name, n_contours, divergence_threshold, mask)
    % generate_divergence_video: Saves a video visualizing the divergence tensor
    % 
    % Parameters:
    % divergence_tensor: Precomputed divergence data (height x width x time)
    % tensor: Original intensity data (height x width x time) for background visualization
    % output_video_name: Name of the output video file (default: 'divergence_plot.mp4')
    % n_contours: Number of contour levels to display (default: 10)
    % divergence_threshold: Threshold for detecting sources and sinks (default: 0.5)
    % mask: Optional logical mask defining the ROI (height x width)
    
    % Set default values if parameters not provided
    if nargin < 3 || isempty(output_video_name)
        output_video_name = 'divergence_plot.mp4';
    end
 
    
    % Get dimensions
    [height, width, time] = size(divergence_tensor);
    
    % Initialize the video writer
    video_writer = VideoWriter(output_video_name, 'MPEG-4');
    video_writer.FrameRate = 10;
    open(video_writer);
    
    % Set up the figure for plotting
    fig = figure('Position', [100, 100, 800, 600]);
    title('Divergence Contour Plot with Sources and Sinks');
    
    % Calculate global min and max for consistent colormap
    if ~isempty(tensor)
        global_min = min(divergence_tensor(:), [], 'omitnan');
        global_max = max(divergence_tensor(:), [], 'omitnan');
        top_prc = prctile(divergence_tensor(:), 98);
        bottom_prc = prctile(divergence_tensor(:), 2);
    end
    
    % Initialize storage for tracking sources and sinks
    tracked_positions = struct('sources', cell(1, time), 'sinks', cell(1, time));
    
    % Loop through each time frame
    for t = 1:time
        % Get current divergence frame and apply mask
        current_divergence = divergence_tensor(:,:,t);
        masked_divergence = current_divergence;

        
        % Display the background tensor if provided
        if ~isempty(tensor)
            imagesc(divergence_tensor(:,:,t), [bottom_prc, top_prc]);
            colormap jet;
            colorbar;
        else
            % Otherwise display the divergence itself as background
            imagesc(masked_divergence);
            colormap jet;
            colorbar;
        end
        
        % Set plot properties
        xlim([1 width]);
        ylim([1 height]);
        set(gca, 'YDir', 'reverse');
        axis equal;
        axis tight;
        
 
        
        set(gca, "Position", [0.1 0.1 0.7 0.8]);
        set(gcf, "InvertHardcopy", 'off');
        
        % Capture the frame for the video
        frame = getframe(fig);
        writeVideo(video_writer, frame);
        
        % Clear for the next frame
        hold off;
        clf;
    end
    
    % Close the video writer and figure
    close(video_writer);
    close(fig);
    
    fprintf('Video saved as %s\n', output_video_name);
end

% Example usage:
% generate_divergence_video(divergence_tensor, tensor, 'divergence_plot.mp4', 10, 0.5)
% 
% If you only have divergence_tensor without the original intensity data:
% generate_divergence_video(divergence_tensor, [], 'divergence_plot.mp4', 10, 0.5)