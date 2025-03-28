function [divergence_tensor,tracked_positions] = improved_tracked_contour_video(u_tensor, v_tensor, tensor, output_video_name, sigma, temporal_window, divergence_threshold, n_contours, mask)
    % save_tracked_contour_video: Saves a video with tracked sources and sinks
    % u_tensor, v_tensor: Vector field components (height x width x time-1)
    % tensor: Original intensity data (height x width x time) for background visualization
    % output_video_name: Name of the output video file
    % sigma: Standard deviation for Gaussian spatial smoothing
    % temporal_window: Number of frames to average for temporal smoothing
    % divergence_threshold: Threshold for detecting sources and sinks
    % n_contours: Number of contour levels for visualization
    % mask: Logical mask defining the ROI (height x width)

    [height, width, time] = size(u_tensor);

    % Initialize the video writer
    video_writer = VideoWriter(output_video_name, 'MPEG-4');
    video_writer.FrameRate = 10;
    open(video_writer);

    % Set up the figure for plotting
    fig = figure('Position', [100, 100, 800, 600]);
    hold on;
    title('Tracked Sources and Sinks in Divergence Contour Plot');

    % Calculate global min and max for consistent colormap
    global_min = min(tensor(:), [], 'omitnan');
    global_max = max(tensor(:), [], 'omitnan');
    top_prc = prctile(tensor(:),98);
    bottom_prc = prctile(tensor(:),2);
    
    % Precompute divergence for each frame
    divergence_tensor = zeros(height, width, time-1);

    for t = 1:time
        % Extract current frame flow fields
        u_current = u_tensor(:,:,t);
        v_current = v_tensor(:,:,t);

        % Compute divergence using gradient but with special handling for boundaries
        % Compute x and y derivatives separately
        [du_dx, du_dy] = gradient(u_current);
        [dv_dx, dv_dy] = gradient(v_current);

        % Compute divergence (∇·v = du/dx + dv/dy)
        divergence = du_dx + dv_dy;

        % The key issue: only apply the mask AFTER computing the full divergence
        masked_divergence = divergence;

        % Apply the mask (keep values inside the mask)
        masked_divergence(~mask) = NaN;

        % Smooth the divergence using Gaussian filter
        % Only operate on valid (non-NaN) regions
        valid_mask = ~isnan(masked_divergence);
        smoothed_divergence = masked_divergence;

        if any(valid_mask(:)) % Only smooth if there are valid points
            % Replace NaNs temporarily for smoothing
            temp_divergence = masked_divergence;
            temp_divergence(~valid_mask) = 0;

            % Apply Gaussian smoothing
            smoothed_temp = imgaussfilt(temp_divergence, sigma);

            % Only keep smoothed values within the valid region
            smoothed_divergence(valid_mask) = smoothed_temp(valid_mask);
        end

        % Store the result
        divergence_tensor(:,:,t) = smoothed_divergence;
    end
    
    % Initialize storage for tracking sources and sinks
    tracked_positions = struct('sources', [], 'sinks', []);

    % Loop through each time frame to detect and track sources and sinks
    for t = 1:time-1
        % Temporal smoothing
        start_frame = max(1, t - floor(temporal_window / 2));
        end_frame = min(time-1, t + floor(temporal_window / 2));
        divergence_avg = mean(divergence_tensor(:,:,start_frame:end_frame), 3, 'omitnan');

        % Display the current frame as an image
        imagesc(tensor(:,:,t), [bottom_prc, top_prc]);
        colormap jet;
        colorbar;
        xlim([1 width]);
        ylim([1 height]);
        set(gca, 'YDir', 'reverse');
        axis equal;
        axis tight;

        % Generate contour data for analysis without plotting
        % Automatically generate contour levels 
        div_min = min(divergence_avg(:), [], 'omitnan');
        div_max = max(divergence_avg(:), [], 'omitnan');
        contour_levels = linspace(div_min, div_max, 10); % 10 levels 
        
        % Compute contours
        [C, h] = contour(divergence_avg, contour_levels, 'Visible', 'off');
        
        % --- Process contours to identify closed ones ---
        % Extract contour data
        contour_data = parseContours(C);
        
        % Initial source and sink detection by thresholding
        potential_sources = divergence_avg > divergence_threshold;
        potential_sinks = divergence_avg < -divergence_threshold;
        
        % Find connected components
        [source_labels, num_sources] = bwlabel(potential_sources);
        [sink_labels, num_sinks] = bwlabel(potential_sinks);
        
        % Get centroids and regions
        source_props = regionprops(source_labels, 'Centroid', 'PixelIdxList');
        sink_props = regionprops(sink_labels, 'Centroid', 'PixelIdxList');
        
        % Filter sources and sinks by closed contour criterion
        valid_sources = [];
        valid_sinks = [];
        
        % Process sources
        for i = 1:num_sources
            centroid = source_props(i).Centroid;
            pixel_indices = source_props(i).PixelIdxList;
            div_value = mean(divergence_avg(pixel_indices));
            
            % Check for closed contours around this source
            num_closed_contours = countClosedContours(contour_data, centroid, div_value, true);
            
            % Only accept sources with at least 2 closed contours
            if num_closed_contours >= 2
                valid_sources = [valid_sources; centroid];
            end
        end
        
        % Process sinks
        for i = 1:num_sinks
            centroid = sink_props(i).Centroid;
            pixel_indices = sink_props(i).PixelIdxList;
            div_value = mean(divergence_avg(pixel_indices));
            
            % Check for closed contours around this sink
            num_closed_contours = countClosedContours(contour_data, centroid, div_value, false);
            
            % Only accept sinks with at least 2 closed contours
            if num_closed_contours >= 2
                valid_sinks = [valid_sinks; centroid];
            end
        end
        
        % Store the filtered results
        tracked_positions(t).sources = valid_sources;
        tracked_positions(t).sinks = valid_sinks;

        % --- Visualization ---
        % Plot contours
        contour(divergence_avg, n_contours, 'LineColor', 'k', 'LineWidth', 1);

        % Plot source and sink centroids
        if ~isempty(tracked_positions(t).sources)
            plot(tracked_positions(t).sources(:,1), tracked_positions(t).sources(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
        if ~isempty(tracked_positions(t).sinks)
            plot(tracked_positions(t).sinks(:,1), tracked_positions(t).sinks(:,2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
        end

        set(gca, "Position",[0.1 0.1 0.8 0.8]);
        set(gcf, "InvertHardcopy", 'off');
        
        % Capture the frame for the video
        frame = getframe(fig);
        writeVideo(video_writer, frame);

        % Clear for the next frame
        cla;
    end

    % Close the video writer and figure
    close(video_writer);
    close(fig);
end

% Helper function to parse contour matrix into usable structure
function contour_data = parseContours(C)
    % Initialize
    contour_data = struct('level', {}, 'x', {}, 'y', {}, 'closed', {});
    
    % Parse the contour matrix
    i = 1;
    contour_idx = 1;
    
    while i < size(C, 2)
        level = C(1, i);
        n_points = C(2, i);
        
        % Get contour coordinates
        x = C(1, i+1:i+n_points);
        y = C(2, i+1:i+n_points);
        
        % Check if contour is closed (first and last points match)
        is_closed = (abs(x(1) - x(end)) < 1e-6) && (abs(y(1) - y(end)) < 1e-6);
        
        % Store the contour
        contour_data(contour_idx).level = level;
        contour_data(contour_idx).x = x;
        contour_data(contour_idx).y = y;
        contour_data(contour_idx).closed = is_closed;
        
        % Move to next contour
        i = i + n_points + 1;
        contour_idx = contour_idx + 1;
    end
end

% Helper function to count closed contours around a point
function count = countClosedContours(contour_data, centroid, div_value, is_source)
    count = 0;
    pt_x = centroid(1);
    pt_y = centroid(2);
    
    for i = 1:length(contour_data)
        % Only consider closed contours
        if ~contour_data(i).closed
            continue;
        end
        
        % For sources, only consider positive contours
        % For sinks, only consider negative contours
        contour_level = contour_data(i).level;
        if (is_source && contour_level <= 0) || (~is_source && contour_level >= 0)
            continue;
        end
        
        % Check if point is inside this contour using inpolygon
        if inpolygon(pt_x, pt_y, contour_data(i).x, contour_data(i).y)
            count = count + 1;
        end
    end
end