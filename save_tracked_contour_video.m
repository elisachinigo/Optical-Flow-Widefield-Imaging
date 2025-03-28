function [divergence_tensor,tracked_positions] = save_tracked_contour_video(u_tensor, v_tensor, tensor, output_video_name, sigma, temporal_window, divergence_threshold,n_contours, mask)
    % save_tracked_contour_video: Saves a video with tracked sources and sinks
    % u_tensor, v_tensor: Vector field components (height x width x time-1)
    % tensor: Original intensity data (height x width x time) for background visualization
    % output_video_name: Name of the output video file
    % sigma: Standard deviation for Gaussian spatial smoothing
    % temporal_window: Number of frames to average for temporal smoothing
    % divergence_threshold: Threshold for detecting sources and sinks
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
        %axis([1 width 1 height]);
        set(gca, 'YDir', 'reverse');
        axis equal;
        axis tight;

        % Detect sources and sinks by thresholding divergence
        sources = divergence_avg > divergence_threshold;
        sinks = divergence_avg < -divergence_threshold;

        % Label connected components in sources and sinks
        [source_labels, num_sources] = bwlabel(sources);
        [sink_labels, num_sinks] = bwlabel(sinks);

        % Calculate centroids for each source and sink
        source_centroids = regionprops(source_labels, 'Centroid');
        sink_centroids = regionprops(sink_labels, 'Centroid');

        % Store centroids for tracking (optional: refine by tracking closest centroids)
        if num_sources > 0 
            tracked_positions(t).sources = cat(1, source_centroids.Centroid);
        else
            tracked_positions(t).sources = [];
        end
        if num_sinks > 0
            tracked_positions(t).sinks = cat(1, sink_centroids.Centroid);
        else 
            tracked_positions(t).sinks = [];
        end

        % Plot contours and mark sources and sinks
        hold on;
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