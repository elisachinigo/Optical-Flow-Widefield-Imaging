% Let's filter the data so that we get rid of the noisy pixels
%load('/Users/elisachinigo/Documents/Widefield_Data/Data_videos/video.mat');
load('/Users/elisachinigo/Documents/Widefield_Data/Data_videos/mask.mat');
% step 1: compute the average intensity for each pixel over time
%avg_values = mean(video,3);
avg_values = mean(NREM_widefield.data,1);
video = NREM_widefield.data;

% step 2: Plot the average values taken by each pixel over time as an image
figure;
imagesc(avg_values);
colorbar;
title("Average Intensity per Pixel Over Time");
xlabel("X Pixel");
ylabel("Y Pixel");
axis image;
colormap jet;

% step 3: Plot the histogram of average values
figure;
histogram(avg_values(:), 'Binwidth', 0.05); % flatten to 1D for histogram
title('Histogram of average Pixel intensities');
xlabel('Average Intensity');
ylabel('Frequency');
grid on;

% Now lets filter out these high brightness and very negative brightness pixels
% Define a threshold for filtering based offf the average intensities
high_threshold = 0.01;
low_threshold = -0.01;
% CCreate a mask for high-intensity average values and filter the data
high_intensity_mask = avg_values > high_threshold;
low_intensity_mask = avg_values < low_threshold;

% step 5: set high-intensity pixels to zero across all time points
for t = 1:size(video,3)
   video(:,:, t) = video(:,:,t).* ~high_intensity_mask; % set high intensity pixels to zero
   video(:,:, t) = video(:,:,t).* ~low_intensity_mask; 
end

% Visualise after filtering
figure;
imagesc(mean(video,3));
colorbar;
title('Average Intensity After Filtering High Values')
axis image;
colormap jet;


%% 

alpha = 1;
num_iterations = 200;
output_video_name = 'optical_flow_video_full.mp4';


shifted_data = video_3d*100 + abs(-2);
top_prc = prctile(shifted_data(:),90);
bottom_prc = prctile(shifted_data(:),10);
disp(top_prc)
disp(bottom_prc)
[u_tensor, v_tensor]=plot_quiver_video(shifted_data(:,:,1:300),alpha, num_iterations, output_video_name,mask);

%% Now let's look at the divergence of the vector field and do some source/sink analysis 
%save_smoothed_contour_video(u_tensor, v_tensor, shifted_data, 'divergence_video_short_2.mp4',20,5);
%[divergence_tensor,tracked_positions,source_tracks,sink_tracks] = save_tracked_contour_video(u_tensor(:,:,1:299), v_tensor(:,:,1:299), shifted_data(:,:,1:299), 'source_sink_video.mp4', 20, 2, 0.0005,mask);
[divergence_tensor,tracked_positions] = save_tracked_contour_video(u_tensor(:,:,1:299), v_tensor(:,:,1:299), shifted_data(:,:,1:299), 'source_sink_video.mp4', 20, 2, 0.005, 10,mask);

    % save_tracked_contour_video: Saves a video with tracked sources and sinks
    % u_tensor, v_tensor: Vector field components (height x width x time-1)
    % tensor: Original intensity data (height x width x time) for background visualization
    % output_video_name: Name of the output video file
    % sigma: Standard deviation for Gaussian spatial smoothing
    % temporal_window: Number of frames to average for temporal smoothing
    % divergence_threshold: Threshold for detecting sources and sinks
    % n_contours: number of contours to plot
    % mask: Logical mask defining the ROI (height x width)
%% 
[divergence, tracked, source_tracks, sink_tracks] = save_tracked_contour_video(u_tensor(:,:,1:299), v_tensor(:,:,1:299), shifted_data(:,:,1:299), 'output.mp4', 20, 2, 2, mask);
%%
abs_divergence = abs(divergence_tensor(:));
threshold_guess = prctile(abs_divergence, 95);
figure;
histogram(divergence_tensor(:), 'Normalization', 'probability');
xlabel('Divergence Value');
ylabel('Probability');
title('Divergence Value Distribution');

%%  Functions for flow field plotting


% note: this function averages gradients twice (might be excessive)
function [u_tensor,v_tensor] =plot_quiver_video(tensor, alpha, num_iterations, output_video_name,mask)
    % tensor: 3D matrix representing the intensity of the pixels over time
    % (height x width x time)
    % alpha: regularization parameter for Horn-Schunck method
    % num_iterations: nubmer of iterations for the Horn-Schunck method 
    % output_video_name: name of the output video file

    % get the size of the tensor
    [height, width, time ] = size(tensor);

    % precompute the flow fields for all frames
    u_tensor = zeros(height, width, time-1);
    v_tensor = zeros(height, width, time-1);

    % Compute flow for each consecutive pair of frames
    for t = 1: time -1
        tensor(isnan(tensor))= 0; % Replace NaNs with 0
        % Extract two consecutive frames
        I1 = tensor(:,:,t); % frame t
        I2 = tensor(:,:,t+1); % frame t+1

        % Compute the optical flow between consecutive frames
        [u,v] = horn_schunck(I1, I2, alpha, num_iterations);

        % Store the flow components in the tensors
        u_tensor(:,:, t) = u;
        v_tensor(:,:, t) = v;
    end
    
    % set the flow fields outside of the mask to NaNs
    % Ensure mask is replicated to match the size of the 3D tensor
    mask_3D = repmat(mask, [1, 1, size(u_tensor, 3)]);

    % Apply the mask to the tensors
    u_tensor(~mask_3D) = NaN;
    v_tensor(~mask_3D) = NaN;

    % Initialize the video writer with a fixed frame size 
    v = VideoWriter(output_video_name , 'MPEG-4');
    v.FrameRate = 10; % Adjust frame rate as desired
    open(v);

    % Set up the figure for plotting
    fig = figure('Position', [100, 100, 640, 480]); % Fixed figure size for consistency
    hold on;
    title('Optical Flow Video - Quiver Plot');

    % Prepare a grid for the quiver plot
    step_size = 4; % Subsampling step size for sparse arrows

    % Create a meshgrid for subsampling (spatial dimensions only)
    [XX, YY] = meshgrid(1: step_size:width, 1:step_size:height);

    % Pre-allocate the subsampled tensors (u and v for all frames)
    u_subsampled = u_tensor(1:step_size:end, 1:step_size:end, :);
    v_subsampled = v_tensor(1:step_size:end, 1:step_size:end, :);

    % Gaussian smoothing (adjust the sigma value for desired smoothness)
    % Should also check this bit

    sigma = 0.5;
    u_smooth = imgaussfilt(u_subsampled, sigma);
    v_smooth = imgaussfilt(v_subsampled, sigma);


    top_prc = prctile(tensor(:),98);
    bottom_prc = prctile(tensor(:),2);

    % Loop through each time slice and update the quiver plot
    for t = 1:time-1
        % Display the current frame as an image
        imagesc(tensor(:,:,t), [bottom_prc, top_prc]);
        axis([1 width 1 height]);
        set(gca, 'YDir', 'reverse'); %Flip the y-axis so 0 is at the top
        axis equal;
        axis tight;
        colormap jet; 
        colorbar;

        % plot the smoothed quiver field
        quiver(XX, YY, u_smooth(:,:,t), v_smooth(:,:,t), 2, 'Color','k');

        % Capture the frame for the video
        frame = getframe(fig);
        writeVideo(v, frame);

        % Pause for display
        pause(1/10); 

        cla; % clear for next frame
    end

    % close video writer and figure
    close(v);
    hold off;
    close(fig);
end






function [u, v] = horn_schunck(I1, I2, alpha, num_iterations)
    % Horn-Schunck optical flow estimation method with averaging of
    % gradients

    % Step 1: Compute image gradients
    %weighted_data = data .* mask;
    %[dx, dy] = gradient(weighted_data);

    [Ix, Iy] = gradient(double(I1));
    It = double(I2) - double(I1);

    % Apply spatial smoothing (averaging) on gradients Ix, Iy, and It
    Ix_avg = (circshift(Ix, [0, -1]) + circshift(Ix, [0, 1]) + circshift(Ix, [-1, 0]) + circshift(Ix, [1, 0])) / 4;
    Iy_avg = (circshift(Iy, [0, -1]) + circshift(Iy, [0, 1]) + circshift(Iy, [-1, 0]) + circshift(Iy, [1, 0])) / 4;
    It_avg = (circshift(It, [0, -1]) + circshift(It, [0, 1]) + circshift(It, [-1, 0]) + circshift(It, [1, 0])) / 4;

    % Step 2: Initialize flow components (u and v)
    [height, width] = size(I1);
    u = zeros(height, width);
    v = zeros(height, width);

    % Step 3: iteratively refione flow estimation
    for iter = 1:num_iterations
        % Compute averages of the flow at neighboring pixels
        u_avg = (circshift(u, [0, -1]) + circshift(u, [0, 1]) + circshift(u, [-1, 0]) + circshift(u, [1, 0])) / 4;
        v_avg = (circshift(v, [0, -1]) + circshift(v, [0, 1]) + circshift(v, [-1, 0]) + circshift(v, [1, 0])) / 4;

        % Update the flow components using the Horn-Schunck equations with smoothed gradients
        u = u_avg - Ix_avg .* ((Ix_avg .* u_avg + Iy_avg .* v_avg + It_avg) ./ (alpha^2 + Ix_avg.^2 + Iy_avg.^2));
        v = v_avg - Iy_avg .* ((Ix_avg .* u_avg + Iy_avg .* v_avg + It_avg) ./ (alpha^2 + Ix_avg.^2 + Iy_avg.^2));
    end
end
%% 

%% 

% % 
% function [divergence_tensor, tracked_positions,source_tracks,sink_tracks] = save_tracked_contour_video(u_tensor, v_tensor, tensor, output_video_name, sigma, temporal_window, divergence_threshold, mask)
%     % save_tracked_contour_video: Saves a video with tracked sources and sinks
%     % u_tensor, v_tensor: Vector field components (height x width x time-1)
%     % tensor: Original intensity data (height x width x time) for background visualization
%     % output_video_name: Name of the output video file
%     % sigma: Standard deviation for Gaussian spatial smoothing
%     % temporal_window: Number of frames to average for temporal smoothing
%     % divergence_threshold: Threshold for detecting sources and sinks
%     % mask: Logical mask defining the ROI (height x width)
% 
%     [height, width, time] = size(u_tensor);
% 
%     % Debug: Print the dimensions to verify everything is as expected
%     fprintf('Dimensions: height=%d, width=%d, time=%d\n', height, width, time);
% 
%     % Initialize the video writer
%     video_writer = VideoWriter(output_video_name, 'MPEG-4');
%     video_writer.FrameRate = 10;
%     open(video_writer);
% 
%     % Set up the figure for plotting
%     fig = figure('Position', [100, 100, 800, 600]);
% 
%     % Calculate percentiles for consistent colormap
%     top_prc = prctile(tensor(:),98);
%     bottom_prc = prctile(tensor(:),2);
% 
%     % Precompute divergence for each frame
%     divergence_tensor = zeros(height, width, time);
% 
%     % Loop through each time frame and compute divergence
%     for t = 1:time
%         % Extract current frame flow fields
%         u_current = u_tensor(:,:,t);
%         v_current = v_tensor(:,:,t);
% 
%         % Compute divergence (∇·v = du/dx + dv/dy)
%         [du_dx, ~] = gradient(u_current);
%         [~, dv_dy] = gradient(v_current);
%         divergence = du_dx + dv_dy;
% 
%         % Apply the mask
%         divergence(~mask) = NaN;
% 
%         % Smooth the divergence using Gaussian filter
%         valid_mask = ~isnan(divergence);
%         smoothed_divergence = divergence;
% 
%         if any(valid_mask(:))
%             % Replace NaNs temporarily for smoothing
%             temp_divergence = divergence;
%             temp_divergence(~valid_mask) = 0;
% 
%             % Apply Gaussian smoothing
%             smoothed_temp = imgaussfilt(temp_divergence, sigma);
% 
%             % Only keep smoothed values within the valid region
%             smoothed_divergence(valid_mask) = smoothed_temp(valid_mask);
%         end
% 
%         % Store the result
%         divergence_tensor(:,:,t) = smoothed_divergence;
%     end
% 
%     % ======== TRACKING IMPLEMENTATION ========
%     % Structure to hold information about tracked features
%     source_tracks = struct();
%     sink_tracks = struct();
% 
%     % Parameters for tracking
%     max_tracking_distance = 10; % Maximum distance a feature can move between frames
%     min_track_length = 5;      % Minimum number of frames a feature must exist to be considered valid
% 
%     % Initialize track counters
%     next_source_id = 1;
%     next_sink_id = 1;
% 
%     % First, detect all sources and sinks across all frames
%     all_sources = cell(time, 1);
%     all_sinks = cell(time, 1);
% 
%     for t = 1:time
%         % Temporal smoothing for better detection
%         start_frame = max(1, t - floor(temporal_window / 2));
%         end_frame = min(time, t + floor(temporal_window / 2));
%         divergence_avg = mean(divergence_tensor(:,:,start_frame:end_frame), 3, 'omitnan');
% 
%         % Detect sources and sinks by thresholding divergence
%         pos_div_threshold = prctile(divergence_threshold(:),80);
%         neg_div_threshold = prctile(divergence_threshold(:),20);
%         sources = divergence_avg > pos_div_threshold;
%         sinks = divergence_avg < -neg_div_threshold;
% 
% 
%         %sources = divergence_avg > divergence_threshold;
%         %sinks = divergence_avg < -divergence_threshold;
% 
%         % Apply the mask to eliminate false detections outside ROI
%         sources = sources & mask;
%         sinks = sinks & mask;
% 
%         % Label connected components
%         [source_labels, num_sources] = bwlabel(sources);
%         [sink_labels, num_sinks] = bwlabel(sinks);
% 
%         % Get properties including centroids and areas
%         source_props = regionprops(source_labels, 'Centroid', 'Area', 'PixelIdxList');
%         sink_props = regionprops(sink_labels, 'Centroid', 'Area', 'PixelIdxList');
% 
%         % Filter out small features (likely noise)
%         min_area = 3; % Minimum number of pixels
%         source_props = source_props([source_props.Area] >= min_area);
%         sink_props = sink_props([sink_props.Area] >= min_area);
% 
%         % Store for tracking
%         all_sources{t} = source_props;
%         all_sinks{t} = sink_props;
%     end
% 
%     % ======== TRACK SOURCES ACROSS FRAMES ========
%     active_source_tracks = []; % IDs of currently active tracks
% 
%     for t = 1:time
%         sources_t = all_sources{t};
% 
%         % No sources in this frame
%         if isempty(sources_t)
%             continue;
%         end
% 
%         % Get centroids of current sources
%         current_centroids = cat(1, sources_t.Centroid);
% 
%         % Match with existing tracks if any are active
%         if ~isempty(active_source_tracks)
%             % Get predicted positions of active tracks
%             predicted_positions = zeros(length(active_source_tracks), 2);
% 
%             for i = 1:length(active_source_tracks)
%                 track_id = active_source_tracks(i);
%                 track = source_tracks(track_id);
% 
%                 % Use last position if no velocity info available yet
%                 if length(track.frames) == 1
%                     predicted_positions(i,:) = track.positions(end,:);
%                 else
%                     % Use simple linear prediction
%                     last_pos = track.positions(end,:);
%                     prev_pos = track.positions(end-1,:);
%                     velocity = last_pos - prev_pos;
%                     predicted_positions(i,:) = last_pos + velocity;
%                 end
%             end
% 
%             % Calculate distance matrix between predicted positions and current centroids
%             n_tracks = length(active_source_tracks);
%             n_sources = length(sources_t);
%             dist_matrix = zeros(n_tracks, n_sources);
% 
%             for i = 1:n_tracks
%                 for j = 1:n_sources
%                     dist_matrix(i,j) = norm(predicted_positions(i,:) - current_centroids(j,:));
%                 end
%             end
% 
%             % Find optimal assignment
%             [assignment, ~] = assignmentoptimal(dist_matrix);
% 
%             % Update existing tracks or start new ones
%             matched_sources = false(n_sources, 1);
% 
%             for i = 1:n_tracks
%                 track_id = active_source_tracks(i);
% 
%                 if assignment(i) > 0 && dist_matrix(i, assignment(i)) <= max_tracking_distance
%                     % Update existing track
%                     source_idx = assignment(i);
%                     matched_sources(source_idx) = true;
% 
%                     % Add new position to track
%                     source_tracks(track_id).frames(end+1) = t;
%                     source_tracks(track_id).positions(end+1,:) = current_centroids(source_idx,:);
%                     source_tracks(track_id).areas(end+1) = sources_t(source_idx).Area;
%                     source_tracks(track_id).last_seen = t;
%                 else
%                     % Mark track as inactive if not matched
%                     source_tracks(track_id).active = false;
%                 end
%             end
% 
%             % Start new tracks for unmatched sources
%             for i = 1:n_sources
%                 if ~matched_sources(i)
%                     source_tracks(next_source_id).id = next_source_id;
%                     source_tracks(next_source_id).frames = t;
%                     source_tracks(next_source_id).positions = current_centroids(i,:);
%                     source_tracks(next_source_id).areas = sources_t(i).Area;
%                     source_tracks(next_source_id).active = true;
%                     source_tracks(next_source_id).last_seen = t;
% 
%                     active_source_tracks(end+1) = next_source_id;
%                     next_source_id = next_source_id + 1;
%                 end
%             end
%         else
%             % First frame or no active tracks, create new tracks for all sources
%             for i = 1:length(sources_t)
%                 source_tracks(next_source_id).id = next_source_id;
%                 source_tracks(next_source_id).frames = t;
%                 source_tracks(next_source_id).positions = sources_t(i).Centroid;
%                 source_tracks(next_source_id).areas = sources_t(i).Area;
%                 source_tracks(next_source_id).active = true;
%                 source_tracks(next_source_id).last_seen = t;
% 
%                 active_source_tracks(end+1) = next_source_id;
%                 next_source_id = next_source_id + 1;
%             end
%         end
% 
%         % Remove tracks that haven't been seen for a while
%         track_age_threshold = 5; % Maximum frames a track can be inactive
%         still_active = [];
% 
%         for i = 1:length(active_source_tracks)
%             track_id = active_source_tracks(i);
%             if source_tracks(track_id).active && (t - source_tracks(track_id).last_seen <= track_age_threshold)
%                 still_active(end+1) = track_id;
%             end
%         end
% 
%         active_source_tracks = still_active;
%     end
% 
%     % ======== SIMILAR TRACKING FOR SINKS ========
%     % (Code would be nearly identical to source tracking, just using sink data)
%     active_sink_tracks = [];
% 
%     for t = 1:time
%         sinks_t = all_sinks{t};
% 
%         % Skip if no sinks in this frame
%         if isempty(sinks_t)
%             continue;
%         end
% 
%         % Get centroids of current sinks
%         current_centroids = cat(1, sinks_t.Centroid);
% 
%         % Match with existing tracks
%         if ~isempty(active_sink_tracks)
%             % (Similar matching logic as for sources)
%             % Code omitted for brevity but would be nearly identical
%             % to the source tracking section above
%         else
%             % First frame or no active tracks, create new tracks for all sinks
%             for i = 1:length(sinks_t)
%                 sink_tracks(next_sink_id).id = next_sink_id;
%                 sink_tracks(next_sink_id).frames = t;
%                 sink_tracks(next_sink_id).positions = sinks_t(i).Centroid;
%                 sink_tracks(next_sink_id).areas = sinks_t(i).Area;
%                 sink_tracks(next_sink_id).active = true;
%                 sink_tracks(next_sink_id).last_seen = t;
% 
%                 active_sink_tracks(end+1) = next_sink_id;
%                 next_sink_id = next_sink_id + 1;
%             end
%         end
% 
%         % (Similar track maintenance as for sources)
%     end
% 
%     % Filter out short tracks
%     valid_source_tracks = [];
%     for i = 1:length(source_tracks)
%         if length(source_tracks(i).frames) >= min_track_length
%             valid_source_tracks(end+1) = i;
%         end
%     end
% 
%     valid_sink_tracks = [];
%     for i = 1:length(sink_tracks)
%         if length(sink_tracks(i).frames) >= min_track_length
%             valid_sink_tracks(end+1) = i;
%         end
%     end
% 
%     % ======== VISUALIZE TRACKING RESULTS ========
%     % Generate random but consistent colors for each track
%     rng(0); % For reproducible colors
%     source_colors = hsv(length(valid_source_tracks));
%     sink_colors = hsv(length(valid_sink_tracks));
% 
%     % Store tracked positions in original format for compatibility
%     tracked_positions = struct('sources', cell(time, 1), 'sinks', cell(time, 1));
% 
%     % Loop through each frame to create video
%     for t = 1:time
%         % Temporal smoothing
%         start_frame = max(1, t - floor(temporal_window / 2));
%         end_frame = min(time, t + floor(temporal_window / 2));
%         divergence_avg = mean(divergence_tensor(:,:,start_frame:end_frame), 3, 'omitnan');
% 
%         % Clear axes before drawing
%         clf;
%         hold on;
%         title('Tracked Sources and Sinks in Divergence Contour Plot');
% 
%         % Display the current frame as an image
%         imagesc(tensor(:,:,t), [bottom_prc, top_prc]);
%         colormap jet;
%         colorbar;
% 
%         % Explicitly set the axis limits to show the full data
%         ax = gca;
%         ax.XLim = [1 width];
%         ax.YLim = [1 height];
%         ax.YDir = 'reverse';
%         axis tight;
% 
%         % Draw contours for the divergence
%         [C, h] = contour(divergence_avg, 8, 'LineColor', 'k', 'LineWidth', 1);
% 
%         % Collect current positions for compatibility with original code
%         current_sources = [];
%         current_sinks = [];
% 
%         % Plot source tracks
%         for i = 1:length(valid_source_tracks)
%             track_id = valid_source_tracks(i);
%             track = source_tracks(track_id);
% 
%             % Check if this track exists in the current frame
%             frame_indices = find(track.frames == t);
% 
%             if ~isempty(frame_indices)
%                 % Current position
%                 current_pos = track.positions(frame_indices,:);
% 
%                 % Add to compatibility structure
%                 current_sources(end+1,:) = current_pos;
% 
%                 % Plot current position with ID
%                 plot(current_pos(1), current_pos(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', source_colors(i,:));
%                 text(current_pos(1) + 2, current_pos(2) + 2, num2str(track_id), 'Color', source_colors(i,:), 'FontWeight', 'bold');
% 
%                 % Plot track history (trail)
%                 history_start = max(1, frame_indices - 10); % Show last 10 frames
%                 if history_start < frame_indices
%                     history_frames = track.frames(history_start:frame_indices);
%                     history_pos = track.positions(history_start:frame_indices,:);
%                     plot(history_pos(:,1), history_pos(:,2), '-', 'Color', source_colors(i,:), 'LineWidth', 1.5);
%                 end
%             end
%         end
% 
%         % Plot sink tracks (similar to source tracks)
%         for i = 1:length(valid_sink_tracks)
%             track_id = valid_sink_tracks(i);
%             track = sink_tracks(track_id);
% 
%             % Check if this track exists in the current frame
%             frame_indices = find(track.frames == t);
% 
%             if ~isempty(frame_indices)
%                 % Current position
%                 current_pos = track.positions(frame_indices,:);
% 
%                 % Add to compatibility structure
%                 current_sinks(end+1,:) = current_pos;
% 
%                 % Plot current position with ID
%                 plot(current_pos(1), current_pos(2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', sink_colors(i,:));
%                 text(current_pos(1) + 2, current_pos(2) + 2, num2str(track_id), 'Color', sink_colors(i,:), 'FontWeight', 'bold');
% 
%                 % Plot track history (trail)
%                 history_start = max(1, frame_indices - 10); % Show last 10 frames
%                 if history_start < frame_indices
%                     history_frames = track.frames(history_start:frame_indices);
%                     history_pos = track.positions(history_start:frame_indices,:);
%                     plot(history_pos(:,1), history_pos(:,2), '-', 'Color', sink_colors(i,:), 'LineWidth', 1.5);
%                 end
%             end
%         end
% 
%         % Store in compatibility structure
%         tracked_positions(t).sources = current_sources;
%         tracked_positions(t).sinks = current_sinks;
% 
%         % Text showing frame number
%         text(5, 15, ['Frame: ' num2str(t)], 'Color', 'white', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);
% 
%         % Add legend
%         h_src = plot(NaN, NaN, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
%         h_snk = plot(NaN, NaN, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
%         legend([h_src, h_snk], {'Source (+ div)', 'Sink (- div)'}, 'Location', 'northeast');
% 
%         % Capture the frame for the video
%         frame = getframe(fig);
%         writeVideo(video_writer, frame);
%     end
% 
%     % Close the video writer and figure
%     close(video_writer);
%     close(fig);
% end
% 
% function [assignment, cost] = assignmentoptimal(distMatrix)
%     % A simplified implementation of the Hungarian algorithm for assignment optimization
%     % For a complete implementation, you might want to use assignmentoptimal from 
%     % the Matlab File Exchange or MATLAB's built-in matchpairs function
% 
%     % Initialize
%     [nOfRows, nOfColumns] = size(distMatrix);
% 
%     % Special cases
%     if nOfRows == 0 || nOfColumns == 0
%         assignment = [];
%         cost = 0;
%         return;
%     end
% 
%     % Ensure the problem is balanced
%     if nOfRows > nOfColumns
%         distMatrix(nOfRows, nOfColumns) = 0;
%     elseif nOfRows < nOfColumns
%         distMatrix(nOfColumns, nOfColumns) = 0;
%     end
% 
%     % Update dimensions
%     [nOfRows, nOfColumns] = size(distMatrix);
% 
%     % Greedy assignment (suboptimal but simple)
%     assignment = zeros(nOfRows, 1);
%     cost = 0;
% 
%     % For each row, find the minimum element and assign it
%     for row = 1:nOfRows
%         [minVal, minCol] = min(distMatrix(row, :));
% 
%         % Check if this column is already assigned
%         if ~any(assignment == minCol)
%             assignment(row) = minCol;
%             cost = cost + minVal;
%         else
%             % Find the next best column
%             tempDist = distMatrix(row, :);
%             tempDist(minCol) = Inf;
% 
%             while true
%                 [minVal, minCol] = min(tempDist);
% 
%                 if ~any(assignment == minCol) || minVal == Inf
%                     break;
%                 end
% 
%                 tempDist(minCol) = Inf;
%             end
% 
%             if minVal < Inf
%                 assignment(row) = minCol;
%                 cost = cost + minVal;
%             else
%                 assignment(row) = 0; % No assignment possible
%             end
%         end
%     end
% end

%%
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
