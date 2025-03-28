% Modified vector_flow_plots function to handle 2D data (pixels × time)
% where pixels are arranged linearly rather than in a spatial grid
% The flow calculation is now restricted to only the brain mask region

% Load the mask
load('/Users/elisachinigo/Documents/Widefield_Data/brainMaskStruct.mat'); %% change this for the MASK.mat file on dropbox
% Load your 2D data (pixels × time)
load('/Users/elisachinigo/Documents/Widefield_Data/Data_videos/video.mat'); %% CHANG E THIS TO 


% Get dimensions from the mask
[height, width] = size(mask);
num_pixels = sum(mask(:)); % Count of non-zero elements in mask
num_frames = size(NREM_widefield.data, 2); % Number of time frames

% Step 1: Reshape the 2D data into 3D using the mask
video_3d = zeros(height, width, num_frames);

% For each time frame, map the linear pixel values to their 2D coordinates using the mask
for t = 1:num_frames
    % Get the data for the current frame
    frame_data = NREM_widefield.data(:, t);
    
    % Create a temporary frame
    temp_frame = zeros(height, width);
    
    % Loop through each masked pixel and place it in the correct 2D position
    for i = 1:num_pixels
        % Convert linear index in masked space to 2D coordinates
        [row, col] = ind2sub_brain_mask(i, brainMaskStruct);
        
        % Place the pixel value in the correct position
        temp_frame(row, col) = frame_data(i);
    end
    
    % Assign the reconstructed frame to the 3D video
    video_3d(:, :, t) = temp_frame;
end

% Now continue with your original code, but using video_3d instead of video

% Step 2: Compute the average intensity for each pixel over time
avg_values = mean(video_3d, 3);

% Step 3: Plot the average values taken by each pixel over time as an image
figure;
imagesc(avg_values);
colorbar;
title("Average Intensity per Pixel Over Time");
xlabel("X Pixel");
ylabel("Y Pixel");
axis image;
colormap jet;
%% 

% Step 4: Plot the histogram of average values
figure;
histogram(avg_values(:), 'Binwidth', 0.0001); % flatten to 1D for histogram
title('Histogram of average Pixel intensities');
xlabel('Average Intensity');
ylabel('Frequency');
grid on;

% Step 5: Filter out high and low brightness pixels
high_threshold = 0.0003;
low_threshold = -0.0003;
high_intensity_mask = avg_values > high_threshold;
low_intensity_mask = avg_values < low_threshold;

% Create combined mask that includes both the brain mask and the intensity filters
combined_mask = mask & ~high_intensity_mask & ~low_intensity_mask;

% Step 6: Apply the filtering to all time points (vectorized for speed)
video_3d = bsxfun(@times, video_3d, combined_mask);

%% 

% Visualize after filtering
figure;
imagesc(mean(video_3d, 3));
colorbar;
title('Average Intensity After Filtering High Values')
axis image;
colormap jet;

%% Run optical flow analysisalpha = 1;
num_iterations = 50;
output_video_name = 'optical_flow_video_full.mp4';

shifted_data = video_3d(mask)*10 +2; % amplify and shift
% Get dimensions of the original data
[height, width, num_frames] = size(video_3d);

% Create a copy of the data for manipulation
shifted_data = zeros(height, width, num_frames);

% Properly amplify only the brain mask pixels while maintaining 3D structure
for t = 1:num_frames
    frame = video_3d(:,:,t);
    
    % Create a new frame with amplified values only within the mask
    new_frame = zeros(height, width);
    new_frame(mask) = frame(mask)*10 + 2;
    
    % Store the new frame in the 3D array
    shifted_data(:,:,t) = new_frame;
end
% For display consistency
global_min = min(shifted_data(:), [], 'omitnan'); % Get min value ignoring NaNs
global_max = max(shifted_data(:), [], 'omitnan'); % Get max value ignoring NaNs
global_average = mean(shifted_data(:), 'omitnan');
disp(global_min);
disp(global_max);
disp(global_average);

figure;
histogram(shifted_data);
[u_tensor, v_tensor] = plot_quiver_video(shifted_data, alpha, num_iterations, output_video_name, mask);

%% Now look at the divergence and source/sink analysis 
[divergence_tensor, tracked_positions] = save_tracked_contour_video(u_tensor, v_tensor, shifted_data, 'source_sink_video.mp4', 10, 3, 0.03, mask);

%% Analyze the divergence
abs_divergence = abs(divergence_tensor(:));
threshold_guess = prctile(abs_divergence, 95);
figure;
histogram(divergence_tensor(:), 'Normalization', 'probability');
xlabel('Divergence Value');
ylabel('Probability');
title('Divergence Value Distribution');

%% Functions for flow field plotting
function [u_tensor, v_tensor] = plot_quiver_video(tensor, alpha, num_iterations, output_video_name, mask)
    % plot_quiver_video: Computes and visualizes optical flow vector fields
    % with special attention to ensure edge coverage
    
    % Display diagnostic information
    disp('Starting plot_quiver_video function');
    disp(['Input tensor size: ' num2str(size(tensor))]);
    disp(['Mask size: ' num2str(size(mask))]);
    
    % Get the size of the tensor
    [height, width, time] = size(tensor);
    
    % Precompute the flow fields for all frames
    u_tensor = zeros(height, width, time-1);
    v_tensor = zeros(height, width, time-1);
    
    disp('Computing optical flow between frames...');
    % Compute flow for each consecutive pair of frames
    for t = 1:time-1
        % Display progress
        if mod(t, 100) == 0
            disp(['Processing frame ' num2str(t) ' of ' num2str(time-1)]);
        end
        tensor(isnan(tensor))=0;
        % Replace NaNs with 0
        current_frame = tensor(:,:,t);
        next_frame = tensor(:,:,t+1);
        
        % Set non-mask regions to 0 (not NaN) to prevent edge effects
        mask_3D = repmat(mask,[1,1,size(u_tensor,3)]);
        disp(size(mask_3D))
        disp(size(u_tensor))
        %current_frame_masked = current_frame .* double(mask);
        %next_frame_masked = next_frame .* double(mask);
        
        % Compute the optical flow between consecutive frames
        %[u, v] = horn_schunck_modified(current_frame_masked, next_frame_masked, alpha, num_iterations, mask);
       

        % Store the flow components in the tensors
        u_tensor(:,:,t) = u;
        v_tensor(:,:,t) = v;
    end
    
    disp('Flow computation complete.');
    
    disp('Creating video...');
    
    % Initialize the video writer with a fixed frame size
    try
        v = VideoWriter(output_video_name, 'MPEG-4');
        v.FrameRate = 10; % Adjust frame rate as desired
        open(v);
        
        % Set up the figure for plotting
        fig = figure('Position', [100, 100, 800, 600]); % Larger figure for better visibility
        
        % CRITICAL IMPROVEMENT: Create a specialized sampling for better edge coverage
        
        % First, get all mask points
        [mask_y, mask_x] = find(mask);
        
        % Find the boundary pixels of the mask
        boundary_mask = mask - imerode(mask, strel('disk', 2));
        [bound_y, bound_x] = find(boundary_mask);
        
        % Ensure we have a good number of edge points
        edge_step = max(1, round(length(bound_y)/100)); % Ensure we have around 100 edge points
        edge_idx = 1:edge_step:length(bound_y);
        edge_x = bound_x(edge_idx);
        edge_y = bound_y(edge_idx);
        
        % Sample interior points less densely
        interior_mask = imerode(mask, strel('disk', 3)); % Interior area
        [int_y, int_x] = find(interior_mask);
        
        % Sample interior points with larger step size
        interior_step = max(1, round(length(int_y)/200)); % Around 200 interior points
        int_idx = 1:interior_step:length(int_y);
        int_x = int_x(int_idx);
        int_y = int_y(int_idx);
        
        % Combine edge and interior points
        X_points = [edge_x; int_x];
        Y_points = [edge_y; int_y];
        
        % Display how many points we're using
        disp(['Number of quiver points: ' num2str(length(X_points)) ' (Edge: ' num2str(length(edge_x)) ...
              ', Interior: ' num2str(length(int_x)) ')']);
        
        % Pre-allocate arrays for u and v values at each point
        u_points = zeros(length(X_points), time-1);
        v_points = zeros(length(X_points), time-1);
        
        % Extract u and v values at each sampled point
        for t = 1:time-1
            for i = 1:length(X_points)
                x = X_points(i);
                y = Y_points(i);
                
                % Get flow values at this point
                if y <= size(u_tensor, 1) && x <= size(u_tensor, 2)
                    u_points(i, t) = u_tensor(y, x, t);
                    v_points(i, t) = v_tensor(y, x, t);
                else
                    u_points(i, t) = 0;
                    v_points(i, t) = 0;
                end
            end
        end
        
        % For display consistency
        global_min = min(tensor(:), [], 'omitnan'); % Get min value ignoring NaNs
        global_max = max(tensor(:), [], 'omitnan'); % Get max value ignoring NaNs
        
        disp(['Creating video frames (total: ' num2str(time-1) ')...']);
        
        % Loop through each time slice and update the quiver plot
        for t = 1:time-1
            % Display progress
            if mod(t, 10) == 0
                disp(['Creating frame ' num2str(t) ' of ' num2str(time-1)]);
            end
            
            % Create a two-panel plot
            subplot(1,2,1);
            
            % Display the current frame data
            imagesc(tensor(:,:,t), [0, 3]);
            colormap(jet);
            colorbar;
            title(['Frame ' num2str(t) ' - Original']);
            axis image;
            
            subplot(1,2,2);
            
            % Display the current frame with flow overlay
            imagesc(tensor(:,:,t), [0, 3]);
            hold on;
            
            % Set axis properties
            axis([1 width 1 height]);
            set(gca, 'YDir', 'reverse');
            axis equal tight;
            colormap(jet);
            colorbar;
            title(['Frame ' num2str(t) ' - With Flow Vectors']);
            
            % Plot quiver with error handling
            try
                % Scale the quiver arrows for better visualization
                u_curr = u_points(:, t);
                v_curr = v_points(:, t);
                
                % Detect NaN or extreme values that might cause plotting issues
                valid = ~isnan(u_curr) & ~isnan(v_curr) & ...
                        abs(u_curr) < 10 & abs(v_curr) < 10;
                                
                if sum(valid) > 0
                    % Plot vectors with appropriate scaling
                    quiver(X_points(valid), Y_points(valid), ...
                           u_curr(valid), v_curr(valid), ...
                           1.5, 'Color', 'k', 'LineWidth', 1);
                           
                    % Highlight edge points in a different color for debugging
                    edge_indices = 1:length(edge_x);
                    valid_edges = valid(edge_indices);
                    if sum(valid_edges) > 0
                        quiver(X_points(edge_indices(valid_edges)), Y_points(edge_indices(valid_edges)), ...
                               u_curr(edge_indices(valid_edges)), v_curr(edge_indices(valid_edges)), ...
                               1.5, 'Color', 'r', 'LineWidth', 1);
                    end
                else
                    disp(['No valid vectors for frame ' num2str(t)]);
                end
                
                % Plot the mask boundary for reference
                boundary = bwboundaries(mask);
                for k = 1:length(boundary)
                    b = boundary{k};
                    plot(b(:,2), b(:,1), 'w-', 'LineWidth', 1);
                end
                
            catch e
                disp(['Error plotting quiver for frame ' num2str(t) ': ' e.message]);
            end
            
            % Ensure figure is rendered before capturing
            drawnow;
            
            % Capture the frame for the video
            frame = getframe(fig);
            writeVideo(v, frame);
            
            % Clear for next frame
            hold off;
            clf;
        end
        
        % Close video writer and figure
        close(v);
        close(fig);
        disp(['Video successfully saved as: ' output_video_name]);
        
    catch e
        % Display any errors that occur
        disp('Error in video creation:');
        disp(e.message);
        disp(e.stack(1));
    end
end

function [u, v] = horn_schunck_modified(I1, I2, alpha, num_iterations, mask)
    % Horn-Schunck optical flow with specific modifications to improve edge flow
    
    % Initialize flow components (u and v)
    [height, width] = size(I1);
    u = zeros(height, width);
    v = zeros(height, width);
    
    % Set initial values outside the mask to 0
    u(~mask) = 0;
    v(~mask) = 0;
    
    % Step 1: Compute image gradients with padding to improve edge estimation
    % Apply a small amount of smoothing before gradient computation
    I1_smooth = imgaussfilt(I1, 0.5);
    I2_smooth = imgaussfilt(I2, 0.5);
    
    % Compute gradients with centered difference
    [Ix, Iy] = gradient(I1_smooth);
    It = I2_smooth - I1_smooth;
    
    % Zero out gradients outside the mask
    Ix = Ix .* double(mask);
    Iy = Iy .* double(mask);
    It = It .* double(mask);
    
    % Create a dilated mask for better edge handling
    dilated_mask = imdilate(mask, strel('square', 3));
    
    % Iteratively refine flow estimation
    for iter = 1:num_iterations
        % Compute neighborhood averages using circular shifting
        u_avg = (circshift(u, [0, -1]) + circshift(u, [0, 1]) + ...
                 circshift(u, [-1, 0]) + circshift(u, [1, 0])) / 4;
        v_avg = (circshift(v, [0, -1]) + circshift(v, [0, 1]) + ...
                 circshift(v, [-1, 0]) + circshift(v, [1, 0])) / 4;
        
        % Apply the dilated mask to averages
        u_avg = u_avg .* double(dilated_mask);
        v_avg = v_avg .* double(dilated_mask);
        
        % Update flow using Horn-Schunck equations
        denom = alpha^2 + Ix.^2 + Iy.^2;
        denom(denom < 1e-10) = 1e-10;  % Avoid division by zero
        
        update_term = (Ix .* u_avg + Iy .* v_avg + It) ./ denom;
        u_new = u_avg - Ix .* update_term;
        v_new = v_avg - Iy .* update_term;
        
        % Special handling for boundary pixels - apply stronger regularization
        boundary = mask & ~imerode(mask, strel('disk', 1));
        
        % For boundary pixels, put more weight on neighborhood average
        u_new(boundary) = 0.8 * u_avg(boundary) + 0.2 * u_new(boundary);
        v_new(boundary) = 0.8 * v_avg(boundary) + 0.2 * v_new(boundary);
        
        % Update u and v
        u = u_new .* double(mask);
        v = v_new .* double(mask);
    end
    
    % Set values outside the mask to NaN for final output
    u(~mask) = NaN;
    v(~mask) = NaN;
end

function [u, v] = horn_schunck_masked(I1, I2, alpha, num_iterations, mask)
    % Horn-Schunck optical flow estimation method with masking to restrict calculation
    % to the brain region only - optimized for performance with vectorized operations
    
    % Initialize flow components (u and v)
    [height, width] = size(I1);
    u = zeros(height, width);
    v = zeros(height, width);
    
    % Set initial values outside the mask to 0 (will be masked at end)
    u(~mask) = 0;
    v(~mask) = 0;
    
    % Step 1: Compute image gradients
    [Ix, Iy] = gradient(double(I1));
    It = double(I2) - double(I1);
    
    % Zero out gradients outside the mask
    Ix(~mask) = 0;
    Iy(~mask) = 0;
    It(~mask) = 0;
    
    % Create a dilated mask for safer averaging (to avoid edge effects)
    dilated_mask = imdilate(mask, strel('square', 3));
    
    % Iteratively refine flow estimation
    for iter = 1:num_iterations
        % Use vectorized operations for computing neighborhood averages
        u_avg = (circshift(u, [0, -1]) + circshift(u, [0, 1]) + ...
                 circshift(u, [-1, 0]) + circshift(u, [1, 0])) / 4;
        v_avg = (circshift(v, [0, -1]) + circshift(v, [0, 1]) + ...
                 circshift(v, [-1, 0]) + circshift(v, [1, 0])) / 4;
        
        % Zero out the averages outside the dilated mask to prevent influence from outside
        u_avg(~dilated_mask) = 0;
        v_avg(~dilated_mask) = 0;
        
        % Update flow using Horn-Schunck equations (vectorized)
        denom = alpha^2 + Ix.^2 + Iy.^2;
        
        % Avoid division by zero
        safe_denom = denom;
        safe_denom(denom < 1e-10) = 1e-10;
        
        % Update only within the mask
        update_term = (Ix .* u_avg + Iy .* v_avg + It) ./ safe_denom;
        u = u_avg - Ix .* update_term;
        v = v_avg - Iy .* update_term;
        
        % Apply mask in each iteration to maintain boundaries
        u = u .* mask;
        v = v .* mask;
    end
    
    % Set values outside the mask to NaN for final output
    u(~mask) = NaN;
    v(~mask) = NaN;
end

function [divergence_tensor, tracked_positions] = save_tracked_contour_video(u_tensor, v_tensor, tensor, output_video_name, sigma, temporal_window, divergence_threshold, mask)
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
    fig = figure('Position', [100, 100, 640, 480]);
    hold on;
    title('Tracked Sources and Sinks in Divergence Contour Plot');

    % Calculate global min and max for consistent colormap
    global_min = min(tensor(:), [], 'omitnan');
    global_max = max(tensor(:), [], 'omitnan');

    % Create a slightly eroded mask for safer gradient calculation
    eroded_mask = imerode(mask, strel('disk', 1));
    
    % Precompute divergence for each frame - vectorized for speed
    divergence_tensor = zeros(height, width, time-1);
    disp('Computing divergence for all frames...');
    for t = 1:time-1
        if mod(t, 100) == 0
            disp(['Processing frame ' num2str(t) ' of ' num2str(time-1)]);
        end
        
        % Compute gradients using vectorized operations
        [u_x, ~] = gradient(u_tensor(:,:,t));
        [~, v_y] = gradient(v_tensor(:,:,t));
        
        % Combine gradients to get divergence
        divergence = u_x + v_y;
        
        % Apply the mask
        divergence = divergence .* eroded_mask;
        divergence(~eroded_mask) = NaN;
        
        % Apply Gaussian smoothing efficiently
        valid_div = divergence;
        valid_div(isnan(valid_div)) = 0;
        smoothed_div = imgaussfilt(valid_div, sigma);
        smoothed_div(~mask) = NaN;
        
        % Store in the divergence tensor
        divergence_tensor(:,:,t) = smoothed_div;
    end
    
    % Preallocate for speed
    tracked_positions(time-1) = struct('sources', [], 'sinks', []);
    
    disp('Creating source/sink video...');
    % Loop through each time frame to detect and track sources and sinks
    for t = 1:time-1
        if mod(t, 10) == 0
            disp(['Creating frame ' num2str(t) ' of ' num2str(time-1)]);
        end
        
        % Temporal smoothing - use efficient slice indexing
        start_frame = max(1, t - floor(temporal_window / 2));
        end_frame = min(time-1, t + floor(temporal_window / 2));
        divergence_avg = mean(divergence_tensor(:,:,start_frame:end_frame), 3, 'omitnan');

        % Display the current frame as an image
        imagesc(tensor(:,:,t), [global_min, global_max]);
        colormap jet;
        colorbar;
        axis([1 width 1 height]);
        set(gca, 'YDir', 'reverse');
        axis equal;
        axis tight;

        % Vectorized thresholding for speed
        sources = divergence_avg > divergence_threshold & mask;
        sinks = divergence_avg < -divergence_threshold & mask;

        % Label connected components in sources and sinks
        [source_labels, ~] = bwlabel(sources);
        [sink_labels, ~] = bwlabel(sinks);

        % Calculate centroids for each source and sink
        source_centroids = regionprops(source_labels, 'Centroid');
        sink_centroids = regionprops(sink_labels, 'Centroid');
        
        % Store centroids for tracking - use direct assignment for speed
        if ~isempty(source_centroids)
            tracked_positions(t).sources = cat(1, source_centroids.Centroid);
        else
            tracked_positions(t).sources = [];
        end
        
        if ~isempty(sink_centroids)
            tracked_positions(t).sinks = cat(1, sink_centroids.Centroid);
        else
            tracked_positions(t).sinks = [];
        end

        % Plot contours and mark sources and sinks
        hold on;
        
        % Create a temporary masked version for contour plotting
        masked_div = divergence_avg;
        masked_div(~mask) = NaN;
        contour(masked_div, 5, 'LineColor', 'k', 'LineWidth', 1);

        % Plot source and sink centroids
        if ~isempty(tracked_positions(t).sources)
            plot(tracked_positions(t).sources(:,1), tracked_positions(t).sources(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
        if ~isempty(tracked_positions(t).sinks)
            plot(tracked_positions(t).sinks(:,1), tracked_positions(t).sinks(:,2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
        end

        % Capture the frame for the video
        frame = getframe(fig);
        writeVideo(video_writer, frame);

        % Clear for next frame
        cla;
    end

    % Close the video writer and figure
    close(video_writer);
    close(fig);
end