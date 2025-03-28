%% Main code to run optical flow analysis

% support code:
% plot_quiver_video.m
% ind2sub_brain_mask.m
% horn_schunck.m
% save_tracked_contour_video.m

%% Load and process the data

% load the mask 
% note: this is just the 2D mask containing 1s and 0s extracted from
% BrainMaskStruct
load('/Users/elisachinigo/Documents/Widefield_Data/brainMaskStruct.mat');
mask = brainMaskStruct.WarpData.brainMask;
% load the video data
% note: because this is has linear arrangement of pixels we need to convert
% to a 3D tensor using sub_2ind
load('/Users/elisachinigo/Documents/Widefield_Data/mouse47/Conv/9102020/Processedvid4/NREM_WF_2.mat');

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

%% 

% some of the pixels are bad as they have super large signal, we need to
% filter these out. To do this let's plot the average intensity of each
% pixel to figure out an appropriate threshold to eliminate bad pixels

% Step 2: compute average intensity per pixel over time
avg_values = mean(video_3d,3);

% Step 3: Plot the average values taken by each pixel over time as an image
figure;
imagesc(avg_values);
colorbar;
title("Average Intensity per Pixel Over Time");
xlabel("X Pixel");
ylabel("Y Pixel");
axis image;
colormap jet;

% note: for mouse 47 this is a group of approx 10 pixels on the medial
% boundary

% Step 4: Plot the histogram of average values
figure;
histogram(avg_values(:), 'Binwidth', 0.00001); % flatten to 1D for histogram
title('Histogram of average Pixel intensities');
xlabel('Average Intensity');
ylabel('Frequency');
grid on;

% Step 5: Filter out high and low brightness pixels

high_threshold = 0.00003;
low_threshold = -0.0002;
high_intensity_mask = avg_values > high_threshold;
low_intensity_mask = avg_values < low_threshold;

% Create combined mask that includes both the brain mask and the intensity filters
combined_mask = mask & ~high_intensity_mask & ~low_intensity_mask;

% Step 6: Apply the filtering to all time points (vectorized for speed)
video_3d = bsxfun(@times, video_3d, combined_mask);

% Step 7: Revisualize the averages after filtering to make sure we've
% removed all the correct pixels

% Visualize after filtering
figure;
imagesc(mean(video_3d, 3));
colorbar;
title('Average Intensity After Filtering High Values')
axis image;
colormap jet;


%% RUN OPTICAL FLOW ANALYSIS

% NOTE: the function takes 2 parameters:
% - alpha: regularization parameter (larger means more smoothing typically 0.1-100)
% - num_iterations: more iterations mean more refined solutions (tradeoff,
% also takes longer to run)

alpha = 10;
num_iterations = 200;
output_video_name = 'optical_flow_video_WF_2.mp4'; % change to whatever you want this, currently saves to current directory

% let's amplilfy the data and make it positive 
shifted_data = video_3d*100 + abs(-2);

% now run the vector flow analysis, this produces a quiver plot and saves
% it as an mp4 file
% note, it takes a while to run, to check things are working properly I
% recommend running for maybe 100 frames before running for the whole video
[u_tensor, v_tensor]=plot_quiver_video(shifted_data(:,:,:),alpha, num_iterations, output_video_name,mask);

%% source and sink calculations

% NOTE: this function takes the following parameters:
% - sigma: std of Gaussian smoothing. Larger values = more smoothing
% - temporal_window: number of frames it smooths over. 
% - divergence_threshold: threshold for source/sink (the smaller the
% threshold the greater the number of sources/sinks it finds)
% - n_contours: number of contours to plot (this is just for visualization
% doesnt affect calculations)

sigma = 20;
temporal_window = 2;
divergence_threshold = 0.005;
n_contours = 10;

% run the source/sink plots and save the video as an mp4 file
[divergence_tensor,tracked_positions] = save_tracked_contour_video(u_tensor(:,:,:), v_tensor(:,:,:), shifted_data(:,:,:), 'source_sink_video.mp4', sigma, temporal_window, divergence_threshold, n_contours,mask);

