function [u_tensor,v_tensor] =plot_quiver_video(tensor, alpha, num_iterations, output_video_name,mask)
% note: this function averages gradients twice (might be excessive)
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




