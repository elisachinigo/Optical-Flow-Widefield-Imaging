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