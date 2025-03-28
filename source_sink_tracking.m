% Create the figure with two subplots
figure;

% Plot sources
subplot(1, 2, 1);
set(gca, 'YDir', 'reverse');
hold on;
for i = 1:length(tracked_positions)
    source = tracked_positions(i).sources;
    if ~isempty(source)
        scatter(source(:,1), source(:,2), 'filled', 'b');
    end
end
title('Sources');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
axis equal;

% Plot sinks
subplot(1, 2, 2);
set(gca, 'YDir', 'reverse');
hold on;
for i = 1:length(tracked_positions)
    sink = tracked_positions(i).sinks;
    if ~isempty(sink)
        scatter(sink(:,1), sink(:,2), 'filled', 'r');
    end
end
title('Sinks');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
axis equal;

% Add a main title
sgtitle('Source and Sink Points from Video Frames');

% Optional: Adjust the figure size
set(gcf, 'Position', [100, 100, 1000, 500]);
saveas(gcf, 'source_sink_scatters.pdf');


%% 

% Create a figure with two subplots for heatmaps
figure;

% Define grid dimensions for the heatmap
xEdges = linspace(0, 130, 50);  % Adjust range based on your data
yEdges = linspace(0, 200, 50);  % Adjust range based on your data

% Collect all source points
all_sources_x = [];
all_sources_y = [];
for i = 1:length(tracked_positions)
    source = tracked_positions(i).sources;
    if ~isempty(source)
        all_sources_x = [all_sources_x; source(:,1)];
        all_sources_y = [all_sources_y; source(:,2)];
    end
end

% Collect all sink points
all_sinks_x = [];
all_sinks_y = [];
for i = 1:length(tracked_positions)
    sink = tracked_positions(i).sinks;
    if ~isempty(sink)
        all_sinks_x = [all_sinks_x; sink(:,1)];
        all_sinks_y = [all_sinks_y; sink(:,2)];
    end
end

% Create heatmap for sources
subplot(1, 2, 1);
set(gca, 'YDir', 'reverse');
if ~isempty(all_sources_x)
    % Create 2D histogram
    [N_sources, ~, ~] = histcounts2(all_sources_x, all_sources_y, xEdges, yEdges);
    % Display as heatmap with proper orientation
    imagesc(xEdges, yEdges, N_sources');
    set(gca, 'YDir', 'reverse');  % Ensure Y-axis is pointing upward
    colormap(hot);
    colorbar;
    title('Source Points Heatmap');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    axis equal tight;
else
    title('No Source Points Found');
    axis([min(xEdges) max(xEdges) min(yEdges) max(yEdges)]);
end

% Create heatmap for sinks
subplot(1, 2, 2);
set(gca, 'YDir', 'reverse');
if ~isempty(all_sinks_x)
    % Create 2D histogram
    [N_sinks, ~, ~] = histcounts2(all_sinks_x, all_sinks_y, xEdges, yEdges);
    % Display as heatmap with proper orientation
    imagesc(xEdges, yEdges, N_sinks');
    set(gca, 'YDir', 'reverse');  % Ensure Y-axis is pointing upward
    colormap(hot);
    colorbar;
    title('Sink Points Heatmap');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    axis equal tight;
else
    title('No Sink Points Found');
    axis([min(xEdges) max(xEdges) min(yEdges) max(yEdges)]);
end

% Add a main title
sgtitle('Spatial Distribution of Source and Sink Points');

% Optional: Adjust the figure size
set(gcf, 'Position', [100, 100, 1000, 500]);
saveas(gcf, 'source_sink_density_coarse.pdf');

%% % Alternative: Use kernel density estimation for smoother results
% Create a figure with two subplots for heatmaps
figure;

% Collect all source points
all_sources_x = [];
all_sources_y = [];
for i = 1:length(tracked_positions)
    source = tracked_positions(i).sources;
    if ~isempty(source)
        all_sources_x = [all_sources_x; source(:,1)];
        all_sources_y = [all_sources_y; source(:,2)];
    end
end

% Collect all sink points
all_sinks_x = [];
all_sinks_y = [];
for i = 1:length(tracked_positions)
    sink = tracked_positions(i).sinks;
    if ~isempty(sink)
        all_sinks_x = [all_sinks_x; sink(:,1)];
        all_sinks_y = [all_sinks_y; sink(:,2)];
    end
end

% Create KDE heatmap for sources
subplot(1, 2, 1);
if ~isempty(all_sources_x)
    % Define grid (adjust min/max if needed)
    x_min = min(all_sources_x) - 5;
    x_max = max(all_sources_x) + 5;
    y_min = min(all_sources_y) - 5;
    y_max = max(all_sources_y) + 5;
    
    [X_sources, Y_sources] = meshgrid(linspace(x_min, x_max, 200), linspace(y_min, y_max, 200));
    
    % Compute KDE
    Z_sources = zeros(size(X_sources));
    bandwidth = 3;  % Adjust for desired smoothness
    
    for i = 1:length(all_sources_x)
        Z_sources = Z_sources + exp(-((X_sources - all_sources_x(i)).^2 + (Y_sources - all_sources_y(i)).^2) / (2*bandwidth^2));
    end
    
    % Normalize and plot
    Z_sources = Z_sources / sum(Z_sources(:));
    
    contourf(X_sources, Y_sources, Z_sources, 30, 'LineStyle', 'none');
    hold on;
    scatter(all_sources_x, all_sources_y, 10, 'w', 'filled', 'MarkerFaceAlpha', 0.3);
    colormap(hot);
    colorbar;
    title('Source Points Density');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    axis equal tight;
    set(gca, 'YDir', 'reverse');
else
    title('No Source Points Found');
    axis([0 100 0 180]);  % Default range if no points
end

% Create KDE heatmap for sinks
subplot(1, 2, 2);

if ~isempty(all_sinks_x)
    % Define grid (adjust min/max if needed)
    x_min = min(all_sinks_x) - 5;
    x_max = max(all_sinks_x) + 5;
    y_min = min(all_sinks_y) - 5;
    y_max = max(all_sinks_y) + 5;
    
    [X_sinks, Y_sinks] = meshgrid(linspace(x_min, x_max, 200), linspace(y_min, y_max, 200));
    
    % Compute KDE
    Z_sinks = zeros(size(X_sinks));
    bandwidth = 3;  % Adjust for desired smoothness
    
    for i = 1:length(all_sinks_x)
        Z_sinks = Z_sinks + exp(-((X_sinks - all_sinks_x(i)).^2 + (Y_sinks - all_sinks_y(i)).^2) / (2*bandwidth^2));
    end
    
    % Normalize and plot
    Z_sinks = Z_sinks / sum(Z_sinks(:));
    
    contourf(X_sinks, Y_sinks, Z_sinks, 30, 'LineStyle', 'none');
    hold on;
    scatter(all_sinks_x, all_sinks_y, 10, 'w', 'filled', 'MarkerFaceAlpha', 0.3);
    colormap(hot);
    colorbar;
    title('Sink Points Density');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    axis equal tight;
    set(gca, 'YDir', 'reverse');
else
    title('No Sink Points Found');
    axis([0 100 0 180]);  % Default range if no points
    set(gca, 'YDir', 'reverse');
end

% Add a main title
sgtitle('Spatial Distribution Density of Source and Sink Points');

% Optional: Adjust the figure size
set(gcf, 'Position', [100, 100, 1000, 500]);
saveas(gcf, 'source_sink_density_kernel_smoothing.pdf');