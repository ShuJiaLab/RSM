%% Load Image
[file, path] = uigetfile('*.tif', 'Select the raw foci image'); 
if isequal(file, 0)
    disp('User canceled file selection.');
    return;
end
imagePath = fullfile(path, file);
rawImage = double(imread(imagePath));

%% Normalize & Background Correction
normalizedImage = rawImage - min(rawImage(:));
normalizedImage = normalizedImage / max(normalizedImage(:));
enhancedImage = adapthisteq(normalizedImage);
background = imgaussfilt(enhancedImage, 10);
correctedImage = enhancedImage - background;

%% Smoothing & Thresholding
filteredImage = imgaussfilt(correctedImage, 1);
level = graythresh(filteredImage);
binaryMask = filteredImage > level;
localMaxima = imregionalmax(filteredImage) & binaryMask;
[y, x] = find(localMaxima);

%% Generate fociMap
fociMap = zeros(size(rawImage));
for i = 1:length(x)
    fociMap(y(i), x(i)) = 1;
end

%% Estimate ideal grid from central region
centerX = round(size(fociMap, 2) / 2);
centerY = round(size(fociMap, 1) / 2);
xMin = centerX - 0.25 * size(fociMap, 2);
xMax = centerX + 0.25 * size(fociMap, 2);
yMin = centerY - 0.25 * size(fociMap, 1);
yMax = centerY + 0.25 * size(fociMap, 1);
inCenter = (x >= xMin & x <= xMax) & (y >= yMin & y <= yMax);
centerFociX = x(inCenter);
centerFociY = y(inCenter);
dx = mean(diff(sort(centerFociX)));
dy = mean(diff(sort(centerFociY)));
[Xgrid, Ygrid] = meshgrid(min(x):dx:max(x), min(y):dy:max(y));

%% Estimate Deformation Field
[indices, ~] = knnsearch([Xgrid(:) Ygrid(:)], [x y]);
idealNearest = [Xgrid(indices), Ygrid(indices)];
deviation = [x y] - idealNearest;
Fx = scatteredInterpolant(idealNearest(:,1), idealNearest(:,2), deviation(:,1));
Fy = scatteredInterpolant(idealNearest(:,1), idealNearest(:,2), deviation(:,2));

%% Apply Inverse Deformation
[xx, yy] = meshgrid(1:size(fociMap,2), 1:size(fociMap,1));
dx = Fx(xx, yy);
dy = Fy(xx, yy);
dx(isnan(dx)) = 0;
dy(isnan(dy)) = 0;
warpedX = xx - dx;
warpedY = yy - dy;
correctedFociMap = interp2(double(fociMap), warpedX, warpedY, 'linear', 0);

%% Final Binary Map with Cleaned Foci
binaryFoci = correctedFociMap > 0.2;
labeled = bwlabel(binaryFoci);
correctedBinaryFociMap = zeros(size(binaryFoci));
props = regionprops(labeled, correctedFociMap, 'PixelIdxList', 'MaxIntensity');
for i = 1:length(props)
    [~, idx] = max(correctedFociMap(props(i).PixelIdxList));
    correctedBinaryFociMap(props(i).PixelIdxList(idx)) = 1;
end
% Save the corrected binary foci map
save('corrected_foci_map.mat', 'correctedBinaryFociMap');
disp('Corrected foci map saved to corrected_foci_map.mat');
%% Metrics to verify corection

%% Generate deformation model from real grid
Fx = scatteredInterpolant(idealNearest(:,1), idealNearest(:,2), x - idealNearest(:,1));
Fy = scatteredInterpolant(idealNearest(:,1), idealNearest(:,2), y - idealNearest(:,2));

%% Inverse Correction Test (x - Fx instead of x + Fx)

% Step 1: Filter out NaNs
valid_input = all(~isnan([x y]), 2);
x_clean = x(valid_input);
y_clean = y(valid_input);

% Step 2: Apply INVERSE correction
dx = Fx(x_clean, y_clean);
dy = Fy(x_clean, y_clean);
x_corr = x_clean - dx;  % Inverted correction
y_corr = y_clean - dy;

% Step 3: Remove any NaNs introduced by interpolation
valid_corr = all(~isnan([x_corr y_corr]), 2);
x_clean = x_clean(valid_corr);
y_clean = y_clean(valid_corr);
x_corr = x_corr(valid_corr);
y_corr = y_corr(valid_corr);

% Step 4: Flatten ideal grid
Xgrid_flat = Xgrid(:);
Ygrid_flat = Ygrid(:);
ideal_coords = [Xgrid_flat, Ygrid_flat];

% Step 5: KNN to match each detected foci to closest ideal grid point
[ideal_idx, ~] = knnsearch(ideal_coords, [x_clean, y_clean]);

% Step 6: Filter valid indices
n_coords = size(ideal_coords, 1);
valid_idx = ideal_idx > 0 & ideal_idx <= n_coords;
ideal_idx = ideal_idx(valid_idx);
x_clean = x_clean(valid_idx);
y_clean = y_clean(valid_idx);
x_corr  = x_corr(valid_idx);
y_corr  = y_corr(valid_idx);

% Step 7: Extract matched ideal coordinates
ideal_coords_matched = ideal_coords(ideal_idx, :);

% Step 8: Compute deviation before and after correction
deviation_before = vecnorm([x_clean y_clean] - ideal_coords_matched, 2, 2);
deviation_after  = vecnorm([x_corr  y_corr]  - ideal_coords_matched, 2, 2);

% Step 9: Plot results
figure;
histogram(deviation_before, 40); hold on;
histogram(deviation_after, 40);
legend('Before correction', 'After inverse correction');
xlabel('Deviation (pixels)');
ylabel('Count');
title('Foci Deviation from Ideal Grid (Using Inverse Correction)');

% Step 10: Output stats
fprintf('--- Inverse Correction Deviation Summary ---\n');
fprintf('Mean Before Correction: %.3f px\n', mean(deviation_before));
fprintf('Mean After  Correction: %.3f px\n', mean(deviation_after));
fprintf('Max  Before Correction: %.3f px\n', max(deviation_before));
fprintf('Max  After  Correction: %.3f px\n', max(deviation_after));
fprintf('Std  Before Correction: %.3f px\n', std(deviation_before));
fprintf('Std  After  Correction: %.3f px\n', std(deviation_after));
