# %% Import external libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import exposure, filters, morphology, measure
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from sklearn.neighbors import NearestNeighbors

# %% Import internal utility functions
from util import (
    load_image, 
    normalize_mat, 
    save_matfile
)

# %% Set file path
file_path = ''

# %% Load image from file path
#raw_image_loaded = load_image(file_path)
raw_image_loaded = load_image()
if raw_image_loaded is None:
    raise ValueError("Failed to load image using utils.load_image")
raw_image = raw_image_loaded.astype(np.float32)

# %% Normalize Image
normalized_image = normalize_mat(raw_image)

# %% Background Correction
enhanced_image = exposure.equalize_adapthist(normalized_image, clip_limit=0.01, nbins=256)
sigma = 20
kernel_size = int(np.ceil(2 * sigma) * 2 + 1)
background = cv.GaussianBlur(enhanced_image, 
                             ksize=(kernel_size, kernel_size), 
                             sigmaX=sigma,
                             sigmaY=sigma,
                             borderType=cv.BORDER_REPLICATE)
corrected_image = enhanced_image - background

# %% Smoothing
sigma = 1
kernel_size = int(np.ceil(2 * sigma) * 2 + 1)
filtered_image = cv.GaussianBlur(corrected_image, 
                                 ksize=(kernel_size, kernel_size), 
                                 sigmaX=sigma,
                                 sigmaY=sigma,
                                 borderType=cv.BORDER_REPLICATE)

# %% Thresholding
normalized_filtered_image = normalize_mat(filtered_image)
thresh_val = filters.threshold_otsu(normalized_filtered_image)
binary_mask = normalized_filtered_image > thresh_val
local_maxima = morphology.local_maxima(normalized_filtered_image) & binary_mask
y, x = np.nonzero(local_maxima)

# %% Generate fociMap
foci_map = np.zeros_like(raw_image)
foci_map[y, x] = 1

# Estimate ideal grid from central region
foci_map_width = foci_map.shape[1]
foci_map_height = foci_map.shape[0]
center_x = foci_map_width // 2
center_y = foci_map_height // 2
x_min = int(center_x - 0.25 * foci_map_width)
x_max = int(center_x + 0.25 * foci_map_width)
y_min = int(center_y - 0.25 * foci_map_height)
y_max = int(center_y + 0.25 * foci_map_height)

in_center = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
center_foci_x = x[in_center]
center_foci_y = y[in_center]

dx = np.mean(np.diff(np.sort(center_foci_x)))
dy = np.mean(np.diff(np.sort(center_foci_y)))
grid_x_coords = np.arange(np.min(x), np.max(x) + dx * 0.5, dx)
grid_y_coords = np.arange(np.min(y), np.max(y) + dy * 0.5, dy)
grid_x, grid_y = np.meshgrid(grid_x_coords, grid_y_coords)

# %% Estimate Deformation Field
# Step 1: Flatten grid to N x 2
grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

# Step 2: Stack query points
query_points = np.column_stack((x.ravel(), y.ravel()))

# Step 3: Nearest neighbor search
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(grid_points)
distances, indices = nbrs.kneighbors(query_points)

# Step 4: Extract ideal nearest from grid
ideal_nearest = grid_points[indices[:, 0]]

# Step 5: Compute deviation vectors
deviation = query_points - ideal_nearest

# Step 6: Interpolation
Fx = LinearNDInterpolator(ideal_nearest, deviation[:, 0])
Fy = LinearNDInterpolator(ideal_nearest, deviation[:, 1])

# %% Apply Inverse Deformation
xx, yy = np.meshgrid(np.arange(foci_map_width), np.arange(foci_map_height))
dx = Fx(xx, yy)
dy = Fy(xx, yy)
dx = np.nan_to_num(dx, nan=0.0)
dy = np.nan_to_num(dy, nan=0.0)
warpedX = xx - dx
warpedY = yy - dy
# RegularGridInterpolator expects axes in (y, x) order
interp = RegularGridInterpolator(
    (np.arange(foci_map_height), np.arange(foci_map_width)),
    foci_map.astype(np.float64),
    method='linear',
    bounds_error=False,
    fill_value=0.0
)
coords = np.stack([warpedY.ravel(), warpedX.ravel()], axis=-1)
corrected_foci_map = interp(coords).reshape(foci_map_height, foci_map_width)

# %% Final Binary Map with Cleaned Foci
binary_foci = corrected_foci_map > 0.2
labeled = measure.label(binary_foci, connectivity=2)
corrected_binary_foci_map = np.zeros_like(binary_foci)
props = measure.regionprops(labeled, intensity_image=corrected_foci_map)
for prop in props:
    if len(prop.coords) == 0:
        continue
    max_idx = np.argmax(corrected_foci_map[tuple(prop.coords.T)])
    max_pixel_coord = prop.coords[max_idx]
    corrected_binary_foci_map[tuple(max_pixel_coord)] = 1
    
# %% Save the corrected binary foci map
save_directory_path = ''
if not os.path.exists(save_directory_path):
    os.makedirs(save_directory_path)
    
save_matfile(os.path.join(save_directory_path, "corrected_foci_map.mat"), 
             {"correctedBinaryFociMap": corrected_binary_foci_map})

#Metrics to verify correction
# Step 1: Regenerate ideal_nearest from KNN
nn = NearestNeighbors(n_neighbors=1)
grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
nn.fit(grid_coords)
_, indices = nn.kneighbors(np.column_stack([x, y]))
ideal_nearest = grid_coords[indices.flatten()]

# Step 2: Create interpolants (deformation model)
Fx_interp = LinearNDInterpolator(ideal_nearest, x - ideal_nearest[:, 0])
Fy_interp = LinearNDInterpolator(ideal_nearest, y - ideal_nearest[:, 1])

# Step 3: Filter valid inputs
valid_input = ~np.isnan(x) & ~np.isnan(y)
x_clean = x[valid_input]
y_clean = y[valid_input]

# Step 4: Apply inverse correction
dx = Fx_interp(x_clean, y_clean)
dy = Fy_interp(x_clean, y_clean)
x_corr = x_clean - dx
y_corr = y_clean - dy

# Step 5: Remove new NaNs
valid_corr = ~np.isnan(x_corr) & ~np.isnan(y_corr)
x_clean = x_clean[valid_corr]
y_clean = y_clean[valid_corr]
x_corr = x_corr[valid_corr]
y_corr = y_corr[valid_corr]

# Step 6: KNN match each detected foci to the ideal grid point
nn.fit(grid_coords)
ideal_idx = nn.kneighbors(np.column_stack([x_clean, y_clean]), return_distance=False).flatten()

# Step 7: Filter valid indices
valid_idx = (ideal_idx >= 0) & (ideal_idx < len(grid_coords))
ideal_idx = ideal_idx[valid_idx]
x_clean = x_clean[valid_idx]
y_clean = y_clean[valid_idx]
x_corr  = x_corr[valid_idx]
y_corr  = y_corr[valid_idx]

# Step 8: Extract matched ideal coordinates
ideal_coords_matched = grid_coords[ideal_idx]

# Step 9: Compute deviations
deviation_before = np.linalg.norm(np.column_stack([x_clean, y_clean]) - ideal_coords_matched, axis=1)
deviation_after  = np.linalg.norm(np.column_stack([x_corr, y_corr]) - ideal_coords_matched, axis=1)

# Step 10: Plot histograms
plt.figure(figsize=(8, 5))
plt.hist(deviation_before, bins=40, alpha=0.6, label='Before correction')
plt.hist(deviation_after, bins=40, alpha=0.6, label='After correction')
plt.xlabel('Deviation (pixels)')
plt.ylabel('Count')
plt.title('Foci Deviation from Ideal Grid (Using Inverse Correction)')
plt.legend()
plt.tight_layout()
plt.show()

# Step 11: Print stats
print('--- Inverse Correction Deviation Summary ---')
print(f'Mean Before Correction: {np.mean(deviation_before):.3f} px')
print(f'Mean After  Correction: {np.mean(deviation_after):.3f} px')
print(f'Max  Before Correction: {np.max(deviation_before):.3f} px')
print(f'Max  After  Correction: {np.max(deviation_after):.3f} px')
print(f'Std  Before Correction: {np.std(deviation_before):.3f} px')
print(f'Std  After  Correction: {np.std(deviation_after):.3f} px')