# Recovery of ROI Shift with Correlation Coefficient (Zero-Mean Normalized Cross Correlation)

# %% Import external libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from datetime import datetime
from skimage import filters
from sklearn.linear_model import LinearRegression

# %% Import internal utility functions
from utils import (
    select_file, 
    load_image, 
    load_matfile, 
    save_h5_file,
    normalize_mat, 
    select_ROI, 
    implay
)

# %% Parameters 
sigma = 15
start_frame = 0
num_frames_analyzed = 343   # Total number of frames
w = 5   # additional width padding for search window
h = 5   # additional height padding for top of search window
data_directory_path = ""    # Full path to directory of image data
save_directory_path = ""    # Full path to directory where files are saved
save_file_name = "FILE NAME HERE" + f"{datetime.now().strftime('%m%d%Y%H%M%S')}"

# %% Step 1. Read Frames from File
# Load image data from .tiff file
# Load foci map from .mat from Step 1

# im has shape (# frames, height, width)
im = load_image(select_file(
        filetypes=[("TIFF files", "*.tif*")], 
        dir_path=data_directory_path
))

Mask_MSM = load_matfile(select_file(
        filetypes=[("MAT files", "*.mat*")], 
        dir_path=data_directory_path
))

Foci_Map = Mask_MSM['FociMap']

# %% Step 2. Prepare frames for Analysis
# Smooth out image to minimize effect of point scanning
imgA = filters.gaussian(im[0, :, :], sigma=sigma, mode='nearest')
# Normalize intensity
imgA = normalize_mat(imgA)

# %% Step 3. Select ROI
# Coordinates of the selected region (ROI)
# ROI_x = x top left corner
# ROI_y = y top left corner
ROI_x, ROI_y, ROI_width, ROI_height = select_ROI(imgA)
sect = imgA[ROI_y : ROI_y + ROI_height, ROI_x : ROI_x + ROI_width]
print(f"Coordinates of the selected ROI: (x, y) = {ROI_x, ROI_y}; width = {ROI_width}; height = {ROI_height}")

# %% Tracking Setup & Initialization
# Restrict number of frames analyzed, we do this in case the raw data has less 
# frames than the number we intended to analyze
num_frames_analyzed = min(num_frames_analyzed, im.shape[0] - start_frame)

# Create 3D array of zeros representing a stack of frames with same dimension 
# as `sect`
S = np.zeros((num_frames_analyzed, sect.shape[0], sect.shape[1]))

# Create foci map for S: zero matrix with shape of S
S_focimap = np.zeros(S.shape)

# Create a `num_frames_analyzed` by 2 matrix of zeros, this matrix will hold 
# the global coords of the top left corner of each match
positions = np.zeros((num_frames_analyzed, 2))
# Put position of ROI in first frame
positions[0, :] = [ROI_x, ROI_y]

# Add selected window to the final output video
S[0, :, :] = normalize_mat(im[0, 
                              ROI_y : ROI_y + ROI_height, 
                              ROI_x : ROI_x + ROI_width])

# Define width and height of search window, note that the window has padding
Search_Width = slice(max(0, ROI_x - w), min(imgA.shape[1], ROI_x + ROI_width + w))
Search_Top = max(0, ROI_y - h)

# %% Correlation Coefficient tracking method

# Goes frame by frame and tries to find the ROI in each frame
for i in range(start_frame, start_frame + num_frames_analyzed):
    # Logs tracking process every 10 frames
    if i == 0 or ((i + 1) % 10 == 0 ) or ((i + 1) == num_frames_analyzed):
        print(f"Processing frame {i + 1} / {num_frames_analyzed}")

    # Retrieve and process particular section of (i+1)-th frame
    img_raw = im[i, :, :]
    
    # We only look within the Search Window and not the whole image
    img_vertical_strip = img_raw[Search_Top:, Search_Width].copy()
    foci_vertical_strip = Foci_Map[Search_Top:, Search_Width].copy()
    
    # Pre-process the Search Window to improve searching
    img_processed = filters.gaussian(img_vertical_strip, sigma=sigma, mode='nearest')
    img_processed = normalize_mat(img_processed)
    img_processed = img_processed.astype('float32')

    # % Step 4. Template Matching with Correlation Coefficient
    # Use OpenCV's template matching with normalized correlation coefficient as
    # the similarity metric. This metric is a Zero-Mean Normalized 
    # Cross-Correlation that removes the influence of overall brightness and
    # contrast from both the template and the image being searched. It does so
    # by subtracting the mean from both the template and the image prior to
    # performing cross correlation.
    res = cv.matchTemplate(img_processed, sect, method=cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    # The `max_loc` tuple is the top-left position of the match within the 
    # bounded strip. We can use the local position to find the shifts relative
    # to the original position of the ROI
    local_TL_x, local_TL_y = max_loc    
    positions[i, :] = [local_TL_x + Search_Width.start, local_TL_y + Search_Top]

    # % Step 5. Use Match Position to Cut the New ROI Frame
    S[i, :, :] = normalize_mat(img_vertical_strip[local_TL_y : local_TL_y + ROI_height, 
                                                  local_TL_x : local_TL_x + ROI_width])
    S_focimap[i, :, :] = foci_vertical_strip[local_TL_y : local_TL_y + ROI_height, 
                                             local_TL_x : local_TL_x + ROI_width]
    
    # Dynamically shrink height of future Search Window
    Search_Top = max(0, local_TL_y + Search_Top - h)

# %% Step 6. Show & Save
# Show the plot of the x shifts and y positions
plt.plot(positions[:, 1])
plt.xlabel("Frame Index #"), plt.ylabel("Global y position"), plt.title("Y Position vs Frames")
plt.show()

implay(S)

# %% Saving S, S_focimap
# This does take some time, so you can skip this if not needed
data_dict = {"S": S, "S_focimap": S_focimap}
save_h5_file(f"{save_directory_path}/{save_file_name}.h5", data_dict)

# %% Step 7. Refine Tracking with Linear Fit of Y positions
frames = np.arange(start=0, stop=num_frames_analyzed, step=1).reshape((-1, 1))

# Fit linear model to y positions
model = LinearRegression()
model.fit(frames, positions[:num_frames_analyzed, 1])
y_pred = model.predict(frames)

plt.figure(figsize=(8, 6))
plt.scatter(frames, positions[:, 1], color='blue', label="Y positions")
plt.plot(frames, y_pred, color='red', linewidth=2, label="Regression line")
plt.xlabel("Frame #")
plt.ylabel("Y position")
plt.title("Linear Regression of Y Positions")
plt.legend()
plt.show()

# Align ROI using Linear Regression of Y positions
S_1d = 0 * S

S_focimap_1d = 0 * S_focimap

for i in range(start_frame, start_frame + num_frames_analyzed):
    img_raw = im[i, :, :]
    img_vertical_strip = img_raw[:, Search_Width]
    
    foci_vertical_strip = Foci_Map[:, Search_Width]

    S_1d[i, :, :] = img_vertical_strip[int(y_pred[i]) : int(y_pred[i]) + ROI_height, w : w + ROI_width]

    S_focimap_1d[i, :, :] = foci_vertical_strip[int(y_pred[i]) : int(y_pred[i]) + ROI_height, w : w + ROI_width]

implay(S_1d)

# %% Saving S_1d, S_focimap_1d
# This does take some time, so you can skip this if not needed
data_dict = {"S": S_1d, "S_focimap": S_focimap_1d}
save_h5_file(f"{save_directory_path}/{save_file_name}_1D.h5", data_dict)
