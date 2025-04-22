# %% ScanOSM Reconstruction

# %% Import external libraries
import os
import h5py
import cv2 as cv
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# %% Import internal utility functions
from utils import (
    select_file, 
    normalize_mat, 
    fill_missing_spline, 
    inpaint_nans_biharmonic,
    fspecial_gauss,
    deconvblind
)

# %% Parameters
# Set control switches
load_file = 1
decon_on = 1
save_tiff = 1

# Set Parameters
s_frame = 0     # Starting frame index
n_frames = 271  # Number of frames to be reconstructed
gauss_fit = 0   # Perform 2D Gaussian fit (only for beads)
verbose = 0     # Display resolution for each fit
r = 3           # Set digital pinhole radius
alpha = 1.25
beta = 1.1

data_directory_path = ""    # Full path to directory of image data

# %% Load Data
# Load tracked data cdata0 and focimap
if load_file:
    filepath = select_file(
        filetypes=[("h5 Files", "*.h5")], 
        dir_path=data_directory_path
    )
    with h5py.File(filepath, 'r') as f:
        cdata0 = f['S'][:]
        data_focimap = f['S_focimap'][:]   
else:
    # Modify to use the correct S and S_foci from Step 2
    cdata0 = S_ecc.copy()
    data_focimap = S_focimap_1d.copy()
    
# %% Background removal
print("Background removal...")

n_frames = min(n_frames, cdata0.shape[0] - s_frame)

cdata = np.zeros((n_frames, cdata0.shape[1], cdata0.shape[2]))

sigma_gauss = 30

for i in range(s_frame, s_frame + n_frames):
    frame = cdata0[i].astype(np.float32)
    blurred = cv.GaussianBlur(frame, ksize=(0, 0), sigmaX=sigma_gauss)
    cdata[i, :, :] = cdata0[i, :, :] / blurred

# %% Widefield
widefield = cv.resize(
    (np.sum(cdata[s_frame:s_frame + n_frames, :, :], axis=0)), 
    (2 * cdata.shape[2], 2 * cdata.shape[1]),   # (width, height)
    interpolation=cv.INTER_NEAREST
)

# %% Pixel Reassignment
print("Pixel Reassignment...")

# Pads images with 0's. The pad width is r, likely so that pinholes near the edge of the frame can be processed
Q = np.pad(
    cdata, 
    pad_width=((0, 0), (r, r), (r,r)), 
    mode='constant', 
    constant_values=0
)
Q += 0.0001

# Generates an output image twice the size of Q-padded data of our image
padded_output = np.zeros((2 * Q.shape[1], 2 * Q.shape[2]))

# overlap detector
overlap = padded_output
overlap = np.pad(
    overlap, 
    pad_width=((r, r), (r, r)), 
    mode='constant', 
    constant_values=0
)
overlap = overlap.astype(bool)
checker = overlap.copy()
padded_output_w = np.ones((2 * Q.shape[1], 2 * Q.shape[2]))

sz2 = cdata[0, :, :].shape
for i in range(s_frame, s_frame + n_frames):
    k = np.argwhere(data_focimap[i, :, :] > 0.95)
    
    for j in range(len(k)):
        row, col = k[j]
        row = row + r
        col = col + r
        for jj in range(-r, r + 1):
            for ii in range(-r, r + 1):
                out_y = 2 * row + ii
                out_x = 2 * col + jj
                q_y = row + ii
                q_x = col + jj
                
                if overlap[out_y, out_x] != 0:
                    padded_output[out_y, out_x] = max(padded_output[out_y, out_x], Q[i, q_y, q_x]) # (padded_output[out_y, out_x] + Q[i, q_y, q_x]) / 2# 
                    checker[out_y, out_x] = 1
                else:
                    padded_output[out_y, out_x] = max(padded_output[out_y, out_x], Q[i, q_y, q_x]) # (padded_output[out_y, out_x] + Q[i, q_y, q_x]) / 2# 
                    padded_output_w[out_y, out_x] = padded_output_w[out_y, out_x] + 1
                overlap[out_y, out_x] = 1
                
# %% Weighted Average
# Before filling
print("Image Reconstruction")

# copy() is used to make development easier, can be removed in the future for memory efficiency
output = padded_output[r:-r, r:-r].copy()
output_w = padded_output_w[r:-r, r:-r].copy()

output = normalize_mat(output / (output_w**beta))

plt.figure()
plt.imshow(output[10:-10, 10:-10] / 2, cmap='gray', aspect='equal')
plt.title("Before filling")
plt.axis("image")

# Stage 1
outputz = output.copy()
outputz[outputz == 0] = np.nan
F = fill_missing_spline(outputz, axis=0, max_gap=10)

plt.figure()
plt.imshow(F[10:-10, 10:-10] / 2, cmap='gray', aspect='equal')
plt.title("Stage 1")
plt.axis("image")

# 2D Inpaint
F1 = fill_missing_spline(F, axis=1, max_gap=15)

outputz = output.copy()
outputz[outputz == 0] = np.nan
F = fill_missing_spline(outputz, axis=1, max_gap=10)
F2 = fill_missing_spline(F, axis=0, max_gap=10)


F2D = inpaint_nans_biharmonic(outputz[10:-10, 10:-10])
F2D = cv.GaussianBlur(F2D, ksize=(0, 0), sigmaX=1)

plt.figure()
plt.imshow(F2D[10:-10, 10:-10] / 2, cmap='gray', aspect='equal')
plt.title("2D inpaint")
plt.axis("image")

# Widefield
plt.figure()
plt.imshow(widefield, cmap='gray', aspect='equal')
plt.title("Widefield")
plt.axis("image")

# INT
F = F1.copy()
output[output > 1] = 1

plt.figure()
plt.imshow(output[10:-10, 10:-10] / 2, cmap='gray', aspect='equal')
plt.title("INT")
plt.axis("image")

# %% Deconvolution
if decon_on:
    print("Deconvolution...")
    
    # lambda =515; NA=1.45;.61*lambda/NA/sqrt(2)
    sigma_PSF = 153.198/2.345/32.5; # 488 laser
    # sigma_PSF = 200.4/2.335/32.5; # 647 laser
    
    # Build a square Gaussian kernel
    size = int(2 * np.ceil(2 * sigma_PSF)) + 1
    PSF = fspecial_gauss(size, sigma_PSF)
    
    # Blind deconvolution
    usim, psf_est = deconvblind(output, PSF, num_iter=1, RL_iter=5)
    usim2, _ = deconvblind(F2D, PSF, num_iter=1, RL_iter=5)
    
    plt.figure()
    plt.imshow(normalize_mat(usim), cmap='gray', aspect='equal')
    plt.title("RSM deconv")
    plt.axis("image")

plt.show()
# %% Save Results
save_directory_path = ""    # Full path to directory where files are saved
if save_tiff:
    if not os.path.exists(save_directory_path):
        os.makedirs(save_directory_path)
    
    # Save 2D Inpaint
    F2D_crop = F2D[10:-10, 10:-10]
    F2D_crop_uint16 = (F2D * (2**16)).astype(np.uint16)
    tiff.imwrite(os.path.join(save_directory_path, "INT2.tif"), F2D_crop_uint16)
    
    # Save widefield image
    wf_crop = widefield[10:-10, 10:-10]
    wf_norm = normalize_mat(wf_crop)
    wf_uint16 = (wf_norm * (2**16)).astype(np.uint16)
    tiff.imwrite(os.path.join(save_directory_path, "WF.tif"), wf_uint16)
    
    # Save INT image
    int_crop = output[10:-10, 10:-10]
    int_uint16 = (int_crop * (2 ** 16)).astype(np.uint16)
    tiff.imwrite(os.path.join(save_directory_path, "INT.tif"), int_uint16)
    
    # Save RSM image
    usim_uint16 = (usim * (2**16)).astype(np.uint16)
    tiff.imwrite(os.path.join(save_directory_path, "RSM.tif"), int_uint16)
    
    # Save RSM2 image
    usim2_uint16 = (usim2 * (2**16)).astype(np.uint16)
    tiff.imwrite(os.path.join(save_directory_path, "RSM2.tif"), int_uint16)