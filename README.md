# RSM  System Reconstruction
This project focuses on the reconstruction of data obtained from the Resonant Multifocal Scanning Microscope (RSM) system. The code follows three main steps: mask calibration, data alignment, and the core reconstruction process. These steps are crucial for generating accurate 2D images from the acquired data.
# Steps:

## Step 1: Mask Calibration
This step generates calibration masks necessary for accurate data processing. Proper mask calibration is crucial to ensure reliable reconstruction.

## Step 2: Data Alignment
Data alignment is a key component of the reconstruction process. Beginning with this step, there will be two codebase options: (1) MATLAB and (2) Python. Both the MATLAB and Python code utilize a similar procedure for this step; however, there are certain differences. 

The MATLAB code offers two alignment options:
- Cross Correlation: Utilize cross-correlation for accurate tracking.
- 1D Tracking: Use this method for data alignment.

The Python code offers two alignement options:
- Correlation Coefficient (Zero-Mean Normalized Cross Correlation): Utilize zero-mean normalized cross correlation for accurate tracking.
- 1D Tracking: Use this method for data alignment.

## Step 3: Main Reconstruction
The main reconstruction process involves several important operations, including:
- Pinholing
- Scaling
- Pixel Reassignment
- Deconvolution

# Setup:

## Directions
For MATLAB code:
- Ensure MATLAB is installed on your system.
- Open the provided MATLAB script for this code.
- Execute the script to run the reconstruction process.

For Python code:
- Ensure Python is installed on your system.
- Use the provided `requirements.txt` to install necessary dependencies.
- Executer the script to run the reconstruction process.

## Additional Note:
- This code may require additional libraries or MATLAB toolboxes, depending on the specific reconstruction techniques used.
- Make sure to have the necessary calibration data ready before running the code.

# References:
Please refer to relevant academic papers, documentation, or resources for a deeper understanding of the RSM system and the reconstruction techniques used.

For questions or assistance, please contact Kidan Tadesse at ktadesse3@gatech.edu.
