import os
import sys
import h5py
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
from skimage.restoration import inpaint, richardson_lucy
from scipy.interpolate import interp1d
from scipy.io import savemat
from mat73 import loadmat
from PyQt5.QtWidgets import QApplication, QFileDialog


def normalize_mat(mat_in):
    '''
    Normalize an n-dimensional array using min-max normalization,
    scaling all values to the range [0, 1].

    Parameters
    ----------
    mat_in : np.ndarray
        Input array to be normalized. Should be of numeric type.

    Returns
    -------
    mat_out : np.ndarray
        Normalized array with the same shape as mat_in, where values are scaled
        to the range [0, 1] using the formula:
        (x - min) / (max - min)
    '''
    mat_in = mat_in.astype("float32")
    mat_in_max = np.max(mat_in)
    mat_in_min = np.min(mat_in)
    mat_out = (mat_in - mat_in_min) / (mat_in_max - mat_in_min)
    return mat_out


def fspecial_gauss(size, sigma):
    '''
    Create a 2D Gaussian kernel with the specified size and standard deviation.

    Mimics MATLAB's `fspecial('gaussian', ...)` function, generating a symmetric
    Gaussian filter for use in image processing tasks such as smoothing or blurring.

    Parameters
    ----------
    size : int
        Size of the kernel. The output will be a (size x size) 2D array.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    g : np.ndarray
        2D Gaussian kernel normalized to sum to 1.
    '''
    x, y = np.mgrid[-size//2 + 1 : size//2 + 1, -size//2 + 1 : size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2 * sigma**2)))
    return g/g.sum()


def crop_psf(psf_full, target_shape):
    '''
    Center-crop a 2D point spread function (PSF) array to a specified target shape.

    Parameters
    ----------
    psf_full : np.ndarray
        2D input array representing the full PSF.
    target_shape : tuple of int
        Desired output shape as (height, width).

    Returns
    -------
    cropped : np.ndarray
        Center-cropped 2D array of shape `target_shape`.
    '''
    h, w = psf_full.shape
    th, tw = target_shape
    cy, cx = h // 2, w // 2
    y1 = cy - th // 2
    x1 = cx - tw // 2
    cropped = psf_full[y1:y1+th, x1:x1+tw]
    return cropped


def deconvblind(img, psf_init, num_iter=1, RL_iter=5, pad_mode='reflect'):
    '''
    Perform blind deconvolution using alternating Richardson-Lucy updates.

    This method jointly estimates both the image and the point spread function (PSF),
    alternating between deconvolving the image and refining the PSF. Padding is applied
    to reduce boundary artifacts during convolution.

    Parameters
    ----------
    img : np.ndarray
        2D input image to be deconvolved.
    psf_init : np.ndarray
        Initial guess for the PSF (must be 2D).
    num_iter : int, optional
        Number of outer iterations alternating between image and PSF updates. Default is 1.
    RL_iter : int, optional
        Number of Richardson-Lucy iterations used for each update step. Default is 5.
    pad_mode : str, optional
        Padding mode passed to np.pad before deconvolution (e.g., 'reflect', 'constant'). Default is 'reflect'.

    Returns
    -------
    im_est : np.ndarray
        Final deconvolved image, cropped to original size.
    psf_est : np.ndarray
        Final estimated PSF, normalized to sum to 1 and same shape as `psf_init`.
    '''
    # Prepare
    psf_est = psf_init.copy()
    pad = psf_init.shape[0] // 2
    img_p = np.pad(img, pad, mode=pad_mode)
    im_est = img_p.copy()
    
    # Alternate Richardson-Lucy Updates
    for _ in range(num_iter):
        # Update image estimate
        im_est = richardson_lucy(im_est, psf_est, num_iter=RL_iter, clip=True)
        # Update PSF estimate, then crop and norm
        psf_full = richardson_lucy(img_p, im_est, num_iter=RL_iter, clip=True)
        psf_cropped = crop_psf(psf_full, psf_init.shape)
        psf_est = psf_cropped / np.sum(psf_cropped)
        
    # Crop back to original FOV
    im_est = im_est[pad:-pad, pad:-pad]
    return im_est, psf_est


def find_nan_runs(mask):
    '''
    Identify contiguous runs of `True` values in a 1D boolean mask.

    This is commonly used to detect sequences of NaNs when `mask` is
    derived from `np.isnan(array)`. Each run is returned as a (start, end)
    index pair, where `end` is exclusive.

    Parameters
    ----------
    mask : np.ndarray
        1D boolean array where `True` values indicate positions of interest
        (e.g., NaNs in the original data).

    Returns
    -------
    runs : list of tuple
        List of (start, end) index pairs for each contiguous run of `True` values.
        The `end` index is exclusive, so the run covers `mask[start:end]`.
    '''
    padded = np.pad(mask.astype(int), (1, 1), mode='constant')
    changes = np.diff(padded)
    starts = np.asarray(changes == 1).nonzero()[0]
    ends = np.asarray(changes == -1).nonzero()[0]
    runs = list(zip(starts, ends))
    return runs
    

def fill_missing_spline(arr, axis=0, max_gap=10):
    '''
    Fill NaNs in a 2D array using cubic spline interpolation along the given axis,
    only for contiguous NaN gaps of length ≤ max_gap.

    Parameters
    ----------
    arr : np.ndarray
        2D array with possible NaN values.
    axis : int, optional
        Axis to interpolate along (0 for columns, 1 for rows). Default is 0.
    max_gap : int, optional
        Maximum gap length to fill. Larger gaps are left as NaN. The default is 10.

    Returns
    -------
    filled : np.ndarry
        Copy of `arr` with NaNs filled in where possible.

    '''
    arr = np.asarray(arr)
    filled = arr.copy()
    n_rows, n_cols = arr.shape
    
    # Determine which dimension we're iterating over
    num_lines = n_cols if axis == 0 else n_rows
    
    for idx in range(num_lines):
        line = arr[:, idx] if axis == 0 else arr[idx, :]
        
        nan_mask = np.isnan(line)
        if np.sum(~nan_mask) < 2:
            continue # Not enough points to build a spline
            
        x_vals = np.arange(len(line))
        valid_mask = ~nan_mask
        spline = interp1d(
            x_vals[valid_mask], 
            line[valid_mask],
            kind='cubic',
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        # Find the contiguous runs of NaNs, and only fill gaps <= max_gap
        start_end_pairs = find_nan_runs(nan_mask)
        for (start, end) in start_end_pairs:
            gap_len = end - start
            if gap_len <= max_gap:
                fill_indices = np.arange(start, end)
                fill_values = spline(fill_indices)
                if axis == 0:
                    filled[fill_indices, idx] = fill_values
                else:
                    filled[idx, fill_indices] = fill_values
                    
    return filled


def inpaint_nans_biharmonic(arr_masked):
    '''
    Fill NaN values in a 2D array using biharmonic inpainting.

    This method uses the biharmonic equation to smoothly interpolate over NaNs,
    leveraging the surrounding pixel values. It is especially useful for
    spatially coherent image data.

    Parameters
    ----------
    arr_masked : np.ndarray
        2D array containing NaNs to be inpainted. Should represent image-like data.

    Returns
    -------
    arr_filled : np.ndarray
        Array with the same shape as `arr_masked`, where NaNs have been replaced
        by smooth inpainting using `skimage.restoration.inpaint_biharmonic`.
    '''
    arr = np.copy(arr_masked)
    nan_mask = np.isnan(arr)
    # Replace nans with 0 for the call, track them in the mask
    arr[nan_mask] = 0
    # skimage inpaint wants a float64 array
    arr_filled = inpaint.inpaint_biharmonic(
        arr.astype(np.float64),
        mask=nan_mask.astype(bool)
    )
    
    return arr_filled


def select_file(filetypes=[], dir_path=os.getcwd(), print_toggle=True) -> str:
    '''
    Open a file dialog for user to select a file, and return the absolute path.

    Uses PyQt5 to display a GUI-based file selection window, with optional filetype
    filtering and customizable starting directory.

    Parameters
    ----------
    filetypes : list of tuple, optional
        List of (description, pattern) pairs to filter selectable files.
        Example: [('TIFF files', '*.tif'), ('Text files', '*.txt')]. Default is no filter.
    dir_path : str, optional
        Starting directory for the file dialog. Default is the current working directory.
    print_toggle : bool, optional
        If True, prints the selected file path. Default is True.

    Returns
    -------
    file_path : str or None
        Absolute path to the selected file, or None if no file was selected.
    '''
    # Ensures there is an application instance
    app = QApplication([])
    if app is None:
        app = QApplication(sys.argv)
    
    # Create filters for particular filetypes
    if filetypes:
        filters = ";;".join([f"{desc} ({pattern})" for desc, pattern in filetypes])
    else:
        filters = "All Files (*)"
    
    # Get filepath
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select file",
        dir_path,
        filter=filters
    )
    
    # Error handling for when no file is selected
    if file_path:
        if print_toggle:
            print(f"Selected file path: {file_path}")
        return file_path
    else:
        print("No file selected")
        return None


def load_image(file_path: str, print_toggle=True):
    '''
    Load a TIFF image from disk using memory mapping if possible, with fallback to imread.

    Attempts to load the image using `tifffile.memmap` for memory-efficient access.
    If that fails, falls back to `tifffile.imread` to fully load the image into memory.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the TIFF image file.
    print_toggle : bool, optional
        If True, prints status messages during loading. Default is True.

    Returns
    -------
    image : np.ndarray or None
        Loaded image data as a NumPy array. Returns None if loading fails.
    '''
    try:
        if print_toggle:
            print("Loading file...")

        # Try using memmap first
        try:
            memmap_vol = tiff.memmap(file_path)

            if print_toggle:
                print(f"Successfully loaded using memmap: {file_path}")

            return memmap_vol
        except Exception as memmap_exception:
            print(f"Memmap failed: {memmap_exception}\nFalling back to imread...")
            image_data = tiff.imread(file_path)
            if print_toggle:
                print(f"Successfully loaded using imread: {file_path}")
            return image_data
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def load_matfile(file_path: str, print_toggle: bool = True):
    '''
    Load a MATLAB .mat file and return its contents as a dictionary.

    Uses `scipy.io.loadmat` to read MATLAB files (typically version 7.2 or earlier),
    returning the variables stored in the file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .mat file.
    print_toggle : bool, optional
        If True, prints status messages during loading. Default is True.

    Returns
    -------
    matfile_dict : dict or None
        Dictionary containing variables from the .mat file. Returns None if loading fails.
    '''
    try:
        if print_toggle:
            print(f"Loading {file_path}")
        
        matfile_dict = loadmat(file_path)
        
        if print_toggle:
            print(f"Successfully loaded: {file_path}")

        return matfile_dict
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def save_h5_file(filename, data_dict: dict):
    '''
    Save a dictionary of arrays to an HDF5 (.h5) file using gzip compression.

    Each key-value pair in the dictionary is stored as a separate dataset in the file.
    Chunking is enabled for efficient I/O, and datasets are compressed using gzip.

    Parameters
    ----------
    filename : str
        Path to the output .h5 file.
    data_dict : dict
        Dictionary where keys are dataset names (str) and values are NumPy arrays.

    Returns
    -------
    None
    '''
    print("\nSaving .h5 file")
    with h5py.File(filename, 'w') as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val, chunks=True, compression='gzip')
    print("Save Complete!")
         
    
def save_matfile(filename, data_dict: dict, do_compression=False):
    '''
    Save a dictionary of variables to a MATLAB .mat file.

    Uses `scipy.io.savemat` to write the contents of `data_dict` to disk,
    compatible with MATLAB version 7.2 and earlier. Optionally enables
    compression to reduce file size.

    Parameters
    ----------
    filename : str
        Path to the output .mat file.
    data_dict : dict
        Dictionary where keys are variable names (str) and values are the data (e.g., NumPy arrays).
    do_compression : bool, optional
        If True, enables zlib compression. Default is False.

    Returns
    -------
    None
    '''
    print("\nSaving .mat file")
    savemat(filename, mdict=data_dict, do_compression=do_compression)
    print("Save Complete!")


def select_ROI(img):
    '''
    Display an image and allow the user to interactively select a rectangular Region of Interest (ROI).

    The user selects the ROI by clicking and dragging a rectangle on the displayed image.
    Once the selection is made, the window closes automatically and the coordinates are returned.

    Parameters
    ----------
    img : np.ndarray
        2D image array to be displayed for ROI selection.

    Returns
    -------
    coords : list of int or None
        List containing [x, y, width, height] of the selected ROI in pixel units.
        Returns None if no selection is made.
    '''
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="hot", aspect="equal")
    plt.axis("image")   # ensures a tight fit of the image to the plot

    print("Draw a rectangle by clicking and dragging on the image.")
    coords = []
    selection_complete = {"done": False}

    def on_select(e_click, e_release):
        '''
        Call back function to capture rectangle coordinates
        e_click: Mouse press event
        e_release: Mouse release event
        '''
        x1, y1 = e_click.xdata, e_click.ydata       # Top-left corner
        x2, y2 = e_release.xdata, e_release.ydata   # Bottom-right corner
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        print(f"({x1:.0f}, {y1:.0f}) --> ({x2:.0f}, {y2:.0f})\twidth: {width:.0f}\theight: {height:.0f}")

        # Store the rectangle coordinates
        coords.clear()
        coords.extend([round(x1), round(y1), round(width), round(height)])
        selection_complete["done"] = True
        plt.close()     # Close the figure after selection

    rect_selector = RectangleSelector(
        ax, 
        on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels",
        interactive=True
    )
    
    plt.show(block=True)    # Makes plt.show() block until window is closed

    if selection_complete["done"]:
        return coords
    else:
        print("No selection made.")
        return None


def imshow(img):
    '''
    Display a 2D image using matplotlib with grayscale colormap.

    Opens a new figure window and renders the image in grayscale.

    Parameters
    ----------
    img : np.ndarray
        2D image array to be displayed.

    Returns
    -------
    None
    '''
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.imshow(img, cmap='gray')
    plt.show()


def implay(img_stack, save=False, save_name=''):
    '''
    Play a stack of 2D grayscale images as an animation with interactive controls.

    Displays the image stack using matplotlib, allowing the user to play/pause 
    and step through frames. Optionally saves the animation as a GIF.

    Controls
    --------
    Spacebar : Toggle play/pause
    Left/Right arrows : Step backward/forward one frame (when paused)

    Parameters
    ----------
    img_stack : np.ndarray or list of np.ndarray
        A list or 3D NumPy array representing the image stack (frames must be 2D).
    save : bool, optional
        If True, saves the animation as a GIF using PillowWriter. Default is False.
    save_name : str, optional
        Output filename for the saved animation (e.g., 'output.gif'). Required if `save` is True.

    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(figsize=(8, 4))
    num_frames = len(img_stack)
    curr_frame = [0]
    playing = [True]
    
    im = ax.imshow(img_stack[curr_frame[0]], cmap='gray')
    title = ax.set_title(f"Frame 1/{num_frames}")

    def update_display(frame_idx: int):
        im.set_array(img_stack[frame_idx])
        title.set_text(f"Frame {frame_idx + 1}/{num_frames}")
        fig.canvas.draw_idle()  # Draws when control goes back to GUI event loop

    def animate(frame_idx: int):
        curr_frame[0] = frame_idx
        update_display(frame_idx)   # This is where im gets updated
        return [im]

    def toggle_play(event):
        if event.key == ' ':
            playing[0] = not playing[0]
            if playing[0]:
                ani.event_source.start()
                print("Playing")
            else:
                ani.event_source.stop()
                print("Paused")

    def on_key(event):
        if not playing[0]:
            if event.key == "right":
                curr_frame[0] = (curr_frame[0] + 1) % num_frames
                update_display(curr_frame[0])
            elif event.key == "left":
                curr_frame[0] = (curr_frame[0] - 1) % num_frames
                update_display(curr_frame[0])

    def dynamic_frame_gen():
        while True:
            if playing[0]:
                yield curr_frame[0]
                curr_frame[0] = (curr_frame[0] + 1) % num_frames
            else:
                yield curr_frame[0]

    ani = animation.FuncAnimation(
        fig, 
        animate, 
        frames=dynamic_frame_gen(), 
        interval=100, 
        blit=False,
        cache_frame_data=False
    )

    # Connect keypress events to the figure
    fig.canvas.mpl_connect("key_press_event", toggle_play)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Show the first frame
    update_display(curr_frame[0])

    if save:
        if not save_name:
            raise ValueError("save_name cannot be empty when save=True")
        ani.save(save_name, writer=animation.PillowWriter(fps=60))

    plt.show()