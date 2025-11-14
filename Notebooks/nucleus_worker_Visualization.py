"""
Multi-channel nucleus analysis worker for parallel processing.
Compatible with Windows multiprocessing and existing task structure.
NOW RETURNS WATERSHED LABELS FOR GLOBAL VISUALIZATION
"""
import numpy as np
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from collections import Counter
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage import exposure
import os


# ===============================================================
# HELPER FUNCTIONS (module-level for multiprocessing)
# ===============================================================

# ===============================================================
# SAVE GLOBAL VISUALIZATION (INCLUDES REAL WATERSHED)
# ===============================================================

def save_global_visualizations(original_image, foci_tritc, foci_fitc, 
                               watershed_labels_tritc, watershed_labels_fitc,
                               well_number, position_number, base_name, output_root):
    """
    Generate 4 full-field visualizations with proper filenames.
    Watershed images show filled colored regions (no borders).
    
    This function creates comprehensive visualizations of the entire microscopy field:
    - Two images showing detected foci as colored dots overlaid on the DAPI background
    - Two images showing watershed segmentation regions as filled colored areas
    
    Parameters:
    -----------
    original_image : ndarray
        Raw DAPI or merged channel image for background display
    foci_tritc : list of tuples
        TRITC foci coordinates [(y, x), ...] from all nuclei in the image
    foci_fitc : list of tuples
        FITC foci coordinates [(y, x), ...] from all nuclei in the image
    watershed_labels_tritc : ndarray
        Labeled watershed segmentation for TRITC channel (entire image, all nuclei combined)
    watershed_labels_fitc : ndarray
        Labeled watershed segmentation for FITC channel (entire image, all nuclei combined)
    well_number : str
        Well identifier extracted from filename (e.g., '00044')
    position_number : str
        Position identifier extracted from filename (e.g., '00021')
    base_name : str
        Original filename base (e.g., 'ATR2_24h--W00044--P00021--Z00000--T00000--')
    output_root : str
        Root directory for saving images (creates debug_images_global subdirectory)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import os
    from skimage import exposure

    try:
        # Create output directory if it doesn't exist
        # This folder will contain for both the FITC and the TRITC channel Images of the foci locations and their detected area
        debug_dir = os.path.join(output_root, "Full_Images_Foci")
        os.makedirs(debug_dir, exist_ok=True)

        # Normalize the background image to 0-1 range for consistent display
        # This ensures the grayscale background is properly visible regardless of original intensity range
        vis_img = exposure.rescale_intensity(original_image, in_range='image', out_range=(0, 1))

        # ================================================================
        # 1️⃣ TRITC FOCI OVERLAY (RED DOTS ON GRAY BACKGROUND)
        # ================================================================
        # Creates an image showing all detected TRITC foci as red dots
        # This gives an overview of TRITC foci distribution across the entire field
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img, cmap='gray')  # Display the DAPI background in grayscale
        
        # Plot each TRITC focus as a small red dot
        # markersize=0.35 makes dots visible but not overwhelming
        # alpha=0.7 adds slight transparency to see overlapping foci
        for (y, x) in foci_tritc:
            plt.plot(x, y, 'ro', markersize=0.35, alpha=0.7)
        
        plt.title(f"TRITC Foci | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')  # Remove axis labels for cleaner image
        plt.tight_layout()
        
        # Save with original filename convention + channel identifier
        filename = f"{base_name}TRITC_foci.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"  ✓ Saved: {filename}")

        # ================================================================
        # 2️⃣ FITC FOCI OVERLAY (GREEN DOTS ON GRAY BACKGROUND)
        # ================================================================
        # Creates an image showing all detected FITC foci as green dots
        # Same logic as TRITC but with green color ('go') for FITC channel
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img, cmap='gray')
        
        # Plot each FITC focus as a small green dot
        for (y, x) in foci_fitc:
            plt.plot(x, y, 'go', markersize=0.35, alpha=0.7)
        
        plt.title(f"FITC Foci | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}FITC_foci.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

        # ================================================================
        # 3️⃣ TRITC WATERSHED (FILLED COLORED REGIONS - NO BORDERS)
        # ================================================================
        # Shows watershed segmentation regions as filled colored areas
        # Each individual focus region gets a unique random color for easy distinction
        plt.figure(figsize=(10, 10))
        
        # Get the maximum label number to determine how many colors we need
        # Each watershed region has a unique label (1, 2, 3, ...)
        num_labels_tritc = int(watershed_labels_tritc.max())
        if num_labels_tritc > 0:
            # Generate a random color for each watershed region
            # This ensures neighboring foci are visually distinguishable
            np.random.seed(42)  # For reproducibility across runs
            colors = np.random.rand(num_labels_tritc + 1, 3)  # +1 because label 0 is background
            colors[0] = [0, 0, 0]  # Force background (label 0) to be black
            cmap_tritc = mcolors.ListedColormap(colors)  # Create custom colormap from random colors
            
            # Display as two layers:
            # Layer 1: DAPI background at 50% opacity (alpha=0.5) to see nucleus structure
            # Layer 2: Colored watershed regions at 70% opacity (alpha=0.7) overlaid on top
            plt.imshow(vis_img, cmap='gray', alpha=0.5)  # Semi-transparent background
            plt.imshow(watershed_labels_tritc, cmap=cmap_tritc, alpha=0.7, interpolation='nearest')
        else:
            # If no foci were detected, just show the background image
            plt.imshow(vis_img, cmap='gray')
        
        plt.title(f"TRITC Watershed | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}TRITC_watershed.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

        # ================================================================
        # 4️⃣ FITC WATERSHED (FILLED COLORED REGIONS - NO BORDERS)
        # ================================================================
        # Same logic as TRITC watershed but for FITC channel
        # Uses different random seed (43) to ensure different colors than TRITC
        plt.figure(figsize=(10, 10))
        
        num_labels_fitc = int(watershed_labels_fitc.max())
        if num_labels_fitc > 0:
            # Generate random colors for FITC watershed regions
            # Different seed (43 vs 42) ensures FITC colors differ from TRITC
            np.random.seed(43)  # Different seed for different color palette
            colors = np.random.rand(num_labels_fitc + 1, 3)
            colors[0] = [0, 0, 0]  # Background is black
            cmap_fitc = mcolors.ListedColormap(colors)
            
            # Display with same transparency settings as TRITC
            plt.imshow(vis_img, cmap='gray', alpha=0.5)
            plt.imshow(watershed_labels_fitc, cmap=cmap_fitc, alpha=0.7, interpolation='nearest')
        else:
            plt.imshow(vis_img, cmap='gray')
        
        plt.title(f"FITC Watershed | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}FITC_watershed.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {filename}")

    except Exception as e:
        # If any error occurs, print detailed error message with traceback
        # This helps debug issues without crashing the entire analysis pipeline
        print(f"⚠️ Failed to save global visualizations for Well {well_number}, Position {position_number}: {e}")
        import traceback
        traceback.print_exc()



def compute_circularity(area, perimeter):
    """
    Calculate circularity factor: 4π * area / perimeter²
    
    Circularity is a shape descriptor that indicates how circular an object is:
    - Value of 1.0 = perfect circle
    - Values approaching 0 = elongated or irregular shapes
    
    This is used to characterize both nuclei and individual foci shapes.
    Formula: C = 4π * A / P²
    where A = area in pixels, P = perimeter in pixels
    
    Parameters:
    -----------
    area : float
        Area of the region in pixels
    perimeter : float
        Perimeter of the region in pixels
        
    Returns:
    --------
    float : Circularity value between 0 and 1
    """
    # Avoid division by zero if perimeter is zero (shouldn't happen but safety check)
    if perimeter == 0:
        return 0.0
    
    # Standard circularity formula
    # A perfect circle has maximum circularity (approaches 1.0)
    # Irregular or elongated shapes have lower values
    return (4 * np.pi * area) / (perimeter ** 2)


def compute_local_percentiles_for_candidates(image, coords, unique_percentiles):
    """
    Calculate local background percentiles around each candidate focus.
    
    For each candidate focus position, this function extracts a small 13x13 pixel
    neighborhood (±6 pixels in each direction) and computes background intensity
    percentiles. This local background measurement is crucial for distinguishing
    true foci from background noise.
    
    Why local background matters:
    - Microscopy images often have uneven illumination
    - Using global thresholds would miss dim foci in bright areas or detect
      noise in dark areas
    - Local background allows adaptive thresholding
    
    Parameters:
    -----------
    image : ndarray
        The image being analyzed (isolated nucleus or filtered version)
    coords : ndarray of shape (N, 2)
        Array of (y, x) coordinates for N candidate foci
    unique_percentiles : array-like
        List of percentile values to compute (e.g., [50, 75, 90])
        
    Returns:
    --------
    ndarray of shape (N, P) : Local percentile values for each candidate
        where N = number of candidates, P = number of percentiles requested
    """
    N = coords.shape[0]  # Number of candidate foci
    P = len(unique_percentiles)  # Number of percentile thresholds to compute
    out = np.zeros((N, P), dtype=float)  # Preallocate output array
    
    # Process each candidate focus location
    for i, (y, x) in enumerate(coords):
        # Define 13x13 pixel neighborhood around the candidate focus
        # ±6 pixels in each direction, bounded by image edges
        y_min, y_max = max(0, y - 6), min(image.shape[0], y + 7)
        x_min, x_max = max(0, x - 6), min(image.shape[1], x + 7)
        
        # Extract the local neighborhood square
        square = image[y_min:y_max, x_min:x_max]
        
        # Handle edge case: if somehow we got an empty region (shouldn't happen)
        # just use the center pixel value
        if square.size == 0:
            out[i, :] = image[y, x]
        else:
            # Compute the requested percentiles of the local background
            # These percentiles represent local background intensity levels
            # For example, 75th percentile = threshold where 75% of pixels are dimmer
            out[i, :] = np.percentile(square, unique_percentiles)
    
    return out


def apply_foci_filters(p_idx, bright_pcts, contrast_threshs, percentile_vals,
                       min_brightness_per_param, bright_to_idx,
                       unf_intensities, filt_intensities,
                       local_percentiles_unf, local_percentiles_filt,
                       distances, unf_yx, tolerance):
    """
    Apply filtering to detect valid foci for one parameter combination.
    
    This function tests whether candidate foci pass both absolute brightness and
    local contrast criteria for a specific set of detection parameters. It's called
    repeatedly with different parameter combinations to assess detection robustness.
    
    The filtering process:
    1. ABSOLUTE brightness filter: Is the focus bright enough overall?
    2. LOCAL CONTRAST filter: Is the focus brighter than its local background?
    3. SPATIAL MATCHING: Does the same focus appear in both filtered and unfiltered images?
    
    Only foci that pass all three criteria are considered valid detections.
    
    Parameters:
    -----------
    p_idx : int
        Index of the parameter combination being tested
    bright_pcts : array
        Array of brightness percentile thresholds (one per parameter combo)
    contrast_threshs : array
        Array of contrast threshold multipliers (one per parameter combo)
    percentile_vals : array
        Array of global percentile values for absolute brightness (one per combo)
    min_brightness_per_param : array
        Precomputed minimum brightness thresholds from global percentiles
    bright_to_idx : dict
        Mapping from brightness percentile to column index in local_percentiles arrays
    unf_intensities : array
        Peak intensities in the unfiltered image
    filt_intensities : array
        Peak intensities in the filtered (DoG) image
    local_percentiles_unf : ndarray
        Local background percentiles for unfiltered peaks
    local_percentiles_filt : ndarray
        Local background percentiles for filtered peaks
    distances : ndarray
        Distance matrix between unfiltered and filtered peak coordinates
    unf_yx : ndarray
        Coordinates of unfiltered peaks
    tolerance : int
        Maximum pixel distance to consider two peaks as "the same" (typically 2)
        
    Returns:
    --------
    tuple : (confirmed_coords, count)
        confirmed_coords : ndarray of shape (M, 2)
            Coordinates of foci that passed all filters
        count : int
            Number of confirmed foci
    """
    # Extract the specific parameters for this iteration
    bright_pct = bright_pcts[p_idx]           # Local background percentile threshold
    contrast_thresh = contrast_threshs[p_idx]  # Contrast multiplier (e.g., 2.5x background)
    min_brightness = min_brightness_per_param[p_idx]  # Absolute brightness threshold
    
    # Map the brightness percentile to the correct column in the local percentiles array
    bright_key = np.round(bright_pct, 6)  # Round to avoid floating point comparison issues
    b_idx = bright_to_idx[bright_key]
    
    # ---- STEP 1: ABSOLUTE BRIGHTNESS FILTER ----
    # Check if peak intensities exceed the global minimum brightness threshold
    # This filters out very dim spots that are likely noise regardless of local context
    unf_mask_abs = unf_intensities >= min_brightness
    filt_mask_abs = filt_intensities >= min_brightness
    
    # Early exit: if no peaks pass absolute brightness in either image, return empty result
    if not np.any(unf_mask_abs) or not np.any(filt_mask_abs):
        return np.array([]).reshape(0, 2), 0
    
    # ---- STEP 2: LOCAL CONTRAST FILTER ----
    # Check if peaks are sufficiently brighter than their local background
    # Extract the local background value at the specified percentile for each peak
    unf_local_bg = local_percentiles_unf[:, b_idx]
    filt_local_bg = local_percentiles_filt[:, b_idx]
    
    # Apply contrast threshold: peak must be > (local_background × contrast_thresh)
    # Example: if contrast_thresh=2.5, peak must be 2.5× brighter than local background
    unf_mask_con = unf_intensities > (unf_local_bg * contrast_thresh)
    filt_mask_con = filt_intensities > (filt_local_bg * contrast_thresh)
    
    # ---- STEP 3: COMBINE FILTERS ----
    # A valid peak must pass BOTH absolute brightness AND local contrast filters
    unf_final_mask = unf_mask_abs & unf_mask_con
    filt_final_mask = filt_mask_abs & filt_mask_con
    
    # Get the indices of peaks that passed all filters
    unf_idxs = np.where(unf_final_mask)[0]
    filt_idxs = np.where(filt_final_mask)[0]
    
    # Early exit: if no peaks passed filters in either image, return empty result
    if unf_idxs.size == 0 or filt_idxs.size == 0:
        return np.array([]).reshape(0, 2), 0
    
    # ---- STEP 4: SPATIAL MATCHING ----
    # Match filtered and unfiltered foci: only keep foci that appear in BOTH images
    # This confirms that the focus is a real feature, not an artifact of filtering
    
    # Extract the sub-matrix of distances between valid unfiltered and filtered peaks
    distances_sub = distances[unf_idxs][:, filt_idxs]
    
    # For each valid unfiltered peak, find the distance to its nearest valid filtered peak
    nearest_dist = np.min(distances_sub, axis=1)
    
    # Keep only unfiltered peaks that have a matching filtered peak within tolerance
    # Tolerance of 2 pixels allows for slight spatial shifts due to filtering
    confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
    
    # Get the final coordinates of confirmed foci
    confirmed_coords = unf_yx[confirmed_unf_idxs]
    
    # Return both the coordinates and the count
    return confirmed_coords, len(confirmed_coords)

# ===============================================================
# INTENSITY ANALYSIS
# ===============================================================

def analyze_channel_intensity(nucleus_mask, image, channel_name):
    """
    Compute total and mean intensity for one nucleus in one channel.
    
    This function calculates basic intensity statistics for an entire nucleus region.
    These measurements represent the overall signal in the nucleus, including both
    background and any foci present.
    
    Why measure whole-nucleus intensity:
    - Provides context for foci measurements (foci intensity relative to background)
    - Detects overall expression levels or staining intensity
    - Can indicate technical issues (e.g., uneven staining)
    
    Parameters:
    -----------
    nucleus_mask : ndarray (boolean)
        Binary mask indicating which pixels belong to this nucleus
    image : ndarray
        The image to measure (should be float, 0-1 range)
    channel_name : str
        Name of the channel (e.g., 'TRITC', 'FITC', 'Cy5', 'DAPI')
        
    Returns:
    --------
    dict : Dictionary with two keys:
        '{channel_name}_total_intensity' : Sum of all pixel intensities in nucleus
        '{channel_name}_mean_intensity' : Average pixel intensity in nucleus
    """
    # Extract only the pixels belonging to this nucleus
    nucleus_pixels = image[nucleus_mask]
    
    # Calculate total intensity: sum of all pixel values
    # This represents the total amount of signal in the nucleus
    # Higher values = more fluorescence (more protein, more RNA, etc.)
    total_intensity = float(np.sum(nucleus_pixels))
    
    # Calculate mean intensity: average pixel value
    # This represents the average brightness, normalized by nucleus size
    # Useful for comparing nuclei of different sizes
    mean_intensity = float(np.mean(nucleus_pixels))
    
    # Return as dictionary with channel-specific keys for easy DataFrame creation
    return {
        f"{channel_name}_total_intensity": total_intensity,
        f"{channel_name}_mean_intensity": mean_intensity,
    }


# ===============================================================
# FOCI DETECTION FOR ONE CHANNEL (MODIFIED TO RETURN WATERSHED)
# ===============================================================

def detect_foci_single_channel(
    nucleus_mask, image, original_image, channel_name, cell_id,
    valid_param_samples, total_iterations,
    well_number=None, position_number=None
):
    """
    Detect foci in a single nucleus region for one channel.
    Returns: (foci_list, summary_dict, watershed_labels)
    
    NEW: Now returns watershed_labels for global visualization
    """
    isolated_img = img_as_float(image.copy())
    isolated_img[~nucleus_mask] = 0
    
    if isolated_img.max() == 0:
        return [], {}, None  # ← CHANGED: Added None for watershed
    
    # Apply DoG filter
    filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
    filtered_img = np.clip(filtered_img, 0, None)
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', 
                                             out_range=(0, isolated_img.max()))
    
    # Extract parameters
    bright_pcts = valid_param_samples[:, 0]
    contrast_threshs = valid_param_samples[:, 1]
    percentile_vals = valid_param_samples[:, 2]
    
    # Use original_image for global percentile calculations
    pos_pixels = original_image[original_image > 0]
    if pos_pixels.size == 0:
        return [], {}, None  # ← CHANGED: Added None
    
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    global_min_brightness = np.min(min_brightness_per_param)
    
    # Find candidate foci
    candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                        threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                          threshold_abs=global_min_brightness)
    
    if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
        return [], {}, None  # ← CHANGED: Added None
    
    # Extract coordinates and intensities
    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    filt_intensities = filtered_img[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_img[unf_yx[:, 0], unf_yx[:, 1]]
    
    # Prepare brightness percentile mapping
    unique_brights = np.unique(np.round(bright_pcts, 6))
    bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}
    
    # Compute local backgrounds
    local_percentiles_unf = compute_local_percentiles_for_candidates(
        isolated_img, unf_yx, unique_brights)
    local_percentiles_filt = compute_local_percentiles_for_candidates(
        filtered_img, filt_yx, unique_brights)
    
    distances = cdist(unf_yx, filt_yx)
    tolerance = 2
    
    # Test all parameter combinations
    foci_counts = []
    all_detected_foci = []
    
    for p_idx in range(len(valid_param_samples)):
        confirmed_coords, count = apply_foci_filters(
            p_idx, bright_pcts, contrast_threshs, percentile_vals,
            min_brightness_per_param, bright_to_idx,
            unf_intensities, filt_intensities,
            local_percentiles_unf, local_percentiles_filt,
            distances, unf_yx, tolerance
        )
        foci_counts.append(count)
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord))
    
    if not foci_counts:
        return [], {}, None  # ← CHANGED: Added None
    
    # Calculate statistics
    foci_detection_count = Counter(all_detected_foci)
    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = int(min(foci_counts))
    max_foci = int(max(foci_counts))
    
    # Run watershed with best parameters
    max_idx = np.argmax(foci_counts)
    best_params = valid_param_samples[max_idx]
    best_bright_pct = best_params[0]
    best_contrast_thresh = best_params[1]
    percentile_val = best_params[2]
    
    # Calculate minimum brightness from original image
    min_brightness = np.percentile(original_image[original_image > 0], percentile_val)
    
    # Find candidate peaks in both filtered and unfiltered
    coordinates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                           threshold_abs=min_brightness)
    coordinates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                         threshold_abs=min_brightness)
    
    if coordinates_unfiltered.size == 0 or coordinates_filtered.size == 0:
        return [], {}, None  # ← CHANGED: Added None
    
    # Extract intensities at peak locations
    unf_y, unf_x = coordinates_unfiltered[:, 0], coordinates_unfiltered[:, 1]
    filt_y, filt_x = coordinates_filtered[:, 0], coordinates_filtered[:, 1]
    unf_peak_intensities = isolated_img[unf_y, unf_x]
    filt_peak_intensities = filtered_img[filt_y, filt_x]
    
    # Apply ABSOLUTE brightness filter
    unf_bright_mask = unf_peak_intensities >= min_brightness
    filt_bright_mask = filt_peak_intensities >= min_brightness
    
    # Apply LOCAL BACKGROUND contrast filter
    unf_local_bg = compute_local_percentiles_for_candidates(
        isolated_img, coordinates_unfiltered, [best_bright_pct])[:, 0]
    filt_local_bg = compute_local_percentiles_for_candidates(
        filtered_img, coordinates_filtered, [best_bright_pct])[:, 0]
    
    unf_contrast_mask = unf_peak_intensities > (unf_local_bg * best_contrast_thresh)
    filt_contrast_mask = filt_peak_intensities > (filt_local_bg * best_contrast_thresh)
    
    # Combine all filters
    unf_final_mask = unf_bright_mask & unf_contrast_mask
    filt_final_mask = filt_bright_mask & filt_contrast_mask
    
    coordinates_unfiltered_filtered = coordinates_unfiltered[unf_final_mask]
    coordinates_filtered_filtered = coordinates_filtered[filt_final_mask]
    
    if coordinates_unfiltered_filtered.size == 0 or coordinates_filtered_filtered.size == 0:
        return [], {}, None  # ← CHANGED: Added None
    
    # Match filtered and unfiltered with tolerance
    distances_final = cdist(coordinates_unfiltered_filtered, coordinates_filtered_filtered)
    final_coords = coordinates_unfiltered_filtered[np.min(distances_final, axis=1) <= tolerance]
    
    if len(final_coords) == 0:
        return [], {}, None  # ← CHANGED: Added None
    
    # Perform watershed segmentation
    gradient = filters.sobel(filtered_img) # Using the DoG filtered image to create topographical map for watershed
    
    markers = np.zeros_like(isolated_img, dtype=int)
    for idx, (y, x) in enumerate(final_coords, start=1):
        markers[y, x] = idx
    
    watershed_threshold = min_brightness * 1.5
    watershed_mask = (isolated_img > watershed_threshold) | (markers > 0)
    
    water_labels = watershed(gradient, markers, mask=watershed_mask)
    
    # Measure each focus
    foci_list = []
    DETECTION_THRESHOLD = 50.0
    confident_foci_intensities = []
    
    for idx, (y, x) in enumerate(final_coords):
        region_id = water_labels[y, x]
        spot_mask = (water_labels == region_id)
        spot_area = int(np.sum(spot_mask))
        spot_intensity = float(np.sum(isolated_img[spot_mask]))
        spot_mean_intensity = float(np.mean(isolated_img[spot_mask]))
        detection_prob = (foci_detection_count.get((y, x), 0) / total_iterations) * 100
        
        # Calculate circularity for this focus
        focus_props = measure.regionprops(spot_mask.astype(int))
        if len(focus_props) > 0:
            focus_perimeter = focus_props[0].perimeter
            focus_circularity = compute_circularity(spot_area, focus_perimeter)
        else:
            focus_circularity = 0.0
        
        if detection_prob > 0:
            if detection_prob >= DETECTION_THRESHOLD:
                confident_foci_intensities.append(spot_intensity)
            
            foci_list.append({
                'cell_num': cell_id,
                'centr_y': int(y),
                'centr_x': int(x),
                'foci_area': spot_area,
                'foci_circularity': focus_circularity,
                'foci_total_intensity': spot_intensity,
                'foci_mean_intensity': spot_mean_intensity,
                'detection_prob': detection_prob,
                'channel': channel_name
            })
    
    # Calculate nucleus-level statistics
    sum_foci_intensity = float(np.sum(confident_foci_intensities)) if confident_foci_intensities else 0.0
    mean_foci_intensity = float(np.mean(confident_foci_intensities)) if confident_foci_intensities else 0.0
    num_confident_foci = len(confident_foci_intensities)
    
    summary = {
        f"{channel_name}_mean_foci": mean_foci,
        f"{channel_name}_std_foci": std_foci,
        f"{channel_name}_min_foci": min_foci,
        f"{channel_name}_max_foci": max_foci,
        f"{channel_name}_confident_foci_count": num_confident_foci,
        f"{channel_name}_sum_foci_intensity": sum_foci_intensity,
        f"{channel_name}_mean_foci_intensity": mean_foci_intensity,
    }

    # ← NEW: Return watershed labels for global visualization
    return foci_list, summary, water_labels


# ===============================================================
# MAIN WORKER FUNCTION (MODIFIED TO RETURN WATERSHED LABELS)
# ===============================================================

def process_single_nucleus(args):
    """
    Process one nucleus across all provided channels.
    
    NOW RETURNS: (foci_data_list, nuclei_data_list, watershed_data_list)
    watershed_data_list contains dictionaries with watershed labels for each channel
    """
    (cellnumber, masks, channel_images, valid_param_samples, 
     total_iterations, well_number, position_number) = args
    
    # Create mask for current nucleus
    masks_reduced = (masks == cellnumber)
    
    if not np.any(masks_reduced):
        return [], [], []  # ← CHANGED: Added empty list for watershed
    
    # Initialize result containers
    foci_data_list = []
    nucleus_data = {
        'cell_num': cellnumber,
        'Well': well_number,
        'Position': position_number
    }
    watershed_data_list = []  # ← NEW: Store watershed labels here
    
    # Extract DAPI properties (nucleus shape/size)
    if 'DAPI' in channel_images:
        nucleus_props = measure.regionprops(masks_reduced.astype(int))
        if len(nucleus_props) > 0:
            region = nucleus_props[0]
            nucleus_area = region.area
            nucleus_perimeter = region.perimeter
            nucleus_circularity = compute_circularity(nucleus_area, nucleus_perimeter)
            
            nucleus_data.update({
                'DAPI_area': nucleus_area,
                'DAPI_perimeter': nucleus_perimeter,
                'DAPI_circularity': nucleus_circularity,
                'centr_y': region.centroid[0],
                'centr_x': region.centroid[1],
            })
    
    # Process each channel
    for channel_name, channel_image in channel_images.items():
        channel_image_float = img_as_float(channel_image)
        
        # Calculate intensity for all channels
        intensity_data = analyze_channel_intensity(masks_reduced, channel_image_float, channel_name)
        nucleus_data.update(intensity_data)
        
        # Detect foci ONLY for TRITC and FITC
        if channel_name in ["TRITC", "FITC"]:
            foci_list, foci_summary, water_labels = detect_foci_single_channel(  # ← CHANGED: Now gets 3 returns
                masks_reduced,
                channel_image_float,
                channel_image_float,
                channel_name,
                cellnumber,
                valid_param_samples,
                total_iterations,
                well_number,
                position_number
            )
            
            # Add well and position to each focus
            for focus in foci_list:
                focus['Well'] = well_number
                focus['Position'] = position_number
            
            foci_data_list.extend(foci_list)
            nucleus_data.update(foci_summary)
            
            # ← NEW: Store watershed labels if valid
            if water_labels is not None:
                watershed_data_list.append({
                    'cell_id': cellnumber,
                    'channel': channel_name,
                    'labels': water_labels,
                    'mask': masks_reduced  # Include the nucleus mask for proper placement
                })
    
    nuclei_data_list = [nucleus_data]
    
    # ← CHANGED: Now returns 3 items instead of 2
    return foci_data_list, nuclei_data_list, watershed_data_list