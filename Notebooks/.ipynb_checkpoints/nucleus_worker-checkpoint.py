"""
Multi-channel nucleus analysis worker for parallel processing.
Compatible with Windows multiprocessing and existing task structure.
"""
import numpy as np
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from collections import Counter


# ===============================================================
# HELPER FUNCTIONS (module-level for multiprocessing)
# ===============================================================

def compute_circularity(area, perimeter):
    """Calculate circularity factor: 4π * area / perimeter^2"""
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)



def compute_local_percentiles_for_candidates(image, coords, unique_percentiles):
    """Calculate local background percentiles around each candidate focus."""
    N = coords.shape[0]
    P = len(unique_percentiles)
    out = np.zeros((N, P), dtype=float)
    for i, (y, x) in enumerate(coords):
        y_min, y_max = max(0, y - 6), min(image.shape[0], y + 7)
        x_min, x_max = max(0, x - 6), min(image.shape[1], x + 7)
        square = image[y_min:y_max, x_min:x_max]
        if square.size == 0:
            out[i, :] = image[y, x]
        else:
            out[i, :] = np.percentile(square, unique_percentiles)
    return out


def apply_foci_filters(p_idx, bright_pcts, contrast_threshs, percentile_vals,
                       min_brightness_per_param, bright_to_idx,
                       unf_intensities, filt_intensities,
                       local_percentiles_unf, local_percentiles_filt,
                       distances, unf_yx, tolerance):
    """Apply filtering to detect valid foci for one parameter combination."""
    bright_pct = bright_pcts[p_idx]
    contrast_thresh = contrast_threshs[p_idx]
    min_brightness = min_brightness_per_param[p_idx]
    
    bright_key = np.round(bright_pct, 6)
    b_idx = bright_to_idx[bright_key]
    
    # Absolute brightness filters
    unf_mask_abs = unf_intensities >= min_brightness
    filt_mask_abs = filt_intensities >= min_brightness
    if not np.any(unf_mask_abs) or not np.any(filt_mask_abs):
        return np.array([]).reshape(0, 2), 0
    
    # Contrast filters
    unf_local_bg = local_percentiles_unf[:, b_idx]
    filt_local_bg = local_percentiles_filt[:, b_idx]
    unf_mask_con = unf_intensities > (unf_local_bg * contrast_thresh)
    filt_mask_con = filt_intensities > (filt_local_bg * contrast_thresh)
    
    # Combine filters
    unf_final_mask = unf_mask_abs & unf_mask_con
    filt_final_mask = filt_mask_abs & filt_mask_con
    
    unf_idxs = np.where(unf_final_mask)[0]
    filt_idxs = np.where(filt_final_mask)[0]
    if unf_idxs.size == 0 or filt_idxs.size == 0:
        return np.array([]).reshape(0, 2), 0
    
    # Match filtered and unfiltered foci
    distances_sub = distances[unf_idxs][:, filt_idxs]
    nearest_dist = np.min(distances_sub, axis=1)
    confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
    confirmed_coords = unf_yx[confirmed_unf_idxs]
    
    return confirmed_coords, len(confirmed_coords)


# ===============================================================
# INTENSITY ANALYSIS
# ===============================================================

def analyze_channel_intensity(nucleus_mask, image, channel_name):
    """Compute total and mean intensity for one nucleus in one channel."""
    nucleus_pixels = image[nucleus_mask]
    total_intensity = float(np.sum(nucleus_pixels))
    mean_intensity = float(np.mean(nucleus_pixels))
    
    return {
        f"{channel_name}_total_intensity": total_intensity,
        f"{channel_name}_mean_intensity": mean_intensity,
    }
    

# ===============================================================
# FOCI DETECTION FOR ONE CHANNEL
# ===============================================================

def detect_foci_single_channel(nucleus_mask, image, original_image, channel_name, cell_id,
                                valid_param_samples, total_iterations):
    """
    Detect foci in a single nucleus region for one channel.
    Returns: (foci_list, summary_dict)
    """
    isolated_img = img_as_float(image.copy())
    isolated_img[~nucleus_mask] = 0
    
    if isolated_img.max() == 0:
        return [], {}
    
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
        return [], {}
    
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    global_min_brightness = np.min(min_brightness_per_param)
    
    # Find candidate foci
    candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                        threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                          threshold_abs=global_min_brightness)
    
    if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
        return [], {}
    
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
        return [], {}
    
    # Calculate statistics
    foci_detection_count = Counter(all_detected_foci)
    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = int(min(foci_counts))
    max_foci = int(max(foci_counts))
    
    # Run watershed with best parameters
    max_idx = np.argmax(foci_counts)
    best_params = valid_param_samples[max_idx]
    percentile_val = best_params[2]
    min_brightness = np.percentile(original_image[original_image > 0], percentile_val)
    
    coordinates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                           threshold_abs=min_brightness)
    coordinates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                         threshold_abs=min_brightness)
    
    if coordinates_unfiltered.size > 0 and coordinates_filtered.size > 0:
        distances_final = cdist(coordinates_unfiltered, coordinates_filtered)
        final_coords = coordinates_unfiltered[np.min(distances_final, axis=1) <= tolerance]
    else:
        final_coords = np.empty((0, 2), int)
    
    # Perform watershed segmentation
    gradient = filters.sobel(filtered_img)
    markers = np.zeros_like(filtered_img, dtype=int)
    for idx, (y, x) in enumerate(final_coords, start=1):
        markers[y, x] = idx
    
    watershed_mask = (filtered_img > min_brightness) | (markers > 0)
    water_labels = watershed(gradient, markers, mask=watershed_mask)
    
    # Measure each focus
    foci_list = []
    
    # Set detection probability threshold for "confident" foci
    # Adjust this value based on your needs (50-70% is typical)
    DETECTION_THRESHOLD = 50.0  # Only include foci detected in ≥50% of iterations
    
    confident_foci_intensities = []  # For nucleus-level summary
    
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
        
        # Track intensity for confident foci only (for nucleus summary)
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
    
    # Calculate nucleus-level foci intensity statistics (ONLY from confident foci)
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
    
    return foci_list, summary



# ===============================================================
# MAIN WORKER FUNCTION 
# ===============================================================

def process_single_nucleus(args):
    """
    Process one nucleus across all provided channels.
    Compatible with existing task structure: (cellnum, masks, channel_dict, valid_param_samples, 
                                             total_iterations, well_number, position_number)
    
    Args:
        tuple: (cellnumber, masks, channel_images_dict, valid_param_samples, 
                total_iterations, well_number, position_number)
        
        channel_images_dict should be: {'TRITC': TRITC_pic, 'FITC': FITC_pic, 'Cy5': Cy5_pic, 'DAPI': DAPI_pic}
    
    Returns:
        tuple: (foci_data_list, nuclei_data_list)
            - foci_data_list: list of dicts with individual foci (includes 'channel' key)
            - nuclei_data_list: list with single dict containing nucleus summary
    """
    (cellnumber, masks, channel_images, valid_param_samples, 
     total_iterations, well_number, position_number) = args
    
    # Create mask for current nucleus
    masks_reduced = (masks == cellnumber)
    
    if not np.any(masks_reduced):
        return [], []
    
    # Initialize result containers
    foci_data_list = []
    nucleus_data = {
        'cell_num': cellnumber,
        'Well': well_number,
        'Position': position_number
    }
    
    # Extract DAPI properties (nucleus shape/size) - ENHANCED
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
        # Ensure image is float
        channel_image_float = img_as_float(channel_image)
        
        # Calculate intensity for all channels
        intensity_data = analyze_channel_intensity(masks_reduced, channel_image_float, channel_name)
        nucleus_data.update(intensity_data)
        
        # Detect foci ONLY for TRITC and FITC (FIXED LOGIC!)
        if channel_name in ["TRITC", "FITC"]:
            foci_list, foci_summary = detect_foci_single_channel(
                masks_reduced, 
                channel_image_float, 
                channel_image_float,  # Use same image for global percentiles
                channel_name, 
                cellnumber,
                valid_param_samples, 
                total_iterations
            )
            
            # Add well and position to each focus
            for focus in foci_list:
                focus['Well'] = well_number
                focus['Position'] = position_number
            
            foci_data_list.extend(foci_list)
            nucleus_data.update(foci_summary)
    
    nuclei_data_list = [nucleus_data]
    
    return foci_data_list, nuclei_data_list


# ===============================================================
# EXAMPLE: How to modify your main script
# ===============================================================

"""
INTEGRATION GUIDE:

In your main loop, change from:
    tasks = [
        (cellnum, masks, TRITC_pic, valid_param_samples, total_iterations, well_number, position_number)
        for cellnum in cell_numbers
    ]

To:
    # Create channel dictionary
    channel_images = {
        'TRITC': TRITC_pic,
        'FITC': FITC_pic,
        'Cy5': Cy5_pic,
        'DAPI': DAPI_pic
    }
    
    tasks = [
        (cellnum, masks, channel_images, valid_param_samples, total_iterations, well_number, position_number)
        for cellnum in cell_numbers
    ]

Everything else stays the same! The worker will return:
    - foci_data_list: now includes a 'channel' column to distinguish TRITC/FITC/Cy5 foci
    - nuclei_data_list: includes columns like:
        * DAPI_total_intensity, DAPI_mean_intensity
        * TRITC_total_intensity, TRITC_mean_intensity, TRITC_mean_foci, TRITC_std_foci, etc.
        * FITC_total_intensity, FITC_mean_intensity, FITC_mean_foci, FITC_std_foci, etc.
        * Cy5_total_intensity, Cy5_mean_intensity, Cy5_mean_foci, Cy5_std_foci, etc.

Your existing CSV saving code will work without modification!

OPTIONAL: To save foci by channel separately:
    if all_foci_data:
        all_foci_df = pd.DataFrame(all_foci_data)
        
        # Save all foci together
        all_foci_df.to_csv('all_foci_combined.csv', index=False)
        
        # Or save by channel
        for channel in ['TRITC', 'FITC', 'Cy5']:
            channel_foci = all_foci_df[all_foci_df['channel'] == channel]
            save_dataframe_to_csv(channel_foci, folder_path, '3_param_iteration', f'foci_{channel}')
"""