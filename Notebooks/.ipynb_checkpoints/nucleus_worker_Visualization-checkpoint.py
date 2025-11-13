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
"""
Enhanced visualization function for full-field foci and watershed overlays.
Saves 4 separate images per position:
1. TRITC foci on original image
2. FITC foci on original image  
3. TRITC watershed outlines (yellow)
4. FITC watershed outlines (yellow)
"""

def save_global_visualizations(original_image, foci_tritc, foci_fitc, 
                               watershed_labels_tritc, watershed_labels_fitc,
                               well_number, position_number, base_name, output_root):
    """
    Generate 4 full-field visualizations with proper filenames.
    
    Parameters:
    -----------
    original_image : ndarray
        Raw DAPI or merged channel image for background
    foci_tritc : list of tuples
        TRITC foci coordinates [(y, x), ...]
    foci_fitc : list of tuples
        FITC foci coordinates [(y, x), ...]
    watershed_labels_tritc : ndarray
        Labeled watershed segmentation for TRITC channel
    watershed_labels_fitc : ndarray
        Labeled watershed segmentation for FITC channel
    well_number : str
        Well identifier (e.g., '00044')
    position_number : str
        Position identifier (e.g., '00021')
    base_name : str
        Original filename base (e.g., 'ATR2_24h--W00044--P00021--Z00000--T00000--')
    output_root : str
        Root directory for saving images
    """
    import matplotlib.pyplot as plt
    import os
    from skimage import exposure
    from skimage.segmentation import mark_boundaries

    try:
        debug_dir = os.path.join(output_root, "debug_images_global")
        os.makedirs(debug_dir, exist_ok=True)

        # Normalize image once for all visualizations
        vis_img = exposure.rescale_intensity(original_image, in_range='image', out_range=(0, 1))

        # --- 1️⃣ TRITC FOCI OVERLAY ---
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img, cmap='gray')
        
        for (y, x) in foci_tritc:
            plt.plot(x, y, 'ro', markersize=0.3, alpha=1)
        
        plt.title(f"TRITC Foci | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}TRITC_foci.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved TRITC foci visualization: {filename}")

        # --- 2️⃣ FITC FOCI OVERLAY ---
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img, cmap='gray')
        
        for (y, x) in foci_fitc:
            plt.plot(x, y, 'go', markersize=0.3, alpha=1)
        
        plt.title(f"FITC Foci | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}FITC_foci.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved FITC foci visualization: {filename}")

        # --- 3️⃣ TRITC WATERSHED OUTLINES ---
        outlined_img_tritc = mark_boundaries(vis_img, watershed_labels_tritc, 
                                            color=(1, 1, 0), mode='thick')
        plt.figure(figsize=(10, 10))
        plt.imshow(outlined_img_tritc)
        plt.title(f"TRITC Watershed | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}TRITC_watershed.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved TRITC watershed visualization: {filename}")

        # --- 4️⃣ FITC WATERSHED OUTLINES ---
        outlined_img_fitc = mark_boundaries(vis_img, watershed_labels_fitc, 
                                           color=(1, 1, 0), mode='thick')
        plt.figure(figsize=(10, 10))
        plt.imshow(outlined_img_fitc)
        plt.title(f"FITC Watershed | Well {well_number} Position {position_number}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"{base_name}FITC_watershed.png"
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved FITC watershed visualization: {filename}")

    except Exception as e:
        print(f"⚠️ Failed to save global visualizations for Well {well_number}, Position {position_number}: {e}")
        import traceback
        traceback.print_exc()






def save_debug_image(isolated_img, water_labels, final_coords, foci_detection_count,
                     total_iterations, cellnumber, channel_name, well_number, position_number,
                     output_root, detection_threshold=50):
    """
    Save visualization of nucleus with watershed and foci overlay.
    """
    try:
        debug_dir = os.path.join(output_root, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)

        vis_img = exposure.rescale_intensity(isolated_img, in_range='image', out_range=(0, 1))
        overlay_img = mark_boundaries(vis_img, water_labels, color=(1, 0, 0), mode='thick')

        plt.figure(figsize=(5, 5))
        plt.imshow(overlay_img, cmap='gray')

        # Overlay detected foci
        for (y, x) in final_coords:
            detection_prob = (foci_detection_count.get((y, x), 0) / total_iterations) * 100
            color = 'go' if detection_prob >= detection_threshold else 'yo'
            plt.plot(x, y, color, markersize=4)

        plt.title(f"Cell {cellnumber} | {channel_name} | W{well_number} P{position_number}")
        plt.axis('off')
        plt.tight_layout()

        out_name = f"W{well_number}_P{position_number}_Cell{cellnumber}_{channel_name}.png"
        out_path = os.path.join(debug_dir, out_name)
        plt.savefig(out_path, dpi=200)
        plt.close()

    except Exception as e:
        print(f"⚠️ Debug image save failed for cell {cellnumber} ({channel_name}): {e}")


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
    gradient = filters.sobel(isolated_img)
    
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