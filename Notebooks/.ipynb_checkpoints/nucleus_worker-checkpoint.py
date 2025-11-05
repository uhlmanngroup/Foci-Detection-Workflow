# nucleus_worker.py
"""
Worker function for parallel nucleus processing.
Must be in separate file for Windows multiprocessing compatibility.
"""

import numpy as np
import os, re
from skimage import exposure, measure, filters, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from collections import Counter


def process_single_nucleus(args):
    """
    Parallel worker that processes a single nucleus:
    - Isolates its region
    - Applies filtering
    - Runs parameter sweeps
    - Performs segmentation
    - Returns results
    """
    (
        cellnumber,
        masks,
        TRITC_pic,
        valid_param_samples,
        total_iterations,
        well_number,
        position_number
    ) = args

    # Initialize output containers
    foci_data_list = []
    nuclei_data_list = []
    parameter_results = []

    # Create mask for the current nucleus only
    masks_reduced = (masks == cellnumber)
    
    # Safety check
    if not np.any(masks_reduced):
        return [], [], []

    # Copy and float-convert the fluorescence image
    isolated_TRITC = img_as_float(TRITC_pic.copy())
    isolated_TRITC[masks_reduced == False] = 0
    
    # Safety check for signal
    if isolated_TRITC.max() == 0:
        return [], [], []

    # Apply Difference of Gaussians filter
    filtered_TRITC = filters.difference_of_gaussians(isolated_TRITC, low_sigma=1, high_sigma=2)
    filtered_TRITC = np.clip(filtered_TRITC, 0, None)
    filtered_TRITC = exposure.rescale_intensity(filtered_TRITC, in_range='image', out_range=(0, isolated_TRITC.max()))

    # Extract parameters from array
    bright_pcts = valid_param_samples[:, 0]
    contrast_threshs = valid_param_samples[:, 1]
    percentile_vals = valid_param_samples[:, 2]

    # Get pixel values > 0 to compute percentile thresholds
    pos_pixels = TRITC_pic[TRITC_pic > 0]
    if pos_pixels.size == 0:
        return [], [], []

    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    global_min_brightness = np.min(min_brightness_per_param)

    # Detect candidate foci
    candidates_filtered = peak_local_max(filtered_TRITC, min_distance=2, threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_TRITC, min_distance=2, threshold_abs=global_min_brightness)

    if candidates_filtered.shape[0] == 0 or candidates_unfiltered.shape[0] == 0:
        return [], [], []

    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    filt_intensities = filtered_TRITC[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_TRITC[unf_yx[:, 0], unf_yx[:, 1]]

    unique_brights = np.unique(np.round(bright_pcts, 6))
    bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}

    def compute_local_percentiles_for_candidates(image, coords, unique_percentiles):
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

    local_percentiles_unf = compute_local_percentiles_for_candidates(isolated_TRITC, unf_yx, unique_brights)
    local_percentiles_filt = compute_local_percentiles_for_candidates(filtered_TRITC, filt_yx, unique_brights)

    distances = cdist(unf_yx, filt_yx)
    tolerance = 2

    foci_counts = []
    all_detected_foci = []

    for p_idx in range(len(valid_param_samples)):
        bright_pct = bright_pcts[p_idx]
        contrast_thresh = contrast_threshs[p_idx]
        percentile_val = percentile_vals[p_idx]
        min_brightness = min_brightness_per_param[p_idx]

        bright_key = np.round(bright_pct, 6)
        b_idx = bright_to_idx[bright_key]

        unf_mask_by_abs = unf_intensities >= min_brightness
        filt_mask_by_abs = filt_intensities >= min_brightness
        if not np.any(unf_mask_by_abs) or not np.any(filt_mask_by_abs):
            foci_counts.append(0)
            continue

        unf_local_bg = local_percentiles_unf[:, b_idx]
        filt_local_bg = local_percentiles_filt[:, b_idx]
        unf_mask_by_contrast = unf_intensities > (unf_local_bg * contrast_thresh)
        filt_mask_by_contrast = filt_intensities > (filt_local_bg * contrast_thresh)

        unf_final_mask = unf_mask_by_abs & unf_mask_by_contrast
        filt_final_mask = filt_mask_by_abs & filt_mask_by_contrast

        unf_idxs = np.where(unf_final_mask)[0]
        filt_idxs = np.where(filt_final_mask)[0]
        if unf_idxs.size == 0 or filt_idxs.size == 0:
            foci_counts.append(0)
            continue

        distances_sub = distances[unf_idxs][:, filt_idxs]
        nearest_dist = np.min(distances_sub, axis=1)
        confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
        confirmed_coords = unf_yx[confirmed_unf_idxs]

        foci_counts.append(len(confirmed_coords))
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord))

        parameter_results.append({
            "cell_num": cellnumber,
            "bright_pct": bright_pct,
            "contrast_thresh": contrast_thresh,
            "percentile_val": percentile_val,
            "foci_count": len(confirmed_coords)
        })

    if not foci_counts:
        return [], [], []

    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = min(foci_counts)
    max_foci = max(foci_counts)
    foci_detection_count = Counter(all_detected_foci)

    max_foci_count = max(foci_counts)
    max_foci_idx = foci_counts.index(max_foci_count)
    best_params = valid_param_samples[max_foci_idx]
    bright_pct, contrast_thresh, percentile_val = best_params
    min_brightness = np.percentile(TRITC_pic[TRITC_pic > 0], percentile_val)

    coordinates_unfiltered = peak_local_max(isolated_TRITC, min_distance=2, threshold_abs=min_brightness)
    coordinates_filtered = peak_local_max(filtered_TRITC, min_distance=2, threshold_abs=min_brightness)
    distances = cdist(coordinates_unfiltered, coordinates_filtered)
    tolerance = 2
    final_coordinates = coordinates_unfiltered[np.min(distances, axis=1) <= tolerance] if coordinates_unfiltered.size > 0 and coordinates_filtered.size > 0 else np.array([]).reshape(0, 2)

    coordinates = final_coordinates
    water_threshold = min_brightness

    markers = np.zeros_like(filtered_TRITC, dtype=int)
    for idx, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = idx

    markers_expanded = markers.copy()
    distance = ndi.distance_transform_edt(filtered_TRITC < water_threshold)
    gradient = filters.sobel(filtered_TRITC)
    watershed_mask = (filtered_TRITC > water_threshold) | (markers_expanded > 0)
    water_labels = watershed(gradient, markers_expanded, mask=watershed_mask)

    for idx, (y, x) in enumerate(coordinates):
        region_id = water_labels[y, x]
        spot_mask = (water_labels == region_id)
        spot_area = np.sum(spot_mask)
        spot_intensity = np.sum(isolated_TRITC[spot_mask])

        focus_tuple = (y, x)
        times_detected = foci_detection_count.get(focus_tuple, 1)
        detection_probability = (times_detected / total_iterations) * 100

        foci_data_list.append({
            'cell_num': cellnumber,
            'centr_y': y,
            'centr_x': x,
            'foci_area': spot_area,
            'intensity': spot_intensity,
            'detection_prob': detection_probability,
            'Well': well_number,
            'Position': position_number
        })

    nucleus_props = measure.regionprops(masks_reduced.astype(int))
    if len(nucleus_props) > 0:
        region = nucleus_props[0]
        nuclei_data_list.append({
            'cell_num': cellnumber,
            'DAPI_area': region.area,
            'centr_y': region.centroid[0],
            'centr_x': region.centroid[1],
            'perim': region.perimeter,
            'mean_foci': mean_foci,
            'stndrd_dev': std_foci,
            'foci_min': min_foci,
            'foci_max': max_foci,
            'Well': well_number,
            'Position': position_number
        })

    return foci_data_list, nuclei_data_list, parameter_results