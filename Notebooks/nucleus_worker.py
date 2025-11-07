# nucleus_worker.py
"""
Worker function for parallel nucleus processing.
Must be in separate file for Windows multiprocessing compatibility.
"""
# These imports need to be inside here, because the workers need to initialize everything from the start
import numpy as np
import os, re
from skimage import exposure, measure, filters, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from collections import Counter


# Function that processes a single nucleus region in parallel
# It isolates the nucleus, applies image filtering and parameter iteration
# and returns a list of detected foci and nucleus metrics
def process_single_nucleus(args):
    # Unpacking the input arguments passed from the main script
    (
        cellnumber,           # Label of the current nucleus to analyze
        masks,                # Full DAPI mask image (contains all labeled nuclei)
        TRITC_pic,            # TRITC fluorescence image (used for foci detection)
        valid_param_samples,  # Array of parameter combinations to iterate through
        total_iterations,     # Total number of valid parameter combinations
        well_number,          # Well identifier for saving/tracking results
        position_number       # Position identifier for saving/tracking results
    ) = args

    # Initialize containers that will store the results for this nucleus
    foci_data_list = []       # Stores per-foci measurements
    nuclei_data_list = []     # Stores nucleus-level metrics
    parameter_results = []    # Stores results for each parameter iteration

    # Create mask for the current nucleus only (True where current cell label matches)
    masks_reduced = (masks == cellnumber)
    
    # Safety check – skip if the mask is empty (shouldn’t happen but prevents crashes)
    if not np.any(masks_reduced):
        return [], [], []

    # Copy the TRITC image and convert to float to ensure consistent pixel intensity scaling
    isolated_TRITC = img_as_float(TRITC_pic.copy())
    # Set all pixels outside of the current nucleus to 0, isolating the nucleus
    isolated_TRITC[masks_reduced == False] = 0
    
    # Skip processing if there is no fluorescence signal in this nucleus
    if isolated_TRITC.max() == 0:
        return [], [], []

    # Apply Difference of Gaussians filter (enhances small bright spots like foci)
    filtered_TRITC = filters.difference_of_gaussians(isolated_TRITC, low_sigma=1, high_sigma=2)
    # Clip any negative values to zero and normalize intensity range
    filtered_TRITC = np.clip(filtered_TRITC, 0, None)
    filtered_TRITC = exposure.rescale_intensity(filtered_TRITC, in_range='image', out_range=(0, isolated_TRITC.max()))
    
    # Extract the three parameter columns for readability
    bright_pcts = valid_param_samples[:, 0]        # Local background percentile
    contrast_threshs = valid_param_samples[:, 1]   # Minimum contrast ratio
    percentile_vals = valid_param_samples[:, 2]    # Global intensity percentile


    # Get all positive pixel values to determine brightness thresholds
    pos_pixels = TRITC_pic[TRITC_pic > 0]
    if pos_pixels.size == 0:
        return [], [], []

    # Compute the global brightness threshold for each parameter iteration
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    # Takes the smallest brightness threshold that was detected in all of the parameter iterations
    # This is later used to detect ALL of the potential foci locations outside of the parameter iteration loop (saves computing time)
    global_min_brightness = np.min(min_brightness_per_param)

    # Detect candidate foci in both filtered and unfiltered TRITC images
    candidates_filtered = peak_local_max(filtered_TRITC, min_distance=2, threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_TRITC, min_distance=2, threshold_abs=global_min_brightness)

    # Skip nucleus if no foci candidates were detected
    if candidates_filtered.shape[0] == 0 or candidates_unfiltered.shape[0] == 0:
        return [], [], []

    # Convert candidate coordinate arrays to integer form and extract intensity values
    # This is also done to save computing time, like this the program has to only check the values inside the vector instead of 
    # computing it for every iteration
    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    filt_intensities = filtered_TRITC[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_TRITC[unf_yx[:, 0], unf_yx[:, 1]]

    # Prepare a mapping from each unique bright percentile to its index
    # This saves only the unique brightness percentages, meaning if multiple parameter iterations use the same percentage it only 
    # gets computed once. It saves the value of the background brightness for each focus and unique percentile to later just access 
    # them without recalculating them
    unique_brights = np.unique(np.round(bright_pcts, 6))
    bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}

    # Helper function to calculate local brightness percentiles for all candidate foci
    # This function needs to be defined inside this function, so that the workers can use them(same like the imports)
    def compute_local_percentiles_for_candidates(image, coords, unique_percentiles):
        N = coords.shape[0]
        P = len(unique_percentiles)
        out = np.zeros((N, P), dtype=float)
        for i, (y, x) in enumerate(coords):
            # Define 13x13 neighborhood around each potential focus (ideally from -6 to +6 or picture border if it would portrude)
            y_min, y_max = max(0, y - 6), min(image.shape[0], y + 7)
            x_min, x_max = max(0, x - 6), min(image.shape[1], x + 7)
            square = image[y_min:y_max, x_min:x_max]
            # Compute local percentiles or use the focus intensity if square is empty
            if square.size == 0:
                out[i, :] = image[y, x]
            else:
                out[i, :] = np.percentile(square, unique_percentiles)
        return out
        
    # Compute local backgrounds for both filtered and unfiltered foci candidates with the function above
    local_percentiles_unf = compute_local_percentiles_for_candidates(isolated_TRITC, unf_yx, unique_brights)
    local_percentiles_filt = compute_local_percentiles_for_candidates(filtered_TRITC, filt_yx, unique_brights)

    # Compute pairwise distances between unfiltered and filtered candidates (for matching)
    distances = cdist(unf_yx, filt_yx)
    tolerance = 2  # Max distance in pixels for matching filtered/unfiltered foci

    foci_counts = []        # List storing number of foci detected per parameter set
    all_detected_foci = []  # Collects coordinates of all confirmed foci

    # Function to filter out all the potential foci that don't match the set requirments(not bright enough, not brighter than 
    # background, too far apart in both pictures etc.)
    def apply_foci_filters(
        p_idx,
        bright_pcts,
        contrast_threshs,
        percentile_vals,
        min_brightness_per_param,
        bright_to_idx,
        unf_intensities,
        filt_intensities,
        local_percentiles_unf,
        local_percentiles_filt,
        distances,
        unf_yx,
        tolerance,
    ):
        """
        Applies all absolute and contrast-based filtering steps for one parameter iteration.
        Returns:
            confirmed_coords (np.ndarray): Coordinates of foci that passed all filters
            foci_count (int): Total number of confirmed foci for this parameter set
        """
    
        # Saves the parameters for this iteration
        bright_pct = bright_pcts[p_idx]            # Background brightness percentage
        contrast_thresh = contrast_threshs[p_idx]  # Contrast threshold
        percentile_val = percentile_vals[p_idx]    # Minimal brightness percentile
        min_brightness = min_brightness_per_param[p_idx]  # Actual pixel intensity corresponding to that percentile
    
        # Identify corresponding background brightness percentile column in the previously created bright_to_idx
        # to get the actual pixel background brightness for this iteration
        bright_key = np.round(bright_pct, 6)
        b_idx = bright_to_idx[bright_key]
    
        # Apply absolute and contrast-based filters to candidate foci
        # Potential foci that have been previously identified with the smallest minimal brightness now need to be brighter than the 
        # minimal brightness for this iteration (for both the filtered and unfiltered picture)
        unf_mask_by_abs = unf_intensities >= min_brightness
        filt_mask_by_abs = filt_intensities >= min_brightness
    
        # If no foci make the cut, return empty arrays and a foci count of 0
        if not np.any(unf_mask_by_abs) or not np.any(filt_mask_by_abs):
            return np.array([]), 0
    
        # Compare each focus’ brightness to its local background * contrast threshold
        unf_local_bg = local_percentiles_unf[:, b_idx]  # Accessing the previously saved background brightness value
        filt_local_bg = local_percentiles_filt[:, b_idx]
    
        # Filtering the foci that aren't significantly brighter than the chosen background
        unf_mask_by_contrast = unf_intensities > (unf_local_bg * contrast_thresh)
        filt_mask_by_contrast = filt_intensities > (filt_local_bg * contrast_thresh)
        
        # Combine both absolute and contrast filters that were independently tested
        unf_final_mask = unf_mask_by_abs & unf_mask_by_contrast
        filt_final_mask = filt_mask_by_abs & filt_mask_by_contrast
    
        # Get the indices of all the foci that passed all the checks
        unf_idxs = np.where(unf_final_mask)[0]
        filt_idxs = np.where(filt_final_mask)[0]
    
        # If no foci remain after all the checks, return empty arrays and a foci count of 0
        if unf_idxs.size == 0 or filt_idxs.size == 0:
            return np.array([]), 0
    
        # Match filtered and unfiltered foci based on spatial proximity
        # Extracts submatrix that only includes the distances from foci that passed all the checks
        distances_sub = distances[unf_idxs][:, filt_idxs]
        # Finding the closest two foci to each other (gives a 1D array where each unfiltered focus is listed 
        # with the distance to the nearest filtered focus)
        nearest_dist = np.min(distances_sub, axis=1)
        # Keeping only the unfiltered foci within the tolerance distance
        confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
        # Get the coordinates of the remaining foci
        confirmed_coords = unf_yx[confirmed_unf_idxs]
    
        # Return coordinates and the total number of confirmed foci for this parameter iteration
        return confirmed_coords, len(confirmed_coords)


    # Iterate through all valid parameter combinations
    #for p_idx in range(len(valid_param_samples)):
    for p_idx in range(1):
        # Apply filtering for this parameter iteration
        confirmed_coords, count = apply_foci_filters(
            p_idx,
            bright_pcts,
            contrast_threshs,
            percentile_vals,
            min_brightness_per_param,
            bright_to_idx,
            unf_intensities,
            filt_intensities,
            local_percentiles_unf,
            local_percentiles_filt,
            distances,
            unf_yx,
            tolerance,
        )
        print("parameter combination done: "+str(p_idx))

        # Save total number of confirmed foci for this parameter iteration
        foci_counts.append(count) # total foci count
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord)) # Location of foci

        parameter_results.append({
            "cell_num": cellnumber,
            "bright_pct": bright_pct,
            "contrast_thresh": contrast_thresh,
            "percentile_val": percentile_val,
            "foci_count": len(confirmed_coords)
        })

    # Skip if no foci were found under any parameter combination
    if not foci_counts:
        return [], [], []

    # Compute statistics for this nucleus across all tested parameters
    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = min(foci_counts)
    max_foci = max(foci_counts)
    foci_detection_count = Counter(all_detected_foci)

    # Identify the parameter combination that yielded the highest foci count
    # This is used to run the highest detected foci count in the watershed segmentation
    max_foci_count = max(foci_counts)
    max_foci_idx = foci_counts.index(max_foci_count)
    best_params = valid_param_samples[max_foci_idx]
    bright_pct, contrast_thresh, percentile_val = best_params
    min_brightness = np.percentile(TRITC_pic[TRITC_pic > 0], percentile_val)

    # Re-run peak detection with best parameters for final segmentation
    coordinates_unfiltered = peak_local_max(isolated_TRITC, min_distance=2, threshold_abs=min_brightness)
    coordinates_filtered = peak_local_max(filtered_TRITC, min_distance=2, threshold_abs=min_brightness)
    distances = cdist(coordinates_unfiltered, coordinates_filtered)
    tolerance = 2
    final_coordinates = coordinates_unfiltered[np.min(distances, axis=1) <= tolerance] if coordinates_unfiltered.size > 0 and coordinates_filtered.size > 0 else np.array([]).reshape(0, 2)

    final_coordinates, count = apply_foci_filters(
        max_foci_idx,                # index of the best parameter set
        bright_pcts,             # all tested background percentiles
        contrast_threshs,        # all tested contrast thresholds
        percentile_vals,         # all tested global percentile thresholds
        min_brightness_per_param,# per-parameter minimal brightness levels
        bright_to_idx,           # mapping of unique brightness percentiles to column indices
        unf_intensities,         # intensities of foci in the unfiltered image
        filt_intensities,        # intensities of foci in the filtered image
        local_percentiles_unf,   # local background levels around each unfiltered focus
        local_percentiles_filt,  # local background levels around each filtered focus
        distances,               # distance matrix between unfiltered and filtered foci
        unf_yx,                  # coordinates of unfiltered foci
        tolerance,               # maximum spatial distance for matching filtered/unfiltered foci
    )


    # Generate final markers for watershed segmentation
    coordinates = final_coordinates
    water_threshold = min_brightness

    # Creation of new image "markers" same shape as TRITC pic
    markers = np.zeros_like(filtered_TRITC, dtype=int)
    # All the detected foci start points are saved with a unique label
    for idx, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = idx

    # Compute distance transform and gradient for watershed segmentation
    markers_expanded = markers.copy()
    distance = ndi.distance_transform_edt(filtered_TRITC < water_threshold)
    gradient = filters.sobel(filtered_TRITC) # Detects ridges(intensity changes) that are then used in the watershed segmentation

    # watershed_mask restricts the watershed algorithm to areas that are bright enough
    watershed_mask = (filtered_TRITC > water_threshold) | (markers_expanded > 0)
    # Creates mask that contains all the pixels that would get segmented in the watershed and the pixel that was detected as the
    # center of the foci
    water_labels = watershed(gradient, markers_expanded, mask=watershed_mask)  # Applying Watershed, -distance to invert map height 
    # map (to make peaks to valleys)
    # The watershed was now done within the borders of the previously defined mask
    

    # Loop through each detected focus and measure properties (area, intensity, detection probability)
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