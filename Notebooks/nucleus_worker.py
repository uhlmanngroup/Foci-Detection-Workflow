import numpy as np
import pandas as pd
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


# ===============================================================
# 1️⃣ INTENSITY ANALYSIS FUNCTION
# ===============================================================

def analyze_nucleus_intensity(nucleus_mask, image, channel_name, cell_id, nuclei_dict):
    """
    Compute total and mean intensity for one nucleus in one channel.
    Appends results to nuclei_dict.
    """
    nucleus_pixels = image[nucleus_mask]
    total_intensity = float(np.sum(nucleus_pixels))
    mean_intensity = float(np.mean(nucleus_pixels))

    # Create entry if not exists
    if cell_id not in nuclei_dict:
        nuclei_dict[cell_id] = {
            "cell_num": cell_id,
        }

    nuclei_dict[cell_id][f"{channel_name}_total_intensity"] = total_intensity
    nuclei_dict[cell_id][f"{channel_name}_mean_intensity"] = mean_intensity


# ===============================================================
# 2️⃣ FOCI DETECTION FUNCTION
# ===============================================================

def detect_foci(nucleus_mask, image, channel_name, cell_id,
                valid_param_samples, total_iterations,
                nuclei_dict, foci_dict):
    """
    Detects foci in a single nucleus region and updates:
      - foci_dict[channel_name]: individual foci entries
      - nuclei_dict[cell_id]: summary stats for that channel
    """

    isolated_img = img_as_float(image.copy())
    isolated_img[~nucleus_mask] = 0
    if isolated_img.max() == 0:
        return

    filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
    filtered_img = np.clip(filtered_img, 0, None)
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', out_range=(0, isolated_img.max()))

    bright_pcts = valid_param_samples[:, 0]
    contrast_threshs = valid_param_samples[:, 1]
    percentile_vals = valid_param_samples[:, 2]

    pos_pixels = image[image > 0]
    if pos_pixels.size == 0:
        return
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    global_min_brightness = np.min(min_brightness_per_param)

    candidates_filtered = peak_local_max(filtered_img, min_distance=2, threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, threshold_abs=global_min_brightness)
    if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
        return

    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    filt_intensities = filtered_img[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_img[unf_yx[:, 0], unf_yx[:, 1]]

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

    local_percentiles_unf = compute_local_percentiles_for_candidates(isolated_img, unf_yx, unique_brights)
    local_percentiles_filt = compute_local_percentiles_for_candidates(filtered_img, filt_yx, unique_brights)

    distances = cdist(unf_yx, filt_yx)
    tolerance = 2
    foci_counts = []
    all_detected_foci = []

    def apply_foci_filters(p_idx):
        bright_pct = bright_pcts[p_idx]
        contrast_thresh = contrast_threshs[p_idx]
        percentile_val = percentile_vals[p_idx]
        min_brightness = min_brightness_per_param[p_idx]
        bright_key = np.round(bright_pct, 6)
        b_idx = bright_to_idx[bright_key]

        unf_mask_abs = unf_intensities >= min_brightness
        filt_mask_abs = filt_intensities >= min_brightness
        if not np.any(unf_mask_abs) or not np.any(filt_mask_abs):
            return np.array([]), 0

        unf_local_bg = local_percentiles_unf[:, b_idx]
        filt_local_bg = local_percentiles_filt[:, b_idx]
        unf_mask_con = unf_intensities > (unf_local_bg * contrast_thresh)
        filt_mask_con = filt_intensities > (filt_local_bg * contrast_thresh)

        unf_final_mask = unf_mask_abs & unf_mask_con
        filt_final_mask = filt_mask_abs & filt_mask_con

        unf_idxs = np.where(unf_final_mask)[0]
        filt_idxs = np.where(filt_final_mask)[0]
        if unf_idxs.size == 0 or filt_idxs.size == 0:
            return np.array([]), 0

        distances_sub = distances[unf_idxs][:, filt_idxs]
        nearest_dist = np.min(distances_sub, axis=1)
        confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
        confirmed_coords = unf_yx[confirmed_unf_idxs]
        return confirmed_coords, len(confirmed_coords)

    for p_idx in range(len(valid_param_samples)):
        confirmed_coords, count = apply_foci_filters(p_idx)
        foci_counts.append(count)
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord))

    if not foci_counts:
        return

    foci_detection_count = Counter(all_detected_foci)
    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = min(foci_counts)
    max_foci = max(foci_counts)

    # Watershed on best parameters
    max_idx = np.argmax(foci_counts)
    best_params = valid_param_samples[max_idx]
    percentile_val = best_params[2]
    min_brightness = np.percentile(image[image > 0], percentile_val)
    coordinates_unfiltered = peak_local_max(isolated_img, min_distance=2, threshold_abs=min_brightness)
    coordinates_filtered = peak_local_max(filtered_img, min_distance=2, threshold_abs=min_brightness)
    distances = cdist(coordinates_unfiltered, coordinates_filtered)
    final_coords = coordinates_unfiltered[np.min(distances, axis=1) <= tolerance] \
        if coordinates_unfiltered.size > 0 and coordinates_filtered.size > 0 else np.empty((0, 2), int)

    # Add foci results to foci_dict
    if channel_name not in foci_dict:
        foci_dict[channel_name] = []

    gradient = filters.sobel(filtered_img)
    markers = np.zeros_like(filtered_img, dtype=int)
    for idx, (y, x) in enumerate(final_coords, start=1):
        markers[y, x] = idx
    water_labels = watershed(gradient, markers, mask=(filtered_img > min_brightness))

    for idx, (y, x) in enumerate(final_coords):
        region_id = water_labels[y, x]
        spot_mask = (water_labels == region_id)
        spot_area = int(np.sum(spot_mask))
        spot_intensity = float(np.sum(isolated_img[spot_mask]))
        detection_probability = (foci_detection_count.get((y, x), 0) / total_iterations) * 100

        foci_dict[channel_name].append({
            'cell_num': cell_id,
            'centr_y': y,
            'centr_x': x,
            'foci_area': spot_area,
            'intensity': spot_intensity,
            'detection_prob': detection_probability,
            'channel': channel_name
        })

    # Update nucleus summary
    if cell_id not in nuclei_dict:
        nuclei_dict[cell_id] = {"cell_num": cell_id}

    nuclei_dict[cell_id].update({
        f"{channel_name}_mean_foci": mean_foci,
        f"{channel_name}_std_foci": std_foci,
        f"{channel_name}_min_foci": min_foci,
        f"{channel_name}_max_foci": max_foci,
    })


# ===============================================================
# 3️⃣ WRAPPER TO PROCESS ALL NUCLEI
# ===============================================================

def process_all_nuclei(masks, channel_images, valid_param_samples,
                       total_iterations, parallel=True):
    """
    Iterate through all nuclei and all channels.
    Returns nuclei_dict and foci_dict.
    """

    nuclei_dict = {}
    foci_dict = {}

    cell_numbers = np.unique(masks)
    cell_numbers = cell_numbers[cell_numbers != 0]

    def process_single(cell_id):
        local_nuclei_dict = {}
        local_foci_dict = {}

        nucleus_mask = (masks == cell_id)

        # Channel intensity
        for cname, img in channel_images.items():
            analyze_nucleus_intensity(nucleus_mask, img, cname, cell_id, local_nuclei_dict)
            if cname != "DAPI":  # Skip foci detection for DAPI
                detect_foci(nucleus_mask, img, cname, cell_id,
                            valid_param_samples, total_iterations,
                            local_nuclei_dict, local_foci_dict)
        return local_nuclei_dict, local_foci_dict

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single, c) for c in cell_numbers]
            for fut in as_completed(futures):
                n_dict, f_dict = fut.result()
                for k, v in n_dict.items():
                    nuclei_dict[k] = {**nuclei_dict.get(k, {}), **v}
                for cname, lst in f_dict.items():
                    foci_dict.setdefault(cname, []).extend(lst)
    else:
        for c in cell_numbers:
            n_dict, f_dict = process_single(c)
            for k, v in n_dict.items():
                nuclei_dict[k] = {**nuclei_dict.get(k, {}), **v}
            for cname, lst in f_dict.items():
                foci_dict.setdefault(cname, []).extend(lst)

    return nuclei_dict, foci_dict


# ===============================================================
# 4️⃣ MAIN EXECUTION EXAMPLE
# ===============================================================

if __name__ == "__main__":
    # Example: assume you already have these loaded
    # masks = ...         # DAPI mask with labeled nuclei
    # DAPI_pic = ...
    # TRITC_pic = ...
    # FITC_pic = ...
    # valid_param_samples = np.array([...])
    # total_iterations = len(valid_param_samples)

    channel_images = {
        "DAPI": DAPI_pic,
        "TRITC": TRITC_pic,
        "FITC": FITC_pic
    }

    nuclei_dict, foci_dict = process_all_nuclei(
        masks, channel_images, valid_param_samples, total_iterations, parallel=True
    )

    # Convert to DataFrames
    nuclei_df = pd.DataFrame.from_dict(list(nuclei_dict.values()))
    for cname, foc_list in foci_dict.items():
        foci_df = pd.DataFrame(foc_list)
        foci_df.to_csv(f"foci_{cname}.csv", index=False)

    nuclei_df.to_csv("nuclei_summary.csv", index=False)
    print("✅ Analysis complete.")
