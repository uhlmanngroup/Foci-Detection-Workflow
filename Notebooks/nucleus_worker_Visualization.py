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
import skimage as ski
import os
from PIL import Image
from skimage.morphology import binary_erosion, disk


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
    if perimeter == 0 or area == 0:
        return 0.0
    
    # Standard circularity formula
    # A perfect circle has maximum circularity (approaches 1.0)
    # Irregular or elongated shapes have lower values
    circularity = (4 * np.pi * area) / (perimeter ** 2)
        
    # Cap at 1.0 to handle numerical artifacts from discrete pixel measurements
    # This can happen with very small regions where perimeter approximation
    # is less accurate
    return min(circularity, 1.0)


def compute_adaptive_background_texture_nucleus_fallback(
    image, coords, unique_percentiles, 
    nucleus_mask=None,
    nucleus_labels=None,
    inner_radius=2, 
    outer_radius=6,
    edge_outer_radius=12,
    density_threshold=0.15,
    edge_zone_distance=6,
    return_texture_info=False
):
    """
    Adaptive background with NUCLEUS fallback for dense regions + texture metrics.
    
    KEY FEATURES:
    1. Dense regions fall back to PER-NUCLEUS background (not global)
    2. Computes texture metrics (CV, variance) for each nucleus
    3. Returns texture info so researchers can filter post-hoc
    
    WORKFLOW:
    - Detects dense region around each candidate (local median > nucleus median)
    - If dense: uses NUCLEUS background (handles uniformly bright nuclei)
    - If not dense: uses LOCAL annulus background
    - Edges: always use expanded local annulus
    
    Parameters:
    -----------
    image : 2D numpy array
        Grayscale microscopy image
    coords : Nx2 numpy array  
        Candidate spot coordinates [[y1,x1], [y2,x2], ...]
    unique_percentiles : array-like
        Percentile values to compute (e.g., [25, 50, 75])
    nucleus_mask : 2D boolean array, optional
        Binary mask defining nucleus interior
    nucleus_labels : 2D int array, optional
        Labeled mask where each nucleus has unique integer ID
    inner_radius : int
        Inner radius of annulus (default=2)
    outer_radius : int  
        Outer radius for interior regions (default=6)
    edge_outer_radius : int
        Larger outer radius for edge regions (default=12)
    density_threshold : float
        Local median must be this much above nucleus median to trigger
        dense mode (default=0.15 = 15%)
    edge_zone_distance : int
        Distance from edge to be considered "edge region" (default=6)
    return_texture_info : bool
        If True, also return dictionary with nucleus texture metrics
        
    Returns:
    --------
    backgrounds : NxP numpy array
        Background estimates for each coordinate at each percentile
    texture_info : dict (only if return_texture_info=True)
        Dictionary with keys:
        - 'nucleus_stats': dict mapping nucleus_id -> {mean, std, cv, median, ...}
        - 'coord_nucleus_ids': array of nucleus IDs for each coordinate
        - 'coord_texture_flags': array of bools indicating if nucleus is "spotty"
    """
    
    N = coords.shape[0]
    P = len(unique_percentiles)
    backgrounds = np.zeros((N, P), dtype=float)
    
    # ============================================================================
    # STEP 1: Compute edge distances
    # ============================================================================
    edge_distances = None
    if nucleus_mask is not None:
        edge_distances = distance_transform_edt(nucleus_mask)
    
    # ============================================================================
    # STEP 2: Compute per-nucleus backgrounds AND texture metrics
    # ============================================================================
    nucleus_backgrounds = {}  # nucleus_id -> background percentiles
    nucleus_stats = {}        # nucleus_id -> {mean, std, cv, median, ...}
    
    if nucleus_mask is not None:
        # Get labeled nuclei if not provided
        if nucleus_labels is None:
            nucleus_labels, num_nuclei = label(nucleus_mask)
        else:
            num_nuclei = int(nucleus_labels.max())
        
        # Compute statistics for each nucleus
        for nuc_id in range(1, num_nuclei + 1):
            nuc_mask_single = nucleus_labels == nuc_id
            nuc_pixels = image[nuc_mask_single]
            
            if len(nuc_pixels) == 0:
                continue
            
            # Basic statistics
            mean_intensity = np.mean(nuc_pixels)
            std_intensity = np.std(nuc_pixels)
            median_intensity = np.median(nuc_pixels)
            
            # Coefficient of Variation (CV) = std/mean
            # HIGH CV (>0.3) = spotty/variable = likely has foci
            # LOW CV (<0.2) = uniform = likely just bright baseline
            cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            # Additional texture metrics
            # Percentile range: difference between high and low percentiles
            p10 = np.percentile(nuc_pixels, 10)
            p90 = np.percentile(nuc_pixels, 90)
            percentile_range = p90 - p10
            
            # Store background percentiles for this nucleus
            nucleus_backgrounds[nuc_id] = np.percentile(nuc_pixels, unique_percentiles)
            
            # Store comprehensive statistics
            nucleus_stats[nuc_id] = {
                'mean': mean_intensity,
                'median': median_intensity,
                'std': std_intensity,
                'cv': cv,  # Coefficient of variation
                'p10': p10,
                'p90': p90,
                'percentile_range': percentile_range,
                'num_pixels': len(nuc_pixels),
                # Flag for post-hoc filtering
                'is_spotty': cv > 0.25,  # Can adjust this threshold
                'is_uniform': cv < 0.15   # Very uniform (suspiciously bright?)
            }
    
    # ============================================================================
    # STEP 3: Pre-compute annulus masks
    # ============================================================================
    # Standard annulus for interior regions
    y_grid, x_grid = np.ogrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]
    distances = np.sqrt(x_grid**2 + y_grid**2)
    std_annulus = (distances >= inner_radius) & (distances <= outer_radius)
    std_y, std_x = np.where(std_annulus)
    std_y -= outer_radius
    std_x -= outer_radius
    
    # Expanded annulus for edge regions  
    y_grid_big, x_grid_big = np.ogrid[-edge_outer_radius:edge_outer_radius+1, 
                                       -edge_outer_radius:edge_outer_radius+1]
    distances_big = np.sqrt(x_grid_big**2 + y_grid_big**2)
    edge_annulus = (distances_big >= inner_radius) & (distances_big <= edge_outer_radius)
    edge_y, edge_x = np.where(edge_annulus)
    edge_y -= edge_outer_radius
    edge_x -= edge_outer_radius
    
    # ============================================================================
    # STEP 4: Process each candidate coordinate
    # ============================================================================
    coord_nucleus_ids = np.zeros(N, dtype=int)  # Track which nucleus each coord is in
    
    for i, (y, x) in enumerate(coords):
        
        # ------------------------------------------------------------------------
        # 4.1: Determine if near edge
        # ------------------------------------------------------------------------
        is_near_edge = False
        if edge_distances is not None:
            dist_from_edge = edge_distances[y, x]
            is_near_edge = dist_from_edge <= edge_zone_distance
        
        # ------------------------------------------------------------------------
        # 4.2: Get nucleus ID for this coordinate
        # ------------------------------------------------------------------------
        nuc_id = 0
        if nucleus_labels is not None:
            nuc_id = int(nucleus_labels[y, x])
        coord_nucleus_ids[i] = nuc_id
        
        # ------------------------------------------------------------------------
        # 4.3: Select appropriate annulus based on location
        # ------------------------------------------------------------------------
        if is_near_edge:
            # Edge: use expanded annulus
            annulus_y, annulus_x = edge_y, edge_x
            min_pixels = 15
        else:
            # Interior: use standard annulus
            annulus_y, annulus_x = std_y, std_x
            min_pixels = 5
        
        # ------------------------------------------------------------------------
        # 4.4: Compute absolute pixel positions
        # ------------------------------------------------------------------------
        abs_y = y + annulus_y
        abs_x = x + annulus_x
        
        # ------------------------------------------------------------------------
        # 4.5: Filter for valid pixels (in bounds + in nucleus)
        # ------------------------------------------------------------------------
        valid = (abs_y >= 0) & (abs_y < image.shape[0]) & \
                (abs_x >= 0) & (abs_x < image.shape[1])
        
        if nucleus_mask is not None:
            valid_indices = np.where(valid)[0]
            nucleus_valid = nucleus_mask[abs_y[valid_indices], abs_x[valid_indices]] > 0
            valid[valid_indices] = nucleus_valid
        
        # ------------------------------------------------------------------------
        # 4.6: Compute background with nucleus-aware fallback
        # ------------------------------------------------------------------------
        if valid.sum() >= min_pixels:
            # Extract annulus pixels
            annulus_pixels = image[abs_y[valid], abs_x[valid]]
            local_percentiles = np.percentile(annulus_pixels, unique_percentiles)
            
            if not is_near_edge and nuc_id > 0 and nuc_id in nucleus_stats:
                # INTERIOR with valid nucleus: Check for dense region
                
                # Compute local median from annulus
                local_median = np.median(annulus_pixels)
                
                # Get nucleus median for comparison
                nucleus_median = nucleus_stats[nuc_id]['median']
                
                # Dense detection: local median significantly above nucleus median
                # This means we're in a region that's brighter than the nucleus baseline
                is_dense = local_median > nucleus_median * (1 + density_threshold)
                
                if is_dense:
                    # Dense region detected: use NUCLEUS background
                    # This prevents false positives in uniformly bright nuclei
                    # AND correctly handles truly dense foci regions
                    backgrounds[i, :] = nucleus_backgrounds[nuc_id]
                else:
                    # Normal region: use LOCAL annulus background
                    backgrounds[i, :] = local_percentiles
            else:
                # Edge region or no nucleus info: always use local
                backgrounds[i, :] = local_percentiles
        
        else:
            # Insufficient valid pixels: use fallback
            if nuc_id > 0 and nuc_id in nucleus_backgrounds:
                # Have nucleus info: use nucleus background
                backgrounds[i, :] = nucleus_backgrounds[nuc_id]
            else:
                # No nucleus info: use pixel's own value (conservative)
                backgrounds[i, :] = image[y, x]
    
    # ============================================================================
    # STEP 5: Return results with optional texture info
    # ============================================================================
    if return_texture_info:
        texture_info = {
            'nucleus_stats': nucleus_stats,
            'coord_nucleus_ids': coord_nucleus_ids,
            'coord_texture_flags': np.array([
                nucleus_stats.get(nid, {}).get('is_spotty', False) 
                for nid in coord_nucleus_ids
            ])
        }
        return backgrounds, texture_info
    else:
        return backgrounds


# ============================================================================
# HELPER: Post-hoc filtering based on texture
# ============================================================================
def filter_foci_by_texture(foci_coords, nucleus_labels, texture_info,
                          filter_uniform_bright=True,
                          min_cv_for_foci=0.15):
    """
    Filter detected foci based on nucleus texture characteristics.
    
    Use this AFTER foci detection to remove likely false positives from
    uniformly bright nuclei.
    
    Parameters:
    -----------
    foci_coords : Nx2 array
        Detected foci coordinates
    nucleus_labels : 2D array
        Labeled nucleus mask
    texture_info : dict
        Texture information from compute_adaptive_background_texture_nucleus_fallback
    filter_uniform_bright : bool
        If True, remove foci from nuclei with CV < min_cv_for_foci
    min_cv_for_foci : float
        Minimum coefficient of variation for nucleus to be considered
        "spotty enough" to have real foci (default=0.15)
        
    Returns:
    --------
    filtered_coords : Mx2 array (M <= N)
        Filtered foci coordinates
    filter_mask : N-length bool array
        True = kept, False = filtered out
    """
    N = foci_coords.shape[0]
    filter_mask = np.ones(N, dtype=bool)
    
    nucleus_stats = texture_info['nucleus_stats']
    
    for i, (y, x) in enumerate(foci_coords):
        nuc_id = int(nucleus_labels[y, x])
        
        if nuc_id == 0:
            # Not in a nucleus - keep it
            continue
        
        if nuc_id not in nucleus_stats:
            # No stats for this nucleus - keep it to be safe
            continue
        
        stats = nucleus_stats[nuc_id]
        
        # Filter based on texture
        if filter_uniform_bright and stats['is_uniform']:
            # Nucleus is very uniform (low CV) - likely false positives
            filter_mask[i] = False
    
    filtered_coords = foci_coords[filter_mask]
    return filtered_coords, filter_mask


# ============================================================================
# HELPER: Generate report on nucleus characteristics
# ============================================================================
def generate_nucleus_texture_report(texture_info):
    """
    Generate a summary report of nucleus texture characteristics.
    Useful for QC and deciding filtering thresholds.
    
    Returns:
    --------
    report : dict
        Summary statistics about nuclei in the image
    """
    nucleus_stats = texture_info['nucleus_stats']
    
    if not nucleus_stats:
        return {"error": "No nucleus statistics available"}
    
    cvs = [stats['cv'] for stats in nucleus_stats.values()]
    means = [stats['mean'] for stats in nucleus_stats.values()]
    
    num_spotty = sum(1 for stats in nucleus_stats.values() if stats['is_spotty'])
    num_uniform = sum(1 for stats in nucleus_stats.values() if stats['is_uniform'])
    
    report = {
        'num_nuclei': len(nucleus_stats),
        'num_spotty_nuclei': num_spotty,
        'num_uniform_nuclei': num_uniform,
        'cv_mean': np.mean(cvs),
        'cv_median': np.median(cvs),
        'cv_range': (np.min(cvs), np.max(cvs)),
        'intensity_mean': np.mean(means),
        'intensity_median': np.median(means),
        'intensity_range': (np.min(means), np.max(means))
    }
    
    return report


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
    valid_param_samples, total_iterations, water_threshold_percentile,
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
    local_percentiles_unf = compute_adaptive_background_expanded_edge(
        image=isolated_img,
        coords=unf_yx,
        unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask  
    )

 
    local_percentiles_filt = compute_adaptive_background_expanded_edge(
        image=filtered_img, 
        coords=filt_yx, 
        unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask,  

    )

    
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
    unf_local_bg = compute_adaptive_background_expanded_edge(
        image=isolated_img,
        coords=coordinates_unfiltered,
        unique_percentiles=[best_bright_pct],
        nucleus_mask=nucleus_mask, 
    )[:, 0]

    filt_local_bg = compute_adaptive_background_expanded_edge(
        image=filtered_img,
        coords=coordinates_filtered,
        unique_percentiles=[best_bright_pct],
        nucleus_mask=nucleus_mask,
    )[:, 0]

    
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


    
    # Rescale filtered image intensity to 0-100 range for consistent thresholding
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', out_range=(0, 100))
    
    # ========== WATERSHED SEGMENTATION WITH DISTANCE TRANSFORM ==========
    # This approach combines distance transform with compactness constraints to
    # segment foci while preventing over-segmentation and edge spillage
    
    # Erode nucleus mask to create safety margin from edges
    # This prevents foci from spilling along nucleus boundaries
    nucleus_mask_eroded = binary_erosion(nucleus_mask, disk(2))
    
    # Create marker array from detected foci coordinates
    # Each seed gets a unique integer label (1, 2, 3, ...)
    # Markers serve as starting points for watershed basins
    markers = np.zeros_like(isolated_img, dtype=int)
    for idx, (y, x) in enumerate(final_coords, start=1):
        markers[y, x] = idx
    
    # Define watershed mask combining multiple constraints
    # Start with pixels above intensity threshold AND within eroded nucleus
    binary_mask = (filtered_img > water_threshold_percentile) & nucleus_mask_eroded
    # Then force inclusion of all marker seeds, even if below threshold or in eroded region
    # This ensures every detected focus gets segmented
    binary_mask = binary_mask | (markers > 0)
    
    # Compute distance transform
    # Creates smooth "bowl-shaped" basins centered at high-intensity regions
    # Distance values are higher at region centers, lower at edges
    # Negative distance is used because watershed finds basins (low points)
    distance = ndi.distance_transform_edt(binary_mask)
    
    # Run watershed segmentation
    # -distance: inverted distance map (peaks become valleys for watershed)
    # markers: seed points defining basin centers
    # mask: limits where watershed can flow
    # compactness: penalizes irregular shapes, keeps foci compact and circular
    water_labels = watershed(-distance, markers, mask=binary_mask, compactness=0.005)

    
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
     total_iterations, well_number, position_number, water_threshold_percentile_TRITC, water_threshold_percentile_FITC) = args
    
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
        if channel_name in ["TRITC"]:
            foci_list, foci_summary, water_labels = detect_foci_single_channel(  # ← CHANGED: Now gets 3 returns
                masks_reduced,
                channel_image_float,
                channel_image_float,
                channel_name,
                cellnumber,
                valid_param_samples,
                total_iterations,
                water_threshold_percentile_TRITC, 
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

        if channel_name in ["FITC"]:
            foci_list, foci_summary, water_labels = detect_foci_single_channel(  # ← CHANGED: Now gets 3 returns
                masks_reduced,
                channel_image_float,
                channel_image_float,
                channel_name,
                cellnumber,
                valid_param_samples,
                total_iterations,
                water_threshold_percentile_FITC, 
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