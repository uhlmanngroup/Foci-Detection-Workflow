"""
Multi-channel nucleus analysis worker for parallel processing.
Compatible with Windows multiprocessing and existing task structure.
NOW RETURNS WATERSHED LABELS FOR GLOBAL VISUALIZATION

This module contains all functions needed by worker processes to:
1. Detect foci in individual nuclei using adaptive parameter sweeps
2. Compute texture-aware local backgrounds for robust detection
3. Generate watershed segmentation of detected foci
4. Create full-field visualizations of results

The module is designed for parallel processing on Windows, where all
worker functions must be at module level (not nested) for pickling.
"""

# ===============================================================
# IMPORTS
# ===============================================================
# Standard library imports
import os
from collections import Counter

# Scientific computing
import numpy as np

# Image processing - scikit-image
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries
from skimage.morphology import binary_erosion, disk

# Scipy
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, label, binary_dilation

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Other
from PIL import Image


# ===============================================================
# HELPER FUNCTIONS (module-level for multiprocessing)
# ===============================================================
# All functions must be defined at module level (not nested inside other
# functions) to be picklable for Windows multiprocessing.
# ===============================================================




# ===============================================================
# SAVE GLOBAL VISUALIZATION (INCLUDES REAL WATERSHED)
# ===============================================================

def save_global_visualizations(original_image, foci_tritc, foci_fitc, 
                               watershed_labels_tritc, watershed_labels_fitc,
                               well_number, position_number, base_name, output_root):
    """
    Generate 4 full-field visualizations(2 each for FITC and TRITC) with proper filenames.
    Watershed images show filled colored regions (no borders).
    
    This function creates comprehensive visualizations of the entire microscopy field:
    - Two images showing detected foci as colored dots overlaid on the DAPI background
    - Two images showing watershed segmentation regions as filled colored areas
    
    These visualizations are saved as PNG files for quality control and publication.
    Each image is 10x10 inches at 300 DPI for high-resolution output.
    
    Parameters:
    -----------
    original_image : ndarray
        Raw DAPI or merged channel image for background display
        Can be uint16 or float - will be normalized for display
    foci_tritc : list of tuples
        TRITC foci coordinates [(y, x), ...] from all nuclei in the image
        Empty list if no foci detected
    foci_fitc : list of tuples
        FITC foci coordinates [(y, x), ...] from all nuclei in the image
        Empty list if no foci detected
    watershed_labels_tritc : ndarray
        Labeled watershed segmentation for TRITC channel (entire image, all nuclei combined)
        Each detected focus has a unique integer label (0 = background)
    watershed_labels_fitc : ndarray
        Labeled watershed segmentation for FITC channel (entire image, all nuclei combined)
        Each detected focus has a unique integer label (0 = background)
    well_number : str
        Well identifier extracted from filename (e.g., '00044')
        Used in title and filename
    position_number : str
        Position identifier extracted from filename (e.g., '00021')
        Used in title and filename
    base_name : str
        Original filename base (e.g., 'ATR2_24h--W00044--P00021--Z00000--T00000--')
        Preserves original naming convention in output files
    output_root : str
        Root directory for saving images (creates Full_Images_Foci subdirectory)
        Typically the main data folder path
        
    Returns:
    --------
    None
        Saves 4 PNG files to disk:
        1. {base_name}TRITC_foci.png - Red dots on gray background
        2. {base_name}FITC_foci.png - Green dots on gray background
        3. {base_name}TRITC_watershed.png - Colored watershed regions
        4. {base_name}FITC_watershed.png - Colored watershed regions
        
    Notes:
    ------
    - Uses try-except to prevent single visualization failure from crashing pipeline
    - Prints confirmation message for each saved file
    - Random seed ensures reproducible colors across runs
    - Different seeds (42 vs 43) ensure TRITC and FITC use different color palettes
    """

    try:
        # ----------------------------------------------------------------
        # Create output directory if it doesn't exist
        # ----------------------------------------------------------------
        # This folder will contain images for both FITC and TRITC channels:
        # - Foci location overlays (dots on DAPI)
        # - Watershed segmentation results (colored regions)
        debug_dir = os.path.join(output_root, "Full_Images_Foci")
        os.makedirs(debug_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # Normalize the background image for consistent display
        # ----------------------------------------------------------------
        # Rescale image to 0-1 range regardless of input type (uint16, float, etc.)
        # This ensures the grayscale background is properly visible with consistent brightness
        # 'in_range='image'' auto-detects the input range (e.g., 0-65535 for uint16)
        # 'out_range=(0, 1)' sets output to standard float range for matplotlib
        vis_img = exposure.rescale_intensity(original_image, in_range='image', out_range=(0, 1))

        # ================================================================
        # 1️⃣ TRITC FOCI OVERLAY (RED DOTS ON GRAY BACKGROUND)
        # ================================================================
        # Creates an image showing all detected TRITC foci as red dots
        # This gives an overview of TRITC foci distribution across the entire field
        # Useful for: quality control, pattern recognition, spatial distribution analysis
        
        # Create new figure with specified size (10x10 inches)
        # Large size ensures visibility when zooming in on dense regions
        plt.figure(figsize=(10, 10))
        
        # Display the DAPI background in grayscale
        # This provides anatomical context (nucleus locations/shapes)
        plt.imshow(vis_img, cmap='gray')
        
        # Plot each TRITC focus as a small red dot
        # Loop through all foci coordinates and plot individually
        # Why individual plots instead of scatter? Better control over appearance
        for (y, x) in foci_tritc:
            # Plot at (x, y) because matplotlib uses (x, y) convention
            # 'ro' = red circles
            # markersize=0.35 makes dots visible but not overwhelming (can adjust for publication)
            # alpha=0.7 adds slight transparency to see overlapping foci and background
            plt.plot(x, y, 'ro', markersize=0.35, alpha=0.7)
        
        # Add informative title with well and position identifiers
        # fontsize=14 for readability
        plt.title(f"TRITC Foci | Well {well_number} Position {position_number}", fontsize=14)
        
        # Remove axis labels and ticks for cleaner image (no pixel coordinates shown)
        plt.axis('off')
        
        # Adjust layout to minimize white space around the image
        plt.tight_layout()
        
        # Save with original filename convention + channel identifier
        # Preserves experimental metadata in filename for easy organization
        filename = f"{base_name}TRITC_foci.png"
        
        # Save at high resolution (300 DPI = publication quality)
        # bbox_inches='tight' removes extra whitespace around figure
        plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight')
        
        # Close figure to free memory (important when processing many images)
        # Without this, memory usage grows with each image processed
        plt.close()
        
        # Print confirmation (helps track progress during batch processing)
        print(f"  ✓ Saved: {filename}")

        # ================================================================
        # 2️⃣ FITC FOCI OVERLAY (GREEN DOTS ON GRAY BACKGROUND)
        # ================================================================
        # Creates an image showing all detected FITC foci as green dots
        # Same logic as TRITC but with green color ('go') for FITC channel
        
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img, cmap='gray')
        
        # Plot each FITC focus as a small green dot
        # 'go' = green circles
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
        # This allows visual assessment of:
        # - Focus sizes and shapes
        # - Segmentation quality (over/under-segmentation)
        # - Spatial relationships between foci
        
        plt.figure(figsize=(10, 10))
        
        # Get the maximum label number to determine how many colors we need
        # Each watershed region has a unique label (1, 2, 3, ...)
        # Label 0 is always background (not a focus)
        num_labels_tritc = int(watershed_labels_tritc.max())
        
        # Only create visualization if foci were detected
        if num_labels_tritc > 0:
            # --------------------------------------------------------
            # Generate a random color for each watershed region
            # --------------------------------------------------------
            # This ensures neighboring foci are visually distinguishable
            # Without random colors, adjacent foci might look like one region
            
            # Set random seed for reproducibility across runs
            # Same seed = same color palette every time you run the script
            # Seed 42 chosen arbitrarily
            np.random.seed(42)
            
            # Generate random RGB colors for each label
            # Shape: (num_labels + 1, 3) because we need one color per label plus background
            # +1 because labels go from 0 to num_labels (inclusive)
            # Each color is [R, G, B] with values in [0, 1] range
            colors = np.random.rand(num_labels_tritc + 1, 3)
            
            # Force background (label 0) to be black
            # This makes foci stand out against dark background
            colors[0] = [0, 0, 0]
            
            # Create custom colormap from random colors
            # ListedColormap maps integer labels to specific RGB colors
            cmap_tritc = mcolors.ListedColormap(colors)
            
            # --------------------------------------------------------
            # Display as two layers for better visualization
            # --------------------------------------------------------
            # Layer 1: DAPI background at 50% opacity (alpha=0.5) to see nucleus structure
            # This provides anatomical context while not overwhelming the colored foci
            plt.imshow(vis_img, cmap='gray', alpha=0.5)
            
            # Layer 2: Colored watershed regions at 70% opacity (alpha=0.7) overlaid on top
            # Higher opacity for foci so they're clearly visible
            # interpolation='nearest' preserves sharp edges (no blurring between regions)
            plt.imshow(watershed_labels_tritc, cmap=cmap_tritc, alpha=0.7, interpolation='nearest')
        else:
            # If no foci were detected, just show the background image
            # This prevents error when trying to visualize empty labels
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
        # This prevents confusion when comparing channels side-by-side
        
        plt.figure(figsize=(10, 10))
        
        num_labels_fitc = int(watershed_labels_fitc.max())
        if num_labels_fitc > 0:
            # Generate random colors for FITC watershed regions
            # Different seed (43 vs 42) ensures FITC colors differ from TRITC
            np.random.seed(43)
            colors = np.random.rand(num_labels_fitc + 1, 3)
            colors[0] = [0, 0, 0]  # Background is black
            cmap_fitc = mcolors.ListedColormap(colors)
            
            # Display with same transparency settings as TRITC for consistency
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
        # ----------------------------------------------------------------
        # Error handling to prevent pipeline crashes
        # ----------------------------------------------------------------
        # If any error occurs, print detailed error message with traceback
        # This helps debug issues without crashing the entire analysis pipeline
        # Visualization is not critical to analysis - we can continue without it
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
    Circular foci suggest point-like sources (typical for well-defined foci).
    Elongated foci might indicate:
    - Multiple overlapping foci
    - Edge effects (focus touching nucleus boundary)
    - Segmentation artifacts
    
    Formula: C = 4π * A / P²
    where A = area in pixels, P = perimeter in pixels
    
    Mathematical basis:
    - For a circle: P = 2πr, A = πr², so C = 4π(πr²)/(2πr)² = 1.0
    - For other shapes: C < 1.0, with more irregular shapes having lower values
    
    Parameters:
    -----------
    area : float
        Area of the region in pixels (from regionprops or watershed)
    perimeter : float
        Perimeter of the region in pixels (from regionprops)
        
    Returns:
    --------
    float : Circularity value between 0 and 1
        1.0 = perfect circle
        0.0 = degenerate shape (zero area or perimeter)
        
    Examples:
    ---------
    >>> compute_circularity(100, 35.4)  # Nearly circular focus
    1.0
    >>> compute_circularity(100, 50)  # Elongated focus
    0.5
    """
    # Avoid division by zero if perimeter is zero (shouldn't happen but safety check)
    # Zero area or perimeter indicates degenerate region (single pixel or no boundary)
    if perimeter == 0 or area == 0:
        return 0.0
    
    # Standard circularity formula
    # 4 * π * area / perimeter²
    # A perfect circle has maximum circularity (approaches 1.0)
    # Irregular or elongated shapes have lower values
    circularity = (4 * np.pi * area) / (perimeter ** 2)
        
    # Cap at 1.0 to handle numerical artifacts from discrete pixel measurements
    # This can happen with very small regions where perimeter approximation
    # is less accurate due to pixelation (perimeter might be underestimated)
    # For example, a 3x3 pixel circle might calculate to circularity > 1.0
    return min(circularity, 1.0)

# ===============================================================
# TEXTURE-AWARE BACKGROUND WITH NUCLEUS FALLBACK
# ===============================================================


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
    
    This is the core background estimation function that handles complex scenarios:
    1. Computes local background around each potential focus (annulus method)
    2. Detects dense foci regions where local background is unreliable
    3. Falls back to per-nucleus background in dense regions (not global)
    4. Computes texture metrics (CV, variance) for each nucleus
    5. Returns texture info for post-hoc filtering
    
    KEY INNOVATION: Dense regions use PER-NUCLEUS background, not global.
    This is critical because:
    - Global background fails for bright nuclei (over-corrects)
    - Local annulus fails in dense foci regions (samples other foci)
    - Per-nucleus background is the appropriate middle ground
    
    TEXTURE METRICS: Coefficient of Variation (CV) = std / mean
    - High CV (>0.25) = spotty, textured nucleus (real foci expected)
    - Low CV (<0.15) = uniform nucleus (foci may be artifacts)
    - Moderate CV (0.15-0.25) = intermediate texture
    --> Needs to be tested, above values are a rough estimate
    
    The function operates in TWO passes:
    Pass 1: Compute per-nucleus statistics (backgrounds and texture metrics)
    Pass 2: For each candidate focus, choose appropriate background method
    
    Parameters:
    -----------
    image : ndarray
        Fluorescence image (float, 0-1 range) to measure backgrounds from
    coords : ndarray, shape (N, 2)
        Coordinates of candidate foci as (y, x) pairs
    unique_percentiles : array
        Background percentile values to compute (e.g., [10, 20, 30, ...])
        Multiple percentiles allow testing different brightness thresholds
    nucleus_mask : ndarray, optional
        Boolean mask of nucleus region (True = inside nucleus)
        Required for nucleus-aware background estimation
    nucleus_labels : ndarray, optional
        Labeled nucleus mask if multiple nuclei in image
        If None, will be computed from nucleus_mask
    inner_radius : int, default=2
        Inner radius of background annulus (pixels)
        Annulus excludes focus center to avoid sampling the focus itself
    outer_radius : int, default=6
        Outer radius of background annulus for standard regions (pixels)
    edge_outer_radius : int, default=12
        Outer radius for edge regions (pixels) - larger to get more samples
    density_threshold : float, default=0.15
        Threshold for detecting dense regions
        If local_median > nucleus_median * (1 + threshold), region is "dense"
    edge_zone_distance : int, default=6
        Distance from nucleus edge to be considered "edge region" (pixels)
    return_texture_info : bool, default=False
        If True, also return texture metrics for each nucleus
        
    Returns:
    --------
    backgrounds : ndarray, shape (N, P)
        Background values for each coordinate at each percentile
        N = number of coordinates, P = number of percentiles
    texture_info : dict (only if return_texture_info=True)
        Dictionary containing:
        - 'nucleus_stats': Per-nucleus texture metrics (CV, mean, std, etc.)
        - 'coord_nucleus_ids': Which nucleus each coordinate belongs to
        - 'coord_texture_flags': Boolean flags for spotty nuclei
        
    Algorithm Overview:
    -------------------
    1. PASS 1: Per-nucleus statistics
       - For each nucleus: compute mean, std, CV, percentiles
       - Store nucleus-wide backgrounds for fallback
       
    2. PASS 2: Per-coordinate backgrounds
       For each candidate focus coordinate:
       a. Determine if near nucleus edge (use larger annulus if so)
       b. Sample pixels in annulus around coordinate
       c. Compute local percentiles from annulus
       d. Check if region is "dense" (many nearby foci)
       e. If dense: use nucleus background (fallback)
          If not dense: use local annulus background
       f. If insufficient annulus pixels: use nucleus background
    """
    
    # ----------------------------------------------------------------
    # Initialize output array
    # ----------------------------------------------------------------
    # N coordinates × P percentiles = N×P background values
    N = coords.shape[0]  # Number of candidate foci
    P = len(unique_percentiles)  # Number of percentile values to compute
    backgrounds = np.zeros((N, P), dtype=float)
    

    # ----------------------------------------------------------------
    # Compute distance from nucleus edge (for edge detection)
    # ----------------------------------------------------------------
    # Edge distances used to identify foci near nucleus boundary
    # These need special handling (larger annulus) because:
    # - Less area available for background sampling
    # - Background may be contaminated by outside-nucleus regions
    edge_distances = None
    if nucleus_mask is not None:
        # Distance transform: each pixel = distance to nearest False pixel
        # Pixels near edge have small values, center pixels have large values
        edge_distances = distance_transform_edt(nucleus_mask)
    
    # ================================================================
    # PASS 1: COMPUTE PER-NUCLEUS BACKGROUNDS AND TEXTURE METRICS
    # ================================================================
    # Build lookup tables of nucleus-wide statistics
    # These will be used as fallback in dense regions
    
    nucleus_backgrounds = {}  # Dict: nucleus_id → array of percentile values
    nucleus_stats = {}        # Dict: nucleus_id → dict of texture metrics
    
    if nucleus_mask is not None:
        # --------------------------------------------------------
        # Get labeled nuclei (one unique ID per nucleus)
        # --------------------------------------------------------
        if nucleus_labels is None:
            # Auto-label connected regions in mask
            # Returns: (labels, num_nuclei)
            nucleus_labels, num_nuclei = label(nucleus_mask)
        else:
            # User provided labels, just count them
            num_nuclei = int(nucleus_labels.max())
        
        # --------------------------------------------------------
        # Handle single nucleus case
        # --------------------------------------------------------
        # Single nucleus case - treat the whole mask as one nucleus
        # This happens when processing one nucleus at a time (typical workflow)
        if num_nuclei == 0 and nucleus_mask.any():
            nucleus_labels = nucleus_mask.astype(int)
            num_nuclei = 1
        
        # --------------------------------------------------------
        # Compute statistics for each nucleus
        # --------------------------------------------------------
        # Loop through each nucleus ID (1, 2, 3, ...) - 0 is background
        for nuc_id in range(1, num_nuclei + 1):
            # Extract just this nucleus's mask
            nuc_mask_single = nucleus_labels == nuc_id
            
            # Get all pixel values in this nucleus
            nuc_pixels = image[nuc_mask_single]
            
            # Skip if no pixels (shouldn't happen but safety check)
            if len(nuc_pixels) == 0:
                continue
            
            # ------------------------------------------------
            # Compute basic statistics
            # ------------------------------------------------
            mean_intensity = np.mean(nuc_pixels)
            std_intensity = np.std(nuc_pixels)
            median_intensity = np.median(nuc_pixels)
            
            # ------------------------------------------------
            # Coefficient of Variation (CV) = std/mean
            # ------------------------------------------------
            # CV is a texture measure: high CV = spotty, low CV = uniform
            # 
            # ✅ FIX: Use proper epsilon threshold (1e-6) instead of exact 0
            # This prevents floating-point precision issues with dim TRITC images
            # where mean_intensity rounds to 0.0 after img_as_float() normalization
            # 
            # Why 1e-6?
            # - img_as_float() scales uint16 (0-65535) to float (0-1)
            # - Very dim images (e.g., mean=10 in uint16) become 10/65535 ≈ 0.00015
            # - But floating point arithmetic can introduce tiny errors
            # - 1e-6 is safely above floating point noise but below any real signal
            if mean_intensity > 1e-6:  # Use epsilon threshold instead of exact 0
                cv = std_intensity / mean_intensity
            else:
                # Nucleus has negligible signal - CV is undefined
                # Setting to 0 is appropriate (indicates no texture to measure)
                # This happens with very dim or empty nuclei
                cv = 0.0
            
            # ------------------------------------------------
            # Additional texture metrics
            # ------------------------------------------------
            # Percentile range (p10 to p90) gives robust measure of intensity spread
            # More robust than std because not affected by outliers
            p10 = np.percentile(nuc_pixels, 10)
            p90 = np.percentile(nuc_pixels, 90)
            percentile_range = p90 - p10
            
            # ------------------------------------------------
            # Store background percentiles for this nucleus
            # ------------------------------------------------
            # These are the fallback backgrounds used in dense regions
            # Compute at all requested percentiles (same as local backgrounds)
            nucleus_backgrounds[nuc_id] = np.percentile(nuc_pixels, unique_percentiles)
            
            # ------------------------------------------------
            # Store comprehensive statistics
            # ------------------------------------------------
            # These enable post-hoc filtering and quality control
            nucleus_stats[nuc_id] = {
                'mean': mean_intensity,
                'median': median_intensity,
                'std': std_intensity,
                'cv': cv,  # KEY METRIC for texture
                'p10': p10,
                'p90': p90,
                'percentile_range': percentile_range,
                'num_pixels': len(nuc_pixels),
                
                # Pre-computed flags for common filtering thresholds
                # Can adjust thresholds based on your data
                'is_spotty': cv > 0.25 and mean_intensity > 1e-6,   # High texture
                'is_uniform': cv < 0.15 and mean_intensity > 1e-6   # Low texture (suspicious)
            }
    
    # ================================================================
    # PRE-COMPUTE ANNULUS MASKS (for efficiency)
    # ================================================================
    # Annulus = ring-shaped region around each focus
    # Used to sample local background while excluding the focus itself
    # 
    # Pre-computing saves time: we use the same mask for every coordinate
    # Just shift it to each coordinate's position
    
    # ----------------------------------------------------------------
    # Standard annulus (for non-edge regions)
    # ----------------------------------------------------------------
    # Create coordinate grids from -outer_radius to +outer_radius
    y_grid, x_grid = np.ogrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]
    
    # Compute distance from center for each point in grid
    distances = np.sqrt(x_grid**2 + y_grid**2)
    
    # Define annulus: points between inner and outer radius
    # inner_radius creates hole in center (excludes the focus)
    # outer_radius limits how far we sample (only nearby background)
    std_annulus = (distances >= inner_radius) & (distances <= outer_radius)
    
    # Get relative coordinates of annulus pixels
    # These will be added to each candidate focus coordinate
    std_y, std_x = np.where(std_annulus)
    std_y -= outer_radius  # Convert to relative coordinates (centered at 0)
    std_x -= outer_radius
    
    # ----------------------------------------------------------------
    # Expanded annulus (for edge regions)
    # ----------------------------------------------------------------
    # Edge regions need larger annulus because less area available
    # Same logic as standard annulus but with edge_outer_radius
    y_grid_big, x_grid_big = np.ogrid[-edge_outer_radius:edge_outer_radius+1, 
                                       -edge_outer_radius:edge_outer_radius+1]
    distances_big = np.sqrt(x_grid_big**2 + y_grid_big**2)
    edge_annulus = (distances_big >= inner_radius) & (distances_big <= edge_outer_radius)
    edge_y, edge_x = np.where(edge_annulus)
    edge_y -= edge_outer_radius  # Convert to relative coordinates
    edge_x -= edge_outer_radius
    
    # ================================================================
    # PASS 2: PROCESS EACH CANDIDATE COORDINATE
    # ================================================================
    # For each potential focus, choose appropriate background method
    
    # Track which nucleus each coordinate belongs to (for return value)
    coord_nucleus_ids = np.zeros(N, dtype=int)
    

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////////////////////////////////
#-----------HERE------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////////////////////////////////



    # Loop through each candidate focus coordinate
    for i, (y, x) in enumerate(coords):
        # --------------------------------------------------------
        # Determine if coordinate is near nucleus edge
        # --------------------------------------------------------
        is_near_edge = False
        if edge_distances is not None:
            # Get distance from this pixel to nucleus edge
            dist_from_edge = edge_distances[y, x]
            
            # Consider "near edge" if within edge_zone_distance pixels
            # These coordinates get special handling (larger annulus)
            is_near_edge = dist_from_edge <= edge_zone_distance
        
        # --------------------------------------------------------
        # Get nucleus ID for this coordinate
        # --------------------------------------------------------
        nuc_id = 0  # 0 means not in any nucleus (shouldn't happen)
        if nucleus_labels is not None:
            nuc_id = int(nucleus_labels[y, x])
        coord_nucleus_ids[i] = nuc_id  # Store for return value
        
        # --------------------------------------------------------
        # Select appropriate annulus based on location
        # --------------------------------------------------------
        if is_near_edge:
            # Near edge: use larger annulus and require more pixels
            annulus_y, annulus_x = edge_y, edge_x
            min_pixels = 15  # Need more pixels because less reliable
        else:
            # Center of nucleus: use standard annulus
            annulus_y, annulus_x = std_y, std_x
            min_pixels = 5  # Fewer pixels sufficient in good regions
        
        # --------------------------------------------------------
        # Compute absolute pixel positions for annulus
        # --------------------------------------------------------
        # Add relative annulus coordinates to focus coordinate
        # This gives us the actual image coordinates to sample
        abs_y = y + annulus_y
        abs_x = x + annulus_x
        
        # --------------------------------------------------------
        # Filter for valid pixels (inside image bounds)
        # --------------------------------------------------------
        # Check that coordinates are within image dimensions
        valid = (abs_y >= 0) & (abs_y < image.shape[0]) & \
                (abs_x >= 0) & (abs_x < image.shape[1])
        
        # --------------------------------------------------------
        # Further filter to only include pixels inside nucleus
        # --------------------------------------------------------
        # This prevents sampling outside-nucleus regions
        # Important near nucleus boundaries
        if nucleus_mask is not None:
            # Get indices where pixels are inside image
            valid_indices = np.where(valid)[0]
            
            # Check if those pixels are also inside nucleus
            nucleus_valid = nucleus_mask[abs_y[valid_indices], abs_x[valid_indices]] > 0
            
            # Update valid mask: pixel must be both in-bounds AND in-nucleus
            valid[valid_indices] = nucleus_valid
        
        # --------------------------------------------------------
        # Compute background based on available data
        # --------------------------------------------------------
        if valid.sum() >= min_pixels:
            # ------------------------------------------------
            # Case 1: Sufficient annulus pixels available
            # ------------------------------------------------
            # Extract pixel values from annulus
            annulus_pixels = image[abs_y[valid], abs_x[valid]]
            
            # Compute local percentiles from annulus
            local_percentiles = np.percentile(annulus_pixels, unique_percentiles)
            
            # ------------------------------------------------
            # Density check (only for non-edge regions)
            # ------------------------------------------------
            # Dense regions have many nearby foci → annulus samples other foci
            # In this case, nucleus background is more appropriate
            if not is_near_edge and nuc_id > 0 and nuc_id in nucleus_stats:
                # Compare local median to nucleus median
                local_median = np.median(annulus_pixels)
                nucleus_median = nucleus_stats[nuc_id]['median']
                
                # Dense detection: local median much higher than nucleus median
                # This indicates annulus is sampling foci, not background
                is_dense = local_median > nucleus_median * (1 + density_threshold)
                
                if is_dense:
                    # Dense region detected: use NUCLEUS background
                    # This is the key innovation - not global, but per-nucleus
                    backgrounds[i, :] = nucleus_backgrounds[nuc_id]
                else:
                    # Normal region: use LOCAL annulus background
                    # This is the standard case for well-separated foci
                    backgrounds[i, :] = local_percentiles
            else:
                # Edge region or no nucleus info: use local annulus background
                # Edge regions can't be checked for density (annulus may be outside)
                backgrounds[i, :] = local_percentiles
        else:
            # ------------------------------------------------
            # Case 2: Insufficient annulus pixels (fallback)
            # ------------------------------------------------
            # Not enough valid pixels in annulus (e.g., near image edge)
            # Fall back to nucleus-wide background if available
            if nuc_id > 0 and nuc_id in nucleus_backgrounds:
                backgrounds[i, :] = nucleus_backgrounds[nuc_id]
            else:
                # Last resort: use intensity at focus coordinate itself
                # This shouldn't happen often but prevents crashes
                backgrounds[i, :] = image[y, x]
    
    # ================================================================
    # RETURN RESULTS
    # ================================================================
    # Return backgrounds, and optionally texture info for post-hoc filtering
    if return_texture_info:
        # Package texture information for return
        texture_info = {
            'nucleus_stats': nucleus_stats,          # Dict of per-nucleus metrics
            'coord_nucleus_ids': coord_nucleus_ids,  # Which nucleus each coord belongs to
            'coord_texture_flags': np.array([        # Boolean array: is nucleus spotty?
                nucleus_stats.get(nid, {}).get('is_spotty', False) 
                for nid in coord_nucleus_ids
            ])
        }
        return backgrounds, texture_info
    else:
        # Just return backgrounds (default behavior)
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
    
    RATIONALE: Uniformly bright nuclei (low CV) often produce false positive foci
    because any small intensity variation looks like a "peak" against the flat background.
    Real foci are usually in nuclei with some texture variation (moderate to high CV).
    
    This function allows post-hoc cleanup based on texture metrics computed
    during background estimation.
    
    Parameters:
    -----------
    foci_coords : Nx2 array
        Detected foci coordinates (y, x)
    nucleus_labels : 2D array
        Labeled nucleus mask (0 = background, 1,2,3,... = nuclei)
    texture_info : dict
        Texture information from compute_adaptive_background_texture_nucleus_fallback
        Must contain 'nucleus_stats' key
    filter_uniform_bright : bool, default=True
        If True, remove foci from nuclei with CV < min_cv_for_foci
    min_cv_for_foci : float, default=0.15
        Minimum coefficient of variation for nucleus to be considered
        "spotty enough" to have real foci
        Lower threshold = more permissive (keep more foci)
        Higher threshold = more stringent (remove more suspicious foci)
        
    Returns:
    --------
    filtered_coords : Mx2 array (M <= N)
        Filtered foci coordinates (M ≤ N, some foci removed)
    filter_mask : N-length bool array
        True = kept, False = filtered out
        Useful for tracking which foci were removed
        
    Example:
    --------
    >>> filtered_coords, mask = filter_foci_by_texture(
    ...     foci_coords, nucleus_labels, texture_info,
    ...     filter_uniform_bright=True, min_cv_for_foci=0.20
    ... )
    >>> print(f"Removed {(~mask).sum()} suspicious foci from uniform nuclei")
    """
    N = foci_coords.shape[0]
    filter_mask = np.ones(N, dtype=bool)  # Start with all foci kept
    
    # Extract nucleus statistics from texture info
    nucleus_stats = texture_info['nucleus_stats']
    
    # Check each focus
    for i, (y, x) in enumerate(foci_coords):
        # Get which nucleus this focus belongs to
        nuc_id = int(nucleus_labels[y, x])
        
        if nuc_id == 0:
            # Not in a nucleus - keep it (shouldn't happen but safety check)
            continue
        
        if nuc_id not in nucleus_stats:
            # No stats for this nucleus - keep it to be safe
            # Missing stats shouldn't happen but avoid removing valid foci
            continue
        
        # Get texture stats for this nucleus
        stats = nucleus_stats[nuc_id]
        
        # Apply texture-based filtering
        if filter_uniform_bright and stats['is_uniform']:
            # Nucleus is very uniform (low CV) - likely false positives
            # Mark this focus for removal
            filter_mask[i] = False
    
    # Extract coordinates that passed filter
    filtered_coords = foci_coords[filter_mask]
    
    return filtered_coords, filter_mask


# ============================================================================
# HELPER: Generate report on nucleus characteristics
# ============================================================================
def generate_nucleus_texture_report(texture_info):
    """
    Generate a summary report of nucleus texture characteristics.
    Useful for QC and deciding filtering thresholds.
    
    This function helps researchers understand their data by providing
    summary statistics about nucleus texture across the image.
    
    Use this to:
    - Assess data quality (are most nuclei uniform or spotty?)
    - Choose appropriate CV thresholds for filtering
    - Identify problematic images (e.g., all nuclei very dim)
    
    Parameters:
    -----------
    texture_info : dict
        Texture information from compute_adaptive_background_texture_nucleus_fallback
        Must contain 'nucleus_stats' key
        
    Returns:
    --------
    report : dict
        Summary statistics about nuclei in the image:
        - num_nuclei: Total number of nuclei
        - num_spotty_nuclei: Count with CV > 0.25
        - num_uniform_nuclei: Count with CV < 0.15
        - cv_mean: Average CV across all nuclei
        - cv_median: Median CV
        - cv_range: (min CV, max CV)
        - intensity_mean: Average mean intensity across nuclei
        - intensity_median: Median mean intensity
        - intensity_range: (min mean, max mean)
        
    Example:
    --------
    >>> report = generate_nucleus_texture_report(texture_info)
    >>> print(f"Found {report['num_uniform_nuclei']} suspicious uniform nuclei")
    >>> print(f"Median CV: {report['cv_median']:.3f}")
    """
    nucleus_stats = texture_info['nucleus_stats']
    
    # Check if we have any nucleus data
    if not nucleus_stats:
        return {"error": "No nucleus statistics available"}
    
    # Extract CV and mean intensity for all nuclei
    cvs = [stats['cv'] for stats in nucleus_stats.values()]
    means = [stats['mean'] for stats in nucleus_stats.values()]
    
    # Count nuclei by texture category
    num_spotty = sum(1 for stats in nucleus_stats.values() if stats['is_spotty'])
    num_uniform = sum(1 for stats in nucleus_stats.values() if stats['is_uniform'])
    
    # Build report dictionary
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
    
    The three-stage filtering process:
    1. ABSOLUTE brightness filter: Is the focus bright enough overall?
       - Filters out very dim spots that are likely noise
       - Uses global percentile threshold computed from entire image
       
    2. LOCAL CONTRAST filter: Is the focus brighter than its local background?
       - Filters out spots that aren't elevated above surroundings
       - Uses adaptive local backgrounds computed per focus
       
    3. SPATIAL MATCHING: Does the same focus appear in both filtered and unfiltered images?
       - Confirms focus is a real feature, not filtering artifact
       - Matches coordinates within tolerance distance (typically 2 pixels)
    
    Only foci that pass all three criteria are considered valid detections.
    
    Parameters:
    -----------
    p_idx : int
        Index of the parameter combination being tested (0 to len(valid_param_samples)-1)
    bright_pcts : array
        Array of brightness percentile thresholds (one per parameter combo)
        e.g., [10, 15, 20, ...] - which percentile of pixels to use as local background
    contrast_threshs : array
        Array of contrast threshold multipliers (one per parameter combo)
        e.g., [2.0, 2.5, 3.0, ...] - how much brighter than background must focus be?
    percentile_vals : array
        Array of global percentile values for absolute brightness (one per combo)
        e.g., [5, 10, 15, ...] - global percentile for minimum acceptable brightness
    min_brightness_per_param : array
        Precomputed minimum brightness thresholds from global percentiles
        Shape: (num_params,) - one threshold per parameter combination
    bright_to_idx : dict
        Mapping from brightness percentile to column index in local_percentiles arrays
        e.g., {10.0: 0, 15.0: 1, 20.0: 2, ...}
        Used to look up correct column for current parameter
    unf_intensities : array
        Peak intensities in the unfiltered image (raw intensity at each candidate)
    filt_intensities : array
        Peak intensities in the filtered (DoG) image (enhanced intensity at each candidate)
    local_percentiles_unf : ndarray, shape (N_unf, P)
        Local background percentiles for unfiltered peaks
        Each row = one candidate, each column = one percentile value
    local_percentiles_filt : ndarray, shape (N_filt, P)
        Local background percentiles for filtered peaks
        Each row = one candidate, each column = one percentile value
    distances : ndarray, shape (N_unf, N_filt)
        Distance matrix between unfiltered and filtered peak coordinates
        Used for spatial matching (finding same focus in both images)
    unf_yx : ndarray, shape (N_unf, 2)
        Coordinates of unfiltered peaks as (y, x) pairs
    tolerance : int
        Maximum pixel distance to consider two peaks as "the same" (typically 2)
        Allows for slight spatial shifts due to filtering
        
    Returns:
    --------
    tuple : (confirmed_coords, count)
        confirmed_coords : ndarray of shape (M, 2)
            Coordinates of foci that passed all filters
            M ≤ N_unf (some candidates filtered out)
        count : int
            Number of confirmed foci (convenience, same as len(confirmed_coords))
            
    Algorithm Flow:
    ---------------
    1. Extract parameters for this iteration
    2. Apply absolute brightness filter → get unf_mask_abs, filt_mask_abs
    3. Apply local contrast filter → get unf_mask_con, filt_mask_con
    4. Combine filters (AND) → get unf_final_mask, filt_final_mask
    5. Spatially match candidates between images
    6. Return coordinates that passed all stages
    """
    # ================================================================
    # EXTRACT PARAMETERS FOR THIS ITERATION
    # ================================================================
    # Each parameter combination defines one set of detection thresholds
    bright_pct = bright_pcts[p_idx]           # Local background percentile threshold
    contrast_thresh = contrast_threshs[p_idx]  # Contrast multiplier (e.g., 2.5x background)
    min_brightness = min_brightness_per_param[p_idx]  # Absolute brightness threshold
    
    # Map the brightness percentile to the correct column in the local percentiles array
    # Round to avoid floating point comparison issues (e.g., 10.0 vs 10.0000001)
    bright_key = np.round(bright_pct, 6)
    b_idx = bright_to_idx[bright_key]
    
    # ================================================================
    # STEP 1: ABSOLUTE BRIGHTNESS FILTER
    # ================================================================
    # Check if peak intensities exceed the global minimum brightness threshold
    # This filters out very dim spots that are likely noise regardless of local context
    # 
    # Rationale: Even if a spot is brighter than local background, if it's too dim
    # overall (below global percentile), it's probably just noise/background variation
    unf_mask_abs = unf_intensities >= min_brightness
    filt_mask_abs = filt_intensities >= min_brightness
    
    # Early exit: if no peaks pass absolute brightness in either image, return empty result
    # No point continuing if we have no candidates
    if not np.any(unf_mask_abs) or not np.any(filt_mask_abs):
        return np.array([]).reshape(0, 2), 0
    
    # ================================================================
    # STEP 2: LOCAL CONTRAST FILTER
    # ================================================================
    # Check if peaks are sufficiently brighter than their local background
    # This is the key adaptive filter that handles variable background intensity
    
    # Extract the local background value at the specified percentile for each peak
    # b_idx is the column corresponding to current bright_pct
    unf_local_bg = local_percentiles_unf[:, b_idx]
    filt_local_bg = local_percentiles_filt[:, b_idx]
    
    # Apply contrast threshold: peak must be > (local_background × contrast_thresh)
    # Example: if contrast_thresh=2.5, peak must be 2.5× brighter than local background
    # 
    # Why multiply instead of subtract?
    # - Multiplicative: adapts to local intensity level (2× bright spot in bright region
    #   vs 2× bright spot in dim region both require same relative contrast)
    # - Additive would fail: fixed threshold too strict in bright regions, too loose in dim
    unf_mask_con = unf_intensities > (unf_local_bg * contrast_thresh)
    filt_mask_con = filt_intensities > (filt_local_bg * contrast_thresh)
    
    # ================================================================
    # STEP 3: COMBINE FILTERS
    # ================================================================
    # A valid peak must pass BOTH absolute brightness AND local contrast filters
    # This is a logical AND operation
    # 
    # Why both filters?
    # - Absolute brightness alone: would miss dim but high-contrast foci
    # - Local contrast alone: would accept noise in bright regions
    # - Together: ensures foci are both bright enough overall AND elevated locally
    unf_final_mask = unf_mask_abs & unf_mask_con
    filt_final_mask = filt_mask_abs & filt_mask_con
    
    # Get the indices of peaks that passed all filters
    # These are the row indices in the arrays
    unf_idxs = np.where(unf_final_mask)[0]
    filt_idxs = np.where(filt_final_mask)[0]
    
    # Early exit: if no peaks passed filters in either image, return empty result
    if unf_idxs.size == 0 or filt_idxs.size == 0:
        return np.array([]).reshape(0, 2), 0
    
    # ================================================================
    # STEP 4: SPATIAL MATCHING
    # ================================================================
    # Match filtered and unfiltered foci: only keep foci that appear in BOTH images
    # This confirms that the focus is a real feature, not an artifact of filtering
    # 
    # Rationale: If a bright spot appears in unfiltered image but disappears after
    # DoG filtering, it's likely a large diffuse region (not a point focus).
    # If it appears only in filtered image, it's likely a filtering artifact.
    # Real foci appear in both images (though position may shift slightly due to filtering).
    
    # Extract the sub-matrix of distances between valid unfiltered and filtered peaks
    # This is a smaller matrix containing only peaks that passed filters
    # Shape: (num_valid_unf, num_valid_filt)
    distances_sub = distances[unf_idxs][:, filt_idxs]
    
    # For each valid unfiltered peak, find the distance to its nearest valid filtered peak
    # axis=1 means "min across columns" = for each unf peak, find closest filt peak
    nearest_dist = np.min(distances_sub, axis=1)
    
    # Keep only unfiltered peaks that have a matching filtered peak within tolerance
    # Tolerance of 2 pixels allows for slight spatial shifts due to filtering
    # (DoG can slightly shift peak locations while preserving the feature)
    confirmed_unf_idxs = unf_idxs[nearest_dist <= tolerance]
    
    # Get the final coordinates of confirmed foci
    # These are the foci that passed all three stages
    confirmed_coords = unf_yx[confirmed_unf_idxs]
    
    # Return both the coordinates and the count
    # Count is redundant (could compute len(confirmed_coords)) but convenient for caller
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
    - Useful for normalization across images or experiments
    - Enables comparison between channels (e.g., TRITC vs FITC expression levels)
    
    This is separate from foci detection - it measures the total cellular signal
    regardless of whether that signal is concentrated in foci or diffuse.
    
    Parameters:
    -----------
    nucleus_mask : ndarray (boolean)
        Binary mask indicating which pixels belong to this nucleus
        True = inside nucleus, False = outside
    image : ndarray
        The image to measure (should be float, 0-1 range)
        Already converted by img_as_float in caller
    channel_name : str
        Name of the channel (e.g., 'TRITC', 'FITC', 'Cy5', 'DAPI')
        Used to create descriptive column names in output
        
    Returns:
    --------
    dict : Dictionary with two keys:
        '{channel_name}_total_intensity' : Sum of all pixel intensities in nucleus
        '{channel_name}_mean_intensity' : Average pixel intensity in nucleus
        
    Example:
    --------
    >>> nucleus_mask = (labels == 5)  # Extract nucleus #5
    >>> tritc_img = img_as_float(tritc_raw)
    >>> result = analyze_channel_intensity(nucleus_mask, tritc_img, 'TRITC')
    >>> print(result)
    {'TRITC_total_intensity': 0.45, 'TRITC_mean_intensity': 0.00023}
    """
    # Extract only the pixels belonging to this nucleus
    # Boolean indexing: takes only pixels where mask is True
    nucleus_pixels = image[nucleus_mask]
    
    # Calculate total intensity: sum of all pixel values
    # This represents the total amount of signal in the nucleus
    # Higher values = more fluorescence (more protein, more RNA, etc.)
    # Interpretation: if nucleus is 1000 pixels and mean intensity is 0.5,
    # total intensity is 500 (integrated signal across entire nucleus)
    total_intensity = float(np.sum(nucleus_pixels))
    
    # Calculate mean intensity: average pixel value
    # This represents the average brightness, normalized by nucleus size
    # Useful for comparing nuclei of different sizes
    # Interpretation: mean intensity reflects average concentration/expression level
    mean_intensity = float(np.mean(nucleus_pixels))
    
    # Return as dictionary with channel-specific keys for easy DataFrame creation
    # These keys will become column names when accumulated into pandas DataFrame
    # Format: {CHANNEL}_total_intensity, {CHANNEL}_mean_intensity
    return {
        f"{channel_name}_total_intensity": total_intensity,
        f"{channel_name}_mean_intensity": mean_intensity,
    }


# ===============================================================
# FOCI DETECTION FOR ONE CHANNEL - FIXED VERSION
# ===============================================================
# KEY FIX: Texture CV is now calculated BEFORE foci detection
# This allows distinguishing between:
#   - Case A: Uniform bright (no foci) → low CV
#   - Case B: Dense foci (detection failed) → high CV
# ===============================================================

def detect_foci_single_channel(
    nucleus_mask, image, original_image, channel_name, cell_id,
    valid_param_samples, total_iterations, water_threshold_percentile,
    watershed_min_detection_prob=0.0, 
    well_number=None, position_number=None,
    calibration_mode=False,
    calibration_tracker=None,
    image_id=None,
    min_cv_threshold=0.20,              
    uniform_contrast_multiplier=1.0   
):
    """
    Detect foci in a single nucleus region for one channel.
    Returns: (foci_list, summary_dict, watershed_labels)
    
    ✅ FIXED: Texture CV calculated FIRST, independent of foci detection
    
    This is the core foci detection function that:
    1. Calculates nucleus texture (CV) before any detection
    2. Applies DoG filtering to enhance foci
    3. Finds candidate foci using peak detection
    4. Tests candidates against multiple parameter combinations
    5. Performs watershed segmentation on confirmed foci
    6. Measures properties of each detected focus
    
    The function supports both production and calibration modes:
    - Production: detect foci with optimized parameters
    - Calibration: test many parameters to find optimal ones
    
    Parameters:
    -----------
    nucleus_mask : ndarray (boolean)
        Binary mask for this nucleus (True = inside nucleus)
    image : ndarray (float, 0-1)
        Channel image to analyze (e.g., TRITC or FITC)
        Already converted to float by caller
    original_image : ndarray (float, 0-1)
        Original (unfiltered) image for global percentile calculations
        Used to set absolute brightness thresholds
    channel_name : str
        Name of channel ('TRITC', 'FITC', etc.)
    cell_id : int
        Nucleus identifier (cell number)
    valid_param_samples : ndarray, shape (N, 3)
        Parameter combinations to test: [bright_pct, contrast_thresh, percentile_val]
        N = 256 during calibration, 1-3 during production
    total_iterations : int
        Total number of parameter combinations (len(valid_param_samples))
    water_threshold_percentile : float
        Brightness threshold for watershed segmentation (0-100)
        Defines minimum brightness for watershed regions
    watershed_min_detection_prob : float, default=0.0
        Minimum detection probability to include focus in watershed (0-100)
        If 50%, focus must be detected by >50% of parameter combinations
    well_number : str, optional
        Well identifier for tracking
    position_number : str, optional
        Position identifier for tracking
    calibration_mode : bool, default=False
        If True, record results for calibration analysis
    calibration_tracker : object, optional
        Tracker object to store calibration results
    image_id : str, optional
        Image identifier for calibration tracking
        
    Returns:
    --------
    foci_list : list of dict
        List of detected foci, each dict containing:
        - cell_num: nucleus ID
        - centr_y, centr_x: focus coordinates
        - foci_area: size in pixels
        - foci_circularity: shape measure (0-1)
        - foci_total_intensity: integrated intensity
        - foci_mean_intensity: average intensity
        - detection_prob: % of parameters that detected this focus
        - channel: channel name
    summary_dict : dict
        Nucleus-level summary statistics:
        - {channel}_nucleus_cv: texture coefficient of variation
        - {channel}_texture_applied: was texture filtering used?
        - {channel}_mean_foci_intensity: average across all foci
        - {channel}_std_foci_intensity: standard deviation across foci
        - {channel}_min/max_foci_intensity: range across foci
        - {channel}_mean/std/min/max_foci: statistics on foci counts
          across parameter combinations
    watershed_labels : ndarray (int)
        Labeled watershed segmentation (0 = background, 1,2,3,... = foci)
        Same shape as input image
        None if no foci detected
        
    Algorithm Overview:
    -------------------
    1. Create isolated image (zero outside nucleus)
    2. Calculate texture CV (KEY: before detection)
    3. Apply DoG filter to enhance foci
    4. Find candidate foci via peak detection
    5. Compute adaptive local backgrounds
    6. Test all parameter combinations
    7. Record calibration data if in calibration mode
    8. Perform watershed segmentation
    9. Measure each detected focus
    10. Return results
    """
    
    # ============================================================
    # HELPER FUNCTION FOR EARLY EXITS
    # ============================================================
    def return_empty(reason="", nucleus_cv_val=None):
        """
        Return empty results but with COMPLETE nucleus summary.
        
        This helper ensures consistent return format even when detection fails.
        Critical: texture CV must be included even when no foci detected!
        
        Parameters:
        -----------
        reason : str
            Why we're returning empty (for logging)
        nucleus_cv_val : float
            Texture CV value to include in summary
            
        Returns:
        --------
        (empty_foci_list, summary_dict, None_watershed)
        """
        
        # Use provided CV if available, otherwise 0.0 (fallback for very early exits)
        cv_to_use = nucleus_cv_val if nucleus_cv_val is not None else 0.0
        
        # Build complete summary with texture info even when no foci
        empty_summary = {
            # ✅ FIXED: Include texture even when no foci
            f"{channel_name}_nucleus_cv": cv_to_use,
            f"{channel_name}_texture_applied": False,
            
            # Foci intensity statistics (None when no foci)
            # These will show as "None" in CSV, not as missing columns
            f"{channel_name}_mean_foci_intensity": "None",
            f"{channel_name}_std_foci_intensity": "None",
            f"{channel_name}_min_foci_intensity": "None",
            f"{channel_name}_max_foci_intensity": "None",
            
            # Parameter sweep statistics (zeros when no foci)
            f"{channel_name}_mean_foci": 0.0,
            f"{channel_name}_std_foci": 0.0,
            f"{channel_name}_min_foci": 0,
            f"{channel_name}_max_foci": 0,
        }
        
        # Print diagnostic message if reason provided
        if reason:
            print(f"      Cell {cell_id} ({channel_name}): {reason} - returning empty (CV={cv_to_use:.3f})")
        
        # Return: (empty foci list, summary with texture, no watershed)
        return [], empty_summary, None
        
    # ============================================================
    # STEP 1: Create isolated image
    # ============================================================
    # Zero out pixels outside nucleus to isolate signal
    # This prevents edge effects and ensures we only analyze this nucleus
    
    # Create copy and convert to float (may already be float but ensure consistency)
    isolated_img = img_as_float(image.copy())
    
    # Zero out all pixels outside nucleus mask
    # After this: isolated_img has signal only within nucleus, zeros elsewhere
    isolated_img[~nucleus_mask] = 0
    
    # Early exit: if no signal in isolated image (all zeros)
    # This can happen with very dim images or segmentation errors
    if isolated_img.max() == 0:
        return return_empty("No signal in isolated nucleus", nucleus_cv_val=0.0)

    
    # ============================================================
    # ✅ NEW STEP 2: CALCULATE TEXTURE FIRST (BEFORE foci detection!)
    # ============================================================
    # This is the KEY FIX that distinguishes:
    # - Case A: Uniformly bright nucleus (low CV) with false positive "foci" from noise
    # - Case B: Nucleus with many real foci (high CV) where detection happened to fail
    #
    # Previously, texture was only calculated IF foci were detected
    # → couldn't distinguish these cases
    # Now, texture calculated FIRST, independent of detection success
    
    t_texture_start = time.time()
    
    # Calculate texture on WHOLE NUCLEUS, independent of foci detection
    # Extract all pixels within nucleus mask
    nucleus_pixels = isolated_img[nucleus_mask]
    mean_intensity = np.mean(nucleus_pixels)
    std_intensity = np.std(nucleus_pixels)
    
    # Calculate CV with proper epsilon threshold
    # CV = coefficient of variation = std / mean
    # High CV = spotty, textured nucleus (real foci expected)
    # Low CV = uniform nucleus (foci likely false positives)
    if mean_intensity > 1e-6:  # Epsilon threshold for numerical stability
        nucleus_cv = float(std_intensity / mean_intensity)
    else:
        # Mean intensity too close to zero - CV undefined
        # Set to 0 (indicates no measurable texture)
        nucleus_cv = 0.0
    
    # Classify nucleus texture using standard thresholds
    min_cv_threshold = 0.20  # Below this = uniform (suspicious)
    nucleus_is_uniform = nucleus_cv < min_cv_threshold
    nucleus_is_spotty = nucleus_cv > 0.25  # Above this = spotty (good for foci)
    
    # Create human-readable label for logging
    texture_label = 'uniform' if nucleus_is_uniform else ('spotty' if nucleus_is_spotty else 'moderate')
    
    # Print texture classification (helps track what's happening)
    print(f"      Cell {cell_id} ({channel_name}): Texture CV={nucleus_cv:.3f} ({texture_label})")
    
    
    # ============================================================
    # STEP 3: Apply DoG filter
    # ============================================================
    # Difference of Gaussians enhances foci by subtracting smoothed versions
    # This acts as a band-pass filter highlighting features of specific size
    
    t1 = time.time()
    
    # Apply DoG with low_sigma=1, high_sigma=2
    # This highlights features ~1-2 pixels in size (typical focus size)
    # Larger features (diffuse signal) are suppressed
    filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
    
    # Clip negative values (DoG can produce negatives where dark surrounds bright spots)
    # We only care about positive features (bright spots)
    filtered_img = np.clip(filtered_img, 0, None)
    
    # Rescale intensity to match original image range
    # This ensures filtered image has same dynamic range as unfiltered
    # Important for consistent thresholding across images
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', 
                                             out_range=(0, isolated_img.max()))

    
    # ============================================================
    # STEP 4: Detect foci candidates
    # ============================================================
    # Find local maxima in both filtered and unfiltered images
    # Candidates must appear in both to be considered real
    
    t2 = time.time()
    
    # Extract parameters from valid_param_samples array
    # Each row is [bright_pct, contrast_thresh, percentile_val]
    bright_pcts = valid_param_samples[:, 0]      # Column 0: brightness percentile
    contrast_threshs = valid_param_samples[:, 1]  # Column 1: contrast threshold
    percentile_vals = valid_param_samples[:, 2]   # Column 2: global percentile
    
    # Use original_image for global percentile calculations
    # This ensures absolute brightness thresholds are consistent across nuclei
    # original_image is the full-field image (not isolated to this nucleus)
    pos_pixels = original_image[original_image > 0]
    
    # Early exit: if no positive pixels in original image
    # This shouldn't happen but prevents crashes
    if pos_pixels.size == 0:
        return return_empty("No positive pixels in original image", nucleus_cv_val=nucleus_cv)
    
    # Compute minimum brightness for each parameter combination
    # These are the absolute brightness thresholds (global percentiles)
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    
    # Get the minimum across all parameters (most permissive threshold)
    # Use this for initial candidate finding (we'll apply stricter filters later)
    global_min_brightness = np.min(min_brightness_per_param)
    
    # Find candidate foci using peak_local_max
    # min_distance=2: peaks must be at least 2 pixels apart
    # threshold_abs=global_min_brightness: peaks must exceed this absolute value
    candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                        threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                          threshold_abs=global_min_brightness)

    
    # ============================================================
    # ✅ CRITICAL: If NO foci detected, we STILL have texture CV!
    # ============================================================
    # This is the key improvement: texture is independent of detection
    # We can return meaningful summary even when no foci found
    if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
        # Build complete summary with texture info
        # This distinguishes "no foci, uniform nucleus" from "no foci, spotty nucleus"
        summary = {
            f"{channel_name}_nucleus_cv": nucleus_cv,  # ← KEY: includes texture
            f"{channel_name}_texture_applied": nucleus_is_uniform,
            
            # No foci detected, so intensity stats are "None"
            f"{channel_name}_mean_foci_intensity": "None",
            f"{channel_name}_std_foci_intensity": "None",
            f"{channel_name}_min_foci_intensity": "None",
            f"{channel_name}_max_foci_intensity": "None",
            
            # Parameter sweep found zero foci for all combinations
            f"{channel_name}_mean_foci": 0.0,
            f"{channel_name}_std_foci": 0.0,
            f"{channel_name}_min_foci": 0,
            f"{channel_name}_max_foci": 0,
        }
        
        print(f"      Cell {cell_id} ({channel_name}): No foci detected, texture CV={nucleus_cv:.3f}")
        return [], summary, None
    
    # ============================================================
    # STEP 5: Extract coordinates and intensities
    # ============================================================
    # Convert candidate coordinates to arrays and extract intensities
    
    # Convert coordinates to integer arrays (peak_local_max returns them as lists)
    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    
    # Extract intensity at each candidate coordinate
    # These are the peak intensities we'll compare against thresholds
    filt_intensities = filtered_img[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_img[unf_yx[:, 0], unf_yx[:, 1]]
    
    # Prepare brightness percentile mapping
    # bright_pcts may have duplicates, get unique values
    unique_brights = np.unique(np.round(bright_pcts, 6))
    
    # Create mapping: brightness percentile → column index in local_percentiles array
    # This allows quick lookup during filtering
    bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}
    
    # ============================================================
    # STEP 6: Compute local backgrounds
    # ============================================================
    # Calculate adaptive backgrounds for each candidate focus
    # This is the most computationally expensive step
    
    # Use optimized vectorized version for background computation
    # return_texture_info=True gives us texture metrics (for post-hoc filtering)
    local_percentiles_unf, texture_info_unf = compute_adaptive_background_texture_nucleus_fallback(
        image=isolated_img,
        coords=unf_yx,
        unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask,
        return_texture_info=True
    )
    
    # Same for filtered image candidates
    local_percentiles_filt, texture_info_filt = compute_adaptive_background_texture_nucleus_fallback(
        image=filtered_img, 
        coords=filt_yx, 
        unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask,
        return_texture_info=True
    )

    
    # ============================================================
    # STEP 7: Apply contrast adjustments for uniform nuclei
    # ============================================================
    # Uniform nuclei get stricter filtering to reduce false positives
    
    contrast_multiplier = 1.0  # Default: no adjustment
    
    if nucleus_is_uniform:
        # Uniform nucleus detected - apply stricter filters
        # contrast_multiplier = 1.5 would make contrast threshold 1.5× stricter
        # Currently set to 1.0 (no adjustment) but infrastructure is in place
        print(f"    ⚠️ Cell {cell_id}: Low texture (CV={nucleus_cv:.3f}) - applying stricter filters")
        contrast_multiplier = uniform_contrast_multiplier
    
    # ============================================================
    # STEP 8: Test all parameter combinations
    # ============================================================
    # Loop through each parameter combination and count detected foci
    
    # Pre-compute distance matrix for spatial matching (used in every iteration)
    # Shape: (num_unfiltered_candidates, num_filtered_candidates)
    distances = cdist(unf_yx, filt_yx)
    
    # Tolerance for spatial matching (2 pixels)
    tolerance = 2
    
    # Initialize accumulators
    foci_counts = []  # Number of foci detected by each parameter combination
    all_detected_foci = []  # All foci coordinates across all parameters
    
    # Test each parameter combination
    for p_idx in range(len(valid_param_samples)):
        # Adjust contrast threshold for uniform nuclei
        adjusted_contrast_threshs = contrast_threshs.copy()
        if nucleus_is_uniform:
            # Apply contrast multiplier (currently 1.0, no effect)
            adjusted_contrast_threshs = contrast_threshs * contrast_multiplier
        
        # Apply filters for this parameter combination
        # Returns: (coordinates of confirmed foci, count)
        confirmed_coords, count = apply_foci_filters(
            p_idx, bright_pcts, adjusted_contrast_threshs, percentile_vals,
            min_brightness_per_param, bright_to_idx,
            unf_intensities, filt_intensities,
            local_percentiles_unf, local_percentiles_filt,
            distances, unf_yx, tolerance
        )
        
        # Record count for this parameter combination
        foci_counts.append(count)
        
        # Add confirmed foci to global list (for detection probability calculation)
        # Store as tuples so we can count occurrences with Counter
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord))


    # ============================================================
    # STEP 9: Record calibration data
    # ============================================================
    # If in calibration mode, record results for parameter optimization
    
    if calibration_mode and calibration_tracker is not None and image_id is not None:
        # Record result for each parameter combination tested
        for p_idx in range(len(valid_param_samples)):
            # Extract parameter combination as tuple
            param_combo = tuple(valid_param_samples[p_idx])
            
            # Record in tracker (for later analysis)
            calibration_tracker.record_calibration_result(
                image_id=image_id,              # Which image
                cell_id=cell_id,                # Which nucleus
                param_combo=param_combo,        # Which parameters
                foci_count=foci_counts[p_idx],  # How many foci detected
                detection_prob=100.0,           # Placeholder (not used currently)
                channel=channel_name            # Which channel
            )
    
    # Early exit: if no foci counts recorded (shouldn't happen but safety check)
    if not foci_counts:
        return return_empty("No foci counts recorded", nucleus_cv_val=nucleus_cv)
    
    # ============================================================
    # STEP 10: Calculate statistics
    # ============================================================
    # Compute statistics across all parameter combinations
    
    # Count how many times each coordinate was detected
    # This gives "detection probability" for each focus
    foci_detection_count = Counter(all_detected_foci)
    
    # Calculate statistics on foci counts across parameters
    mean_foci = np.mean(foci_counts)  # Average number of foci across parameters
    std_foci = np.std(foci_counts)    # Std dev (measures parameter sensitivity)
    min_foci = int(min(foci_counts))  # Minimum foci found by any parameter
    max_foci = int(max(foci_counts))  # Maximum foci found by any parameter
    
    # ============================================================
    # STEP 11: Watershed segmentation
    # ============================================================
    # Segment the area around each detected focus
    # This defines the spatial extent of each focus
    
    # Filter foci by detection probability threshold
    # Only include foci detected by sufficient percentage of parameters
    watershed_foci = []
    for coord, count in foci_detection_count.items():
        # Calculate what percentage of parameters detected this focus
        detection_prob = (count / total_iterations) * 100
        
        # Include if probability exceeds threshold
        if detection_prob >= watershed_min_detection_prob:
            watershed_foci.append(coord)
    
    # Early exit: if no foci passed detection threshold
    if len(watershed_foci) == 0:
        return return_empty("No foci passed detection threshold", nucleus_cv_val=nucleus_cv)
    
    # Convert to numpy array for watershed
    final_coords = np.array(watershed_foci)
    
    # Safety check (redundant but prevents crashes)
    if len(final_coords) == 0:
        return return_empty("No final coordinates", nucleus_cv_val=nucleus_cv)
    
    # --------------------------------------------------------
    # Prepare images for watershed
    # --------------------------------------------------------
    # Rescale filtered image intensity to 0-100 range for consistent thresholding
    # This ensures water_threshold_percentile has consistent meaning
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', out_range=(0, 100))
    
    # Crop to bounding box around foci (for efficiency)
    # No need to process entire image when foci are localized
    y_coords, x_coords = final_coords[:, 0], final_coords[:, 1]
    pad = 25  # Extra padding around foci
    y_min = max(0, y_coords.min() - pad)
    y_max = min(filtered_img.shape[0], y_coords.max() + pad)
    x_min = max(0, x_coords.min() - pad)
    x_max = min(filtered_img.shape[1], x_coords.max() + pad)
    
    # Crop images to bounding box
    filtered_crop = filtered_img[y_min:y_max, x_min:x_max]
    isolated_crop = isolated_img[y_min:y_max, x_min:x_max]
    nucleus_mask_crop = nucleus_mask[y_min:y_max, x_min:x_max]
    
    # Adjust coordinates to cropped space
    # Subtract offset so coordinates are relative to crop
    final_coords_crop = final_coords - np.array([y_min, x_min])
    
    # Erode nucleus mask to avoid edge artifacts
    # disk(2) removes 2-pixel border from nucleus
    nucleus_mask_eroded = binary_erosion(nucleus_mask_crop, disk(2))
    
    # --------------------------------------------------------
    # Create marker image for watershed
    # --------------------------------------------------------
    # Markers are seeds for watershed: each focus gets a unique ID
    markers_crop = np.zeros_like(filtered_crop, dtype=int)
    for idx, (y, x) in enumerate(final_coords_crop, start=1):
        # Place marker at each focus coordinate
        # idx starts at 1 (0 reserved for background)
        markers_crop[int(y), int(x)] = idx
    
    # --------------------------------------------------------
    # Define watershed mask
    # --------------------------------------------------------
    # Watershed will only operate within this mask
    # Include: bright regions above threshold AND eroded nucleus AND marker locations
    binary_mask = (filtered_crop > water_threshold_percentile) & nucleus_mask_eroded
    binary_mask = binary_mask | (markers_crop > 0)  # Ensure markers are included

    # Dilate marker regions slightly to ensure connectivity
    # This prevents markers from being isolated islands
    marker_mask = markers_crop > 0
    dilated_markers = binary_dilation(marker_mask, structure=disk(1))
    binary_mask = binary_mask | dilated_markers
    
    # --------------------------------------------------------
    # Run watershed
    # --------------------------------------------------------
    # Compute distance transform (negative because watershed finds valleys)
    distance = ndi.distance_transform_edt(binary_mask)
    
    # Run watershed segmentation
    # -distance: watershed finds valleys (we want to expand from peaks)
    # markers_crop: starting seeds
    # mask=binary_mask: limit watershed to this region
    # compactness=0.005: slight preference for compact regions
    water_labels_crop = watershed(-distance, markers_crop, mask=binary_mask, compactness=0.005)

    # --------------------------------------------------------
    # Diagnostic: Check if any markers were lost
    # --------------------------------------------------------
    # Sometimes watershed fails to segment all markers (connectivity issues)
    markers_found = np.unique(markers_crop[markers_crop > 0])
    labels_found = np.unique(water_labels_crop[water_labels_crop > 0])

    if len(labels_found) < len(markers_found):
        # Some markers disappeared during watershed
        lost_markers = set(markers_found) - set(labels_found)
        print(f"⚠️ WARNING Cell {cell_id} ({channel_name}): {len(lost_markers)} foci lost in watershed!")
        print(f"   Lost marker IDs: {lost_markers}")
        print(f"   Expected: {len(markers_found)}, Got: {len(labels_found)}")
        # Note: We continue anyway - lost foci will be missing from results
    
    # --------------------------------------------------------
    # Place watershed results back into full-size array
    # --------------------------------------------------------
    # Initialize full-size array (same shape as input image)
    water_labels = np.zeros_like(isolated_img, dtype=int)
    
    # Copy cropped watershed labels into correct position
    water_labels[y_min:y_max, x_min:x_max] = water_labels_crop


    
    # ============================================================
    # STEP 12: Measure each focus
    # ============================================================
    # Extract properties for each detected focus
    
    foci_list = []
    
    # Loop through final confirmed foci coordinates
    for idx, (y, x) in enumerate(final_coords):
        # Get watershed label at this coordinate
        region_id = water_labels[y, x]
        
        # Extract mask for this watershed region
        spot_mask = (water_labels == region_id)
        
        # Measure properties
        spot_area = int(np.sum(spot_mask))  # Area in pixels
        spot_intensity = float(np.sum(isolated_img[spot_mask]))  # Total intensity
        spot_mean_intensity = float(np.mean(isolated_img[spot_mask]))  # Average intensity
        
        # Calculate detection probability for this focus
        # How many parameter combinations detected this focus?
        detection_prob = (foci_detection_count.get((y, x), 0) / total_iterations) * 100
        
        # Calculate circularity for this focus
        # Measure shape quality (1.0 = perfect circle)
        focus_props = measure.regionprops(spot_mask.astype(int))
        if len(focus_props) > 0:
            focus_perimeter = focus_props[0].perimeter
            focus_circularity = compute_circularity(spot_area, focus_perimeter)
        else:
            # No props found (shouldn't happen) - set to 0
            focus_circularity = 0.0
        
        # Add focus to results list
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
    
    # ============================================================
    # STEP 13: Build complete summary
    # ============================================================
    # Create nucleus-level summary with all metrics
    
    # Calculate foci intensity statistics (across all foci in this nucleus)
    if len(foci_list) > 0:
        # Extract mean intensities for all foci
        foci_intensities = [f['foci_mean_intensity'] for f in foci_list]
        
        # Calculate statistics on foci intensities
        mean_foci_intensity = float(np.mean(foci_intensities))
        std_foci_intensity = float(np.std(foci_intensities))
        min_foci_intensity = float(np.min(foci_intensities))
        max_foci_intensity = float(np.max(foci_intensities))
    else:
        # No foci detected - set to "None" string
        # This shows in CSV as "None" rather than empty cell
        mean_foci_intensity = "None"
        std_foci_intensity = "None"
        min_foci_intensity = "None"
        max_foci_intensity = "None"
    
    # Build complete summary dictionary
    # This will be added to nucleus-level DataFrame
    summary = {
        # ✅ FIXED: Texture is ALWAYS included now (calculated early)
        f"{channel_name}_nucleus_cv": float(nucleus_cv),
        f"{channel_name}_texture_applied": nucleus_is_uniform,
        
        # Foci intensity statistics
        f"{channel_name}_mean_foci_intensity": mean_foci_intensity,
        f"{channel_name}_std_foci_intensity": std_foci_intensity,
        f"{channel_name}_min_foci_intensity": min_foci_intensity,
        f"{channel_name}_max_foci_intensity": max_foci_intensity,
        
        # Parameter sweep statistics (variation across parameter combinations)
        f"{channel_name}_mean_foci": float(mean_foci),
        f"{channel_name}_std_foci": float(std_foci),
        f"{channel_name}_min_foci": int(min_foci),
        f"{channel_name}_max_foci": int(max_foci),
    }

    # Return all results
    return foci_list, summary, water_labels


# ===============================================================
# MAIN WORKER FUNCTION (MODIFIED TO RETURN WATERSHED LABELS)
# ===============================================================

def process_single_nucleus(args):
    """
    Process one nucleus across all provided channels.
    
    This is the main worker function called by parallel executor.
    It receives a tuple of arguments, processes one nucleus, and returns results.
    
    NOW ACCEPTS SEPARATE PARAMETER SPACES FOR TRITC AND FITC:
    - valid_param_samples_TRITC: TRITC-specific parameter space (256 or 1-3 params)
    - valid_param_samples_FITC: FITC-specific parameter space (256 or 1-3 params)
    - total_iterations_TRITC: number of TRITC iterations
    - total_iterations_FITC: number of FITC iterations
    
    This separation allows:
    - Different optimal parameters for different fluorophores
    - Independent calibration of each channel
    - Channel-specific adaptive parameter optimization
    
    Parameters (unpacked from args tuple):
    ---------------------------------------
    cellnumber : int
        Nucleus ID to process
    masks : ndarray
        Full segmentation mask (all nuclei labeled)
    channel_images : dict
        Dictionary of channel images {channel_name: image_array}
    valid_param_samples_TRITC : ndarray
        TRITC parameter space (Nx3 array)
    valid_param_samples_FITC : ndarray
        FITC parameter space (Nx3 array)
    total_iterations_TRITC : int
        Number of TRITC parameter combinations
    total_iterations_FITC : int
        Number of FITC parameter combinations
    well_number : str
        Well identifier
    position_number : str
        Position identifier
    water_threshold_percentile_TRITC : float
        Watershed threshold for TRITC (0-100)
    water_threshold_percentile_FITC : float
        Watershed threshold for FITC (0-100)
    watershed_min_detection_prob : float
        Minimum detection probability for watershed
    min_cv_threshold : float
        CV threshold for uniform/texture classification
    uniform_contrast_multiplier : float
        Contrast adjustment for uniform nuclei
    enable_texture_filtering : bool
        Enable texture-based filtering
    calibration_mode : bool
        Are we in calibration mode?
    tritc_tracker : object or None
        Calibration tracker for TRITC
    fitc_tracker : object or None
        Calibration tracker for FITC
    image_id : str
        Image identifier for calibration
        
    Returns:
    --------
    tuple : (foci_data_list, nuclei_data_list, watershed_data_list, calibration_data)
        foci_data_list : list of dict
            All detected foci across all channels
        nuclei_data_list : list with one dict
            Nucleus-level summary (one nucleus = one entry)
        watershed_data_list : list of dict
            Watershed labels for each channel
        calibration_data : list
            Calibration results if in calibration mode
    """
    # ----------------------------------------------------------------
    # Unpack arguments from tuple
    # ----------------------------------------------------------------
    # This unpacking pattern is required for multiprocessing compatibility
    (cellnumber, masks, channel_images, 
     valid_param_samples_TRITC, valid_param_samples_FITC,
     total_iterations_TRITC, total_iterations_FITC,
     well_number, position_number, 
     water_threshold_percentile_TRITC, water_threshold_percentile_FITC,
     watershed_min_detection_prob,
     min_cv_threshold,
     uniform_contrast_multiplier,
     enable_texture_filtering,
     calibration_mode,
     tritc_tracker,
     fitc_tracker,
     image_id
    ) = args
    
    # ----------------------------------------------------------------
    # Create mask for current nucleus
    # ----------------------------------------------------------------
    # Extract just this nucleus from the full segmentation mask
    masks_reduced = (masks == cellnumber)
    
    # Early exit: if mask is empty (shouldn't happen but safety check)
    if not np.any(masks_reduced):
        # Return empty results (4-tuple format for consistency)
        return [], [], [], []
    
    # ----------------------------------------------------------------
    # Initialize result containers
    # ----------------------------------------------------------------
    foci_data_list = []  # Will accumulate foci from all channels
    
    # Initialize nucleus-level data with identifiers
    nucleus_data = {
        'cell_num': cellnumber,
        'Well': well_number,
        'Position': position_number
    }
    
    watershed_data_list = []  # Will store watershed labels for visualization
    
    # ----------------------------------------------------------------
    # Extract DAPI properties (nucleus shape/size)
    # ----------------------------------------------------------------
    # DAPI channel defines nucleus boundaries via segmentation
    # Measure geometric properties that characterize nucleus shape and size
    # These are independent of fluorescence intensity (purely geometric)
    if 'DAPI' in channel_images:
        # Get region properties from binary mask
        # regionprops extracts geometric and intensity properties of labeled regions
        # Input must be integer labels, not boolean mask
        nucleus_props = measure.regionprops(masks_reduced.astype(int))
        
        if len(nucleus_props) > 0:
            # Extract first region (should only be one since we extracted single nucleus)
            # If somehow multiple regions, take the first
            region = nucleus_props[0]
            
            # --------------------------------------------------------
            # Measure geometric properties
            # --------------------------------------------------------
            nucleus_area = region.area  # Area in pixels (size of nucleus)
            nucleus_perimeter = region.perimeter  # Perimeter in pixels (boundary length)
            
            # Calculate circularity (shape measure)
            # 1.0 = perfect circle, lower values = more irregular
            # Useful for identifying abnormal nucleus shapes (mitotic, apoptotic, artifacts)
            nucleus_circularity = compute_circularity(nucleus_area, nucleus_perimeter)
            
            # Add geometric properties to nucleus data dictionary
            # These become columns in the nuclei CSV output
            nucleus_data.update({
                'DAPI_area': nucleus_area,
                'DAPI_perimeter': nucleus_perimeter,
                'DAPI_circularity': nucleus_circularity,
                'centr_y': region.centroid[0],  # Nucleus centroid Y coordinate
                'centr_x': region.centroid[1],  # Nucleus centroid X coordinate
            })
    
    # ================================================================
    # PROCESS EACH CHANNEL
    # ================================================================
    # Loop through all channels in the channel_images dictionary
    # Different channels get different processing:
    # - All channels: intensity measurements
    # - TRITC/FITC: foci detection
    # - Others (Cy5, DAPI): intensity only
    for channel_name, channel_image in channel_images.items():
        # --------------------------------------------------------
        # Convert to float for consistent processing
        # --------------------------------------------------------
        # Ensures [0,1] range for all calculations
        # img_as_float is idempotent: if already float, returns unchanged
        # This line may be redundant if images already converted in main,
        # but ensures consistency regardless of caller behavior
        channel_image_float = img_as_float(channel_image)
        
        # --------------------------------------------------------
        # Calculate whole-nucleus intensity for all channels
        # --------------------------------------------------------
        # Measure total and mean intensity across entire nucleus
        # This includes both background and any foci present
        # Provides context for foci measurements and overall expression levels
        intensity_data = analyze_channel_intensity(masks_reduced, channel_image_float, channel_name)
        
        # Add intensity measurements to nucleus data
        # Keys are like "TRITC_total_intensity", "TRITC_mean_intensity"
        nucleus_data.update(intensity_data)
        
        # --------------------------------------------------------
        # Detect foci with CHANNEL-SPECIFIC parameters
        # --------------------------------------------------------
        # Only detect foci in specified channels (TRITC and FITC)
        # Other channels (Cy5, DAPI) only get intensity measurements
        # 
        # Key feature: Each channel uses its own optimized parameters
        # TRITC and FITC have different fluorophore properties, requiring
        # different detection thresholds for optimal results
        
        if channel_name == "TRITC":
            # ------------------------------------------------
            # TRITC FOCI DETECTION
            # ------------------------------------------------
            # Detect TRITC foci using TRITC-specific parameter space
            # During calibration: tests 256 parameters
            # During production: uses 1-3 optimized parameters
            foci_list, foci_summary, water_labels = detect_foci_single_channel(
                masks_reduced,                    # Boolean mask for this nucleus
                channel_image_float,              # TRITC image (float, 0-1)
                channel_image_float,              # Use same image for global percentiles
                channel_name,                     # "TRITC"
                cellnumber,                       # Nucleus ID
                valid_param_samples_TRITC,        # ← TRITC-specific parameters (Nx3 array)
                total_iterations_TRITC,           # ← TRITC iteration count (N)
                water_threshold_percentile_TRITC, # ← TRITC watershed threshold
                watershed_min_detection_prob=watershed_min_detection_prob,
                well_number=well_number,
                position_number=position_number,
                calibration_mode=calibration_mode,      # Are we calibrating?
                calibration_tracker=tritc_tracker,      # Tracker for calibration data
                image_id=image_id,                      # Image identifier
                min_cv_threshold=min_cv_threshold,                      # cv threshold to decide if a nucleus is uniform
                uniform_contrast_multiplier=uniform_contrast_multiplier  # adiddtional contrast threshold modifier for uniform nuclei
            )

            # ------------------------------------------------
            # Add metadata to each detected focus
            # ------------------------------------------------
            # Add well and position to each focus dictionary
            # This allows tracking which image each focus came from
            # Essential for organizing results across multi-well plates
            for focus in foci_list:
                focus['Well'] = well_number
                focus['Position'] = position_number
            
            # ------------------------------------------------
            # Accumulate results
            # ------------------------------------------------
            # Add TRITC foci to global foci list
            # This list accumulates foci from all channels
            foci_data_list.extend(foci_list)
            
            # Add TRITC summary statistics to nucleus data
            # Keys are like "TRITC_mean_foci", "TRITC_nucleus_cv", etc.
            nucleus_data.update(foci_summary)
            
            # ------------------------------------------------
            # Store watershed labels if valid (for visualization)
            # ------------------------------------------------
            # Watershed labels define the spatial extent of each focus
            # Stored separately for later visualization generation
            if water_labels is not None:
                watershed_data_list.append({
                    'cell_id': cellnumber,      # Which nucleus
                    'channel': channel_name,    # Which channel ("TRITC")
                    'labels': water_labels,     # Full-size array with watershed labels
                    'mask': masks_reduced       # Nucleus mask for proper placement in global image
                })

        if channel_name in ["FITC"]:
            # ------------------------------------------------
            # FITC FOCI DETECTION
            # ------------------------------------------------
            # Detect FITC foci using FITC-specific parameter space
            # Same logic as TRITC but with FITC parameters
            # Separate parameter space accounts for different fluorophore properties:
            # - FITC may have different signal-to-noise ratio
            # - Optimal brightness/contrast thresholds may differ
            # - Background characteristics may differ
            foci_list, foci_summary, water_labels = detect_foci_single_channel(
                masks_reduced,
                channel_image_float,
                channel_image_float,
                channel_name,                     # "FITC"
                cellnumber,
                valid_param_samples_FITC,         # ← FITC-specific parameters
                total_iterations_FITC,            # ← FITC iteration count
                water_threshold_percentile_FITC,  # ← FITC watershed threshold
                watershed_min_detection_prob=watershed_min_detection_prob, 
                well_number=well_number,
                position_number=position_number,
                calibration_mode=calibration_mode,      # Are we calibrating?
                calibration_tracker=fitc_tracker,       # Tracker for calibration data
                image_id=image_id,                      # Image identifier
                min_cv_threshold=min_cv_threshold,                      # cv threshold to decide if a nucleus is uniform
                uniform_contrast_multiplier=uniform_contrast_multiplier  # adiddtional contrast threshold modifier for uniform nuclei
            )
                    
            # ------------------------------------------------
            # Add metadata to each detected focus
            # ------------------------------------------------
            # Same as TRITC: add well and position tracking
            for focus in foci_list:
                focus['Well'] = well_number
                focus['Position'] = position_number
            
            # ------------------------------------------------
            # Accumulate results
            # ------------------------------------------------
            # Add FITC foci to global list
            foci_data_list.extend(foci_list)
            
            # Add FITC summary statistics to nucleus data
            nucleus_data.update(foci_summary)
            
            # ------------------------------------------------
            # Store watershed labels if valid
            # ------------------------------------------------
            if water_labels is not None:
                watershed_data_list.append({
                    'cell_id': cellnumber,
                    'channel': channel_name,    # "FITC"
                    'labels': water_labels,
                    'mask': masks_reduced
                })
    
    # ----------------------------------------------------------------
    # Package nucleus data for return
    # ----------------------------------------------------------------
    # Wrap single nucleus dictionary in list for consistency
    # Main process expects list format (even though only one nucleus per worker)
    # This allows consistent handling: main process can always extend() the list
    nuclei_data_list = [nucleus_data]
    
    # ----------------------------------------------------------------
    # Collect calibration data if in calibration mode
    # ----------------------------------------------------------------
    # During calibration, trackers accumulate results from each nucleus
    # Each parameter combination tested gets recorded
    # We need to extract and return these results to the main process
    # Main process will aggregate across all nuclei to find optimal parameters
    calibration_data = []
    if calibration_mode:
        # ------------------------------------------------
        # Extract calibration results from TRITC tracker
        # ------------------------------------------------
        # Each tracker has a calibration_results list populated during detection
        # Each entry records: (image_id, cell_id, param_combo, foci_count, channel)
        if tritc_tracker is not None and len(tritc_tracker.calibration_results) > 0:
            # Add all TRITC calibration results to return list
            calibration_data.extend(tritc_tracker.calibration_results)
        
        # ------------------------------------------------
        # Extract calibration results from FITC tracker
        # ------------------------------------------------
        # Same for FITC channel
        if fitc_tracker is not None and len(fitc_tracker.calibration_results) > 0:
            # Add all FITC calibration results to return list
            calibration_data.extend(fitc_tracker.calibration_results)
    
    # ----------------------------------------------------------------
    # Return all results as 4-tuple
    # ----------------------------------------------------------------
    # Return format: (foci_list, nuclei_list, watershed_list, calibration_list)
    # 
    # Main process will:
    # 1. Extend all_foci_data with foci_list
    # 2. Extend all_nuclei_data with nuclei_list
    # 3. Process watershed_list to merge labels into global arrays
    # 4. Aggregate calibration_list for parameter optimization
    #
    # All four components must be present even if empty (for consistent unpacking)
    return foci_data_list, nuclei_data_list, watershed_data_list, calibration_data