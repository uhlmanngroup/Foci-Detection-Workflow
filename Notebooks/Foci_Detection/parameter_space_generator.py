"""
Complete Parameter Space Generator with Full Foci Detection

This module provides tools for finding optimal parameter ranges for foci detection
through an interactive, ground-truth-based calibration process:

1. INTERACTIVE SELECTION: User manually counts foci in representative nuclei
2. GRID SEARCH: Test parameter space systematically on ground truth nuclei
3. INTERSECTION: Find parameters that work for ALL ground truth nuclei
4. KDE MODELING: Generate smooth parameter space boundary using kernel density estimation
5. DELAUNAY HULL: Create convex hull for efficient parameter sampling

The result is a validated parameter space that can be used to generate reliable
parameter combinations for the main detection pipeline.

This approach ensures parameters are individual and specialized for each data set.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# Import detection functions from main pipeline
# This ensures parameter space generator uses IDENTICAL detection logic
from nucleus_worker_Visualization import (
    compute_adaptive_background_texture_nucleus_fallback,
    apply_foci_filters
)


class ParameterSpaceGenerator:
    """
    Interactive parameter space calibration system for foci detection.
    
    This class implements a complete workflow for determining valid parameter ranges:
    
    WORKFLOW:
    ---------
    1. User selects representative nuclei and counts foci (ground truth)
    2. System tests parameter grid on these nuclei (grid search)
    3. Finds parameters that correctly detect foci in ALL ground truth nuclei
    4. Builds statistical model (KDE) of valid parameter space
    5. Creates Delaunay triangulation for efficient parameter sampling
    
    WHY THIS APPROACH:
    ------------------
    - Grounded in real data, not arbitrary parameter choices
    - Validates parameters against known ground truth
    - Finds robust parameters that work across diverse nuclei
    - Provides statistical model for parameter generation
    
    USAGE EXAMPLE:
    --------------
```python
    # Initialize with parameter ranges to explore
    generator = ParameterSpaceGenerator(
        param_ranges={
            'bright_pct': (0, 100),      # Background percentile
            'contrast_thresh': (1, 10),   # Local contrast threshold
            'percentile_val': (0, 100)    # Global brightness percentile
        }
    )
    
    # Step 1: Interactive ground truth collection
    nucleus_ids = generator.interactive_nucleus_selection(
        masks=segmentation_masks,
        channel_image=tritc_image,
        num_nuclei=10
    )
    
    # Step 2: Grid search on ground truth
    results = generator.generate_grid_search(
        masks=segmentation_masks,
        channel_image=tritc_image,
        original_image=original_tritc
    )
    
    # Step 3: Find valid parameter intersection
    valid_params = generator.find_valid_intersection()
    
    # Step 4: Build KDE model and Delaunay hull
    hull = generator.generate_kde_parameter_space(coverage_percentile=85)
    
    # Step 5: Save for use in main pipeline
    generator.save_complete('output_folder/')
```
    
    Attributes:
    -----------
    param_ranges : dict
        Parameter ranges to explore: {param_name: (min, max)}
    resolution : int
        Grid resolution for parameter space (not currently used)
    ground_truth_nuclei : dict
        {nucleus_id: (min_count, max_count)} - manually annotated foci counts
    grid_results : pd.DataFrame
        Results from testing all parameters on all ground truth nuclei
    valid_points : np.ndarray
        Parameter combinations that work for ALL ground truth nuclei
    kde_model : KernelDensity
        Statistical model of valid parameter space
    hull : scipy.spatial.Delaunay
        Delaunay triangulation of valid parameter space
    bounds : dict
        Bounding box of valid parameters: {param_name: (min, max)}
    kde_metadata : dict
        KDE configuration (bandwidth, threshold, isosurface points)
    """
    
    def __init__(self, param_ranges, resolution=20):
        """
        Initialize the parameter space generator.
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary defining parameter ranges to explore:
            {
                'bright_pct': (min, max),      # Background brightness percentile
                'contrast_thresh': (min, max),  # Local contrast threshold
                'percentile_val': (min, max)    # Global brightness percentile
            }
            Example: {
                'bright_pct': (0, 100),
                'contrast_thresh': (1, 10),
                'percentile_val': (0, 100)
            }
        resolution : int, default=20
            Grid resolution for parameter space
            Currently not used in implementation (grid is hard-coded to 31×15×31)
            Kept for future flexibility
        """
        # Store parameter configuration
        self.param_ranges = param_ranges
        self.resolution = resolution
        
        # ----------------------------------------------------------------
        # Initialize data storage
        # ----------------------------------------------------------------
        # Ground truth: nucleus_id → (min_foci_count, max_foci_count)
        # User manually counts foci and provides acceptable range
        self.ground_truth_nuclei = {}
        
        # Grid search results: DataFrame with columns
        # [cell_num, bright_pct, contrast_thresh, percentile_val, foci_count]
        self.grid_results = None
        
        # Valid parameters: (N, 3) array of parameters that work for ALL nuclei
        # Each row: [bright_pct, contrast_thresh, percentile_val]
        self.valid_points = None
        
        # KDE model: Sklearn KernelDensity estimator
        # Models probability density of valid parameters
        self.kde_model = None
        
        # Delaunay hull: scipy.spatial.Delaunay triangulation
        # Defines convex hull of valid parameter space
        # Used for efficient random sampling within valid region
        self.hull = None
        
        # Bounds: Bounding box of valid parameters
        # {param_name: (min_val, max_val)}
        self.bounds = None
        
        # KDE metadata: Configuration and derived data
        # {bandwidth, threshold, coverage_percentile, isosurface_points}
        self.kde_metadata = None
        
    def add_nucleus(self, cell_id, min_count, max_count):
        """
        Register a nucleus with its ground truth foci count range.
        
        This function is called during interactive selection to record
        the user's manual foci count for each nucleus. The count can be:
        - Exact: min_count = max_count (user is certain)
        - Range: min_count < max_count (user allows some uncertainty)
        
        The range approach is useful when:
        - Foci are ambiguous or overlapping
        - User is not 100% certain of exact count
        - Want to allow slight parameter variation
        
        Parameters:
        -----------
        cell_id : int
            Unique nucleus identifier (from segmentation mask)
        min_count : int
            Minimum acceptable foci count (inclusive)
            Parameters that detect fewer than this are rejected
        max_count : int
            Maximum acceptable foci count (inclusive)
            Parameters that detect more than this are rejected
            
        Raises:
        -------
        ValueError
            If max_count < min_count (invalid range)
            
        Examples:
        ---------
        >>> generator.add_nucleus(5, 3, 3)   # Exactly 3 foci
        ✓ Added nucleus 5 with exact count: 3
        
        >>> generator.add_nucleus(12, 4, 6)  # Between 4-6 foci
        ✓ Added nucleus 12 with acceptable range: 4-6 foci
        """
        # ----------------------------------------------------------------
        # Validate input
        # ----------------------------------------------------------------
        if max_count < min_count:
            raise ValueError(f"max_count ({max_count}) must be >= min_count ({min_count})")
        
        # ----------------------------------------------------------------
        # Store ground truth
        # ----------------------------------------------------------------
        # Format: {nucleus_id: (min, max)}
        # This will be used later to evaluate parameter performance
        self.ground_truth_nuclei[cell_id] = (min_count, max_count)
        
        # ----------------------------------------------------------------
        # User feedback
        # ----------------------------------------------------------------
        if min_count == max_count:
            # Exact count (certainty)
            print(f"✓ Added nucleus {cell_id} with exact count: {min_count}")
        else:
            # Range (uncertainty)
            print(f"✓ Added nucleus {cell_id} with acceptable range: {min_count}-{max_count} foci")
    
    def interactive_nucleus_selection(self, masks, channel_image, num_nuclei=10, 
                                      mask_data_paths=None, channel_data_paths=None):
        """
        Interactive workflow for selecting and annotating ground truth nuclei.
        
        This is the MAIN INTERFACE for ground truth collection. The workflow:
        
        1. Shows overview of all nuclei in current image
        2. User selects a nucleus by entering its ID
        3. Shows detailed zoom view of that nucleus (3 panels)
        4. User counts foci and enters count or range
        5. Repeats until target number reached
        
        The user can:
        - Navigate between images ('next', 'prev', 'picture')
        - Regenerate overview ('generate')
        - Abort current nucleus without saving ('abort', 'abort+')
        - Finish early with fewer nuclei ('done')
        
        ADVANCED FEATURE: Picture Navigation
        -------------------------------------
        If mask_data_paths and channel_data_paths are provided, user can switch
        between images during selection. This is useful when:
        - Some images have better examples than others
        - Want to ensure diversity across multiple images
        - Initial image doesn't have enough good nuclei
        
        Parameters:
        -----------
        masks : numpy.ndarray or will be loaded
            Segmentation mask (initially loaded, updated when changing pictures)
            Shape: (H, W) with integer labels for each nucleus
            Label 0 = background, 1+ = nuclei
        channel_image : numpy.ndarray or will be loaded
            Fluorescence image (initially loaded, updated when changing pictures)
            Shape: (H, W) with intensity values
        num_nuclei : int, default=10
            Target number of nuclei to annotate
            User can finish early with 'done' command
            Recommended: 5-15 nuclei for good coverage
        mask_data_paths : list of str, optional
            List of ALL segmentation mask file paths
            Enables picture navigation feature
            Format: [..., 'path/to/mask_W0001_P0001.npy', ...]
        channel_data_paths : list of str, optional
            List of ALL channel image file paths
            Must match order of mask_data_paths
            Format: [..., 'path/to/image_W0001_P0001.tif', ...]
            
        Returns:
        --------
        list of int
            List of selected nucleus IDs
            These IDs are keys in self.ground_truth_nuclei
            
        User Commands:
        --------------
        During nucleus ID prompt:
        - Integer (e.g., '5'): Select nucleus 5 for annotation
        - 'generate': Regenerate overview plot
        - 'done': Finish annotation (must have ≥1 nucleus)
        - 'next' or 'n': Next picture (if navigation enabled)
        - 'prev' or 'p': Previous picture (if navigation enabled)
        - 'picture' or 'pic': Choose specific picture (if navigation enabled)
        
        During foci count prompt:
        - Integer (e.g., '3'): Exactly 3 foci
        - Range (e.g., '2-4'): Between 2-4 foci
        - 'abort': Cancel this nucleus, go back to selection
        - 'abort+': Cancel this nucleus and regenerate overview
        
        Picture selection commands (when using 'picture'):
        - Integer (e.g., '5'): Go to picture index 5
        - Well+position (e.g., 'w50p18'): Go to Well 50, Position 18
        - 'list': Show all available pictures
        - 'cancel': Return to nucleus selection
        
        Example Session:
        ----------------
        >>> generator.interactive_nucleus_selection(masks, image, num_nuclei=5)
        
        INTERACTIVE VISUAL NUCLEUS SELECTION
        Goal: Select and annotate up to 5 nuclei
        
        [Overview plot appears]
        
        [0/5 nuclei selected]
        Enter nucleus ID to examine (or 'generate' to show overview, 'done' to finish): 12
        
        [Detailed view of nucleus 12 appears]
        
        Nucleus 12 - Enter foci count (e.g., '3' or '2-4' for range), 'abort', or 'abort+': 4
        ✓ Added nucleus 12 with exact count: 4
        ✓ Saved! (1/5 complete)
        
        [1/5 nuclei selected]
        Enter nucleus ID: 23
        ...
        
        Notes:
        ------
        - Plots are shown non-blocking (plt.show(block=False))
        - All plots are closed (plt.close('all')) when changing context
        - Image is converted to float internally (img_as_float)
        - Selected nuclei are tracked to prevent re-selection
        """
        # ================================================================
        # IMPORT DEPENDENCIES
        # ================================================================
        # These are imported locally to avoid requiring them at module level
        from skimage.segmentation import find_boundaries
        import os
        import re
        import imageio
        
        # ================================================================
        # PICTURE NAVIGATION SETUP
        # ================================================================
        # Feature is enabled only if both path lists are provided
        picture_navigation_enabled = (mask_data_paths is not None and 
                                       channel_data_paths is not None)
        current_picture_idx = 0  # Track which picture we're viewing
        
        # ----------------------------------------------------------------
        # Load initial picture if navigation is enabled
        # ----------------------------------------------------------------
        if picture_navigation_enabled:
            # Load from file paths instead of using provided arrays(not implemented)
            masks = np.load(mask_data_paths[current_picture_idx], allow_pickle=True)
            channel_image = imageio.imread(channel_data_paths[current_picture_idx])
        
        # ================================================================
        # HELPER FUNCTIONS
        # ================================================================
        
        def extract_well_position(filepath):
            """
            Extract well and position identifiers from filename.
            
            Assumes filenames contain patterns like:
            - --W00050-- (well number, padded to 5 digits)
            - --P00018-- (position number, padded to 5 digits)
            
            Parameters:
            -----------
            filepath : str or None
                Full path to file
                
            Returns:
            --------
            tuple of (str, str)
                (well_number, position_number)
                Returns ("?", "?") if filepath is None or pattern not found
            """
            if filepath is None:
                return "?", "?"
            
            # Extract just the filename (no directory path)
            basename = os.path.basename(filepath)
            
            # Search for well pattern: --W followed by digits
            well_match = re.search(r'--W(\d+)', basename)
            # Search for position pattern: --P followed by digits
            pos_match = re.search(r'--P(\d+)', basename)
            
            # Extract matched groups or use "?" if not found
            well = well_match.group(1) if well_match else "?"
            pos = pos_match.group(1) if pos_match else "?"
            
            return well, pos
        
        def change_picture(direction):
            """
            Navigate to next or previous picture in the sequence.
            
            Parameters:
            -----------
            direction : str
                'next': Move forward one picture
                'prev': Move backward one picture
                
            Returns:
            --------
            bool
                True if picture changed successfully
                False if already at boundary or navigation disabled
                
            Side Effects:
            -------------
            - Updates nonlocal variables: current_picture_idx, masks, channel_image
            - Loads new data from file paths
            - Prints confirmation message with well/position info
            """
            nonlocal current_picture_idx, masks, channel_image
            
            # --------------------------------------------------------
            # Check if navigation is available
            # --------------------------------------------------------
            if not picture_navigation_enabled:
                print("❌ Picture navigation not available (no paths provided)")
                return False
            
            # --------------------------------------------------------
            # Update index based on direction
            # --------------------------------------------------------
            if direction == 'next':
                # Check if we can go forward
                if current_picture_idx < len(mask_data_paths) - 1:
                    current_picture_idx += 1
                else:
                    print("❌ Already at last picture!")
                    return False
                    
            elif direction == 'prev':
                # Check if we can go backward
                if current_picture_idx > 0:
                    current_picture_idx -= 1
                else:
                    print("❌ Already at first picture!")
                    return False
            
            # --------------------------------------------------------
            # Load new picture data
            # --------------------------------------------------------
            masks = np.load(mask_data_paths[current_picture_idx], allow_pickle=True)
            channel_image = imageio.imread(channel_data_paths[current_picture_idx])
            
            # --------------------------------------------------------
            # User feedback
            # --------------------------------------------------------
            well, pos = extract_well_position(mask_data_paths[current_picture_idx])
            print(f"→ Switched to picture {current_picture_idx + 1}/{len(mask_data_paths)}: "
                  f"Well {well}, Position {pos}")
            return True
        
        def choose_specific_picture():
            """
            Interactive menu for choosing a specific picture.
            
            User can specify picture by:
            - Index number (0, 1, 2, ...)
            - Well+position shorthand (e.g., 'w50p18' → Well 00050, Position 00018)
            
            Also provides:
            - 'list': Show all available pictures with indices
            - 'cancel': Return to nucleus selection without changing picture
            
            Side Effects:
            -------------
            - Updates nonlocal variables: current_picture_idx, masks, channel_image
            - Loads new data if picture is changed
            - Prints confirmation or error messages
            """
            nonlocal current_picture_idx, masks, channel_image
            
            # --------------------------------------------------------
            # Check if navigation is available
            # --------------------------------------------------------
            if not picture_navigation_enabled:
                print("❌ Picture navigation not available")
                return
            
            # --------------------------------------------------------
            # Show menu
            # --------------------------------------------------------
            print("\n" + "="*60)
            print("🔍 CHOOSE PICTURE")
            print("="*60)
            print("Options:")
            print("  1. Enter picture index (e.g., '5')")
            print("  2. Enter well+position (e.g., 'w50p18' or 'W0050P0018')")
            print("  3. Type 'list' to see all pictures")
            print("  4. Type 'cancel' to go back")
            print("="*60)
            
            # --------------------------------------------------------
            # Input loop
            # --------------------------------------------------------
            while True:
                choice = input("\n➤ ").strip()
                
                # ----------------------------------------------------
                # Handle cancel
                # ----------------------------------------------------
                if choice.lower() == 'cancel':
                    print("Cancelled.")
                    return
                
                # ----------------------------------------------------
                # Handle list command
                # ----------------------------------------------------
                if choice.lower() == 'list':
                    print("\n📋 Available pictures:")
                    for idx, path in enumerate(mask_data_paths):
                        well, pos = extract_well_position(path)
                        # Mark current picture
                        marker = "← CURRENT" if idx == current_picture_idx else ""
                        print(f"  [{idx:3d}] Well {well}, Position {pos} {marker}")
                    continue  # Stay in loop, ask for input again
                
                # ----------------------------------------------------
                # Try to parse as well+position format
                # ----------------------------------------------------
                # Pattern: w50p18 (case-insensitive)
                well_pos_match = re.match(r'w(\d+)p(\d+)', choice.lower())
                if well_pos_match:
                    # Extract well and position numbers
                    target_well = well_pos_match.group(1).zfill(5)  # Pad to 5 digits
                    target_pos = well_pos_match.group(2).zfill(5)   # Pad to 5 digits
                    
                    # Search for matching picture in file list
                    for idx, path in enumerate(mask_data_paths):
                        well, pos = extract_well_position(path)
                        if well == target_well and pos == target_pos:
                            # Found match - switch to this picture
                            current_picture_idx = idx
                            masks = np.load(mask_data_paths[current_picture_idx], allow_pickle=True)
                            channel_image = imageio.imread(channel_data_paths[current_picture_idx])
                            print(f"✅ Switched to Well {well}, Position {pos}")
                            return  # Exit function, return to nucleus selection
                    
                    # Not found in any file
                    print(f"❌ No picture found for Well {target_well}, Position {target_pos}")
                    continue  # Stay in loop
                
                # ----------------------------------------------------
                # Try to parse as picture index
                # ----------------------------------------------------
                if choice.isdigit():
                    idx = int(choice)
                    
                    # Validate index range
                    if 0 <= idx < len(mask_data_paths):
                        # Valid index - switch to this picture
                        current_picture_idx = idx
                        masks = np.load(mask_data_paths[current_picture_idx], allow_pickle=True)
                        channel_image = imageio.imread(channel_data_paths[current_picture_idx])
                        well, pos = extract_well_position(mask_data_paths[current_picture_idx])
                        print(f"✅ Switched to picture #{idx}: Well {well}, Position {pos}")
                        return  # Exit function
                    else:
                        # Index out of range
                        print(f"❌ Invalid index (must be 0-{len(mask_data_paths)-1})")
                        continue  # Stay in loop
                else:
                    # Not a digit, not recognized format
                    print("❌ Invalid format. Use: index (e.g., '5') or 'w50p18'")
                    continue  # Stay in loop
        
        # ================================================================
        # MAIN INTERACTIVE WORKFLOW
        # ================================================================
        
        # ----------------------------------------------------------------
        # Print welcome banner and instructions
        # ----------------------------------------------------------------
        print("\n" + "="*60)
        print("INTERACTIVE VISUAL NUCLEUS SELECTION")
        print("="*60)
        print(f"\nGoal: Select and annotate up to {num_nuclei} nuclei")
        
        # Show current picture info if navigation is enabled
        if picture_navigation_enabled:
            well, pos = extract_well_position(mask_data_paths[current_picture_idx])
            print(f"Current picture: {current_picture_idx + 1}/{len(mask_data_paths)} - "
                  f"Well {well}, Position {pos}")
        
        print("\nWorkflow:")
        print("  1. View full overview with all nucleus IDs")
        print("  2. Select a nucleus by entering its ID")
        print("  3. View detailed zoom of that nucleus")
        print("  4. Enter foci count or choose different nucleus")
        
        # Show navigation commands if enabled
        if picture_navigation_enabled:
            print("\n📸 Picture Navigation:")
            print("  • 'next' or 'n' - Next picture")
            print("  • 'prev' or 'p' - Previous picture")
            print("  • 'picture' or 'pic' - Choose specific picture")
        print("="*60)
        
        # ----------------------------------------------------------------
        # Prepare data
        # ----------------------------------------------------------------
        # Convert image to float (required for processing)
        channel_image = img_as_float(channel_image)
        
        # Get all valid nucleus IDs from mask
        # Label 0 is background, so skip it
        all_nucleus_ids = np.unique(masks)[1:]
        
        # ----------------------------------------------------------------
        # Initialize selection tracking
        # ----------------------------------------------------------------
        selected_count = 0  # How many nuclei have been annotated
        used_ids = set()    # Set of nucleus IDs already selected
        
        # ----------------------------------------------------------------
        # STEP 1: Show initial overview
        # ----------------------------------------------------------------
        print("\n📊 Showing full overview of all nuclei...")
        self._show_all_nuclei_overview(masks, channel_image, used_ids)
        
        # ================================================================
        # MAIN SELECTION LOOP
        # ================================================================
        while selected_count < num_nuclei:
            print(f"\n[{selected_count}/{num_nuclei} nuclei selected]")
            
            # ============================================================
            # STEP 2: Get nucleus ID from user
            # ============================================================
            while True:  # Loop until valid ID or command
                # Show current picture info if navigation enabled
                if picture_navigation_enabled:
                    well, pos = extract_well_position(mask_data_paths[current_picture_idx])
                    print(f"📸 Current: Picture {current_picture_idx + 1}/{len(mask_data_paths)}, "
                          f"Well {well}, Pos {pos}")
                
                # Build prompt with appropriate options
                prompt = "\n🔍 Enter nucleus ID to examine (or 'generate' to show overview, 'done' to finish"
                if picture_navigation_enabled:
                    prompt += ", 'next'/'prev'/'picture' to change picture"
                prompt += "): "
                
                nucleus_input = input(prompt).strip()
                
                # --------------------------------------------------------
                # Handle 'done' command
                # --------------------------------------------------------
                if nucleus_input.lower() == 'done':
                    if selected_count > 0:
                        # Have at least one nucleus - can finish
                        print(f"\n✓ Finished with {selected_count} nuclei")
                        plt.close('all')
                        return list(self.ground_truth_nuclei.keys())
                    else:
                        # Must select at least one nucleus
                        print("❌ Please select at least one nucleus first!")
                        continue
                
                # --------------------------------------------------------
                # Handle 'generate' command (regenerate overview)
                # --------------------------------------------------------
                elif nucleus_input.lower() == 'generate':
                    print("\n📊 Regenerating full overview...")
                    plt.close('all')
                    # Update nucleus IDs for current picture
                    all_nucleus_ids = np.unique(masks)[1:]
                    self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    continue  # Back to top of input loop
                
                # --------------------------------------------------------
                # Handle 'next' command (next picture)
                # --------------------------------------------------------
                elif nucleus_input.lower() in ['next', 'n']:
                    if change_picture('next'):
                        # Successfully changed picture - update display
                        plt.close('all')
                        all_nucleus_ids = np.unique(masks)[1:]
                        channel_image = img_as_float(channel_image)
                        self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    continue  # Back to top of input loop
                
                # --------------------------------------------------------
                # Handle 'prev' command (previous picture)
                # --------------------------------------------------------
                elif nucleus_input.lower() in ['prev', 'p']:
                    if change_picture('prev'):
                        # Successfully changed picture - update display
                        plt.close('all')
                        all_nucleus_ids = np.unique(masks)[1:]
                        channel_image = img_as_float(channel_image)
                        self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    continue  # Back to top of input loop
                
                # --------------------------------------------------------
                # Handle 'picture' command (choose specific picture)
                # --------------------------------------------------------
                elif nucleus_input.lower() in ['picture', 'pic']:
                    choose_specific_picture()  # Interactive submenu
                    # May or may not have changed picture
                    plt.close('all')
                    all_nucleus_ids = np.unique(masks)[1:]
                    channel_image = img_as_float(channel_image)
                    self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    continue  # Back to top of input loop
                
                # --------------------------------------------------------
                # Try to parse as nucleus ID (integer)
                # --------------------------------------------------------
                else:
                    try:
                        nucleus_id = int(nucleus_input)
                        
                        # Validate that this nucleus exists
                        if nucleus_id not in all_nucleus_ids:
                            print(f"❌ Nucleus {nucleus_id} not found. "
                                  f"Try again or enter 'generate' to see overview.")
                            continue  # Back to top of input loop
                        
                        # Valid nucleus ID - break to show detailed view
                        break
                        
                    except ValueError:
                        # Not a valid integer
                        print("❌ Invalid input. Enter a number, 'generate', or 'done'.")
                        continue  # Back to top of input loop
            
            # ============================================================
            # STEP 3: Show detailed view of selected nucleus
            # ============================================================
            plt.close('all')  # Close overview
            self._visualize_single_nucleus(masks, channel_image, nucleus_id)
            plt.show(block=False)  # Non-blocking so user can interact
            plt.pause(0.1)  # Small pause to ensure plot appears
            
            # ============================================================
            # STEP 4: Get foci count from user
            # ============================================================
            while True:  # Loop until valid count or abort
                action_input = input(
                    f"\n🎯 Nucleus {nucleus_id} - Enter foci count "
                    f"(e.g., '3' or '2-4' for range), 'abort', or 'abort+': "
                ).strip()
                
                # --------------------------------------------------------
                # Handle 'abort' command (cancel without regenerating)
                # --------------------------------------------------------
                if action_input.lower() == 'abort':
                    print("→ Aborting to nucleus selection...")
                    plt.close('all')
                    break  # Break inner loop (count input), stay in outer loop (ID selection)
                
                # --------------------------------------------------------
                # Handle 'abort+' command (cancel and regenerate overview)
                # --------------------------------------------------------
                elif action_input.lower() == 'abort+':
                    print("→ Aborting and regenerating overview...")
                    plt.close('all')
                    self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    break  # Break inner loop, stay in outer loop
                
                # --------------------------------------------------------
                # Try to parse as foci count
                # --------------------------------------------------------
                else:
                    try:
                        # ----------------------------------------------------
                        # Parse range format "2-4" or single value "3"
                        # ----------------------------------------------------
                        if '-' in action_input:
                            # Range format: "min-max"
                            parts = action_input.split('-')
                            
                            # Validate format (must be exactly 2 parts)
                            if len(parts) != 2:
                                print("❌ Invalid range format. Use 'min-max' (e.g., '2-4')")
                                continue
                            
                            # Parse min and max
                            min_count = int(parts[0].strip())
                            max_count = int(parts[1].strip())
                            
                            # Validate non-negative
                            if min_count < 0 or max_count < 0:
                                print("❌ Counts must be non-negative")
                                continue
                            
                            # Validate range ordering
                            if max_count < min_count:
                                print("❌ Max count must be >= min count")
                                continue
                        
                        else:
                            # Single value: "3" → treat as exact count (3-3)
                            foci_count = int(action_input)
                            
                            # Validate non-negative
                            if foci_count < 0:
                                print("❌ Count must be non-negative")
                                continue
                            
                            # Set both min and max to same value (exact count)
                            min_count = foci_count
                            max_count = foci_count
                        
                        # ----------------------------------------------------
                        # Valid count/range - save it
                        # ----------------------------------------------------
                        self.add_nucleus(nucleus_id, min_count, max_count)
                        used_ids.add(nucleus_id)  # Mark as used
                        selected_count += 1  # Increment counter
                        plt.close('all')
                        
                        # Check if we've reached target
                        if selected_count >= num_nuclei:
                            print(f"\n✅ Completed! Selected {selected_count} nuclei.")
                            return list(self.ground_truth_nuclei.keys())
                        
                        # Not done yet - prompt for next nucleus
                        print(f"✓ Saved! ({selected_count}/{num_nuclei} complete)")
                        break  # Break inner loop (count input), continue outer loop (ID selection)
                        
                    except ValueError:
                        # Not a valid integer or range
                        print("❌ Invalid input. Enter a number, 'abort', or 'abort+'.")
                        continue  # Stay in count input loop
        
        # ================================================================
        # COMPLETION
        # ================================================================
        # Reached target number of nuclei
        plt.close('all')
        return list(self.ground_truth_nuclei.keys())
    
    def generate_grid_search(self, masks, channel_image, original_image):
        """
        Test parameter space systematically on ground truth nuclei.
        
        This is the CORE CALIBRATION STEP where we:
        1. Generate a grid of parameter combinations to test
        2. Run ACTUAL foci detection (identical to main pipeline) on each ground truth nucleus
        3. Record how many foci each parameter combination detects
        4. Store results for later analysis
        
        The grid is defined by:
        - bright_pct: 31 values from min to max (background brightness percentile)
        - contrast_thresh: 15 values from min to max (local contrast threshold)
        - percentile_val: 31 values from min to max (global brightness percentile)
        
        Total combinations: 31 × 15 × 31 = 14,415 tests per nucleus
        
        WHY SO MANY COMBINATIONS:
        - Need fine resolution to find optimal parameters
        - Some parameters interact (non-linear effects)
        - Want to explore full parameter space systematically
        
        Parameters:
        -----------
        masks : numpy.ndarray
            Segmentation mask, shape (H, W)
            Integer labels where each nucleus has unique ID
        channel_image : numpy.ndarray
            Fluorescence channel image, shape (H, W)
            This is the image where foci are detected
        original_image : numpy.ndarray
            Original unprocessed image, shape (H, W)
            Used for global percentile calculations
            
        Returns:
        --------
        pd.DataFrame
            Results with columns:
            - cell_num: Nucleus ID
            - bright_pct: Background percentile parameter
            - contrast_thresh: Contrast threshold parameter
            - percentile_val: Global percentile parameter
            - foci_count: Number of foci detected with this parameter combo
            - min_count: Ground truth minimum (from user annotation)
            - max_count: Ground truth maximum (from user annotation)
            
        Raises:
        -------
        ValueError
            If no ground truth nuclei registered (must call add_nucleus() first)
            
        Side Effects:
        -------------
        - Stores results in self.grid_results (DataFrame)
        - Generates and displays visualization of results (_visualize_grid_search_results)
        - Prints progress for each nucleus
        
        Example Output:
        ---------------
        STEP 2: GRID SEARCH WITH ACTUAL FOCI DETECTION
        ============================================================
        
        Testing 14415 parameter combinations
        On 5 nuclei
          Bright %: 31 values from 0.0 to 100.0
          Contrast: 15 values from 1.0 to 10.0
          Percentile: 31 values from 0.0 to 100.0
        
          Processing nucleus 5 (expected: exactly 3 foci)...
            ✓ 2847/14415 parameters gave acceptable count
        
          Processing nucleus 12 (expected: 4-6 foci)...
            ✓ 4521/14415 parameters gave acceptable count
        
        ...
        
        ✓ Grid search complete: 72075 total results
        
        Notes:
        ------
        - Uses IDENTICAL detection algorithm as main pipeline
        - Detection is texture-aware (applies contrast multiplier if CV < 0.20) --> right now disabled(additional multiplier set at 1)
        - Each parameter combo is independent (parallelizable if needed)
        - Results are visualized as 3D scatter plots (green=correct, gray=incorrect)
        """
        # ================================================================
        # BANNER AND VALIDATION
        # ================================================================
        print("\n" + "="*60)
        print("STEP 2: GRID SEARCH WITH ACTUAL FOCI DETECTION")
        print("="*60)
        
        # Ensure we have ground truth data
        if not self.ground_truth_nuclei:
            raise ValueError("No nuclei registered! Use add_nucleus() first.")
         
        # ================================================================
        # DEFINE PARAMETER GRID
        # ================================================================
        # Number of samples per parameter dimension
        # These define the resolution of our search grid
        n_bright = 31      # Background percentile samples
        n_contrast = 15    # Contrast threshold samples
        n_percentile = 31  # Global percentile samples
        
        # ----------------------------------------------------------------
        # Create 1D arrays of parameter values
        # ----------------------------------------------------------------
        # Use linspace to create evenly-spaced values across each range
        # dtype=int ensures we get clean integer values (not floats)
        bright_vals = np.linspace(
            self.param_ranges['bright_pct'][0],      # Min value
            self.param_ranges['bright_pct'][1],      # Max value
            n_bright,                                # Number of samples
            dtype=int                                # Cast to integer
        )
        
        contrast_vals = np.linspace(
            self.param_ranges['contrast_thresh'][0],
            self.param_ranges['contrast_thresh'][1],
            n_contrast,
            dtype=int
        )
        
        percentile_vals = np.linspace(
            self.param_ranges['percentile_val'][0],
            self.param_ranges['percentile_val'][1],
            n_percentile,
            dtype=int
        )
        
        # ----------------------------------------------------------------
        # Create meshgrid to generate ALL combinations
        # ----------------------------------------------------------------
        # meshgrid creates 3D grids where each point is a parameter combination
        # indexing='ij' ensures standard matrix indexing (i,j,k not i,k,j)
        bright_mesh, contrast_mesh, percentile_mesh = np.meshgrid(
            bright_vals, contrast_vals, percentile_vals, indexing='ij'
        )
        
        # ----------------------------------------------------------------
        # Flatten meshgrid to get 1D arrays
        # ----------------------------------------------------------------
        # This converts 3D grids to 1D lists of all combinations
        # Shape goes from (31, 15, 31) to (14415,) for each parameter
        bright_grid = bright_mesh.flatten()      # All bright_pct values
        contrast_grid = contrast_mesh.flatten()  # All contrast_thresh values
        percentile_grid = percentile_mesh.flatten()  # All percentile_val values
        
        # ----------------------------------------------------------------
        # Print configuration summary
        # ----------------------------------------------------------------
        total_combinations = len(bright_grid)
        print(f"\nTesting {total_combinations} parameter combinations")
        print(f"On {len(self.ground_truth_nuclei)} nuclei")
        print(f"  Bright %: {n_bright} values from {bright_vals.min():.1f} to {bright_vals.max():.1f}")
        print(f"  Contrast: {n_contrast} values from {contrast_vals.min():.2f} to {contrast_vals.max():.2f}")
        print(f"  Percentile: {n_percentile} values from {percentile_vals.min():.1f} to {percentile_vals.max():.1f}")
        
        # ================================================================
        # RUN GRID SEARCH ON EACH NUCLEUS
        # ================================================================
        results = []  # Accumulator for all results
        
        # Process each ground truth nucleus
        for cell_id, (min_count, max_count) in self.ground_truth_nuclei.items():
            # --------------------------------------------------------
            # Print progress
            # --------------------------------------------------------
            if min_count == max_count:
                print(f"\n  Processing nucleus {cell_id} (expected: exactly {min_count} foci)...")
            else:
                print(f"\n  Processing nucleus {cell_id} (expected: {min_count}-{max_count} foci)...")
            
            # --------------------------------------------------------
            # Extract nucleus mask
            # --------------------------------------------------------
            nucleus_mask = (masks == cell_id)
            
            # Validate that nucleus exists
            if not np.any(nucleus_mask):
                print(f"    ⚠️ Warning: Nucleus {cell_id} not found in mask")
                continue
            
            # --------------------------------------------------------
            # Run detection with all parameter combinations
            # --------------------------------------------------------
            # This is where the actual work happens
            nucleus_results = self._detect_foci_for_nucleus(
                nucleus_mask, channel_image, original_image,
                bright_grid, contrast_grid, percentile_grid,
                cell_id
            )
            
            # --------------------------------------------------------
            # Add ground truth metadata to results
            # --------------------------------------------------------
            # Store min/max counts so we can later identify valid parameters
            for result in nucleus_results:
                result['min_count'] = min_count
                result['max_count'] = max_count
                results.append(result)
            
            # --------------------------------------------------------
            # Show summary for this nucleus
            # --------------------------------------------------------
            # Count how many parameter combos gave acceptable foci count
            correct = sum(1 for r in nucleus_results 
                         if min_count <= r['foci_count'] <= max_count)
            print(f"    ✓ {correct}/{len(nucleus_results)} parameters gave acceptable count")
        
        # ================================================================
        # STORE RESULTS AND VISUALIZE
        # ================================================================
        # Convert list of dicts to DataFrame for easy analysis
        self.grid_results = pd.DataFrame(results)
        print(f"\n✓ Grid search complete: {len(self.grid_results)} total results")
        
        # Generate 3D scatter plot visualization
        # Shows parameter space colored by correctness (green=good, gray=bad)
        self._visualize_grid_search_results()
        
        return self.grid_results
    
    def _detect_foci_for_nucleus(self, nucleus_mask, channel_image, original_image,
                                 bright_grid, contrast_grid, percentile_grid, cell_id):
        """
        Run actual foci detection for one nucleus with all parameter combinations.
        
        This function uses THE SAME detection algorithm as the main pipeline.
        It imports functions directly from nucleus_worker_Visualization to ensure
        consistency. Any changes to the main detection logic will automatically
        apply here.
        
        The detection process:
        1. Isolate nucleus (zero out pixels outside mask)
        2. Apply DoG filter (difference of Gaussians)
        3. Find candidates (peak_local_max on both filtered and unfiltered)
        4. Compute adaptive backgrounds (texture-aware, per-focus)
        5. Apply filters (brightness, contrast, spatial matching)
        6. Count confirmed foci for each parameter combination
        
        This function includes EXTENSIVE DEBUG OUTPUT to help diagnose issues.
        
        Parameters:
        -----------
        nucleus_mask : numpy.ndarray
            Boolean mask, shape (H, W)
            True for pixels inside this nucleus, False elsewhere
        channel_image : numpy.ndarray
            Fluorescence image, shape (H, W)
            Float array with intensity values
        original_image : numpy.ndarray
            Original unprocessed image, shape (H, W)
            Used for global percentile calculations
        bright_grid : numpy.ndarray
            Background percentile values to test, shape (N,)
            Example: [0, 5, 10, ..., 95, 100]
        contrast_grid : numpy.ndarray
            Contrast threshold values to test, shape (N,)
            Example: [1.0, 1.5, 2.0, ..., 9.5, 10.0]
        percentile_grid : numpy.ndarray
            Global percentile values to test, shape (N,)
            Example: [0, 5, 10, ..., 95, 100]
        cell_id : int
            Nucleus identifier (for debug messages)
            
        Returns:
        --------
        list of dict
            One dict per parameter combination tested
            Each dict contains:
            {
                'cell_num': cell_id,
                'bright_pct': background_percentile,
                'contrast_thresh': contrast_threshold,
                'percentile_val': global_percentile,
                'foci_count': number_of_detected_foci
            }
            Returns empty list [] if detection fails (no signal, no candidates, etc.)
            
        Debug Output:
        -------------
        Prints diagnostic information including:
        - Max intensity in isolated nucleus
        - Nucleus mask size (pixels)
        - Number of positive pixels
        - Min brightness thresholds
        - Number of candidates found
        - Intensity ranges
        - Nucleus texture (CV)
        - Contrast multiplier (if applied)
        - Final results summary
        
        Early Exit Conditions:
        ----------------------
        Returns [] if:
        - isolated_img.max() == 0 (no signal in nucleus)
        - pos_pixels.size == 0 (no positive pixels in image)
        - No candidates found (neither filtered nor unfiltered)
        
        Notes:
        ------
        - Uses texture-aware contrast multiplier (1.5× if CV < 0.20)
        - Applies same background estimation as main pipeline
        - Uses vectorized filtering for efficiency
        - All warnings/errors are printed but don't raise exceptions
        """
        # ================================================================
        # DEBUG BANNER
        # ================================================================
        print(f"\n    🔍 Debug nucleus {cell_id}:")

        # ================================================================
        # PREPARE IMAGES
        # ================================================================
        # Convert original image to float for percentile calculations
        original_image_float = img_as_float(original_image)
        
        # ----------------------------------------------------------------
        # Isolate nucleus (zero out background)
        # ----------------------------------------------------------------
        # Create isolated image: original intensity inside nucleus, 0 outside
        isolated_img = img_as_float(channel_image.copy())
        isolated_img[~nucleus_mask] = 0
        
        # ----------------------------------------------------------------
        # Debug check: Is there any signal?
        # ----------------------------------------------------------------
        print(f"      Max intensity in isolated nucleus: {isolated_img.max():.6f}")
        print(f"      Nucleus mask size: {np.sum(nucleus_mask)} pixels")
        
        # Early exit if nucleus has no signal
        if isolated_img.max() == 0:
            print(f"  Warning: Nucleus {cell_id} has no signal, skipping")
            return []
        
        # ================================================================
        # APPLY DOG FILTER (same as main pipeline)
        # ================================================================
        # Difference of Gaussians enhances foci while suppressing large features
        filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
        
        # Clip negative values (DoG can produce negatives)
        filtered_img = np.clip(filtered_img, 0, None)
        
        # Rescale to match original intensity range
        # This preserves absolute brightness information
        filtered_img = exposure.rescale_intensity(
            filtered_img, 
            in_range='image',              # Auto-detect current range
            out_range=(0, isolated_img.max())  # Scale to original max
        )
        
        # ================================================================
        # PREPARE FOR PERCENTILE CALCULATIONS
        # ================================================================
        # Get all positive pixels from original image for global percentiles
        pos_pixels = original_image_float[original_image_float > 0]
        
        # ----------------------------------------------------------------
        # Debug check: Are there positive pixels?
        # ----------------------------------------------------------------
        print(f"      Positive pixels in original: {pos_pixels.size}")
        
        # Early exit if no positive pixels
        if pos_pixels.size == 0:
            print(f"      ❌ EARLY EXIT: No positive pixels in original image")
            return []
        
        # ----------------------------------------------------------------
        # Convert parameter grids to arrays
        # ----------------------------------------------------------------
        # Ensure we have proper numpy arrays for vectorized operations
        bright_grid_arr = np.array(bright_grid)
        contrast_grid_arr = np.array(contrast_grid)
        percentile_grid_arr = np.array(percentile_grid)
        
        # ----------------------------------------------------------------
        # Compute minimum brightness thresholds
        # ----------------------------------------------------------------
        # For each percentile value, compute the corresponding brightness threshold
        # This is used as absolute brightness filter
        min_brightness_per_param = np.percentile(pos_pixels, percentile_grid_arr)
        
        # Global minimum across all parameters
        # This is used as threshold for candidate detection
        global_min_brightness = np.min(min_brightness_per_param)
        
        # ----------------------------------------------------------------
        # Debug check: What's the minimum brightness?
        # ----------------------------------------------------------------
        print(f"      Global min brightness: {global_min_brightness:.6f}")
        print(f"      Range of min_brightness_per_param: "
              f"{min_brightness_per_param.min():.6f} - {min_brightness_per_param.max():.6f}")
        
        # ================================================================
        # FIND CANDIDATE FOCI (same as main pipeline)
        # ================================================================
        # Find local maxima in FILTERED image
        # min_distance=2: Peaks must be at least 2 pixels apart
        # threshold_abs: Peaks must be at least this bright (absolute)
        candidates_filtered = peak_local_max(
            filtered_img, 
            min_distance=2, 
            threshold_abs=global_min_brightness
        )
        
        # Find local maxima in UNFILTERED image
        # Used for spatial matching (foci should appear in both)
        candidates_unfiltered = peak_local_max(
            isolated_img, 
            min_distance=2, 
            threshold_abs=global_min_brightness
        )
        
        # ----------------------------------------------------------------
        # Debug check: Are candidates found?
        # ----------------------------------------------------------------
        print(f"      Candidates (filtered): {len(candidates_filtered)}")
        print(f"      Candidates (unfiltered): {len(candidates_unfiltered)}")
        
        # Early exit if no candidates
        if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
            print(f"      ❌ EARLY EXIT: No candidates found")
            return []
        
        # ================================================================
        # EXTRACT INTENSITIES
        # ================================================================
        # Get intensity values at candidate locations
        # These are used for brightness and contrast filtering
        unf_intensities = isolated_img[candidates_unfiltered[:, 0], candidates_unfiltered[:, 1]]
        filt_intensities = filtered_img[candidates_filtered[:, 0], candidates_filtered[:, 1]]
        
        # ----------------------------------------------------------------
        # Debug check: What are the candidate intensities?
        # ----------------------------------------------------------------
        print(f"      Unfiltered intensity range: "
              f"{unf_intensities.min():.6f} - {unf_intensities.max():.6f}")
        print(f"      Filtered intensity range: "
              f"{filt_intensities.min():.6f} - {filt_intensities.max():.6f}")
        
        # ================================================================
        # COMPUTE ADAPTIVE BACKGROUNDS (same as main pipeline)
        # ================================================================
        # Get unique brightness percentiles to compute backgrounds for
        # Round to avoid near-duplicates
        unique_brights = np.unique(np.round(bright_grid_arr, 6))
        
        # Create mapping from percentile to index
        # This is used to look up pre-computed backgrounds efficiently
        bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}
        
        print(f"      Computing backgrounds for {len(unique_brights)} unique brightness percentiles...")
        
        # ----------------------------------------------------------------
        # Compute backgrounds for unfiltered candidates
        # ----------------------------------------------------------------
        # This uses the SAME adaptive background function as main pipeline
        # return_texture_info=True gives us nucleus statistics
        local_percentiles_unf, texture_info_unf = compute_adaptive_background_texture_nucleus_fallback(
            image=isolated_img,
            coords=candidates_unfiltered,
            unique_percentiles=unique_brights,
            nucleus_mask=nucleus_mask,
            return_texture_info=True  # Get texture stats
        )
        
        # ----------------------------------------------------------------
        # Compute backgrounds for filtered candidates
        # ----------------------------------------------------------------
        # Same function, but we don't need texture info again
        local_percentiles_filt = compute_adaptive_background_texture_nucleus_fallback(
            image=filtered_img,
            coords=candidates_filtered,
            unique_percentiles=unique_brights,
            nucleus_mask=nucleus_mask,
            return_texture_info=False  # Don't need texture info twice
        )
        
        # ================================================================
        # TEXTURE-AWARE CONTRAST MULTIPLIER
        # ================================================================
        # Check if nucleus is uniform (low texture)
        # Uniform nuclei get contrast multiplier to avoid false positives
        contrast_multiplier = 1.0  # Default: no multiplier
        
        if texture_info_unf['nucleus_stats']:
            # Extract nucleus statistics (there should be exactly one nucleus)
            stats = list(texture_info_unf['nucleus_stats'].values())[0]
            nucleus_cv = stats['cv']  # Coefficient of variation
            
            print(f"      Nucleus CV: {nucleus_cv:.3f}")
            
            # Apply multiplier for uniform nuclei (CV < 0.20) 
            #--> currently disabled(multiplier at 1x), but left code construction for future use
            if nucleus_cv < 0.20:
                print(f"      ⚠️ Low texture - applying 1.0x contrast multiplier")
                contrast_multiplier = 1.0
                # Scale up all contrast thresholds
                contrast_grid_arr = contrast_grid_arr * contrast_multiplier
        
        # ================================================================
        # COMPUTE DISTANCES FOR SPATIAL MATCHING
        # ================================================================
        # Calculate distances between unfiltered and filtered candidates
        # Used to match foci: same focus should appear in both images
        distances = cdist(candidates_unfiltered, candidates_filtered)
        
        # ================================================================
        # TEST ALL PARAMETER COMBINATIONS
        # ================================================================
        results = []  # Accumulator for results
        
        print(f"      Testing {len(bright_grid)} parameter combinations...")
        
        # Loop through each parameter combination
        for p_idx in range(len(bright_grid)):
            # --------------------------------------------------------
            # Apply filters using main pipeline function
            # --------------------------------------------------------
            # This function applies all three filters:
            # 1. Absolute brightness filter
            # 2. Local contrast filter
            # 3. Spatial matching filter
            # Returns confirmed coordinates and count
            confirmed_coords, count = apply_foci_filters(
                p_idx, bright_grid_arr, contrast_grid_arr, percentile_grid_arr,
                min_brightness_per_param, bright_to_idx,
                unf_intensities, filt_intensities,
                local_percentiles_unf, local_percentiles_filt,
                distances, candidates_unfiltered, tolerance=2
            )
            
            # --------------------------------------------------------
            # Store result for this parameter combination
            # --------------------------------------------------------
            results.append({
                'cell_num': cell_id,
                'bright_pct': bright_grid[p_idx],
                'contrast_thresh': contrast_grid[p_idx],
                'percentile_val': percentile_grid[p_idx],
                'foci_count': count
            })
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        print(f"      ✅ Generated {len(results)} results")
        print(f"      Foci counts range: "
              f"{min([r['foci_count'] for r in results])} - "
              f"{max([r['foci_count'] for r in results])}")
        
        return results

    def find_valid_intersection(self):
        """
        Find parameter combinations that work for ALL ground truth nuclei.
        
        This is a CRITICAL step that filters the parameter space to only
        include parameters that give acceptable foci counts for every
        ground truth nucleus.
        
        LOGIC:
        ------
        1. For each nucleus, find all parameters where:
           min_count ≤ detected_count ≤ max_count
        2. Find the INTERSECTION of these parameter sets
           (parameters must work for nucleus 1 AND nucleus 2 AND nucleus 3 ...)
        3. Store the intersection as "valid parameters"
        
        WHY INTERSECTION:
        -----------------
        We want parameters that are ROBUST - they work across diverse nuclei.
        If a parameter works great for nucleus 1 but fails on nucleus 2, it's
        not reliable for the full dataset.
        
        EXAMPLE:
        --------
        Nucleus 1 (ground truth: 3 foci):
          Parameter (10, 2.0, 5) → detects 3 foci ✓
          Parameter (15, 2.5, 10) → detects 3 foci ✓
          Parameter (20, 3.0, 15) → detects 5 foci ✗
        
        Nucleus 2 (ground truth: 7-9 foci):
          Parameter (10, 2.0, 5) → detects 11 foci ✗
          Parameter (15, 2.5, 10) → detects 8 foci ✓
          Parameter (20, 3.0, 15) → detects 8 foci ✓
        
        Intersection (valid for BOTH):
          Parameter (15, 2.5, 10) ✓ (only this one works for both)
        
        Returns:
        --------
        numpy.ndarray or None
            Valid parameter combinations, shape (N, 3)
            Each row: [bright_pct, contrast_thresh, percentile_val]
            Returns None if no valid parameters found
            Also stored in self.valid_points
            
        Raises:
        -------
        ValueError
            If grid_results is None (must run generate_grid_search() first)
            
        Side Effects:
        -------------
        - Stores valid parameters in self.valid_points
        - Generates and displays visualization (_visualize_valid_intersection)
        - Prints summary statistics for each nucleus
        - Prints warnings if no valid parameters found
        
        Example Output:
        ---------------
        STEP 3: FINDING VALID PARAMETER INTERSECTION
        ============================================================
        
          Nucleus 5 (exact: 3): 2847 valid combinations (19.8%)
          Nucleus 12 (range: 4-6): 4521 valid combinations (31.4%)
          Nucleus 23 (exact: 8): 1834 valid combinations (12.7%)
        
        ✓ Found 521 parameters valid for ALL nuclei
        
        OR if no intersection:
        
        ✓ Found 0 parameters valid for ALL nuclei
        
        ⚠️ WARNING: No parameters work for all nuclei!
           Consider:
           - Checking ground truth annotations
           - Expanding parameter ranges
           - Using fewer or different nuclei
        
        Notes:
        ------
        - Uses set intersection (efficient for large parameter spaces)
        - Supports both exact counts (min=max) and ranges (min<max)
        - Visualization shows all tested parameters (very light gray) and valid ones (green)
        - If no valid parameters, suggests corrective actions
        """
        # ================================================================
        # BANNER AND VALIDATION
        # ================================================================
        print("\n" + "="*60)
        print("STEP 3: FINDING VALID PARAMETER INTERSECTION")
        print("="*60)
        
        # Ensure grid search has been run
        if self.grid_results is None:
            raise ValueError("Must run generate_grid_search() first")
        
        # ================================================================
        # FIND VALID PARAMETERS FOR EACH NUCLEUS
        # ================================================================
        # Dictionary: {nucleus_id: array_of_valid_parameter_combos}
        valid_params_per_nucleus = {}
        
        # Process each ground truth nucleus
        for cell_id, (min_count, max_count) in self.ground_truth_nuclei.items():
            # --------------------------------------------------------
            # Get all grid search results for this nucleus
            # --------------------------------------------------------
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            
            # --------------------------------------------------------
            # Filter to acceptable foci counts
            # --------------------------------------------------------
            # Accept any count within [min_count, max_count] (inclusive)
            # This handles both exact counts (min=max) and ranges (min<max)
            valid_for_nucleus = nucleus_data[
                (nucleus_data['foci_count'] >= min_count) & 
                (nucleus_data['foci_count'] <= max_count)
            ]
            
            # --------------------------------------------------------
            # Extract parameter combinations
            # --------------------------------------------------------
            # Get just the parameter columns, convert to numpy array
            valid_params = valid_for_nucleus[['bright_pct', 'contrast_thresh', 'percentile_val']].values
            valid_params_per_nucleus[cell_id] = valid_params
            
            # --------------------------------------------------------
            # Print summary for this nucleus
            # --------------------------------------------------------
            total_tested = len(nucleus_data)
            num_valid = len(valid_params)
            percent_valid = (num_valid / total_tested * 100) if total_tested > 0 else 0
            
            if min_count == max_count:
                # Exact count
                print(f"  Nucleus {cell_id} (exact: {min_count}): "
                      f"{num_valid} valid combinations ({percent_valid:.1f}%)")
            else:
                # Range
                print(f"  Nucleus {cell_id} (range: {min_count}-{max_count}): "
                      f"{num_valid} valid combinations ({percent_valid:.1f}%)")
        
        # ================================================================
        # COMPUTE INTERSECTION
        # ================================================================
        # Find parameters that work for ALL nuclei
        
        # ----------------------------------------------------------------
        # Handle edge case: no nuclei processed
        # ----------------------------------------------------------------
        if len(valid_params_per_nucleus) == 0:
            print("⚠️ No valid parameters found!")
            return None
        
        # ----------------------------------------------------------------
        # Convert to sets of tuples for intersection
        # ----------------------------------------------------------------
        # Sets allow efficient intersection operations
        # Must convert arrays to tuples (hashable) for set operations
        param_sets = []
        for cell_id, params in valid_params_per_nucleus.items():
            # Convert each parameter combo to tuple, create set
            param_tuples = set([tuple(p) for p in params])
            param_sets.append(param_tuples)
        
        # ----------------------------------------------------------------
        # Find intersection across all nuclei
        # ----------------------------------------------------------------
        # Start with first nucleus's valid parameters
        intersection = param_sets[0]
        
        # Intersect with each subsequent nucleus
        # Result: only parameters that appear in ALL sets
        for param_set in param_sets[1:]:
            intersection = intersection.intersection(param_set)
        
        # ----------------------------------------------------------------
        # Convert back to numpy array
        # ----------------------------------------------------------------
        if len(intersection) > 0:
            # Have valid parameters - convert tuples back to array
            self.valid_points = np.array(list(intersection))
        else:
            # No intersection - empty array
            self.valid_points = np.array([])
        
        # ================================================================
        # SUMMARY AND WARNINGS
        # ================================================================
        print(f"\n✓ Found {len(self.valid_points)} parameters valid for ALL nuclei")
        
        # ----------------------------------------------------------------
        # Warning if no valid parameters
        # ----------------------------------------------------------------
        if len(self.valid_points) == 0:
            print("\n⚠️ WARNING: No parameters work for all nuclei!")
            print("   Consider:")
            print("   - Checking ground truth annotations")
            print("   - Expanding parameter ranges")
            print("   - Using fewer or different nuclei")
        
        # ================================================================
        # VISUALIZE INTERSECTION
        # ================================================================
        # Show 3D scatter plot: all tested (gray) vs valid (green)
        self._visualize_valid_intersection()
        
        return self.valid_points
    
    def generate_kde_parameter_space(self, coverage_percentile=85):
        """
        Generate smooth parameter space boundary using Kernel Density Estimation.
        
        This function takes the discrete set of valid parameters (from intersection)
        and creates a CONTINUOUS statistical model of the parameter space. This
        enables smooth parameter sampling and defines a convex hull for efficient
        random parameter generation.
        
        WHAT IS KDE:
        ------------
        Kernel Density Estimation models the probability density of valid parameters.
        Instead of just discrete points, we get a smooth surface showing "how valid"
        each region of parameter space is.
        
        WHY USE KDE:
        ------------
        1. Smooth interpolation between valid points
        2. Can generate new parameters within valid region
        3. Defines a boundary around valid space (isosurface)
        4. Provides statistical confidence about parameter validity
        
        WORKFLOW:
        ---------
        1. Normalize valid parameters to [0,1]³ cube
        2. Find optimal KDE bandwidth (cross-validation)
        3. Fit KDE model to normalized valid parameters
        4. Generate isosurface at specified coverage percentile
        5. Build Delaunay triangulation of isosurface
        6. Denormalize and compute bounding box
        
        Parameters:
        -----------
        coverage_percentile : float, default=85
            What percentage of valid parameters should be enclosed by isosurface
            - 85: Conservative (captures 85% of valid points)
            - 95: More inclusive (captures 95% of valid points)
            - 99: Very inclusive (captures nearly all valid points)
            Higher values create larger parameter spaces with more uncertainty
            
        Returns:
        --------
        scipy.spatial.Delaunay or None
            Delaunay triangulation of parameter space
            Can be used for point-in-hull tests and random sampling
            Returns None if insufficient valid points (<10)
            Also stored in self.hull
            
        Side Effects:
        -------------
        - Stores KDE model in self.kde_model
        - Stores Delaunay hull in self.hull
        - Stores bounding box in self.bounds
        - Stores metadata in self.kde_metadata (bandwidth, threshold, isosurface points)
        - Generates and displays 3D mesh visualization
        
        Raises:
        -------
        Prints warning and returns None if:
        - self.valid_points is None (no valid parameters)
        - len(self.valid_points) < 10 (insufficient data for KDE)
        
        Example Output:
        ---------------
        STEP 4: GENERATING KDE PARAMETER SPACE
        ============================================================
        
        1. Finding optimal KDE bandwidth...
           Optimal bandwidth: 0.127
        
        2. Fitting KDE model...
        
        3. Generating isosurface...
           Generated 3847 isosurface points
           Coverage: 85%
        
        4. Building Delaunay triangulation...
        
        ✓ KDE parameter space generated
          Bounding box:
            bright_pct: 12.34 - 67.89
            contrast_thresh: 1.87 - 7.23
            percentile_val: 8.45 - 72.31
        
        Technical Details:
        ------------------
        BANDWIDTH SELECTION:
        - Uses GridSearchCV with cross-validation
        - Tests bandwidths from 0.01 to 1.0 (log-spaced)
        - Requires ≥20 points for cross-validation
        - Falls back to 0.1 if <20 points
        
        ISOSURFACE GENERATION:
        - Importance sampling around valid points (5000 samples)
        - Additional uniform sampling (2500 samples)
        - Computes density for all samples
        - Selects samples above threshold (defined by coverage_percentile)
        
        DELAUNAY TRIANGULATION:
        - Built on DENORMALIZED isosurface points (original scale)
        - Defines convex hull of valid parameter space
        - Used for efficient point-in-hull testing
        - Enables random sampling within valid region
        
        Notes:
        ------
        - Normalization is essential for KDE (parameters have different scales)
        - Isosurface smooths out discrete grid artifacts
        - Delaunay hull is built in original (denormalized) coordinates
        - Visualization shows both KDE isosurface and Delaunay hull
        - If no isosurface points generated, uses valid_points directly
        """
        # ================================================================
        # VALIDATION
        # ================================================================
        # Need sufficient valid points for meaningful KDE
        if self.valid_points is None or len(self.valid_points) < 10:
            print("⚠️ Not enough valid points for KDE")
            return None
        
        # ================================================================
        # BANNER
        # ================================================================
        print("\n" + "="*60)
        print("STEP 4: GENERATING KDE PARAMETER SPACE")
        print("="*60)
        
        # ================================================================
        # STEP 1: NORMALIZE PARAMETERS
        # ================================================================
        # KDE works best with normalized data (all dimensions on same scale)
        # Transform parameters from their original ranges to [0, 1]³
        normalized_points = self._normalize_parameters(self.valid_points)
        
        # ================================================================
        # STEP 2: FIND OPTIMAL BANDWIDTH
        # ================================================================
        # Bandwidth controls KDE smoothness
        # Too small: Overfitting (lumpy surface)
        # Too large: Oversmoothing (loses detail)
        
        if len(normalized_points) >= 20:
            # --------------------------------------------------------
            # Use cross-validation to find optimal bandwidth
            # --------------------------------------------------------
            print("\n1. Finding optimal KDE bandwidth...")
            
            # Test bandwidths from 0.01 to 1.0 (logarithmically spaced)
            bandwidths = np.logspace(-2, 0, 10)
            
            # Set up cross-validation
            # cv: Number of folds (min of 5 or n_points//4)
            kde_cv = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': bandwidths},
                cv=min(5, len(normalized_points) // 4)
            )
            
            # Fit and find best bandwidth
            kde_cv.fit(normalized_points)
            best_bandwidth = kde_cv.best_params_['bandwidth']
            print(f"   Optimal bandwidth: {best_bandwidth:.3f}")
        else:
            # --------------------------------------------------------
            # Not enough points for cross-validation - use default
            # --------------------------------------------------------
            best_bandwidth = 0.1
            print(f"   Using default bandwidth: {best_bandwidth}")
        
        # ================================================================
        # STEP 3: FIT KDE MODEL
        # ================================================================
        print("\n2. Fitting KDE model...")
        
        # Create and fit KDE model
        # kernel='gaussian': Use Gaussian kernel (smooth, standard choice)
        self.kde_model = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
        self.kde_model.fit(normalized_points)
        
        # ================================================================
        # STEP 4: GENERATE ISOSURFACE POINTS
        # ================================================================
        print("\n3. Generating isosurface...")
        
        n_samples = 5000  # Number of samples to generate
        
        # ----------------------------------------------------------------
        # Importance sampling around valid points
        # ----------------------------------------------------------------
        # Sample near existing valid points (more likely to be valid)
        base_samples = normalized_points[
            np.random.choice(len(normalized_points), n_samples, replace=True)
        ]
        
        # Add Gaussian noise to explore nearby regions
        noise = np.random.normal(0, best_bandwidth, base_samples.shape)
        samples = np.clip(base_samples + noise, 0, 1)  # Clip to [0,1] cube
        
        # ----------------------------------------------------------------
        # Add uniform samples for coverage
        # ----------------------------------------------------------------
        # Sample uniformly to ensure we explore empty regions too
        uniform_samples = np.random.uniform(0, 1, (n_samples // 2, 3))
        
        # Combine importance and uniform samples
        all_samples = np.vstack([samples, uniform_samples])
        
        # ----------------------------------------------------------------
        # Compute density for all samples
        # ----------------------------------------------------------------
        # KDE returns log density (for numerical stability)
        log_densities = self.kde_model.score_samples(all_samples)
        densities = np.exp(log_densities)  # Convert to actual density
        
        # ----------------------------------------------------------------
        # Find density threshold for isosurface
        # ----------------------------------------------------------------
        # Compute densities at actual valid points
        valid_densities = np.exp(self.kde_model.score_samples(normalized_points))
        
        # Threshold: density that captures specified percentile of valid points
        # Example: 85th percentile means isosurface encloses 85% of valid points
        threshold = np.percentile(valid_densities, 100 - coverage_percentile)
        
        # ----------------------------------------------------------------
        # Select points above threshold (on or inside isosurface)
        # ----------------------------------------------------------------
        isosurface_mask = densities > threshold
        isosurface_points = all_samples[isosurface_mask]
        
        print(f"   Generated {len(isosurface_points)} isosurface points")
        print(f"   Coverage: {coverage_percentile}%")
        
        # ================================================================
        # STEP 5: BUILD DELAUNAY TRIANGULATION
        # ================================================================
        print("\n4. Building Delaunay triangulation...")
        
        if len(isosurface_points) > 0:
            # --------------------------------------------------------
            # Use isosurface points for hull
            # --------------------------------------------------------
            # IMPORTANT: Denormalize before creating hull
            # Hull should be in original parameter space, not [0,1]³
            denorm_isosurface = self._denormalize_parameters(isosurface_points)
            self.hull = Delaunay(denorm_isosurface)
        else:
            # --------------------------------------------------------
            # Fallback: use valid points directly
            # --------------------------------------------------------
            print("   ⚠️ No isosurface points, using valid points directly")
            self.hull = Delaunay(self.valid_points)
        
        # ================================================================
        # STEP 6: CALCULATE BOUNDING BOX
        # ================================================================
        # Compute min/max for each parameter dimension
        # This defines a rectangular bounding box around the valid space
        
        if len(isosurface_points) > 0:
            # Use denormalized isosurface points
            denorm_points = denorm_isosurface
        else:
            # Use original valid points
            denorm_points = self.valid_points
        
        # Create bounds dictionary
        self.bounds = {
            'bright_pct': (denorm_points[:, 0].min(), denorm_points[:, 0].max()),
            'contrast_thresh': (denorm_points[:, 1].min(), denorm_points[:, 1].max()),
            'percentile_val': (denorm_points[:, 2].min(), denorm_points[:, 2].max())
        }
        
        # ================================================================
        # PRINT SUMMARY
        # ================================================================
        print(f"\n✓ KDE parameter space generated")
        print(f"  Bounding box:")
        for key, (min_val, max_val) in self.bounds.items():
            print(f"    {key}: {min_val:.2f} - {max_val:.2f}")
        
        # ================================================================
        # STORE METADATA
        # ================================================================
        # Save KDE configuration for later reference or reproduction
        self.kde_metadata = {
            'bandwidth': best_bandwidth,           # KDE bandwidth used
            'threshold': threshold,                # Density threshold for isosurface
            'coverage_percentile': coverage_percentile,  # Target coverage
            'isosurface_points': isosurface_points  # Generated isosurface points (normalized)
        }
        
        # ================================================================
        # VISUALIZE
        # ================================================================
        # Generate 3D mesh visualization showing KDE isosurface and Delaunay hull
        self._visualize_kde_and_delaunay()
        
        return self.hull
    
    def _visualize_grid_search_results(self):
        """
        Create 3D scatter plot visualization for each ground truth nucleus.
        
        This function generates one plot per nucleus showing how different
        parameter combinations performed. The visualization helps understand:
        - Which regions of parameter space work well (green points)
        - Which regions fail (gray points)
        - How parameter choices affect foci count
        
        Each plot shows parameter space in normalized coordinates [0,1]³
        but with axis labels showing original parameter values for readability.
        
        PLOT ELEMENTS:
        --------------
        - Gray points (small, transparent): Parameters giving incorrect count
        - Green points (larger, opaque): Parameters giving correct count
        - Axes: Background %, Contrast Threshold, Global Percentile
        - Title: Nucleus ID, expected count, success rate
        
        COLOR CODING:
        -------------
        - Green: Parameter gives foci count within acceptable range
        - Gray: Parameter gives foci count outside acceptable range
        
        Side Effects:
        -------------
        - Creates one figure per nucleus
        - Displays all figures at once (plt.show())
        - Blocks until user closes plots
        
        Notes:
        ------
        - Skips nuclei with no results (shouldn't happen)
        - Uses normalized coordinates [0,1] for display
        - Customizes tick labels to show real parameter values
        - Automatically handles both exact counts and ranges
        """
        # ================================================================
        # VALIDATION
        # ================================================================
        if self.grid_results is None:
            return
        
        # ================================================================
        # CREATE PLOT FOR EACH NUCLEUS
        # ================================================================
        for idx, (cell_id, (min_count, max_count)) in enumerate(self.ground_truth_nuclei.items()):
            # --------------------------------------------------------
            # Create figure
            # --------------------------------------------------------
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # --------------------------------------------------------
            # Get data for this nucleus
            # --------------------------------------------------------
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            
            # Skip if no data (shouldn't happen)
            if len(nucleus_data) == 0:
                plt.close(fig)
                continue
            
            # --------------------------------------------------------
            # Normalize coordinates for display [0,1]
            # --------------------------------------------------------
            # Transform parameter values to [0,1] range for plotting
            x_norm = (nucleus_data['bright_pct'] - self.param_ranges['bright_pct'][0]) / \
                     (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            y_norm = (nucleus_data['contrast_thresh'] - self.param_ranges['contrast_thresh'][0]) / \
                     (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            z_norm = (nucleus_data['percentile_val'] - self.param_ranges['percentile_val'][0]) / \
                     (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            # --------------------------------------------------------
            # Classify parameters as correct/incorrect
            # --------------------------------------------------------
            # Correct: foci count within [min_count, max_count]
            # Incorrect: foci count outside this range
            correct_mask = (nucleus_data['foci_count'] >= min_count) & \
                          (nucleus_data['foci_count'] <= max_count)
            
            # --------------------------------------------------------
            # Plot incorrect points (gray background)
            # --------------------------------------------------------
            if (~correct_mask).any():
                ax.scatter(
                    x_norm[~correct_mask], 
                    y_norm[~correct_mask], 
                    z_norm[~correct_mask],
                    c='lightgray',  # Light gray color
                    s=2,            # Small size
                    alpha=0.1,      # Very transparent
                    label='Incorrect'
                )
            
            # --------------------------------------------------------
            # Plot correct points (green foreground)
            # --------------------------------------------------------
            if correct_mask.any():
                ax.scatter(
                    x_norm[correct_mask], 
                    y_norm[correct_mask], 
                    z_norm[correct_mask],
                    c='green',      # Green color
                    s=10,           # Larger size
                    alpha=0.8,      # More opaque
                    label='Correct'
                )
            
            # ================================================================
            # CONFIGURE AXES
            # ================================================================
            
            # --------------------------------------------------------
            # Set axis labels
            # --------------------------------------------------------
            ax.set_xlabel('Background %')
            ax.set_ylabel('Contrast Thresh')
            ax.set_zlabel('Global Percentile')
            
            # --------------------------------------------------------
            # Set axis limits to [0,1] (normalized)
            # --------------------------------------------------------
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            
            # --------------------------------------------------------
            # Customize tick labels to show real parameter values
            # --------------------------------------------------------
            # X-axis: Background % (0-100)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0', '25', '50', '75', '100'])
            
            # Y-axis: Contrast Threshold (1-10)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['1.0', '3.25', '5.5', '7.75', '10.0'])
            
            # Z-axis: Global Percentile (0-100)
            ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_zticklabels(['0', '25', '50', '75', '100'])
            
            # --------------------------------------------------------
            # Set title (different for exact vs range)
            # --------------------------------------------------------
            if min_count == max_count:
                # Exact count expected
                ax.set_title(
                    f'Nucleus {cell_id} - Expected: exactly {min_count} foci\n'
                    f'Green: {correct_mask.sum()} correct / '
                    f'Gray: {(~correct_mask).sum()} incorrect'
                )
            else:
                # Range expected
                ax.set_title(
                    f'Nucleus {cell_id} - Expected: {min_count}-{max_count} foci\n'
                    f'Green: {correct_mask.sum()} acceptable / '
                    f'Gray: {(~correct_mask).sum()} outside range'
                )

            ax.legend()
        
        # ================================================================
        # DISPLAY ALL PLOTS
        # ================================================================
        plt.show()
    
    def _visualize_valid_intersection(self):
        """
        Visualize the intersection of valid parameters across all nuclei.
        
        This creates a single 3D scatter plot showing:
        - All tested parameter combinations (light gray, background)
        - Parameters valid for ALL nuclei (green, highlighted)
        
        This visualization helps assess:
        - How restrictive the intersection is (few vs many valid parameters)
        - Where valid parameters cluster in parameter space
        - Whether valid region is contiguous or fragmented
        
        INTERPRETATION:
        ---------------
        - Dense green cluster: Strong consensus, robust parameters
        - Scattered green points: Fragmented valid space, less robust
        - Few/no green points: Ground truth may be inconsistent or ranges too tight
        
        Side Effects:
        -------------
        - Creates one figure with single 3D subplot
        - Displays immediately (plt.show())
        - Blocks until user closes plot
        
        Notes:
        ------
        - Uses normalized coordinates [0,1] for display
        - Tick labels show real parameter values
        - Green points have dark green edge for visibility
        - Title shows total number of ground truth nuclei and valid parameters
        """
        # ================================================================
        # VALIDATION
        # ================================================================
        if self.grid_results is None:
            return
        
        # ================================================================
        # CREATE FIGURE
        # ================================================================
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Valid Parameter Intersection', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111, projection='3d')
        
        # ================================================================
        # PLOT ALL TESTED PARAMETERS (gray background)
        # ================================================================
        # Get unique parameter combinations from grid search
        all_params = self.grid_results[
            ['bright_pct', 'contrast_thresh', 'percentile_val']
        ].drop_duplicates()
        
        # ----------------------------------------------------------------
        # Normalize coordinates to [0,1]
        # ----------------------------------------------------------------
        x_all = (all_params['bright_pct'] - self.param_ranges['bright_pct'][0]) / \
                (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
        y_all = (all_params['contrast_thresh'] - self.param_ranges['contrast_thresh'][0]) / \
                (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
        z_all = (all_params['percentile_val'] - self.param_ranges['percentile_val'][0]) / \
                (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
        
        # ----------------------------------------------------------------
        # Plot as very light gray background
        # ----------------------------------------------------------------
        ax.scatter(
            x_all, y_all, z_all, 
            c='lightgray',  # Very light gray
            s=2,            # Small points
            alpha=0.1,      # Very transparent
            label='All tested'
        )
        
        # ================================================================
        # PLOT VALID INTERSECTION (green foreground)
        # ================================================================
        if self.valid_points is not None and len(self.valid_points) > 0:
            # --------------------------------------------------------
            # Normalize valid points to [0,1]
            # --------------------------------------------------------
            x_valid = (self.valid_points[:, 0] - self.param_ranges['bright_pct'][0]) / \
                     (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            y_valid = (self.valid_points[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                     (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            z_valid = (self.valid_points[:, 2] - self.param_ranges['percentile_val'][0]) / \
                     (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            # --------------------------------------------------------
            # Plot as green with dark green edge
            # --------------------------------------------------------
            ax.scatter(
                x_valid, y_valid, z_valid, 
                c='green',           # Green color
                s=10,                # Larger than background
                alpha=0.8,           # More opaque
                edgecolor='darkgreen',  # Dark edge for definition
                linewidth=1,         # Edge thickness
                label='Valid for ALL nuclei'
            )
        
        # ================================================================
        # CONFIGURE AXES
        # ================================================================
        
        # ----------------------------------------------------------------
        # Set axis labels
        # ----------------------------------------------------------------
        ax.set_xlabel('Background %')
        ax.set_ylabel('Contrast Thresh')
        ax.set_zlabel('Global Percentile')
        
        # ----------------------------------------------------------------
        # Set axis limits to [0,1]
        # ----------------------------------------------------------------
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # ----------------------------------------------------------------
        # Customize tick labels to show real values
        # ----------------------------------------------------------------
        # X-axis: Background % (0-100)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '25', '50', '75', '100'])
        
        # Y-axis: Contrast Threshold (1-10)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['1.0', '3.25', '5.5', '7.75', '10.0'])
        
        # Z-axis: Global Percentile (0-100)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '25', '50', '75', '100'])
        
        # ----------------------------------------------------------------
        # Set title with summary statistics
        # ----------------------------------------------------------------
        num_nuclei = len(self.ground_truth_nuclei)
        num_valid = len(self.valid_points) if self.valid_points is not None else 0
        
        ax.set_title(
            f'Parameters valid for all {num_nuclei} nuclei\n'
            f'Green: {num_valid} valid combinations'
        )
        
        ax.legend()
        
        # ================================================================
        # DISPLAY
        # ================================================================
        plt.show()
    
    def _visualize_kde_and_delaunay(self):
        """
        Visualize KDE isosurface and Delaunay triangulation as 3D meshes.
        
        This creates a dual visualization showing:
        LEFT PLOT: KDE isosurface (smooth probability boundary)
        RIGHT PLOT: Delaunay hull (convex polytope)
        
        Both plots also show valid parameters as red dots for reference.
        
        PURPOSE:
        --------
        - Understand the shape and structure of valid parameter space
        - Compare smooth KDE boundary vs discrete Delaunay hull
        - Assess coverage and gaps in parameter space
        - Visualize 3D structure that's used for parameter sampling
        
        PLOT ELEMENTS:
        --------------
        KDE Isosurface (LEFT):
        - Light blue mesh: KDE isosurface at specified coverage percentile
        - Red dots: Actual valid parameters (for reference)
        - Transparent mesh allows seeing interior structure
        
        Delaunay Hull (RIGHT):
        - Cyan mesh: Convex hull triangulation
        - Blue dots: Hull vertices
        - Red dots: Actual valid parameters (for reference)
        - More angular than KDE (uses actual points)
        
        AXIS CONFIGURATION:
        -------------------
        IMPORTANT: Axes are SWAPPED to match grid search visualization
        - X-axis: Contrast Threshold (INVERTED for consistency)
        - Y-axis: Background %
        - Z-axis: Global Percentile
        
        This non-standard ordering matches how users see grid search results.
        
        Side Effects:
        -------------
        - Creates one figure with two side-by-side 3D subplots
        - Displays immediately (plt.show())
        - Blocks until user closes plot
        - May print warning if mesh creation fails
        
        Technical Details:
        ------------------
        MESH CREATION:
        - Uses ConvexHull from scipy.spatial
        - Converts hull simplices to 3D polygon collection
        - Alpha blending for transparency
        
        AXIS SWAPPING:
        - Original: [background, contrast, percentile]
        - Display: [contrast, background, percentile]
        - Swapping done via [:, [1, 0, 2]] indexing
        
        COORDINATE SYSTEMS:
        - KDE isosurface: Uses normalized points from kde_metadata
        - Delaunay hull: Uses denormalized points (original scale)
        - Both normalized to [0,1] for display
        
        Notes:
        ------
        - If mesh creation fails (degenerate hull), falls back to scatter plot
        - Inverted X-axis makes high contrast on left (matches grid search)
        - view_init(20, 45) sets camera angle for best viewing
        - Both plots use same axis configuration for direct comparison
        """
        # ================================================================
        # VALIDATION
        # ================================================================
        if self.kde_metadata is None:
            return
        
        # ================================================================
        # SETUP
        # ================================================================
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create figure with two side-by-side subplots
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('KDE Isosurface and Delaunay Triangulation', fontsize=16, fontweight='bold')
        
        # ================================================================
        # PREPARE VALID POINTS (for reference overlay)
        # ================================================================
        # Normalize and swap axes for consistent display
        if self.valid_points is not None and len(self.valid_points) > 0:
            # --------------------------------------------------------
            # Normalize to [0,1]
            # --------------------------------------------------------
            bright_norm = (self.valid_points[:, 0] - self.param_ranges['bright_pct'][0]) / \
                         (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            contrast_norm = (self.valid_points[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                           (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            percentile_norm = (self.valid_points[:, 2] - self.param_ranges['percentile_val'][0]) / \
                             (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            # --------------------------------------------------------
            # SWAP AXES: X=Contrast, Y=Background (to match grid search)
            # --------------------------------------------------------
            x_valid = contrast_norm      # X-axis = Contrast Thresh
            y_valid = bright_norm        # Y-axis = Background %
            z_valid = percentile_norm    # Z-axis = Percentile
        
        # ================================================================
        # LEFT PLOT: KDE ISOSURFACE
        # ================================================================
        ax1 = fig.add_subplot(121, projection='3d')
        
        # ----------------------------------------------------------------
        # Plot valid points as red dots (background reference)
        # ----------------------------------------------------------------
        if self.valid_points is not None and len(self.valid_points) > 0:
            ax1.scatter(
                x_valid, y_valid, z_valid, 
                c='red',            # Red for visibility against blue mesh
                s=3,                # Small dots
                alpha=0.1,          # Very transparent
                edgecolor='darkred',
                linewidth=1,
                label='Valid points', 
                zorder=10           # Draw on top
            )
        
        # ----------------------------------------------------------------
        # Create KDE isosurface mesh
        # ----------------------------------------------------------------
        # Get isosurface points from metadata (these are normalized [0,1])
        iso_points = self.kde_metadata['isosurface_points']
        
        if len(iso_points) > 3:
            try:
                from scipy.spatial import ConvexHull
                
                # --------------------------------------------------------
                # SWAP columns before creating hull
                # --------------------------------------------------------
                # Original: [background, contrast, percentile]
                # Display: [contrast, background, percentile] --> again to match the rest
                iso_points_swapped = iso_points[:, [1, 0, 2]]
                
                # Create convex hull
                hull_iso = ConvexHull(iso_points_swapped)
                
                # --------------------------------------------------------
                # Create mesh from hull simplices
                # --------------------------------------------------------
                # Each simplex is a triangle face of the hull
                faces = []
                for simplex in hull_iso.simplices:
                    faces.append(iso_points_swapped[simplex])
                
                # Create 3D polygon collection
                mesh = Poly3DCollection(
                    faces, 
                    alpha=0.2,              # Transparent
                    facecolor='lightblue',  # Light blue color
                    edgecolor='blue',       # Blue edges
                    linewidth=0.2
                )
                ax1.add_collection3d(mesh)
                
            except Exception as e:
                # Hull creation can fail for degenerate cases
                print(f"⚠️ Could not create KDE surface: {e}")
        
        # ----------------------------------------------------------------
        # Configure axes for LEFT plot
        # ----------------------------------------------------------------
        # Set axis labels (swapped order)
        ax1.set_xlabel('Contrast Thresh')
        ax1.set_ylabel('Background %')
        ax1.set_zlabel('Global Percentile')
        
        # Set axis limits to [0,1] (normalized)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_zlim(0, 1)
        
        # Customize tick labels to show real values
        # X-axis: Contrast Threshold (1-10, INVERTED)
        ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_xticklabels(['10.0', '7.75', '5.5', '3.25', '1.0'])
        ax1.invert_xaxis()  # Invert so high values on left
        
        # Y-axis: Background % (0-100)
        ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_yticklabels(['0', '25', '50', '75', '100'])
        
        # Z-axis: Global Percentile (0-100)
        ax1.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_zticklabels(['0', '25', '50', '75', '100'])
        
        # Set title and camera angle
        ax1.set_title(f'KDE Isosurface\nBandwidth: {self.kde_metadata["bandwidth"]:.3f}')
        ax1.legend()
        ax1.view_init(elev=20, azim=45)  # Set camera angle
        
        # ================================================================
        # RIGHT PLOT: DELAUNAY HULL
        # ================================================================
        ax2 = fig.add_subplot(122, projection='3d')
        
        if self.hull is not None:
            # --------------------------------------------------------
            # Get hull points and normalize for display
            # --------------------------------------------------------
            # Hull points are in REAL VALUES (original scale)
            hull_points_real = self.hull.points
            
            # Normalize to [0,1] for display
            hull_points = self._normalize_parameters(hull_points_real)
                 
            # --------------------------------------------------------
            # Create outer hull mesh
            # --------------------------------------------------------
            try:
                from scipy.spatial import ConvexHull
                
                # SWAP columns for consistent display
                hull_points_swapped = hull_points[:, [1, 0, 2]]
                
                # Create convex hull
                outer_hull = ConvexHull(hull_points_swapped)
                
                # Create mesh from simplices
                faces = []
                for simplex in outer_hull.simplices:
                    faces.append(hull_points_swapped[simplex])
                
                # Create 3D polygon collection
                mesh = Poly3DCollection(
                    faces, 
                    alpha=0.25,         # Semi-transparent
                    facecolor='cyan',   # Cyan color
                    edgecolor='darkblue',
                    linewidth=0.5
                )
                ax2.add_collection3d(mesh)
                
                # --------------------------------------------------------
                # Plot hull vertices
                # --------------------------------------------------------
                ax2.scatter(
                    hull_points_swapped[:, 0], 
                    hull_points_swapped[:, 1], 
                    hull_points_swapped[:, 2],
                    c='blue',           # Blue dots
                    s=5,                # Small
                    alpha=0.4,          # Semi-transparent
                    label='Hull vertices', 
                    zorder=10
                )
                
            except Exception as e:
                # Fallback: just plot points if mesh creation fails
                print(f"⚠️ Could not create Delaunay surface: {e}")
                hull_points_swapped = hull_points[:, [1, 0, 2]]
                ax2.scatter(
                    hull_points_swapped[:, 0], 
                    hull_points_swapped[:, 1], 
                    hull_points_swapped[:, 2],
                    c='blue', 
                    s=5, 
                    alpha=0.6
                )
        
        # ----------------------------------------------------------------
        # Show valid points (red dots) for reference
        # ----------------------------------------------------------------
        if self.valid_points is not None and len(self.valid_points) > 0:
            ax2.scatter(
                x_valid, y_valid, z_valid, 
                c='red',
                s=3,
                alpha=0.1,
                edgecolor='darkred', 
                linewidth=1,
                label='Valid points', 
                zorder=10
            )
        
        # ----------------------------------------------------------------
        # Configure axes for RIGHT plot
        # ----------------------------------------------------------------
        # Set axis labels (swapped order)
        ax2.set_xlabel('Contrast Thresh')
        ax2.set_ylabel('Background %')
        ax2.set_zlabel('Global Percentile')
        
        # Set axis limits to [0,1]
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_zlim(0, 1)
        
        # Customize tick labels (same as left plot)
        # X-axis: Contrast Threshold (1-10, INVERTED)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xticklabels(['10.0', '7.75', '5.5', '3.25', '1.0'])
        ax2.invert_xaxis()
        
        # Y-axis: Background % (0-100)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(['0', '25', '50', '75', '100'])
        
        # Z-axis: Global Percentile (0-100)
        ax2.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_zticklabels(['0', '25', '50', '75', '100'])
        
        # Set title and camera angle
        ax2.set_title(f'Delaunay Hull\n{len(hull_points) if self.hull else 0} points')
        ax2.legend()
        ax2.view_init(elev=20, azim=45)
        
        # ================================================================
        # DISPLAY
        # ================================================================
        plt.show()
    
    def _normalize_parameters(self, params):
        """
        Normalize parameters from original ranges to [0, 1]³ cube.
        
        This transformation is essential for KDE (Kernel Density Estimation)
        because parameters have vastly different scales:
        - bright_pct: 0-100 (large range)
        - contrast_thresh: 1-10 (small range)
        - percentile_val: 0-100 (large range)
        
        Without normalization, KDE would be dominated by large-scale parameters.
        
        FORMULA:
        --------
        normalized = (value - min) / (max - min)
        
        This maps:
        - min → 0
        - max → 1
        - Linear interpolation in between
        
        Parameters:
        -----------
        params : numpy.ndarray
            Parameters in original scale, shape (N, 3)
            Each row: [bright_pct, contrast_thresh, percentile_val]
            
        Returns:
        --------
        numpy.ndarray
            Normalized parameters, shape (N, 3)
            Each row: [bright_norm, contrast_norm, percentile_norm]
            All values in range [0, 1]
            
        Example:
        --------
        >>> params = np.array([[50, 5.5, 75]])
        >>> normalized = self._normalize_parameters(params)
        >>> print(normalized)
        [[0.5, 0.5, 0.75]]  # All scaled to [0,1]
        
        Notes:
        ------
        - Assumes param_ranges has been set correctly
        - Each dimension normalized independently
        - Preserves relative distances within each dimension
        - Does NOT preserve distances across dimensions
        """
        # Initialize output array with same shape
        normalized = np.zeros_like(params, dtype=float)
        
        # Normalize each parameter dimension
        # Column 0: bright_pct
        normalized[:, 0] = (params[:, 0] - self.param_ranges['bright_pct'][0]) / \
                          (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
        
        # Column 1: contrast_thresh
        normalized[:, 1] = (params[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                          (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
        
        # Column 2: percentile_val
        normalized[:, 2] = (params[:, 2] - self.param_ranges['percentile_val'][0]) / \
                          (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
        
        return normalized
    
    def _denormalize_parameters(self, params_norm):
        """
        Denormalize parameters from [0, 1]³ cube to original ranges.
        
        This is the inverse transformation of _normalize_parameters().
        Used after KDE operations to convert back to real parameter values.
        
        FORMULA:
        --------
        value = normalized × (max - min) + min
        
        This maps:
        - 0 → min
        - 1 → max
        - Linear interpolation in between
        
        Parameters:
        -----------
        params_norm : numpy.ndarray
            Normalized parameters, shape (N, 3)
            Each row: [bright_norm, contrast_norm, percentile_norm]
            All values should be in range [0, 1]
            
        Returns:
        --------
        numpy.ndarray
            Parameters in original scale, shape (N, 3)
            Each row: [bright_pct, contrast_thresh, percentile_val]
            
        Example:
        --------
        >>> params_norm = np.array([[0.5, 0.5, 0.75]])
        >>> params = self._denormalize_parameters(params_norm)
        >>> print(params)
        [[50, 5.5, 75]]  # Back to original scale
        
        Notes:
        ------
        - Exact inverse of _normalize_parameters()
        - Round-trip should recover original values (within floating point precision)
        - Used for creating Delaunay hull in original coordinate system
        """
        # Initialize output array with same shape
        params = np.zeros_like(params_norm)
        
        # Denormalize each parameter dimension
        # Column 0: bright_pct
        params[:, 0] = params_norm[:, 0] * (self.param_ranges['bright_pct'][1] - 
                                           self.param_ranges['bright_pct'][0]) + \
                      self.param_ranges['bright_pct'][0]
        
        # Column 1: contrast_thresh
        params[:, 1] = params_norm[:, 1] * (self.param_ranges['contrast_thresh'][1] - 
                                           self.param_ranges['contrast_thresh'][0]) + \
                      self.param_ranges['contrast_thresh'][0]
        
        # Column 2: percentile_val
        params[:, 2] = params_norm[:, 2] * (self.param_ranges['percentile_val'][1] - 
                                           self.param_ranges['percentile_val'][0]) + \
                      self.param_ranges['percentile_val'][0]
        
        return params
    
    def save_complete(self, output_dir):
        """
        Save complete parameter space to disk for later use.
        
        This function saves all components needed to use the calibrated
        parameter space in production. The saved files can be loaded by
        the main detection pipeline to generate valid parameter combinations.
        
        WHAT IS SAVED:
        --------------
        1. valid_parameter_hull.pkl: Delaunay triangulation (for point-in-hull tests)
        2. parameter_bounds.npz: Bounding box (min/max for each parameter)
        3. valid_points.npy: Raw valid parameter combinations
        4. kde_metadata.pkl: KDE configuration (bandwidth, threshold, isosurface)
        
        These files enable:
        - Random sampling within valid parameter space
        - Point-in-hull testing (is parameter combination valid?)
        - Bounding box for rejection sampling
        - Reproducible parameter generation
        
        Parameters:
        -----------
        output_dir : str
            Directory path where files will be saved
            Will be created if it doesn't exist
            Example: 'calibration_results/tritc/'
            
        Side Effects:
        -------------
        - Creates output_dir if needed (including parent directories)
        - Writes 4 files to output_dir (if corresponding data exists)
        - Prints confirmation for each saved file
        - Prints final summary
        
        File Formats:
        -------------
        - .pkl files: Python pickle (preserves object structure exactly)
        - .npz file: Numpy compressed archive (efficient for arrays)
        - .npy file: Numpy array format (simple, efficient)
        
        Example Output:
        ---------------
        >>> generator.save_complete('output/tritc_params/')
        ✓ Saved hull: output/tritc_params/valid_parameter_hull.pkl
        ✓ Saved bounds: output/tritc_params/parameter_bounds.npz
        ✓ Saved valid points: output/tritc_params/valid_points.npy
        ✓ Saved KDE metadata: output/tritc_params/kde_metadata.pkl
        
        ✓ Complete parameter space saved to: output/tritc_params/
        
        Loading Saved Data:
        -------------------
```python
        # In main pipeline
        import pickle
        import numpy as np
        
        # Load Delaunay hull
        with open('output/tritc_params/valid_parameter_hull.pkl', 'rb') as f:
            hull = pickle.load(f)
        
        # Load bounds
        bounds_data = np.load('output/tritc_params/parameter_bounds.npz')
        bounds = dict(bounds_data)
        
        # Load valid points
        valid_points = np.load('output/tritc_params/valid_points.npy')
        
        # Load KDE metadata
        with open('output/tritc_params/kde_metadata.pkl', 'rb') as f:
            kde_meta = pickle.load(f)
```
        
        Notes:
        ------
        - Only saves data that exists (checks for None before saving)
        - Safe to call multiple times (will overwrite existing files)
        - All paths are relative to output_dir
        - Pickle files are not portable across Python versions (use with caution)
        """
        # ================================================================
        # CREATE OUTPUT DIRECTORY
        # ================================================================
        # Create directory and any necessary parent directories
        # exist_ok=True prevents error if directory already exists
        os.makedirs(output_dir, exist_ok=True)
        
        # ================================================================
        # SAVE DELAUNAY HULL
        # ================================================================
        if self.hull is not None:
            hull_path = os.path.join(output_dir, "valid_parameter_hull.pkl")
            with open(hull_path, 'wb') as f:
                pickle.dump(self.hull, f)
            print(f"✓ Saved hull: {hull_path}")
        
        # ================================================================
        # SAVE PARAMETER BOUNDS
        # ================================================================
        if self.bounds is not None:
            bounds_path = os.path.join(output_dir, "parameter_bounds.npz")
            # **self.bounds unpacks dictionary as keyword arguments
            # Creates .npz with one array per parameter
            np.savez(bounds_path, **self.bounds)
            print(f"✓ Saved bounds: {bounds_path}")
        
        # ================================================================
        # SAVE VALID POINTS
        # ================================================================
        if self.valid_points is not None:
            points_path = os.path.join(output_dir, "valid_points.npy")
            np.save(points_path, self.valid_points)
            print(f"✓ Saved valid points: {points_path}")
        
        # ================================================================
        # SAVE KDE METADATA
        # ================================================================
        if self.kde_metadata is not None:
            kde_path = os.path.join(output_dir, "kde_metadata.pkl")
            with open(kde_path, 'wb') as f:
                pickle.dump(self.kde_metadata, f)
            print(f"✓ Saved KDE metadata: {kde_path}")
        
        # ================================================================
        # COMPLETION MESSAGE
        # ================================================================
        print(f"\n✓ Complete parameter space saved to: {output_dir}")
    
    # ====================================================================
    # HELPER VISUALIZATION METHODS
    # ====================================================================
    
    def _get_diverse_nuclei(self, masks, channel_image, n_suggest=20):
        """
        Suggest diverse nuclei for ground truth annotation.
        
        This is a HELPER FUNCTION (not currently used in main workflow)
        that could be used to automatically suggest which nuclei to annotate
        for ground truth. The goal is to select nuclei that are representative
        of the full dataset diversity.
        
        DIVERSITY CRITERIA:
        -------------------
        - Intensity: Select nuclei across intensity quartiles (dim to bright)
        - Random sampling: Fill remaining slots with random nuclei
        
        This ensures ground truth covers different nucleus types rather than
        being biased toward bright or dim nuclei.
        
        Parameters:
        -----------
        masks : numpy.ndarray
            Segmentation mask, shape (H, W)
        channel_image : numpy.ndarray
            Fluorescence image, shape (H, W)
        n_suggest : int, default=20
            Number of nuclei to suggest
            
        Returns:
        --------
        list of int
            List of suggested nucleus IDs
            Empty list if no valid nuclei found
            
        Algorithm:
        ----------
        1. Extract features for each nucleus (area, intensity, CV)
        2. Select nuclei at intensity quartiles (25th, 50th, 75th)
        3. Fill remaining slots with random samples
        4. Return up to n_suggest nucleus IDs
        
        Notes:
        ------
        - Not currently integrated into interactive_nucleus_selection
        - Could be added as an "auto-suggest" feature
        - Skips background (label 0)
        - Skips nuclei with no pixels
        """
        # ================================================================
        # EXTRACT NUCLEUS FEATURES
        # ================================================================
        # Use regionprops to get properties of each labeled region
        props = measure.regionprops(masks, intensity_image=channel_image)
        
        nucleus_features = []
        for prop in props:
            # Skip background
            if prop.label == 0:
                continue
            
            # Get pixels for this nucleus
            pixels = channel_image[masks == prop.label]
            if len(pixels) == 0:
                continue
                
            # Compute features
            features = {
                'label': prop.label,
                'area': prop.area,
                'mean_intensity': prop.mean_intensity,
                'cv': np.std(pixels) / prop.mean_intensity if prop.mean_intensity > 0 else 0
            }
            nucleus_features.append(features)
        
        # Handle empty case
        if not nucleus_features:
            return []
        
        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(nucleus_features)
        
        # ================================================================
        # SELECT DIVERSE NUCLEI
        # ================================================================
        selected = []
        
        if len(df) >= 4:
            # --------------------------------------------------------
            # Select nuclei at intensity quartiles
            # --------------------------------------------------------
            for q in [0.25, 0.5, 0.75]:
                # Find quartile value
                quartile_val = df['mean_intensity'].quantile(q)
                # Find nucleus closest to this quartile
                closest_idx = (df['mean_intensity'] - quartile_val).abs().idxmin()
                selected.append(df.loc[closest_idx, 'label'])
        
        # ----------------------------------------------------------------
        # Fill remaining slots with random samples
        # ----------------------------------------------------------------
        remaining = df[~df['label'].isin(selected)]
        if len(remaining) > 0:
            # Calculate how many more we need
            n_random = min(n_suggest - len(selected), len(remaining))
            if n_random > 0:
                # Random sample without replacement
                random_picks = remaining.sample(n=n_random)['label'].tolist()
                selected.extend(random_picks)
        
        # Return up to n_suggest nuclei
        return selected[:n_suggest]
    
    def _visualize_single_nucleus(self, masks, channel_image, nucleus_id):
        """
        Show detailed 3-panel view of a single nucleus.
        
        This visualization helps users accurately count foci by showing:
        1. Original image (raw intensities)
        2. DoG filtered image (foci enhanced)
        3. Adaptively enhanced image (extreme contrast)
        
        The three views complement each other:
        - Original: See absolute brightness, context
        - DoG: See foci as enhanced spots
        - Enhanced: See dim foci that might be missed
        
        Used during interactive nucleus selection to help users make
        accurate foci counts.
        
        Parameters:
        -----------
        masks : numpy.ndarray
            Segmentation mask, shape (H, W)
        channel_image : numpy.ndarray
            Fluorescence image, shape (H, W)
        nucleus_id : int
            ID of nucleus to visualize
            
        Side Effects:
        -------------
        - Creates figure with 3 subplots
        - Shows immediately (plt.show())
        - Displays nucleus statistics in text box
        
        Plot Elements:
        --------------
        PANEL 1 (LEFT): Original Image
        - Raw fluorescence intensities
        - Cyan contour showing nucleus boundary
        - Good for assessing absolute brightness
        
        PANEL 2 (MIDDLE): DoG Filtered
        - Difference of Gaussians enhancement
        - Highlights foci while suppressing background
        - Same filter used in actual detection
        
        PANEL 3 (RIGHT): Adaptive Histogram Equalization
        - Extreme local contrast enhancement
        - Good for finding dim/ambiguous foci
        - Hot colormap (black=dim, white=bright)
        
        Statistics Box:
        - Area: Number of pixels in nucleus
        - Mean: Average intensity
        - CV: Coefficient of variation (texture measure)
        
        Notes:
        ------
        - Automatically crops to nucleus with 30-pixel padding
        - All three views show same nucleus region
        - Contours use cyan for visibility
        - DoG uses same parameters as detection (sigma=1,2)
        """
        # ================================================================
        # EXTRACT NUCLEUS REGION
        # ================================================================
        nucleus_mask = (masks == nucleus_id)
        y_coords, x_coords = np.where(nucleus_mask)
        
        # Handle empty mask
        if len(y_coords) == 0:
            return
        
        # ----------------------------------------------------------------
        # Calculate crop boundaries with padding
        # ----------------------------------------------------------------
        padding = 30
        y_min = max(0, y_coords.min() - padding)
        y_max = min(channel_image.shape[0], y_coords.max() + padding)
        x_min = max(0, x_coords.min() - padding)
        x_max = min(channel_image.shape[1], x_coords.max() + padding)
        
        # ----------------------------------------------------------------
        # Crop image and mask
        # ----------------------------------------------------------------
        crop_img = channel_image[y_min:y_max, x_min:x_max]
        crop_mask = nucleus_mask[y_min:y_max, x_min:x_max]
        
        # ================================================================
        # CREATE ENHANCED VIEWS
        # ================================================================
        
        # ----------------------------------------------------------------
        # DoG filtered view (same as detection pipeline)
        # ----------------------------------------------------------------
        crop_dog = filters.difference_of_gaussians(crop_img, low_sigma=1, high_sigma=2)
        crop_dog = np.clip(crop_dog, 0, None)  # Remove negative values
        crop_dog = exposure.rescale_intensity(
            crop_dog, 
            in_range='image', 
            out_range=(0, crop_img.max())
        )
        
        # ----------------------------------------------------------------
        # Adaptive histogram equalization (extreme enhancement)
        # ----------------------------------------------------------------
        enhanced = exposure.equalize_adapthist(crop_img)
        enhanced = exposure.rescale_intensity(
            enhanced, 
            in_range='image', 
            out_range=(0, crop_img.max())
        )
        
        # ================================================================
        # CREATE FIGURE
        # ================================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ----------------------------------------------------------------
        # PANEL 1: Original
        # ----------------------------------------------------------------
        axes[0].imshow(crop_img, cmap='gray')
        axes[0].contour(crop_mask, colors='cyan', linewidths=2)
        axes[0].set_title(f'Original - Nucleus {nucleus_id}')
        axes[0].axis('off')
        
        # ----------------------------------------------------------------
        # PANEL 2: DoG Filtered
        # ----------------------------------------------------------------
        axes[1].imshow(crop_dog, cmap='gray')
        axes[1].contour(crop_mask, colors='cyan', linewidths=2)
        axes[1].set_title('DoG Filtered')
        axes[1].axis('off')
        
        # ----------------------------------------------------------------
        # PANEL 3: Enhanced
        # ----------------------------------------------------------------
        axes[2].imshow(enhanced, cmap='hot')
        axes[2].contour(crop_mask, colors='cyan', linewidths=2)
        axes[2].set_title('Enhanced')
        axes[2].axis('off')
        
        # ================================================================
        # ADD STATISTICS TEXT BOX
        # ================================================================
        pixels = channel_image[nucleus_mask]
        stats_text = (
            f"Area: {len(pixels)} px\n"
            f"Mean: {np.mean(pixels):.3f}\n"
            f"CV: {np.std(pixels)/np.mean(pixels):.3f}"
        )
        
        # Place text box on left side of figure
        fig.text(
            0.02, 0.5,  # Position (x, y)
            stats_text, 
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat')
        )
        
        # ================================================================
        # FINALIZE
        # ================================================================
        plt.suptitle(f'Nucleus {nucleus_id} - Count visible foci', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_all_nuclei_overview(self, masks, channel_image, used_ids):
        """
        Show overview of all nuclei with ID labels.
        
        This provides a "map" of available nuclei for the user to choose from.
        Each nucleus is labeled with its ID, and already-selected nuclei are
        highlighted in green.
        
        VISUALIZATION ELEMENTS:
        -----------------------
        - Background: Grayscale fluorescence image
        - Boundaries: Cyan contours around all nuclei (faint)
        - Labels: White text for unselected, green text for selected
        - Checkmark: Selected nuclei show "ID✓" instead of just "ID"
        
        Parameters:
        -----------
        masks : numpy.ndarray
            Segmentation mask, shape (H, W)
        channel_image : numpy.ndarray
            Fluorescence image, shape (H, W)
        used_ids : set
            Set of nucleus IDs already selected
            Used to highlight selected nuclei in green
            
        Side Effects:
        -------------
        - Creates figure with single axis
        - Shows immediately (plt.show())
        - Blocks until user closes (for interactive workflow)
        
        Label Styling:
        --------------
        Unselected nuclei:
        - Color: White
        - Alpha: 0.3 (transparent)
        - Text: Just the ID number
        
        Selected nuclei:
        - Color: Lime green
        - Alpha: 1.0 (opaque)
        - Text: ID followed by checkmark (e.g., "5✓")
        
        Notes:
        ------
        - Labels placed at nucleus centroid
        - Boundaries shown very faintly (alpha=0.1)
        - Large figure (12×10) for visibility
        - Skips background (label 0)
        """
        # ================================================================
        # SETUP
        # ================================================================
        from skimage.segmentation import find_boundaries
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Show fluorescence image as background
        ax.imshow(channel_image, cmap='gray')
        
        # ================================================================
        # SHOW NUCLEUS BOUNDARIES (faint cyan)
        # ================================================================
        # Find boundaries between labeled regions
        boundaries = find_boundaries(masks, mode='outer')
        
        # Show as very faint contour
        ax.contour(
            boundaries, 
            colors='cyan', 
            linewidths=0.5, 
            alpha=0.1  # Very transparent
        )
        
        # ================================================================
        # LABEL EACH NUCLEUS
        # ================================================================
        # Loop through all nucleus IDs (skip background 0)
        for nucleus_id in np.unique(masks)[1:]:
            nucleus_mask = (masks == nucleus_id)
            y_coords, x_coords = np.where(nucleus_mask)
            
            # Skip if empty (shouldn't happen)
            if len(y_coords) == 0:
                continue
            
            # --------------------------------------------------------
            # Calculate centroid (label position)
            # --------------------------------------------------------
            cy = np.mean(y_coords)
            cx = np.mean(x_coords)
            
            # --------------------------------------------------------
            # Style based on selection status
            # --------------------------------------------------------
            if nucleus_id in self.ground_truth_nuclei:
                # Already selected - show in green
                color = 'lime'
                transparent = 1  # Opaque
                text = f"{nucleus_id}✓"  # Add checkmark
            else:
                # Not selected - show in white
                color = 'white'
                transparent = 0.3  # Transparent
                text = str(nucleus_id)  # Just ID
            
            # --------------------------------------------------------
            # Place text label
            # --------------------------------------------------------
            ax.text(
                cx, cy,  # Position at centroid
                text, 
                color=color, 
                fontsize=8,
                ha='center',   # Horizontal alignment
                va='center',   # Vertical alignment
                alpha=transparent
            )
        
        # ================================================================
        # FINALIZE
        # ================================================================
        ax.set_title('All Nuclei - Green=Selected')
        ax.axis('off')
        plt.tight_layout()
        plt.show()