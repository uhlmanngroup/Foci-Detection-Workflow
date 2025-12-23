"""
Watershed threshold configuration helper with multi-image distribution analysis.

This module provides tools for determining optimal watershed thresholds by analyzing
brightness distributions across multiple images. It helps researchers choose appropriate
thresholds that work consistently across their entire dataset.

Key features:
- Analyzes multiple images at once to account for image-to-image variability
- Applies the SAME preprocessing as the detection pipeline (DoG filter, 0-100 scaling)
- Provides statistical summaries and visualizations
- Supports both interactive configuration and standalone testing
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure, img_as_float
from typing import Dict, Tuple, List
import imageio


class WatershedConfigurator:
    """
    Helper class for configuring watershed thresholds with multi-image analysis.
    
    This class analyzes brightness distributions across multiple images to help
    determine appropriate watershed thresholds that work consistently across
    an entire dataset. It accounts for image-to-image variability and provides
    statistical guidance for threshold selection.
    
    Attributes:
    -----------
    thresholds : dict
        Dictionary to store configured thresholds (currently unused but 
        available for future expansion)
        
    Methods:
    --------
    analyze_multi_image_distribution():
        Analyze brightness across multiple images and create histogram
    print_statistics_summary():
        Print detailed statistics including per-image variability
    configure_threshold_from_multi_image():
        Interactive workflow for threshold configuration
    
    Usage:
    ------
    >>> configurator = WatershedConfigurator()
    >>> tritc_thresh, fitc_thresh = configurator.configure_threshold_from_multi_image(
    ...     tritc_paths, fitc_paths, n_images=10, folder_path=output_dir
    ... )
    """
    
    def __init__(self):
        """Initialize the configurator with empty threshold storage."""
        self.thresholds = {}
    
    def analyze_multi_image_distribution(self,
                                        image_paths: List[str],
                                        channel_name: str,
                                        n_images: int,
                                        folder_path: str,
                                        save_plot: bool = True) -> Tuple[plt.Figure, Dict]:
        """
        Analyze brightness distribution across multiple images.
        
        This function loads N images, applies the SAME preprocessing as the detection
        pipeline (DoG filter + 0-100 rescaling), combines all pixels into one
        distribution, and creates a histogram showing the combined brightness profile.
        
        The goal is to understand how brightness varies:
        - Within each image (pixel-to-pixel variation)
        - Between images (image-to-image variation)
        
        This helps choose a threshold that works for ALL images, not just one.
        
        Parameters:
        -----------
        image_paths : List[str]
            List of full paths to image files to analyze
            Example: ['path/to/img1.tif', 'path/to/img2.tif', ...]
        channel_name : str
            Name of the channel being analyzed ('TRITC' or 'FITC')
            Used for labeling plots and output files
        n_images : int
            Number of images to analyze from the list
            Using first N images (e.g., 10 out of 100 total)
            More images = better statistics but slower
        folder_path : str
            Directory path where histogram plots will be saved
            Should be a valid existing directory
        save_plot : bool, default=True
            Whether to save the histogram plot as a PNG file
            File will be named: {channel_name}_multi_image_brightness_distribution.png
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The histogram figure object (for display or further modification)
        global_stats : dict
            Comprehensive statistics dictionary containing:
            - 'n_images': Number of images analyzed
            - 'total_pixels': Total number of non-zero pixels across all images
            - 'global_mean': Mean brightness across all pixels
            - 'global_median': Median brightness across all pixels
            - 'global_std': Standard deviation of brightness
            - 'global_p10/p25/p50/p75/p90': Brightness percentiles
            - 'min_brightness': Minimum brightness value observed
            - 'max_brightness': Maximum brightness value observed
            - 'per_image_stats': List of per-image statistics for variability analysis
            
        Algorithm:
        ----------
        1. Loop through N images
        2. For each image:
           a. Load image file
           b. Convert to float (0-1 range)
           c. Apply DoG filter (same as detection pipeline)
           d. Clip negative values
           e. Rescale to 0-100 range (same as detection pipeline)
           f. Extract non-zero pixels
           g. Store pixels and compute per-image statistics
        3. Combine all pixels from all images
        4. Compute global statistics (across all images)
        5. Create combined histogram
        6. Return figure and statistics
        
        Notes:
        ------
        - Only non-zero pixels are analyzed (background excluded)
        - Preprocessing MUST match the detection pipeline exactly
        - Per-image statistics allow assessment of image-to-image variability
        """
        print(f"\n📊 Analyzing {channel_name} brightness across {n_images} images...")
        
        # ----------------------------------------------------------------
        # Initialize accumulators
        # ----------------------------------------------------------------
        all_pixels = []       # Will accumulate ALL pixels from all images
        image_stats = []      # Will store statistics for each individual image
        
        # ================================================================
        # PROCESS EACH IMAGE
        # ================================================================
        # Loop through first N images in the list
        for idx in range(min(n_images, len(image_paths))):
            # Print progress (overwrites same line with \r)
            print(f"   Loading image {idx+1}/{n_images}...", end='\r')
            
            # --------------------------------------------------------
            # Load and convert image
            # --------------------------------------------------------
            # Load raw image (typically uint16, 0-65535 range)
            img = imageio.imread(image_paths[idx])
            
            # Convert to float with 0-1 range
            # This is CRITICAL: must match what detection pipeline does
            img_float = img_as_float(img)
            
            # --------------------------------------------------------
            # Apply DoG filter (SAME as detection pipeline)
            # --------------------------------------------------------
            # Difference of Gaussians enhances foci
            # low_sigma=1, high_sigma=2: highlights 1-2 pixel features
            filtered = filters.difference_of_gaussians(img_float, low_sigma=1, high_sigma=2)
            
            # Clip negative values (DoG can produce negatives)
            # Only care about positive features (bright spots)
            filtered = np.clip(filtered, 0, None)
            
            # --------------------------------------------------------
            # Rescale to 0-100 range (SAME as detection pipeline)
            # --------------------------------------------------------
            # This is CRITICAL: watershed threshold is in 0-100 range
            # Must use ACTUAL max from filtered image for correct scaling
            max_filtered = filtered.max()
            
            if max_filtered > 0:
                # Rescale: 0 → 0, max_filtered → 100
                # This matches what detect_foci_single_channel does:
                # filtered_img = exposure.rescale_intensity(filtered_img, 
                #                   in_range='image', out_range=(0, isolated_img.max()))
                # Then later: filtered_img rescaled to 0-100 before watershed
                filtered = exposure.rescale_intensity(
                    filtered, 
                    in_range=(0, max_filtered), 
                    out_range=(0, 100)
                )
            else:
                # Edge case: if image is completely black, keep as zeros
                # Shouldn't happen with real data but prevents crashes
                filtered = np.zeros_like(filtered)
            
            # --------------------------------------------------------
            # Extract non-zero pixels (ignore background)
            # --------------------------------------------------------
            # Flatten to 1D array for easier processing
            pixels = filtered.flatten()
            
            # Keep only positive pixels (background is typically 0)
            pixels = pixels[pixels > 0]
            
            # --------------------------------------------------------
            # Accumulate pixels and compute per-image statistics
            # --------------------------------------------------------
            if len(pixels) > 0:
                # Add this image's pixels to global accumulator
                all_pixels.extend(pixels)
                
                # Store per-image statistics for variability analysis
                # These statistics help identify:
                # - Outlier images (very different from others)
                # - Consistent vs variable datasets
                # - Whether a single threshold will work for all images
                image_stats.append({
                    'index': idx,                                    # Image number
                    'filename': image_paths[idx].split('\\')[-1],    # Filename only (no path)
                    'mean': np.mean(pixels),                         # Average brightness
                    'median': np.median(pixels),                     # Median brightness (robust)
                    'std': np.std(pixels),                           # Variability within image
                    'p10': np.percentile(pixels, 10),                # 10th percentile
                    'p25': np.percentile(pixels, 25),                # 25th percentile (Q1)
                    'p50': np.percentile(pixels, 50),                # 50th percentile (median)
                    'p75': np.percentile(pixels, 75),                # 75th percentile (Q3)
                    'p90': np.percentile(pixels, 90)                 # 90th percentile
                })
    
        # Clear progress message
        print(f"   ✓ Loaded {len(image_stats)} images successfully" + " "*20)
        
        # ----------------------------------------------------------------
        # Convert accumulated pixels to numpy array
        # ----------------------------------------------------------------
        # all_pixels is a Python list - convert to numpy for efficient computation
        all_pixels = np.array(all_pixels)
        
        # ================================================================
        # CALCULATE GLOBAL STATISTICS
        # ================================================================
        # These are computed across ALL pixels from ALL images combined
        # This gives us the overall distribution characteristics
        global_stats = {
            # Image counts
            'n_images': len(image_stats),        # How many images analyzed
            'total_pixels': len(all_pixels),     # Total non-zero pixels
            
            # Central tendency (combined across all images)
            'global_mean': np.mean(all_pixels),      # Mean brightness
            'global_median': np.median(all_pixels),  # Median brightness (more robust)
            'global_std': np.std(all_pixels),        # Standard deviation
            
            # Percentiles (combined across all images)
            # These are KEY for threshold selection
            'global_p10': np.percentile(all_pixels, 10),   # 10% of pixels below this
            'global_p25': np.percentile(all_pixels, 25),   # 25% below (Q1)
            'global_p50': np.percentile(all_pixels, 50),   # 50% below (median)
            'global_p75': np.percentile(all_pixels, 75),   # 75% below (Q3)
            'global_p90': np.percentile(all_pixels, 90),   # 90% below
            
            # Range
            'min_brightness': np.min(all_pixels),    # Dimmest pixel
            'max_brightness': np.max(all_pixels),    # Brightest pixel
            
            # Per-image data (for variability analysis)
            'per_image_stats': image_stats
        }
        
        # ================================================================
        # CREATE HISTOGRAM VISUALIZATION
        # ================================================================
        # Generate single combined histogram showing all images together
        fig = self._create_single_histogram(all_pixels, global_stats, channel_name)
        
        # ----------------------------------------------------------------
        # Save plot if requested
        # ----------------------------------------------------------------
        if save_plot:
            # Construct filename with channel name
            plot_path = f"{folder_path}/{channel_name}_multi_image_brightness_distribution.png"
            
            # Save at 150 DPI (good balance of quality vs file size)
            # bbox_inches='tight' removes extra whitespace
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            print(f"   ✅ Saved combined plot: {plot_path}")
        
        # Return both figure (for display) and statistics (for analysis)
        return fig, global_stats
    
    def _create_single_histogram(self, all_pixels, global_stats, channel_name):
        """
        Create a single histogram plot showing combined brightness distribution.
        
        This is the SAME style as the original single-image analyzer, but now
        showing combined data from multiple images. The histogram helps visualize:
        - Overall brightness distribution
        - Where most pixels fall (mode)
        - Spread of the distribution (variance)
        - Appropriate threshold range (suggested region)
        
        Parameters:
        -----------
        all_pixels : ndarray
            1D array of all non-zero pixel values from all images combined
            Already preprocessed (DoG filtered, 0-100 scaled)
        global_stats : dict
            Statistics computed from all_pixels (mean, median, percentiles, etc.)
        channel_name : str
            Channel being analyzed ('TRITC' or 'FITC') for plot title
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The histogram figure object
            
        Visual Elements:
        ----------------
        - Blue line/area: Pixel count distribution (histogram)
        - Green dashed line: Median brightness
        - Red dashed line: Mean brightness
        - Orange shaded region: 25th-75th percentile (interquartile range)
        - Green shaded region: Suggested threshold range
        - Logarithmic Y-axis: Better visibility of full range
        - Grid lines: Help read values accurately
        """
        # ----------------------------------------------------------------
        # Create figure
        # ----------------------------------------------------------------
        fig = plt.figure(figsize=(12, 6))
        
        # ----------------------------------------------------------------
        # Create histogram with 100 bins (one per percentile unit)
        # ----------------------------------------------------------------
        # bins=100: One bin per brightness value (0-100 range)
        # This gives fine-grained resolution for threshold selection
        counts, bin_edges = np.histogram(all_pixels, bins=100, range=(0, 100))
        
        # Compute bin centers for plotting (mid-point of each bin)
        # bin_edges has 101 values (boundaries), bin_centers has 100 (midpoints)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # ----------------------------------------------------------------
        # Plot histogram as smooth line with filled area
        # ----------------------------------------------------------------
        # Plot counts vs brightness (line plot, not bar chart)
        # linewidth=2: Thick line for visibility
        # color='steelblue': Professional blue color
        plt.plot(bin_centers, counts, linewidth=2, color='steelblue', 
                label='Pixel distribution')
        
        # Fill area under curve for better visualization
        # alpha=0.3: 30% opacity (semi-transparent)
        plt.fill_between(bin_centers, counts, alpha=0.3, color='steelblue')
        
        # ----------------------------------------------------------------
        # Add statistical markers (vertical lines)
        # ----------------------------------------------------------------
        # Median line (green dashed)
        # Median is more robust than mean (not affected by outliers)
        plt.axvline(global_stats['global_median'], color='green', linestyle='--', 
                   linewidth=2, label=f"Median: {global_stats['global_median']:.1f}")
        
        # Mean line (red dashed)
        # Mean can be pulled by outliers but still informative
        plt.axvline(global_stats['global_mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {global_stats['global_mean']:.1f}")
        
        # ----------------------------------------------------------------
        # Add interquartile range (IQR) shading
        # ----------------------------------------------------------------
        # Orange shaded region: 25th to 75th percentile
        # This is where the "middle 50%" of pixels fall
        # Helps identify the main concentration of the distribution
        plt.axvspan(global_stats['global_p25'], global_stats['global_p75'], 
                   alpha=0.1, color='orange', label='25th-75th percentile')
        
        
        # ----------------------------------------------------------------
        # Configure axes and labels
        # ----------------------------------------------------------------
        # X-axis label
        plt.xlabel('Brightness Percentile (0-100)', fontsize=12)
        
        # Y-axis label
        plt.ylabel('Number of Pixels', fontsize=12)
        
        # Title with number of images analyzed
        plt.title(f'{channel_name} Brightness Distribution | Combined from {global_stats["n_images"]} Images', 
                 fontsize=14, fontweight='bold')
        
        # Set x-axis range (0 to 100)
        plt.xlim(0, 100)
        
        # ----------------------------------------------------------------
        # X-axis tick marks and grid
        # ----------------------------------------------------------------
        # X-axis labels every 2 units: 0, 2, 4, 6, ..., 98, 100
        # This gives fine resolution for reading threshold values
        plt.xticks(np.arange(0, 101, step=2))
        
        # Vertical grid lines at every x-axis tick (every 2 units)
        # Helps read exact threshold values from the plot
        # alpha=0.3: Light gray, not too distracting
        plt.grid(axis='x', which='major', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # ----------------------------------------------------------------
        # Y-axis configuration
        # ----------------------------------------------------------------
        # Logarithmic scale for Y-axis
        # This is IMPORTANT because pixel counts vary by orders of magnitude:
        # - Many pixels at common brightness values (thousands)
        # - Few pixels at extreme values (tens)
        # Log scale makes both visible
        plt.yscale('log')
        
        # Y-axis grid (both major and minor ticks)
        # which='both': Shows grid at 1, 10, 100, 1000, ... AND intermediate values
        plt.grid(axis='y', alpha=0.3, which='both')
        
        # ----------------------------------------------------------------
        # Add legend and finalize
        # ----------------------------------------------------------------
        # Legend shows all the markers and regions
        # loc='upper right': Place in top-right corner (usually least crowded)
        plt.legend(loc='upper right', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    
    def print_statistics_summary(self, global_stats: Dict, channel_name: str):
        """
        Print comprehensive statistics including per-image variability.
        
        This function provides a detailed text summary of the brightness analysis,
        helping researchers understand:
        1. Overall brightness characteristics (combined across images)
        2. Image-to-image variability (consistency check)
        3. Recommended threshold range
        
        The summary helps answer critical questions:
        - Are my images consistent enough for a single threshold?
        - What's a reasonable threshold range for my data?
        - Are there outlier images I should exclude?
        
        Parameters:
        -----------
        global_stats : dict
            Statistics dictionary from analyze_multi_image_distribution()
            Must contain keys: global_mean, global_median, per_image_stats, etc.
        channel_name : str
            Name of the channel ('TRITC' or 'FITC') for labeling
            
        Returns:
        --------
        None
            Prints formatted statistics to console
            
        Output Sections:
        ----------------
        1. COMBINED STATISTICS: Overall distribution characteristics
        2. PER-IMAGE VARIABILITY: Consistency analysis across images
        """
        # ================================================================
        # HEADER
        # ================================================================
        print(f"\n" + "="*70)
        print(f"📈 {channel_name} BRIGHTNESS STATISTICS SUMMARY")
        print("="*70)
        
        # ================================================================
        # SECTION 1: COMBINED STATISTICS (GLOBAL)
        # ================================================================
        # These are computed from ALL pixels from ALL images combined
        print(f"\n🌍 COMBINED STATISTICS (All {global_stats['n_images']} images):")
        
        # Total pixel count (across all images)
        print(f"   Total pixels analyzed: {global_stats['total_pixels']:,}")
        print(f"   ")  # Blank line for readability
        
        # Central tendency measures
        # Mean: Average brightness (affected by outliers)
        # Median: Middle value (robust to outliers)
        # Std: Spread of the distribution
        print(f"   Mean brightness:   {global_stats['global_mean']:.2f}")
        print(f"   Median brightness: {global_stats['global_median']:.2f}")
        print(f"   Std deviation:     {global_stats['global_std']:.2f}")
        print(f"   ")
        
        # Brightness range (min to max)
        # Helps identify if there are extreme outliers
        print(f"   Brightness range:  {global_stats['min_brightness']:.2f} - {global_stats['max_brightness']:.2f}")
        
        # ----------------------------------------------------------------
        # Percentiles (key for threshold selection)
        # ----------------------------------------------------------------
        print(f"\n   Percentiles:")
        print(f"      10th: {global_stats['global_p10']:.2f}")   # 10% of pixels below this
        print(f"      25th: {global_stats['global_p25']:.2f}")   # Q1 (lower quartile)
        print(f"      50th: {global_stats['global_p50']:.2f}")   # Median
        print(f"      75th: {global_stats['global_p75']:.2f}")   # Q3 (upper quartile)
        print(f"      90th: {global_stats['global_p90']:.2f}")   # 90% below
        
        # ================================================================
        # SECTION 2: PER-IMAGE VARIABILITY ANALYSIS
        # ================================================================
        # This is CRITICAL: if images vary a lot, a single threshold won't work well
        print(f"\n📊 PER-IMAGE VARIABILITY:")
        
        # Extract per-image statistics
        per_image = global_stats['per_image_stats']
        
        # Get median and mean for each image
        medians = [s['median'] for s in per_image]
        means = [s['mean'] for s in per_image]
        
        # ----------------------------------------------------------------
        # Compute variability metrics
        # ----------------------------------------------------------------
        # Range: difference between brightest and dimmest image
        median_range = max(medians) - min(medians)
        mean_range = max(means) - min(means)
        
        # Coefficient of Variation (CV): normalized measure of spread
        # CV = (std / mean) × 100%
        # CV < 10%: Very consistent
        # CV 10-20%: Moderately consistent
        # CV > 20%: High variability (problematic)
        median_cv = (np.std(medians) / np.mean(medians)) * 100
        
        print(f"   ")
        print(f"   Individual image medians:")
        print(f"      Min: {min(medians):.2f}")      # Dimmest image
        print(f"      Max: {max(medians):.2f}")      # Brightest image
        print(f"      Range: {median_range:.2f}")    # Spread
        print(f"      Coefficient of Variation: {median_cv:.1f}%")
        
        # ----------------------------------------------------------------
        # Interpret variability (provide guidance)
        # ----------------------------------------------------------------
        if median_cv < 10:
            # Very consistent: single threshold will work well
            print(f"      → Very consistent across images ✓")
        elif median_cv < 20:
            # Moderately consistent: single threshold acceptable
            print(f"      → Moderately consistent across images")
        else:
            # High variability: may need per-image thresholds or data QC
            print(f"      → High variability - consider checking image quality")
        
        # ----------------------------------------------------------------
        # Per-image breakdown (detailed view)
        # ----------------------------------------------------------------
        print(f"   ")
        print(f"   Per-image breakdown:")
        for s in per_image:
            # For each image, show:
            # - Median: typical brightness
            # - IQR: interquartile range (spread within image)
            # - Std: standard deviation (variability within image)
            print(f"      Image {s['index']+1}: median={s['median']:.1f}, "
                  f"IQR={s['p75']-s['p25']:.1f}, std={s['std']:.1f}")
        

        
        print("="*70)
    
    def configure_threshold_from_multi_image(self,
                                           tritc_paths: List[str],
                                           fitc_paths: List[str],
                                           n_images: int,
                                           folder_path: str) -> Tuple[float, float]:
        """
        Configure watershed thresholds by analyzing multiple images at once.
        
        This is the MAIN INTERACTIVE WORKFLOW for threshold configuration.
        It guides the user through:
        1. Analyzing sample images from the dataset
        2. Viewing brightness distributions and statistics
        3. Selecting appropriate thresholds for TRITC and FITC
        4. Applying those thresholds to the entire dataset
        
        WHY analyze multiple images?
        - Single image might be atypical (too bright, too dim, artifacts)
        - Multiple images reveal true variability in the dataset
        - Chosen threshold must work for ALL images, not just one
        
        Workflow:
        ---------
        1. Load first N images (user-specified sample size)
        2. Apply SAME preprocessing as detection pipeline
           - DoG filter (sigma 1→2)
           - Rescale to 0-100 range
        3. Combine all pixels into one histogram
        4. Show statistics + per-image variability
        5. Display histogram with suggested range
        6. Get user input for threshold
        7. Repeat for second channel
        8. Return both thresholds for use in full dataset processing
        
        Parameters:
        -----------
        tritc_paths : List[str]
            List of all TRITC image file paths in the dataset
            Only first n_images will be analyzed, but user is told total count
        fitc_paths : List[str]
            List of all FITC image file paths in the dataset
        n_images : int
            Number of sample images to analyze (typically 5-20)
            More images = better statistics but slower
            Recommendation: 10 images is usually sufficient
        folder_path : str
            Directory where histogram plots will be saved
            Must be an existing writable directory
            
        Returns:
        --------
        tuple : (tritc_threshold, fitc_threshold)
            tritc_threshold : float
                User-selected watershed threshold for TRITC (0-100)
            fitc_threshold : float
                User-selected watershed threshold for FITC (0-100)
            These will be applied to ALL images in the dataset
            
        Example:
        --------
        >>> configurator = WatershedConfigurator()
        >>> tritc_thresh, fitc_thresh = configurator.configure_threshold_from_multi_image(
        ...     tritc_paths=all_tritc_files,
        ...     fitc_paths=all_fitc_files,
        ...     n_images=10,
        ...     folder_path=r"C:\output"
        ... )
        >>> print(f"Will use TRITC={tritc_thresh}, FITC={fitc_thresh} for all images")
        """
        # ================================================================
        # INTRODUCTION BANNER
        # ================================================================
        print("\n" + "="*70)
        print("🎨 MULTI-IMAGE WATERSHED THRESHOLD CONFIGURATION")
        print("="*70)
        
        # Explain what's happening
        print(f"Analyzing first {n_images} images to determine optimal thresholds")
        print(f"Using SAME preprocessing as detection pipeline:")
        print(f"  - Difference of Gaussians filter (sigma 1→2)")
        print(f"  - Rescale intensity to 0-100 range")
        print("="*70 + "\n")
        
        # ================================================================
        # TRITC CHANNEL ANALYSIS
        # ================================================================
        print("🔴 TRITC Channel Analysis")
        print("-" * 70)
        
        # Analyze TRITC brightness distribution
        fig_tritc, stats_tritc = self.analyze_multi_image_distribution(
            image_paths=tritc_paths,       # All TRITC image paths
            channel_name='TRITC',          # Label for plots
            n_images=n_images,             # How many to analyze
            folder_path=folder_path,       # Where to save plot
            save_plot=True                 # Save histogram PNG
        )
        
        # Print detailed statistics
        self.print_statistics_summary(stats_tritc, 'TRITC')
        
        # Display histogram (blocks until user closes window)
        plt.show()
        
        # ----------------------------------------------------------------
        # Get user input for TRITC threshold
        # ----------------------------------------------------------------
        print("\n" + "="*70)
        tritc_threshold = float(input("🎯 Enter watershed threshold for TRITC (e.g., 26): "))
        print("="*70)
        
        # Close the figure to free memory
        plt.close(fig_tritc)
        
        # ================================================================
        # FITC CHANNEL ANALYSIS
        # ================================================================
        # Same process as TRITC but for FITC channel
        print("\n🟢 FITC Channel Analysis")
        print("-" * 70)
        
        # Analyze FITC brightness distribution
        fig_fitc, stats_fitc = self.analyze_multi_image_distribution(
            image_paths=fitc_paths,        # All FITC image paths
            channel_name='FITC',           # Label for plots
            n_images=n_images,             # How many to analyze
            folder_path=folder_path,       # Where to save plot
            save_plot=True                 # Save histogram PNG
        )
        
        # Print detailed statistics
        self.print_statistics_summary(stats_fitc, 'FITC')
        
        # Display histogram
        plt.show()
        
        # ----------------------------------------------------------------
        # Get user input for FITC threshold
        # ----------------------------------------------------------------
        print("\n" + "="*70)
        fitc_threshold = float(input("🎯 Enter watershed threshold for FITC (e.g., 26): "))
        print("="*70)
        
        # Close the figure
        plt.close(fig_fitc)
        
        # ================================================================
        # CONFIGURATION COMPLETE - SUMMARY
        # ================================================================
        print("\n" + "="*70)
        print("✅ THRESHOLD CONFIGURATION COMPLETE")
        print("="*70)
        
        # Show selected thresholds
        print(f"   TRITC threshold: {tritc_threshold}")
        print(f"   FITC threshold:  {fitc_threshold}")
        
        # Inform user these will be applied to ALL images
        print(f"\n💡 These thresholds will be applied to ALL images in the dataset")
        
        # ----------------------------------------------------------------
        # Provide config.yaml template for future runs
        # ----------------------------------------------------------------
        # This allows user to skip interactive configuration next time
        # by saving these values in the config file
        print("💡 To save these for future runs, add to your config.yaml:")
        print(f"\n   watershed:")
        print(f"     mode: \"manual_preset\"")
        print(f"     manual_threshold_tritc: {tritc_threshold}")
        print(f"     manual_threshold_fitc: {fitc_threshold}")
        print(f"     skip_interactive_config: true")
        print("="*70 + "\n")
        
        # Return both thresholds as tuple
        return tritc_threshold, fitc_threshold


# ============================================================
# STANDALONE TESTING FUNCTION
# ============================================================
def test_multi_image_distribution(tritc_paths, fitc_paths, n_images, output_folder):
    """
    Test function to preview multi-image distributions without full pipeline run.
    
    This is a CONVENIENCE FUNCTION for testing the configurator without running
    the full detection pipeline. Use this to:
    - Preview brightness distributions
    - Test different sample sizes (n_images)
    - Generate histogram plots for documentation
    - Check image consistency before running full analysis
    
    Unlike configure_threshold_from_multi_image(), this function:
    - Does NOT ask for user input
    - Does NOT return threshold values
    - Only shows plots and statistics
    
    Usage Example:
    --------------
```python
    # At Python console or Jupyter notebook
    from watershed_configurator import test_multi_image_distribution
    
    # Define image paths (typically loaded from file browser or glob)
    tritc_files = [...]  # List of TRITC image paths
    fitc_files = [...]   # List of FITC image paths
    
    # Run test
    test_multi_image_distribution(
        tritc_paths=tritc_files,
        fitc_paths=fitc_files,
        n_images=5,                           # Analyze first 5 images
        output_folder=r"Y:\path\to\output"    # Where to save plots
    )
```
    
    Parameters:
    -----------
    tritc_paths : list
        List of TRITC image file paths
    fitc_paths : list
        List of FITC image file paths
    n_images : int
        Number of images to analyze
    output_folder : str
        Directory where histogram plots will be saved
        
    Returns:
    --------
    None
        Displays plots and prints statistics, but does not return values
        
    Notes:
    ------
    - Plots are displayed with plt.show() - must close manually to continue
    - Histogram PNGs are saved to output_folder
    - Statistics are printed to console
    - Does NOT configure anything - just for preview/testing
    """
    # Create configurator instance
    configurator = WatershedConfigurator()
    
    # Print test mode banner
    print("\n🧪 TEST MODE: Multi-Image Distribution Analysis")
    print("   (No thresholds will be set, only visualization)\n")
    
    # ================================================================
    # TRITC ANALYSIS
    # ================================================================
    # Analyze and display TRITC brightness distribution
    fig_tritc, stats_tritc = configurator.analyze_multi_image_distribution(
        image_paths=tritc_paths,
        channel_name='TRITC',
        n_images=n_images,
        folder_path=output_folder,
        save_plot=True
    )
    
    # Print statistics
    configurator.print_statistics_summary(stats_tritc, 'TRITC')
    
    # Display plot (blocks until closed)
    plt.show()
    
    # ================================================================
    # FITC ANALYSIS
    # ================================================================
    # Same for FITC channel
    fig_fitc, stats_fitc = configurator.analyze_multi_image_distribution(
        image_paths=fitc_paths,
        channel_name='FITC',
        n_images=n_images,
        folder_path=output_folder,
        save_plot=True
    )
    
    # Print statistics
    configurator.print_statistics_summary(stats_fitc, 'FITC')
    
    # Display plot
    plt.show()
    
    # Completion message
    print("\n✅ Test complete. Check output folder for saved plots.")