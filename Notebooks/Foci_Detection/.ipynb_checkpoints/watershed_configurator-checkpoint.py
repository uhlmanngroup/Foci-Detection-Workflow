"""
Watershed threshold configuration helper with multi-image distribution analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure, img_as_float
from typing import Dict, Tuple, List
import imageio


class WatershedConfigurator:
    """Helper class for configuring watershed thresholds with multi-image analysis."""
    
    def __init__(self):
        self.thresholds = {}
    
    def analyze_multi_image_distribution(self,
                                        image_paths: List[str],
                                        channel_name: str,
                                        n_images: int,
                                        folder_path: str,
                                        save_plot: bool = True) -> Tuple[plt.Figure, Dict]:
        """
        Analyze brightness distribution across multiple images.
        """
        print(f"\n📊 Analyzing {channel_name} brightness across {n_images} images...")
        
        all_pixels = []
        image_stats = []
        
        # Process each image
        for idx in range(min(n_images, len(image_paths))):
            print(f"   Loading image {idx+1}/{n_images}...", end='\r')
            
            # Load image
            img = imageio.imread(image_paths[idx])
            
            # Convert to float (0-1 range)
            img_float = img_as_float(img)
            
            # Apply DoG filter (same as in detection pipeline)
            filtered = filters.difference_of_gaussians(img_float, low_sigma=1, high_sigma=2)
            
            # Clip negative values
            filtered = np.clip(filtered, 0, None)
            
            # ============================================================
            # Rescale using ACTUAL maximum from filtered image
            # ============================================================
            max_filtered = filtered.max()
            
            if max_filtered > 0:
                # Rescale: 0 → 0, max_filtered → 100
                filtered = exposure.rescale_intensity(
                    filtered, 
                    in_range=(0, max_filtered), 
                    out_range=(0, 100)
                )
            else:
                # Edge case: if image is completely black, keep as is
                filtered = np.zeros_like(filtered)
            
            # Extract non-zero pixels (ignore background)
            pixels = filtered.flatten()
            pixels = pixels[pixels > 0]
            
            if len(pixels) > 0:
                all_pixels.extend(pixels)
                
                # Store per-image statistics
                image_stats.append({
                    'index': idx,
                    'filename': image_paths[idx].split('\\')[-1],
                    'mean': np.mean(pixels),
                    'median': np.median(pixels),
                    'std': np.std(pixels),
                    'p10': np.percentile(pixels, 10),
                    'p25': np.percentile(pixels, 25),
                    'p50': np.percentile(pixels, 50),
                    'p75': np.percentile(pixels, 75),
                    'p90': np.percentile(pixels, 90)
                })
    
        
        print(f"   ✓ Loaded {len(image_stats)} images successfully" + " "*20)
        
        # Convert to numpy array
        all_pixels = np.array(all_pixels)
        
        # Calculate global statistics (combined across all images)
        global_stats = {
            'n_images': len(image_stats),
            'total_pixels': len(all_pixels),
            'global_mean': np.mean(all_pixels),
            'global_median': np.median(all_pixels),
            'global_std': np.std(all_pixels),
            'global_p10': np.percentile(all_pixels, 10),
            'global_p25': np.percentile(all_pixels, 25),
            'global_p50': np.percentile(all_pixels, 50),
            'global_p75': np.percentile(all_pixels, 75),
            'global_p90': np.percentile(all_pixels, 90),
            'min_brightness': np.min(all_pixels),
            'max_brightness': np.max(all_pixels),
            'per_image_stats': image_stats  # Store for variability analysis
        }
        
        # Create single histogram plot (like your original)
        fig = self._create_single_histogram(all_pixels, global_stats, channel_name)
        
        # Save plot if requested
        if save_plot:
            plot_path = f"{folder_path}/{channel_name}_multi_image_brightness_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved combined plot: {plot_path}")
        
        return fig, global_stats
    
    def _create_single_histogram(self, all_pixels, global_stats, channel_name):
            """
            Create a SINGLE histogram plot (like your original analyze_brightness_percentiles).
            
            This is the SAME style as before, just with combined pixels from multiple images.
            """
            fig = plt.figure(figsize=(12, 6))
            
            # Create histogram with 100 bins (one per percentile)
            counts, bin_edges = np.histogram(all_pixels, bins=100, range=(0, 100))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot with smooth line (SAME as your original)
            plt.plot(bin_centers, counts, linewidth=2, color='steelblue', label='Pixel distribution')
            plt.fill_between(bin_centers, counts, alpha=0.3, color='steelblue')
            
            # Add statistical markers
            plt.axvline(global_stats['global_median'], color='green', linestyle='--', 
                       linewidth=2, label=f"Median: {global_stats['global_median']:.1f}")
            plt.axvline(global_stats['global_mean'], color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {global_stats['global_mean']:.1f}")
            
            # Add quartile shading
            plt.axvspan(global_stats['global_p25'], global_stats['global_p75'], 
                       alpha=0.1, color='orange', label='25th-75th percentile')
            
            # Suggest typical threshold range
            suggested_min = max(global_stats['global_p25'], 15)
            suggested_max = min(global_stats['global_p75'], 50)
            plt.axvspan(suggested_min, suggested_max, 
                       alpha=0.15, color='green', 
                       label=f'Suggested: {suggested_min:.0f}-{suggested_max:.0f}')
            
            # ============================================================
            # X-AXIS CONFIGURATION (UPDATED)
            # ============================================================
            plt.xlabel('Brightness Percentile (0-100)', fontsize=12)
            plt.ylabel('Number of Pixels', fontsize=12)
            plt.title(f'{channel_name} Brightness Distribution | Combined from {global_stats["n_images"]} Images', 
                     fontsize=14, fontweight='bold')
            plt.xlim(0, 100)
            
            # X-axis labels every 2 units: 0, 2, 4, 6, ..., 98, 100
            plt.xticks(np.arange(0, 101, step=2))
            
            # Vertical grid lines at every x-axis tick (every 2 units)
            plt.grid(axis='x', which='major', alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Y-axis grid (keep existing)
            plt.yscale('log')
            plt.grid(axis='y', alpha=0.3, which='both')
            
            plt.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            
            return fig
    
    def print_statistics_summary(self, global_stats: Dict, channel_name: str):
        """
        Print comprehensive statistics including per-image variability.
        """
        print(f"\n" + "="*70)
        print(f"📈 {channel_name} BRIGHTNESS STATISTICS SUMMARY")
        print("="*70)
        
        # Global statistics (combined)
        print(f"\n🌍 COMBINED STATISTICS (All {global_stats['n_images']} images):")
        print(f"   Total pixels analyzed: {global_stats['total_pixels']:,}")
        print(f"   ")
        print(f"   Mean brightness:   {global_stats['global_mean']:.2f}")
        print(f"   Median brightness: {global_stats['global_median']:.2f}")
        print(f"   Std deviation:     {global_stats['global_std']:.2f}")
        print(f"   ")
        print(f"   Brightness range:  {global_stats['min_brightness']:.2f} - {global_stats['max_brightness']:.2f}")
        
        # Percentiles
        print(f"\n   Percentiles:")
        print(f"      10th: {global_stats['global_p10']:.2f}")
        print(f"      25th: {global_stats['global_p25']:.2f}")
        print(f"      50th: {global_stats['global_p50']:.2f}")
        print(f"      75th: {global_stats['global_p75']:.2f}")
        print(f"      90th: {global_stats['global_p90']:.2f}")
        
        # Per-image variability analysis
        print(f"\n📊 PER-IMAGE VARIABILITY:")
        per_image = global_stats['per_image_stats']
        
        medians = [s['median'] for s in per_image]
        means = [s['mean'] for s in per_image]
        
        median_range = max(medians) - min(medians)
        mean_range = max(means) - min(means)
        median_cv = (np.std(medians) / np.mean(medians)) * 100  # Coefficient of variation in %
        
        print(f"   ")
        print(f"   Individual image medians:")
        print(f"      Min: {min(medians):.2f}")
        print(f"      Max: {max(medians):.2f}")
        print(f"      Range: {median_range:.2f}")
        print(f"      Coefficient of Variation: {median_cv:.1f}%")
        
        if median_cv < 10:
            print(f"      → Very consistent across images ✓")
        elif median_cv < 20:
            print(f"      → Moderately consistent across images")
        else:
            print(f"      → High variability - consider checking image quality")
        
        print(f"   ")
        print(f"   Per-image breakdown:")
        for s in per_image:
            print(f"      Image {s['index']+1}: median={s['median']:.1f}, "
                  f"IQR={s['p75']-s['p25']:.1f}, std={s['std']:.1f}")
        
        # Threshold recommendation
        suggested_min = max(global_stats['global_p25'], 15)
        suggested_max = min(global_stats['global_p75'], 50)
        
        print(f"\n💡 THRESHOLD RECOMMENDATION:")
        print(f"   Suggested range: {suggested_min:.0f} - {suggested_max:.0f}")
        print(f"   ")
        print(f"   Rationale:")
        print(f"   - Based on combined 25th-75th percentile")
        print(f"   - Clamped to reasonable range (15-50)")
        print(f"   - Accounts for variability across {global_stats['n_images']} images")
        
        print("="*70)
    
    def configure_threshold_from_multi_image(self,
                                           tritc_paths: List[str],
                                           fitc_paths: List[str],
                                           n_images: int,
                                           folder_path: str) -> Tuple[float, float]:
        """
        Configure watershed thresholds by analyzing multiple images at once.
        
        Workflow:
        1. Load first N images
        2. Apply SAME preprocessing as detection pipeline (DoG filter, 0-100 scaling)
        3. Combine all pixels into one histogram
        4. Show statistics + per-image variability
        5. Get user input ONCE
        6. Apply to entire dataset
        
        Parameters:
        -----------
        tritc_paths : list
            List of TRITC image paths
        fitc_paths : list
            List of FITC image paths
        n_images : int
            Number of images to analyze
        folder_path : str
            Path to save plots
            
        Returns:
        --------
        tuple : (tritc_threshold, fitc_threshold)
        """
        print("\n" + "="*70)
        print("🎨 MULTI-IMAGE WATERSHED THRESHOLD CONFIGURATION")
        print("="*70)
        print(f"Analyzing first {n_images} images to determine optimal thresholds")
        print(f"Using SAME preprocessing as detection pipeline:")
        print(f"  - Difference of Gaussians filter (sigma 1→2)")
        print(f"  - Rescale intensity to 0-100 range")
        print("="*70 + "\n")
        
        # ============================================================
        # TRITC CHANNEL
        # ============================================================
        print("🔴 TRITC Channel Analysis")
        print("-" * 70)
        
        fig_tritc, stats_tritc = self.analyze_multi_image_distribution(
            image_paths=tritc_paths,
            channel_name='TRITC',
            n_images=n_images,
            folder_path=folder_path,
            save_plot=True
        )
        
        self.print_statistics_summary(stats_tritc, 'TRITC')
        
        plt.show()
        
        # Get user input
        print("\n" + "="*70)
        tritc_threshold = float(input("🎯 Enter watershed threshold for TRITC (e.g., 26): "))
        print("="*70)
        
        plt.close(fig_tritc)
        
        # ============================================================
        # FITC CHANNEL
        # ============================================================
        print("\n🟢 FITC Channel Analysis")
        print("-" * 70)
        
        fig_fitc, stats_fitc = self.analyze_multi_image_distribution(
            image_paths=fitc_paths,
            channel_name='FITC',
            n_images=n_images,
            folder_path=folder_path,
            save_plot=True
        )
        
        self.print_statistics_summary(stats_fitc, 'FITC')
        
        plt.show()
        
        # Get user input
        print("\n" + "="*70)
        fitc_threshold = float(input("🎯 Enter watershed threshold for FITC (e.g., 26): "))
        print("="*70)
        
        plt.close(fig_fitc)
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "="*70)
        print("✅ THRESHOLD CONFIGURATION COMPLETE")
        print("="*70)
        print(f"   TRITC threshold: {tritc_threshold}")
        print(f"   FITC threshold:  {fitc_threshold}")
        print("\n💡 These thresholds will be applied to ALL {0} images in the dataset")
        print("💡 To save these for future runs, add to your config.yaml:")
        print(f"\n   watershed:")
        print(f"     mode: \"manual_preset\"")
        print(f"     manual_threshold_tritc: {tritc_threshold}")
        print(f"     manual_threshold_fitc: {fitc_threshold}")
        print(f"     skip_interactive_config: true")
        print("="*70 + "\n")
        
        return tritc_threshold, fitc_threshold


# ============================================================
# STANDALONE TESTING FUNCTION
# ============================================================
def test_multi_image_distribution(tritc_paths, fitc_paths, n_images, output_folder):
    """
    Test function to preview multi-image distributions without full pipeline run.
    
    Usage:
    ------
    from watershed_configurator import test_multi_image_distribution
    
    test_multi_image_distribution(
        tritc_paths=TRITC_data,
        fitc_paths=FITC_data,
        n_images=5,
        output_folder=r"Y:\path\to\output"
    )
    """
    configurator = WatershedConfigurator()
    
    print("\n🧪 TEST MODE: Multi-Image Distribution Analysis")
    print("   (No thresholds will be set, only visualization)\n")
    
    # TRITC
    fig_tritc, stats_tritc = configurator.analyze_multi_image_distribution(
        image_paths=tritc_paths,
        channel_name='TRITC',
        n_images=n_images,
        folder_path=output_folder,
        save_plot=True
    )
    configurator.print_statistics_summary(stats_tritc, 'TRITC')
    plt.show()
    
    # FITC
    fig_fitc, stats_fitc = configurator.analyze_multi_image_distribution(
        image_paths=fitc_paths,
        channel_name='FITC',
        n_images=n_images,
        folder_path=output_folder,
        save_plot=True
    )
    configurator.print_statistics_summary(stats_fitc, 'FITC')
    plt.show()
    
    print("\n✅ Test complete. Check output folder for saved plots.")