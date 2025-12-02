"""
Complete Parameter Space Generator with Full Foci Detection
Includes all visualizations and actual detection algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity
import pandas as pd
import pickle
import os
from skimage import exposure, filters, measure, img_as_float
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D



from nucleus_worker_Visualization import (
    compute_adaptive_background_texture_nucleus_fallback,
    apply_foci_filters
)



class ParameterSpaceGenerator:
    """
    Full implementation with actual foci detection and comprehensive visualizations.
    """
    
    def __init__(self, param_ranges, resolution=20):
        self.param_ranges = param_ranges
        self.resolution = resolution
        self.ground_truth_nuclei = {}
        self.grid_results = None
        self.valid_points = None
        self.kde_model = None
        self.hull = None
        self.bounds = None
        self.kde_metadata = None
        
    def add_nucleus(self, cell_id, min_count, max_count):
        """
        Add a nucleus with its expected foci count range.
        
        Parameters:
        -----------
        cell_id : int
            Nucleus identifier
        min_count : int
            Minimum acceptable foci count (inclusive)
        max_count : int
            Maximum acceptable foci count (inclusive)
        """
        if max_count < min_count:
            raise ValueError(f"max_count ({max_count}) must be >= min_count ({min_count})")
        
        self.ground_truth_nuclei[cell_id] = (min_count, max_count)
        
        if min_count == max_count:
            print(f"✓ Added nucleus {cell_id} with exact count: {min_count}")
        else:
            print(f"✓ Added nucleus {cell_id} with acceptable range: {min_count}-{max_count} foci")
    
    def interactive_nucleus_selection(self, masks, channel_image, num_nuclei=10):
        """
        Interactively select and annotate nuclei with visual feedback.
        Streamlined workflow with minimal scrolling.
        """
        from skimage.segmentation import find_boundaries
        
        print("\n" + "="*60)
        print("INTERACTIVE VISUAL NUCLEUS SELECTION")
        print("="*60)
        print(f"\nGoal: Select and annotate up to {num_nuclei} nuclei")
        print("\nWorkflow:")
        print("  1. View full overview with all nucleus IDs")
        print("  2. Select a nucleus by entering its ID")
        print("  3. View detailed zoom of that nucleus")
        print("  4. Enter foci count or choose different nucleus")
        print("="*60)
        
        channel_image = img_as_float(channel_image)
        
        # Get all valid nucleus IDs
        all_nucleus_ids = np.unique(masks)[1:]  # Skip background
        
        selected_count = 0
        used_ids = set()
        
        # ✅ STEP 1: Show full overview at the start
        print("\n📊 Showing full overview of all nuclei...")
        self._show_all_nuclei_overview(masks, channel_image, used_ids)
        
        while selected_count < num_nuclei:
            print(f"\n[{selected_count}/{num_nuclei} nuclei selected]")
            
            # ✅ STEP 2: Choose a nucleus to examine
            while True:
                nucleus_input = input("\n🔍 Enter nucleus ID to examine (or 'generate' to show overview, 'done' to finish): ").strip()
                
                if nucleus_input.lower() == 'done':
                    if selected_count > 0:
                        print(f"\n✓ Finished with {selected_count} nuclei")
                        plt.close('all')
                        return list(self.ground_truth_nuclei.keys())
                    else:
                        print("❌ Please select at least one nucleus first!")
                        continue
                
                elif nucleus_input.lower() == 'generate':
                    print("\n📊 Regenerating full overview...")
                    plt.close('all')
                    self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    continue
                
                else:
                    try:
                        nucleus_id = int(nucleus_input)
                        if nucleus_id not in all_nucleus_ids:
                            print(f"❌ Nucleus {nucleus_id} not found. Try again or enter 'generate' to see overview.")
                            continue
                        # Valid nucleus ID - break to show detailed view
                        break
                    except ValueError:
                        print("❌ Invalid input. Enter a number, 'generate', or 'done'.")
                        continue
            
            # ✅ STEP 3: Show detailed view of selected nucleus
            plt.close('all')
            self._visualize_single_nucleus(masks, channel_image, nucleus_id)
            plt.show(block=False)
            plt.pause(0.1)
            
            # ✅ STEP 4: Count foci or abort
            while True:
                action_input = input(f"\n🎯 Nucleus {nucleus_id} - Enter foci count (e.g., '3' or '2-4' for range), 'abort', or 'abort+': ").strip()
                
                if action_input.lower() == 'abort':
                    # Abort without regenerating - go back to nucleus selection
                    print("→ Aborting to nucleus selection...")
                    plt.close('all')
                    break  # Break inner loop, stay in outer loop
                
                elif action_input.lower() == 'abort+':
                    # Abort and regenerate overview
                    print("→ Aborting and regenerating overview...")
                    plt.close('all')
                    self._show_all_nuclei_overview(masks, channel_image, used_ids)
                    break  # Break inner loop, stay in outer loop
                
                else:
                    try:
                        # ✅ NEW: Parse range format "2-4" or single value "3"
                        if '-' in action_input:
                            # Range format: "2-4"
                            parts = action_input.split('-')
                            if len(parts) != 2:
                                print("❌ Invalid range format. Use 'min-max' (e.g., '2-4')")
                                continue
                            
                            min_count = int(parts[0].strip())
                            max_count = int(parts[1].strip())
                            
                            if min_count < 0 or max_count < 0:
                                print("❌ Counts must be non-negative")
                                continue
                            
                            if max_count < min_count:
                                print("❌ Max count must be >= min count")
                                continue
                        
                        else:
                            # Single value: "3" → treat as exact count (3-3)
                            foci_count = int(action_input)
                            if foci_count < 0:
                                print("❌ Count must be non-negative")
                                continue
                            
                            min_count = foci_count
                            max_count = foci_count
                        
                        # ✅ Valid range - save it
                        self.add_nucleus(nucleus_id, min_count, max_count)
                        used_ids.add(nucleus_id)
                        selected_count += 1
                        plt.close('all')
                        
                        # Check if we're done
                        if selected_count >= num_nuclei:
                            print(f"\n✅ Completed! Selected {selected_count} nuclei.")
                            return list(self.ground_truth_nuclei.keys())
                        
                        # Not done - prompt for next nucleus without regenerating
                        print(f"✓ Saved! ({selected_count}/{num_nuclei} complete)")
                        break  # Break inner loop to select next nucleus
                        
                    except ValueError:
                        print("❌ Invalid input. Enter a number, 'abort', or 'abort+'.")
                        continue
        
        plt.close('all')
        return list(self.ground_truth_nuclei.keys())
    
    def generate_grid_search(self, masks, channel_image, original_image):
        """
        Run ACTUAL foci detection with parameter grid search.
        This is the complete implementation, not simplified.
        """
        print("\n" + "="*60)
        print("STEP 2: GRID SEARCH WITH ACTUAL FOCI DETECTION")
        print("="*60)
        
        if not self.ground_truth_nuclei:
            raise ValueError("No nuclei registered! Use add_nucleus() first.")
         
        # number of sampled values per parameter
        n_bright = 31
        n_contrast = 15
        n_percentile = 31
        
        # ✅ Step 1: Create 1D arrays of parameter values (no dtype=int!)
        bright_vals = np.linspace(
            self.param_ranges['bright_pct'][0],
            self.param_ranges['bright_pct'][1],
            n_bright,
            dtype=int
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
        
        # ✅ Step 2: Create meshgrid to generate ALL combinations
        bright_mesh, contrast_mesh, percentile_mesh = np.meshgrid(
            bright_vals, contrast_vals, percentile_vals, indexing='ij'
        )
        
        # ✅ Step 3: Flatten meshgrid to get 1D arrays of all 125 combinations
        bright_grid = bright_mesh.flatten()      # Shape: (125,)
        contrast_grid = contrast_mesh.flatten()  # Shape: (125,)
        percentile_grid = percentile_mesh.flatten()  # Shape: (125,)
        
        total_combinations = len(bright_grid)  # Now correctly 125
        print(f"\nTesting {total_combinations} parameter combinations")
        print(f"On {len(self.ground_truth_nuclei)} nuclei")
        print(f"  Bright %: {n_bright} values from {bright_vals.min():.1f} to {bright_vals.max():.1f}")
        print(f"  Contrast: {n_contrast} values from {contrast_vals.min():.2f} to {contrast_vals.max():.2f}")
        print(f"  Percentile: {n_percentile} values from {percentile_vals.min():.1f} to {percentile_vals.max():.1f}")
        print(f"\nTesting {total_combinations} parameter combinations")
        print(f"On {len(self.ground_truth_nuclei)} nuclei")
        
        results = []
        
        # Process each nucleus
        for cell_id, (min_count, max_count) in self.ground_truth_nuclei.items():
            if min_count == max_count:
                print(f"\n  Processing nucleus {cell_id} (expected: exactly {min_count} foci)...")
            else:
                print(f"\n  Processing nucleus {cell_id} (expected: {min_count}-{max_count} foci)...")
            
            nucleus_mask = (masks == cell_id)
            if not np.any(nucleus_mask):
                print(f"    ⚠️ Warning: Nucleus {cell_id} not found in mask")
                continue
            
            # Run detection for all parameter combinations
            nucleus_results = self._detect_foci_for_nucleus(
                nucleus_mask, channel_image, original_image,
                bright_grid, contrast_grid, percentile_grid,
                cell_id
            )
            
            # ✅ NEW: Add min and max counts to results
            for result in nucleus_results:
                result['min_count'] = min_count
                result['max_count'] = max_count
                results.append(result)
            
            # ✅ NEW: Show progress (count within range)
            correct = sum(1 for r in nucleus_results 
                         if min_count <= r['foci_count'] <= max_count)
            print(f"    ✓ {correct}/{len(nucleus_results)} parameters gave acceptable count")
        
        self.grid_results = pd.DataFrame(results)
        print(f"\n✓ Grid search complete: {len(self.grid_results)} total results")
        
        # Generate visualization of grid search results
        self._visualize_grid_search_results()
        
        return self.grid_results
    
    def _detect_foci_for_nucleus(self, nucleus_mask, channel_image, original_image,
                                 bright_grid, contrast_grid, percentile_grid, cell_id):
        """
        Run actual foci detection for one nucleus across all parameters.
        NOW USES THE SAME DETECTION PIPELINE AS MAIN PROGRAM.
        """
        # ✅ ADD DEBUG OUTPUT
        print(f"\n    🔍 Debug nucleus {cell_id}:")

        # ✅ FIX: Ensure original_image is also float
        original_image_float = img_as_float(original_image)
        
        # Isolate nucleus
        isolated_img = img_as_float(channel_image.copy())
        isolated_img[~nucleus_mask] = 0
        
        # ✅ CHECK 1: Is there any signal?
        print(f"      Max intensity in isolated nucleus: {isolated_img.max():.6f}")
        print(f"      Nucleus mask size: {np.sum(nucleus_mask)} pixels")
        
        if isolated_img.max() == 0:
            print(f"      ❌ EARLY EXIT: No signal in isolated nucleus")
            return []
        
        # Apply DoG filter (same as main program)
        filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
        filtered_img = np.clip(filtered_img, 0, None)
        filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', 
                                                 out_range=(0, isolated_img.max()))
        
        # Get positive pixels for percentile calculation
        pos_pixels = original_image_float[original_image_float > 0]
        
        # ✅ CHECK 2: Are there positive pixels?
        print(f"      Positive pixels in original: {pos_pixels.size}")
        
        if pos_pixels.size == 0:
            print(f"      ❌ EARLY EXIT: No positive pixels in original image")
            return []
        
        # Prepare parameter arrays
        bright_grid_arr = np.array(bright_grid)
        contrast_grid_arr = np.array(contrast_grid)
        percentile_grid_arr = np.array(percentile_grid)
        
        # Compute minimum brightness thresholds
        min_brightness_per_param = np.percentile(pos_pixels, percentile_grid_arr)
        global_min_brightness = np.min(min_brightness_per_param)
        
        # ✅ CHECK 3: What's the minimum brightness?
        print(f"      Global min brightness: {global_min_brightness:.6f}")
        print(f"      Range of min_brightness_per_param: {min_brightness_per_param.min():.6f} - {min_brightness_per_param.max():.6f}")
        
        # Find candidates (same as main program)
        candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                            threshold_abs=global_min_brightness)
        candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                              threshold_abs=global_min_brightness)
        
        # ✅ CHECK 4: Are candidates found?
        print(f"      Candidates (filtered): {len(candidates_filtered)}")
        print(f"      Candidates (unfiltered): {len(candidates_unfiltered)}")
        
        if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
            print(f"      ❌ EARLY EXIT: No candidates found")
            return []
        
        # Extract intensities
        unf_intensities = isolated_img[candidates_unfiltered[:, 0], candidates_unfiltered[:, 1]]
        filt_intensities = filtered_img[candidates_filtered[:, 0], candidates_filtered[:, 1]]
        
        # ✅ CHECK 5: What are the candidate intensities?
        print(f"      Unfiltered intensity range: {unf_intensities.min():.6f} - {unf_intensities.max():.6f}")
        print(f"      Filtered intensity range: {filt_intensities.min():.6f} - {filt_intensities.max():.6f}")
        
        # ✅ USE REAL BACKGROUND CALCULATION FROM MAIN PROGRAM
        unique_brights = np.unique(np.round(bright_grid_arr, 6))
        bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}
        
        print(f"      Computing backgrounds for {len(unique_brights)} unique brightness percentiles...")
        
        local_percentiles_unf, texture_info_unf = compute_adaptive_background_texture_nucleus_fallback(
            image=isolated_img,
            coords=candidates_unfiltered,
            unique_percentiles=unique_brights,
            nucleus_mask=nucleus_mask,
            return_texture_info=True
        )
        
        local_percentiles_filt = compute_adaptive_background_texture_nucleus_fallback(
            image=filtered_img,
            coords=candidates_filtered,
            unique_percentiles=unique_brights,
            nucleus_mask=nucleus_mask,
            return_texture_info=False  # Or just omit this parameter (False is default)
        )
        
        # ✅ CHECK 6: Nucleus texture
        contrast_multiplier = 1.0
        if texture_info_unf['nucleus_stats']:
            stats = list(texture_info_unf['nucleus_stats'].values())[0]
            nucleus_cv = stats['cv']
            print(f"      Nucleus CV: {nucleus_cv:.3f}")
            if nucleus_cv < 0.20:
                print(f"      ⚠️ Low texture - applying 1.5x contrast multiplier")
                contrast_multiplier = 1.5
                contrast_grid_arr = contrast_grid_arr * contrast_multiplier
        
        # Compute distances for matching
        distances = cdist(candidates_unfiltered, candidates_filtered)
        
        results = []
        
        # ✅ USE VECTORIZED FILTERING FROM MAIN PROGRAM
        print(f"      Testing {len(bright_grid)} parameter combinations...")
        
        for p_idx in range(len(bright_grid)):
            confirmed_coords, count = apply_foci_filters(
                p_idx, bright_grid_arr, contrast_grid_arr, percentile_grid_arr,
                min_brightness_per_param, bright_to_idx,
                unf_intensities, filt_intensities,
                local_percentiles_unf, local_percentiles_filt,
                distances, candidates_unfiltered, tolerance=2
            )
            
            results.append({
                'cell_num': cell_id,
                'bright_pct': bright_grid[p_idx],
                'contrast_thresh': contrast_grid[p_idx],
                'percentile_val': percentile_grid[p_idx],
                'foci_count': count
            })
        
        print(f"      ✅ Generated {len(results)} results")
        print(f"      Foci counts range: {min([r['foci_count'] for r in results])} - {max([r['foci_count'] for r in results])}")
        
        return results

    def find_valid_intersection(self):
        """
        Find parameter combinations valid for ALL registered nuclei.
        Now supports min-max ranges instead of exact counts.
        """
        print("\n" + "="*60)
        print("STEP 3: FINDING VALID PARAMETER INTERSECTION")
        print("="*60)
        
        if self.grid_results is None:
            raise ValueError("Must run generate_grid_search() first")
        
        # Find parameters where detected count is within acceptable range for each nucleus
        valid_params_per_nucleus = {}
        
        for cell_id, (min_count, max_count) in self.ground_truth_nuclei.items():
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            
            # ✅ NEW: Accept any count within [min_count, max_count]
            valid_for_nucleus = nucleus_data[
                (nucleus_data['foci_count'] >= min_count) & 
                (nucleus_data['foci_count'] <= max_count)
            ]
            
            valid_params = valid_for_nucleus[['bright_pct', 'contrast_thresh', 'percentile_val']].values
            valid_params_per_nucleus[cell_id] = valid_params
            
            # ✅ UPDATED: Show range instead of exact count
            if min_count == max_count:
                print(f"  Nucleus {cell_id} (exact: {min_count}): {len(valid_params)} valid combinations "
                      f"({len(valid_params)/len(nucleus_data)*100:.1f}%)")
            else:
                print(f"  Nucleus {cell_id} (range: {min_count}-{max_count}): {len(valid_params)} valid combinations "
                      f"({len(valid_params)/len(nucleus_data)*100:.1f}%)")
        
        # Find intersection - parameters valid for ALL nuclei
        if len(valid_params_per_nucleus) == 0:
            print("⚠️ No valid parameters found!")
            return None
        
        # Convert to sets of tuples for intersection
        param_sets = []
        for cell_id, params in valid_params_per_nucleus.items():
            param_tuples = set([tuple(p) for p in params])
            param_sets.append(param_tuples)
        
        # Find intersection
        intersection = param_sets[0]
        for param_set in param_sets[1:]:
            intersection = intersection.intersection(param_set)
        
        # Convert back to array
        if len(intersection) > 0:
            self.valid_points = np.array(list(intersection))
        else:
            self.valid_points = np.array([])
        
        print(f"\n✓ Found {len(self.valid_points)} parameters valid for ALL nuclei")
        
        if len(self.valid_points) == 0:
            print("\n⚠️ WARNING: No parameters work for all nuclei!")
            print("   Consider:")
            print("   - Checking ground truth annotations")
            print("   - Expanding parameter ranges")
            print("   - Using fewer or different nuclei")
        
        # Visualize the intersection
        self._visualize_valid_intersection()
        
        return self.valid_points
    
    def generate_kde_parameter_space(self, coverage_percentile=85):
        """
        Generate KDE-based parameter space from valid points.
        Full implementation with isosurface generation.
        """
        if self.valid_points is None or len(self.valid_points) < 10:
            print("⚠️ Not enough valid points for KDE")
            return None
        
        print("\n" + "="*60)
        print("STEP 4: GENERATING KDE PARAMETER SPACE")
        print("="*60)
        
        # Normalize points for KDE
        normalized_points = self._normalize_parameters(self.valid_points)
        
        # Find optimal bandwidth
        from sklearn.model_selection import GridSearchCV
        
        if len(normalized_points) >= 20:
            print("\n1. Finding optimal KDE bandwidth...")
            bandwidths = np.logspace(-2, 0, 10)
            kde_cv = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': bandwidths},
                cv=min(5, len(normalized_points) // 4)
            )
            kde_cv.fit(normalized_points)
            best_bandwidth = kde_cv.best_params_['bandwidth']
            print(f"   Optimal bandwidth: {best_bandwidth:.3f}")
        else:
            best_bandwidth = 0.1
            print(f"   Using default bandwidth: {best_bandwidth}")
        
        # Fit KDE
        print("\n2. Fitting KDE model...")
        self.kde_model = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
        self.kde_model.fit(normalized_points)
        
        # Generate isosurface points
        print("\n3. Generating isosurface...")
        n_samples = 5000
        
        # Importance sampling around valid points
        base_samples = normalized_points[np.random.choice(len(normalized_points), 
                                                         n_samples, replace=True)]
        noise = np.random.normal(0, best_bandwidth, base_samples.shape)
        samples = np.clip(base_samples + noise, 0, 1)
        
        # Add uniform samples
        uniform_samples = np.random.uniform(0, 1, (n_samples // 2, 3))
        all_samples = np.vstack([samples, uniform_samples])
        
        # Compute densities
        log_densities = self.kde_model.score_samples(all_samples)
        densities = np.exp(log_densities)
        
        # Find threshold
        valid_densities = np.exp(self.kde_model.score_samples(normalized_points))
        threshold = np.percentile(valid_densities, 100 - coverage_percentile)
        
        # Select points above threshold
        isosurface_mask = densities > threshold
        isosurface_points = all_samples[isosurface_mask]
        
        print(f"   Generated {len(isosurface_points)} isosurface points")
        print(f"   Coverage: {coverage_percentile}%")
        
        # Build Delaunay triangulation
        print("\n4. Building Delaunay triangulation...")
        if len(isosurface_points) > 0:
            # ✅ Denormalize before creating hull
            denorm_isosurface = self._denormalize_parameters(isosurface_points)
            self.hull = Delaunay(denorm_isosurface)  
        else:
            print("   ⚠️ No isosurface points, using valid points directly")
            self.hull = Delaunay(self.valid_points)  
        
        # Calculate bounds
        if len(isosurface_points) > 0:
            denorm_points = denorm_isosurface  
        else:
            denorm_points = self.valid_points
        
        self.bounds = {
            'bright_pct': (denorm_points[:, 0].min(), denorm_points[:, 0].max()),
            'contrast_thresh': (denorm_points[:, 1].min(), denorm_points[:, 1].max()),
            'percentile_val': (denorm_points[:, 2].min(), denorm_points[:, 2].max())
        }
        
        print(f"\n✓ KDE parameter space generated")
        print(f"  Bounding box:")
        for key, (min_val, max_val) in self.bounds.items():
            print(f"    {key}: {min_val:.2f} - {max_val:.2f}")
        
        # Store metadata
        self.kde_metadata = {
            'bandwidth': best_bandwidth,
            'threshold': threshold,
            'coverage_percentile': coverage_percentile,
            'isosurface_points': isosurface_points
        }
        
        # Visualize KDE and Delaunay
        self._visualize_kde_and_delaunay()
        
        return self.hull
    
    def _visualize_grid_search_results(self):
        """
        Visualize grid search results for each nucleus.
        Shows green points for correct counts, light gray for incorrect.
        """
        if self.grid_results is None:
            return
        
        for idx, (cell_id, (min_count, max_count)) in enumerate(self.ground_truth_nuclei.items()):
            fig = plt.figure(figsize=(12, 8))  # Fixed size for each
            ax = fig.add_subplot(111, projection='3d')
            
            # Get data for this nucleus
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            
            if len(nucleus_data) == 0:
                plt.close(fig)
                continue
            
            # Normalize coordinates for display
            x_norm = (nucleus_data['bright_pct'] - self.param_ranges['bright_pct'][0]) / \
                     (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            y_norm = (nucleus_data['contrast_thresh'] - self.param_ranges['contrast_thresh'][0]) / \
                     (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            z_norm = (nucleus_data['percentile_val'] - self.param_ranges['percentile_val'][0]) / \
                     (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            # Color by correctness
            correct_mask = (nucleus_data['foci_count'] >= min_count) & (nucleus_data['foci_count'] <= max_count)
            
            # Plot incorrect points (light gray, smaller)
            if (~correct_mask).any():
                ax.scatter(x_norm[~correct_mask], y_norm[~correct_mask], z_norm[~correct_mask],
                          c='lightgray', s=2, alpha=0.1, label='Incorrect')
            
            # Plot correct points (green, larger)
            if correct_mask.any():
                ax.scatter(x_norm[correct_mask], y_norm[correct_mask], z_norm[correct_mask],
                          c='green', s=10, alpha=0.8, label='Correct')
            
            # Set axis labels
            ax.set_xlabel('Background %')
            ax.set_ylabel('Contrast Thresh')
            ax.set_zlabel('Global Percentile')
            
            # Set axis limits (still 0-1 for normalized data)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            
            # ✅ CUSTOMIZE TICK LABELS TO SHOW REAL VALUES
            # X-axis: 0-100 (Background %)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0', '25', '50', '75', '100'])
            
            # Y-axis: 1-10 (Contrast Thresh)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['1.0', '3.25', '5.5', '7.75', '10.0'])
            
            # Z-axis: 0-100 (Global Percentile)
            ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_zticklabels(['0', '25', '50', '75', '100'])
            if min_count == max_count:
                ax.set_title(f'Nucleus {cell_id} - Expected: exactly {min_count} foci\n'
                            f'Green: {correct_mask.sum()} correct / Gray: {(~correct_mask).sum()} incorrect')
            else:
                ax.set_title(f'Nucleus {cell_id} - Expected: {min_count}-{max_count} foci\n'
                            f'Green: {correct_mask.sum()} acceptable / Gray: {(~correct_mask).sum()} outside range')

            ax.legend()
        
            
        plt.show()
    
    def _visualize_valid_intersection(self):
        """
        Visualize the intersection of valid parameters across all nuclei.
        """
        if self.grid_results is None:
            return
        
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Valid Parameter Intersection', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all tested points in very light gray
        all_params = self.grid_results[['bright_pct', 'contrast_thresh', 'percentile_val']].drop_duplicates()
        
        x_all = (all_params['bright_pct'] - self.param_ranges['bright_pct'][0]) / \
                (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
        y_all = (all_params['contrast_thresh'] - self.param_ranges['contrast_thresh'][0]) / \
                (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
        z_all = (all_params['percentile_val'] - self.param_ranges['percentile_val'][0]) / \
                (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
        
        ax.scatter(x_all, y_all, z_all, c='lightgray', s=2, alpha=0.1, label='All tested')
        
        # Plot valid intersection in green
        if self.valid_points is not None and len(self.valid_points) > 0:
            x_valid = (self.valid_points[:, 0] - self.param_ranges['bright_pct'][0]) / \
                     (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            y_valid = (self.valid_points[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                     (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            z_valid = (self.valid_points[:, 2] - self.param_ranges['percentile_val'][0]) / \
                     (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            ax.scatter(x_valid, y_valid, z_valid, c='green', s=10, alpha=0.8, 
                      edgecolor='darkgreen', linewidth=1, label='Valid for ALL nuclei')
        
        # Set axis labels
        ax.set_xlabel('Background %')
        ax.set_ylabel('Contrast Thresh')
        ax.set_zlabel('Global Percentile')
        
        # Set axis limits (still 0-1 for normalized data)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # ✅ CUSTOMIZE TICK LABELS TO SHOW REAL VALUES
        # X-axis: 0-100 (Background %)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '25', '50', '75', '100'])
        
        # Y-axis: 1-10 (Contrast Thresh)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['1.0', '3.25', '5.5', '7.75', '10.0'])
        
        # Z-axis: 0-100 (Global Percentile)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '25', '50', '75', '100'])
        ax.set_title(f'Parameters valid for all {len(self.ground_truth_nuclei)} nuclei\n'
                    f'Green: {len(self.valid_points) if self.valid_points is not None else 0} valid combinations')
        ax.legend()

        
        
        plt.show()
    
    def _visualize_kde_and_delaunay(self):
        """
        Visualize KDE isosurface and final Delaunay triangulation as 3D meshes.
        """
        if self.kde_metadata is None:
            return
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('KDE Isosurface and Delaunay Triangulation', fontsize=16, fontweight='bold')
        
        # ✅ NORMALIZE VALID POINTS - THEN SWAP X AND Y
        if self.valid_points is not None and len(self.valid_points) > 0:
            bright_norm = (self.valid_points[:, 0] - self.param_ranges['bright_pct'][0]) / \
                         (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
            contrast_norm = (self.valid_points[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                           (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
            percentile_norm = (self.valid_points[:, 2] - self.param_ranges['percentile_val'][0]) / \
                             (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
            
            # ✅ SWAP: X = Contrast, Y = Background (to match grid search)
            x_valid = contrast_norm      # X-axis = Contrast Thresh
            y_valid = bright_norm        # Y-axis = Background %
            z_valid = percentile_norm    # Z-axis = Percentile
        
        # ================================================================
        # LEFT PLOT: KDE ISOSURFACE
        # ================================================================
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot valid points (red dots) - NOW WITH SWAPPED AXES
        if self.valid_points is not None and len(self.valid_points) > 0:
            ax1.scatter(x_valid, y_valid, z_valid, c='red', s=3, alpha=0.1,
                       edgecolor='darkred', linewidth=1, label='Valid points', zorder=10)
        
        # KDE isosurface as mesh - SWAP COLUMNS [1,0,2] = [Contrast, Background, Percentile]
        iso_points = self.kde_metadata['isosurface_points']
        if len(iso_points) > 3:
            try:
                from scipy.spatial import ConvexHull
                
                # ✅ SWAP columns before creating hull
                iso_points_swapped = iso_points[:, [1, 0, 2]]  # [Contrast, Background, Percentile]
                hull_iso = ConvexHull(iso_points_swapped)
                
                # Create mesh collection
                faces = []
                for simplex in hull_iso.simplices:
                    faces.append(iso_points_swapped[simplex])
                
                mesh = Poly3DCollection(faces, alpha=0.2, facecolor='lightblue', 
                                       edgecolor='blue', linewidth=0.2)
                ax1.add_collection3d(mesh)
                
            except Exception as e:
                print(f"⚠️ Could not create KDE surface: {e}")
        
        # ✅ SWAPPED AXIS LABELS (X=Contrast, Y=Background)
        ax1.set_xlabel('Contrast Thresh')
        ax1.set_ylabel('Background %')
        ax1.set_zlabel('Global Percentile')
        
        # Set axis limits (still 0-1 for normalized data)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_zlim(0, 1)
        
        # ✅ SWAPPED TICK LABELS
        # X-axis: 1-10 (Contrast Thresh)
        ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_xticklabels(['10.0', '7.75', '5.5', '3.25', '1.0'])  # ✅ Reversed order
        ax1.invert_xaxis()  # ✅ INVERT X-AXIS
        
        # Y-axis: 0-100 (Background %)
        ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_yticklabels(['0', '25', '50', '75', '100'])
        
        # Z-axis: 0-100 (Global Percentile)
        ax1.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_zticklabels(['0', '25', '50', '75', '100'])
        
        ax1.set_title(f'KDE Isosurface\nBandwidth: {self.kde_metadata["bandwidth"]:.3f}')
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
        
        # ================================================================
        # RIGHT PLOT: DELAUNAY HULL
        # ================================================================
        ax2 = fig.add_subplot(122, projection='3d')
        
        if self.hull is not None:
            # 🔧 OLD CODE (causes double denormalization):
            # hull_points = self.hull.points  # These are NOW ALREADY REAL VALUES!
            # # Then it would try to denormalize them again
            
            # ✅ NEW CODE: Hull points are already in real values, need to NORMALIZE for display
            hull_points_real = self.hull.points  # Already in real values
            
            # Normalize hull points for 0-1 display
            hull_points = self._normalize_parameters(hull_points_real)
                 
            # Extract surface triangles
            try:
                from scipy.spatial import ConvexHull
                
                # ✅ SWAP columns before creating hull
                hull_points_swapped = hull_points[:, [1, 0, 2]]  # [Contrast, Background, Percentile]
                outer_hull = ConvexHull(hull_points_swapped)
                
                # Create mesh
                faces = []
                for simplex in outer_hull.simplices:
                    faces.append(hull_points_swapped[simplex])
                
                mesh = Poly3DCollection(faces, alpha=0.25, facecolor='cyan',
                                       edgecolor='darkblue', linewidth=0.5)
                ax2.add_collection3d(mesh)
                
                # ✅ Plot vertices with SWAPPED coordinates
                ax2.scatter(hull_points_swapped[:, 0], hull_points_swapped[:, 1], hull_points_swapped[:, 2],
                           c='blue', s=5, alpha=0.4, label='Hull vertices', zorder=10)
                
            except Exception as e:
                print(f"⚠️ Could not create Delaunay surface: {e}")
                # Fallback with swapped coordinates
                hull_points_swapped = hull_points[:, [1, 0, 2]]
                ax2.scatter(hull_points_swapped[:, 0], hull_points_swapped[:, 1], hull_points_swapped[:, 2],
                           c='blue', s=5, alpha=0.6)
        
        # Show valid points (red dots) - NOW WITH SWAPPED AXES
        if self.valid_points is not None and len(self.valid_points) > 0:
            ax2.scatter(x_valid, y_valid, z_valid, c='red', s=3, alpha=0.1,
                       edgecolor='darkred', linewidth=1, label='Valid points', zorder=10)
        
        # ✅ SWAPPED AXIS LABELS (X=Contrast, Y=Background)
        ax2.set_xlabel('Contrast Thresh')
        ax2.set_ylabel('Background %')
        ax2.set_zlabel('Global Percentile')
        
        # Set axis limits (still 0-1 for normalized data)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_zlim(0, 1)
        
        # ✅ SWAPPED TICK LABELS
        # X-axis: 1-10 (Contrast Thresh)
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_xticklabels(['10.0', '7.75', '5.5', '3.25', '1.0'])  # ✅ Reversed order
        ax2.invert_xaxis()  # ✅ INVERT X-AXIS
        
        # Y-axis: 0-100 (Background %)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(['0', '25', '50', '75', '100'])
        
        # Z-axis: 0-100 (Global Percentile)
        ax2.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_zticklabels(['0', '25', '50', '75', '100'])
        
        ax2.set_title(f'Delaunay Hull\n{len(hull_points) if self.hull else 0} points')
        ax2.legend()
        ax2.view_init(elev=20, azim=45)
        
        plt.show()
    
    def _normalize_parameters(self, params):
        """Normalize parameters to [0, 1]³."""
        normalized = np.zeros_like(params, dtype=float)
        normalized[:, 0] = (params[:, 0] - self.param_ranges['bright_pct'][0]) / \
                          (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
        normalized[:, 1] = (params[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                          (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
        normalized[:, 2] = (params[:, 2] - self.param_ranges['percentile_val'][0]) / \
                          (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
        return normalized
    
    def _denormalize_parameters(self, params_norm):
        """Denormalize from [0, 1]³ to original ranges."""
        params = np.zeros_like(params_norm)
        params[:, 0] = params_norm[:, 0] * (self.param_ranges['bright_pct'][1] - 
                                           self.param_ranges['bright_pct'][0]) + \
                      self.param_ranges['bright_pct'][0]
        params[:, 1] = params_norm[:, 1] * (self.param_ranges['contrast_thresh'][1] - 
                                           self.param_ranges['contrast_thresh'][0]) + \
                      self.param_ranges['contrast_thresh'][0]
        params[:, 2] = params_norm[:, 2] * (self.param_ranges['percentile_val'][1] - 
                                           self.param_ranges['percentile_val'][0]) + \
                      self.param_ranges['percentile_val'][0]
        return params
    
    def save_complete(self, output_dir):
        """Save complete parameter space."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save hull
        if self.hull is not None:
            hull_path = os.path.join(output_dir, "valid_parameter_hull.pkl")
            with open(hull_path, 'wb') as f:
                pickle.dump(self.hull, f)
            print(f"✓ Saved hull: {hull_path}")
        
        # Save bounds
        if self.bounds is not None:
            bounds_path = os.path.join(output_dir, "parameter_bounds.npz")
            np.savez(bounds_path, **self.bounds)
            print(f"✓ Saved bounds: {bounds_path}")
        
        # Save valid points
        if self.valid_points is not None:
            points_path = os.path.join(output_dir, "valid_points.npy")
            np.save(points_path, self.valid_points)
            print(f"✓ Saved valid points: {points_path}")
        
        # Save KDE metadata
        if self.kde_metadata is not None:
            kde_path = os.path.join(output_dir, "kde_metadata.pkl")
            with open(kde_path, 'wb') as f:
                pickle.dump(self.kde_metadata, f)
            print(f"✓ Saved KDE metadata: {kde_path}")
        
        print(f"\n✓ Complete parameter space saved to: {output_dir}")
    
    # Keep the helper visualization methods from before
    def _get_diverse_nuclei(self, masks, channel_image, n_suggest=20):
        """Get diverse nucleus suggestions."""
        props = measure.regionprops(masks, intensity_image=channel_image)
        
        nucleus_features = []
        for prop in props:
            if prop.label == 0:
                continue
            
            pixels = channel_image[masks == prop.label]
            if len(pixels) == 0:
                continue
                
            features = {
                'label': prop.label,
                'area': prop.area,
                'mean_intensity': prop.mean_intensity,
                'cv': np.std(pixels) / prop.mean_intensity if prop.mean_intensity > 0 else 0
            }
            nucleus_features.append(features)
        
        if not nucleus_features:
            return []
        
        df = pd.DataFrame(nucleus_features)
        
        # Simple diverse selection
        selected = []
        if len(df) >= 4:
            # Get quartiles of intensity
            for q in [0.25, 0.5, 0.75]:
                quartile_val = df['mean_intensity'].quantile(q)
                closest_idx = (df['mean_intensity'] - quartile_val).abs().idxmin()
                selected.append(df.loc[closest_idx, 'label'])
        
        # Add random samples to fill
        remaining = df[~df['label'].isin(selected)]
        if len(remaining) > 0:
            n_random = min(n_suggest - len(selected), len(remaining))
            if n_random > 0:
                random_picks = remaining.sample(n=n_random)['label'].tolist()
                selected.extend(random_picks)
        
        return selected[:n_suggest]
    
    def _visualize_single_nucleus(self, masks, channel_image, nucleus_id):
        """Show detailed view of single nucleus."""
        nucleus_mask = (masks == nucleus_id)
        y_coords, x_coords = np.where(nucleus_mask)
        
        if len(y_coords) == 0:
            return
        
        padding = 30
        y_min = max(0, y_coords.min() - padding)
        y_max = min(channel_image.shape[0], y_coords.max() + padding)
        x_min = max(0, x_coords.min() - padding)
        x_max = min(channel_image.shape[1], x_coords.max() + padding)
        
        crop_img = channel_image[y_min:y_max, x_min:x_max]
        crop_mask = nucleus_mask[y_min:y_max, x_min:x_max]
        
        crop_dog = filters.difference_of_gaussians(crop_img, low_sigma=1, high_sigma=2)
        crop_dog = np.clip(crop_dog, 0, None)
        crop_dog = exposure.rescale_intensity(crop_dog, in_range='image', 
                                                 out_range=(0, crop_img.max()))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(crop_img, cmap='gray')
        axes[0].contour(crop_mask, colors='cyan', linewidths=2)
        axes[0].set_title(f'Original - Nucleus {nucleus_id}')
        axes[0].axis('off')
        
        axes[1].imshow(crop_dog, cmap='gray')
        axes[1].contour(crop_mask, colors='cyan', linewidths=2)
        axes[1].set_title('DoG Filtered')
        axes[1].axis('off')
        
        enhanced = exposure.equalize_adapthist(crop_img)
        enhanced = exposure.rescale_intensity(enhanced, in_range='image', 
                                                 out_range=(0, crop_img.max()))
        axes[2].imshow(enhanced, cmap='hot')
        axes[2].contour(crop_mask, colors='cyan', linewidths=2)
        axes[2].set_title('Enhanced')
        axes[2].axis('off')
        
        pixels = channel_image[nucleus_mask]
        stats_text = (f"Area: {len(pixels)} px\n"
                     f"Mean: {np.mean(pixels):.3f}\n"
                     f"CV: {np.std(pixels)/np.mean(pixels):.3f}")
        fig.text(0.02, 0.5, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.suptitle(f'Nucleus {nucleus_id} - Count visible foci', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_all_nuclei_overview(self, masks, channel_image, used_ids):
        """Show overview of all nuclei."""
        from skimage.segmentation import find_boundaries
        from matplotlib import patheffects
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(channel_image, cmap='gray')
        
        boundaries = find_boundaries(masks, mode='outer')
        ax.contour(boundaries, colors='cyan', linewidths=0.5, alpha=0.1)
        
        for nucleus_id in np.unique(masks)[1:]:
            nucleus_mask = (masks == nucleus_id)
            y_coords, x_coords = np.where(nucleus_mask)
            
            if len(y_coords) == 0:
                continue
            
            cy = np.mean(y_coords)
            cx = np.mean(x_coords)
            
            if nucleus_id in self.ground_truth_nuclei:
                color = 'lime'
                transparent = 1
                text = f"{nucleus_id}✓"
            else:
                color = 'white'
                transparent = 0.3
                text = str(nucleus_id)
            
            # ✅ Remove background box entirely - just text with outline
            ax.text(cx, cy, text, color=color, fontsize=8,
                   ha='center', va='center',
                   alpha = transparent
                   ) 
        
        ax.set_title('All Nuclei - Green=Selected')
        ax.axis('off')
        plt.tight_layout()
        plt.show()