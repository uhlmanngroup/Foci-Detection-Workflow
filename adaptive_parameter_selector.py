"""
Adaptive Parameter Selection for Foci Detection Pipeline
Two-phase approach: Calibration phase with full sweep, then production with optimized parameters
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


class AdaptiveParameterSelector:
    """
    Learns optimal parameter combinations from initial images,
    then selects best 1-3 combinations for production use.
    """
    
    def __init__(self, n_calibration_images=5, n_final_params=3):
        """
        Parameters:
        -----------
        n_calibration_images : int
            Number of images to use for calibration phase
        n_final_params : int
            Number of parameter combinations to use in production (1-3)
        """
        self.n_calibration_images = n_calibration_images
        self.n_final_params = n_final_params
        
        # Storage for calibration results
        self.calibration_results = []
        self.parameter_performance = defaultdict(list)
        self.selected_params = None
        
    
    def record_calibration_result(self, image_id, cell_id, param_combo, 
                                   foci_count, detection_prob, channel):
        """
        Record results from one nucleus during calibration phase.
        
        Parameters:
        -----------
        image_id : str
            Image identifier (e.g., "Well_00044_Pos_00021")
        cell_id : int
            Nucleus ID within that image
        param_combo : tuple
            (bright_pct, contrast_thresh, percentile_val)
        foci_count : int
            Number of foci detected with this parameter combo
        detection_prob : float
            Detection probability for this focus
        channel : str
            Channel name (TRITC/FITC)
        """
        self.calibration_results.append({
            'image_id': image_id,
            'cell_id': cell_id,
            'param_combo': param_combo,
            'foci_count': foci_count,
            'detection_prob': detection_prob,
            'channel': channel
        })
    
    
    def analyze_calibration_data(self, channel='TRITC'):
        """
        Analyze calibration results to find optimal parameter combinations.
        
        Strategy:
        1. Group by parameter combination
        2. Calculate mean foci count per parameter combo
        3. Find parameters that give closest to MEAN across all nuclei
        4. Select most consistent parameters (low variance)
        
        Returns:
        --------
        pd.DataFrame : Performance metrics for each parameter combination
        """
        df = pd.DataFrame(self.calibration_results)
        
        # Filter to specific channel
        df = df[df['channel'] == channel]
        
        if len(df) == 0:
            raise ValueError(f"No calibration data for channel {channel}")
        
        # Calculate global mean foci count across all nuclei
        global_mean_foci = df.groupby(['image_id', 'cell_id'])['foci_count'].mean().mean()
        
        print(f"\n📊 {channel} Calibration Analysis:")
        print(f"   Global mean foci per nucleus: {global_mean_foci:.2f}")
        
        # Group by parameter combination
        param_stats = df.groupby('param_combo').agg({
            'foci_count': ['mean', 'std', 'count']
        }).reset_index()
        
        param_stats.columns = ['param_combo', 'mean_foci', 'std_foci', 'n_nuclei']
        
        # Calculate deviation from global mean
        param_stats['deviation_from_mean'] = np.abs(param_stats['mean_foci'] - global_mean_foci)
        
        # Calculate coefficient of variation (lower = more consistent)
        param_stats['cv'] = param_stats['std_foci'] / (param_stats['mean_foci'] + 1e-6)
        
        # Composite score: prioritize closeness to mean, penalize high variance
        # Lower score = better
        param_stats['score'] = (
            param_stats['deviation_from_mean'] * 2.0 +  # Deviation is most important
            param_stats['cv'] * 1.0  # Consistency is secondary
        )
        
        # Sort by score (best first)
        param_stats = param_stats.sort_values('score').reset_index(drop=True)
        
        return param_stats, global_mean_foci
    
    
    def select_optimal_parameters(self, channel='TRITC', diversity_weight=0.3):
        """
        Select n_final_params optimal parameter combinations.
        
        Strategy:
        1. Start with single best parameter
        2. Add diverse parameters that cover different regions of parameter space
        
        Parameters:
        -----------
        channel : str
            Channel to optimize for
        diversity_weight : float
            How much to value diversity vs pure performance (0-1)
            Higher = more diverse parameters selected
        
        Returns:
        --------
        list : Selected parameter combinations
        """
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        selected = []
        selected_indices = []
        
        # Extract parameter arrays for distance calculations
        param_arrays = np.array([list(p) for p in param_stats['param_combo'].values])
        
        # 1. Select best performing parameter
        best_idx = 0
        selected.append(param_stats.iloc[best_idx]['param_combo'])
        selected_indices.append(best_idx)
        
        print(f"\n🎯 Selected Parameter 1/{self.n_final_params}:")
        print(f"   {selected[0]}")
        print(f"   Mean foci: {param_stats.iloc[best_idx]['mean_foci']:.2f}")
        print(f"   Std: {param_stats.iloc[best_idx]['std_foci']:.2f}")
        print(f"   Score: {param_stats.iloc[best_idx]['score']:.3f}")
        
        # 2. Select additional parameters balancing performance and diversity
        for i in range(1, self.n_final_params):
            # Calculate distances from already selected parameters
            selected_params = param_arrays[selected_indices]
            
            # For each remaining parameter, calculate min distance to selected set
            min_distances = []
            for idx in range(len(param_arrays)):
                if idx in selected_indices:
                    min_distances.append(0)
                else:
                    distances = cdist([param_arrays[idx]], selected_params)[0]
                    min_distances.append(np.min(distances))
            
            min_distances = np.array(min_distances)
            
            # Normalize distances to 0-1 range
            if min_distances.max() > 0:
                normalized_distances = min_distances / min_distances.max()
            else:
                normalized_distances = min_distances
            
            # Normalize scores to 0-1 range (lower is better)
            normalized_scores = param_stats['score'].values / param_stats['score'].max()
            
            # Combined score: balance performance and diversity
            combined_scores = (
                (1 - diversity_weight) * normalized_scores +  # Performance (lower better)
                (-diversity_weight) * normalized_distances     # Diversity (higher better)
            )
            
            # Mask out already selected
            combined_scores[selected_indices] = np.inf
            
            # Select best combined score
            next_idx = np.argmin(combined_scores)
            selected.append(param_stats.iloc[next_idx]['param_combo'])
            selected_indices.append(next_idx)
            
            print(f"\n🎯 Selected Parameter {i+1}/{self.n_final_params}:")
            print(f"   {selected[i]}")
            print(f"   Mean foci: {param_stats.iloc[next_idx]['mean_foci']:.2f}")
            print(f"   Std: {param_stats.iloc[next_idx]['std_foci']:.2f}")
            print(f"   Score: {param_stats.iloc[next_idx]['score']:.3f}")
            print(f"   Distance from selected: {min_distances[next_idx]:.3f}")
        
        self.selected_params = {channel: selected}
        return selected
    
    
    def visualize_parameter_space(self, channel='TRITC', save_path=None):
        """
        Visualize parameter space and selected parameters.
        """
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        # Extract parameters
        params = np.array([list(p) for p in param_stats['param_combo'].values])
        bright_pct = params[:, 0]
        contrast_thresh = params[:, 1]
        percentile_val = params[:, 2]
        
        scores = param_stats['score'].values
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Bright % vs Contrast (colored by score)
        ax1 = fig.add_subplot(131)
        scatter1 = ax1.scatter(bright_pct, contrast_thresh, c=scores, 
                              cmap='RdYlGn_r', s=50, alpha=0.6)
        
        # Mark selected parameters
        if self.selected_params and channel in self.selected_params:
            selected = np.array([list(p) for p in self.selected_params[channel]])
            ax1.scatter(selected[:, 0], selected[:, 1], 
                       marker='*', s=500, c='blue', 
                       edgecolors='black', linewidths=2,
                       label='Selected', zorder=5)
        
        ax1.set_xlabel('Brightness Percentile')
        ax1.set_ylabel('Contrast Threshold')
        ax1.set_title('Bright % vs Contrast')
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Score (lower=better)')
        
        # Plot 2: Percentile vs Contrast
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(percentile_val, contrast_thresh, c=scores,
                              cmap='RdYlGn_r', s=50, alpha=0.6)
        
        if self.selected_params and channel in self.selected_params:
            ax2.scatter(selected[:, 2], selected[:, 1],
                       marker='*', s=500, c='blue',
                       edgecolors='black', linewidths=2,
                       label='Selected', zorder=5)
        
        ax2.set_xlabel('Global Percentile')
        ax2.set_ylabel('Contrast Threshold')
        ax2.set_title('Percentile vs Contrast')
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='Score (lower=better)')
        
        # Plot 3: Score distribution
        ax3 = fig.add_subplot(133)
        ax3.hist(param_stats['mean_foci'], bins=30, alpha=0.6, label='All params')
        ax3.axvline(global_mean, color='red', linestyle='--', 
                   linewidth=2, label=f'Global mean: {global_mean:.2f}')
        
        if self.selected_params and channel in self.selected_params:
            selected_means = [param_stats[param_stats['param_combo'] == p]['mean_foci'].values[0] 
                            for p in self.selected_params[channel]]
            for i, mean_val in enumerate(selected_means):
                ax3.axvline(mean_val, color='blue', linestyle='-', 
                           linewidth=2, alpha=0.7, label=f'Selected {i+1}')
        
        ax3.set_xlabel('Mean Foci Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Foci Count Distribution')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        
        return fig
    
    
    def save_calibration(self, filepath):
        """Save calibration results and selected parameters."""
        data = {
            'calibration_results': self.calibration_results,
            'selected_params': self.selected_params,
            'n_calibration_images': self.n_calibration_images,
            'n_final_params': self.n_final_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved calibration data to {filepath}")
    
    
    def load_calibration(self, filepath):
        """Load previously saved calibration."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.calibration_results = data['calibration_results']
        self.selected_params = data['selected_params']
        self.n_calibration_images = data['n_calibration_images']
        self.n_final_params = data['n_final_params']
        
        print(f"✅ Loaded calibration data from {filepath}")
        print(f"   Calibration images: {self.n_calibration_images}")
        print(f"   Selected parameters: {self.n_final_params}")


# ============================================================================
# MODIFIED DETECTION FUNCTION FOR CALIBRATION TRACKING
# ============================================================================

def detect_foci_with_calibration_tracking(
    nucleus_mask, image, original_image, channel_name, cell_id,
    valid_param_samples, total_iterations, water_threshold_percentile,
    watershed_min_detection_prob=0.0,
    well_number=None, position_number=None,
    calibration_mode=False,
    calibration_tracker=None,
    image_id=None
):
    """
    Modified version of detect_foci_single_channel that tracks parameter performance.
    
    NEW PARAMETERS:
    ---------------
    calibration_mode : bool
        If True, records detailed parameter performance
    calibration_tracker : AdaptiveParameterSelector
        Tracker object to record results
    image_id : str
        Identifier for current image
    """
    
    # [... SAME PREPROCESSING CODE AS ORIGINAL ...]
    isolated_img = img_as_float(image.copy())
    isolated_img[~nucleus_mask] = 0
    
    if isolated_img.max() == 0:
        return [], {}, None
    
    filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
    filtered_img = np.clip(filtered_img, 0, None)
    filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', 
                                             out_range=(0, isolated_img.max()))
    
    # [... SAME PARAMETER EXTRACTION ...]
    bright_pcts = valid_param_samples[:, 0]
    contrast_threshs = valid_param_samples[:, 1]
    percentile_vals = valid_param_samples[:, 2]
    
    pos_pixels = original_image[original_image > 0]
    if pos_pixels.size == 0:
        return [], {}, None
    
    min_brightness_per_param = np.percentile(pos_pixels, percentile_vals)
    global_min_brightness = np.min(min_brightness_per_param)
    
    # [... SAME CANDIDATE DETECTION ...]
    candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                        threshold_abs=global_min_brightness)
    candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                          threshold_abs=global_min_brightness)
    
    if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
        return [], {}, None
    
    # [... SAME SETUP ...]
    filt_yx = np.asarray(candidates_filtered, dtype=int)
    unf_yx = np.asarray(candidates_unfiltered, dtype=int)
    filt_intensities = filtered_img[filt_yx[:, 0], filt_yx[:, 1]]
    unf_intensities = isolated_img[unf_yx[:, 0], unf_yx[:, 1]]
    
    unique_brights = np.unique(np.round(bright_pcts, 6))
    bright_to_idx = {b: idx for idx, b in enumerate(unique_brights)}
    
    # [... SAME LOCAL BACKGROUND CALCULATION ...]
    from nucleus_worker_Visualization import compute_adaptive_background_texture_nucleus_fallback
    
    local_percentiles_unf, texture_info_unf = compute_adaptive_background_texture_nucleus_fallback(
        image=isolated_img, coords=unf_yx, unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask, return_texture_info=True
    )
    
    local_percentiles_filt, texture_info_filt = compute_adaptive_background_texture_nucleus_fallback(
        image=filtered_img, coords=filt_yx, unique_percentiles=unique_brights, 
        nucleus_mask=nucleus_mask, return_texture_info=True
    )
    
    # [... SAME UNIFORMITY CHECK ...]
    nucleus_is_uniform = False
    contrast_multiplier = 1.0
    nucleus_cv = 0.0
    
    if texture_info_unf['nucleus_stats']:
        stats = list(texture_info_unf['nucleus_stats'].values())[0]
        nucleus_cv = stats['cv']
        
        if nucleus_cv < 0.20:
            nucleus_is_uniform = True
            contrast_multiplier = 1.0
    
    distances = cdist(unf_yx, filt_yx)
    tolerance = 2
    
    # ==========================================
    # PARAMETER SWEEP WITH CALIBRATION TRACKING
    # ==========================================
    foci_counts = []
    all_detected_foci = []
    
    # NEW: Track per-parameter results
    param_to_foci = {}
    
    for p_idx in range(len(valid_param_samples)):
        adjusted_contrast_threshs = contrast_threshs.copy()
        if nucleus_is_uniform:
            adjusted_contrast_threshs = contrast_threshs * contrast_multiplier
        
        confirmed_coords, count = apply_foci_filters(
            p_idx, bright_pcts, adjusted_contrast_threshs, percentile_vals,
            min_brightness_per_param, bright_to_idx,
            unf_intensities, filt_intensities,
            local_percentiles_unf, local_percentiles_filt,
            distances, unf_yx, tolerance
        )
        
        foci_counts.append(count)
        
        # Store parameter-specific results
        param_combo = tuple(valid_param_samples[p_idx])
        param_to_foci[param_combo] = count
        
        for coord in confirmed_coords:
            all_detected_foci.append(tuple(coord))
    
    # NEW: Record calibration data if in calibration mode
    if calibration_mode and calibration_tracker is not None and image_id is not None:
        for param_combo, foci_count in param_to_foci.items():
            calibration_tracker.record_calibration_result(
                image_id=image_id,
                cell_id=cell_id,
                param_combo=param_combo,
                foci_count=foci_count,
                detection_prob=100.0,  # All params tested in calibration
                channel=channel_name
            )
    
    # [... REST OF ORIGINAL FUNCTION UNCHANGED ...]
    # Continue with watershed, measurements, etc.
    
    if not foci_counts:
        return [], {}, None
    
    foci_detection_count = Counter(all_detected_foci)
    mean_foci = np.mean(foci_counts)
    std_foci = np.std(foci_counts)
    min_foci = int(min(foci_counts))
    max_foci = int(max(foci_counts))
    
    # [... Continue with watershed as in original ...]
    
    return foci_list, summary, water_labels
