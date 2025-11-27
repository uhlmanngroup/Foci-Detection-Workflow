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



    
    def select_optimal_parameters_auto(self, channel='TRITC'):
        """
        Automatically determine optimal diversity weight based on:
        1. Data variance (how much parameters differ in performance)
        2. Number of parameters requested (n_final_params)
        
        Logic:
        - If requesting 1 parameter: diversity_weight = 0 (just pick the best)
        - If requesting 2-3 parameters: scale diversity based on variance
        - If requesting 4+ parameters: higher diversity to cover parameter space
        """
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        # Calculate coefficient of variation across all parameters
        cv_across_params = param_stats['mean_foci'].std() / param_stats['mean_foci'].mean()
        
        # Base diversity on number of parameters requested
        if self.n_final_params == 1:
            # Only 1 parameter → no diversity needed, just pick the best
            diversity_weight = 0.0
            print(f"  Requesting 1 parameter → diversity_weight=0.0 (pure performance)")
            
        elif self.n_final_params == 2:
            # 2 parameters → light diversity
            if cv_across_params > 0.4:
                diversity_weight = 0.3  # High variance → moderate diversity
                print(f"  2 parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.3")
            else:
                diversity_weight = 0.15  # Low variance → slight diversity
                print(f"  2 parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.15")
                
        elif self.n_final_params == 3:
            # 3 parameters → balanced diversity (your default case)
            if cv_across_params > 0.5:
                diversity_weight = 0.5  # High variance → more diversity
                print(f"  3 parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.5")
            elif cv_across_params > 0.3:
                diversity_weight = 0.3  # Medium variance → balanced
                print(f"  3 parameters, medium variance (CV={cv_across_params:.2f}) → diversity_weight=0.3")
            else:
                diversity_weight = 0.2  # Low variance → light diversity
                print(f"  3 parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.2")
                
        else:  # 4+ parameters
            # Many parameters → need more diversity to cover space
            if cv_across_params > 0.4:
                diversity_weight = 0.6  # High variance → high diversity
                print(f"  {self.n_final_params} parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.6")
            else:
                diversity_weight = 0.4  # Low variance → moderate diversity
                print(f"  {self.n_final_params} parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.4")
        
        print(f"  Rationale: With {self.n_final_params} parameter(s), balancing performance vs coverage")
        
        return self.select_optimal_parameters(channel, diversity_weight)
