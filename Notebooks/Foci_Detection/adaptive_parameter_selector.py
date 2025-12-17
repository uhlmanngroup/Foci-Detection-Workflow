"""
Adaptive Parameter Selection for Foci Detection Pipeline

This module implements a two-phase approach for optimizing foci detection parameters:
1. CALIBRATION PHASE: Test 256 parameter combinations on first N images
2. PRODUCTION PHASE: Use only the best 1-3 parameters on remaining images

The goal is to find parameter combinations that:
- Detect a consistent number of foci across different nuclei
- Are close to the average detection rate (not too sensitive, not too conservative)
- Cover diverse regions of parameter space (avoid redundancy)

This dramatically speeds up processing: instead of testing 256 parameters per nucleus
throughout the entire dataset, we identify the optimal few and use only those.
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from typing import Tuple 


class AdaptiveParameterSelector:
    """
    Learns optimal parameter combinations from initial images,
    then selects best 1-X combinations for production use.
    
    The adaptive selection process:
    1. Calibration: Run full parameter sweep (256 combinations) on first N images
    2. Analysis: Identify which parameters give most consistent, reliable results
    3. Selection: Pick 1-X best parameters balancing performance and diversity
    4. Production: Use only these optimized parameters on remaining images
    
    This reduces computation time by ~100× while maintaining detection quality.
    
    Attributes:
    -----------
    n_calibration_images : int
        How many images to use for calibration (typically 5-10)
    n_final_params : int
        How many parameter combinations to use in production (1-X)
        1 = fastest, 3 = more robust, etc.
    calibration_results : list
        Accumulated results from calibration phase
    selected_params : dict
        Optimal parameters for each channel {channel: [param_tuples]}
        
    Usage Example:
    --------------
```python
    # Phase 1: Calibration
    selector = AdaptiveParameterSelector(n_calibration_images=5, n_final_params=3)
    
    # During calibration, record results for each nucleus:
    selector.record_calibration_result(
        image_id='Well_00044_Pos_00021',
        cell_id=5,
        param_combo=(15.0, 2.5, 10.0),
        foci_count=12,
        detection_prob=100.0,
        channel='TRITC'
    )
    
    # Phase 2: Analysis and selection
    optimal_params = selector.select_optimal_parameters_auto(channel='TRITC')
    # Returns: [(15.0, 2.5, 10.0), (20.0, 3.0, 15.0), (25.0, 2.0, 20.0)]
    
    # Phase 3: Save for production use
    selector.save_reduced_parameters('params_tritc.pkl', 'TRITC')
    
    # Later: Load for production
    selector.load_reduced_parameters('params_tritc.pkl', 'TRITC')
```
    """
    
    def __init__(self, n_calibration_images=5, n_final_params=3):
        """
        Initialize the adaptive parameter selector.
        
        Parameters:
        -----------
        n_calibration_images : int, default=5
            Number of images to use for calibration phase
            Recommendations:
            - 5 images: Fast, good for homogeneous data
            - 10 images: Balanced, recommended for most cases
            - 20+ images: Slower but better for heterogeneous data
            
        n_final_params : int, default=3
            Number of parameter combinations to use in production
            Trade-offs:
            - 1 param: Fastest (3× faster than 3), less robust to edge cases
            - 2 params: Balanced speed/robustness
            - 3 params: Most robust, still 85× faster than full sweep
            - 4+ params: Diminishing returns
        """
        self.n_calibration_images = n_calibration_images
        self.n_final_params = n_final_params
        
        # ----------------------------------------------------------------
        # Storage for calibration results
        # ----------------------------------------------------------------
        # List of dicts, one per nucleus per parameter tested
        # Each entry: {image_id, cell_id, param_combo, foci_count, channel}
        self.calibration_results = []
        
        # Currently unused, kept for potential future features
        self.parameter_performance = defaultdict(list)
        
        # Will store selected optimal parameters after analysis
        # Format: {channel_name: [param_tuple1, param_tuple2, ...]}
        self.selected_params = None
        
    

    def save_reduced_parameters(self, save_path: str, channel: str):
        """
        Save optimized parameters for this channel to disk.
        
        After calibration and selection, save the optimal parameter combinations
        so they can be loaded for future runs without re-calibration.
        
        This enables:
        - Reusing calibration across multiple analysis runs
        - Sharing optimized parameters between users
        - Archiving parameters with publication for reproducibility
        
        Parameters:
        -----------
        save_path : str
            Full path where to save the parameters (e.g., 'params_tritc.pkl')
            Parent directory will be created if it doesn't exist
        channel : str
            Channel name ('TRITC' or 'FITC')
            Only parameters for this channel will be saved
            
        Raises:
        -------
        ValueError
            If no parameters have been selected yet (must run selection first)
            
        File Format:
        ------------
        Pickle file containing list of parameter tuples:
        [(bright_pct, contrast_thresh, percentile_val), ...]
        
        Example:
        --------
        >>> selector.select_optimal_parameters_auto('TRITC')
        >>> selector.save_reduced_parameters('tritc_params.pkl', 'TRITC')
        ✅ Saved reduced TRITC parameters to tritc_params.pkl
        """
        # ----------------------------------------------------------------
        # Validation: Ensure parameters have been selected
        # ----------------------------------------------------------------
        if self.selected_params is None:
            raise ValueError("No parameters selected yet. Run select_optimal_parameters first.")
        
        # ----------------------------------------------------------------
        # Save parameters if they exist for this channel
        # ----------------------------------------------------------------
        if channel in self.selected_params:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save as pickle (preserves tuple structure exactly)
            with open(save_path, 'wb') as f:
                pickle.dump(self.selected_params[channel], f)
            
            print(f"✅ Saved reduced {channel} parameters to {save_path}")
        else:
            # Channel not in selected_params (wasn't calibrated)
            print(f"⚠️ No {channel} parameters found in tracker")


    def load_reduced_parameters(self, load_path: str, channel: str) -> np.ndarray:
        """
        Load pre-optimized parameters for this channel from disk.
        
        Use this to skip the calibration phase entirely by loading previously
        optimized parameters. Useful for:
        - Running analysis on new data with same characteristics
        - Reproducing published results with exact parameters
        - Saving time when re-analyzing after pipeline changes
        
        Parameters:
        -----------
        load_path : str
            Full path to the saved parameters file (e.g., 'params_tritc.pkl')
        channel : str
            Channel name ('TRITC' or 'FITC')
            Must match the channel that was saved
            
        Returns:
        --------
        np.ndarray : The loaded parameter combinations, shape (N, 3)
            Each row is [bright_pct, contrast_thresh, percentile_val]
            N = number of parameters (typically 1-3)
            
        Raises:
        -------
        FileNotFoundError
            If the parameter file doesn't exist at load_path
        ValueError
            If file is corrupted or contains invalid data
            
        Example:
        --------
        >>> selector = AdaptiveParameterSelector()
        >>> params = selector.load_reduced_parameters('tritc_params.pkl', 'TRITC')
        ✅ Loaded 3 TRITC parameters from tritc_params.pkl
        >>> print(params)
        [[15.0, 2.5, 10.0],
         [20.0, 3.0, 15.0],
         [25.0, 2.0, 20.0]]
        """
        # ----------------------------------------------------------------
        # Validation: Check file exists
        # ----------------------------------------------------------------
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Parameter file not found: {load_path}")
        
        # ----------------------------------------------------------------
        # Load and validate parameters
        # ----------------------------------------------------------------
        try:
            with open(load_path, 'rb') as f:
                params = pickle.load(f)
        except Exception as e:
            # Catch any unpickling errors (corrupted file, wrong format, etc.)
            raise ValueError(f"Failed to load parameters from {load_path}: {e}")
        
        # Validate that loaded data has correct type
        # Should be either list of tuples or numpy array
        if not isinstance(params, (list, np.ndarray)):
            raise ValueError(f"Loaded parameters have invalid type: {type(params)}")
        
        print(f"✅ Loaded {len(params)} {channel} parameters from {load_path}")
        
        # ----------------------------------------------------------------
        # Store in selected_params for consistency with selection workflow
        # ----------------------------------------------------------------
        # This allows the rest of the code to work identically whether
        # parameters were selected via calibration or loaded from file
        if self.selected_params is None:
            self.selected_params = {}
        self.selected_params[channel] = params
        
        # Return as numpy array for convenience
        return np.array(params)            






    def record_calibration_result(self, image_id, cell_id, param_combo, 
                                   foci_count, detection_prob, channel):
        """
        Record results from one nucleus during calibration phase.
        
        This function is called repeatedly during calibration to accumulate
        performance data for each parameter combination tested. The data is
        later analyzed to identify optimal parameters.
        
        Call this function ONCE per nucleus per parameter combination tested.
        For 256 parameters tested on 5 images with 50 nuclei each:
        → 256 × 5 × 50 = 64,000 calls to this function
        
        Parameters:
        -----------
        image_id : str
            Unique image identifier (e.g., "Well_00044_Pos_00021")
            Used to group results by image for variance analysis
        cell_id : int
            Nucleus ID within that image (1, 2, 3, ...)
            Used to group results by nucleus
        param_combo : tuple of 3 floats
            The parameter combination tested: (bright_pct, contrast_thresh, percentile_val)
            Example: (15.0, 2.5, 10.0)
        foci_count : int
            Number of foci detected with this parameter combination
            This is the KEY metric used to evaluate parameter performance
        detection_prob : float
            Detection probability for this focus (0-100)
            Currently recorded but not used in analysis
            Reserved for future weighted scoring
        channel : str
            Channel name ('TRITC' or 'FITC')
            Allows separate optimization per channel
            
        Example Usage:
        --------------
```python
        # During calibration phase, after testing each parameter:
        for param in parameter_space:
            foci_detected = detect_foci(nucleus, param)
            selector.record_calibration_result(
                image_id=f"Well_{well}_Pos_{pos}",
                cell_id=nucleus_id,
                param_combo=param,
                foci_count=len(foci_detected),
                detection_prob=100.0,
                channel='TRITC'
            )
```
        
        Notes:
        ------
        - Results are stored in memory (self.calibration_results list)
        - Must call analyze_calibration_data() after all recording is complete
        - detection_prob is currently unused but kept for future extensions
        """
        # ----------------------------------------------------------------
        # Append result to accumulator
        # ----------------------------------------------------------------
        # Store as dictionary for easy conversion to pandas DataFrame later
        self.calibration_results.append({
            'image_id': image_id,          # Which image
            'cell_id': cell_id,            # Which nucleus
            'param_combo': param_combo,    # Which parameters
            'foci_count': foci_count,      # How many foci detected
            'detection_prob': detection_prob,  # Detection confidence (unused currently)
            'channel': channel             # Which channel
        })
    
    
    def analyze_calibration_data(self, channel='TRITC'):
        """
        Analyze calibration results to find optimal parameter combinations.
        
        This is the CORE ANALYSIS FUNCTION that processes all calibration data
        to identify which parameter combinations perform best. "Best" is defined as:
        1. Detects close to the AVERAGE number of foci (not too many, not too few)
        2. Consistent across different nuclei (low variance)
        
        Why target the average?
        - Parameters that detect many foci might be over-sensitive (false positives)
        - Parameters that detect few foci might be under-sensitive (false negatives)
        - Parameters near the average balance sensitivity and specificity
        
        Strategy:
        ---------
        1. Calculate global mean foci count across all nuclei
        2. For each parameter combination, calculate:
           - Mean foci detected per nucleus
           - Standard deviation (consistency measure)
           - Deviation from global mean (accuracy measure)
        3. Composite score = 2×deviation + 1×coefficient_of_variation
           (Lower score = better)
        4. Sort parameters by score (best first)
        
        Parameters:
        -----------
        channel : str, default='TRITC'
            Which channel to analyze ('TRITC' or 'FITC')
            Analysis is performed separately for each channel
            
        Returns:
        --------
        param_stats : pd.DataFrame
            Statistics for each parameter combination, sorted by score
            Columns:
            - param_combo: (bright_pct, contrast_thresh, percentile_val)
            - mean_foci: Average foci count across all nuclei
            - std_foci: Standard deviation of foci counts
            - n_nuclei: Number of nuclei tested with this parameter
            - deviation_from_mean: |mean_foci - global_mean|
            - cv: Coefficient of variation (std/mean)
            - score: Composite score (lower = better)
        global_mean_foci : float
            Average foci count across all nuclei and all parameters
            This is the target value we want parameters to achieve
            
        Raises:
        -------
        ValueError
            If no calibration data exists for the specified channel
            (Must call record_calibration_result() first)
            
        Example Output:
        ---------------
        📊 TRITC Calibration Analysis:
           Global mean foci per nucleus: 8.23
           
        param_stats DataFrame:
        | param_combo      | mean_foci | std_foci | score |
        |------------------|-----------|----------|-------|
        | (15.0, 2.5, 10)  | 8.15      | 2.1      | 0.42  | ← Best
        | (20.0, 3.0, 15)  | 8.45      | 2.3      | 0.51  |
        | (25.0, 2.0, 20)  | 8.67      | 3.1      | 0.82  |
        """
        # ----------------------------------------------------------------
        # Convert calibration results to DataFrame for analysis
        # ----------------------------------------------------------------
        df = pd.DataFrame(self.calibration_results)
        
        # Filter to specific channel (TRITC and FITC analyzed separately)
        df = df[df['channel'] == channel]
        
        # Validation: Ensure we have data for this channel
        if len(df) == 0:
            raise ValueError(f"No calibration data for channel {channel}")
        
        # ----------------------------------------------------------------
        # Calculate global mean foci count (the target)
        # ----------------------------------------------------------------
        # Two-step grouping to get mean per nucleus, then mean across nuclei
        # Step 1: groupby(['image_id', 'cell_id']) → mean per nucleus
        # Step 2: .mean() → average across all nuclei
        # This gives the overall average detection rate across the dataset
        global_mean_foci = df.groupby(['image_id', 'cell_id'])['foci_count'].mean().mean()
        
        print(f"\n📊 {channel} Calibration Analysis:")
        print(f"   Global mean foci per nucleus: {global_mean_foci:.2f}")
        
        # ----------------------------------------------------------------
        # Group by parameter combination and compute statistics
        # ----------------------------------------------------------------
        # For each unique parameter combination, calculate:
        # - mean: average foci count
        # - std: standard deviation (consistency)
        # - count: how many nuclei tested
        param_stats = df.groupby('param_combo').agg({
            'foci_count': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten multi-level column names for easier access
        param_stats.columns = ['param_combo', 'mean_foci', 'std_foci', 'n_nuclei']
        
        # ----------------------------------------------------------------
        # Calculate deviation from global mean
        # ----------------------------------------------------------------
        # Absolute difference between this parameter's mean and the global mean
        # Lower deviation = closer to target = better
        # Example: if global mean = 8.0 and param mean = 7.5, deviation = 0.5
        param_stats['deviation_from_mean'] = np.abs(param_stats['mean_foci'] - global_mean_foci)
        
        # ----------------------------------------------------------------
        # Calculate coefficient of variation (CV = std/mean)
        # ----------------------------------------------------------------
        # CV is a normalized measure of consistency:
        # - Low CV (e.g., 0.2): Consistent across nuclei
        # - High CV (e.g., 0.8): Variable, unreliable
        # Adding 1e-6 prevents division by zero if mean_foci = 0
        param_stats['cv'] = param_stats['std_foci'] / (param_stats['mean_foci'] + 1e-6)

        # ----------------------------------------------------------------
        # Composite score: Balance accuracy and consistency
        # ----------------------------------------------------------------
        # Score = 2×deviation + 1×CV
        # 
        # Why this weighting?
        # - Deviation weighted 2×: Accuracy is most important (close to mean)
        # - CV weighted 1×: Consistency is secondary (low variance)
        # 
        # Lower score = better parameter
        # Example: deviation=0.5, CV=0.3 → score = 2×0.5 + 1×0.3 = 1.3
        param_stats['score'] = (
            param_stats['deviation_from_mean'] * 2.0 +  # Deviation is most important
            param_stats['cv'] * 1.0  # Consistency is secondary
        )

        # ----------------------------------------------------------------
        # Clean up any NaN/inf values in scores
        # ----------------------------------------------------------------
        # NaN can occur if std=0 and mean=0 (never happens in practice)
        # inf can occur if mean=0 but std>0 (rare edge case)
        # Replace with 999 (worst possible score) so they're ranked last
        param_stats['score'] = param_stats['score'].fillna(999)
        param_stats['score'] = param_stats['score'].replace([np.inf, -np.inf], 999)
        
        # ----------------------------------------------------------------
        # Sort by score (best parameters first)
        # ----------------------------------------------------------------
        param_stats = param_stats.sort_values('score').reset_index(drop=True)
        
        # Return both the statistics and the global mean (used by other functions)
        return param_stats, global_mean_foci
    
    
    def select_optimal_parameters(self, channel='TRITC', diversity_weight=0.3):
        """
        Select n_final_params optimal parameter combinations.
        
        This function picks the best N parameter combinations from the calibration
        results, balancing two competing objectives:
        1. PERFORMANCE: Select parameters with best scores (closest to mean, low variance)
        2. DIVERSITY: Select parameters that cover different regions of parameter space
        
        Why diversity matters:
        - Similar parameters are redundant (all behave nearly the same)
        - Diverse parameters provide robustness to edge cases
        - Example: (15, 2.5, 10) and (15.5, 2.6, 10.5) are redundant
        -          (15, 2.5, 10) and (25, 3.0, 20) are diverse
        
        Selection Strategy:
        -------------------
        1. Pick the single best performing parameter (best score)
        2. For each additional parameter:
           a. Calculate distance from already-selected parameters
           b. Compute combined score = performance + diversity
           c. Pick parameter with best combined score
        3. Repeat until n_final_params selected
        
        Parameters:
        -----------
        channel : str, default='TRITC'
            Channel to optimize for ('TRITC' or 'FITC')
        diversity_weight : float, default=0.3
            How much to value diversity vs pure performance (0-1)
            - 0.0: Pure performance (pick top N best scores, ignore diversity)
            - 0.5: Equal weight to performance and diversity
            - 1.0: Pure diversity (maximize distance, ignore performance)
            Recommended: 0.2-0.4 for most cases
        
        Returns:
        --------
        list : Selected parameter combinations as list of tuples
            Each tuple: (bright_pct, contrast_thresh, percentile_val)
            Length: n_final_params (typically 1-3)
            Example: [(15.0, 2.5, 10.0), (20.0, 3.0, 15.0), (25.0, 2.0, 20.0)]
        
        Side Effects:
        -------------
        - Stores selected parameters in self.selected_params[channel]
        - Prints selection progress with scores and distances
        
        Example Output:
        ---------------
        🎯 Selected Parameter 1/3:
           (15.0, 2.5, 10.0)
           Mean foci: 8.15
           Std: 2.1
           Score: 0.42
        
        🎯 Selected Parameter 2/3:
           (25.0, 3.0, 20.0)
           Mean foci: 8.45
           Std: 2.8
           Score: 0.68
           Distance from selected: 12.5
        
        🎯 Selected Parameter 3/3:
           (10.0, 2.0, 5.0)
           Mean foci: 7.89
           Std: 2.3
           Score: 0.71
           Distance from selected: 11.2
        """
        # ----------------------------------------------------------------
        # Analyze calibration data to get parameter statistics
        # ----------------------------------------------------------------
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        # Initialize lists to track selected parameters
        selected = []          # Selected parameter tuples
        selected_indices = []  # Indices in param_stats DataFrame
        
        # ----------------------------------------------------------------
        # Extract parameter arrays for distance calculations
        # ----------------------------------------------------------------
        # Convert tuples to numpy array for efficient distance computation
        # Shape: (N_params, 3) where 3 = [bright_pct, contrast_thresh, percentile_val]
        param_arrays = np.array([list(p) for p in param_stats['param_combo'].values])
        
        # ================================================================
        # STEP 1: Select best performing parameter (no diversity yet)
        # ================================================================
        # Always start with the absolute best parameter (index 0 after sorting)
        best_idx = 0
        selected.append(param_stats.iloc[best_idx]['param_combo'])
        selected_indices.append(best_idx)
        
        # Print selection details
        print(f"\n🎯 Selected Parameter 1/{self.n_final_params}:")
        print(f"   {selected[0]}")
        print(f"   Mean foci: {param_stats.iloc[best_idx]['mean_foci']:.2f}")
        print(f"   Std: {param_stats.iloc[best_idx]['std_foci']:.2f}")
        print(f"   Score: {param_stats.iloc[best_idx]['score']:.3f}")
        
        # ================================================================
        # STEP 2: Select additional parameters balancing performance and diversity
        # ================================================================
        for i in range(1, self.n_final_params):
            # --------------------------------------------------------
            # Calculate distances from already selected parameters
            # --------------------------------------------------------
            # Get array of already-selected parameters
            selected_params = param_arrays[selected_indices]
            
            # Compute distance matrix: all parameters vs selected parameters
            # cdist is vectorized and efficient (much faster than loop)
            # Shape: (N_all_params, N_selected)
            all_distances = cdist(param_arrays, selected_params)
            
            # For each parameter, get minimum distance to ANY selected parameter
            # This tells us "how far is this parameter from the nearest selection"
            # Shape: (N_all_params,)
            min_distances = np.min(all_distances, axis=1)
            
            # Set distance to 0 for already-selected parameters
            # This prevents re-selecting them (they'll have worst combined score)
            min_distances[selected_indices] = 0
            
            # Convert to array (may already be, but ensure for consistency)
            min_distances = np.array(min_distances)
            
            # --------------------------------------------------------
            # Normalize distances to 0-1 range
            # --------------------------------------------------------
            # Normalization allows fair combination with normalized scores
            # max() > 0 check prevents division by zero (shouldn't happen)
            if min_distances.max() > 0:
                normalized_distances = min_distances / min_distances.max()
            else:
                # All distances are zero (shouldn't happen) - keep as is
                normalized_distances = min_distances
            
            # --------------------------------------------------------
            # Normalize scores to 0-1 range (lower is better)
            # --------------------------------------------------------
            # Divide by max so worst score = 1.0, best score ≈ 0.0
            normalized_scores = param_stats['score'].values / param_stats['score'].max()
            
            # --------------------------------------------------------
            # Combined score: balance performance and diversity
            # --------------------------------------------------------
            # Formula: (1-w)×score - w×distance
            # 
            # Performance term: (1 - diversity_weight) × normalized_scores
            # - Lower score = better performance
            # - Weight decreases as diversity_weight increases
            # 
            # Diversity term: -diversity_weight × normalized_distances
            # - Higher distance = more diverse (better)
            # - Negative sign because we MINIMIZE combined_scores
            # - Weight increases as diversity_weight increases
            # 
            # Example with diversity_weight=0.3:
            # - Performance contributes 70% to combined score
            # - Diversity contributes 30% to combined score
            combined_scores = (
                (1 - diversity_weight) * normalized_scores +  # Performance (lower better)
                (-diversity_weight) * normalized_distances     # Diversity (higher better)
            )
            
            # --------------------------------------------------------
            # Mask out already selected parameters
            # --------------------------------------------------------
            # Set combined score to infinity for selected parameters
            # This ensures they won't be selected again (argmin will skip them)
            combined_scores[selected_indices] = np.inf
            
            # --------------------------------------------------------
            # Select parameter with best combined score
            # --------------------------------------------------------
            next_idx = np.argmin(combined_scores)
            selected.append(param_stats.iloc[next_idx]['param_combo'])
            selected_indices.append(next_idx)
            
            # Print selection details
            print(f"\n🎯 Selected Parameter {i+1}/{self.n_final_params}:")
            print(f"   {selected[i]}")
            print(f"   Mean foci: {param_stats.iloc[next_idx]['mean_foci']:.2f}")
            print(f"   Std: {param_stats.iloc[next_idx]['std_foci']:.2f}")
            print(f"   Score: {param_stats.iloc[next_idx]['score']:.3f}")
            print(f"   Distance from selected: {min_distances[next_idx]:.3f}")
        
        # ----------------------------------------------------------------
        # Store selected parameters for later use
        # ----------------------------------------------------------------
        self.selected_params = {channel: selected}
        return selected
    
    
    def visualize_parameter_space(self, channel='TRITC', save_path=None):
        """
        Visualize parameter space and selected parameters.
        
        Creates a 3-panel figure showing:
        1. Brightness % vs Contrast Threshold (colored by score)
        2. Global Percentile vs Contrast Threshold (colored by score)
        3. Histogram of mean foci counts (showing selected parameters)
        
        This visualization helps understand:
        - How parameters are distributed in parameter space
        - Which regions of parameter space perform well (green) vs poorly (red)
        - Whether selected parameters are diverse or clustered
        - How selected parameters relate to the global mean foci count
        
        Parameters:
        -----------
        channel : str, default='TRITC'
            Channel to visualize ('TRITC' or 'FITC')
        save_path : str, optional
            If provided, save figure to this path as PNG
            If None, figure is only returned (not saved)
            
        Returns:
        --------
        matplotlib.figure.Figure : The generated figure object
            Note: Figure is CLOSED after creation to prevent display
            If you want to display it, remove the plt.close(fig) line
            
        Example Usage:
        --------------
```python
        # After calibration and selection:
        fig = selector.visualize_parameter_space(
            channel='TRITC',
            save_path='parameter_space_tritc.png'
        )
        ✅ Saved visualization to parameter_space_tritc.png
```
        
        Plot Elements:
        --------------
        - Green dots: Good parameters (low score)
        - Red dots: Poor parameters (high score)
        - Blue stars: Selected parameters for production
        - Red dashed line: Global mean foci count
        - Blue solid lines: Mean foci count for selected parameters
        """
        # ----------------------------------------------------------------
        # Analyze calibration data to get statistics
        # ----------------------------------------------------------------
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        # ----------------------------------------------------------------
        # Extract parameter components for plotting
        # ----------------------------------------------------------------
        # Convert parameter tuples to array and extract each component
        params = np.array([list(p) for p in param_stats['param_combo'].values])
        bright_pct = params[:, 0]      # Brightness percentile (column 0)
        contrast_thresh = params[:, 1]  # Contrast threshold (column 1)
        percentile_val = params[:, 2]   # Global percentile (column 2)
        
        # Extract scores for color-coding
        scores = param_stats['score'].values
        
        # ================================================================
        # Create figure with 3 subplots
        # ================================================================
        fig = plt.figure(figsize=(15, 5))
        
        # ================================================================
        # PLOT 1: Brightness % vs Contrast (colored by score)
        # ================================================================
        ax1 = fig.add_subplot(131)
        
        # Scatter plot of all tested parameters
        # Color represents score: green=good (low score), red=bad (high score)
        scatter1 = ax1.scatter(bright_pct, contrast_thresh, c=scores, 
                              cmap='RdYlGn_r', s=50, alpha=0.6)
        
        # ----------------------------------------------------------------
        # Mark selected parameters with blue stars
        # ----------------------------------------------------------------
        if self.selected_params and channel in self.selected_params:
            # Extract selected parameters
            selected = np.array([list(p) for p in self.selected_params[channel]])
            
            # Plot as large blue stars with black outline
            # zorder=5 ensures stars appear on top of other points
            ax1.scatter(selected[:, 0], selected[:, 1], 
                       marker='*', s=500, c='blue', 
                       edgecolors='black', linewidths=2,
                       label='Selected', zorder=5)
        
        # Labels and formatting
        ax1.set_xlabel('Brightness Percentile')
        ax1.set_ylabel('Contrast Threshold')
        ax1.set_title('Bright % vs Contrast')
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Score (lower=better)')
        
        # ================================================================
        # PLOT 2: Global Percentile vs Contrast (colored by score)
        # ================================================================
        ax2 = fig.add_subplot(132)
        
        # Same logic as Plot 1 but with different parameter axes
        scatter2 = ax2.scatter(percentile_val, contrast_thresh, c=scores,
                              cmap='RdYlGn_r', s=50, alpha=0.6)
        
        # Mark selected parameters
        if self.selected_params and channel in self.selected_params:
            # Use column 2 (percentile_val) for x-axis, column 1 (contrast) for y-axis
            ax2.scatter(selected[:, 2], selected[:, 1],
                       marker='*', s=500, c='blue',
                       edgecolors='black', linewidths=2,
                       label='Selected', zorder=5)
        
        # Labels and formatting
        ax2.set_xlabel('Global Percentile')
        ax2.set_ylabel('Contrast Threshold')
        ax2.set_title('Percentile vs Contrast')
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='Score (lower=better)')
        
        # ================================================================
        # PLOT 3: Histogram of mean foci counts
        # ================================================================
        ax3 = fig.add_subplot(133)
        
        # Histogram showing distribution of mean foci counts across all parameters
        # This shows the range of detection rates from different parameters
        ax3.hist(param_stats['mean_foci'], bins=30, alpha=0.6, label='All params')
        
        # ----------------------------------------------------------------
        # Add vertical line showing global mean
        # ----------------------------------------------------------------
        # Red dashed line = target foci count we want parameters to achieve
        ax3.axvline(global_mean, color='red', linestyle='--', 
                   linewidth=2, label=f'Global mean: {global_mean:.2f}')
        
        # ----------------------------------------------------------------
        # Add vertical lines for selected parameters' mean foci counts
        # ----------------------------------------------------------------
        if self.selected_params and channel in self.selected_params:
            selected_means = []
            
            # Extract mean foci count for each selected parameter
            for p in self.selected_params[channel]:
                # Find matching row in param_stats
                matches = param_stats[param_stats['param_combo'] == p]['mean_foci'].values
                
                if len(matches) > 0:
                    # Found matching parameter, add its mean to list
                    selected_means.append(matches[0])
                else:
                    # Selected parameter not found in stats (shouldn't happen)
                    print(f"⚠️ Warning: Selected parameter {p} not found in param_stats")
            
            # Plot vertical line for each selected parameter's mean
            # Blue solid lines show where selected parameters fall on distribution
            for i, mean_val in enumerate(selected_means):
                ax3.axvline(mean_val, color='blue', linestyle='-', 
                           linewidth=2, alpha=0.7, label=f'Selected {i+1}')
        
        # Labels and formatting
        ax3.set_xlabel('Mean Foci Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Foci Count Distribution')
        ax3.legend()
        
        # ----------------------------------------------------------------
        # Finalize figure
        # ----------------------------------------------------------------
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")

        # Close figure to prevent automatic display at end of script
        # If you want to display the figure, comment out this line
        plt.close(fig)
        
        return fig
    
    
    def save_calibration(self, filepath):
        """
        Save calibration results and selected parameters to disk.
        
        Saves all calibration data to a pickle file for later use. This allows:
        - Archiving calibration results with your data
        - Re-running selection with different diversity weights
        - Reproducing analysis with exact same parameters
        - Sharing calibration results with collaborators
        
        Parameters:
        -----------
        filepath : str
            Full path where to save calibration data (e.g., 'calibration.pkl')
            
        Saved Data:
        -----------
        - calibration_results: All recorded nucleus results
        - selected_params: Optimized parameter combinations
        - n_calibration_images: How many images were used
        - n_final_params: How many parameters were selected
        
        Example:
        --------
        >>> selector.save_calibration('calibration_results.pkl')
        ✅ Saved calibration data to calibration_results.pkl
        """
        # Package all relevant data into dictionary
        data = {
            'calibration_results': self.calibration_results,
            'selected_params': self.selected_params,
            'n_calibration_images': self.n_calibration_images,
            'n_final_params': self.n_final_params
        }
        
        # Save as pickle (preserves exact data structures)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved calibration data to {filepath}")
    
    
    def load_calibration(self, filepath):
        """
        Load previously saved calibration results.
        
        Restores a complete calibration session from disk, allowing you to:
        - Resume analysis from a previous session
        - Re-run selection with different parameters
        - Access calibration results without re-running calibration
        
        Parameters:
        -----------
        filepath : str
            Path to saved calibration file (from save_calibration())
            
        Side Effects:
        -------------
        Overwrites current calibration state with loaded data
        
        Example:
        --------
        >>> selector = AdaptiveParameterSelector()
        >>> selector.load_calibration('calibration_results.pkl')
        ✅ Loaded calibration data from calibration_results.pkl
           Calibration images: 5
           Selected parameters: 3
        """
        # Load data from pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore all calibration state
        self.calibration_results = data['calibration_results']
        self.selected_params = data['selected_params']
        self.n_calibration_images = data['n_calibration_images']
        self.n_final_params = data['n_final_params']
        
        # Print summary of loaded data
        print(f"✅ Loaded calibration data from {filepath}")
        print(f"   Calibration images: {self.n_calibration_images}")
        print(f"   Selected parameters: {self.n_final_params}")



    
    def select_optimal_parameters_auto(self, channel='TRITC'):
        """
        Automatically determine optimal diversity weight and select parameters.
        
        This is a CONVENIENCE WRAPPER around select_optimal_parameters() that
        automatically chooses an appropriate diversity_weight based on:
        1. How many parameters you're requesting (n_final_params)
        2. How variable the data is (coefficient of variation across parameters)
        
        Why auto-tune diversity weight?
        - Different datasets need different diversity levels
        - Homogeneous data (low variance) → less diversity needed
        - Heterogeneous data (high variance) → more diversity needed
        - More parameters requested → more diversity needed
        
        Decision Logic:
        ---------------
        If requesting 1 parameter:
            diversity_weight = 0.0 (just pick the best, no diversity)
        
        If requesting 2 parameters:
            Low variance (CV < 0.4): diversity_weight = 0.15
            High variance (CV ≥ 0.4): diversity_weight = 0.3
        
        If requesting 3 parameters:
            Low variance (CV < 0.3): diversity_weight = 0.2
            Medium variance (0.3 ≤ CV < 0.5): diversity_weight = 0.3
            High variance (CV ≥ 0.5): diversity_weight = 0.5
        
        If requesting 4+ parameters:
            Low variance (CV < 0.4): diversity_weight = 0.4
            High variance (CV ≥ 0.4): diversity_weight = 0.6
        
        Parameters:
        -----------
        channel : str, default='TRITC'
            Channel to optimize for ('TRITC' or 'FITC')
            
        Returns:
        --------
        list : Selected parameter combinations
            Same return value as select_optimal_parameters()
            
        Example Usage:
        --------------
```python
        # Simple one-line selection with automatic tuning
        optimal_params = selector.select_optimal_parameters_auto('TRITC')
        
        # Instead of manually choosing diversity_weight:
        # optimal_params = selector.select_optimal_parameters('TRITC', diversity_weight=0.3)
```
        
        Output Example:
        ---------------
        📊 TRITC Calibration Analysis:
           Global mean foci per nucleus: 8.23
        
        Requesting 3 parameters → diversity_weight=0.3
        Rationale: With 3 parameter(s), balancing performance vs coverage
        
        🎯 Selected Parameter 1/3: ...
        """
        # ----------------------------------------------------------------
        # Analyze data to get variance metrics
        # ----------------------------------------------------------------
        param_stats, global_mean = self.analyze_calibration_data(channel)
        
        # ----------------------------------------------------------------
        # Calculate coefficient of variation across all parameters
        # ----------------------------------------------------------------
        # CV = std / mean of the 'mean_foci' column
        # High CV = parameters give very different foci counts (variable)
        # Low CV = parameters give similar foci counts (consistent)
        cv_across_params = param_stats['mean_foci'].std() / param_stats['mean_foci'].mean()
        
        # ================================================================
        # Auto-select diversity weight based on n_final_params and CV
        # ================================================================
        
        if self.n_final_params == 1:
            # --------------------------------------------------------
            # Only 1 parameter: No diversity needed
            # --------------------------------------------------------
            # Just pick the best performer, diversity is meaningless
            diversity_weight = 0.0
            print(f"  Requesting 1 parameter → diversity_weight=0.0 (pure performance)")
            
        elif self.n_final_params == 2:
            # --------------------------------------------------------
            # 2 parameters: Light diversity
            # --------------------------------------------------------
            if cv_across_params > 0.4:
                # High variance: parameters behave very differently
                # Use moderate diversity to get two complementary approaches
                diversity_weight = 0.3
                print(f"  2 parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.3")
            else:
                # Low variance: parameters behave similarly
                # Use light diversity, focus more on performance
                diversity_weight = 0.15
                print(f"  2 parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.15")
                
        elif self.n_final_params == 3:
            # --------------------------------------------------------
            # 3 parameters: Balanced diversity (default case)
            # --------------------------------------------------------
            if cv_across_params > 0.5:
                # Very high variance: wide range of behaviors
                # Use high diversity to sample different approaches
                diversity_weight = 0.5
                print(f"  3 parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.5")
            elif cv_across_params > 0.3:
                # Medium variance: moderate variation
                # Use balanced diversity (default recommended)
                diversity_weight = 0.3
                print(f"  3 parameters, medium variance (CV={cv_across_params:.2f}) → diversity_weight=0.3")
            else:
                # Low variance: parameters behave very similarly
                # Use light diversity, emphasize performance
                diversity_weight = 0.2
                print(f"  3 parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.2")
                
        else:  # 4+ parameters
            # --------------------------------------------------------
            # Many parameters: Need more diversity
            # --------------------------------------------------------
            # With 4+ parameters, we want good coverage of parameter space
            if cv_across_params > 0.4:
                # High variance: use high diversity
                diversity_weight = 0.6
                print(f"  {self.n_final_params} parameters, high variance (CV={cv_across_params:.2f}) → diversity_weight=0.6")
            else:
                # Low variance: use moderate diversity
                diversity_weight = 0.4
                print(f"  {self.n_final_params} parameters, low variance (CV={cv_across_params:.2f}) → diversity_weight=0.4")
        
        # Print rationale
        print(f"  Rationale: With {self.n_final_params} parameter(s), balancing performance vs coverage")
        
        # ----------------------------------------------------------------
        # Call main selection function with auto-tuned diversity weight
        # ----------------------------------------------------------------
        return self.select_optimal_parameters(channel, diversity_weight)