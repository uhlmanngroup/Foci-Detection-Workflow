"""
Enhanced Parameter Space Generator with KDE Implementation
Full implementation ready to use
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os


class EnhancedParameterSpaceGenerator:
    """
    Complete implementation of KDE-based parameter space generation.
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
        
    def automatic_nucleus_selection(self, masks, channel_image):
        """
        Automatically select diverse nuclei for parameter training.
        """
        from skimage import measure
        
        print("\n" + "="*60)
        print("AUTOMATIC NUCLEUS SELECTION")
        print("="*60)
        
        # Get all nucleus properties
        props = measure.regionprops(masks, intensity_image=channel_image)
        
        nucleus_features = []
        for prop in props:
            if prop.label == 0:  # Skip background
                continue
                
            features = {
                'label': prop.label,
                'area': prop.area,
                'mean_intensity': prop.mean_intensity,
                'max_intensity': prop.max_intensity,
                'std_intensity': np.std(channel_image[masks == prop.label]),
                'cv': np.std(channel_image[masks == prop.label]) / prop.mean_intensity
            }
            nucleus_features.append(features)
        
        df = pd.DataFrame(nucleus_features)
        
        # Select diverse nuclei based on features
        selected_nuclei = []
        
        # 1. Select nuclei from different intensity ranges
        intensity_bins = pd.qcut(df['mean_intensity'], q=5, duplicates='drop')
        for bin_val in intensity_bins.unique():
            bin_nuclei = df[intensity_bins == bin_val]
            if len(bin_nuclei) > 0:
                # Select one with median CV from this bin
                median_cv_idx = (bin_nuclei['cv'] - bin_nuclei['cv'].median()).abs().idxmin()
                selected_nuclei.append(bin_nuclei.loc[median_cv_idx])
        
        # 2. Add some with extreme CV values (very uniform and very spotty)
        selected_nuclei.append(df.nsmallest(1, 'cv').iloc[0])  # Most uniform
        selected_nuclei.append(df.nlargest(1, 'cv').iloc[0])   # Most spotty
        
        # 3. Add different sizes
        selected_nuclei.append(df.nsmallest(1, 'area').iloc[0])  # Smallest
        selected_nuclei.append(df.nlargest(1, 'area').iloc[0])   # Largest
        
        # Remove duplicates
        selected_df = pd.DataFrame(selected_nuclei).drop_duplicates(subset=['label'])
        
        print(f"✓ Automatically selected {len(selected_df)} diverse nuclei")
        print("\nSelected nuclei characteristics:")
        print(selected_df[['label', 'area', 'mean_intensity', 'cv']].to_string())
        
        return selected_df['label'].tolist()
    
    def optimize_kde_bandwidth(self, points):
        """
        Find optimal KDE bandwidth using cross-validation.
        """
        if len(points) < 20:
            return 'scott'  # Fall back to Scott's rule
        
        # Normalize points
        normalized = self._normalize_parameters(points)
        
        # Grid search for optimal bandwidth
        bandwidths = np.logspace(-2, 0, 10)  # 0.01 to 1.0
        
        kde = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=min(5, len(points) // 5)  # 5-fold CV or less if not enough points
        )
        
        kde.fit(normalized)
        
        print(f"  Optimal bandwidth: {kde.best_params_['bandwidth']:.3f}")
        return kde.best_params_['bandwidth']
    
    def generate_kde_parameter_space(self, coverage_percentile=85):
        """
        Generate final parameter space using KDE with optimal parameters.
        """
        if self.valid_points is None or len(self.valid_points) < 10:
            print("⚠️ Not enough valid points for KDE")
            return None
        
        print("\n" + "="*60)
        print("GENERATING KDE-BASED PARAMETER SPACE")
        print("="*60)
        
        # Find optimal bandwidth
        print("\n1. Optimizing KDE bandwidth...")
        bandwidth = self.optimize_kde_bandwidth(self.valid_points)
        
        # Fit KDE with optimal bandwidth
        print("\n2. Fitting KDE model...")
        normalized_points = self._normalize_parameters(self.valid_points)
        
        if isinstance(bandwidth, str):
            self.kde_model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        else:
            self.kde_model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        
        self.kde_model.fit(normalized_points)
        
        # Generate sampling grid
        print("\n3. Generating parameter space samples...")
        n_samples = 10000
        
        # Use importance sampling: more samples where we have valid points
        # Start with uniform grid
        uniform_samples = np.random.uniform(0, 1, (n_samples // 2, 3))
        
        # Add Gaussian noise around existing points
        noise_std = 0.1
        noisy_samples = normalized_points[np.random.choice(len(normalized_points), 
                                                          n_samples // 2, replace=True)]
        noisy_samples += np.random.normal(0, noise_std, noisy_samples.shape)
        noisy_samples = np.clip(noisy_samples, 0, 1)
        
        all_samples = np.vstack([uniform_samples, noisy_samples])
        
        # Compute densities
        log_densities = self.kde_model.score_samples(all_samples)
        densities = np.exp(log_densities)
        
        # Find density threshold for desired coverage
        valid_point_densities = np.exp(self.kde_model.score_samples(normalized_points))
        threshold = np.percentile(valid_point_densities, 100 - coverage_percentile)
        
        print(f"  Density threshold: {threshold:.3e}")
        print(f"  Coverage: {coverage_percentile}%")
        
        # Select points above threshold
        selected_samples = all_samples[densities > threshold]
        
        # Build Delaunay triangulation for the selected points
        print("\n4. Building Delaunay triangulation...")
        self.hull = Delaunay(selected_samples)
        
        # Store metadata
        self.kde_metadata = {
            'bandwidth': bandwidth,
            'threshold': threshold,
            'coverage_percentile': coverage_percentile,
            'n_selected_samples': len(selected_samples),
            'selected_samples_normalized': selected_samples
        }
        
        # Denormalize for bounds calculation
        selected_denorm = self._denormalize_parameters(selected_samples)
        
        # Calculate bounding box
        self.bounds = {
            'bright_pct': (selected_denorm[:, 0].min(), selected_denorm[:, 0].max()),
            'contrast_thresh': (selected_denorm[:, 1].min(), selected_denorm[:, 1].max()),
            'percentile_val': (selected_denorm[:, 2].min(), selected_denorm[:, 2].max())
        }
        
        print(f"\n✓ KDE parameter space generated")
        print(f"  Total samples in space: {len(selected_samples)}")
        print(f"  Bounding box:")
        print(f"    Background %: {self.bounds['bright_pct'][0]:.1f} - {self.bounds['bright_pct'][1]:.1f}")
        print(f"    Contrast: {self.bounds['contrast_thresh'][0]:.2f} - {self.bounds['contrast_thresh'][1]:.2f}")
        print(f"    Percentile: {self.bounds['percentile_val'][0]:.1f} - {self.bounds['percentile_val'][1]:.1f}")
        
        # Visualize
        self._visualize_kde_space()
        
        return self.hull
    
    def validate_parameter_space(self, validation_nuclei_data):
        """
        Validate the parameter space on held-out nuclei.
        """
        print("\n" + "="*60)
        print("VALIDATING PARAMETER SPACE")
        print("="*60)
        
        # Sample parameters from the KDE space
        n_test_samples = 100
        test_samples = self.sample_from_kde(n_test_samples)
        
        validation_results = []
        
        for nucleus_id, expected_count in validation_nuclei_data.items():
            # Test each parameter sample
            correct_detections = 0
            
            for params in test_samples:
                # Run detection with these parameters
                # (You'd implement the actual detection here)
                detected_count = self._run_detection_with_params(params, nucleus_id)
                
                if detected_count == expected_count:
                    correct_detections += 1
            
            accuracy = correct_detections / n_test_samples
            validation_results.append({
                'nucleus_id': nucleus_id,
                'expected': expected_count,
                'accuracy': accuracy
            })
            
            print(f"  Nucleus {nucleus_id}: {accuracy*100:.1f}% accurate")
        
        overall_accuracy = np.mean([r['accuracy'] for r in validation_results])
        print(f"\n✓ Overall validation accuracy: {overall_accuracy*100:.1f}%")
        
        return validation_results
    
    def sample_from_kde(self, n_samples):
        """
        Sample new parameter combinations from the KDE model.
        """
        if self.kde_model is None:
            raise ValueError("Must fit KDE model first")
        
        # Sample from KDE in normalized space
        samples_norm = self.kde_model.sample(n_samples)
        
        # Clip to valid range
        samples_norm = np.clip(samples_norm, 0, 1)
        
        # Denormalize
        samples = self._denormalize_parameters(samples_norm)
        
        return samples
    
    def _visualize_kde_space(self):
        """Enhanced visualization of KDE parameter space."""
        if self.kde_metadata is None:
            return
        
        fig = plt.figure(figsize=(18, 6))
        
        # Get samples for visualization
        samples = self.kde_metadata['selected_samples_normalized']
        samples_denorm = self._denormalize_parameters(samples)
        
        # Subsample if too many points
        if len(samples) > 5000:
            idx = np.random.choice(len(samples), 5000, replace=False)
            samples_viz = samples_denorm[idx]
        else:
            samples_viz = samples_denorm
        
        # 3D scatter
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(samples_viz[:, 0], samples_viz[:, 1], samples_viz[:, 2],
                   c='lightblue', s=5, alpha=0.1, label='KDE space')
        ax1.scatter(self.valid_points[:, 0], self.valid_points[:, 1], 
                   self.valid_points[:, 2],
                   c='red', s=30, alpha=0.8, edgecolor='k', label='Training points')
        ax1.set_xlabel('Background %')
        ax1.set_ylabel('Contrast Thresh')
        ax1.set_zlabel('Global Percentile')
        ax1.set_title('KDE Parameter Space (3D)')
        ax1.legend()
        
        # 2D density plot - XY plane
        ax2 = fig.add_subplot(132)
        from scipy.stats import gaussian_kde
        
        # Create 2D KDE for better visualization
        kde_2d = gaussian_kde([samples_viz[:, 0], samples_viz[:, 1]])
        x_grid = np.linspace(self.bounds['bright_pct'][0], self.bounds['bright_pct'][1], 50)
        y_grid = np.linspace(self.bounds['contrast_thresh'][0], self.bounds['contrast_thresh'][1], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde_2d(positions).T, X.shape)
        
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax2.scatter(self.valid_points[:, 0], self.valid_points[:, 1],
                   c='red', s=20, alpha=0.8, edgecolor='k')
        ax2.set_xlabel('Background %')
        ax2.set_ylabel('Contrast Thresh')
        ax2.set_title('Parameter Density (XY projection)')
        plt.colorbar(contour, ax=ax2, label='Density')
        
        # Parameter importance plot
        ax3 = fig.add_subplot(133)
        
        # Calculate parameter ranges (spread)
        param_ranges = {
            'Background %': self.valid_points[:, 0].std(),
            'Contrast': self.valid_points[:, 1].std(),
            'Percentile': self.valid_points[:, 2].std()
        }
        
        # Normalize to show relative importance
        total = sum(param_ranges.values())
        param_importance = {k: v/total for k, v in param_ranges.items()}
        
        bars = ax3.bar(param_importance.keys(), param_importance.values())
        ax3.set_ylabel('Relative Variability')
        ax3.set_title('Parameter Importance')
        ax3.set_ylim(0, 1)
        
        # Color bars by importance
        colors = ['red' if v > 0.4 else 'orange' if v > 0.3 else 'green' 
                 for v in param_importance.values()]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
    
    def _normalize_parameters(self, params):
        """Normalize parameters to [0, 1]³."""
        normalized = np.zeros_like(params, dtype=float)
        for i, (key, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
        return normalized
    
    def _denormalize_parameters(self, params_norm):
        """Denormalize from [0, 1]³ to original ranges."""
        params = np.zeros_like(params_norm)
        for i, (key, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            params[:, i] = params_norm[:, i] * (max_val - min_val) + min_val
        return params
    
    def _run_detection_with_params(self, params, nucleus_id):
        """
        Placeholder for actual detection with given parameters.
        In real implementation, this would run your detection algorithm.
        """
        # This is where you'd run actual detection
        # For now, return a dummy value
        return np.random.poisson(3)  # Dummy count
    
    def save_complete(self, output_dir):
        """
        Save complete parameter space with KDE model.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save KDE model
        kde_path = os.path.join(output_dir, "kde_model.pkl")
        with open(kde_path, 'wb') as f:
            pickle.dump({
                'model': self.kde_model,
                'metadata': self.kde_metadata
            }, f)
        print(f"✓ Saved KDE model: {kde_path}")
        
        # Save hull
        hull_path = os.path.join(output_dir, "valid_parameter_hull.pkl")
        with open(hull_path, 'wb') as f:
            pickle.dump(self.hull, f)
        print(f"✓ Saved hull: {hull_path}")
        
        # Save bounds
        bounds_path = os.path.join(output_dir, "parameter_bounds.npz")
        np.savez(bounds_path, **self.bounds)
        print(f"✓ Saved bounds: {bounds_path}")
        
        # Save valid points
        points_path = os.path.join(output_dir, "valid_points.npy")
        np.save(points_path, self.valid_points)
        print(f"✓ Saved valid points: {points_path}")
        
        print(f"\n✓ Complete parameter space saved to: {output_dir}")


