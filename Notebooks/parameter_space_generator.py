"""
Interactive Parameter Space Generator for Foci Detection
Automated pipeline with visualization and nucleus selection interface
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import qmc
import pandas as pd
import pickle
from skimage import exposure, filters, img_as_float
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from collections import Counter
import os


class ParameterSpaceGenerator:
    """
    Automated pipeline for generating robust parameter sampling spaces
    for foci detection validation.
    """
    
    def __init__(self, param_ranges, resolution=20):
        """
        Initialize the parameter space generator.
        
        Parameters:
        -----------
        param_ranges : dict
            Parameter ranges, e.g.:
            {
                'bright_pct': (0, 100),
                'contrast_thresh': (1, 10),
                'percentile_val': (0, 100)
            }
        resolution : int
            Grid resolution for parameter sweep (default 20)
        """
        self.param_ranges = param_ranges
        self.resolution = resolution
        self.ground_truth_nuclei = {}  # {cell_id: expected_count}
        self.grid_results = None  # Will store full parameter sweep results
        self.valid_points = None  # Points valid for all nuclei
        self.hull = None
        self.bounds = None
        
    def add_nucleus(self, cell_id, expected_count):
        """
        Register a ground truth nucleus for validation.
        
        Parameters:
        -----------
        cell_id : int
            Nucleus ID from segmentation mask
        expected_count : int
            Expected number of foci (ground truth)
        """
        self.ground_truth_nuclei[cell_id] = expected_count
        print(f"✓ Added nucleus {cell_id} with expected count: {expected_count}")
    
    def generate_grid_search(self, masks, channel_image, original_image):
        """
        Run parameter grid search for all registered nuclei.
        
        Parameters:
        -----------
        masks : ndarray
            Segmentation mask with labeled nuclei
        channel_image : ndarray
            Channel image to analyze (e.g., TRITC)
        original_image : ndarray
            Original unfiltered image for global percentile calculations
            
        Returns:
        --------
        pd.DataFrame : Results with columns [cell_num, bright_pct, contrast_thresh, 
                       percentile_val, foci_count]
        """
        print("\n" + "="*60)
        print("STEP 2: GRID SEARCH OVER PARAMETER SPACE")
        print("="*60)
        
        # Generate parameter grid
        bright_grid = np.linspace(self.param_ranges['bright_pct'][0], 
                                 self.param_ranges['bright_pct'][1], 
                                 self.resolution)
        contrast_grid = np.linspace(self.param_ranges['contrast_thresh'][0], 
                                   self.param_ranges['contrast_thresh'][1], 
                                   self.resolution)
        percentile_grid = np.linspace(self.param_ranges['percentile_val'][0], 
                                     self.param_ranges['percentile_val'][1], 
                                     self.resolution)
        
        total_combinations = len(bright_grid) * len(contrast_grid) * len(percentile_grid)
        print(f"\nTesting {total_combinations} parameter combinations per nucleus...")
        print(f"Grid resolution: {self.resolution}³ = {total_combinations}")
        print(f"Testing on {len(self.ground_truth_nuclei)} nuclei")
        
        results = []
        
        # Process each registered nucleus
        for cell_id in self.ground_truth_nuclei.keys():
            print(f"\n  Processing nucleus {cell_id}...")
            nucleus_mask = (masks == cell_id)
            
            if not np.any(nucleus_mask):
                print(f"    ⚠️ Warning: Nucleus {cell_id} not found in mask")
                continue
            
            # Isolate nucleus region
            isolated_img = img_as_float(channel_image.copy())
            isolated_img[~nucleus_mask] = 0
            
            if isolated_img.max() == 0:
                print(f"    ⚠️ Warning: Nucleus {cell_id} has no signal")
                continue
            
            # Apply DoG filter
            filtered_img = filters.difference_of_gaussians(isolated_img, low_sigma=1, high_sigma=2)
            filtered_img = np.clip(filtered_img, 0, None)
            filtered_img = exposure.rescale_intensity(filtered_img, in_range='image', 
                                                     out_range=(0, isolated_img.max()))
            
            # Calculate global percentiles from original image
            pos_pixels = original_image[original_image > 0]
            if pos_pixels.size == 0:
                print(f"    ⚠️ Warning: No positive pixels in original image")
                continue
            
            nucleus_count = 0
            
            # Test each parameter combination
            for bright_pct in bright_grid:
                for contrast_thresh in contrast_grid:
                    for percentile_val in percentile_grid:
                        # Calculate minimum brightness threshold
                        min_brightness = np.percentile(pos_pixels, percentile_val)
                        
                        # Find candidates in both filtered and unfiltered
                        candidates_filtered = peak_local_max(filtered_img, min_distance=2, 
                                                            threshold_abs=min_brightness)
                        candidates_unfiltered = peak_local_max(isolated_img, min_distance=2, 
                                                              threshold_abs=min_brightness)
                        
                        if len(candidates_filtered) == 0 or len(candidates_unfiltered) == 0:
                            foci_count = 0
                        else:
                            # Extract coordinates and intensities
                            filt_yx = np.asarray(candidates_filtered, dtype=int)
                            unf_yx = np.asarray(candidates_unfiltered, dtype=int)
                            filt_intensities = filtered_img[filt_yx[:, 0], filt_yx[:, 1]]
                            unf_intensities = isolated_img[unf_yx[:, 0], unf_yx[:, 1]]
                            
                            # Apply absolute brightness filter
                            unf_bright_mask = unf_intensities >= min_brightness
                            filt_bright_mask = filt_intensities >= min_brightness
                            
                            # Compute local backgrounds (simplified version)
                            unf_local_bg = self._compute_simple_local_background(
                                isolated_img, unf_yx, bright_pct, nucleus_mask
                            )
                            filt_local_bg = self._compute_simple_local_background(
                                filtered_img, filt_yx, bright_pct, nucleus_mask
                            )
                            
                            # Apply contrast filter
                            unf_contrast_mask = unf_intensities > (unf_local_bg * contrast_thresh)
                            filt_contrast_mask = filt_intensities > (filt_local_bg * contrast_thresh)
                            
                            # Combine filters
                            unf_final_mask = unf_bright_mask & unf_contrast_mask
                            filt_final_mask = filt_bright_mask & filt_contrast_mask
                            
                            unf_filtered = unf_yx[unf_final_mask]
                            filt_filtered = filt_yx[filt_final_mask]
                            
                            if len(unf_filtered) == 0 or len(filt_filtered) == 0:
                                foci_count = 0
                            else:
                                # Match with tolerance
                                distances = cdist(unf_filtered, filt_filtered)
                                nearest_dist = np.min(distances, axis=1)
                                confirmed = unf_filtered[nearest_dist <= 2]
                                foci_count = len(confirmed)
                        
                        results.append({
                            'cell_num': cell_id,
                            'bright_pct': bright_pct,
                            'contrast_thresh': contrast_thresh,
                            'percentile_val': percentile_val,
                            'foci_count': foci_count
                        })
                        nucleus_count += 1
            
            print(f"    ✓ Tested {nucleus_count} combinations")
        
        self.grid_results = pd.DataFrame(results)
        print(f"\n✓ Grid search complete: {len(self.grid_results)} total results")
        
        # Visualize grid search results
        self._visualize_grid_search()
        
        return self.grid_results
    
    def _compute_simple_local_background(self, image, coords, percentile, nucleus_mask, 
                                        inner_radius=2, outer_radius=6):
        """Simplified local background calculation for grid search"""
        backgrounds = np.zeros(len(coords))
        
        # Create annulus mask
        y_grid, x_grid = np.ogrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]
        distances = np.sqrt(x_grid**2 + y_grid**2)
        annulus = (distances >= inner_radius) & (distances <= outer_radius)
        annulus_y, annulus_x = np.where(annulus)
        annulus_y -= outer_radius
        annulus_x -= outer_radius
        
        for i, (y, x) in enumerate(coords):
            abs_y = y + annulus_y
            abs_x = x + annulus_x
            
            valid = (abs_y >= 0) & (abs_y < image.shape[0]) & \
                    (abs_x >= 0) & (abs_x < image.shape[1])
            
            if nucleus_mask is not None:
                valid_indices = np.where(valid)[0]
                nucleus_valid = nucleus_mask[abs_y[valid_indices], abs_x[valid_indices]] > 0
                valid[valid_indices] = nucleus_valid
            
            if valid.sum() >= 5:
                annulus_pixels = image[abs_y[valid], abs_x[valid]]
                backgrounds[i] = np.percentile(annulus_pixels, percentile)
            else:
                backgrounds[i] = image[y, x]
        
        return backgrounds
    
    def _visualize_grid_search(self):
        """Visualize parameter sweep results for each nucleus"""
        if self.grid_results is None:
            return
        
        fig = plt.figure(figsize=(16, 4 * len(self.ground_truth_nuclei)))
        
        for idx, (cell_id, expected_count) in enumerate(self.ground_truth_nuclei.items()):
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            
            if len(nucleus_data) == 0:
                continue
            
            ax = fig.add_subplot(len(self.ground_truth_nuclei), 1, idx + 1, projection='3d')
            
            # Color by whether count matches expected
            correct = nucleus_data['foci_count'] == expected_count
            colors = ['green' if c else 'red' for c in correct]
            
            ax.scatter(nucleus_data['bright_pct'],
                      nucleus_data['contrast_thresh'],
                      nucleus_data['percentile_val'],
                      c=colors, s=10, alpha=0.3)
            
            ax.set_xlabel('Background %')
            ax.set_ylabel('Contrast Thresh')
            ax.set_zlabel('Global Percentile')
            ax.set_title(f'Nucleus {cell_id} (Expected: {expected_count} foci)\n'
                        f'Green = Correct, Red = Incorrect')
        
        plt.tight_layout()
        plt.show()
        print("\n✓ Grid search visualization complete")
    
    def find_valid_intersection(self):
        """
        Find parameter combinations valid for ALL registered nuclei.
        
        Returns:
        --------
        pd.DataFrame : Valid parameter combinations
        """
        print("\n" + "="*60)
        print("STEP 3: FINDING VALID PARAMETER INTERSECTION")
        print("="*60)
        
        if self.grid_results is None:
            raise ValueError("Must run generate_grid_search() first")
        
        # Start with all parameter combinations
        all_params = self.grid_results[['bright_pct', 'contrast_thresh', 'percentile_val']].drop_duplicates()
        print(f"\nTotal unique parameter combinations: {len(all_params)}")
        
        # For each nucleus, find valid parameters
        valid_sets = []
        for cell_id, expected_count in self.ground_truth_nuclei.items():
            nucleus_data = self.grid_results[self.grid_results['cell_num'] == cell_id]
            valid_for_nucleus = nucleus_data[nucleus_data['foci_count'] == expected_count]
            valid_sets.append(valid_for_nucleus[['bright_pct', 'contrast_thresh', 'percentile_val']])
            print(f"  Nucleus {cell_id}: {len(valid_for_nucleus)} valid combinations")
        
        # Find intersection (parameters valid for ALL nuclei)
        if len(valid_sets) == 1:
            intersection = valid_sets[0]
        else:
            intersection = valid_sets[0]
            for valid_set in valid_sets[1:]:
                intersection = pd.merge(intersection, valid_set, 
                                       on=['bright_pct', 'contrast_thresh', 'percentile_val'])
        
        self.valid_points = intersection[['bright_pct', 'contrast_thresh', 'percentile_val']].values
        
        print(f"\n✓ Found {len(self.valid_points)} valid parameter combinations")
        print(f"  Coverage: {len(self.valid_points) / len(all_params) * 100:.1f}% of parameter space")
        
        if len(self.valid_points) == 0:
            print("\n⚠️ WARNING: No valid parameters found!")
            print("   Consider:")
            print("   - Expanding parameter ranges")
            print("   - Using fewer validation nuclei")
            print("   - Checking ground truth labels")
            return None
        
        # Visualize intersection
        self._visualize_intersection()
        
        return pd.DataFrame(self.valid_points, 
                          columns=['bright_pct', 'contrast_thresh', 'percentile_val'])
    
    def _visualize_intersection(self):
        """Visualize the valid parameter intersection"""
        if self.valid_points is None or len(self.valid_points) == 0:
            return
        
        fig = plt.figure(figsize=(14, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(self.valid_points[:, 0],
                   self.valid_points[:, 1],
                   self.valid_points[:, 2],
                   c='limegreen', s=30, alpha=0.6, edgecolor='k')
        ax1.set_xlabel('Background %')
        ax1.set_ylabel('Contrast Thresh')
        ax1.set_zlabel('Global Percentile')
        ax1.set_title(f'Valid Parameter Space\n({len(self.valid_points)} points)')
        
        # 2D projections
        ax2 = fig.add_subplot(122)
        ax2.scatter(self.valid_points[:, 0], self.valid_points[:, 1],
                   c='limegreen', s=20, alpha=0.5, edgecolor='k')
        ax2.set_xlabel('Background %')
        ax2.set_ylabel('Contrast Thresh')
        ax2.set_title('2D Projection (Background vs Contrast)')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("\n✓ Intersection visualization complete")
    
    def fit_robust_hull(self, method='alpha_shape', alpha=None, target_coverage=0.95):
        """
        Fit a hull around the valid parameter region.
        
        Parameters:
        -----------
        method : str
            'alpha_shape' or 'convex_hull'
        alpha : float or None
            Alpha value for alpha shape. If None, automatically determined.
        target_coverage : float
            Target fraction of points to include (for auto alpha)
            
        Returns:
        --------
        scipy.spatial.Delaunay : The fitted hull
        """
        print("\n" + "="*60)
        print("STEP 4: FITTING ROBUST HULL")
        print("="*60)
        
        if self.valid_points is None or len(self.valid_points) == 0:
            raise ValueError("Must run find_valid_intersection() first")
        
        # Normalize parameters to [0, 1] for better geometry
        points_normalized = self._normalize_parameters(self.valid_points)
        
        if method == 'alpha_shape':
            print(f"\nFitting alpha shape (method: {method})")
            
            if alpha is None:
                print(f"  Auto-detecting optimal alpha (target coverage: {target_coverage*100:.0f}%)...")
                alpha = self._find_optimal_alpha(points_normalized, target_coverage)
                print(f"  ✓ Optimal alpha: {alpha:.3f}")
            else:
                print(f"  Using provided alpha: {alpha:.3f}")
            
            # Build alpha shape
            hull_normalized = self._alpha_shape_3d(points_normalized, alpha)
            
            # Check coverage
            inside = hull_normalized.find_simplex(points_normalized) >= 0
            coverage = inside.mean()
            print(f"  ✓ Coverage: {coverage*100:.1f}% of valid points")
            
        else:  # convex_hull
            print(f"\nFitting convex hull (method: {method})")
            hull_normalized = Delaunay(points_normalized)
            coverage = 1.0
            print(f"  ✓ Coverage: 100% (convex hull includes all points)")
        
        # Convert back to original scale for storage
        self.hull = hull_normalized
        self.hull._original_points = self.valid_points  # Store for denormalization
        self.hull._param_ranges = self.param_ranges
        
        # Calculate bounding box
        self.bounds = {
            'bright_pct': (self.valid_points[:, 0].min(), self.valid_points[:, 0].max()),
            'contrast_thresh': (self.valid_points[:, 1].min(), self.valid_points[:, 1].max()),
            'percentile_val': (self.valid_points[:, 2].min(), self.valid_points[:, 2].max())
        }
        
        print(f"\n✓ Hull fitted successfully")
        print(f"  Bounding box:")
        print(f"    Background %: {self.bounds['bright_pct'][0]:.1f} - {self.bounds['bright_pct'][1]:.1f}")
        print(f"    Contrast: {self.bounds['contrast_thresh'][0]:.2f} - {self.bounds['contrast_thresh'][1]:.2f}")
        print(f"    Percentile: {self.bounds['percentile_val'][0]:.1f} - {self.bounds['percentile_val'][1]:.1f}")
        
        # Visualize hull
        self._visualize_hull()
        
        return self.hull
    
    def _normalize_parameters(self, params):
        """Normalize parameters to [0, 1]³"""
        normalized = np.zeros_like(params, dtype=float)
        normalized[:, 0] = (params[:, 0] - self.param_ranges['bright_pct'][0]) / \
                          (self.param_ranges['bright_pct'][1] - self.param_ranges['bright_pct'][0])
        normalized[:, 1] = (params[:, 1] - self.param_ranges['contrast_thresh'][0]) / \
                          (self.param_ranges['contrast_thresh'][1] - self.param_ranges['contrast_thresh'][0])
        normalized[:, 2] = (params[:, 2] - self.param_ranges['percentile_val'][0]) / \
                          (self.param_ranges['percentile_val'][1] - self.param_ranges['percentile_val'][0])
        return normalized
    
    def _denormalize_parameters(self, params_norm):
        """Convert normalized parameters back to original scale"""
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
    
    def _find_optimal_alpha(self, points, target_coverage=0.95):
        """Automatically find optimal alpha for alpha shape"""
        alphas = np.logspace(-1, 1, 20)  # 0.1 to 10
        
        best_alpha = None
        best_score = -np.inf
        
        for alpha in alphas:
            try:
                hull = self._alpha_shape_3d(points, alpha)
                inside = hull.find_simplex(points) >= 0
                coverage = inside.mean()
                
                if coverage >= target_coverage:
                    # Prefer smaller alpha (tighter fit) among valid options
                    score = coverage - 0.1 * alpha
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
            except:
                continue
        
        if best_alpha is None:
            # Fallback to convex hull
            print("  ⚠️ Could not find suitable alpha, using convex hull")
            return np.inf
        
        return best_alpha
    
    def _alpha_shape_3d(self, points, alpha):
        """Compute 3D alpha shape"""
        if alpha == np.inf:
            # Just return Delaunay of all points (convex hull)
            return Delaunay(points)
        
        delaunay = Delaunay(points)
        
        # Filter simplices by circumradius
        valid_simplices = []
        for simplex in delaunay.simplices:
            tet = points[simplex]
            circumradius = self._compute_circumradius_3d(tet)
            
            if circumradius < alpha:
                valid_simplices.append(simplex)
        
        if len(valid_simplices) == 0:
            # Alpha too small, fallback to all simplices
            return delaunay
        
        # Build new Delaunay from filtered simplices vertices
        unique_vertices = np.unique(np.array(valid_simplices).flatten())
        filtered_points = points[unique_vertices]
        
        return Delaunay(filtered_points)
    
    def _compute_circumradius_3d(self, tet):
        """Compute circumradius of a tetrahedron"""
        a, b, c, d = tet[0], tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]
        
        vol = np.abs(np.dot(b, np.cross(c, d))) / 6.0
        if vol < 1e-10:
            return np.inf
        
        edges = [np.linalg.norm(vec) for vec in [b, c, d, c-b, d-b, d-c]]
        edges_prod = np.prod(edges)
        
        return edges_prod / (24.0 * vol)
    
    def _visualize_hull(self):
        """Visualize the fitted hull"""
        if self.hull is None or self.valid_points is None:
            return
        
        fig = plt.figure(figsize=(14, 5))
        
        # Generate grid to show hull boundary
        n_grid = 30
        bright_lin = np.linspace(self.bounds['bright_pct'][0], 
                                self.bounds['bright_pct'][1], n_grid)
        contrast_lin = np.linspace(self.bounds['contrast_thresh'][0], 
                                  self.bounds['contrast_thresh'][1], n_grid)
        percentile_lin = np.linspace(self.bounds['percentile_val'][0], 
                                    self.bounds['percentile_val'][1], n_grid)
        
        grid = np.array(np.meshgrid(bright_lin, contrast_lin, percentile_lin)).T.reshape(-1, 3)
        grid_normalized = self._normalize_parameters(grid)
        
        # Test which grid points are inside hull
        inside_mask = self.hull.find_simplex(grid_normalized) >= 0
        inside_grid = grid[inside_mask]
        
        # 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot hull interior (light blue cloud)
        if len(inside_grid) > 0:
            ax1.scatter(inside_grid[:, 0], inside_grid[:, 1], inside_grid[:, 2],
                       c='lightblue', s=5, alpha=0.1, label='Hull interior')
        
        # Plot original valid points (green)
        ax1.scatter(self.valid_points[:, 0], self.valid_points[:, 1], self.valid_points[:, 2],
                   c='limegreen', s=30, alpha=0.6, edgecolor='k', label='Valid points')
        
        ax1.set_xlabel('Background %')
        ax1.set_ylabel('Contrast Thresh')
        ax1.set_zlabel('Global Percentile')
        ax1.set_title('Fitted Hull (3D View)')
        ax1.legend()
        
        # 2D slice at median contrast
        ax2 = fig.add_subplot(122)
        median_contrast = np.median(self.valid_points[:, 1])
        
        # Filter grid points near median contrast
        contrast_tolerance = (self.bounds['contrast_thresh'][1] - 
                             self.bounds['contrast_thresh'][0]) / (2 * n_grid)
        slice_mask = np.abs(inside_grid[:, 1] - median_contrast) < contrast_tolerance
        slice_points = inside_grid[slice_mask]
        
        if len(slice_points) > 0:
            ax2.scatter(slice_points[:, 0], slice_points[:, 2],
                       c='lightblue', s=20, alpha=0.3, label='Hull interior')
        
        valid_slice_mask = np.abs(self.valid_points[:, 1] - median_contrast) < contrast_tolerance
        valid_slice = self.valid_points[valid_slice_mask]
        
        if len(valid_slice) > 0:
            ax2.scatter(valid_slice[:, 0], valid_slice[:, 2],
                       c='limegreen', s=40, alpha=0.7, edgecolor='k', label='Valid points')
        
        ax2.set_xlabel('Background %')
        ax2.set_ylabel('Global Percentile')
        ax2.set_title(f'2D Slice (Contrast ≈ {median_contrast:.2f})')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        print("\n✓ Hull visualization complete")
    
    def save(self, output_dir):
        """
        Save the generated parameter space to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save files
        """
        print("\n" + "="*60)
        print("SAVING PARAMETER SPACE")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save hull (Delaunay object)
        hull_path = os.path.join(output_dir, "valid_parameter_hull.pkl")
        with open(hull_path, 'wb') as f:
            pickle.dump(self.hull, f)
        print(f"✓ Saved hull: {hull_path}")
        
        # Save bounding box
        bounds_path = os.path.join(output_dir, "parameter_bounds.npz")
        np.savez(bounds_path, **self.bounds)
        print(f"✓ Saved bounds: {bounds_path}")
        
        # Save metadata
        metadata = {
            'ground_truth_nuclei': self.ground_truth_nuclei,
            'param_ranges': self.param_ranges,
            'resolution': self.resolution,
            'n_valid_points': len(self.valid_points) if self.valid_points is not None else 0
        }
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Save grid results if available
        if self.grid_results is not None:
            csv_path = os.path.join(output_dir, "grid_search_results.csv")
            self.grid_results.to_csv(csv_path, index=False)
            print(f"✓ Saved grid results: {csv_path}")
        
        # Save valid points
        if self.valid_points is not None:
            valid_path = os.path.join(output_dir, "valid_points.npy")
            np.save(valid_path, self.valid_points)
            print(f"✓ Saved valid points: {valid_path}")
        
        print(f"\n✓ All files saved to: {output_dir}")
    
    @staticmethod
    def load(output_dir):
        """
        Load a previously saved parameter space.
        
        Parameters:
        -----------
        output_dir : str
            Directory containing saved files
            
        Returns:
        --------
        tuple : (delaunay_hull, param_bounds, metadata)
        """
        print(f"\n📂 Loading parameter space from: {output_dir}")
        
        # Load hull
        hull_path = os.path.join(output_dir, "valid_parameter_hull.pkl")
        with open(hull_path, 'rb') as f:
            hull = pickle.load(f)
        print(f"✓ Loaded hull")
        
        # Load bounds
        bounds_path = os.path.join(output_dir, "parameter_bounds.npz")
        bounds = dict(np.load(bounds_path))
        print(f"✓ Loaded bounds")
        
        # Load metadata
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✓ Loaded metadata")
        
        print(f"\n✓ Parameter space loaded successfully")
        print(f"  Ground truth nuclei: {metadata['ground_truth_nuclei']}")
        print(f"  Valid points: {metadata['n_valid_points']}")
        
        return hull, bounds, metadata


class NucleusSelector:
    """
    Interactive interface for selecting nuclei and specifying expected foci counts.
    Shows full image overview, then detailed crops for annotation.
    """
    
    def __init__(self, masks, channel_image, channel_name='TRITC'):
        """
        Initialize nucleus selector.
        
        Parameters:
        -----------
        masks : ndarray
            Segmentation mask with labeled nuclei
        channel_image : ndarray
            Channel image to display (e.g., TRITC)
        channel_name : str
            Name of the channel for display
        """
        self.masks = masks
        self.channel_image = img_as_float(channel_image)
        self.channel_name = channel_name
        self.selected_nuclei = {}  # {cell_id: expected_count}
        
    def run(self):
        """
        Run the interactive selection interface.
        
        Returns:
        --------
        dict : {cell_id: expected_count}
        """
        print("\n" + "="*60)
        print("INTERACTIVE NUCLEUS SELECTION")
        print("="*60)
        print("\nInstructions:")
        print("  1. View the full image with all nuclei labeled")
        print("  2. Enter nucleus IDs to inspect in detail")
        print("  3. For each nucleus, enter expected foci count")
        print("  4. Type 'back' to return to overview")
        print("  5. Type 'done' when finished selecting nuclei")
        print("="*60)
        
        while True:
            # Show full image overview
            self._show_overview()
            
            # Get nucleus IDs to inspect
            user_input = input("\n🔍 Enter nucleus IDs to inspect (comma-separated, or 'done' to finish): ").strip()
            
            if user_input.lower() == 'done':
                break
            
            # Parse nucleus IDs
            try:
                nucleus_ids = [int(x.strip()) for x in user_input.split(',')]
            except ValueError:
                print("❌ Invalid input. Please enter comma-separated numbers or 'done'")
                continue
            
            # Inspect each nucleus
            for nucleus_id in nucleus_ids:
                if nucleus_id not in np.unique(self.masks):
                    print(f"❌ Nucleus {nucleus_id} not found in mask")
                    continue
                
                # Show detailed crop
                action = self._show_nucleus_detail(nucleus_id)
                
                if action == 'back':
                    break  # Return to overview
        
        print(f"\n✓ Selection complete: {len(self.selected_nuclei)} nuclei selected")
        for cell_id, count in self.selected_nuclei.items():
            print(f"   Nucleus {cell_id}: {count} foci")
        
        return self.selected_nuclei
    
    def _show_overview(self):
        """Display full image with all nuclei labeled"""
        from skimage.segmentation import find_boundaries
        from skimage.color import label2rgb
        
        print("\n" + "-"*60)
        print("FULL IMAGE OVERVIEW")
        print("-"*60)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Show channel image
        ax.imshow(self.channel_image, cmap='gray')
        
        # Overlay nucleus boundaries
        boundaries = find_boundaries(self.masks, mode='outer')
        ax.contour(boundaries, colors='cyan', linewidths=1.5, alpha=0.8)
        
        # Label each nucleus with its ID
        cell_ids = np.unique(self.masks)[1:]  # Skip background (0)
        
        for cell_id in cell_ids:
            nucleus_mask = (self.masks == cell_id)
            
            # Find centroid
            y_coords, x_coords = np.where(nucleus_mask)
            if len(y_coords) == 0:
                continue
            
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Color code based on whether it's been selected
            if cell_id in self.selected_nuclei:
                color = 'lime'
                text = f"{cell_id}\n({self.selected_nuclei[cell_id]} foci)"
                fontsize = 10
                fontweight = 'bold'
            else:
                color = 'yellow'
                text = str(cell_id)
                fontsize = 9
                fontweight = 'normal'
            
            ax.text(centroid_x, centroid_y, text,
                   color=color, fontsize=fontsize, fontweight=fontweight,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                           alpha=0.7, edgecolor=color, linewidth=2))
        
        ax.set_title(f'{self.channel_name} Channel - All Nuclei\n'
                    f'(Yellow = Unselected, Green = Selected with foci count)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Total nuclei in image: {len(cell_ids)}")
        print(f"✓ Already selected: {len(self.selected_nuclei)}")
    
    def _show_nucleus_detail(self, nucleus_id):
        """
        Show detailed crop of a single nucleus and get user annotation.
        
        Parameters:
        -----------
        nucleus_id : int
            ID of nucleus to display
            
        Returns:
        --------
        str : 'back' or 'continue'
        """
        # Extract nucleus region
        nucleus_mask = (self.masks == nucleus_id)
        y_coords, x_coords = np.where(nucleus_mask)
        
        if len(y_coords) == 0:
            print(f"❌ Nucleus {nucleus_id} has no pixels")
            return 'continue'
        
        # Calculate bounding box with padding
        padding = 30
        y_min = max(0, y_coords.min() - padding)
        y_max = min(self.channel_image.shape[0], y_coords.max() + padding)
        x_min = max(0, x_coords.min() - padding)
        x_max = min(self.channel_image.shape[1], x_coords.max() + padding)
        
        # Crop image and mask
        crop_image = self.channel_image[y_min:y_max, x_min:x_max]
        crop_mask = nucleus_mask[y_min:y_max, x_min:x_max]
        
        # Apply DoG filter for foci visualization
        from skimage import filters
        filtered_crop = filters.difference_of_gaussians(crop_image, low_sigma=1, high_sigma=2)
        filtered_crop = np.clip(filtered_crop, 0, None)
        filtered_crop = exposure.rescale_intensity(filtered_crop, in_range='image', 
                                                   out_range=(0, 1))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(crop_image, cmap='gray')
        axes[0].contour(crop_mask, colors='cyan', linewidths=2)
        axes[0].set_title(f'Nucleus {nucleus_id} - Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Filtered
        axes[1].imshow(filtered_crop, cmap='gray')
        axes[1].contour(crop_mask, colors='cyan', linewidths=2)
        axes[1].set_title(f'Nucleus {nucleus_id} - DoG Filtered', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Enhanced contrast
        enhanced = exposure.equalize_adapthist(crop_image)
        axes[2].imshow(enhanced, cmap='hot')
        axes[2].contour(crop_mask, colors='cyan', linewidths=2)
        axes[2].set_title(f'Nucleus {nucleus_id} - Enhanced', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Get user input
        while True:
            user_input = input(f"\n🎯 Enter expected foci count for nucleus {nucleus_id} "
                             f"(or 'back' to return to overview, 'skip' to skip): ").strip()
            
            if user_input.lower() == 'back':
                return 'back'
            
            if user_input.lower() == 'skip':
                return 'continue'
            
            try:
                foci_count = int(user_input)
                if foci_count < 0:
                    print("❌ Foci count must be non-negative")
                    continue
                
                self.selected_nuclei[nucleus_id] = foci_count
                print(f"✓ Nucleus {nucleus_id} annotated with {foci_count} foci")
                return 'continue'
                
            except ValueError:
                print("❌ Invalid input. Please enter a number, 'back', or 'skip'")



# ===============================================================
# EXAMPLE USAGE SCRIPT
# ===============================================================

def run_parameter_space_pipeline(masks, channel_image, original_image, 
                                 param_ranges, output_dir, 
                                 channel_name='TRITC', resolution=20):
    """
    Complete pipeline for generating parameter space.
    
    Parameters:
    -----------
    masks : ndarray
        Segmentation mask with labeled nuclei
    channel_image : ndarray
        Channel image for foci detection (e.g., TRITC)
    original_image : ndarray
        Original unfiltered image for global percentiles
    param_ranges : dict
        Parameter ranges to search
    output_dir : str
        Directory to save results
    channel_name : str
        Name of channel being analyzed
    resolution : int
        Grid resolution for parameter sweep
        
    Returns:
    --------
    ParameterSpaceGenerator : Fitted generator object
    """
    print("\n" + "="*70)
    print("PARAMETER SPACE GENERATION PIPELINE")
    print("="*70)
    
    # STEP 1: Interactive nucleus selection
    print("\nSTEP 1: NUCLEUS SELECTION")
    print("-"*70)
    
    selector = NucleusSelector(masks, channel_image, channel_name)
    selected_nuclei = selector.run()
    
    if len(selected_nuclei) == 0:
        print("❌ No nuclei selected. Aborting.")
        return None
    
    # STEP 2-4: Parameter space generation
    generator = ParameterSpaceGenerator(param_ranges, resolution=resolution)
    
    # Add selected nuclei
    for cell_id, expected_count in selected_nuclei.items():
        generator.add_nucleus(cell_id, expected_count)
    
    # Generate grid search
    generator.generate_grid_search(masks, channel_image, original_image)
    
    # Find valid intersection
    valid_params = generator.find_valid_intersection()
    
    if valid_params is None or len(valid_params) == 0:
        print("❌ No valid parameter intersection found. Aborting.")
        return None
    
    # Fit hull
    generator.fit_robust_hull(method='alpha_shape', alpha=None, target_coverage=0.95)
    
    # STEP 5: Save results
    generator.save(output_dir)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nYou can now load the parameter space in your main script:")
    print(f"```python")
    print(f"with open('{output_dir}/valid_parameter_hull.pkl', 'rb') as f:")
    print(f"    delaunay = pickle.load(f)")
    print(f"param_bounds = dict(np.load('{output_dir}/parameter_bounds.npz'))")
    print(f"```")
    
    return generator

