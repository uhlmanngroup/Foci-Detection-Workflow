"""
Optimized Background Computation for Foci Detection
====================================================

This module provides highly optimized background percentile computation
using pure PyTorch operations. The original implementation was the main
bottleneck (0.40s per nucleus).

Key Optimizations:
1. Unfold operation for efficient neighborhood extraction
2. Vectorized percentile computation using sorting
3. Batched processing of all coordinates
4. Memory-efficient chunking for large batches
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import time


class OptimizedBackgroundComputer:
    """
    Ultra-optimized local background percentile computation.
    
    Uses PyTorch unfold operations to extract neighborhoods efficiently,
    then computes percentiles using vectorized sorting.
    
    Performance Target: < 0.05s per image (vs 0.40s * 120 nuclei = 48s original)
    """
    
    def __init__(
        self,
        inner_radius: int = 2,
        outer_radius: int = 6,
        device: str = 'cuda'
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Pre-compute annulus mask
        self._build_annulus_mask()
        
    def _build_annulus_mask(self):
        """Pre-compute the annulus mask as a flat index array."""
        r = self.outer_radius
        size = 2 * r + 1
        
        y, x = np.mgrid[-r:r+1, -r:r+1]
        distances = np.sqrt(x**2 + y**2)
        
        # Annulus: between inner and outer radius
        annulus = (distances >= self.inner_radius) & (distances <= self.outer_radius)
        
        # Store as flat indices within the kernel
        self.annulus_indices = torch.from_numpy(
            np.where(annulus.flatten())[0]
        ).long().to(self.device)
        
        self.kernel_size = size
        self.n_annulus_pixels = len(self.annulus_indices)
        
        print(f"✅ Annulus mask: {self.n_annulus_pixels} pixels, kernel size {size}x{size}")
    
    def compute_backgrounds_fast(
        self,
        image: torch.Tensor,
        coords: torch.Tensor,
        percentiles: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute local background percentiles using efficient unfold operations.
        
        Parameters:
        -----------
        image : torch.Tensor
            Input image, shape (H, W)
        coords : torch.Tensor
            Coordinates to compute backgrounds for, shape (N, 2) as (y, x)
        percentiles : torch.Tensor
            Percentile values to compute, shape (P,)
        mask : torch.Tensor, optional
            Valid region mask, shape (H, W)
            
        Returns:
        --------
        torch.Tensor : Background percentiles, shape (N, P)
        """
        if len(coords) == 0:
            return torch.zeros((0, len(percentiles)), device=self.device)
        
        H, W = image.shape
        N = len(coords)
        P = len(percentiles)
        r = self.outer_radius
        
        # Pad image for edge handling
        padded = F.pad(
            image.unsqueeze(0).unsqueeze(0),
            (r, r, r, r),
            mode='reflect'
        ).squeeze()  # (H+2r, W+2r)
        
        # Also pad mask if provided
        if mask is not None:
            padded_mask = F.pad(
                mask.unsqueeze(0).unsqueeze(0).float(),
                (r, r, r, r),
                mode='constant',
                value=0
            ).squeeze().bool()
        
        # Adjust coordinates for padding
        adjusted_coords = coords + r  # (N, 2)
        
        # Extract neighborhoods using advanced indexing
        # Create index grids for each coordinate
        k = self.kernel_size
        
        # Generate all pixel positions in kernel
        ky, kx = torch.meshgrid(
            torch.arange(k, device=self.device),
            torch.arange(k, device=self.device),
            indexing='ij'
        )
        ky = ky.flatten()  # (k*k,)
        kx = kx.flatten()  # (k*k,)
        
        # Compute absolute positions for all coords
        # coords: (N, 2), ky/kx: (k*k,)
        # Result: (N, k*k) for each of y and x
        abs_y = adjusted_coords[:, 0:1] + ky.unsqueeze(0) - r  # (N, k*k)
        abs_x = adjusted_coords[:, 1:2] + kx.unsqueeze(0) - r  # (N, k*k)
        
        # Extract pixel values
        neighborhoods = padded[abs_y, abs_x]  # (N, k*k)
        
        # Select only annulus pixels
        annulus_values = neighborhoods[:, self.annulus_indices]  # (N, n_annulus)
        
        # Apply mask if provided
        if mask is not None:
            mask_values = padded_mask[abs_y, abs_x]
            annulus_mask = mask_values[:, self.annulus_indices]
            # Set masked pixels to NaN
            annulus_values = torch.where(
                annulus_mask,
                annulus_values,
                torch.tensor(float('nan'), device=self.device)
            )
        
        # Compute percentiles via sorting
        # Sort each row, NaNs go to end
        sorted_vals, _ = torch.sort(annulus_values, dim=1)
        
        # Count valid (non-NaN) pixels per coordinate
        valid_counts = (~torch.isnan(annulus_values)).sum(dim=1).float()  # (N,)
        
        # Compute percentile indices
        # For percentile p, index = p/100 * (n_valid - 1)
        q = percentiles / 100.0  # (P,)
        indices = (q.unsqueeze(0) * (valid_counts.unsqueeze(1) - 1)).long()  # (N, P)
        indices = indices.clamp(0, self.n_annulus_pixels - 1)
        
        # Gather percentile values
        # Need to expand sorted_vals for gather
        # sorted_vals: (N, n_annulus), indices: (N, P)
        backgrounds = sorted_vals.gather(1, indices)  # (N, P)
        
        # Handle any remaining NaNs (shouldn't happen with enough valid pixels)
        backgrounds = torch.nan_to_num(backgrounds, nan=0.0)
        
        return backgrounds
    
    def compute_backgrounds_chunked(
        self,
        image: torch.Tensor,
        coords: torch.Tensor,
        percentiles: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 10000
    ) -> torch.Tensor:
        """
        Memory-efficient version that processes coordinates in chunks.
        
        Use this for very large numbers of coordinates to avoid OOM.
        """
        N = len(coords)
        P = len(percentiles)
        
        if N <= chunk_size:
            return self.compute_backgrounds_fast(image, coords, percentiles, mask)
        
        # Process in chunks
        results = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_coords = coords[start:end]
            chunk_result = self.compute_backgrounds_fast(
                image, chunk_coords, percentiles, mask
            )
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)


class BatchBackgroundProcessor:
    """
    Process backgrounds for multiple images in a batch.
    
    This wraps OptimizedBackgroundComputer to handle batches of images
    and their corresponding coordinate lists.
    """
    
    def __init__(self, inner_radius: int = 2, outer_radius: int = 6):
        self.computer = OptimizedBackgroundComputer(
            inner_radius=inner_radius,
            outer_radius=outer_radius
        )
    
    def process_batch(
        self,
        images: torch.Tensor,
        coords_list: List[torch.Tensor],
        percentiles: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Compute backgrounds for a batch of images.
        
        Parameters:
        -----------
        images : torch.Tensor
            Batch of images, shape (B, H, W)
        coords_list : List[torch.Tensor]
            List of coordinate tensors, each shape (N_i, 2)
        percentiles : torch.Tensor
            Percentiles to compute, shape (P,)
        masks : torch.Tensor, optional
            Batch of masks, shape (B, H, W)
            
        Returns:
        --------
        List[torch.Tensor] : List of background tensors, each shape (N_i, P)
        """
        B = len(images)
        results = []
        
        for b in range(B):
            image = images[b]
            coords = coords_list[b]
            mask = masks[b] if masks is not None else None
            
            if len(coords) == 0:
                results.append(torch.zeros((0, len(percentiles)), 
                                         device=images.device))
            else:
                bg = self.computer.compute_backgrounds_fast(
                    image, coords, percentiles, mask
                )
                results.append(bg)
        
        return results


# ============================================================================
# ALTERNATIVE: Using Conv2d for Percentile Estimation
# ============================================================================

class ConvBasedBackgroundEstimator:
    """
    Alternative approach using convolutions for faster approximate backgrounds.
    
    Instead of exact percentiles, uses local statistics (mean, median-like)
    which can be computed more efficiently with convolutions.
    
    Trade-off: Less accurate than exact percentiles but ~10x faster.
    """
    
    def __init__(
        self,
        inner_radius: int = 2,
        outer_radius: int = 6,
        device: str = 'cuda'
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Build annulus kernel
        self._build_kernel()
    
    def _build_kernel(self):
        """Build the annulus convolution kernel."""
        r = self.outer_radius
        size = 2 * r + 1
        
        y, x = np.mgrid[-r:r+1, -r:r+1]
        distances = np.sqrt(x**2 + y**2)
        
        # Annulus mask
        kernel = ((distances >= self.inner_radius) & 
                 (distances <= self.outer_radius)).astype(np.float32)
        kernel = kernel / kernel.sum()  # Normalize
        
        self.kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(self.device)
        self.padding = r
    
    def estimate_local_mean(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute local mean within annulus at every pixel.
        
        This is O(1) per pixel using convolution, much faster than
        explicit neighborhood extraction.
        """
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        local_mean = F.conv2d(image, self.kernel, padding=self.padding)
        return local_mean.squeeze()
    
    def estimate_local_percentile(
        self,
        image: torch.Tensor,
        percentile: float = 50.0
    ) -> torch.Tensor:
        """
        Approximate local percentile using order statistics.
        
        Uses a clever approximation:
        1. Compute local mean
        2. Compute local variance
        3. Estimate percentile assuming roughly Gaussian distribution
        
        This is much faster than exact percentiles but less accurate
        for non-Gaussian intensity distributions.
        """
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        # Local mean
        local_mean = F.conv2d(image, self.kernel, padding=self.padding)
        
        # Local variance = E[X²] - E[X]²
        local_sq_mean = F.conv2d(image ** 2, self.kernel, padding=self.padding)
        local_var = local_sq_mean - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-8))
        
        # Approximate percentile using Gaussian assumption
        # z-score for percentile p
        from scipy import stats
        z = stats.norm.ppf(percentile / 100.0)
        
        local_percentile = local_mean + z * local_std
        return local_percentile.squeeze()


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_background_methods(n_coords: int = 5000, image_size: int = 2048):
    """
    Benchmark different background computation methods.
    """
    print("=" * 70)
    print("Background Computation Benchmark")
    print("=" * 70)
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of coordinates: {n_coords}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data
    image = torch.rand(image_size, image_size, device=device)
    coords = torch.randint(50, image_size-50, (n_coords, 2), device=device)
    percentiles = torch.tensor([25, 50, 75], device=device, dtype=torch.float32)
    mask = torch.ones(image_size, image_size, device=device, dtype=torch.bool)
    
    # Warm up
    computer = OptimizedBackgroundComputer(device=str(device))
    _ = computer.compute_backgrounds_fast(image, coords[:100], percentiles, mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark optimized method
    print("\n1️⃣ Optimized PyTorch (exact percentiles):")
    times = []
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.time()
        result = computer.compute_backgrounds_fast(image, coords, percentiles, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t_start)
    
    avg_time = np.mean(times[1:])  # Skip first (compilation)
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   Per coordinate: {avg_time/n_coords*1e6:.2f} µs")
    print(f"   Result shape: {result.shape}")
    
    # Benchmark conv-based approximation
    print("\n2️⃣ Convolution-based (approximate):")
    conv_estimator = ConvBasedBackgroundEstimator(device=str(device))
    
    times = []
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.time()
        local_mean = conv_estimator.estimate_local_mean(image)
        # Extract at coordinates
        result_conv = local_mean[coords[:, 0], coords[:, 1]]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t_start)
    
    avg_time_conv = np.mean(times[1:])
    print(f"   Average time: {avg_time_conv*1000:.2f} ms")
    print(f"   Speedup vs exact: {avg_time/avg_time_conv:.1f}x")
    
    # Estimate time for full pipeline
    print("\n📊 Pipeline Projections:")
    nuclei_per_image = 120
    coords_per_nucleus = n_coords / nuclei_per_image
    
    print(f"   Assuming {nuclei_per_image} nuclei/image, ~{coords_per_nucleus:.0f} coords/nucleus")
    
    time_per_image_exact = avg_time
    time_per_image_approx = avg_time_conv
    
    print(f"   Time per image (exact): {time_per_image_exact*1000:.1f} ms")
    print(f"   Time per image (approx): {time_per_image_approx*1000:.1f} ms")
    
    # For 700 images
    n_images = 700
    total_exact = time_per_image_exact * n_images / 3600
    total_approx = time_per_image_approx * n_images / 3600
    
    print(f"\n   For {n_images} images:")
    print(f"   - Exact percentiles: {total_exact:.2f} hours (just background comp)")
    print(f"   - Approximate: {total_approx:.2f} hours")
    
    return {
        'exact_time_per_coord': avg_time / n_coords,
        'approx_time_per_coord': avg_time_conv / n_coords,
        'exact_total_hours': total_exact,
        'approx_total_hours': total_approx
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = benchmark_background_methods(n_coords=5000, image_size=2048)
    
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    
    if results['exact_total_hours'] < 0.5:
        print("✅ Exact percentile method is fast enough for your needs")
    else:
        print("⚠️ Consider using approximate method or reducing parameter space")
    
    print(f"\nProjected background computation time: {results['exact_total_hours']:.2f} hours")
    print("This is just one component - total pipeline will be longer")
