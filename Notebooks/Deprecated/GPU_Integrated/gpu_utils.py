"""
GPU acceleration utilities for foci detection
Windows-compatible version using CuPy instead of cuCIM
"""

import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as distance_transform_gpu
    from cupyx.scipy.ndimage import binary_erosion as binary_erosion_gpu
    GPU_AVAILABLE = True
    print("✅ CuPy loaded successfully")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU libraries not available, falling back to CPU")


def gaussian_filter_gpu(image, sigma):
    """GPU-accelerated Gaussian filter using CuPy"""
    from cupyx.scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma)


def difference_of_gaussians_gpu(image, low_sigma, high_sigma):
    """GPU-accelerated DoG filter"""
    low = gaussian_filter_gpu(image, low_sigma)
    high = gaussian_filter_gpu(image, high_sigma)
    return low - high


def peak_local_max_gpu(image, min_distance=2, threshold_abs=0):
    """
    GPU-accelerated peak detection using CuPy
    Simple implementation using maximum filter
    """
    from cupyx.scipy.ndimage import maximum_filter
    
    # Apply maximum filter
    max_filtered = maximum_filter(image, size=min_distance*2+1)
    
    # Find local maxima
    mask = (image == max_filtered) & (image > threshold_abs)
    
    # Get coordinates
    coords = cp.argwhere(mask)
    return coords


def watershed_gpu(image, markers, mask=None, compactness=0.001):
    """
    GPU-accelerated watershed using simple region growing
    Falls back to CPU for actual watershed (most reliable)
    """
    # For watershed, we'll use CPU version wrapped with GPU transfers
    # This is because watershed is complex and CPU version is reliable
    from skimage.segmentation import watershed as watershed_cpu
    
    # Convert to CPU, run watershed, convert back
    image_cpu = cp.asnumpy(image) if isinstance(image, cp.ndarray) else image
    markers_cpu = cp.asnumpy(markers) if isinstance(markers, cp.ndarray) else markers
    mask_cpu = cp.asnumpy(mask) if isinstance(mask, cp.ndarray) else mask if mask is not None else None
    
    result_cpu = watershed_cpu(image_cpu, markers_cpu, mask=mask_cpu, compactness=compactness)
    
    return cp.asarray(result_cpu)


class GPUAccelerator:
    """
    Handles GPU acceleration with automatic CPU fallback
    Windows-compatible version
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            try:
                # Test GPU
                test = cp.array([1, 2, 3])
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = device_props['name'].decode() if isinstance(device_props['name'], bytes) else device_props['name']
                print(f"✅ GPU acceleration enabled: {gpu_name}")
            except Exception as e:
                print(f"⚠️ GPU test failed: {e}")
                self.use_gpu = False
    
    def to_gpu(self, array):
        """Move numpy array to GPU"""
        if self.use_gpu:
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array):
        """Move GPU array to CPU"""
        if self.use_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def difference_of_gaussians(self, image, low_sigma=1, high_sigma=2):
        """GPU-accelerated DoG filter"""
        import time
        t_start = time.time()
        
        if self.use_gpu:
            try:
                image_gpu = self.to_gpu(image)
                result_gpu = difference_of_gaussians_gpu(image_gpu, low_sigma, high_sigma)
                result = self.to_cpu(result_gpu)
                elapsed = (time.time() - t_start) * 1000
                if elapsed > 10:  # Only print if > 10ms
                    print(f"      ⚡ DoG (GPU): {elapsed:.1f}ms")
                return result
            except Exception as e:
                print(f"      ❌ DoG GPU FAILED: {e}")
                self.use_gpu = False
        
        # CPU fallback
        from skimage.filters import difference_of_gaussians
        result = difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma)
        elapsed = (time.time() - t_start) * 1000
        if elapsed > 10:
            print(f"      🐢 DoG (CPU): {elapsed:.1f}ms")
        return result
    
    def peak_local_max(self, image, min_distance=2, threshold_abs=0):
        """GPU-accelerated peak detection"""
        import time
        t_start = time.time()
        
        if self.use_gpu:
            try:
                image_gpu = self.to_gpu(image)
                result_gpu = peak_local_max_gpu(image_gpu, min_distance=min_distance, 
                                               threshold_abs=threshold_abs)
                result = self.to_cpu(result_gpu)
                elapsed = (time.time() - t_start) * 1000
                if elapsed > 10:
                    print(f"      ⚡ Peak detection (GPU): {elapsed:.1f}ms")
                return result
            except Exception as e:
                print(f"      ❌ Peak detection GPU FAILED: {e}")
                self.use_gpu = False
        
        from skimage.feature import peak_local_max
        result = peak_local_max(image, min_distance=min_distance, 
                             threshold_abs=threshold_abs)
        elapsed = (time.time() - t_start) * 1000
        if elapsed > 10:
            print(f"      🐢 Peak detection (CPU): {elapsed:.1f}ms")
        return result
    
    def watershed_segmentation(self, distance, markers, mask, compactness=0.005):
        """GPU-accelerated watershed"""
        if self.use_gpu:
            # Distance and markers are already numpy arrays in most cases
            # Watershed is run on CPU but with GPU-computed distance transform
            from skimage.segmentation import watershed
            return watershed(-distance, markers, mask=mask, compactness=compactness)
        else:
            from skimage.segmentation import watershed
            return watershed(-distance, markers, mask=mask, compactness=compactness)
    
    def distance_transform(self, binary_mask):
        """GPU-accelerated distance transform"""
        import time
        t_start = time.time()
        
        if self.use_gpu:
            try:
                mask_gpu = self.to_gpu(binary_mask)
                result_gpu = distance_transform_gpu(mask_gpu)
                result = self.to_cpu(result_gpu)
                elapsed = (time.time() - t_start) * 1000
                if elapsed > 10:
                    print(f"      ⚡ Distance transform (GPU): {elapsed:.1f}ms")
                return result
            except Exception as e:
                print(f"      ❌ Distance transform GPU FAILED: {e}")
                self.use_gpu = False
        
        from scipy.ndimage import distance_transform_edt
        result = distance_transform_edt(binary_mask)
        elapsed = (time.time() - t_start) * 1000
        if elapsed > 10:
            print(f"      🐢 Distance transform (CPU): {elapsed:.1f}ms")
        return result
    
    def binary_erosion(self, mask, footprint):
        """GPU-accelerated binary erosion"""
        import time
        t_start = time.time()
        
        if self.use_gpu:
            try:
                mask_gpu = self.to_gpu(mask)
                footprint_gpu = self.to_gpu(footprint)
                result_gpu = binary_erosion_gpu(mask_gpu, structure=footprint_gpu)
                result = self.to_cpu(result_gpu)
                elapsed = (time.time() - t_start) * 1000
                if elapsed > 10:
                    print(f"      ⚡ Binary erosion (GPU): {elapsed:.1f}ms")
                return result
            except Exception as e:
                print(f"      ❌ Binary erosion GPU FAILED: {e}")
                self.use_gpu = False
        
        from skimage.morphology import binary_erosion
        result = binary_erosion(mask, footprint)
        elapsed = (time.time() - t_start) * 1000
        if elapsed > 10:
            print(f"      🐢 Binary erosion (CPU): {elapsed:.1f}ms")
        return result
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()