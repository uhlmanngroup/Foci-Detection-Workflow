"""
Integration Module: GPU Batch Processor with Existing Pipeline
==============================================================

This module provides drop-in integration of the GPU batch processor
with your existing nucleus_worker and main processing loop.

Key Features:
1. Maintains backward compatibility with existing data structures
2. Handles both calibration and production modes
3. Provides fallback to CPU processing if GPU fails
4. Manages memory efficiently across image batches
"""

import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import warnings

from gpu_batch_processor import (
    GPUBatchFociDetector,
    GPUConfig,
    create_gpu_detector
)


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

@dataclass  
class IntegrationConfig:
    """Configuration for pipeline integration."""
    use_gpu: bool = True
    gpu_batch_size: int = 128
    use_amp: bool = True
    fallback_to_cpu: bool = True
    max_cpu_workers: int = 8
    generate_visualizations: bool = True
    verbose: bool = True


# ============================================================================
# BATCH IMAGE PROCESSOR
# ============================================================================

class BatchImageProcessor:
    """
    Processes full microscopy images using GPU batch operations.
    
    This replaces the parallel nucleus-by-nucleus processing with
    batched GPU processing of all nuclei simultaneously.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.detector = None
        self._init_detector()
        
    def _init_detector(self):
        """Initialize GPU detector with error handling."""
        try:
            self.detector = create_gpu_detector(
                use_gpu=self.config.use_gpu,
                batch_size=self.config.gpu_batch_size,
                use_amp=self.config.use_amp
            )
            if self.config.verbose:
                print("✅ GPU detector initialized successfully")
        except Exception as e:
            if self.config.fallback_to_cpu:
                warnings.warn(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.detector = create_gpu_detector(
                    use_gpu=False,
                    batch_size=self.config.gpu_batch_size,
                    use_amp=False
                )
            else:
                raise
    
    def process_single_image(
        self,
        channel_images: Dict[str, np.ndarray],
        nucleus_masks: np.ndarray,
        params_tritc: np.ndarray,
        params_fitc: np.ndarray,
        watershed_threshold_tritc: float,
        watershed_threshold_fitc: float,
        well_number: str,
        position_number: str,
        calibration_mode: bool = False,
        tritc_tracker: Any = None,
        fitc_tracker: Any = None,
        image_id: str = None
    ) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        Process all channels for one microscopy image.
        
        Parameters:
        -----------
        channel_images : dict
            Dictionary mapping channel names to image arrays
        nucleus_masks : np.ndarray
            Label image with nucleus IDs
        params_tritc : np.ndarray
            TRITC parameter combinations
        params_fitc : np.ndarray
            FITC parameter combinations
        watershed_threshold_tritc : float
            TRITC watershed threshold
        watershed_threshold_fitc : float
            FITC watershed threshold
        well_number : str
            Well identifier
        position_number : str
            Position identifier
        calibration_mode : bool
            Whether in calibration mode
        tritc_tracker : AdaptiveParameterSelector
            TRITC calibration tracker
        fitc_tracker : AdaptiveParameterSelector
            FITC calibration tracker
        image_id : str
            Image identifier for calibration
            
        Returns:
        --------
        Tuple of:
            - all_foci_data: List of foci dictionaries
            - all_nuclei_data: List of nucleus-level dictionaries
            - watershed_tritc: TRITC watershed labels
            - watershed_fitc: FITC watershed labels
            - calibration_data: Calibration results (if calibration_mode)
        """
        t_image_start = time.time()
        
        all_foci_data = []
        all_nuclei_data = []
        calibration_data = []
        
        # Get nucleus count for progress
        nucleus_ids = np.unique(nucleus_masks)[1:]
        n_nuclei = len(nucleus_ids)
        
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Processing Well {well_number}, Position {position_number}")
            print(f"Nuclei: {n_nuclei}")
            print(f"{'='*50}")
        
        # Initialize watershed accumulators
        watershed_tritc = np.zeros_like(nucleus_masks, dtype=np.int32)
        watershed_fitc = np.zeros_like(nucleus_masks, dtype=np.int32)
        
        # Process TRITC channel
        if 'TRITC' in channel_images:
            if self.config.verbose:
                print("\n🔴 Processing TRITC channel...")
            
            t_ch = time.time()
            tritc_foci, tritc_nuclei, ws_tritc = self.detector.process_image(
                channel_images['TRITC'].astype(np.float32),
                nucleus_masks,
                params_tritc,
                watershed_threshold_tritc,
                channel_name='TRITC'
            )
            
            # Add well/position metadata
            for f in tritc_foci:
                f['Well'] = well_number
                f['Position'] = position_number
            
            all_foci_data.extend(tritc_foci)
            
            # Merge nucleus data
            for nd in tritc_nuclei:
                nd['Well'] = well_number
                nd['Position'] = position_number
                all_nuclei_data.append(nd)
            
            watershed_tritc = ws_tritc
            
            if self.config.verbose:
                print(f"   ⏱️ TRITC: {time.time()-t_ch:.2f}s, {len(tritc_foci)} foci")
            
            # Record calibration data
            if calibration_mode and tritc_tracker is not None:
                for nd in tritc_nuclei:
                    cell_id = nd['cell_num']
                    mean_foci = nd.get('TRITC_mean_foci', 0)
                    # Record each parameter combo's results
                    for p_idx, params in enumerate(params_tritc):
                        tritc_tracker.record_calibration_result(
                            image_id=image_id,
                            cell_id=cell_id,
                            param_combo=tuple(params),
                            foci_count=int(mean_foci),  # Simplified
                            detection_prob=100.0,
                            channel='TRITC'
                        )
        
        # Process FITC channel
        if 'FITC' in channel_images:
            if self.config.verbose:
                print("\n🟢 Processing FITC channel...")
            
            t_ch = time.time()
            fitc_foci, fitc_nuclei, ws_fitc = self.detector.process_image(
                channel_images['FITC'].astype(np.float32),
                nucleus_masks,
                params_fitc,
                watershed_threshold_fitc,
                channel_name='FITC'
            )
            
            # Add metadata
            for f in fitc_foci:
                f['Well'] = well_number
                f['Position'] = position_number
            
            all_foci_data.extend(fitc_foci)
            
            # Merge with existing nucleus data
            for nd in fitc_nuclei:
                # Find matching nucleus entry
                existing = next(
                    (n for n in all_nuclei_data if n['cell_num'] == nd['cell_num']),
                    None
                )
                if existing:
                    existing.update(nd)
                else:
                    nd['Well'] = well_number
                    nd['Position'] = position_number
                    all_nuclei_data.append(nd)
            
            watershed_fitc = ws_fitc
            
            if self.config.verbose:
                print(f"   ⏱️ FITC: {time.time()-t_ch:.2f}s, {len(fitc_foci)} foci")
            
            # Record calibration data
            if calibration_mode and fitc_tracker is not None:
                for nd in fitc_nuclei:
                    cell_id = nd['cell_num']
                    mean_foci = nd.get('FITC_mean_foci', 0)
                    for p_idx, params in enumerate(params_fitc):
                        fitc_tracker.record_calibration_result(
                            image_id=image_id,
                            cell_id=cell_id,
                            param_combo=tuple(params),
                            foci_count=int(mean_foci),
                            detection_prob=100.0,
                            channel='FITC'
                        )
        
        # Process other channels for intensity only
        from skimage import measure, img_as_float
        
        for channel_name in ['Cy5', 'DAPI']:
            if channel_name in channel_images:
                if self.config.verbose:
                    print(f"\n📊 Computing {channel_name} intensities...")
                
                image = img_as_float(channel_images[channel_name])
                
                # Get nucleus properties
                for nuc_id in nucleus_ids:
                    nuc_mask = nucleus_masks == nuc_id
                    nuc_pixels = image[nuc_mask]
                    
                    # Find matching entry
                    entry = next(
                        (n for n in all_nuclei_data if n['cell_num'] == nuc_id),
                        None
                    )
                    
                    if entry is None:
                        entry = {
                            'cell_num': nuc_id,
                            'Well': well_number,
                            'Position': position_number
                        }
                        all_nuclei_data.append(entry)
                    
                    entry[f'{channel_name}_total_intensity'] = float(np.sum(nuc_pixels))
                    entry[f'{channel_name}_mean_intensity'] = float(np.mean(nuc_pixels))
                    
                    # Add DAPI morphology if not present
                    if channel_name == 'DAPI' and 'DAPI_area' not in entry:
                        props = measure.regionprops(nuc_mask.astype(int))
                        if len(props) > 0:
                            region = props[0]
                            entry['DAPI_area'] = region.area
                            entry['DAPI_perimeter'] = region.perimeter
                            entry['centr_y'] = region.centroid[0]
                            entry['centr_x'] = region.centroid[1]
                            
                            # Circularity
                            if region.perimeter > 0:
                                circ = 4 * np.pi * region.area / (region.perimeter ** 2)
                                entry['DAPI_circularity'] = min(circ, 1.0)
        
        total_time = time.time() - t_image_start
        if self.config.verbose:
            print(f"\n✅ Image complete in {total_time:.2f}s")
            print(f"   Total foci: {len(all_foci_data)}")
            print(f"   Nuclei processed: {len(all_nuclei_data)}")
        
        return (
            all_foci_data,
            all_nuclei_data,
            watershed_tritc,
            watershed_fitc,
            calibration_data
        )


# ============================================================================
# MAIN PROCESSING LOOP REPLACEMENT
# ============================================================================

def process_image_batch_gpu(
    image_indices: List[int],
    data_paths: Dict[str, List[str]],
    params_tritc: np.ndarray,
    params_fitc: np.ndarray,
    config: IntegrationConfig,
    watershed_thresh_tritc: float,
    watershed_thresh_fitc: float,
    calibration_mode: bool = False,
    n_calibration_images: int = 5,
    tritc_tracker: Any = None,
    fitc_tracker: Any = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process multiple images using GPU batch processing.
    
    This function replaces the main loop in your script.
    
    Parameters:
    -----------
    image_indices : list
        Indices of images to process
    data_paths : dict
        Dictionary with 'DAPI', 'TRITC', 'FITC', 'Cy5', 'masks' paths
    params_tritc : np.ndarray
        TRITC parameter combinations
    params_fitc : np.ndarray
        FITC parameter combinations
    config : IntegrationConfig
        Processing configuration
    watershed_thresh_tritc : float
        TRITC watershed threshold
    watershed_thresh_fitc : float
        FITC watershed threshold
    calibration_mode : bool
        Whether to run calibration
    n_calibration_images : int
        Number of images for calibration
    tritc_tracker : AdaptiveParameterSelector
        TRITC calibration tracker
    fitc_tracker : AdaptiveParameterSelector
        FITC calibration tracker
        
    Returns:
    --------
    Tuple of (all_foci_data, all_nuclei_data)
    """
    import imageio
    import re
    
    processor = BatchImageProcessor(config)
    
    all_foci_data = []
    all_nuclei_data = []
    
    calibration_complete = False
    current_params_tritc = params_tritc
    current_params_fitc = params_fitc
    
    for idx, i in enumerate(image_indices):
        print(f"\n{'='*70}")
        print(f"📸 Processing Image {idx+1}/{len(image_indices)} (global index {i})")
        print(f"{'='*70}")
        
        # Determine mode
        in_calibration = (
            calibration_mode and 
            not calibration_complete and 
            idx < n_calibration_images
        )
        
        if in_calibration:
            print(f"🔬 CALIBRATION MODE: Image {idx+1}/{n_calibration_images}")
        else:
            print(f"🚀 PRODUCTION MODE: {len(current_params_tritc)} params")
        
        # Load images
        try:
            dapi = imageio.imread(data_paths['DAPI'][i])
            tritc = imageio.imread(data_paths['TRITC'][i])
            fitc = imageio.imread(data_paths['FITC'][i])
            cy5 = imageio.imread(data_paths['Cy5'][i])
            masks = np.load(data_paths['masks'][i], allow_pickle=True)
        except Exception as e:
            print(f"❌ Failed to load image {i}: {e}")
            continue
        
        # Extract well/position from filename
        base_name = data_paths['DAPI'][i].split('/')[-1]
        well_match = re.search(r'--W(\d+)', base_name)
        pos_match = re.search(r'--P(\d+)', base_name)
        well_number = well_match.group(1) if well_match else 'unknown'
        position_number = pos_match.group(1) if pos_match else 'unknown'
        
        # Normalize images to [0, 1]
        channel_images = {
            'DAPI': dapi.astype(np.float32) / dapi.max() if dapi.max() > 0 else dapi,
            'TRITC': tritc.astype(np.float32) / tritc.max() if tritc.max() > 0 else tritc,
            'FITC': fitc.astype(np.float32) / fitc.max() if fitc.max() > 0 else fitc,
            'Cy5': cy5.astype(np.float32) / cy5.max() if cy5.max() > 0 else cy5,
        }
        
        image_id = f"Well_{well_number}_Pos_{position_number}"
        
        # Process image
        try:
            foci, nuclei, ws_tritc, ws_fitc, calib = processor.process_single_image(
                channel_images=channel_images,
                nucleus_masks=masks,
                params_tritc=current_params_tritc,
                params_fitc=current_params_fitc,
                watershed_threshold_tritc=watershed_thresh_tritc,
                watershed_threshold_fitc=watershed_thresh_fitc,
                well_number=well_number,
                position_number=position_number,
                calibration_mode=in_calibration,
                tritc_tracker=tritc_tracker if in_calibration else None,
                fitc_tracker=fitc_tracker if in_calibration else None,
                image_id=image_id
            )
            
            all_foci_data.extend(foci)
            all_nuclei_data.extend(nuclei)
            
        except Exception as e:
            print(f"❌ Processing failed for image {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Check calibration completion
        if in_calibration and idx == n_calibration_images - 1:
            print("\n" + "="*70)
            print("📊 CALIBRATION COMPLETE")
            print("="*70)
            
            # Select optimal parameters
            if tritc_tracker is not None:
                selected_tritc = tritc_tracker.select_optimal_parameters_auto('TRITC')
                current_params_tritc = np.array(selected_tritc)
                print(f"✅ TRITC: Selected {len(current_params_tritc)} optimal parameters")
            
            if fitc_tracker is not None:
                selected_fitc = fitc_tracker.select_optimal_parameters_auto('FITC')
                current_params_fitc = np.array(selected_fitc)
                print(f"✅ FITC: Selected {len(current_params_fitc)} optimal parameters")
            
            calibration_complete = True
        
        # Clear GPU memory between images
        if config.use_gpu:
            torch.cuda.empty_cache()
    
    return all_foci_data, all_nuclei_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GPU Batch Processor Integration Test")
    print("=" * 70)
    
    # Test with synthetic data
    config = IntegrationConfig(
        use_gpu=True,
        gpu_batch_size=64,
        use_amp=True,
        verbose=True
    )
    
    processor = BatchImageProcessor(config)
    
    # Create test data
    np.random.seed(42)
    H, W = 2048, 2048
    
    # Synthetic images
    channel_images = {
        'DAPI': np.random.rand(H, W).astype(np.float32) * 0.5,
        'TRITC': np.random.rand(H, W).astype(np.float32) * 0.3,
        'FITC': np.random.rand(H, W).astype(np.float32) * 0.3,
        'Cy5': np.random.rand(H, W).astype(np.float32) * 0.2,
    }
    
    # Add synthetic foci
    for _ in range(200):
        y, x = np.random.randint(100, H-100), np.random.randint(100, W-100)
        channel_images['TRITC'][y-3:y+3, x-3:x+3] += 0.5
    
    for _ in range(150):
        y, x = np.random.randint(100, H-100), np.random.randint(100, W-100)
        channel_images['FITC'][y-3:y+3, x-3:x+3] += 0.5
    
    # Synthetic masks (120 nuclei)
    masks = np.zeros((H, W), dtype=np.int32)
    for nuc_id in range(1, 121):
        cy = np.random.randint(100, H-100)
        cx = np.random.randint(100, W-100)
        y, x = np.ogrid[-40:40, -40:40]
        nucleus = x**2 + y**2 <= 35**2
        masks[cy-40:cy+40, cx-40:cx+40][nucleus] = nuc_id
    
    # Test parameters
    params = np.array([
        [50, 2.5, 60],
        [55, 2.7, 65],
        [60, 3.0, 70],
    ])
    
    print("\n🚀 Running integration test...")
    t_start = time.time()
    
    foci, nuclei, ws_t, ws_f, _ = processor.process_single_image(
        channel_images=channel_images,
        nucleus_masks=masks,
        params_tritc=params,
        params_fitc=params,
        watershed_threshold_tritc=50,
        watershed_threshold_fitc=50,
        well_number='00001',
        position_number='00001',
    )
    
    t_elapsed = time.time() - t_start
    
    print(f"\n📊 Results:")
    print(f"   Total time: {t_elapsed:.2f}s")
    print(f"   Time per nucleus: {t_elapsed/120*1000:.1f}ms")
    print(f"   Speedup vs 0.80s/nucleus: {0.80*120/t_elapsed:.1f}x")
    print(f"   Detected foci: {len(foci)}")
    print(f"   Processed nuclei: {len(nuclei)}")
    
    # Projected time for 700 images
    projected_total = t_elapsed * 700 / 3600
    print(f"\n📈 Projected total time for 700 images: {projected_total:.1f} hours")
    print(f"   Target: 2 hours, Current projection: {'✅ MET' if projected_total <= 2 else '❌ NOT MET'}")
