"""
GPU-Accelerated Microscopy Foci Detection Pipeline
===================================================

Complete main script integrating GPU batch processing.

Files required:
- gpu_batch_processor.py
- pipeline_integration.py  
- adaptive_params.py (your existing AdaptiveParameterSelector)

Usage:
    python main_gpu_pipeline.py --test     # Quick GPU test
    python main_gpu_pipeline.py --run      # Full pipeline
"""

import os
import sys
import time
import re
import pickle
from glob import glob
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import imageio
from scipy.stats import qmc
from skimage import img_as_float

# Import GPU modules
import torch
from gpu_batch_processor import GPUBatchFociDetector, GPUConfig, create_gpu_detector
from pipeline_integration import BatchImageProcessor, IntegrationConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration matching your existing config structure."""
    
    # Paths - UPDATE THESE TO YOUR PATHS
    DATA_FOLDER = r"Y:\Group Members\Valentin Aubry\01_Data\Test_Data_Andreas_hard"
    OUTPUT_FOLDER = r"Y:\Group Members\Valentin Aubry\01_Data\Results"
    TRITC_PARAMETER_SPACE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\Complete_KDE\TRITC_parameter_space"
    FITC_PARAMETER_SPACE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\Complete_KDE\FITC_parameter_space"
    CALIBRATION_SAVE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\calibration.pkl"
    
    # Processing settings
    N_PARAMETER_SAMPLES = 256
    N_SOBOL_SAMPLES = 65536
    
    # GPU settings
    USE_GPU = True
    GPU_BATCH_SIZE = 128
    USE_AMP = True  # Automatic mixed precision
    
    # Adaptive parameters
    USE_ADAPTIVE_PARAMETERS = True
    N_CALIBRATION_IMAGES = 5
    N_PRODUCTION_PARAMS = 3
    RECALIBRATE = False
    
    # Watershed thresholds (set after calibration)
    CALIBRATION_MODE = False
    MANUAL_WATERSHED_THRESHOLD_TRITC = 50.0
    MANUAL_WATERSHED_THRESHOLD_FITC = 50.0
    
    # Visualization
    GENERATE_VISUALIZATIONS = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Select maximally spread points using farthest-point sampling."""
    n = len(points)
    if n_samples > n:
        raise ValueError(f"Cannot sample {n_samples} from {n} points")
    
    selected_idx = [np.random.randint(0, n)]
    dists = np.linalg.norm(points - points[selected_idx[0]], axis=1)
    
    for _ in range(1, n_samples):
        next_idx = np.argmax(dists)
        selected_idx.append(next_idx)
        new_dists = np.linalg.norm(points - points[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    
    return points[selected_idx]


def in_hull(p: np.ndarray, delaunay_obj) -> np.ndarray:
    """Test if points are inside a convex hull."""
    return delaunay_obj.find_simplex(p) >= 0


def save_dataframe_to_csv(df: pd.DataFrame, folder_path: str, base_name: str, data_type: str):
    """Save DataFrame to CSV with Excel compatibility."""
    filename = f"{folder_path}/{base_name}--{data_type}_DF_SEM.csv"
    
    with open(filename, 'w') as f:
        f.write('sep=,\n')
    df.to_csv(filename, mode='a', header=True, index=False)
    print(f"   ✅ Saved: {filename}")


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    if img.max() > 0:
        return img.astype(np.float32) / img.max()
    return img.astype(np.float32)


def load_parameter_space(hull_path: str, bounds_path: str, 
                         n_samples: int, n_sobol: int) -> np.ndarray:
    """Load and sample from parameter space."""
    
    # Load Delaunay hull
    try:
        with open(hull_path, "rb") as f:
            delaunay = pickle.load(f)
    except FileNotFoundError:
        delaunay = None
        print(f"   ⚠️ Hull not found: {hull_path}")
    
    # Load bounds
    try:
        bounds = dict(np.load(bounds_path))
    except FileNotFoundError:
        # Default bounds
        bounds = {
            'bright_pct': np.array([20, 80]),
            'contrast_thresh': np.array([1.5, 5.0]),
            'percentile_val': np.array([50, 90])
        }
        print(f"   ⚠️ Using default bounds")
    
    # Generate Sobol samples
    sampler = qmc.Sobol(d=3, scramble=True)
    unit_samples = sampler.random(n=n_sobol)
    
    # Scale to bounds
    scaled = qmc.scale(
        unit_samples,
        l_bounds=[bounds["bright_pct"][0], bounds["contrast_thresh"][0], bounds["percentile_val"][0]],
        u_bounds=[bounds["bright_pct"][1], bounds["contrast_thresh"][1], bounds["percentile_val"][1]]
    )
    
    # Filter to hull if available
    if delaunay is not None:
        inside = in_hull(scaled, delaunay)
        valid = scaled[inside]
        print(f"   Hull filtering: {len(valid)}/{n_sobol} points inside")
    else:
        valid = scaled
    
    # Check we have enough points
    if len(valid) < n_samples:
        print(f"   ⚠️ Only {len(valid)} valid points, using all")
        return valid
    
    # Apply farthest point sampling
    return farthest_point_sampling(valid, n_samples)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main():
    """Main processing function."""
    
    config = PipelineConfig()
    
    print("=" * 70)
    print("🚀 GPU-ACCELERATED MICROSCOPY FOCI DETECTION PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # ================================================================
    # 1. CHECK GPU
    # ================================================================
    print("\n1️⃣ Checking GPU...")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   ✅ GPU: {props.name}")
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")
    else:
        print("   ⚠️ No GPU available, using CPU")
        config.USE_GPU = False
    
    # ================================================================
    # 2. LOAD DATA PATHS
    # ================================================================
    print("\n2️⃣ Loading data paths...")
    
    folder_path = config.DATA_FOLDER
    
    Cy5_data = sorted(glob(os.path.join(folder_path, '*Cy5 SEM.tif')))
    DAPI_data = sorted(glob(os.path.join(folder_path, '*DAPI SEM.tif')))
    DAPI_mask_data = sorted(glob(os.path.join(folder_path, '*DAPI SEM_seg.npy')))
    FITC_data = sorted(glob(os.path.join(folder_path, '*FITC SEM.tif')))
    TRITC_data = sorted(glob(os.path.join(folder_path, '*TRITC SEM.tif')))
    
    n_images = len(DAPI_data)
    print(f"   Found {n_images} images")
    
    if n_images == 0:
        print("   ❌ No images found! Check DATA_FOLDER path.")
        return
    
    # ================================================================
    # 3. LOAD PARAMETER SPACES
    # ================================================================
    print("\n3️⃣ Loading parameter spaces...")
    
    print("   Loading TRITC parameters...")
    valid_param_samples_TRITC = load_parameter_space(
        os.path.join(config.TRITC_PARAMETER_SPACE_PATH, "valid_parameter_hull.pkl"),
        os.path.join(config.TRITC_PARAMETER_SPACE_PATH, "parameter_bounds.npz"),
        config.N_PARAMETER_SAMPLES,
        config.N_SOBOL_SAMPLES
    )
    print(f"   TRITC: {len(valid_param_samples_TRITC)} parameter combinations")
    
    print("   Loading FITC parameters...")
    valid_param_samples_FITC = load_parameter_space(
        os.path.join(config.FITC_PARAMETER_SPACE_PATH, "valid_parameter_hull.pkl"),
        os.path.join(config.FITC_PARAMETER_SPACE_PATH, "parameter_bounds.npz"),
        config.N_PARAMETER_SAMPLES,
        config.N_SOBOL_SAMPLES
    )
    print(f"   FITC: {len(valid_param_samples_FITC)} parameter combinations")
    
    # ================================================================
    # 4. INITIALIZE GPU PROCESSOR
    # ================================================================
    print("\n4️⃣ Initializing GPU processor...")
    
    integration_config = IntegrationConfig(
        use_gpu=config.USE_GPU,
        gpu_batch_size=config.GPU_BATCH_SIZE,
        use_amp=config.USE_AMP,
        fallback_to_cpu=True,
        verbose=True,
        generate_visualizations=config.GENERATE_VISUALIZATIONS
    )
    
    processor = BatchImageProcessor(integration_config)
    
    # ================================================================
    # 5. SETUP ADAPTIVE PARAMETERS (if enabled)
    # ================================================================
    tritc_tracker = None
    fitc_tracker = None
    calibration_complete = False
    
    if config.USE_ADAPTIVE_PARAMETERS:
        print("\n5️⃣ Setting up adaptive parameters...")
        
        try:
            from adaptive_params import AdaptiveParameterSelector
            
            tritc_tracker = AdaptiveParameterSelector(
                n_calibration_images=config.N_CALIBRATION_IMAGES,
                n_final_params=config.N_PRODUCTION_PARAMS
            )
            fitc_tracker = AdaptiveParameterSelector(
                n_calibration_images=config.N_CALIBRATION_IMAGES,
                n_final_params=config.N_PRODUCTION_PARAMS
            )
            
            # Try to load existing calibration
            if not config.RECALIBRATE:
                tritc_path = config.CALIBRATION_SAVE_PATH.replace('.pkl', '_TRITC.pkl')
                fitc_path = config.CALIBRATION_SAVE_PATH.replace('.pkl', '_FITC.pkl')
                
                if os.path.exists(tritc_path) and os.path.exists(fitc_path):
                    tritc_tracker.load_calibration(tritc_path)
                    fitc_tracker.load_calibration(fitc_path)
                    
                    valid_param_samples_TRITC = np.array(tritc_tracker.selected_params['TRITC'])
                    valid_param_samples_FITC = np.array(fitc_tracker.selected_params['FITC'])
                    calibration_complete = True
                    
                    print(f"   ✅ Loaded calibration")
                    print(f"      TRITC: {len(valid_param_samples_TRITC)} params")
                    print(f"      FITC: {len(valid_param_samples_FITC)} params")
                else:
                    print(f"   ℹ️ No existing calibration, will calibrate on first {config.N_CALIBRATION_IMAGES} images")
        
        except ImportError:
            print("   ⚠️ adaptive_params module not found, disabling adaptive parameters")
            config.USE_ADAPTIVE_PARAMETERS = False
    
    # Current parameters (may be updated after calibration)
    current_params_tritc = valid_param_samples_TRITC
    current_params_fitc = valid_param_samples_FITC
    
    # ================================================================
    # 6. PROCESS ALL IMAGES
    # ================================================================
    print("\n6️⃣ Processing images...")
    print(f"   Mode: {'CALIBRATION' if config.CALIBRATION_MODE else 'PRODUCTION'}")
    print(f"   Thresholds: TRITC={config.MANUAL_WATERSHED_THRESHOLD_TRITC}, FITC={config.MANUAL_WATERSHED_THRESHOLD_FITC}")
    
    all_foci_data = []
    all_nuclei_data = []
    image_times = []
    
    for i in range(n_images):
        t_img_start = time.time()
        
        # Determine if in calibration phase
        in_calibration = (
            config.USE_ADAPTIVE_PARAMETERS and
            not calibration_complete and
            i < config.N_CALIBRATION_IMAGES
        )
        
        print(f"\n{'='*60}")
        print(f"📸 Image {i+1}/{n_images}" + (" [CALIBRATION]" if in_calibration else ""))
        print(f"{'='*60}")
        
        # Load images
        try:
            dapi = imageio.imread(DAPI_data[i])
            tritc = imageio.imread(TRITC_data[i])
            fitc = imageio.imread(FITC_data[i])
            cy5 = imageio.imread(Cy5_data[i])
            masks = np.load(DAPI_mask_data[i], allow_pickle=True)
        except Exception as e:
            print(f"❌ Failed to load image {i}: {e}")
            continue
        
        # Extract metadata from filename
        base_name = os.path.basename(DAPI_data[i]).removesuffix('DAPI SEM.tif')
        well_match = re.search(r'--W(\d+)', base_name)
        pos_match = re.search(r'--P(\d+)', base_name)
        well_number = well_match.group(1) if well_match else 'unknown'
        position_number = pos_match.group(1) if pos_match else 'unknown'
        
        # Prepare channel images
        channel_images = {
            'DAPI': normalize_image(dapi),
            'TRITC': normalize_image(tritc),
            'FITC': normalize_image(fitc),
            'Cy5': normalize_image(cy5),
        }
        
        image_id = f"Well_{well_number}_Pos_{position_number}"
        
        # Process image
        try:
            foci, nuclei, ws_tritc, ws_fitc, _ = processor.process_single_image(
                channel_images=channel_images,
                nucleus_masks=masks,
                params_tritc=current_params_tritc,
                params_fitc=current_params_fitc,
                watershed_threshold_tritc=config.MANUAL_WATERSHED_THRESHOLD_TRITC,
                watershed_threshold_fitc=config.MANUAL_WATERSHED_THRESHOLD_FITC,
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
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        t_img = time.time() - t_img_start
        image_times.append(t_img)
        
        # Check calibration completion
        if in_calibration and i == config.N_CALIBRATION_IMAGES - 1:
            print("\n" + "="*50)
            print("📊 CALIBRATION COMPLETE")
            print("="*50)
            
            if tritc_tracker is not None:
                selected = tritc_tracker.select_optimal_parameters_auto('TRITC')
                current_params_tritc = np.array(selected)
                tritc_tracker.save_calibration(
                    config.CALIBRATION_SAVE_PATH.replace('.pkl', '_TRITC.pkl')
                )
                print(f"✅ TRITC: {len(current_params_tritc)} optimal parameters")
            
            if fitc_tracker is not None:
                selected = fitc_tracker.select_optimal_parameters_auto('FITC')
                current_params_fitc = np.array(selected)
                fitc_tracker.save_calibration(
                    config.CALIBRATION_SAVE_PATH.replace('.pkl', '_FITC.pkl')
                )
                print(f"✅ FITC: {len(current_params_fitc)} optimal parameters")
            
            calibration_complete = True
        
        # Progress report every 10 images
        if (i + 1) % 10 == 0 and image_times:
            avg_time = np.mean(image_times)
            remaining = (n_images - i - 1) * avg_time / 3600
            print(f"\n📊 Progress: {i+1}/{n_images} | Avg: {avg_time:.1f}s/img | ETA: {remaining:.1f}h")
        
        # Clear GPU memory periodically
        if config.USE_GPU and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # ================================================================
    # 7. SAVE RESULTS
    # ================================================================
    print("\n7️⃣ Saving results...")
    
    output_folder = config.OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)
    
    if all_nuclei_data:
        nuclei_df = pd.DataFrame(all_nuclei_data)
        nuclei_df = nuclei_df.sort_values(['Well', 'Position', 'cell_num']).reset_index(drop=True)
        save_dataframe_to_csv(nuclei_df, output_folder, 'gpu_batch', 'nuclei')
        print(f"   Saved {len(nuclei_df)} nucleus records")
    
    if all_foci_data:
        foci_df = pd.DataFrame(all_foci_data)
        foci_df = foci_df.sort_values(['Well', 'Position', 'cell_num']).reset_index(drop=True)
        save_dataframe_to_csv(foci_df, output_folder, 'gpu_batch', 'foci')
        print(f"   Saved {len(foci_df)} foci records")
    
    # ================================================================
    # 8. FINAL REPORT
    # ================================================================
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print("\n" + "=" * 70)
    print("📊 FINAL REPORT")
    print("=" * 70)
    print(f"Total runtime: {hours}h {minutes}m {seconds:.1f}s")
    print(f"Images processed: {n_images}")
    print(f"Total nuclei: {len(all_nuclei_data)}")
    print(f"Total foci: {len(all_foci_data)}")
    
    if image_times:
        avg_per_image = np.mean(image_times)
        print(f"\nPer-image statistics:")
        print(f"   Average: {avg_per_image:.1f}s")
        print(f"   Min: {np.min(image_times):.1f}s")
        print(f"   Max: {np.max(image_times):.1f}s")
        
        # Speedup calculation
        original_time = 18 * 3600  # 18 hours
        speedup = original_time / total_time if total_time > 0 else float('inf')
        print(f"\n🚀 Speedup vs original: {speedup:.1f}x")
        
        if total_time / 3600 <= 2:
            print("✅ TARGET MET: Under 2 hours!")
        else:
            print(f"⚠️ Target: {total_time/3600:.1f}h > 2h target")


# ============================================================================
# QUICK TEST MODE
# ============================================================================

def quick_test():
    """Quick test with synthetic data to verify GPU processing works."""
    
    print("=" * 70)
    print("🧪 QUICK TEST MODE")
    print("=" * 70)
    
    # Check GPU
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create test data
    print("\nCreating synthetic test data...")
    np.random.seed(42)
    
    H, W = 2048, 2048
    n_nuclei = 120
    
    # Create images with realistic structure
    dapi = np.random.rand(H, W).astype(np.float32) * 0.3
    tritc = np.random.rand(H, W).astype(np.float32) * 0.2
    fitc = np.random.rand(H, W).astype(np.float32) * 0.2
    cy5 = np.random.rand(H, W).astype(np.float32) * 0.1
    
    # Add synthetic foci
    for _ in range(200):
        y, x = np.random.randint(100, H-100), np.random.randint(100, W-100)
        tritc[max(0,y-3):min(H,y+3), max(0,x-3):min(W,x+3)] += 0.6
    
    for _ in range(150):
        y, x = np.random.randint(100, H-100), np.random.randint(100, W-100)
        fitc[max(0,y-3):min(H,y+3), max(0,x-3):min(W,x+3)] += 0.6
    
    # Create nucleus masks
    masks = np.zeros((H, W), dtype=np.int32)
    for nuc_id in range(1, n_nuclei + 1):
        cy = np.random.randint(100, H-100)
        cx = np.random.randint(100, W-100)
        y, x = np.ogrid[-40:40, -40:40]
        nucleus = x**2 + y**2 <= 35**2
        y_start, y_end = max(0, cy-40), min(H, cy+40)
        x_start, x_end = max(0, cx-40), min(W, cx+40)
        # Adjust nucleus slice if needed
        ny_start = 40 - (cy - y_start)
        ny_end = 40 + (y_end - cy)
        nx_start = 40 - (cx - x_start)
        nx_end = 40 + (x_end - cx)
        masks[y_start:y_end, x_start:x_end][nucleus[ny_start:ny_end, nx_start:nx_end]] = nuc_id
    
    channel_images = {
        'DAPI': dapi,
        'TRITC': tritc,
        'FITC': fitc,
        'Cy5': cy5,
    }
    
    # Test parameters (small set for speed)
    params = np.array([
        [50, 2.5, 60],
        [55, 2.7, 65],
        [60, 3.0, 70],
        [65, 3.2, 75],
        [70, 3.5, 80],
    ])
    
    # Process
    print("\nProcessing with GPU batch processor...")
    
    config = IntegrationConfig(
        use_gpu=True,
        gpu_batch_size=64,
        use_amp=True,
        verbose=True
    )
    
    processor = BatchImageProcessor(config)
    
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
    
    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"Processing time: {t_elapsed:.2f}s")
    print(f"Nuclei processed: {len(nuclei)}")
    print(f"Foci detected: {len(foci)}")
    print(f"Time per nucleus: {t_elapsed/n_nuclei*1000:.1f}ms")
    
    # Speedup projection
    original_per_nucleus = 0.80  # seconds (your reported time)
    original_per_image = original_per_nucleus * n_nuclei
    speedup = original_per_image / t_elapsed
    
    print(f"\n🚀 Speedup vs original (0.80s/nucleus): {speedup:.1f}x")
    
    projected_hours = t_elapsed * 700 / 3600
    print(f"Projected time for 700 images: {projected_hours:.1f} hours")
    
    if projected_hours <= 2:
        print("✅ Would meet 2-hour target!")
    else:
        print(f"⚠️ Would exceed target by {projected_hours - 2:.1f} hours")
    
    return t_elapsed, len(foci), len(nuclei)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU-accelerated microscopy foci detection"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run quick GPU test with synthetic data"
    )
    parser.add_argument(
        "--run", 
        action="store_true", 
        help="Run full pipeline on real data"
    )
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.run:
        main()
    else:
        # Default behavior: check if data exists
        config = PipelineConfig()
        if os.path.exists(config.DATA_FOLDER):
            print("Data folder found. Use --run to process or --test for quick test.")
            print("\nRunning quick test...")
            quick_test()
        else:
            print(f"Data folder not found: {config.DATA_FOLDER}")
            print("Running quick test with synthetic data...")
            quick_test()
