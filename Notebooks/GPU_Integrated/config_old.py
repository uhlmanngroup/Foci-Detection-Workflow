"""
Configuration file for Foci Detection Workflow
===============================================
All user-configurable parameters are defined here.
Modify these values to customize the analysis pipeline.
"""

class Config:
    """Configuration container for foci detection analysis."""


    # GPU Settings
    USE_GPU = True  # Set to False to disable GPU
    GPU_BATCH_CLEAR = True  # Clear GPU memory after each image
    
    # ===============================================================
    # DATA INPUT
    # ===============================================================
    # Path to folder containing microscopy images
    # Use raw string (r'...') to handle Windows backslashes correctly
    FOLDER_PATH = r'Y:\Group Members\Valentin Aubry\01_Data\Test_Data_Andreas_hard'
    
    # ===============================================================
    # PARAMETER SPACE CONFIGURATION
    # ===============================================================
    # Set to True when you want to generate a new parameter space
    # Set to False when you want to use an existing one
    GENERATE_NEW_PARAMETER_SPACE = False
    
    # Paths to pre-computed parameter spaces (used when GENERATE_NEW_PARAMETER_SPACE = False)
    TRITC_PARAMETER_SPACE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\Complete_KDE\TRITC_parameter_space"
    FITC_PARAMETER_SPACE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\Complete_KDE\FITC_parameter_space"
    
    # Number of parameter samples to use (via farthest point sampling)
    N_PARAMETER_SAMPLES = 256
    
    # Number of Sobol samples to generate before filtering (must be power of 2)
    N_SOBOL_SAMPLES = 65536  # 2^16
    
    # ===============================================================
    # CALIBRATION VS PRODUCTION MODE
    # ===============================================================
    # Set to True for first few images to calibrate watershed thresholds
    # Set to False for production runs with pre-calibrated thresholds
    CALIBRATION_MODE = False
    
    # Number of images to process in calibration mode (0 = all images)
    CALIBRATION_IMAGE_LIMIT = 1
    
    # Manual watershed threshold values (used when CALIBRATION_MODE = False)
    # These should be determined during calibration and then hardcoded here
    # Values typically range from 0-100 (percentile of filtered image brightness)
    MANUAL_WATERSHED_THRESHOLD_TRITC = 26.0
    MANUAL_WATERSHED_THRESHOLD_FITC = 26.0
    
    # ===============================================================
    # VISUALIZATION SETTINGS
    # ===============================================================
    # Set to False to disable visualization generation (faster processing)
    GENERATE_VISUALIZATIONS = False
    
    # Minimum detection probability (0-100%) for a focus to be included
    # in watershed visualization
    # Examples:
    #   0   = Show ALL detected foci (even if only detected once)
    #   50  = Show foci detected in at least 50% of parameter combinations
    #   80  = Show only highly robust foci (detected in 80%+ of combinations)
    WATERSHED_MIN_DETECTION_PROB = 0.0
    
    # ===============================================================
    # TEXTURE-BASED FILTERING CONFIGURATION
    # ===============================================================
    # Enable/disable texture-based filtering entirely
    ENABLE_TEXTURE_FILTERING = True
    
    # Minimum coefficient of variation (CV) threshold for nucleus texture
    # CV = std_dev / mean_intensity
    # Lower CV = more uniform (less texture variation)
    # Higher CV = more spotty (more texture variation)
    # Examples:
    #   0.15 = Very uniform nuclei (minimal texture)
    #   0.20 = Moderately uniform (default)
    #   0.25 = Spotty nuclei (high texture variation)
    MIN_CV_THRESHOLD = 0.20
    
    # Contrast multiplier for uniform nuclei
    # When a nucleus is detected as uniform (low texture), we apply stricter
    # filtering by multiplying the contrast threshold
    # Examples:
    #   1.0 = No change (treat uniform nuclei the same as textured ones)
    #   1.5 = Require 50% higher contrast for uniform nuclei (default)
    #   2.0 = Require 2x higher contrast for uniform nuclei (very strict)
    UNIFORM_CONTRAST_MULTIPLIER = 1
    
    # ===============================================================
    # PARALLEL PROCESSING
    # ===============================================================
    # Maximum number of parallel workers (None = auto-detect based on CPU cores)
    MAX_WORKERS = None  # Will use min(8, cpu_count - 1)
    
    # ============ ADAPTIVE PARAMETER SELECTION ============  # ← ADD THIS
    USE_ADAPTIVE_PARAMETERS = True
        
    # Calibration phase settings
    N_CALIBRATION_IMAGES = 1  # Images to use for parameter calibration
    N_CALIBRATION_PARAMS = 256  # Full parameter sweep for calibration
        
    # Production phase settings  
    N_PRODUCTION_PARAMS = 1  # Reduced to 3 best parameter combos
    #DIVERSITY_WEIGHT = 0.3  # Balance between performance and diversity (0-1)
        
    # File paths
    CALIBRATION_SAVE_PATH = r"Y:\Group Members\Valentin Aubry\01_Data\Parameters\Complete_KDE\calibration_results.pkl"
    RECALIBRATE = False  # Set to True to force recalibration




    # ===============================================================
    # HELPER METHODS
    # ===============================================================
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary (excludes methods and private attributes)."""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        for key, value in cls.to_dict().items():
            print(f"{key:40s} = {value}")
        print("="*70 + "\n")
    
    @classmethod
    def get_worker_params(cls):
        """
        Get parameters that need to be passed to worker processes.
        Returns a tuple in the correct order for process_single_nucleus.
        """
        return (
            cls.WATERSHED_MIN_DETECTION_PROB,
            cls.MIN_CV_THRESHOLD,
            cls.UNIFORM_CONTRAST_MULTIPLIER,
            cls.ENABLE_TEXTURE_FILTERING
        )