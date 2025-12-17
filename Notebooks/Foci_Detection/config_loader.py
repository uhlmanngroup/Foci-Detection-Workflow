"""
Configuration loader for YAML-based settings.

This module loads config.yaml at import time and provides type-safe property 
access to all settings. Any module can import the global 'config' instance 
to access configuration values without re-parsing the YAML file.

Usage:
    from config_loader import config
    folder = config.FOLDER_PATH
"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration container that loads from YAML file.
    
    Validates structure on initialization and provides property-based access 
    to all settings. Properties are read-only to prevent accidental modification 
    of configuration during runtime.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load and validate configuration from YAML file.
        
        The configuration is loaded once at initialization and stored in 
        self._config dictionary. All properties then read from this dictionary.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file (default: "config.yaml" in current directory)
        
        Raises:
        -------
        FileNotFoundError : If config file doesn't exist
        ValueError : If required sections are missing from config
        """
        # Check file exists before attempting to load
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML file into dictionary
        # safe_load prevents execution of arbitrary Python code in YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Validate that all required sections are present
        self._validate_config()
    
    def _validate_config(self):
        """
        Ensure all required sections exist in config file.
        
        This catches configuration errors early (at import time) rather than 
        failing later when a specific setting is accessed. Helps users identify 
        malformed config files immediately.
        
        Raises:
        -------
        ValueError : If any required section is missing
        """
        required_sections = [
            'data', 'parameter_space', 'adaptive_parameters',
            'watershed', 'visualization', 'texture_filtering', 'processing'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
    
    # ===============================================================
    # PROPERTY ACCESSORS
    # ===============================================================
    # These provide type-safe, read-only access to config values.
    # Grouped by config.yaml section for easy reference.
    #
    # WHY PROPERTIES?
    # - Type hints help IDEs with autocomplete
    # - Centralized access (change path in one place if YAML structure changes)
    # - Read-only (prevents accidental config modification during runtime)
    # - Clear error messages if key is missing
    # ===============================================================
    
    # --- Data Input ---
    # These properties access the 'data' section of config.yaml
    
    @property
    def FOLDER_PATH(self) -> str:
        """
        Path to folder containing microscopy images.
        
        Returns the raw string from config.yaml. No validation is performed here - 
        the main script will check if the folder exists when it tries to load images.
        """
        return self._config['data']['folder_path']
    
    # --- Parameter Space ---
    # These properties control parameter space generation and loading
    
    @property
    def GENERATE_NEW_PARAMETER_SPACE(self) -> bool:
        """Whether to generate new parameter spaces (true) or load existing (false)"""
        return self._config['parameter_space']['generate_new']
    
    @property
    def TRITC_PARAMETER_SPACE_PATH(self) -> str:
        """Path to TRITC parameter space directory (contains hull and bounds files)"""
        return self._config['parameter_space']['tritc_path']
    
    @property
    def FITC_PARAMETER_SPACE_PATH(self) -> str:
        """Path to FITC parameter space directory (contains hull and bounds files)"""
        return self._config['parameter_space']['fitc_path']
    
    @property
    def N_PARAMETER_SAMPLES(self) -> int:
        """Number of parameter combinations to select via farthest-point sampling"""
        return self._config['parameter_space']['n_samples']
    
    @property
    def N_SOBOL_SAMPLES(self) -> int:
        """Number of Sobol samples for parameter space generation (internal use)"""
        return self._config['parameter_space']['n_sobol_samples']
    
    # --- Adaptive Parameters ---
    # These control the calibration and parameter optimization workflow
    
    @property
    def USE_ADAPTIVE_PARAMETERS(self) -> bool:
        """Whether to use adaptive parameter selection (calibration → optimization)"""
        return self._config['adaptive_parameters']['enabled']
    
    @property
    def N_CALIBRATION_IMAGES(self) -> int:
        """Number of images to use for calibration phase"""
        return self._config['adaptive_parameters']['n_calibration_images']
    
    @property
    def N_CALIBRATION_PARAMS(self) -> int:
        """Number of parameters to test during calibration (usually = N_PARAMETER_SAMPLES)"""
        return self._config['adaptive_parameters']['n_calibration_params']
    
    @property
    def N_PRODUCTION_PARAMS(self) -> int:
        """Number of optimized parameters to use after calibration (typically 1-3)"""
        return self._config['adaptive_parameters']['n_production_params']
    
    @property
    def CALIBRATION_SAVE_PATH(self) -> str:
        """Path to save full calibration results (all tested parameters and metrics)"""
        return self._config['adaptive_parameters']['calibration_save_path']
    
    @property
    def RECALIBRATE(self) -> bool:
        """Force fresh calibration even if results exist"""
        return self._config['adaptive_parameters']['recalibrate']
    
    @property
    def LOAD_REDUCED_PARAMETERS(self) -> bool:
        """Load pre-saved optimal parameters (skips calibration entirely)"""
        return self._config['adaptive_parameters']['load_reduced_parameters']
    
    @property
    def REDUCED_TRITC_PATH(self) -> str:
        """Path to saved TRITC optimal parameters (1-3 parameter combinations only)"""
        return self._config['adaptive_parameters']['reduced_tritc_path']
    
    @property
    def REDUCED_FITC_PATH(self) -> str:
        """Path to saved FITC optimal parameters (1-3 parameter combinations only)"""
        return self._config['adaptive_parameters']['reduced_fitc_path']
    
    @property
    def RANDOMIZE_CALIBRATION_ORDER(self) -> bool:
        """Shuffle image processing order using random_seed"""
        return self._config['adaptive_parameters']['randomize_calibration_order']
    
    @property
    def RANDOM_SEED(self) -> Optional[int]:
        """
        Fixed seed for reproducibility (None = random each run).
        CRITICAL: Must stay constant across resume runs to maintain image order.
        """
        return self._config['adaptive_parameters']['random_seed']
    
    # --- Watershed Thresholding ---
    # These control watershed segmentation for separating touching foci
    
    @property
    def WATERSHED_MODE(self) -> str:
        """
        Threshold selection mode.
        Returns 'manual_interactive' or 'manual_preset'
        """
        return self._config['watershed']['mode']
    
    @property
    def MANUAL_WATERSHED_THRESHOLD_TRITC(self) -> float:
        """Pre-configured TRITC watershed threshold (0-100 scale)"""
        return self._config['watershed']['manual_threshold_tritc']
    
    @property
    def MANUAL_WATERSHED_THRESHOLD_FITC(self) -> float:
        """Pre-configured FITC watershed threshold (0-100 scale)"""
        return self._config['watershed']['manual_threshold_fitc']
    
    # --- Visualization ---
    # These control which images get visualization outputs
    
    @property
    def VISUALIZATION_MODE(self) -> str:
        """
        Visualization generation mode.
        Returns 'all', 'none', or 'specific'
        """
        return self._config['visualization']['mode']
    
    @property
    def SPECIFIC_VISUALIZATION_IMAGES(self) -> list:
        """
        List of well_position strings for specific visualization.
        Format: ['00029_00035', '00044_00021']
        Returns empty list if not specified or None in config.
        """
        result = self._config['visualization'].get('specific_images', [])
        # Handle None or [[]] cases from YAML
        return result if result is not None else []
    
    @property
    def SPECIFIC_VISUALIZATION_BASENAMES(self) -> list:
        """
        List of filename prefixes for specific visualization.
        Format: ['ATR1_24h--W00032--P00015--Z00000--T00000--']
        Returns empty list if not specified or None in config.
        """
        result = self._config['visualization'].get('specific_basenames', [])
        # Handle None or [[]] cases from YAML
        return result if result is not None else []
    
    @property
    def WATERSHED_MIN_DETECTION_PROB(self) -> float:
        """Minimum detection probability (0-100%) for foci to appear in visualizations"""
        return self._config['visualization']['watershed_min_detection_prob']
    
    @property
    def GENERATE_VISUALIZATIONS(self) -> bool:
        """
        Legacy property for backward compatibility.
        Returns True if visualization mode is 'all'.
        """
        return self._config['visualization']['mode'] == 'all'
    
    # --- Texture Filtering ---
    # These control texture-based filtering for uniform nuclei
    
    @property
    def ENABLE_TEXTURE_FILTERING(self) -> bool:
        """Apply stricter filtering to uniformly bright nuclei"""
        return self._config['texture_filtering']['enabled']
    
    @property
    def MIN_CV_THRESHOLD(self) -> float:
        """Minimum CV (coefficient of variation) for textured nucleus (0.0-1.0)"""
        return self._config['texture_filtering']['min_cv_threshold']
    
    @property
    def UNIFORM_CONTRAST_MULTIPLIER(self) -> float:
        """Contrast multiplier applied to uniform nuclei (>= 1.0)"""
        return self._config['texture_filtering']['uniform_contrast_multiplier']
    
    # --- Processing ---
    # These control parallel processing and batch management
    
    @property
    def MAX_WORKERS(self) -> Optional[int]:
        """
        Number of parallel workers (None = auto-detect).
        Auto-detect uses: min(8, CPU_cores - 1)
        """
        return self._config['processing']['max_workers']

    @property
    def MAX_IMAGES(self) -> Optional[int]:
        """
        Maximum images to process in this run (None = all images).
        Used for testing or incremental processing.
        """
        return self._config['processing'].get('max_images', None)
    
    @property
    def RESUME_FROM_IMAGE(self) -> int:
        """
        Image index to resume from (0 = start from beginning).
        Skips the first N images in the processing order.
        """
        return self._config['processing'].get('resume_from_image', 0)
    
    # ===============================================================
    # HELPER METHODS
    # ===============================================================
    
    def should_generate_visualization(self, well_number: str, position_number: str, base_name: str) -> bool:
        """
        Check if visualization should be generated for this image.
        
        This method implements the logic for the three visualization modes:
        - 'all': Always return True
        - 'none': Always return False  
        - 'specific': Check if image matches either the well_position list 
                      or the base_name list
        
        Parameters:
        -----------
        well_number : str
            Well number from image filename (e.g., "00032")
        position_number : str
            Position number from image filename (e.g., "00015")
        base_name : str
            Full filename prefix before channel name
            (e.g., "ATR1_24h--W00032--P00015--Z00000--T00000--")
        
        Returns:
        --------
        bool : True if visualization should be generated for this image
        
        Raises:
        -------
        ValueError : If visualization mode is not recognized
        """
        mode = self.VISUALIZATION_MODE
        
        if mode == 'all':
            return True
        
        elif mode == 'none':
            return False
        
        elif mode == 'specific':
            # Check well_position format (e.g., "00032_00015")
            # This format is more compact than full base_name
            image_id = f"{well_number}_{position_number}"
            if image_id in self.SPECIFIC_VISUALIZATION_IMAGES:
                return True
            
            # Check full base name (allows more specific matching)
            # Useful when multiple positions in same well need different treatment
            if base_name in self.SPECIFIC_VISUALIZATION_BASENAMES:
                return True
            
            # No match found in either list
            return False
        
        else:
            # This should never happen if config validation works correctly
            raise ValueError(f"Unknown visualization mode: {mode}")
    
    def print_config(self):
        """
        Print current configuration in readable, nested format.
        
        Useful for debugging and verifying that config.yaml was loaded correctly.
        Prints to stdout in a hierarchical structure matching the YAML file.
        """
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        
        def print_section(title, section_dict, indent=0):
            """
            Recursively print nested config sections with indentation.
            
            Parameters:
            -----------
            title : str
                Section name to print
            section_dict : dict
                Dictionary of settings in this section
            indent : int
                Current indentation level (0 = top level)
            """
            prefix = "  " * indent
            print(f"{prefix}{title}:")
            
            for key, value in section_dict.items():
                if isinstance(value, dict):
                    # Recursively print nested sections
                    print_section(key, value, indent + 1)
                else:
                    # Print leaf values (actual settings)
                    print(f"{prefix}  {key}: {value}")
        
        # Print all top-level sections
        for section, values in self._config.items():
            print_section(section, values)
        
        print("="*70 + "\n")
    
    def get_worker_params(self):
        """
        Get parameters needed by parallel worker processes.
        
        Worker processes are spawned via multiprocessing and need certain 
        configuration values. This method bundles them into a tuple for 
        easy unpacking in the worker function.
        
        These specific parameters are needed because:
        - watershed_min_prob: Workers filter foci by detection probability
        - min_cv: Workers calculate CV to identify uniform nuclei
        - contrast_multiplier: Workers apply stricter filtering to uniform nuclei
        - texture_enabled: Workers need to know if texture filtering is active
        
        Returns:
        --------
        tuple : (watershed_min_prob, min_cv, contrast_multiplier, texture_enabled)
                All values needed for worker process configuration
        """
        return (
            self.WATERSHED_MIN_DETECTION_PROB,
            self.MIN_CV_THRESHOLD,
            self.UNIFORM_CONTRAST_MULTIPLIER,
            self.ENABLE_TEXTURE_FILTERING
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return full configuration as dictionary.
        
        Useful for:
        - Serialization (saving config state to file)
        - Inspection (programmatically checking config values)
        - Logging (recording exact config used for a run)
        
        Returns a copy to prevent external modification of internal state.
        
        Returns:
        --------
        dict : Complete configuration dictionary
        """
        return self._config.copy()


# ===============================================================
# GLOBAL CONFIG INSTANCE
# ===============================================================
# Single global instance loaded at import time.
# Any module can import this to access configuration without re-parsing YAML.
#
# Usage in other modules:
#   from config_loader import config
#   folder = config.FOLDER_PATH
#
# This pattern ensures:
# - Config file is only parsed once (at first import)
# - All modules share the same configuration
# - No risk of accidentally loading different config files
config = Config()