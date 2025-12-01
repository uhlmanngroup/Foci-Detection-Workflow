"""
Configuration loader for Foci Detection Workflow
================================================
Loads configuration from YAML file and provides easy access to settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration container for foci detection analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._set_attributes()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _set_attributes(self):
        """Set configuration values as class attributes for easy access."""
        # GPU Settings
        self.USE_GPU = self._config['gpu']['use_gpu']
        self.GPU_BATCH_CLEAR = self._config['gpu']['gpu_batch_clear']
        
        # Data Input
        self.FOLDER_PATH = self._config['data']['folder_path']
        
        # Parameter Space
        self.GENERATE_NEW_PARAMETER_SPACE = self._config['parameter_space']['generate_new']
        self.TRITC_PARAMETER_SPACE_PATH = self._config['parameter_space']['tritc_path']
        self.FITC_PARAMETER_SPACE_PATH = self._config['parameter_space']['fitc_path']
        self.N_PARAMETER_SAMPLES = self._config['parameter_space']['n_samples']
        self.N_SOBOL_SAMPLES = self._config['parameter_space']['n_sobol_samples']
        
        # Calibration
        self.CALIBRATION_MODE = self._config['calibration']['mode']
        self.CALIBRATION_IMAGE_LIMIT = self._config['calibration']['image_limit']
        self.MANUAL_WATERSHED_THRESHOLD_TRITC = self._config['calibration']['manual_threshold_tritc']
        self.MANUAL_WATERSHED_THRESHOLD_FITC = self._config['calibration']['manual_threshold_fitc']
        
        # Visualization
        self.GENERATE_VISUALIZATIONS = self._config['visualization']['generate']
        self.WATERSHED_MIN_DETECTION_PROB = self._config['visualization']['watershed_min_detection_prob']
        
        # Texture Filtering
        self.ENABLE_TEXTURE_FILTERING = self._config['texture_filtering']['enable']
        self.MIN_CV_THRESHOLD = self._config['texture_filtering']['min_cv_threshold']
        self.UNIFORM_CONTRAST_MULTIPLIER = self._config['texture_filtering']['uniform_contrast_multiplier']
        
        # Parallel Processing
        self.MAX_WORKERS = self._config['parallel']['max_workers']
        
        # Adaptive Parameters
        self.USE_ADAPTIVE_PARAMETERS = self._config['adaptive_parameters']['use_adaptive']
        self.N_CALIBRATION_IMAGES = self._config['adaptive_parameters']['n_calibration_images']
        self.N_CALIBRATION_PARAMS = self._config['adaptive_parameters']['n_calibration_params']
        self.N_PRODUCTION_PARAMS = self._config['adaptive_parameters']['n_production_params']
        self.CALIBRATION_SAVE_PATH = self._config['adaptive_parameters']['calibration_save_path']
        self.RECALIBRATE = self._config['adaptive_parameters']['recalibrate']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excludes methods and private attributes)."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
    
    def print_config(self):
        """Print all configuration settings."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        for key, value in self.to_dict().items():
            if key != 'config_path':
                print(f"{key:40s} = {value}")
        print("="*70 + "\n")
    
    def get_worker_params(self):
        """
        Get parameters that need to be passed to worker processes.
        Returns a tuple in the correct order for process_single_nucleus.
        """
        return (
            self.WATERSHED_MIN_DETECTION_PROB,
            self.MIN_CV_THRESHOLD,
            self.UNIFORM_CONTRAST_MULTIPLIER,
            self.ENABLE_TEXTURE_FILTERING
        )
    
    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_config()
        self._set_attributes()


# For backwards compatibility, create a class-based version
class ConfigClass:
    """
    Class-based config loader (backwards compatible with original Config class).
    Usage: from config import ConfigClass as Config
    """
    _instance = None
    _config_obj = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config_obj = Config()
        return cls._instance
    
    def __getattribute__(self, name):
        if name in ['_instance', '_config_obj', '__class__', '__new__']:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, '_config_obj'), name)

