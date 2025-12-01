"""
Configuration loader for YAML-based configuration
"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration container that loads from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration sections exist."""
        required_sections = [
            'data', 'parameter_space', 'adaptive_parameters',
            'watershed', 'visualization', 'texture_filtering', 'processing'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
    
    # ===============================================================
    # DATA INPUT
    # ===============================================================
    @property
    def FOLDER_PATH(self) -> str:
        return self._config['data']['folder_path']
    
    # ===============================================================
    # PARAMETER SPACE
    # ===============================================================
    @property
    def GENERATE_NEW_PARAMETER_SPACE(self) -> bool:
        return self._config['parameter_space']['generate_new']
    
    @property
    def TRITC_PARAMETER_SPACE_PATH(self) -> str:
        return self._config['parameter_space']['tritc_path']
    
    @property
    def FITC_PARAMETER_SPACE_PATH(self) -> str:
        return self._config['parameter_space']['fitc_path']
    
    @property
    def N_PARAMETER_SAMPLES(self) -> int:
        return self._config['parameter_space']['n_samples']
    
    @property
    def N_SOBOL_SAMPLES(self) -> int:
        return self._config['parameter_space']['n_sobol_samples']
    
    # ===============================================================
    # ADAPTIVE PARAMETERS
    # ===============================================================
    @property
    def USE_ADAPTIVE_PARAMETERS(self) -> bool:
        return self._config['adaptive_parameters']['enabled']
    
    @property
    def N_CALIBRATION_IMAGES(self) -> int:
        return self._config['adaptive_parameters']['n_calibration_images']
    
    @property
    def N_CALIBRATION_PARAMS(self) -> int:
        return self._config['adaptive_parameters']['n_calibration_params']
    
    @property
    def N_PRODUCTION_PARAMS(self) -> int:
        return self._config['adaptive_parameters']['n_production_params']
    
    @property
    def CALIBRATION_SAVE_PATH(self) -> str:
        return self._config['adaptive_parameters']['calibration_save_path']
    
    @property
    def RECALIBRATE(self) -> bool:
        return self._config['adaptive_parameters']['recalibrate']
    
    @property
    def LOAD_REDUCED_PARAMETERS(self) -> bool:
        return self._config['adaptive_parameters']['load_reduced_parameters']
    
    @property
    def REDUCED_TRITC_PATH(self) -> str:
        return self._config['adaptive_parameters']['reduced_tritc_path']
    
    @property
    def REDUCED_FITC_PATH(self) -> str:
        return self._config['adaptive_parameters']['reduced_fitc_path']
    
    @property
    def RANDOMIZE_CALIBRATION_ORDER(self) -> bool:
        return self._config['adaptive_parameters']['randomize_calibration_order']
    
    @property
    def RANDOM_SEED(self) -> Optional[int]:
        return self._config['adaptive_parameters']['random_seed']
    
    # ===============================================================
    # WATERSHED THRESHOLDING
    # ===============================================================
    @property
    def WATERSHED_MODE(self) -> str:
        """Options: 'manual_interactive', 'manual_preset'"""
        return self._config['watershed']['mode']
    
    @property
    def MANUAL_WATERSHED_THRESHOLD_TRITC(self) -> float:
        return self._config['watershed']['manual_threshold_tritc']
    
    @property
    def MANUAL_WATERSHED_THRESHOLD_FITC(self) -> float:
        return self._config['watershed']['manual_threshold_fitc']
    
    # ===============================================================
    # VISUALIZATION
    # ===============================================================
    @property
    def VISUALIZATION_MODE(self) -> str:
        """Options: 'all', 'none', 'specific'"""
        return self._config['visualization']['mode']
    
    @property
    def SPECIFIC_VISUALIZATION_IMAGES(self) -> list:
        result = self._config['visualization'].get('specific_images', [])
        return result if result is not None else []
    
    @property
    def SPECIFIC_VISUALIZATION_BASENAMES(self) -> list:
        result = self._config['visualization'].get('specific_basenames', [])
        return result if result is not None else []
    
    @property
    def WATERSHED_MIN_DETECTION_PROB(self) -> float:
        return self._config['visualization']['watershed_min_detection_prob']
    
    @property
    def GENERATE_VISUALIZATIONS(self) -> bool:
        """Legacy property for backward compatibility."""
        return self._config['visualization']['mode'] == 'all'
    
    # ===============================================================
    # TEXTURE FILTERING
    # ===============================================================
    @property
    def ENABLE_TEXTURE_FILTERING(self) -> bool:
        return self._config['texture_filtering']['enabled']
    
    @property
    def MIN_CV_THRESHOLD(self) -> float:
        return self._config['texture_filtering']['min_cv_threshold']
    
    @property
    def UNIFORM_CONTRAST_MULTIPLIER(self) -> float:
        return self._config['texture_filtering']['uniform_contrast_multiplier']
    
    # ===============================================================
    # PROCESSING
    # ===============================================================
    @property
    def MAX_WORKERS(self) -> Optional[int]:
        return self._config['processing']['max_workers']

    @property
    def MAX_IMAGES(self) -> Optional[int]:
        return self._config['processing'].get('max_images', None)
    
    @property
    def RESUME_FROM_IMAGE(self) -> int:
        return self._config['processing'].get('resume_from_image', 0)
    
    # ===============================================================
    # HELPER METHODS
    # ===============================================================
    def should_generate_visualization(self, well_number: str, position_number: str, base_name: str) -> bool:
        """
        Determine if visualization should be generated for a specific image.
        """
        mode = self.VISUALIZATION_MODE
        
        if mode == 'all':
            return True
        elif mode == 'none':
            return False
        elif mode == 'specific':
            # Check well_position format
            image_id = f"{well_number}_{position_number}"
            if image_id in self.SPECIFIC_VISUALIZATION_IMAGES:
                return True
            
            # Check base name
            if base_name in self.SPECIFIC_VISUALIZATION_BASENAMES:
                return True
            
            return False
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
    
    def print_config(self):
        """Print current configuration in a readable format."""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        
        def print_section(title, section_dict, indent=0):
            prefix = "  " * indent
            print(f"{prefix}{title}:")
            for key, value in section_dict.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{prefix}  {key}: {value}")
        
        for section, values in self._config.items():
            print_section(section, values)
        
        print("="*70 + "\n")
    
    def get_worker_params(self):
        """
        Get parameters that need to be passed to worker processes.
        """
        return (
            self.WATERSHED_MIN_DETECTION_PROB,
            self.MIN_CV_THRESHOLD,
            self.UNIFORM_CONTRAST_MULTIPLIER,
            self.ENABLE_TEXTURE_FILTERING
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self._config.copy()


# Create global config instance
config = Config()