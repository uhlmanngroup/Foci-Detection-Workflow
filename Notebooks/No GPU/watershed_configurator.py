"""
Watershed threshold configuration helper
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure, img_as_float
from typing import Dict, Tuple


class WatershedConfigurator:
    """Helper class for configuring watershed thresholds."""
    
    def __init__(self):
        self.thresholds = {}
    
    def configure_threshold_interactive(self, 
                                       image: np.ndarray, 
                                       channel_name: str,
                                       well_number: str,
                                       position_number: str,
                                       folder_path: str,
                                       base_name: str) -> float:
        """
        Interactively configure watershed threshold for one channel.
        
        Parameters:
        -----------
        image : ndarray
            Channel image
        channel_name : str
            'TRITC' or 'FITC'
        well_number : str
            Well identifier
        position_number : str
            Position identifier
        folder_path : str
            Path to save brightness plot
        base_name : str
            Image base name
            
        Returns:
        --------
        float : Selected threshold value
        """
        print(f"\n📊 Analyzing {channel_name} brightness distribution...")
        
        # Prepare filtered image
        sample_isolated = img_as_float(image.copy())
        sample_filtered = filters.difference_of_gaussians(sample_isolated, low_sigma=1, high_sigma=2)
        sample_filtered = np.clip(sample_filtered, 0, None)
        sample_filtered = exposure.rescale_intensity(sample_filtered, in_range='image', out_range=(0, 100))
        
        # Generate brightness plot
        from nucleus_worker_Visualization import analyze_brightness_percentiles
        fig = analyze_brightness_percentiles(sample_filtered, channel_name)
        
        # Save plot
        plot_path = f"{folder_path}/{base_name}brightness_analysis_{channel_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Brightness plot saved: {plot_path}")
        
        # Get user input
        threshold = float(input(f"\n🎯 Enter watershed threshold for {channel_name} (e.g., 50): "))
        plt.close(fig)
        
        # Store threshold
        image_id = f"{well_number}_{position_number}"
        if image_id not in self.thresholds:
            self.thresholds[image_id] = {}
        self.thresholds[image_id][channel_name.lower()] = threshold
        
        return threshold
    
    def configure_batch_at_start(self, 
                                images_info: list,
                                channels: list = ['TRITC', 'FITC']) -> Dict[str, Dict[str, float]]:
        """
        Configure thresholds for multiple images at the start of a run.
        
        Parameters:
        -----------
        images_info : list
            List of dicts with keys: 'image', 'channel_name', 'well_number', 
            'position_number', 'folder_path', 'base_name'
        channels : list
            List of channel names to configure
            
        Returns:
        --------
        dict : Nested dict of thresholds {image_id: {channel: threshold}}
        """
        print("\n" + "="*70)
        print("🎨 WATERSHED THRESHOLD CONFIGURATION")
        print("="*70)
        print(f"You will configure thresholds for {len(images_info)} image(s)")
        print("="*70 + "\n")
        
        for idx, info in enumerate(images_info, 1):
            print(f"\n--- Image {idx}/{len(images_info)} ---")
            print(f"Well: {info['well_number']}, Position: {info['position_number']}")
            
            for channel in channels:
                if channel in info:
                    self.configure_threshold_interactive(
                        image=info[channel],
                        channel_name=channel,
                        well_number=info['well_number'],
                        position_number=info['position_number'],
                        folder_path=info['folder_path'],
                        base_name=info['base_name']
                    )
        
        print("\n" + "="*70)
        print("✅ THRESHOLD CONFIGURATION COMPLETE")
        print("="*70)
        self.print_summary()
        
        return self.thresholds
    
    def print_summary(self):
        """Print configured thresholds."""
        print("\n📋 Configured Thresholds:")
        for image_id, channels in self.thresholds.items():
            print(f"  {image_id}:")
            for channel, threshold in channels.items():
                print(f"    {channel.upper()}: {threshold}")
    
    def export_to_yaml_format(self) -> str:
        """
        Export thresholds in YAML format for pasting into config.yaml.
        
        Returns:
        --------
        str : YAML-formatted string
        """
        lines = ["  image_thresholds:"]
        for image_id, channels in self.thresholds.items():
            tritc = channels.get('tritc', 26.0)
            fitc = channels.get('fitc', 26.0)
            lines.append(f'    "{image_id}": {{"tritc": {tritc}, "fitc": {fitc}}}')
        
        return "\n".join(lines)