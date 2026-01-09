"""
Platform Simulator - Social Media Image Transformations
Simulates how images are processed by different social media platforms
"""

import io
import numpy as np
from PIL import Image
from typing import Dict, List


class PlatformSimulator:
    """
    Simulates image transformations applied by social media platforms
    """
    
    # Platform-specific transformation parameters
    PLATFORMS = {
        'whatsapp': {
            'max_dimension': 512,
            'jpeg_quality': 40,
            'description': 'WhatsApp compression (aggressive)'
        },
        'instagram': {
            'max_dimension': 1080,
            'jpeg_quality': 70,
            'description': 'Instagram compression (moderate)'
        },
        'facebook': {
            'max_dimension': 960,
            'jpeg_quality': 60,
            'description': 'Facebook compression (moderate-high)'
        }
    }
    
    def transform(self, image: Image.Image, platform: str) -> Image.Image:
        """
        Transform image to simulate platform processing
        
        Args:
            image: Original PIL Image
            platform: Platform name ('whatsapp', 'instagram', 'facebook')
        
        Returns:
            Transformed PIL Image
        """
        if platform not in self.PLATFORMS:
            raise ValueError(f"Unknown platform: {platform}")
        
        params = self.PLATFORMS[platform]
        
        # Step 1: Resize to platform's max dimension
        transformed = self._resize_image(image, params['max_dimension'])
        
        # Step 2: Apply JPEG compression
        transformed = self._apply_jpeg_compression(transformed, params['jpeg_quality'])
        
        return transformed
    
    def _resize_image(self, image: Image.Image, max_dimension: int) -> Image.Image:
        """
        Resize image maintaining aspect ratio
        
        Args:
            image: Original image
            max_dimension: Maximum width or height
        
        Returns:
            Resized image
        """
        width, height = image.size
        
        # Calculate scaling factor
        if width > height:
            if width > max_dimension:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                return image
        else:
            if height > max_dimension:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                return image
        
        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized
    
    def _apply_jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        """
        Apply JPEG compression to simulate platform processing
        
        Args:
            image: Original image
            quality: JPEG quality (0-100)
        
        Returns:
            Compressed image
        """
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[3])
            else:
                rgb_image.paste(image)
            image = rgb_image
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        return compressed
    
    def calculate_stability(self, predictions: List[float]) -> float:
        """
        Calculate prediction stability score across platforms
        
        Stability = consistency of predictions across transformations
        Higher score = more stable/robust model
        
        Args:
            predictions: List of confidence scores (0-1) from different platforms
        
        Returns:
            Stability score (0-100)
        """
        if len(predictions) < 2:
            return 100.0
        
        # Calculate standard deviation
        std_dev = np.std(predictions)
        
        # Convert to stability score (lower variance = higher stability)
        # Scale: std_dev of 0 = 100%, std_dev of 0.5 = 0%
        stability = max(0, 100 - (std_dev * 200))
        
        return stability
    
    def get_platform_info(self, platform: str) -> Dict[str, any]:
        """Get information about a platform's transformation parameters"""
        if platform not in self.PLATFORMS:
            return {}
        return self.PLATFORMS[platform]
    
    def get_all_platforms(self) -> List[str]:
        """Get list of all supported platforms"""
        return list(self.PLATFORMS.keys())


# Utility function for batch platform simulation
def simulate_all_platforms(image: Image.Image) -> Dict[str, Image.Image]:
    """
    Transform image for all supported platforms
    
    Args:
        image: Original PIL Image
    
    Returns:
        Dictionary mapping platform names to transformed images
    """
    simulator = PlatformSimulator()
    results = {}
    
    for platform in simulator.get_all_platforms():
        results[platform] = simulator.transform(image, platform)
    
    return results
