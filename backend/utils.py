"""
Utility functions for image processing and analysis
"""

import numpy as np
from PIL import Image, ImageStat
from typing import Dict, Tuple


def calculate_image_metrics(image: Image.Image) -> Dict[str, float]:
    """
    Calculate various image quality metrics
    
    Args:
        image: PIL Image
    
    Returns:
        Dictionary with metrics
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image statistics
    stat = ImageStat.Stat(image)
    
    # Calculate metrics
    metrics = {
        'mean_brightness': sum(stat.mean) / 3,
        'std_brightness': sum(stat.stddev) / 3,
        'mean_r': stat.mean[0],
        'mean_g': stat.mean[1],
        'mean_b': stat.mean[2],
        'width': image.size[0],
        'height': image.size[1],
        'aspect_ratio': image.size[0] / image.size[1]
    }
    
    return metrics


def detect_color_cast(image: Image.Image) -> str:
    """
    Detect if image has a color cast (common in AI images)
    
    Args:
        image: PIL Image
    
    Returns:
        Color cast description
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    stat = ImageStat.Stat(image)
    r, g, b = stat.mean
    
    # Check for significant color imbalance
    if r > g * 1.2 and r > b * 1.2:
        return "Red color cast detected"
    elif g > r * 1.2 and g > b * 1.2:
        return "Green color cast detected"
    elif b > r * 1.2 and b > g * 1.2:
        return "Blue color cast detected"
    else:
        return "Balanced color distribution"


def estimate_jpeg_quality(image: Image.Image) -> int:
    """
    Estimate JPEG compression quality (rough approximation)
    
    Args:
        image: PIL Image
    
    Returns:
        Estimated quality (0-100)
    """
    # This is a simplified heuristic
    # Real implementation would analyze DCT coefficients
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate edge strength (high quality = sharp edges)
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    
    # Simple gradient calculation
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    
    edge_strength = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
    
    # Map to quality score (heuristic)
    quality = min(100, int(edge_strength * 10))
    
    return quality


def calculate_histogram_variance(image: Image.Image) -> float:
    """
    Calculate histogram variance (AI images often have smoother histograms)
    
    Args:
        image: PIL Image
    
    Returns:
        Variance score
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get histogram
    hist = image.histogram()
    
    # Calculate variance across all channels
    variance = np.var(hist)
    
    return variance


def check_image_size_consistency(image: Image.Image) -> bool:
    """
    Check if image size is typical for AI generators
    
    Args:
        image: PIL Image
    
    Returns:
        True if size is typical for AI generation
    """
    width, height = image.size
    
    # Common AI generator sizes
    common_ai_sizes = [
        (512, 512), (768, 768), (1024, 1024),
        (512, 768), (768, 512),
        (1024, 1536), (1536, 1024)
    ]
    
    return (width, height) in common_ai_sizes


def validate_image_format(image_data: bytes) -> Tuple[bool, str]:
    """
    Validate image format and integrity
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        (is_valid, message)
    """
    try:
        from io import BytesIO
        img = Image.open(BytesIO(image_data))
        img.verify()
        
        # Check format
        if img.format not in ['JPEG', 'PNG']:
            return False, f"Unsupported format: {img.format}"
        
        # Check size
        if img.size[0] < 32 or img.size[1] < 32:
            return False, "Image too small (minimum 32x32)"
        
        if img.size[0] > 4096 or img.size[1] > 4096:
            return False, "Image too large (maximum 4096x4096)"
        
        return True, "Valid image"
    
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def get_file_size_mb(image_data: bytes) -> float:
    """Get file size in megabytes"""
    return len(image_data) / (1024 * 1024)


# Export all utility functions
__all__ = [
    'calculate_image_metrics',
    'detect_color_cast',
    'estimate_jpeg_quality',
    'calculate_histogram_variance',
    'check_image_size_consistency',
    'validate_image_format',
    'get_file_size_mb'
]
