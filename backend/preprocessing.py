"""
Image Preprocessing Module
Handles image preprocessing: resize, normalize, strip metadata
"""

import io
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import piexif


class ImagePreprocessor:
    """Preprocesses images for AI detection analysis"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 jpeg_quality: int = 90):
        """
        Initialize preprocessor with target size and quality
        
        Args:
            target_size: Target image dimensions (width, height)
            jpeg_quality: JPEG compression quality for normalization (1-100)
        """
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        print(f"✅ Image preprocessor initialized: {target_size}, quality={jpeg_quality}")
    
    def preprocess(self, image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
        """
        Complete preprocessing pipeline:
        1. Load image
        2. Strip metadata
        3. Resize to target size
        4. Normalize compression (JPEG quality=90)
        5. Convert to numpy array for analysis
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            Tuple of (PIL Image, numpy array)
        """
        # Step 1: Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        print(f"📷 Original image: {image.size}, mode={image.mode}")
        
        # Step 2: Strip metadata (EXIF, IPTC, XMP)
        image = self._strip_metadata(image)
        
        # Step 3: Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            print(f"🔄 Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Step 4: Resize to target size
        image = self._resize_image(image)
        
        # Step 5: Normalize compression (re-encode as JPEG with quality=90)
        image = self._normalize_compression(image)
        
        # Step 6: Convert to numpy array for analysis
        image_array = np.array(image).astype(np.float32)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0
        
        print(f"✅ Preprocessed image: shape={image_array.shape}, dtype={image_array.dtype}")
        
        return image, image_array
    
    def _strip_metadata(self, image: Image.Image) -> Image.Image:
        """
        Remove all metadata (EXIF, IPTC, XMP) from image
        
        Args:
            image: PIL Image object
            
        Returns:
            Image without metadata
        """
        # Get image data without metadata
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        
        print("🗑️  Stripped metadata from image")
        return image_without_exif
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size using high-quality resampling
        
        Args:
            image: PIL Image object
            
        Returns:
            Resized image
        """
        if image.size != self.target_size:
            # Use LANCZOS for high-quality downsampling
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            print(f"📐 Resized image to {self.target_size}")
        else:
            print(f"📐 Image already at target size {self.target_size}")
        
        return image
    
    def _normalize_compression(self, image: Image.Image) -> Image.Image:
        """
        Normalize compression artifacts by re-encoding as JPEG with quality=90
        This helps standardize compression artifacts across images
        
        Args:
            image: PIL Image object
            
        Returns:
            Re-encoded image
        """
        # Save to bytes buffer with specified JPEG quality
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality)
        buffer.seek(0)
        
        # Reload image from buffer
        normalized_image = Image.open(buffer)
        
        print(f"🔧 Normalized compression (JPEG quality={self.jpeg_quality})")
        
        return normalized_image
    
    def preprocess_for_cnn(self, image_array: np.ndarray) -> np.ndarray:
        """
        Additional preprocessing for CNN models (ImageNet normalization)
        
        Args:
            image_array: Numpy array of image (0-1 normalized)
            
        Returns:
            Normalized array for CNN input
        """
        # ImageNet normalization parameters
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Apply normalization
        normalized = (image_array - mean) / std
        
        # Add batch dimension and convert to channels-first format (CHW)
        normalized = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        normalized = np.expand_dims(normalized, axis=0)    # CHW -> BCHW
        
        print(f"🧠 Preprocessed for CNN: shape={normalized.shape}")
        
        return normalized.astype(np.float32)
    
    def get_image_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes
        
        Args:
            image: PIL Image object
            
        Returns:
            Image as bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality)
        return buffer.getvalue()
