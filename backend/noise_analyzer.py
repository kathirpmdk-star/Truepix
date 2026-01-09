"""
Noise Analyzer Module - High-Frequency Noise Residual Analysis
Extracts and analyzes noise patterns to detect AI-generated images
Weight: 0.1 in final score fusion
"""

import numpy as np
import cv2
from typing import Dict, Any
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter


class NoiseAnalyzer:
    """Noise residual analyzer for detecting AI-generated images"""
    
    def __init__(self):
        """Initialize noise analyzer"""
        print("✅ Noise Analyzer initialized")
    
    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze noise patterns in image to detect AI generation
        AI-generated images often have:
        - Unnatural noise distributions
        - Low noise variance (over-smoothed)
        - Inconsistent noise patterns across regions
        
        Args:
            image_array: Preprocessed image array (H, W, C) with values 0-1
            
        Returns:
            Dictionary containing:
                - score: Noise anomaly score (0-1, higher = more likely AI)
                - variance_score: Noise variance analysis
                - entropy_score: Noise entropy analysis
                - spatial_consistency: Spatial noise consistency
                - explanation: Human-readable explanation
        """
        print("\n🔍 Starting noise analysis...")
        
        # Convert to uint8 for processing
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert to grayscale
        if len(image_uint8.shape) == 3:
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_uint8
        
        # 1. Extract noise residual using high-pass filter
        noise_residual = self._extract_noise_residual(gray)
        
        # 2. Analyze noise variance
        variance_score = self._analyze_noise_variance(noise_residual)
        
        # 3. Analyze noise entropy
        entropy_score = self._analyze_noise_entropy(noise_residual)
        
        # 4. Analyze spatial consistency of noise
        spatial_consistency = self._analyze_spatial_consistency(noise_residual, gray)
        
        # 5. Analyze noise texture
        texture_score = self._analyze_noise_texture(noise_residual)
        
        # 6. Check for color noise inconsistencies (if color image)
        if len(image_uint8.shape) == 3:
            color_noise_score = self._analyze_color_noise(image_uint8)
        else:
            color_noise_score = 0.5
        
        # Combine scores (weighted combination)
        combined_score = (
            0.30 * variance_score +
            0.25 * entropy_score +
            0.25 * spatial_consistency +
            0.10 * texture_score +
            0.10 * color_noise_score
        )
        
        # Normalize to 0-1 range
        noise_score = float(np.clip(combined_score, 0, 1))
        
        # Generate explanation
        explanation = self._generate_explanation(
            noise_score, variance_score, entropy_score, 
            spatial_consistency, texture_score
        )
        
        result = {
            "score": noise_score,
            "variance_score": float(variance_score),
            "entropy_score": float(entropy_score),
            "spatial_consistency": float(spatial_consistency),
            "texture_score": float(texture_score),
            "color_noise_score": float(color_noise_score),
            "explanation": explanation
        }
        
        print(f"✅ Noise Analysis complete: score={noise_score:.3f}")
        
        return result
    
    def _extract_noise_residual(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency noise residual from image
        Uses difference between original and denoised version
        
        Args:
            gray_image: Grayscale image (0-255)
            
        Returns:
            Noise residual
        """
        # Apply Gaussian blur to get low-frequency components
        denoised = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
        
        # Extract high-frequency residual (noise)
        residual = gray_image.astype(np.float32) - denoised.astype(np.float32)
        
        return residual
    
    def _analyze_noise_variance(self, noise_residual: np.ndarray) -> float:
        """
        Analyze variance of noise residual
        AI images tend to have lower noise variance (over-smoothed)
        
        Args:
            noise_residual: Extracted noise residual
            
        Returns:
            Variance anomaly score (0-1)
        """
        # Calculate variance
        noise_var = np.var(noise_residual)
        
        # Natural images typically have noise variance in range 10-100
        # AI images often have much lower variance (<5) due to smoothing
        if noise_var < 5:
            # Very low variance = likely AI-generated (over-smoothed)
            variance_score = 0.8
        elif noise_var < 10:
            # Somewhat low variance
            variance_score = 0.6
        elif noise_var > 100:
            # Very high variance = possibly added noise to fool detection
            variance_score = 0.7
        else:
            # Normal range
            variance_score = 0.3
        
        return float(variance_score)
    
    def _analyze_noise_entropy(self, noise_residual: np.ndarray) -> float:
        """
        Analyze entropy of noise distribution
        Natural noise has high entropy; AI noise is more structured
        
        Args:
            noise_residual: Extracted noise residual
            
        Returns:
            Entropy anomaly score (0-1)
        """
        # Normalize residual to positive range for histogram
        normalized = noise_residual - np.min(noise_residual)
        normalized = normalized / (np.max(normalized) + 1e-10)
        
        # Compute histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 1))
        
        # Normalize histogram to probability distribution
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Calculate entropy
        noise_entropy = entropy(hist + 1e-10)
        
        # Natural noise has high entropy (~7-8 bits)
        # AI-generated images have lower entropy (~4-6 bits)
        max_entropy = np.log2(256)  # ~8 bits
        normalized_entropy = noise_entropy / max_entropy
        
        # Low entropy = more likely AI
        entropy_score = 1 - normalized_entropy
        
        return float(np.clip(entropy_score, 0, 1))
    
    def _analyze_spatial_consistency(self, noise_residual: np.ndarray, 
                                    gray_image: np.ndarray) -> float:
        """
        Analyze spatial consistency of noise across image regions
        Natural noise is consistent; AI noise can be inconsistent
        
        Args:
            noise_residual: Extracted noise residual
            gray_image: Original grayscale image
            
        Returns:
            Spatial inconsistency score (0-1)
        """
        h, w = gray_image.shape
        
        # Divide image into blocks
        block_size = 32
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks < 2 or w_blocks < 2:
            return 0.5
        
        # Calculate noise variance for each block
        block_variances = []
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = noise_residual[i*block_size:(i+1)*block_size,
                                      j*block_size:(j+1)*block_size]
                block_variances.append(np.var(block))
        
        block_variances = np.array(block_variances)
        
        # Analyze consistency of variances across blocks
        # High variation in block variances = inconsistent noise = likely AI
        var_of_vars = np.var(block_variances)
        mean_var = np.mean(block_variances)
        
        if mean_var > 0:
            coefficient_of_variation = np.sqrt(var_of_vars) / mean_var
            # High CV = inconsistent = likely AI
            consistency_score = np.clip(coefficient_of_variation / 2, 0, 1)
        else:
            consistency_score = 0.5
        
        return float(consistency_score)
    
    def _analyze_noise_texture(self, noise_residual: np.ndarray) -> float:
        """
        Analyze texture characteristics of noise
        Natural noise has random texture; AI noise can be structured
        
        Args:
            noise_residual: Extracted noise residual
            
        Returns:
            Texture anomaly score (0-1)
        """
        # Compute gradients
        dy, dx = np.gradient(noise_residual)
        
        # Calculate gradient magnitude
        gradient_mag = np.sqrt(dx**2 + dy**2)
        
        # Natural noise has isotropic gradients (equal in all directions)
        # AI noise can have directional bias
        
        # Compute gradient orientation
        gradient_orient = np.arctan2(dy, dx)
        
        # Compute histogram of orientations
        hist, _ = np.histogram(gradient_orient, bins=36, range=(-np.pi, np.pi))
        
        # Normalize histogram
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Calculate uniformity (should be uniform for natural noise)
        expected_uniform = 1.0 / 36
        chi_square = np.sum((hist - expected_uniform)**2 / (expected_uniform + 1e-10))
        
        # Normalize chi-square
        texture_score = np.clip(chi_square / 10, 0, 1)
        
        return float(texture_score)
    
    def _analyze_color_noise(self, color_image: np.ndarray) -> float:
        """
        Analyze noise consistency across color channels
        Natural images have correlated noise; AI may have uncorrelated noise
        
        Args:
            color_image: Color image (H, W, 3) with values 0-255
            
        Returns:
            Color noise anomaly score (0-1)
        """
        # Extract noise from each channel
        noise_channels = []
        
        for c in range(3):
            channel = color_image[:, :, c]
            denoised = cv2.GaussianBlur(channel, (5, 5), 1.5)
            residual = channel.astype(np.float32) - denoised.astype(np.float32)
            noise_channels.append(residual)
        
        # Calculate correlation between noise in different channels
        r_g_corr = np.corrcoef(noise_channels[0].flatten(), 
                              noise_channels[1].flatten())[0, 1]
        r_b_corr = np.corrcoef(noise_channels[0].flatten(), 
                              noise_channels[2].flatten())[0, 1]
        g_b_corr = np.corrcoef(noise_channels[1].flatten(), 
                              noise_channels[2].flatten())[0, 1]
        
        # Average correlation
        avg_corr = (abs(r_g_corr) + abs(r_b_corr) + abs(g_b_corr)) / 3
        
        # Natural images have some correlation (0.3-0.7)
        # AI images may have very low or very high correlation
        if avg_corr < 0.2 or avg_corr > 0.8:
            color_score = 0.7
        else:
            color_score = 0.3
        
        return float(color_score)
    
    def _generate_explanation(self, noise_score: float, variance: float,
                             entropy_score: float, spatial: float, 
                             texture: float) -> str:
        """
        Generate human-readable explanation of noise analysis
        
        Args:
            noise_score: Overall noise score
            variance: Variance score
            entropy_score: Entropy score
            spatial: Spatial consistency score
            texture: Texture score
            
        Returns:
            Explanation string
        """
        explanations = []
        
        if noise_score > 0.6:
            explanations.append("Significant noise anomalies detected")
        elif noise_score > 0.4:
            explanations.append("Moderate noise irregularities")
        else:
            explanations.append("Normal noise patterns")
        
        if variance > 0.6:
            explanations.append("unusual noise variance (over-smoothed or synthetic)")
        
        if entropy_score > 0.6:
            explanations.append("low noise entropy (structured rather than random)")
        
        if spatial > 0.6:
            explanations.append("spatially inconsistent noise patterns")
        
        if texture > 0.6:
            explanations.append("non-isotropic noise texture")
        
        return "; ".join(explanations) if explanations else "Natural noise characteristics"
