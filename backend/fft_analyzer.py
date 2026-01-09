"""
FFT Analyzer Module - Frequency Domain Analysis
Detects generative frequency artifacts using Fast Fourier Transform
Weight: 0.2 in final score fusion
"""

import numpy as np
import cv2
from typing import Dict, Any
from scipy import fftpack
from scipy.stats import entropy


class FFTAnalyzer:
    """FFT-based analyzer for detecting AI-generated image artifacts"""
    
    def __init__(self):
        """Initialize FFT analyzer"""
        print("✅ FFT Analyzer initialized")
    
    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image using FFT to detect frequency domain artifacts
        AI-generated images often have:
        - Unusual frequency patterns
        - Regular periodic artifacts
        - Anomalous high-frequency components
        
        Args:
            image_array: Preprocessed image array (H, W, C) with values 0-1
            
        Returns:
            Dictionary containing:
                - score: FFT anomaly score (0-1, higher = more likely AI)
                - frequency_anomaly: Frequency pattern irregularity
                - periodic_artifacts: Detection of repeating patterns
                - high_freq_score: High frequency component analysis
                - explanation: Human-readable explanation
        """
        print("\n🔬 Starting FFT analysis...")
        
        # Convert to grayscale for FFT analysis
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        # Ensure proper scale (0-255)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Compute 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Compute power spectrum
        power_spectrum = magnitude_spectrum ** 2
        
        # 1. Analyze frequency distribution
        frequency_anomaly = self._analyze_frequency_distribution(
            power_spectrum, magnitude_spectrum
        )
        
        # 2. Detect periodic artifacts (regular patterns)
        periodic_artifacts = self._detect_periodic_artifacts(magnitude_spectrum)
        
        # 3. Analyze high-frequency components
        high_freq_score = self._analyze_high_frequencies(
            power_spectrum, gray_uint8.shape
        )
        
        # 4. Detect DCT block artifacts (JPEG compression artifacts)
        block_artifacts = self._detect_block_artifacts(gray_uint8)
        
        # 5. Analyze radial frequency profile
        radial_anomaly = self._analyze_radial_profile(power_spectrum)
        
        # Combine scores (weighted combination)
        combined_score = (
            0.30 * frequency_anomaly +
            0.25 * periodic_artifacts +
            0.25 * high_freq_score +
            0.10 * block_artifacts +
            0.10 * radial_anomaly
        )
        
        # Normalize to 0-1 range
        fft_score = float(np.clip(combined_score, 0, 1))
        
        # Generate explanation
        explanation = self._generate_explanation(
            fft_score, frequency_anomaly, periodic_artifacts, 
            high_freq_score, block_artifacts
        )
        
        result = {
            "score": fft_score,
            "frequency_anomaly": float(frequency_anomaly),
            "periodic_artifacts": float(periodic_artifacts),
            "high_freq_score": float(high_freq_score),
            "block_artifacts": float(block_artifacts),
            "radial_anomaly": float(radial_anomaly),
            "explanation": explanation
        }
        
        print(f"✅ FFT Analysis complete: score={fft_score:.3f}")
        
        return result
    
    def _analyze_frequency_distribution(self, power_spectrum: np.ndarray, 
                                       magnitude_spectrum: np.ndarray) -> float:
        """
        Analyze frequency distribution for anomalies
        AI images often have unusual frequency distributions
        
        Args:
            power_spectrum: Power spectrum from FFT
            magnitude_spectrum: Magnitude spectrum from FFT
            
        Returns:
            Anomaly score (0-1)
        """
        # Log-scale power spectrum for better analysis
        log_power = np.log1p(power_spectrum)
        
        # Compute entropy of frequency distribution
        # AI images tend to have lower entropy (more regular)
        freq_entropy = entropy(log_power.flatten() + 1e-10)
        
        # Normalize entropy (typical range: 5-15)
        normalized_entropy = np.clip((15 - freq_entropy) / 10, 0, 1)
        
        # Analyze uniformity of frequency distribution
        freq_std = np.std(log_power)
        freq_mean = np.mean(log_power)
        
        # Coefficient of variation
        if freq_mean > 0:
            cv = freq_std / freq_mean
            # AI images tend to have lower variation
            uniformity_score = np.clip(1 - cv / 2, 0, 1)
        else:
            uniformity_score = 0.5
        
        # Combine metrics
        anomaly_score = 0.6 * normalized_entropy + 0.4 * uniformity_score
        
        return float(anomaly_score)
    
    def _detect_periodic_artifacts(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Detect periodic artifacts (repeating patterns) in frequency domain
        AI generators sometimes create regular patterns
        
        Args:
            magnitude_spectrum: Magnitude spectrum from FFT
            
        Returns:
            Periodic artifact score (0-1)
        """
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Remove DC component (center)
        magnitude_no_dc = magnitude_spectrum.copy()
        magnitude_no_dc[center_y-5:center_y+5, center_x-5:center_x+5] = 0
        
        # Find peaks in frequency domain
        # Strong regular peaks indicate periodic artifacts
        threshold = np.percentile(magnitude_no_dc, 99)
        peaks = magnitude_no_dc > threshold
        num_peaks = np.sum(peaks)
        
        # Check for symmetry (AI artifacts are often symmetric)
        top_half = magnitude_no_dc[:center_y, :]
        bottom_half = magnitude_no_dc[center_y:, :]
        
        # Flip and compare
        bottom_flipped = np.flipud(bottom_half)
        min_h = min(top_half.shape[0], bottom_flipped.shape[0])
        
        if min_h > 0:
            symmetry = np.corrcoef(
                top_half[-min_h:].flatten(), 
                bottom_flipped[:min_h].flatten()
            )[0, 1]
            symmetry = np.clip(symmetry, 0, 1)
        else:
            symmetry = 0.5
        
        # Combine metrics
        peak_score = np.clip(num_peaks / 100, 0, 1)
        periodic_score = 0.5 * peak_score + 0.5 * symmetry
        
        return float(periodic_score)
    
    def _analyze_high_frequencies(self, power_spectrum: np.ndarray, 
                                  shape: tuple) -> float:
        """
        Analyze high-frequency components
        AI images often have unnatural high-frequency characteristics
        
        Args:
            power_spectrum: Power spectrum from FFT
            shape: Image shape
            
        Returns:
            High frequency anomaly score (0-1)
        """
        h, w = shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency masks for different regions
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Define frequency bands
        max_dist = np.sqrt(center_y**2 + center_x**2)
        low_freq_mask = dist_from_center < (max_dist * 0.3)
        mid_freq_mask = (dist_from_center >= (max_dist * 0.3)) & \
                       (dist_from_center < (max_dist * 0.7))
        high_freq_mask = dist_from_center >= (max_dist * 0.7)
        
        # Calculate energy in each band
        low_energy = np.sum(power_spectrum[low_freq_mask])
        mid_energy = np.sum(power_spectrum[mid_freq_mask])
        high_energy = np.sum(power_spectrum[high_freq_mask])
        
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        # Compute ratios
        high_ratio = high_energy / total_energy
        mid_ratio = mid_energy / total_energy
        
        # AI images often have unusual high-frequency characteristics
        # Either too high (synthetic noise) or too low (oversmoothing)
        ideal_high_ratio = 0.15  # Typical for natural images
        high_deviation = abs(high_ratio - ideal_high_ratio) / ideal_high_ratio
        
        high_freq_score = np.clip(high_deviation, 0, 1)
        
        return float(high_freq_score)
    
    def _detect_block_artifacts(self, gray_image: np.ndarray) -> float:
        """
        Detect DCT block artifacts (8x8 blocks from compression)
        AI models sometimes introduce unusual block patterns
        
        Args:
            gray_image: Grayscale image (0-255)
            
        Returns:
            Block artifact score (0-1)
        """
        # Compute DCT on 8x8 blocks
        h, w = gray_image.shape
        block_size = 8
        
        # Ensure image dimensions are multiples of block_size
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks == 0 or w_blocks == 0:
            return 0.0
        
        cropped = gray_image[:h_blocks*block_size, :w_blocks*block_size]
        
        # Compute variance of DCT coefficients across blocks
        block_variances = []
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = cropped[i*block_size:(i+1)*block_size, 
                              j*block_size:(j+1)*block_size]
                
                # Compute DCT
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Variance of AC coefficients (exclude DC)
                ac_coeffs = dct_block.flatten()[1:]
                block_variances.append(np.var(ac_coeffs))
        
        # Analyze uniformity of block variances
        # Too uniform = potential AI artifact
        if len(block_variances) > 0:
            overall_var = np.var(block_variances)
            mean_var = np.mean(block_variances)
            
            if mean_var > 0:
                cv = overall_var / (mean_var + 1e-10)
                # Low coefficient of variation indicates uniformity
                uniformity_score = np.clip(1 / (1 + cv), 0, 1)
            else:
                uniformity_score = 0.5
        else:
            uniformity_score = 0.0
        
        return float(uniformity_score)
    
    def _analyze_radial_profile(self, power_spectrum: np.ndarray) -> float:
        """
        Analyze radial frequency profile
        Natural images have characteristic radial falloff (1/f^2)
        
        Args:
            power_spectrum: Power spectrum from FFT
            
        Returns:
            Radial anomaly score (0-1)
        """
        h, w = power_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Compute radial average
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        max_r = min(center_x, center_y)
        radial_profile = []
        
        for radius in range(1, max_r):
            mask = r == radius
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(power_spectrum[mask]))
        
        if len(radial_profile) < 10:
            return 0.0
        
        radial_profile = np.array(radial_profile)
        
        # Natural images follow ~1/f^2 law
        # Compute deviation from this law
        frequencies = np.arange(1, len(radial_profile) + 1)
        expected = 1 / (frequencies ** 2)
        
        # Normalize both
        radial_norm = radial_profile / (np.max(radial_profile) + 1e-10)
        expected_norm = expected / (np.max(expected) + 1e-10)
        
        # Compute deviation
        deviation = np.mean(np.abs(radial_norm - expected_norm))
        anomaly_score = np.clip(deviation, 0, 1)
        
        return float(anomaly_score)
    
    def _generate_explanation(self, fft_score: float, freq_anomaly: float,
                             periodic: float, high_freq: float, 
                             block: float) -> str:
        """
        Generate human-readable explanation of FFT analysis
        
        Args:
            fft_score: Overall FFT score
            freq_anomaly: Frequency anomaly score
            periodic: Periodic artifact score
            high_freq: High frequency score
            block: Block artifact score
            
        Returns:
            Explanation string
        """
        explanations = []
        
        if fft_score > 0.6:
            explanations.append("Significant frequency domain anomalies detected")
        elif fft_score > 0.4:
            explanations.append("Moderate frequency irregularities")
        else:
            explanations.append("Normal frequency distribution")
        
        if freq_anomaly > 0.6:
            explanations.append("unusual frequency patterns")
        
        if periodic > 0.6:
            explanations.append("regular periodic artifacts (typical of generators)")
        
        if high_freq > 0.6:
            explanations.append("abnormal high-frequency components")
        
        if block > 0.6:
            explanations.append("suspicious DCT block patterns")
        
        return "; ".join(explanations) if explanations else "No significant artifacts"
