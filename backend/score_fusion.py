"""
Score Fusion Module - Combines Analysis Scores
Weighted fusion of CNN, FFT, Noise, and Edge scores
"""

import numpy as np
from typing import Dict, Any, Optional


class ScoreFusion:
    """Combines multiple analysis scores with confidence-aware weighting"""
    
    def __init__(self, 
                 cnn_weight: float = 0.6,
                 fft_weight: float = 0.2,
                 noise_weight: float = 0.1,
                 edge_weight: float = 0.1):
        """
        Initialize score fusion with weights
        
        Args:
            cnn_weight: Weight for CNN analysis (default 0.6)
            fft_weight: Weight for FFT analysis (default 0.2)
            noise_weight: Weight for noise analysis (default 0.1)
            edge_weight: Weight for edge detection (default 0.1)
        """
        # Normalize weights to sum to 1.0
        total = cnn_weight + fft_weight + noise_weight + edge_weight
        
        self.cnn_weight = cnn_weight / total
        self.fft_weight = fft_weight / total
        self.noise_weight = noise_weight / total
        self.edge_weight = edge_weight / total
        
        print(f"✅ Score Fusion initialized:")
        print(f"   CNN: {self.cnn_weight:.2f}")
        print(f"   FFT: {self.fft_weight:.2f}")
        print(f"   Noise: {self.noise_weight:.2f}")
        print(f"   Edge: {self.edge_weight:.2f}")
    
    def fuse_scores(self, 
                    cnn_result: Dict[str, Any],
                    fft_result: Dict[str, Any],
                    noise_result: Dict[str, Any],
                    edge_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine analysis scores from multiple modules
        
        Args:
            cnn_result: Result from CNN analysis
            fft_result: Result from FFT analysis
            noise_result: Result from noise analysis
            edge_result: Optional result from edge detection
            
        Returns:
            Dictionary containing:
                - final_score: Weighted combination of all scores (0-1)
                - prediction: "AI-Generated" or "Real"
                - confidence: Overall confidence in prediction
                - individual_scores: All component scores
                - explanation: Detailed human-readable explanation
                - score_breakdown: Contribution of each module
        """
        print("\n🔗 Starting score fusion...")
        
        # Extract individual scores
        cnn_score = cnn_result.get("score", 0.5)
        fft_score = fft_result.get("score", 0.5)
        noise_score = noise_result.get("score", 0.5)
        edge_score = edge_result.get("score", 0.5) if edge_result else 0.5
        
        # Extract confidences (if available)
        cnn_confidence = cnn_result.get("confidence", 0.5)
        
        # Weighted fusion
        if edge_result is None:
            # Redistribute edge weight to other modules proportionally
            total_weight = self.cnn_weight + self.fft_weight + self.noise_weight
            adjusted_cnn = self.cnn_weight / total_weight
            adjusted_fft = self.fft_weight / total_weight
            adjusted_noise = self.noise_weight / total_weight
            
            final_score = (
                adjusted_cnn * cnn_score +
                adjusted_fft * fft_score +
                adjusted_noise * noise_score
            )
            
            weights_used = {
                "cnn": adjusted_cnn,
                "fft": adjusted_fft,
                "noise": adjusted_noise,
                "edge": 0.0
            }
        else:
            final_score = (
                self.cnn_weight * cnn_score +
                self.fft_weight * fft_score +
                self.noise_weight * noise_score +
                self.edge_weight * edge_score
            )
            
            weights_used = {
                "cnn": self.cnn_weight,
                "fft": self.fft_weight,
                "noise": self.noise_weight,
                "edge": self.edge_weight
            }
        
        # Ensure score is in valid range
        final_score = float(np.clip(final_score, 0, 1))
        
        # Determine prediction
        threshold = 0.5
        if final_score > threshold:
            prediction = "AI-Generated"
            certainty = final_score
        else:
            prediction = "Real"
            certainty = 1 - final_score
        
        # Calculate overall confidence
        # Higher when all modules agree, lower when they disagree
        scores_array = np.array([cnn_score, fft_score, noise_score])
        if edge_result:
            scores_array = np.append(scores_array, edge_score)
        
        score_agreement = 1 - np.std(scores_array)  # Low std = high agreement
        overall_confidence = (score_agreement + cnn_confidence) / 2
        overall_confidence = float(np.clip(overall_confidence, 0, 1))
        
        # Generate comprehensive explanation
        explanation = self._generate_explanation(
            final_score, prediction, cnn_result, fft_result, 
            noise_result, edge_result
        )
        
        # Calculate score breakdown (contribution of each module)
        score_breakdown = {
            "cnn_contribution": weights_used["cnn"] * cnn_score,
            "fft_contribution": weights_used["fft"] * fft_score,
            "noise_contribution": weights_used["noise"] * noise_score,
            "edge_contribution": weights_used["edge"] * edge_score
        }
        
        result = {
            "final_score": final_score,
            "prediction": prediction,
            "certainty": certainty,
            "confidence": overall_confidence,
            "individual_scores": {
                "cnn_score": float(cnn_score),
                "fft_score": float(fft_score),
                "noise_score": float(noise_score),
                "edge_score": float(edge_score) if edge_result else None
            },
            "weights_used": weights_used,
            "score_breakdown": score_breakdown,
            "explanation": explanation,
            "detailed_analysis": {
                "cnn": cnn_result.get("explanation", ""),
                "fft": fft_result.get("explanation", ""),
                "noise": noise_result.get("explanation", ""),
                "edge": edge_result.get("explanation", "") if edge_result else ""
            }
        }
        
        print(f"✅ Score Fusion complete:")
        print(f"   Final Score: {final_score:.3f}")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {overall_confidence:.3f}")
        
        return result
    
    def _generate_explanation(self, 
                            final_score: float,
                            prediction: str,
                            cnn_result: Dict[str, Any],
                            fft_result: Dict[str, Any],
                            noise_result: Dict[str, Any],
                            edge_result: Optional[Dict[str, Any]]) -> str:
        """
        Generate comprehensive human-readable explanation
        
        Args:
            final_score: Final combined score
            prediction: Final prediction
            cnn_result: CNN analysis result
            fft_result: FFT analysis result
            noise_result: Noise analysis result
            edge_result: Edge analysis result (optional)
            
        Returns:
            Detailed explanation string
        """
        parts = []
        
        # Overall assessment
        if final_score > 0.8:
            parts.append(f"**{prediction}** with high confidence ({final_score:.1%}).")
        elif final_score > 0.6:
            parts.append(f"**{prediction}** with moderate confidence ({final_score:.1%}).")
        elif final_score > 0.4:
            parts.append(f"Likely **Real** with moderate confidence ({(1-final_score):.1%}).")
        else:
            parts.append(f"**Real** with high confidence ({(1-final_score):.1%}).")
        
        # Add key findings from each module
        findings = []
        
        # CNN findings
        cnn_score = cnn_result.get("score", 0)
        if cnn_score > 0.6:
            findings.append("CNN detected AI-typical patterns")
        elif cnn_score < 0.4:
            findings.append("CNN found natural image characteristics")
        
        # FFT findings
        fft_score = fft_result.get("score", 0)
        if fft_score > 0.6:
            findings.append("FFT found frequency domain anomalies")
        elif fft_score < 0.4:
            findings.append("FFT showed normal frequency distribution")
        
        # Noise findings
        noise_score = noise_result.get("score", 0)
        if noise_score > 0.6:
            findings.append("Noise analysis revealed synthetic patterns")
        elif noise_score < 0.4:
            findings.append("Noise patterns appear natural")
        
        # Edge findings (if available)
        if edge_result:
            edge_score = edge_result.get("score", 0)
            if edge_score > 0.6:
                findings.append("Edge detection found artificial boundaries")
        
        if findings:
            parts.append(" Key findings: " + "; ".join(findings) + ".")
        
        # Add specific suspicious features
        suspicious_features = []
        
        # Check CNN features
        cnn_features = cnn_result.get("features", {})
        if cnn_features.get("suspicious_smoothness"):
            suspicious_features.append("unnatural texture smoothness")
        if cnn_features.get("suspicious_saturation"):
            suspicious_features.append("abnormal color saturation")
        
        # Check FFT features
        if fft_result.get("periodic_artifacts", 0) > 0.6:
            suspicious_features.append("regular periodic artifacts")
        if fft_result.get("high_freq_score", 0) > 0.6:
            suspicious_features.append("unusual high-frequency components")
        
        # Check noise features
        if noise_result.get("variance_score", 0) > 0.6:
            suspicious_features.append("abnormal noise variance")
        if noise_result.get("spatial_consistency", 0) > 0.6:
            suspicious_features.append("spatially inconsistent noise")
        
        if suspicious_features:
            parts.append(" Detected: " + ", ".join(suspicious_features) + ".")
        
        return "".join(parts)
    
    def get_score_weights(self) -> Dict[str, float]:
        """
        Get current fusion weights
        
        Returns:
            Dictionary of weights
        """
        return {
            "cnn_weight": self.cnn_weight,
            "fft_weight": self.fft_weight,
            "noise_weight": self.noise_weight,
            "edge_weight": self.edge_weight
        }
    
    def update_weights(self, cnn: float = None, fft: float = None, 
                      noise: float = None, edge: float = None):
        """
        Update fusion weights (useful for tuning)
        
        Args:
            cnn: New CNN weight
            fft: New FFT weight
            noise: New noise weight
            edge: New edge weight
        """
        if cnn is not None:
            self.cnn_weight = cnn
        if fft is not None:
            self.fft_weight = fft
        if noise is not None:
            self.noise_weight = noise
        if edge is not None:
            self.edge_weight = edge
        
        # Normalize
        total = self.cnn_weight + self.fft_weight + self.noise_weight + self.edge_weight
        self.cnn_weight /= total
        self.fft_weight /= total
        self.noise_weight /= total
        self.edge_weight /= total
        
        print("✅ Updated fusion weights")
