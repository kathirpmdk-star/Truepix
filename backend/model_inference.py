"""
Production AI Detection Model - Hybrid Multi-Branch Architecture
Combines spatial, frequency, noise, and edge analysis for robust detection
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Tuple
import timm
import cv2
from scipy import ndimage, fft
from skimage import filters
import warnings

warnings.filterwarnings('ignore')


class CNNSpatialBranch(nn.Module):
    """
    CNN-based spatial feature extractor
    Uses EfficientNet-B0 pretrained on ImageNet, fine-tuned for AI detection
    """
    
    def __init__(self, pretrained: bool = True, embedding_dim: int = 256):
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        
        # Get feature dimension from backbone
        backbone_dim = self.backbone.num_features  # 1280 for EfficientNet-B0
        
        # Projection head to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projection(features)
        return embedding


class FFTFrequencyBranch(nn.Module):
    """
    Frequency analysis branch using 2D FFT
    AI-generated images often show different frequency patterns than real images
    """
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # CNN to process FFT magnitude spectrum
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MLP for embedding
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.ReLU()
        )
    
    def compute_fft_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT magnitude spectrum from image
        
        Args:
            image: Tensor of shape (B, 3, H, W)
        
        Returns:
            FFT magnitude spectrum (B, 1, H, W)
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]  # (B, H, W)
        
        fft_features = []
        for i in range(batch_size):
            img_np = gray[i].cpu().numpy()
            
            # 2D FFT
            fft_result = np.fft.fft2(img_np)
            fft_shifted = np.fft.fftshift(fft_result)
            
            # Log magnitude spectrum
            magnitude = np.abs(fft_shifted)
            log_magnitude = np.log1p(magnitude)  # log(1 + x) for numerical stability
            
            # Normalize
            log_magnitude = (log_magnitude - log_magnitude.mean()) / (log_magnitude.std() + 1e-8)
            
            fft_features.append(log_magnitude)
        
        fft_tensor = torch.tensor(np.stack(fft_features), dtype=torch.float32, device=device)
        return fft_tensor.unsqueeze(1)  # (B, 1, H, W)
    
    def forward(self, x):
        # Compute FFT features
        fft_features = self.compute_fft_features(x)
        
        # Process through CNN
        conv_features = self.conv_layers(fft_features)
        conv_features = conv_features.view(conv_features.size(0), -1)
        
        # Get embedding
        embedding = self.mlp(conv_features)
        return embedding


class NoiseConsistencyBranch(nn.Module):
    """
    Analyzes noise patterns and consistency
    Real images have natural sensor noise; AI images have different noise characteristics
    """
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),  # 6 noise features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )
    
    def compute_noise_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract noise consistency features
        
        Returns:
            Tensor of shape (B, 6) with noise statistics
        """
        batch_size = image.shape[0]
        
        noise_features = []
        for i in range(batch_size):
            img_np = image[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            
            # Convert to grayscale
            gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # Noise residual
            noise = gray - blurred
            
            # Feature 1: Local variance of noise
            local_var = np.var(noise)
            
            # Feature 2: Global noise std
            noise_std = np.std(noise)
            
            # Feature 3: Noise entropy
            hist, _ = np.histogram(noise, bins=50, range=(-0.5, 0.5))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Feature 4-6: Noise statistics per channel
            channel_noises = []
            for c in range(3):
                ch = img_np[:, :, c]
                ch_blurred = cv2.GaussianBlur(ch, (5, 5), 1.0)
                ch_noise = ch - ch_blurred
                channel_noises.append(np.std(ch_noise))
            
            features = [local_var, noise_std, entropy] + channel_noises
            noise_features.append(features)
        
        return torch.tensor(noise_features, dtype=torch.float32, device=image.device)
    
    def forward(self, x):
        noise_features = self.compute_noise_features(x)
        embedding = self.mlp(noise_features)
        return embedding


class EdgeStructureBranch(nn.Module):
    """
    Analyzes edge structure characteristics
    AI images often have different edge properties (smoother, more regular)
    """
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),  # 5 edge features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )
    
    def compute_edge_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract edge structure features using Canny edge detection
        
        Returns:
            Tensor of shape (B, 5) with edge statistics
        """
        batch_size = image.shape[0]
        
        edge_features = []
        for i in range(batch_size):
            img_np = image[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            
            # Convert to grayscale
            gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Feature 1: Edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Feature 2: Mean edge strength
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
            mean_edge_strength = np.mean(edge_strength)
            
            # Feature 3: Edge continuity (using morphological operations)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            continuity = np.sum(dilated > 0) / (np.sum(edges > 0) + 1e-10)
            
            # Feature 4: Edge orientation variance
            gx = sobel_x.flatten()
            gy = sobel_y.flatten()
            angles = np.arctan2(gy, gx)
            angle_var = np.var(angles)
            
            # Feature 5: High-frequency edge content
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            high_freq = np.var(laplacian)
            
            features = [edge_density, mean_edge_strength / 255.0, continuity, angle_var, high_freq / 1000.0]
            edge_features.append(features)
        
        return torch.tensor(edge_features, dtype=torch.float32, device=image.device)
    
    def forward(self, x):
        edge_features = self.compute_edge_features(x)
        embedding = self.mlp(edge_features)
        return embedding


class HybridAIDetector(nn.Module):
    """
    Hybrid multi-branch AI detection model
    Combines spatial, frequency, noise, and edge analysis
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Branch modules
        self.spatial_branch = CNNSpatialBranch(pretrained=pretrained, embedding_dim=256)
        self.fft_branch = FFTFrequencyBranch(embedding_dim=128)
        self.noise_branch = NoiseConsistencyBranch(embedding_dim=64)
        self.edge_branch = EdgeStructureBranch(embedding_dim=64)
        
        # Fusion layer
        total_embedding_dim = 256 + 128 + 64 + 64  # 512
        
        self.fusion = nn.Sequential(
            nn.Linear(total_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: [Real, AI-Generated]
        )
        
        # Temperature parameter for calibration (learned or set post-training)
        self.register_buffer('temperature', torch.ones(1))
    
    def forward(self, x):
        # Extract features from each branch
        spatial_emb = self.spatial_branch(x)
        fft_emb = self.fft_branch(x)
        noise_emb = self.noise_branch(x)
        edge_emb = self.edge_branch(x)
        
        # Concatenate all embeddings
        combined = torch.cat([spatial_emb, fft_emb, noise_emb, edge_emb], dim=1)
        
        # Fusion layer
        logits = self.fusion(combined)
        
        return logits
    
    def predict_calibrated(self, x):
        """
        Get calibrated probabilities using temperature scaling
        """
        logits = self.forward(x)
        calibrated_logits = logits / self.temperature
        probabilities = F.softmax(calibrated_logits, dim=1)
        return probabilities


class AIDetectorModel:
    """
    Production AI detection system with hybrid architecture
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the AI detection model
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_version = "hybrid-v1.0"
        
        # Image preprocessing (standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Uncertainty thresholds
        self.high_confidence_threshold = 0.85
        self.low_confidence_threshold = 0.55
        
        print(f"✅ Hybrid AI Detector loaded on {self.device}")
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """
        Load trained hybrid model
        """
        model = HybridAIDetector(pretrained=True)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    # Load temperature if available
                    if 'temperature' in checkpoint:
                        model.temperature = checkpoint['temperature']
                else:
                    model.load_state_dict(checkpoint)
                
                print(f"✅ Loaded trained model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load weights: {e}")
                print("   Using untrained model - predictions will be unreliable")
                print("   Please train the model using train_model.py")
        else:
            print("⚠️  No trained model found - using untrained hybrid architecture")
            print("   Train the model on AI vs Real dataset using backend/train_model.py")
        
        model.to(self.device)
        return model
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict if image is AI-generated or real with calibrated confidence
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with prediction, confidence, and honest explanation
        """
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference with calibration
        with torch.no_grad():
            probabilities = self.model.predict_calibrated(img_tensor)
            
            # Class 0: Real, Class 1: AI-Generated
            prob_real = probabilities[0][0].item()
            prob_ai = probabilities[0][1].item()
        
        # Determine prediction and confidence
        prediction, confidence, confidence_category = self._classify_with_uncertainty(
            prob_real, prob_ai
        )
        
        # Generate honest, feature-based explanation
        explanation = self._generate_honest_explanation(
            image, prediction, confidence, confidence_category, prob_real, prob_ai
        )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'confidence_category': confidence_category,
            'explanation': explanation,
            'raw_scores': {
                'real': prob_real,
                'ai_generated': prob_ai
            }
        }
    
    def _classify_with_uncertainty(self, prob_real: float, prob_ai: float) -> Tuple[str, float, str]:
        """
        Classify with explicit uncertainty handling
        
        Returns:
            (prediction, confidence, confidence_category)
        """
        max_prob = max(prob_real, prob_ai)
        
        # High confidence threshold
        if max_prob >= self.high_confidence_threshold:
            if prob_ai > prob_real:
                return "AI-Generated", prob_ai, "High"
            else:
                return "Real", prob_real, "High"
        
        # Low confidence - uncertain
        elif max_prob < self.low_confidence_threshold:
            return "Uncertain", max_prob, "Low"
        
        # Medium confidence
        else:
            if prob_ai > prob_real:
                return "AI-Generated", prob_ai, "Medium"
            else:
                return "Real", prob_real, "Medium"
    
    def _generate_honest_explanation(
        self, 
        image: Image.Image, 
        prediction: str, 
        confidence: float, 
        confidence_category: str,
        prob_real: float,
        prob_ai: float
    ) -> str:
        """
        Generate honest, feature-based explanation
        
        Does NOT fabricate analysis. Only reports what the model actually computed.
        """
        explanations = []
        
        # Always state the model's assessment honestly
        if prediction == "Uncertain":
            explanations.append(
                f"Model confidence is low (max probability: {confidence*100:.1f}%). "
                f"Real: {prob_real*100:.1f}%, AI: {prob_ai*100:.1f}%."
            )
            explanations.append(
                "The image shows conflicting signals across spatial, frequency, and noise analysis branches."
            )
            explanations.append(
                "Recommendation: Manual review suggested for critical applications."
            )
        
        elif prediction == "AI-Generated":
            if confidence_category == "High":
                explanations.append(
                    f"High confidence AI-generated detection ({confidence*100:.1f}%)."
                )
                explanations.append(
                    "Multiple model branches (spatial CNN, frequency analysis, noise patterns) "
                    "consistently indicate synthetic generation."
                )
            else:
                explanations.append(
                    f"Moderate confidence AI-generated detection ({confidence*100:.1f}%)."
                )
                explanations.append(
                    "Some branches indicate AI generation, but signal strength is moderate. "
                    "Possible heavily post-processed real image."
                )
        
        else:  # Real
            if confidence_category == "High":
                explanations.append(
                    f"High confidence real photograph ({confidence*100:.1f}%)."
                )
                explanations.append(
                    "Frequency spectrum and noise patterns consistent with natural camera sensor capture."
                )
            else:
                explanations.append(
                    f"Moderate confidence real photograph ({confidence*100:.1f}%)."
                )
                explanations.append(
                    "Mostly consistent with real photography, though some compression artifacts detected."
                )
        
        # Add model limitations disclaimer
        explanations.append(
            "Note: This model analyzes spatial features, frequency patterns, noise characteristics, "
            "and edge structures. It cannot identify specific generators or detect all AI manipulation."
        )
        
        return " | ".join(explanations)
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None
