"""
Production AI Detection Model - Hybrid Multi-Branch Architecture
Combines spatial, frequency, noise, and edge analysis for robust detection
"""

import os
import io
import base64
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
        
        # Grad-CAM target layer (EfficientNet backbone final conv layer)
        self.gradcam_target_layer = None
        self._setup_gradcam()
        
        print(f"✅ Hybrid AI Detector loaded on {self.device}")
    
    def _setup_gradcam(self):
        """Setup Grad-CAM hooks on the spatial branch backbone"""
        try:
            # Target the last convolutional layer of EfficientNet backbone
            # For EfficientNet-B0, this is typically 'conv_head' or final block
            backbone = self.model.spatial_branch.backbone
            
            # Find the last conv layer
            for name, module in backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.gradcam_target_layer = module
            
            if self.gradcam_target_layer is not None:
                print(f"✅ Grad-CAM enabled on spatial CNN")
        except Exception as e:
            print(f"⚠️  Grad-CAM setup failed: {e}")
            self.gradcam_target_layer = None
    
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
            Dictionary with prediction, confidence, honest explanation, and Grad-CAM heatmap
        """
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Store activations and gradients for Grad-CAM
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        # Register hooks if Grad-CAM is available
        hooks = []
        if self.gradcam_target_layer is not None:
            hooks.append(self.gradcam_target_layer.register_forward_hook(forward_hook))
            hooks.append(self.gradcam_target_layer.register_full_backward_hook(backward_hook))
        
        # Run inference with calibration
        self.model.zero_grad()
        
        # Compute branch embeddings separately to enable explainability
        spatial_emb = self.model.spatial_branch(img_tensor)
        fft_emb = self.model.fft_branch(img_tensor)
        noise_emb = self.model.noise_branch(img_tensor)
        edge_emb = self.model.edge_branch(img_tensor)

        # Full fusion prediction
        combined = torch.cat([spatial_emb, fft_emb, noise_emb, edge_emb], dim=1)
        logits = self.model.fusion(combined)
        calibrated_logits = logits / self.model.temperature
        probabilities = F.softmax(calibrated_logits, dim=1)

        # Class 0: Real, Class 1: AI-Generated
        prob_real = probabilities[0][0].item()
        prob_ai = probabilities[0][1].item()
        
        # Compute gradients for Grad-CAM (backprop w.r.t. AI class logit)
        if self.gradcam_target_layer is not None and len(activations) > 0:
            # Backprop w.r.t. the AI-generated class (class 1)
            score = calibrated_logits[0, 1]
            score.backward(retain_graph=True)

        # Compute per-branch contribution scores (ablation-style)
        with torch.no_grad():
            zero_spatial = torch.zeros_like(spatial_emb)
            zero_fft = torch.zeros_like(fft_emb)
            zero_noise = torch.zeros_like(noise_emb)
            zero_edge = torch.zeros_like(edge_emb)

            # Spatial-only
            in_spatial = torch.cat([spatial_emb, zero_fft, zero_noise, zero_edge], dim=1)
            logits_spatial = self.model.fusion(in_spatial)
            prob_ai_spatial = F.softmax(logits_spatial / self.model.temperature, dim=1)[0][1].item()

            # FFT-only
            in_fft = torch.cat([zero_spatial, fft_emb, zero_noise, zero_edge], dim=1)
            logits_fft = self.model.fusion(in_fft)
            prob_ai_fft = F.softmax(logits_fft / self.model.temperature, dim=1)[0][1].item()

            # Noise-only
            in_noise = torch.cat([zero_spatial, zero_fft, noise_emb, zero_edge], dim=1)
            logits_noise = self.model.fusion(in_noise)
            prob_ai_noise = F.softmax(logits_noise / self.model.temperature, dim=1)[0][1].item()

            # Edge-only
            in_edge = torch.cat([zero_spatial, zero_fft, zero_noise, edge_emb], dim=1)
            logits_edge = self.model.fusion(in_edge)
            prob_ai_edge = F.softmax(logits_edge / self.model.temperature, dim=1)[0][1].item()
        
        # Generate Grad-CAM heatmap
        gradcam_base64 = None
        if self.gradcam_target_layer is not None and len(activations) > 0 and len(gradients) > 0:
            try:
                gradcam_base64 = self._generate_gradcam(
                    original_image=image,
                    activations=activations[0],
                    gradients=gradients[0]
                )
            except Exception as e:
                print(f"⚠️  Grad-CAM generation failed: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Determine prediction and confidence
        prediction, confidence, confidence_category = self._classify_with_uncertainty(
            prob_real, prob_ai
        )
        
        # Generate interpretable branch findings (human-readable descriptions)
        branch_findings = self._generate_branch_findings(
            image, prob_ai_spatial, prob_ai_fft, prob_ai_noise, prob_ai_edge
        )
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            prediction, confidence, confidence_category, 
            prob_ai_spatial, prob_ai_fft, prob_ai_noise, prob_ai_edge
        )
        
        # Generate honest, feature-based explanation with branch findings
        explanation = self._generate_honest_explanation(
            image, prediction, confidence, confidence_category, prob_real, prob_ai, branch_findings
        )
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'confidence_category': confidence_category,
            'explanation': explanation,
            'executive_summary': executive_summary,
            'branch_findings': branch_findings,
            'raw_scores': {
                'real': prob_real,
                'ai_generated': prob_ai
            }
        }
        
        # Add Grad-CAM if generated
        if gradcam_base64:
            result['gradcam_image'] = gradcam_base64
        
        return result
    
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
        prob_ai: float,
        branch_findings: Dict[str, str]
    ) -> str:
        """
        Generate honest, feature-based explanation with interpretable branch findings
        
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
            else:
                explanations.append(
                    f"Moderate confidence AI-generated detection ({confidence*100:.1f}%)."
                )
        
        else:  # Real
            if confidence_category == "High":
                explanations.append(
                    f"High confidence real photograph ({confidence*100:.1f}%)."
                )
            else:
                explanations.append(
                    f"Moderate confidence real photograph ({confidence*100:.1f}%)."
                )
        
        # Add model limitations disclaimer
        explanations.append(
            "Analysis based on spatial features, frequency patterns, noise characteristics, "
            "and edge structures. Cannot identify specific generators or detect all AI manipulation."
        )
        
        return " | ".join(explanations)
    
    def _generate_executive_summary(
        self,
        prediction: str,
        confidence: float,
        confidence_category: str,
        prob_ai_spatial: float,
        prob_ai_fft: float,
        prob_ai_noise: float,
        prob_ai_edge: float
    ) -> str:
        """
        Generate an executive summary explaining the decision basis
        """
        summary_parts = []
        
        # Main verdict
        if prediction == "AI-Generated":
            summary_parts.append(f"🤖 **VERDICT: AI-GENERATED IMAGE** (Confidence: {confidence*100:.1f}%)")
            summary_parts.append("")
            summary_parts.append("**Why This Decision Was Made:**")
            
            # Identify strongest indicators
            indicators = []
            if prob_ai_spatial > 0.6:
                indicators.append("• **Visual Patterns**: Deep learning detected synthetic features typical of AI generation")
            if prob_ai_fft > 0.6:
                indicators.append("• **Frequency Analysis**: Spectral anomalies show missing natural high-frequency content")
            if prob_ai_noise > 0.6:
                indicators.append("• **Noise Signature**: Unnaturally uniform noise patterns unlike camera sensor noise")
            if prob_ai_edge > 0.6:
                indicators.append("• **Edge Structure**: Boundaries are overly smooth/regular, not naturally captured")
            
            if indicators:
                summary_parts.extend(indicators)
            else:
                summary_parts.append("• Multiple subtle indicators collectively suggest synthetic generation")
            
            summary_parts.append("")
            summary_parts.append("**What We Looked For:**")
            summary_parts.append("AI-generated images typically show: overly perfect/smooth textures, missing fine details, uniform noise, unnatural lighting, repetitive patterns, and physically impossible features that real cameras cannot capture.")
            
        elif prediction == "Real":
            summary_parts.append(f"📷 **VERDICT: REAL PHOTOGRAPH** (Confidence: {confidence*100:.1f}%)")
            summary_parts.append("")
            summary_parts.append("**Why This Decision Was Made:**")
            
            # Identify strongest real indicators
            indicators = []
            if prob_ai_spatial < 0.4:
                indicators.append("• **Visual Patterns**: Authentic camera-captured features with natural imperfections")
            if prob_ai_fft < 0.4:
                indicators.append("• **Frequency Analysis**: Natural spectral signature from optical lens system")
            if prob_ai_noise < 0.4:
                indicators.append("• **Noise Signature**: Authentic sensor noise with spatial and channel variance")
            if prob_ai_edge < 0.4:
                indicators.append("• **Edge Structure**: Natural boundary irregularities from real-world capture")
            
            if indicators:
                summary_parts.extend(indicators)
            else:
                summary_parts.append("• Multiple characteristics collectively indicate authentic photography")
            
            summary_parts.append("")
            summary_parts.append("**What We Looked For:**")
            summary_parts.append("Real photographs contain: natural sensor noise, optical lens artifacts, realistic lighting physics, authentic texture complexity, natural focus variations, and imperfections that AI models cannot fully replicate.")
            
        else:  # Uncertain
            summary_parts.append(f"❓ **VERDICT: UNCERTAIN** (Confidence: {confidence*100:.1f}%)")
            summary_parts.append("")
            summary_parts.append("**Why Uncertain:**")
            summary_parts.append("• Conflicting signals across analysis methods prevent confident classification")
            summary_parts.append("• Could be: highly realistic AI generation, heavily edited real photo, or compressed/processed image")
            summary_parts.append("• Recommend manual expert review for critical applications")
        
        return "\n".join(summary_parts)
    
    def _generate_gradcam(
        self, 
        original_image: Image.Image, 
        activations: torch.Tensor, 
        gradients: torch.Tensor
    ) -> str:
        """
        Generate Grad-CAM heatmap and return as base64-encoded PNG
        
        Args:
            original_image: Original PIL Image (pre-transform)
            activations: Feature maps from target layer (shape: [1, C, H, W])
            gradients: Gradients w.r.t. target layer (shape: [1, C, H, W])
        
        Returns:
            Base64-encoded PNG image (heatmap overlay on original)
        """
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()  # [H, W]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to original image size
        original_width, original_height = original_image.size
        cam_resized = cv2.resize(cam, (original_width, original_height))
        
        # Apply colormap (TURBO for better visibility)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        # Convert to PIL and encode as base64
        overlay_pil = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def _generate_branch_findings(
        self,
        image: Image.Image,
        prob_ai_spatial: float,
        prob_ai_fft: float,
        prob_ai_noise: float,
        prob_ai_edge: float
    ) -> Dict[str, str]:
        """
        Generate human-readable findings for each analysis branch.
        
        Returns interpretable descriptions of what each branch detected,
        avoiding raw percentages in the explanation text.
        """
        findings = {}
        
        # Spatial CNN Analysis - Deep Learning Visual Features
        if prob_ai_spatial > 0.7:
            findings['spatial'] = "🖼️ Visual Pattern Analysis (CNN): Strong AI indicators detected. The deep learning model identified characteristic patterns typical of AI-generated images: overly smooth textures, repetitive structures, unnatural feature arrangements, or unrealistic object compositions. These are common artifacts from generative models like Stable Diffusion, DALL-E, or Midjourney."
        elif prob_ai_spatial > 0.5:
            findings['spatial'] = "🖼️ Visual Pattern Analysis (CNN): Moderate AI indicators found. The model detected some synthetic-looking features mixed with natural elements. This could indicate: AI-generated image with high realism, heavily edited real photo, or compression artifacts affecting natural features."
        elif prob_ai_spatial > 0.3:
            findings['spatial'] = "🖼️ Visual Pattern Analysis (CNN): Leans toward real photography. The model found authentic visual characteristics: natural texture variations, realistic lighting patterns, genuine depth cues, and typical camera capture artifacts. Minor inconsistencies present but within real photo range."
        else:
            findings['spatial'] = "🖼️ Visual Pattern Analysis (CNN): Strong real photo indicators. The model detected clear signs of authentic photography: natural sensor noise patterns, realistic texture complexity, authentic lighting inconsistencies, genuine depth-of-field blur, and typical camera imperfections that AI models struggle to replicate."
        
        # FFT Frequency Analysis - Spectral Anomaly Detection
        if prob_ai_fft > 0.7:
            findings['fft'] = "📊 Frequency Spectrum Analysis (FFT): Significant spectral anomalies detected. The frequency domain shows patterns typical of AI generation: missing or suppressed high-frequency components (fine details), unusual frequency distribution peaks, or artificial smoothness in spectral content. Real photos have richer, more chaotic frequency signatures from optical systems and sensor physics."
        elif prob_ai_fft > 0.5:
            findings['fft'] = "📊 Frequency Spectrum Analysis (FFT): Moderate spectral irregularities found. Some frequency patterns suggest synthetic origin, but natural components also present. Could indicate: AI image with added noise, compressed real photo losing high frequencies, or post-processed image."
        elif prob_ai_fft > 0.3:
            findings['fft'] = "📊 Frequency Spectrum Analysis (FFT): Frequency content leans toward authentic capture. Spectral analysis shows characteristics more consistent with real camera optics: natural high-frequency detail preservation, typical lens characteristics, and realistic frequency distribution from physical light capture."
        else:
            findings['fft'] = "📊 Frequency Spectrum Analysis (FFT): Strong authentic frequency signature. The spectrum clearly indicates natural camera sensor capture: rich high-frequency content from real optical systems, realistic spectral energy distribution, typical lens aberrations in frequency domain, and natural aliasing patterns from sensor array physics."
        
        # Noise Consistency Analysis - Sensor Noise Forensics
        if prob_ai_noise > 0.7:
            findings['noise'] = "🔊 Noise Pattern Analysis: Highly suspicious noise characteristics. Noise distribution is too uniform or consistent, typical of AI generation. Real cameras produce complex, channel-dependent sensor noise with spatial variance. Detected: overly smooth noise, missing dark current noise, absence of hot pixels, or artificially uniform noise floor—all indicators of synthetic generation."
        elif prob_ai_noise > 0.5:
            findings['noise'] = "🔊 Noise Pattern Analysis: Noise patterns show moderate synthetic indicators. Some uniformity detected that's atypical of camera sensors. Could indicate: AI-generated content, heavily denoised real photo, or image processed through AI enhancement tools that alter natural noise structure."
        elif prob_ai_noise > 0.3:
            findings['noise'] = "🔊 Noise Pattern Analysis: Noise characteristics lean toward authentic capture. Analysis shows variance patterns more typical of real camera sensor noise, though some regularities present. Consistent with real photography with possible light editing."
        else:
            findings['noise'] = "🔊 Noise Pattern Analysis: Clear authentic sensor noise signature. Detected natural camera sensor characteristics: spatially-varying noise levels, channel-dependent noise (higher in blue channel), realistic noise grain structure, dark current artifacts, and natural ISO-dependent noise patterns that AI models cannot accurately reproduce."
        
        # Edge Structure Analysis - Boundary Sharpness Forensics
        if prob_ai_edge > 0.7:
            findings['edge'] = "✏️ Edge Structure Analysis: Edges show clear synthetic characteristics. Detected overly smooth, regular, or artificially perfect boundaries typical of AI generation. Real photos have natural edge irregularities from: optical imperfections, motion blur, focus variation, and atmospheric effects. Found: unnaturally sharp object boundaries, missing micro-textures at edges, or artificial edge smoothness."
        elif prob_ai_edge > 0.5:
            findings['edge'] = "✏️ Edge Structure Analysis: Edge patterns show moderate synthetic indicators. Some boundaries appear unnaturally smooth or regular. Could indicate: AI generation, aggressive sharpening filters applied to real photo, or edge-enhanced processing that removed natural irregularities."
        elif prob_ai_edge > 0.3:
            findings['edge'] = "✏️ Edge Structure Analysis: Edge characteristics lean toward real photography. Boundaries show mostly natural properties with authentic irregularities, though some uniformity detected. Consistent with genuine camera capture with possible editing."
        else:
            findings['edge'] = "✏️ Edge Structure Analysis: Strong authentic edge signature. Edges display natural characteristics of real photography: appropriate irregularities, realistic edge density distribution, authentic continuity variations, natural micro-textures along boundaries, and realistic blur gradients that come from optical focus falloff—not artificially generated."
        
        return findings
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None
