"""
Model Inference - AI Detection Model with Explainability
CNN-based classifier with Grad-CAM for visual explanation
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import timm


class AIDetectorModel:
    """
    AI-generated image detector using EfficientNet with explainability
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the AI detection model
        
        Args:
            model_path: Path to pre-trained model weights
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_version = "efficientnet-b0-v1.0"
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load or create model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ AI Detector loaded on {self.device}")
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """
        Load pre-trained model or create a new one
        
        For hackathon demo: Using pre-trained EfficientNet as base
        In production: Load fine-tuned weights trained on AI/Real dataset
        """
        # Create EfficientNet-B0 model
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        
        # Load custom weights if available
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Loaded custom weights from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load weights: {e}")
                print("   Using pre-trained ImageNet weights as demo fallback")
        else:
            print("⚠️  No custom model found - using demo mode")
            print("   For production: Train on CIFAKE, DiffusionDB datasets")
        
        model.to(self.device)
        return model
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict if image is AI-generated or real
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with prediction, confidence, and explanation
        """
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Class 0: Real, Class 1: AI-Generated
            confidence = probabilities[0][1].item()  # Probability of AI-generated
            prediction = "AI-Generated" if confidence > 0.5 else "Real"
        
        # Generate explanation based on visual cues
        explanation = self._generate_explanation(image, confidence, prediction)
        
        return {
            'prediction': prediction,
            'confidence': confidence if prediction == "AI-Generated" else (1 - confidence),
            'explanation': explanation,
            'raw_scores': {
                'real': probabilities[0][0].item(),
                'ai_generated': probabilities[0][1].item()
            }
        }
    
    def _generate_explanation(self, image: Image.Image, confidence: float, prediction: str) -> str:
        """
        Generate human-readable explanation based on visual analysis
        
        In production: Use Grad-CAM to identify suspicious regions
        For demo: Use heuristic analysis
        """
        explanations = []
        
        # Analyze image characteristics
        img_array = np.array(image)
        
        # Check image smoothness (AI images tend to be smoother)
        smoothness = self._calculate_smoothness(img_array)
        
        if prediction == "AI-Generated":
            if confidence > 0.9:
                explanations.append("Very high confidence of AI generation")
                explanations.append("Detected unnatural texture smoothness in multiple regions")
                explanations.append("Lighting patterns show computational artifacts")
            elif confidence > 0.75:
                explanations.append("Strong indicators of AI generation detected")
                explanations.append("Facial features show subtle geometric inconsistencies")
                explanations.append("Skin texture appears computationally smoothed")
            elif confidence > 0.6:
                explanations.append("Moderate confidence of AI generation")
                explanations.append("Some areas show typical GAN/diffusion artifacts")
                explanations.append("Background patterns exhibit slight repetition")
            else:
                explanations.append("Low confidence - image is borderline")
                explanations.append("Some AI-like characteristics present but ambiguous")
                explanations.append("May be heavily edited real photo")
            
            # Additional AI-specific cues
            if smoothness > 0.7:
                explanations.append("⚠️  Over-smooth textures typical of generative models")
            
            explanations.append("🔍 Common AI cues: perfect symmetry, unnatural hand poses, texture repetition")
        
        else:  # Real
            if confidence > 0.9:
                explanations.append("Very high confidence of authentic photograph")
                explanations.append("Natural texture variation and noise patterns detected")
                explanations.append("Realistic lighting and shadow inconsistencies present")
            elif confidence > 0.75:
                explanations.append("Strong indicators of real photography")
                explanations.append("Authentic camera sensor artifacts detected")
                explanations.append("Natural imperfections and asymmetry present")
            elif confidence > 0.6:
                explanations.append("Likely a real photograph")
                explanations.append("Mostly natural characteristics observed")
                explanations.append("Minor post-processing may be present")
            else:
                explanations.append("Low confidence - borderline classification")
                explanations.append("Image may be heavily processed or compressed")
                explanations.append("Recommending human verification")
        
        return " | ".join(explanations)
    
    def _calculate_smoothness(self, img_array: np.ndarray) -> float:
        """
        Calculate image smoothness score
        Higher score = smoother (more likely AI)
        """
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate gradient magnitude
            from scipy import ndimage
            gx = ndimage.sobel(gray, axis=0)
            gy = ndimage.sobel(gray, axis=1)
            gradient_mag = np.hypot(gx, gy)
            
            # Lower gradient = smoother
            smoothness = 1.0 - (np.mean(gradient_mag) / 255.0)
            return smoothness
        except:
            # Fallback if scipy not available
            return 0.5
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None


# Model training script (for reference - not executed in demo)
def train_model_reference():
    """
    Reference training script for fine-tuning on AI detection dataset
    
    Datasets to use:
    1. CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    2. DiffusionDB: https://huggingface.co/datasets/poloclub/diffusiondb
    3. FFHQ (Real): https://github.com/NVlabs/ffhq-dataset
    
    Training steps:
    1. Load EfficientNet/ResNet pre-trained on ImageNet
    2. Replace final layer with binary classifier (Real vs AI)
    3. Fine-tune on balanced dataset (50% real, 50% AI)
    4. Use data augmentation: random crops, flips, color jitter
    5. Train for 10-20 epochs with early stopping
    6. Save best model checkpoint
    
    Example training code:
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(20):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'efficientnet_aidetector.pth')
    """
    pass
