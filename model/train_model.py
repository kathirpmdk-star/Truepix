"""
Model Training Script (Reference Implementation)

This script shows how to fine-tune a CNN model for AI-generated image detection.
Use this as a template for training on your own dataset.

For hackathon demo: Pre-trained ImageNet weights are used.
For production: Train on balanced AI/Real dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm


class AIImageDataset(Dataset):
    """
    Custom dataset for AI-generated vs Real images
    
    Expected directory structure:
    dataset/
    ├── train/
    │   ├── real/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── ai_generated/
    │       ├── img1.jpg
    │       └── img2.jpg
    └── val/
        ├── real/
        └── ai_generated/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)
        
        # Load AI-generated images (label 1)
        ai_dir = os.path.join(self.root_dir, 'ai_generated')
        if os.path.exists(ai_dir):
            for img_name in os.listdir(ai_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(ai_dir, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(dataset_path, batch_size=32):
    """Create train and validation data loaders"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AIImageDataset(dataset_path, split='train', transform=train_transform)
    val_dataset = AIImageDataset(dataset_path, split='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=20):
    """Train the AI detection model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../model/weights/efficientnet_aidetector_best.pth')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
    
    return model


def main():
    """Main training pipeline"""
    
    # Configuration
    DATASET_PATH = 'path/to/your/dataset'  # Update this
    BATCH_SIZE = 32
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting training on {DEVICE}")
    
    # Create model
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(DEVICE)
    
    # Create data loaders
    print("📊 Loading dataset...")
    train_loader, val_loader = create_data_loaders(DATASET_PATH, BATCH_SIZE)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Train model
    print("\n🎯 Training model...")
    model = train_model(model, train_loader, val_loader, DEVICE, EPOCHS)
    
    # Save final model
    torch.save(model.state_dict(), '../model/weights/efficientnet_aidetector_final.pth')
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()


"""
DATASET RECOMMENDATIONS:

1. CIFAKE (Kaggle)
   - 120k images (60k real, 60k AI)
   - Mix of CIFAR-10 style images
   - Download: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

2. DiffusionDB (Hugging Face)
   - 2M+ Stable Diffusion images
   - Diverse prompts and styles
   - Download: https://huggingface.co/datasets/poloclub/diffusiondb

3. FFHQ (Real Faces)
   - 70k high-quality face images
   - Real human portraits
   - Download: https://github.com/NVlabs/ffhq-dataset

4. COCO (Real Photos)
   - 330k diverse real images
   - Natural scenes
   - Download: https://cocodataset.org/

TRAINING TIPS:

1. Balance dataset (50% real, 50% AI)
2. Use data augmentation (flips, rotations, color jitter)
3. Train for 20-30 epochs with early stopping
4. Monitor validation accuracy
5. Use learning rate scheduling
6. Save checkpoints regularly
7. Test on unseen AI models (Midjourney, DALL-E 3, etc.)
"""
