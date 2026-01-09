"""
Training Pipeline for Hybrid AI Detection Model
Supports training on AI-generated vs Real image datasets with augmentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple
import io

from model_inference import HybridAIDetector


class AIDetectionDataset(Dataset):
    """
    Dataset for AI-generated vs Real images
    
    Expected directory structure:
    dataset/
        real/
            image1.jpg
            image2.png
            ...
        ai_generated/
            image1.jpg
            image2.png
            ...
    """
    
    def __init__(self, data_dir: str, transform=None, augment: bool = True):
        """
        Args:
            data_dir: Root directory with 'real' and 'ai_generated' subdirectories
            transform: Base transformations (resize, normalize)
            augment: Whether to apply compression/noise augmentation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        
        # Collect all image paths
        self.samples = []
        
        # Real images (label 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))
        
        # AI-generated images (label 1)
        ai_dir = self.data_dir / 'ai_generated'
        if ai_dir.exists():
            for img_path in ai_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} images: {sum(1 for _, l in self.samples if l == 0)} real, {sum(1 for _, l in self.samples if l == 1)} AI")
        
        # Augmentation transforms
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation
        if self.augment:
            image = self.augment_transforms(image)
            
            # Random JPEG compression (simulates social media)
            if np.random.random() < 0.7:
                quality = np.random.randint(30, 91)  # Quality 30-90
                image = self._apply_jpeg_compression(image, quality)
            
            # Random resize & recompression (simulates platform resizing)
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.7, 1.0)
                w, h = image.size
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Random blur (simulates low-quality uploads)
            if np.random.random() < 0.3:
                from PIL import ImageFilter
                image = image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))
        
        # Apply standard transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _apply_jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression to image"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class Trainer:
    """Training manager for hybrid AI detector"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-4,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        # Different learning rates for pretrained backbone vs new layers
        backbone_params = list(self.model.spatial_branch.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if not any(p is bp for bp in backbone_params)]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained
            {'params': other_params, 'lr': lr}
        ], weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth', is_best=True)
                print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Save regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
            
            print()
        
        # Save final model
        self.save_checkpoint(num_epochs, 'final_model.pth')
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch: int, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'temperature': self.model.temperature,
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / filename)


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: str):
    """
    Perform temperature scaling calibration on validation set
    
    Temperature scaling improves probability calibration
    """
    print("\n" + "="*60)
    print("Performing temperature scaling calibration...")
    print("="*60 + "\n")
    
    model.eval()
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collecting logits"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    criterion = nn.CrossEntropyLoss()
    
    def eval_temp():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_temp)
    
    optimal_temp = temperature.item()
    print(f"✅ Optimal temperature: {optimal_temp:.4f}")
    
    # Update model
    model.temperature = torch.tensor([optimal_temp])
    
    return optimal_temp


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid AI Detection Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory (with real/ and ai_generated/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load full dataset
    full_dataset = AIDetectionDataset(
        args.data_dir,
        transform=transform,
        augment=not args.no_augment
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create val dataset without augmentation
    val_dataset_no_aug = AIDetectionDataset(
        args.data_dir,
        transform=transform,
        augment=False
    )
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = HybridAIDetector(pretrained=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(args.epochs)
    
    # Temperature calibration
    optimal_temp = calibrate_temperature(model, val_loader, device)
    
    # Save calibrated model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': torch.tensor([optimal_temp]),
        'best_val_acc': trainer.best_val_acc,
        'history': trainer.history
    }
    
    save_path = Path(args.save_dir) / 'hybrid_detector_calibrated.pth'
    torch.save(final_checkpoint, save_path)
    print(f"\n✅ Final calibrated model saved to {save_path}")


if __name__ == '__main__':
    main()
