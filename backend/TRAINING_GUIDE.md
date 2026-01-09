# Quick Start Guide - Training the Hybrid AI Detector

## Prerequisites

```bash
# Install additional dependencies
pip install opencv-python scipy scikit-image
```

## 1. Prepare Your Dataset

### Directory Structure
```
my_dataset/
├── real/
│   ├── real_photo_001.jpg
│   ├── real_photo_002.png
│   └── ... (500-5000+ images)
└── ai_generated/
    ├── ai_image_001.jpg
    ├── ai_image_002.png
    └── ... (500-5000+ images)
```

### Dataset Sources

**AI-Generated Images:**
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- DiffusionDB: https://huggingface.co/datasets/poloclub/diffusiondb
- Generate your own using Stable Diffusion/DALL·E/Midjourney

**Real Images:**
- COCO dataset: https://cocodataset.org/
- FFHQ (faces): https://github.com/NVlabs/ffhq-dataset
- Your own photographs

**Important:** Aim for 50/50 balance (equal real and AI images)

## 2. Training Command

### Basic Training (Recommended for First Run)

```bash
cd backend

python train_model.py \
  --data_dir /path/to/my_dataset \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --device cuda
```

### Training on CPU (Slower)

```bash
python train_model.py \
  --data_dir /path/to/my_dataset \
  --epochs 15 \
  --batch_size 16 \
  --device cpu
```

### Advanced Training

```bash
python train_model.py \
  --data_dir /path/to/my_dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_split 0.15 \
  --save_dir my_checkpoints \
  --device cuda
```

## 3. Monitor Training

You'll see output like:

```
Loaded 10000 images: 5000 real, 5000 AI
Training samples: 8500
Validation samples: 1500

======================================================
Starting training for 15 epochs
Device: cuda
======================================================

Epoch 1 [Train]: 100%|████████| 266/266 [02:15<00:00]
Epoch 1 [Val]:   100%|████████| 47/47 [00:23<00:00]

Epoch 1/15
  Train Loss: 0.4521 | Train Acc: 78.34%
  Val Loss:   0.3892 | Val Acc:   82.67%
  ✅ New best model saved! (Val Acc: 82.67%)

...
```

## 4. Output Files

After training, you'll find in `checkpoints/`:

- **best_model.pth** - Model with highest validation accuracy
- **hybrid_detector_calibrated.pth** - Final model with temperature scaling (USE THIS)
- **checkpoint_epoch_5.pth**, **checkpoint_epoch_10.pth**, etc.
- **training_history.json** - Training metrics
- **final_model.pth** - Last epoch model

## 5. Use Trained Model

### Update main.py

```python
# In backend/main.py, update this line:
ai_detector = AIDetectorModel(
    model_path="checkpoints/hybrid_detector_calibrated.pth",
    device='cpu'  # or 'cuda' if available
)
```

### Restart Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 6. Test Your Model

```bash
# Test with a sample image
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@test_image.jpg"
```

Expected response:
```json
{
  "prediction": "AI-Generated",
  "confidence": 87.3,
  "confidence_category": "High",
  "explanation": "High confidence AI-generated detection (87.3%). | Multiple model branches (spatial CNN, frequency analysis, noise patterns) consistently indicate synthetic generation...",
  ...
}
```

## 7. Expected Performance

### Minimum Dataset Size
- **Small:** 1,000 images (500 real + 500 AI) - Quick test, lower accuracy
- **Medium:** 5,000 images (2,500 + 2,500) - Good performance
- **Large:** 10,000+ images (5,000 + 5,000) - Best performance

### Training Time Estimates

| Hardware | Batch Size | Time per Epoch | Total (15 epochs) |
|----------|-----------|----------------|-------------------|
| RTX 3080 | 32 | 2-3 min | 30-45 min |
| RTX 2060 | 32 | 4-5 min | 60-75 min |
| CPU (i7) | 16 | 15-20 min | 4-5 hours |

### Expected Accuracy

After 15-20 epochs on balanced dataset:
- **Validation Accuracy:** 85-95%
- **Stability Score:** 80-90 (robust to compression)

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train_model.py --data_dir /path/to/dataset --batch_size 16
```

### CUDA Not Available
```bash
# Force CPU training
python train_model.py --data_dir /path/to/dataset --device cpu
```

### Low Accuracy (<70%)
- Check dataset balance (should be 50/50)
- Increase training epochs (try 25-30)
- Verify image quality (avoid heavily compressed images)
- Add more diverse data

### Model Not Loading in main.py
```python
# Check the path is correct
import os
print(os.path.exists("checkpoints/hybrid_detector_calibrated.pth"))

# If False, use absolute path
ai_detector = AIDetectorModel(
    model_path="/full/path/to/checkpoints/hybrid_detector_calibrated.pth"
)
```

## Next Steps

1. ✅ Train the model with your dataset
2. ✅ Load trained model in backend
3. ✅ Test with various images (real and AI)
4. ✅ Monitor prediction stability across compression
5. ✅ Update frontend to handle "Uncertain" predictions

**You now have a production-credible AI detection system!**
