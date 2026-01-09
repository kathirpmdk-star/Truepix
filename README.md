# 🎯 TruePix - AI Image Detection Platform

<div align="center">

![TruePix Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=TruePix+-+AI+or+Real%3F)

**Detect AI-generated images with confidence scores and detailed explanations**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Platform Simulation](#platform-simulation)
- [Demo](#demo)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## 🎯 Overview

**TruePix** is a full-stack web application designed to detect whether an uploaded image is **AI-generated** or a **real photograph**. Built for a 24-hour hackathon, it provides:

- ✅ **Binary Classification**: AI-Generated vs Real
- ✅ **Confidence Scores**: Percentage-based reliability
- ✅ **Human-Readable Explanations**: Clear reasons for predictions
- ✅ **Platform Simulation**: Test stability across WhatsApp, Instagram, Facebook
- ✅ **Explainability Focus**: Visual cue analysis (hands, faces, textures, lighting)

---

## ✨ Features

### 🔍 Core Features

- **AI Detection Model**: CNN-based classifier (EfficientNet-B0)
- **Confidence Scoring**: 0-100% prediction confidence
- **Risk Levels**: High / Medium / Uncertain
- **Visual Explanations**: Detects artifacts like:
  - Unnatural hand structure
  - Asymmetrical facial features
  - Over-smooth textures
  - Lighting inconsistencies
  - Repeated patterns

### 📱 Platform Simulation

Test how predictions change after social media compression:

| Platform  | Resolution | JPEG Quality | Use Case           |
|-----------|------------|--------------|---------------------|
| WhatsApp  | 512px      | 40%          | Aggressive compression |
| Instagram | 1080px     | 70%          | Moderate compression   |
| Facebook  | 960px      | 60%          | Balanced compression   |

**Stability Score**: Measures prediction consistency across platforms (0-100%)

### 🎨 UI/UX Highlights

- Beautiful gradient background (blue → cyan)
- Hero section with robot vs human imagery
- Drag-and-drop image upload
- Real-time analysis with loading states
- Responsive design for mobile/desktop

---

## 🛠 Tech Stack

### Frontend
- **React.js** 18.2 - Modern functional components
- **CSS3** - Custom styling with animations
- **Axios** - HTTP client for API calls

### Backend
- **FastAPI** - High-performance Python web framework
- **Uvicorn** - ASGI server
- **Python 3.9+** - Core language

### Machine Learning
- **PyTorch** - Deep learning framework
- **Torchvision** - Pre-trained models
- **timm** (PyTorch Image Models) - EfficientNet architecture
- **Pillow** - Image processing
- **NumPy** - Numerical operations

### Storage
- **Supabase Storage** - Object storage for images
- **PostgreSQL** - Metadata storage (via Supabase)

---

## 📁 Project Structure

```
Truepix/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── model_inference.py       # CNN model + predictions
│   ├── platform_simulator.py    # Image transformations
│   ├── storage_manager.py       # Supabase integration
│   ├── requirements.txt         # Python dependencies
│   └── .env.example             # Environment variables template
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── LandingPage.js
│   │   │   ├── ImageUpload.js
│   │   │   ├── ResultsPanel.js
│   │   │   ├── PlatformSimulation.js
│   │   │   └── *.css
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── .env.example
│
├── model/
│   ├── weights/                 # Trained model checkpoints
│   └── train_model.py           # Training script (reference)
│
├── .gitignore
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Node.js 16+**
- **npm or yarn**
- **(Optional) CUDA** for GPU acceleration

### 1. Clone Repository

```bash
git clone <repository-url>
cd Truepix
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Supabase credentials
```

#### Environment Variables (`.env`)

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_BUCKET=truepix-images

API_HOST=0.0.0.0
API_PORT=8000

MODEL_PATH=../model/weights/efficientnet_aidetector.pth
DEVICE=cpu  # or cuda
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env if needed
```

#### Environment Variables (`.env`)

```env
REACT_APP_API_URL=http://localhost:8000
```

### 4. Supabase Setup (Optional)

1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Go to **Storage** → Create bucket `truepix-images` (public)
4. Copy **Project URL** and **anon key** to backend `.env`

**Note**: App works in demo mode without Supabase (uses mock URLs)

---

## 🎮 Usage

### Start Backend

```bash
cd backend
source venv/bin/activate  # Activate venv
python main.py
```

Backend runs on: `http://localhost:8000`  
API Docs: `http://localhost:8000/docs`

### Start Frontend

```bash
cd frontend
npm start
```

Frontend runs on: `http://localhost:3000`

### Using the Application

1. **Upload Image**: Click or drag-and-drop JPG/PNG
2. **View Results**: See prediction, confidence, risk level, explanation
3. **Test Platforms**: Click "Test Platform Stability" to simulate social media
4. **Compare**: View how prediction changes across platforms
5. **Stability Score**: Check prediction consistency

---

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /
```

Response:
```json
{
  "message": "TruePix API is running",
  "version": "1.0.0",
  "status": "healthy"
}
```

#### 2. Upload Image
```http
POST /api/upload
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "image_id": "uuid-string",
  "image_url": "https://storage.url/path/to/image.jpg",
  "filename": "uuid.jpg"
}
```

#### 3. Analyze Image
```http
POST /api/analyze
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "prediction": "AI-Generated",
  "confidence": 87.5,
  "risk_level": "High",
  "explanation": "Very high confidence of AI generation | Detected unnatural texture smoothness...",
  "image_url": "https://storage.url/...",
  "metadata": {
    "image_id": "uuid",
    "filename": "image.jpg",
    "timestamp": "2026-01-09T12:00:00Z",
    "model_version": "efficientnet-b0-v1.0"
  }
}
```

#### 4. Platform Simulation
```http
POST /api/simulate-platforms
Content-Type: multipart/form-data

file: <image_file>
```

Response:
```json
{
  "platform": "multi-platform",
  "original_result": {
    "prediction": "AI-Generated",
    "confidence": 87.5,
    "explanation": "..."
  },
  "platform_results": {
    "whatsapp": {
      "prediction": "AI-Generated",
      "confidence": 82.3,
      "explanation": "..."
    },
    "instagram": { ... },
    "facebook": { ... }
  },
  "stability_score": 78.5
}
```

---

## 🤖 Model Training

### Datasets

For production, train on:

1. **CIFAKE** - Real vs AI images  
   [Kaggle Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

2. **DiffusionDB** - Stable Diffusion generated images  
   [Hugging Face](https://huggingface.co/datasets/poloclub/diffusiondb)

3. **FFHQ** - High-quality real faces  
   [GitHub](https://github.com/NVlabs/ffhq-dataset)

4. **COCO** - Real photographs  
   [COCO Dataset](https://cocodataset.org/)

### Training Script Reference

```python
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader

# Load pre-trained model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), 'model/weights/efficientnet_aidetector.pth')
```

### Visual Cues to Train For

- ✅ Anatomically incorrect hands (extra/missing fingers)
- ✅ Unnatural facial symmetry
- ✅ Over-smooth skin textures
- ✅ Repeated background patterns
- ✅ Inconsistent lighting/shadows
- ✅ Blurred text/logos
- ✅ Unnatural eye reflections

---

## 📱 Platform Simulation

### How It Works

1. **Original Image**: Analyzed without modification
2. **Transform**: Resize + JPEG compress per platform specs
3. **Re-analyze**: Run model on transformed image
4. **Compare**: Calculate stability score

### Stability Score Formula

```python
stability = 100 - (std_deviation * 200)
```

- **High (75-100%)**: Robust predictions
- **Medium (50-75%)**: Some variation
- **Low (0-50%)**: Significant instability

### Why This Matters

- Images compressed on social media may fool AI detectors
- Stability testing shows model robustness
- Helps identify compression-sensitive features

---

## 🎥 Demo

### Demo Mode Features

Without Supabase setup:
- ✅ Image upload works (stored in memory)
- ✅ Model inference functional
- ✅ Platform simulation active
- ✅ Mock storage URLs generated

### Sample Images to Test

**AI-Generated (try Midjourney/DALL-E outputs)**:
- Perfect portraits with unnatural smoothness
- Complex hands/fingers
- Text-heavy images

**Real Photos**:
- Natural camera photos with noise
- Candid shots with imperfections
- Authentic lighting variations

---

## ⚠️ Limitations

### Current Constraints

- **Demo Model**: Uses pre-trained ImageNet weights (not fine-tuned on AI datasets)
- **Accuracy**: Production model requires training on 50k+ labeled images
- **Compression**: Heavy JPEG compression may reduce accuracy
- **New AI Models**: May not detect latest generation techniques (2026+)
- **Adversarial Attacks**: Vulnerable to purposefully crafted images

### Responsible Use

⚠️ **This tool is not 100% accurate**

- Use as **guidance**, not proof
- Combine with human verification
- Consider context and source
- Don't use for legal decisions without expert review

---

## 🚀 Future Enhancements

### Planned Features

- [ ] **Grad-CAM Visualization**: Highlight suspicious regions
- [ ] **Multi-model Ensemble**: Combine predictions from multiple CNNs
- [ ] **Fine-tuned Weights**: Train on 100k+ labeled images
- [ ] **Metadata Analysis**: Check EXIF data for AI signatures
- [ ] **Batch Processing**: Upload multiple images
- [ ] **History Dashboard**: Track previous analyses
- [ ] **API Rate Limiting**: Prevent abuse
- [ ] **Model Version Selection**: Choose detection model
- [ ] **Advanced Explanations**: NLP-generated reports
- [ ] **Mobile App**: iOS/Android native apps

### Deployment

**Recommended Stack**:
- **Frontend**: Vercel / Netlify
- **Backend**: Railway / Render / AWS Lambda
- **Storage**: Supabase / Cloudinary / AWS S3
- **Database**: Supabase PostgreSQL
- **Model Hosting**: Hugging Face / TorchServe

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** team for excellent deep learning framework
- **timm** library for pre-trained models
- **FastAPI** for blazing-fast API development
- **React** community for modern UI tools
- **Supabase** for free-tier object storage

---

## 📧 Contact

**Project**: TruePix  
**Built for**: Hackathon 2026  
**Purpose**: AI Image Detection with Explainability

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for the AI transparency community

</div>
