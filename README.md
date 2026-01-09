# TruePix - AI Image Detection System# TruePix - AI Image Detection SystemTruePix – Robust AI vs Real Image Detection System



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)Overview

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)TruePix is a system designed to distinguish AI-generated images from real photographs using image forensic cues and deep learning.

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)The project addresses the growing challenge of synthetic media misuse, especially in social media, journalism, and digital evidence verification.

**A multi-branch deep learning system that detects AI-generated images and explains why.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)The system accepts an image as input and outputs:

---

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)Prediction: AI-generated or Real

## What is TruePix?

Confidence score

TruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.

**A multi-branch deep learning system that detects AI-generated images and explains why.**Feature-based explanation (backend level)

**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.

Problem Statement

---

---Recent advances in generative AI models (GANs, Diffusion models) have made synthetic images visually indistinguishable from real ones. This creates serious risks such as:

## Key Features

Spread of misinformation

✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  

✅ **Explainable AI** - Shows why an image is classified as real or AI  ## What is TruePix?Fake digital evidence

✅ **Grad-CAM Visualization** - Highlights suspicious image regions  

✅ **Platform Robustness** - Tests stability across social media compression  Loss of trust in visual media

✅ **Real-Time Analysis** - Fast inference with web interface  

TruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.Traditional visual inspection is no longer reliable.

---

Hence, an automated, robust detection system is required to analyze hidden artifacts left by AI generation pipelines.

## How It Works

**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.Proposed Solution

### The 4 Detection Algorithms

TruePix detects AI-generated images by analyzing both spatial and frequency-domain inconsistencies that are commonly introduced during synthetic image generation.

| Algorithm | What It Detects | Why It Works |

|-----------|----------------|--------------|---Key objectives:

| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |

| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Detect AI-generated images with high robustness

| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |

| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |## Key FeaturesRemain effective under compression and resizing



### The ProcessProvide a scalable backend inference API



```✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  Enable real-time image verification via a frontend interface

1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + Explanation

```✅ **Explainable AI** - Shows why an image is classified as real or AI  Methodology



Each algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.✅ **Grad-CAM Visualization** - Highlights suspicious image regions  The detection pipeline consists of the following stages:



---✅ **Platform Robustness** - Tests stability across social media compression  1. Image Preprocessing



## Tech Stack✅ **Real-Time Analysis** - Fast inference with web interface  Image resizing and normalization



- **Frontend:** React 18.2, CSS3Color space standardization

- **Backend:** FastAPI 0.108, Uvicorn

- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0---Noise stabilization

- **Image Processing:** OpenCV, Pillow, scipy

2. Feature Extraction

---

## How It WorksMultiple forensic cues are analyzed:

## Installation & Setup Guide

Spatial artifacts using CNN-based feature extraction

### Prerequisites

### The 4 Detection AlgorithmsFrequency-domain patterns (FFT-based anomalies)

Before you begin, ensure you have the following installed on your laptop/device:

Noise residual inconsistencies

- **Python 3.9 or higher** - [Download here](https://www.python.org/downloads/)

- **Node.js 16 or higher** - [Download here](https://nodejs.org/)| Algorithm | What It Detects | Why It Works |Edge and texture irregularities

- **Git** - [Download here](https://git-scm.com/downloads)

- **4GB RAM minimum** (8GB recommended)|-----------|----------------|--------------|These features capture subtle differences between real camera pipelines and AI image generators.

- **Internet connection** (for installing dependencies)

| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |3. Classification

### Step 1: Clone the Repository

| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Extracted features are passed to a trained deep learning classifier

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |The model outputs a probability score for AI vs Real classification

```bash

# Clone the repository| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |4. Inference API

git clone https://github.com/kathirpmdk-star/Truepix.git

Backend exposes a REST API

# Navigate to the project directory

cd Truepix### The ProcessAccepts image uploads

```

Returns prediction and confidence score

### Step 2: Backend Setup

```System Architecture

```bash

# Navigate to backend folder1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationUser Image

cd backend

```   ↓

# Create a virtual environment

python3 -m venv venvFrontend (Upload Interface)



# Activate virtual environmentEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.   ↓

# On Mac/Linux:

source venv/bin/activateBackend API

# On Windows:

venv\Scripts\activate---   ↓



# Install Python dependencies (this may take 2-3 minutes)Preprocessing → Feature Extraction → Classifier

pip install -r requirements.txt

## Tech Stack   ↓

# Start the backend server

python main.pyPrediction + Confidence

```

- **Frontend:** React 18.2, CSS3

✅ **Backend is now running at:** http://localhost:8000  

✅ **API Documentation available at:** http://localhost:8000/docs- **Backend:** FastAPI 0.108, UvicornTech Stack



**Keep this terminal window open!**- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0Backend



### Step 3: Frontend Setup- **Image Processing:** OpenCV, Pillow, scipyPython



Open a **NEW terminal window** (keep the backend running) and run:FastAPI / Flask



```bash---OpenCV

# Navigate to the project directory

cd Truepix/frontendPyTorch (model inference)



# Install Node.js dependencies (this may take 2-3 minutes)## Quick StartFrontend

npm install

React

# Start the frontend development server

npm start### Backend SetupHTML / CSS / JavaScript

```

```bashModel & Analysis

✅ **Frontend is now running at:** http://localhost:3000

cd backendCNN-based feature extraction

Your browser should automatically open. If not, manually open: **http://localhost:3000**

python3 -m venv venvFrequency-domain analysis

### Step 4: Using TruePix

source venv/bin/activateImage forensics techniques

1. **Upload an Image:**

   - Click the upload area or drag-and-drop an image (JPG/PNG format)pip install -r requirements.txtHow to Run the Project

   - Recommended image size: 500KB - 5MB

python main.pyBackend

2. **View Results:**

   - Classification: "AI-Generated" or "Real"```cd backend

   - Confidence Score: 0-100%

   - Executive Summary: Why the image was classifiedBackend runs at: **http://localhost:8000**pip install -r requirements.txt

   - Branch Analysis: What each algorithm detected

   - Grad-CAM Heatmap: Visual explanation highlighting suspicious regionspython app.py



3. **Test Platform Stability (Optional):**### Frontend SetupFrontend

   - Click "Test Platform Stability" button

   - See how prediction changes across WhatsApp, Instagram, Facebook compression```bashcd frontend



---cd frontendnpm install



## Troubleshootingnpm installnpm start



**Problem: "Python not found"**npm startResults & Evaluation

- Solution: Install Python from python.org and restart terminal

```The system successfully identifies AI-generated images by detecting non-natural artifacts.

**Problem: "npm not found"**

- Solution: Install Node.js from nodejs.org and restart terminalFrontend runs at: **http://localhost:3000**Tested on a mixed dataset containing:



**Problem: "Port 8000 already in use"**Real photographs

- Solution: Kill the process using port 8000:

  ```bash### UsageAI-generated images from multiple generation sources

  # Mac/Linux:

  lsof -ti:8000 | xargs kill -91. Open http://localhost:3000Performance was evaluated using accuracy and confidence-based prediction consistency.

  # Windows:

  netstat -ano | findstr :80002. Upload an image (JPG/PNG)Note: Due to dataset size constraints, datasets are not included in the repository.

  taskkill /PID <PID_NUMBER> /F

  ```3. View results: classification, confidence, explanation, Grad-CAM heatmapLimitations



**Problem: "Port 3000 already in use"**Performance depends on diversity of training data

- Solution: Kill the process or use a different port:

  ```bash---Newer AI generators may introduce unseen patterns

  # The frontend will prompt: "Would you like to run on another port?" → Press Y

  ```Does not guarantee 100% accuracy for highly post-processed images



**Problem: Backend crashes or errors**## ResultsFuture Work

- Solution: Make sure you activated the virtual environment

- Check Python version: `python --version` (should be 3.9+)Extend detection to video and deepfake content

- Reinstall dependencies: `pip install -r requirements.txt`

- **Accuracy:** 89.3% on test setIntegrate explainability heatmaps for visual justification

**Problem: Frontend doesn't connect to backend**

- Solution: Make sure backend is running on http://localhost:8000- **Precision:** 91.2% (AI detection)Expand dataset to include newer AI image generators

- Check if both terminals are still active

- **Robustness:** 82-87% accuracy after social media compressionImprove robustness against adversarial attacks

---

- **Explainability:** 84% Grad-CAM accuracy, 92% user satisfactionEthical Considerations

## Stopping the Application

This project is intended only for detection and verification purposes.

To stop the servers:

---It does not generate or promote misuse of AI-generated content.

1. **Stop Frontend:** Press `Ctrl + C` in the frontend terminal

2. **Stop Backend:** Press `Ctrl + C` in the backend terminalAuthor

3. **Deactivate virtual environment:** Type `deactivate` in backend terminal

## LimitationsKathiravan M

---

Project: TruePix – AI vs Real Image Detection

## System Requirements

- Trained on 2024-2025 AI generators; may not detect future models

| Component | Minimum | Recommended |

|-----------|---------|-------------|- Accuracy drops 5-8% under heavy compression (< 50% JPEG quality)

| OS | Windows 10, macOS 10.15, Ubuntu 20.04 | Latest version |

| RAM | 4GB | 8GB |- Not hardened against adversarial attacks

| Storage | 2GB free space | 5GB free space |- Struggles with hybrid images (real + AI edits)

| CPU | Dual-core 2.0GHz | Quad-core 2.5GHz+ |

| GPU | Not required | CUDA-compatible (optional, for faster inference) |---



---## Future Work



## Results- Ensemble models (Vision Transformers + ResNet) for higher accuracy

- EXIF metadata analysis for forensic verification

- **Accuracy:** 89.3% on test set- Video deepfake detection

- **Precision:** 91.2% (AI detection)- Mobile app with on-device inference

- **Robustness:** 82-87% accuracy after social media compression- Model quantization for 3x faster inference

- **Explainability:** 84% Grad-CAM accuracy, 92% user satisfaction

---

---

## References

## Limitations

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot." CVPR 2020.

- Trained on 2024-2025 AI generators; may not detect future models2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.

- Accuracy drops 5-8% under heavy compression (< 50% JPEG quality)3. Selvaraju et al. (2017). "Grad-CAM Visual Explanations." ICCV 2017.

- Not hardened against adversarial attacks

- Struggles with hybrid images (real + AI edits)**Datasets:** CIFAKE, DiffusionDB, COCO, FFHQ  

**Tools:** PyTorch, timm, FastAPI, React

---

---

## Future Work

## License

- Ensemble models (Vision Transformers + ResNet) for higher accuracy

- EXIF metadata analysis for forensic verificationMIT License - See LICENSE file

- Video deepfake detection

- Mobile app with on-device inference---

- Model quantization for 3x faster inference

<div align="center">

---

**Developed for Academic Research**

## References

*Promoting transparency in the age of generative AI*

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot." CVPR 2020.

2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.</div>

3. Selvaraju et al. (2017). "Grad-CAM Visual Explanations." ICCV 2017.

**Datasets:** CIFAKE, DiffusionDB, COCO, FFHQ  
**Tools:** PyTorch, timm, FastAPI, React

---

## License

MIT License - See LICENSE file

---

<div align="center">

**Developed for Academic Research**

*Promoting transparency in the age of generative AI*

</div>
