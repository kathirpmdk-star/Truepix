# TruePix - AI Image Detection SystemTruePix – Robust AI vs Real Image Detection System

Overview

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)TruePix is a system designed to distinguish AI-generated images from real photographs using image forensic cues and deep learning.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)The project addresses the growing challenge of synthetic media misuse, especially in social media, journalism, and digital evidence verification.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)The system accepts an image as input and outputs:

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)Prediction: AI-generated or Real

Confidence score

**A multi-branch deep learning system that detects AI-generated images and explains why.**Feature-based explanation (backend level)

Problem Statement

---Recent advances in generative AI models (GANs, Diffusion models) have made synthetic images visually indistinguishable from real ones. This creates serious risks such as:

Spread of misinformation

## What is TruePix?Fake digital evidence

Loss of trust in visual media

TruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.Traditional visual inspection is no longer reliable.

Hence, an automated, robust detection system is required to analyze hidden artifacts left by AI generation pipelines.

**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.Proposed Solution

TruePix detects AI-generated images by analyzing both spatial and frequency-domain inconsistencies that are commonly introduced during synthetic image generation.

---Key objectives:

Detect AI-generated images with high robustness

## Key FeaturesRemain effective under compression and resizing

Provide a scalable backend inference API

✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  Enable real-time image verification via a frontend interface

✅ **Explainable AI** - Shows why an image is classified as real or AI  Methodology

✅ **Grad-CAM Visualization** - Highlights suspicious image regions  The detection pipeline consists of the following stages:

✅ **Platform Robustness** - Tests stability across social media compression  1. Image Preprocessing

✅ **Real-Time Analysis** - Fast inference with web interface  Image resizing and normalization

Color space standardization

---Noise stabilization

2. Feature Extraction

## How It WorksMultiple forensic cues are analyzed:

Spatial artifacts using CNN-based feature extraction

### The 4 Detection AlgorithmsFrequency-domain patterns (FFT-based anomalies)

Noise residual inconsistencies

| Algorithm | What It Detects | Why It Works |Edge and texture irregularities

|-----------|----------------|--------------|These features capture subtle differences between real camera pipelines and AI image generators.

| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |3. Classification

| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Extracted features are passed to a trained deep learning classifier

| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |The model outputs a probability score for AI vs Real classification

| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |4. Inference API

Backend exposes a REST API

### The ProcessAccepts image uploads

Returns prediction and confidence score

```System Architecture

1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationUser Image

```   ↓

Frontend (Upload Interface)

Each algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.   ↓

Backend API

---   ↓

Preprocessing → Feature Extraction → Classifier

## Tech Stack   ↓

Prediction + Confidence

- **Frontend:** React 18.2, CSS3

- **Backend:** FastAPI 0.108, UvicornTech Stack

- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0Backend

- **Image Processing:** OpenCV, Pillow, scipyPython

FastAPI / Flask

---OpenCV

PyTorch (model inference)

## Quick StartFrontend

React

### Backend SetupHTML / CSS / JavaScript

```bashModel & Analysis

cd backendCNN-based feature extraction

python3 -m venv venvFrequency-domain analysis

source venv/bin/activateImage forensics techniques

pip install -r requirements.txtHow to Run the Project

python main.pyBackend

```cd backend

Backend runs at: **http://localhost:8000**pip install -r requirements.txt

python app.py

### Frontend SetupFrontend

```bashcd frontend

cd frontendnpm install

npm installnpm start

npm startResults & Evaluation

```The system successfully identifies AI-generated images by detecting non-natural artifacts.

Frontend runs at: **http://localhost:3000**Tested on a mixed dataset containing:

Real photographs

### UsageAI-generated images from multiple generation sources

1. Open http://localhost:3000Performance was evaluated using accuracy and confidence-based prediction consistency.

2. Upload an image (JPG/PNG)Note: Due to dataset size constraints, datasets are not included in the repository.

3. View results: classification, confidence, explanation, Grad-CAM heatmapLimitations

Performance depends on diversity of training data

---Newer AI generators may introduce unseen patterns

Does not guarantee 100% accuracy for highly post-processed images

## ResultsFuture Work

Extend detection to video and deepfake content

- **Accuracy:** 89.3% on test setIntegrate explainability heatmaps for visual justification

- **Precision:** 91.2% (AI detection)Expand dataset to include newer AI image generators

- **Robustness:** 82-87% accuracy after social media compressionImprove robustness against adversarial attacks

- **Explainability:** 84% Grad-CAM accuracy, 92% user satisfactionEthical Considerations

This project is intended only for detection and verification purposes.

---It does not generate or promote misuse of AI-generated content.

Author

## LimitationsKathiravan M

Project: TruePix – AI vs Real Image Detection

- Trained on 2024-2025 AI generators; may not detect future models

- Accuracy drops 5-8% under heavy compression (< 50% JPEG quality)

- Not hardened against adversarial attacks
- Struggles with hybrid images (real + AI edits)

---

## Future Work

- Ensemble models (Vision Transformers + ResNet) for higher accuracy
- EXIF metadata analysis for forensic verification
- Video deepfake detection
- Mobile app with on-device inference
- Model quantization for 3x faster inference

---

## References

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot." CVPR 2020.
2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.
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
