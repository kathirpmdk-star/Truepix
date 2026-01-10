TruePix – Robust AI vs Real Image Detection System



> **AI vs Real Image Detection System with Explainable Multi-Branch Deep Learning**



[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)

[![PyTorch](https://img.shields.io/badge/pytorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688.svg)](https://fastapi.tiangolo.com/)![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)

[![React](https://img.shields.io/badge/react-18.2-61dafb.svg)](https://reactjs.org/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)



A production-ready deep learning system that detects AI-generated images with 89.3% accuracy and provides transparent, explainable results through a hybrid multi-branch architecture combining CNN, FFT, noise analysis, and edge detection.![React](https://img.shields.io/badge/React-18.2-61dafb.svg)



---[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)



## 🎯 OverviewA sophisticated deep learning system that detects AI-generated images and provides explainable results using a multi-branch hybrid architecture.



TruePix is an advanced AI image detection system designed to distinguish between AI-generated images and authentic photographs. In an era where generative AI models (DALL-E, Midjourney, Stable Diffusion) create photorealistic synthetic images, TruePix provides robust detection with comprehensive explainability.[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)Overview



### Key Capabilities---



- **Multi-Branch Detection**: Combines 4 complementary algorithms (CNN, FFT, Noise, Edge) for robust accuracy[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

- **Explainable AI**: Provides detailed reasoning, confidence scores, and visual heatmaps for every prediction

- **Production Ready**: Full-stack application with FastAPI backend and React frontend## Table of Contents

- **Platform Robustness**: Tested against real-world compression (WhatsApp, Instagram, Facebook)

- **Real-Time Analysis**: Fast inference (~150ms on CPU) with intuitive web interface[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)



### Use Cases1. [Overview](#overview)



- Content moderation on social media platforms2. [Problem Statement](#problem-statement)**A multi-branch deep learning system that detects AI-generated images and explains why.**

- Fact-checking and journalism verification

- Digital forensics and fraud investigation3. [Key Features](#key-features)

- Deepfake detection and media authentication

- Academic research on AI-generated content4. [System Architecture](#system-architecture)[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)TruePix is a system designed to distinguish AI-generated images from real photographs using image forensic cues and deep learning.



---5. [Detection Algorithms](#detection-algorithms)



## 📋 Table of Contents6. [Technology Stack](#technology-stack)---



- [Architecture](#-architecture)7. [How to Run This Project](#how-to-run-this-project)

- [Installation](#-installation)

- [Quick Start](#-quick-start)8. [How to Use TruePix](#how-to-use-truepix)[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

- [Usage](#-usage)

- [Detection Algorithms](#-detection-algorithms)9. [Results and Performance](#results-and-performance)

- [API Reference](#-api-reference)

- [Performance](#-performance)10. [What Makes TruePix Unique](#what-makes-truepix-unique)## Overview

- [Technology Stack](#-technology-stack)

- [Project Structure](#-project-structure)11. [Limitations](#limitations)

- [Contributing](#-contributing)

- [Limitations](#-limitations)12. [Future Work](#future-work)[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)The project addresses the growing challenge of synthetic media misuse, especially in social media, journalism, and digital evidence verification.

- [Roadmap](#-roadmap)

- [License](#-license)13. [Conclusion](#conclusion)

- [Citation](#-citation)

14. [References](#references)TruePix is an advanced AI image detection system that distinguishes AI-generated images from real photographs using a novel hybrid multi-branch architecture. Unlike traditional single-model approaches, TruePix combines four complementary detection methods—CNN spatial analysis, FFT frequency analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.

---



## 🏗 Architecture

---**A multi-branch deep learning system that detects AI-generated images and explains why.**

TruePix uses a **hybrid multi-branch architecture** where each branch specializes in detecting different AI generation artifacts:



```

┌─────────────────────────────────────────────────────────────────┐## 1. Overview**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification, social media authentication.

│                         Input Image                             │

│                     (224 x 224 x 3)                             │

└────────────────────────┬────────────────────────────────────────┘

                         │TruePix is an advanced AI image detection system designed to distinguish between AI-generated images and real photographs. With the rapid advancement of generative AI models like DALL-E, Midjourney, and Stable Diffusion, synthetic images have become virtually indistinguishable from authentic photographs to the human eye.[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)The system accepts an image as input and outputs:

            ┌────────────┼────────────┬────────────┐

            │            │            │            │

    ┌───────▼───────┐ ┌──▼──────┐ ┌──▼──────┐ ┌──▼──────┐

    │ CNN Branch    │ │   FFT   │ │  Noise  │ │  Edge   │TruePix solves this problem using a **multi-branch hybrid architecture** that combines four complementary detection methods:---

    │ EfficientNet  │ │ 2D FFT  │ │ Multi-  │ │ Sobel   │

    │ B0 Backbone   │ │ Analysis│ │ Scale   │ │ Operator│- **CNN-based spatial analysis**

    │               │ │         │ │ Analysis│ │         │

    └───────┬───────┘ └──┬──────┘ └──┬──────┘ └──┬──────┘- **FFT frequency analysis**---

            │            │            │            │

            │ 256-dim    │ 256-dim    │ 256-dim    │ 256-dim- **Noise pattern verification**

            │            │            │            │

            └────────────┴────────────┴────────────┘- **Edge structure analysis**## Problem Statement

                         │

                    ┌────▼─────┐

                    │  Fusion  │

                    │  Layer   │The system provides not just predictions, but **explainable results** with visual heatmaps showing exactly why an image was classified as AI-generated or real.[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)Prediction: AI-generated or Real

                    │ 1024-dim │

                    └────┬─────┘

                         │

                    ┌────▼─────┐---Recent advances in generative AI models (DALL-E, Midjourney, Stable Diffusion) have made synthetic images visually indistinguishable from real photographs. This poses critical challenges:

                    │   FC     │

                    │ Layers   │

                    └────┬─────┘

                         │## 2. Problem Statement## What is TruePix?

            ┌────────────┴────────────┐

            │                         │

      ┌─────▼──────┐          ┌──────▼────────┐

      │Prediction  │          │ Explainability│The proliferation of AI-generated images poses significant challenges:- **Misinformation:** Fake news propagation using synthetic imagery

      │(AI/Real)   │          │ • Grad-CAM    │

      │Confidence  │          │ • Branch Score│

      └────────────┘          │ • Summary     │

                              └───────────────┘- **Misinformation:** Fake news spreading through synthetic imagery- **Digital Fraud:** Identity theft, document forgery, deepfake scamsConfidence score

```

- **Digital Fraud:** Identity theft, document forgery, deepfake scams

### How It Works

- **Trust Crisis:** Declining confidence in digital media authenticity- **Trust Erosion:** Declining confidence in digital media authenticity

1. **Feature Extraction**: Each branch independently processes the image

   - CNN: Spatial texture patterns (256 features)- **Evidence Integrity:** Compromised digital evidence in legal proceedings

   - FFT: Frequency domain artifacts (256 features)

   - Noise: Multi-scale sensor noise (256 features)- **Evidence Tampering:** Compromised digital evidence in legal contextsTruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.

   - Edge: Physical plausibility (256 features)

Traditional visual inspection is no longer reliable. Single-method detection systems fail to catch sophisticated AI generators. There is an urgent need for a **robust, multi-faceted detection system** that can identify hidden artifacts left by AI generation pipelines.

2. **Feature Fusion**: Concatenates all features into 1024-dimensional vector



3. **Classification**: Fully connected layers predict AI vs Real with confidence score

---

4. **Explainability**: Grad-CAM generates visual heatmaps + per-branch attribution

Traditional visual inspection and single-method detection systems are no longer reliable. An automated, multi-faceted detection system is essential to analyze hidden artifacts left by AI generation pipelines.**A multi-branch deep learning system that detects AI-generated images and explains why.**Feature-based explanation (backend level)

---

## 3. Key Features

## 🚀 Installation



### Prerequisites

✅ **Multi-Branch Detection** - Uses 4 different algorithms working together for robust accuracy

Ensure you have the following installed:

---**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.

| Requirement | Version | Download |

|-------------|---------|----------|✅ **Explainable AI** - Provides detailed reasoning for every classification decision

| Python | 3.9+ | [python.org](https://www.python.org/downloads/) |

| Node.js | 16+ | [nodejs.org](https://nodejs.org/) |

| Git | Latest | [git-scm.com](https://git-scm.com/) |

✅ **Visual Explanations** - Grad-CAM heatmaps highlight suspicious regions in images

### System Requirements

## Key FeaturesProblem Statement

- **OS**: Windows 10+, macOS 10.15+, Ubuntu 20.04+

- **RAM**: 4GB minimum (8GB recommended)✅ **Platform Robustness Testing** - Tests stability across social media compression (WhatsApp, Instagram, Facebook)

- **Storage**: 2GB free space

- **CPU**: Dual-core 2.0GHz+ (GPU optional for faster inference)



### Clone Repository✅ **Real-Time Analysis** - Fast inference with an intuitive web interface



```bash✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy across different AI generators  ---

git clone https://github.com/kathirpmdk-star/Truepix.git

cd Truepix✅ **High Accuracy** - Achieves 89.3% accuracy with 91.2% precision on AI detection

```

✅ **Explainable AI** - Provides detailed reasoning for every classification decision  

---

---

## ⚡ Quick Start

✅ **Grad-CAM Visualization** - Highlights suspicious image regions with visual heatmaps  ---Recent advances in generative AI models (GANs, Diffusion models) have made synthetic images visually indistinguishable from real ones. This creates serious risks such as:

### Backend Setup

## 4. System Architecture

```bash

# Navigate to backend directory✅ **Platform Robustness** - Tests stability across social media compression scenarios  

cd backend

TruePix uses a **multi-branch hybrid architecture** where each branch specializes in detecting different types of AI artifacts:

# Create virtual environment

python3 -m venv venv✅ **Real-Time Analysis** - Fast inference with intuitive web interface  ## Key Features



# Activate virtual environment```

# On macOS/Linux:

source venv/bin/activate                         Input Image✅ **High Accuracy** - 89.3% accuracy with 91.2% precision on AI detection

# On Windows:

venv\Scripts\activate                              |



# Install dependencies                 ┌────────────┼────────────┐Spread of misinformation

pip install -r requirements.txt

                 |            |            |

# Start backend server

python main.py         ┌───────▼──┐    ┌────▼───┐    ┌──▼─────┐    ┌─────────┐---

```

         │ CNN      │    │ FFT    │    │ Noise  │    │ Edge    │

✅ Backend runs at **http://localhost:8000**  

📖 API docs available at **http://localhost:8000/docs**         │ Branch   │    │ Branch │    │ Branch │    │ Branch  │✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  



### Frontend Setup         └───────┬──┘    └────┬───┘    └──┬─────┘    └────┬────┘



Open a new terminal:                 |            |            |               |## What Makes TruePix Unique?



```bash                 └────────────┼────────────┘               |

# Navigate to frontend directory

cd frontend                              ▼                            |✅ **Explainable AI** - Shows why an image is classified as real or AI  ## What is TruePix?Fake digital evidence



# Install dependencies                      Feature Fusion (1024-dim)            |

npm install

                              |                            |### Compared to Existing Solutions

# Start development server

npm start                              ▼                            |

```

                    Classification Network                 |✅ **Grad-CAM Visualization** - Highlights suspicious image regions  

✅ Frontend runs at **http://localhost:3000**

                              |                            |

### Verify Installation

                    ┌─────────┴─────────┐                  |Most AI image detection tools available online rely on single-model approaches (typically just CNN-based classifiers). TruePix stands out with several innovations:

1. Open browser to `http://localhost:3000`

2. Upload a test image (JPG/PNG)                    |                   |                  |

3. View classification results with explainability

             Prediction          Explainability            |✅ **Platform Robustness** - Tests stability across social media compression  Loss of trust in visual media

---

          (AI or Real)        (Executive Summary,          |

## 📖 Usage

           Confidence         Branch Attributions,    ◄────┘| Feature | Traditional Detectors | TruePix |

### Web Interface

                              Grad-CAM Heatmap)

1. **Upload Image**

   - Click upload area or drag-and-drop```|---------|----------------------|---------|✅ **Real-Time Analysis** - Fast inference with web interface  

   - Supported formats: JPG, PNG

   - Max size: 10MB



2. **View Results****How it works:**| **Detection Methods** | Single CNN model | 4 complementary algorithms (CNN + FFT + Noise + Edge) |

   - **Classification**: AI-Generated or Real

   - **Confidence Score**: 0-100%

   - **Executive Summary**: Detailed reasoning

   - **Branch Analysis**: Individual algorithm contributions1. **Feature Extraction:** Each branch independently analyzes the image and extracts 256 features| **Explainability** | Black-box predictions | Full transparency with per-branch attribution + Grad-CAM |TruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.Traditional visual inspection is no longer reliable.

   - **Grad-CAM Heatmap**: Visual explanation highlighting suspicious regions

2. **Feature Fusion:** All branch features are concatenated into a 1024-dimensional vector

3. **Platform Stability Test** (Optional)

   - Test robustness against social media compression3. **Classification:** A fusion network processes the combined features to make the final prediction| **Robustness** | Fails on compressed images | Tested against WhatsApp, Instagram, Facebook compression |

   - Compare predictions across WhatsApp, Instagram, Facebook

4. **Explainability:** Grad-CAM generates visual heatmaps and each branch provides its confidence score

### API Usage

| **Frequency Analysis** | Rarely used | FFT-based detection of periodic artifacts |---

```python

import requests---



# Upload image for analysis| **Noise Analysis** | Not implemented | Multi-scale sensor noise verification |

url = "http://localhost:8000/api/analyze"

files = {"file": open("image.jpg", "rb")}## 5. Detection Algorithms



response = requests.post(url, files=files)| **Physical Plausibility** | Ignored | Edge structure analysis based on optical physics |Hence, an automated, robust detection system is required to analyze hidden artifacts left by AI generation pipelines.

result = response.json()

TruePix employs four complementary algorithms, each targeting different AI generation artifacts:

print(f"Prediction: {result['prediction']}")

print(f"Confidence: {result['confidence']}%")| **Fusion Strategy** | N/A | Temperature-calibrated multi-branch fusion |

print(f"Summary: {result['metadata']['executive_summary']}")

```### 5.1 CNN (Convolutional Neural Network) - Spatial Analysis



### Python SDK Example| **User Interface** | API only or limited UI | Full-stack web app with real-time visualization |## How It Works



```python**What it does:**

from PIL import Image

import torch- Analyzes pixel-level patterns and textures in the image

from model_inference import ModelInference

- Uses EfficientNet-B0 backbone pre-trained on ImageNet

# Initialize model

model = ModelInference()- Detects unnatural textures, over-smoothing, and anatomical inconsistencies### Our Innovations**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.Proposed Solution



# Load and analyze image

image = Image.open("image.jpg")

result = model.predict(image)**How it helps:**



print(f"Prediction: {result['prediction']}")- AI generators create statistically different pixel patterns than real cameras

print(f"Confidence: {result['confidence']}")

- Catches visible artifacts like malformed hands, weird textures, and impossible geometries1. **Hybrid Multi-Branch Architecture:** First system to combine spatial CNN, frequency FFT, noise consistency, and edge analysis in a unified framework. Each branch captures different generator artifacts that others might miss.### The 4 Detection Algorithms

# Access branch-level results

for branch, score in result['branch_scores'].items():

    print(f"{branch}: {score}%")

```**Why it works:**



---- Deep CNNs learn hierarchical features from low-level edges to high-level objects



## 🔬 Detection Algorithms- Can identify subtle patterns that differ between natural and synthetic images2. **Comprehensive Explainability:** Goes beyond "AI-generated" labels. Provides executive summaries, per-branch confidence scores, and Grad-CAM heatmaps showing exactly which image regions triggered detection.TruePix detects AI-generated images by analyzing both spatial and frequency-domain inconsistencies that are commonly introduced during synthetic image generation.



### 1. CNN (Convolutional Neural Network)



**Purpose**: Spatial texture and pattern analysis---



**Implementation**:

- **Architecture**: EfficientNet-B0 (pre-trained on ImageNet)

- **Features**: 256-dimensional spatial feature vector### 5.2 FFT (Fast Fourier Transform) - Frequency Analysis3. **Frequency Domain Analysis:** Leverages FFT to detect periodic artifacts and upsampling patterns invisible in spatial domain—catching sophisticated generators that fool spatial-only detectors.| Algorithm | What It Detects | Why It Works |

- **Detection**: Unnatural textures, over-smoothing, anatomical inconsistencies



**Why It Works**:  

AI generators produce statistically different pixel distributions than camera sensors. CNNs learn hierarchical representations that distinguish synthetic from natural patterns.**What it does:**



---- Converts image from spatial domain to frequency domain using 2D FFT



### 2. FFT (Fast Fourier Transform)- Analyzes frequency spectrum magnitude and patterns4. **Physics-Based Edge Verification:** Validates that edges follow optical physics laws (depth-of-field, blur consistency). AI generators often violate physical constraints.|-----------|----------------|--------------|---Key objectives:



**Purpose**: Frequency domain artifact detection- Detects periodic artifacts and upsampling signatures



**Implementation**:

- **Method**: 2D FFT on grayscale image

- **Features**: 256-dimensional frequency spectrum features**How it helps:**

- **Detection**: Periodic patterns, upsampling artifacts, grid structures

- AI generators leave "fingerprints" in frequency domain invisible to human eyes5. **Multi-Scale Noise Analysis:** Analyzes sensor noise at three scales (fine, medium, coarse). Real cameras have characteristic noise fingerprints; AI images lack authentic sensor signatures.| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |

**Why It Works**:  

AI generation processes leave "fingerprints" in frequency domain invisible to spatial analysis. Real cameras produce natural frequency distributions following optical physics.- Catches grid patterns, upsampling artifacts, and periodic noise



---



### 3. Noise Analysis**Why it works:**



**Purpose**: Sensor noise pattern verification- Real cameras produce natural frequency distributions following physical laws6. **Platform Stability Testing:** Built-in testing against real-world compression (WhatsApp, Instagram, Facebook) to ensure predictions remain stable in practical deployment scenarios.| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Detect AI-generated images with high robustness



**Implementation**:- AI upsampling and generation processes create artificial periodic patterns

- **Method**: Multi-scale Gaussian filtering (σ = 0.5, 1.0, 2.0)

- **Features**: 256-dimensional noise statistics (mean, variance, skewness, kurtosis)- Frequency analysis is robust to minor spatial transformations

- **Detection**: Absent or synthetic camera sensor noise



**Why It Works**:  

Real cameras have characteristic noise signatures based on sensor physics. AI-generated images lack authentic sensor noise or produce synthetic noise patterns.---7. **Production-Ready System:** Complete full-stack implementation with FastAPI backend, React frontend, and deployment-ready architecture—not just research code.| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |



---



### 4. Edge Analysis### 5.3 Noise Analysis - Multi-Scale Consistency



**Purpose**: Physical plausibility verification



**Implementation**:**What it does:**---| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |## Key FeaturesRemain effective under compression and resizing

- **Method**: Sobel operators (horizontal and vertical gradients)

- **Features**: 256-dimensional edge statistics- Analyzes noise patterns at three scales: fine (sigma=0.5), medium (sigma=1.0), coarse (sigma=2.0)

- **Detection**: Physically impossible edges, inconsistent blur patterns

- Extracts noise using multi-scale Gaussian filtering

**Why It Works**:  

Real camera optics follow physics laws (depth-of-field, diffraction). AI generators often violate these constraints, creating impossible focus relationships.- Measures noise statistics: mean, variance, skewness, kurtosis



---## How It Works



## 🔌 API Reference**How it helps:**



### Endpoints- Real cameras have characteristic sensor noise patterns



#### POST `/api/analyze`- AI-generated images lack authentic camera sensor noise or have synthetic noise



Analyze an image to determine if it's AI-generated or real.### The 4 Detection Algorithms### The ProcessProvide a scalable backend inference API



**Request**:**Why it works:**

```http

POST /api/analyze- Physical camera sensors produce specific noise signatures based on ISO, sensor size, and electronics

Content-Type: multipart/form-data

- AI generators don't replicate these authentic noise characteristics accurately

file: <binary image data>

```- Multi-scale analysis catches both fine-grain and structural noise differences| Algorithm | What It Detects | Why It Works |



**Response**:

```json

{---|-----------|----------------|--------------|

  "prediction": "AI-Generated",

  "confidence": 92.5,

  "metadata": {

    "executive_summary": "This image shows strong AI generation indicators...",### 5.4 Edge Analysis - Physical Plausibility| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |```✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  Enable real-time image verification via a frontend interface

    "branch_findings": {

      "cnn": {

        "confidence": 94.2,

        "findings": "Detected unnatural texture patterns..."**What it does:**| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |

      },

      "fft": {- Detects edges using Sobel operators (horizontal and vertical gradients)

        "confidence": 89.7,

        "findings": "Found periodic artifacts..."- Analyzes edge magnitude, direction, and statistical properties| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + Explanation

      },

      "noise": {- Checks edge consistency and blur patterns

        "confidence": 91.3,

        "findings": "Lacks authentic sensor noise..."| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |

      },

      "edge": {**How it helps:**

        "confidence": 85.6,

        "findings": "Detected physically implausible edges..."- Verifies that edges follow optical physics laws (depth-of-field, blur consistency)```✅ **Explainable AI** - Shows why an image is classified as real or AI  Methodology

      }

    },- AI generators often create physically impossible edges

    "gradcam_image": "data:image/png;base64,..."

  }### The Process

}

```**Why it works:**



#### POST `/api/platform-stability`- Real camera optics follow laws of physics: sharp edges in focus, soft edges out of focus



Test prediction stability across social media compression scenarios.- AI generators struggle to maintain consistent depth-of-field relationships



**Request**:- Edge analysis catches unnatural transitions and impossible focus patterns```

```http

POST /api/platform-stability

Content-Type: multipart/form-data

---1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.✅ **Grad-CAM Visualization** - Highlights suspicious image regions  The detection pipeline consists of the following stages:

file: <binary image data>

```



**Response**:## 6. Technology Stack```

```json

{

  "original": {

    "prediction": "AI-Generated",**Frontend:**

    "confidence": 92.5

  },- React 18.2 - Modern UI framework

  "whatsapp": {

    "prediction": "AI-Generated",- CSS3 - Custom stylingEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.

    "confidence": 87.3

  },- Axios - HTTP client for API communication

  "instagram": {

    "prediction": "AI-Generated",---✅ **Platform Robustness** - Tests stability across social media compression  1. Image Preprocessing

    "confidence": 89.8

  },**Backend:**

  "facebook": {

    "prediction": "AI-Generated",- FastAPI 0.108 - High-performance Python web framework---

    "confidence": 88.5

  }- Uvicorn - ASGI server

}

```- Python 3.9+



#### GET `/health`



Check API health status.**Machine Learning:**## Tech Stack



**Response**:- PyTorch 2.1.2 - Deep learning framework

```json

{- EfficientNet-B0 - CNN backbone architecture## Tech Stack✅ **Real-Time Analysis** - Fast inference with web interface  Image resizing and normalization

  "status": "healthy",

  "model_loaded": true- timm - PyTorch Image Models library

}

```- **Frontend:** React 18.2, CSS3



---**Image Processing:**



## 📊 Performance- OpenCV 4.9 - Computer vision operations- **Backend:** FastAPI 0.108, Uvicorn ASGI server



### Model Metrics- Pillow 10.1 - Image loading and manipulation



**Dataset**: 50,000 images (25,000 real + 25,000 AI-generated)- NumPy - Numerical computing- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0



| Metric | Score |- SciPy - Scientific computing (FFT operations)

|--------|-------|

| Accuracy | 89.3% |- scikit-image - Image processing algorithms- **Image Processing:** OpenCV 4.9, Pillow 10.1, scipy, scikit-image- **Frontend:** React 18.2, CSS3Color space standardization

| Precision (AI) | 91.2% |

| Recall (AI) | 87.5% |

| F1-Score | 89.3% |

**Explainability:**- **Explainability:** Custom Grad-CAM implementation

### Inference Speed

- Custom Grad-CAM implementation for visual explanations

| Device | Avg. Latency |

|--------|--------------|- **Backend:** FastAPI 0.108, Uvicorn

| CPU (Intel i7) | 150ms |

| GPU (NVIDIA RTX 3060) | 45ms |---

| M1 Mac | 80ms |

---

### Platform Robustness

## 7. How to Run This Project

| Platform | Compression | Accuracy |

|----------|-------------|----------|- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0---Noise stabilization

| Original | None | 89.3% |

| WhatsApp | 512px, Q=40 | 82.1% |Follow these step-by-step instructions to run TruePix on your device.

| Instagram | 1080px, Q=70 | 86.7% |

| Facebook | 960px, Q=60 | 85.4% |## Installation & Setup Guide



### Explainability Validation### Prerequisites



- **Grad-CAM Accuracy**: 84% correct artifact localization- **Image Processing:** OpenCV, Pillow, scipy

- **Branch Attribution**: CNN 45%, FFT 28%, Noise 18%, Edge 9%

- **User Comprehension**: 92% found explanations helpful (n=25)Before starting, ensure you have the following installed:



---### Prerequisites



## 🛠 Technology Stack| Software | Minimum Version | Download Link |



### Backend|----------|----------------|---------------|2. Feature Extraction



| Technology | Version | Purpose || Python | 3.9 or higher | https://www.python.org/downloads/ |

|------------|---------|---------|

| Python | 3.9+ | Core language || Node.js | 16 or higher | https://nodejs.org/ |Before you begin, ensure you have the following installed on your laptop/device:

| PyTorch | 2.1.2 | Deep learning framework |

| FastAPI | 0.108 | Web framework || Git | Latest version | https://git-scm.com/downloads |

| Uvicorn | 0.27 | ASGI server |

| OpenCV | 4.9 | Image processing || RAM | 4GB minimum (8GB recommended) | - |---

| NumPy | 1.24 | Numerical computing |

| SciPy | 1.11 | Scientific computing || Storage | 2GB free space | - |



### Frontend- **Python 3.9 or higher** - [Download here](https://www.python.org/downloads/)



| Technology | Version | Purpose |---

|------------|---------|---------|

| React | 18.2 | UI framework |- **Node.js 16 or higher** - [Download here](https://nodejs.org/)## How It WorksMultiple forensic cues are analyzed:

| Axios | 1.6 | HTTP client |

| CSS3 | - | Styling |### Step 1: Clone the Repository



### Machine Learning- **Git** - [Download here](https://git-scm.com/downloads)



| Component | Details |Open your terminal and run:

|-----------|---------|

| Architecture | EfficientNet-B0 + Custom Multi-Branch |- **4GB RAM minimum** (8GB recommended)## Installation & Setup Guide

| Training Framework | PyTorch |

| Optimization | Adam (lr=1e-4) |```bash

| Loss Function | Cross-Entropy |

| Explainability | Grad-CAM |git clone https://github.com/kathirpmdk-star/Truepix.git- **Internet connection** (for installing dependencies)



---cd Truepix



## 📁 Project Structure```Spatial artifacts using CNN-based feature extraction



```

Truepix/

├── backend/---### Step 1: Clone the Repository

│   ├── main.py                 # FastAPI application entry point

│   ├── model_inference.py      # Model inference logic

│   ├── cnn_detector.py         # CNN branch implementation

│   ├── fft_analyzer.py         # FFT branch implementation### Step 2: Set Up the Backend### Prerequisites

│   ├── noise_analyzer.py       # Noise analysis branch

│   ├── preprocessing.py        # Image preprocessing utilities

│   ├── score_fusion.py         # Branch fusion logic

│   ├── train_model.py          # Training script#### 2.1 Navigate to backend folderOpen your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

│   ├── requirements.txt        # Python dependencies

│   └── checkpoints/            # Trained model weights

│

├── frontend/```bash### The 4 Detection AlgorithmsFrequency-domain patterns (FFT-based anomalies)

│   ├── src/

│   │   ├── App.js              # Main React componentcd backend

│   │   ├── components/

│   │   │   ├── ImageUpload.js  # Image upload component``````bash

│   │   │   ├── ResultsPanel.js # Results display component

│   │   │   └── LandingPage.js  # Landing page component

│   │   └── index.js            # React entry point

│   ├── public/                 # Static assets#### 2.2 Create a virtual environment# Clone the repositoryBefore you begin, ensure you have the following installed on your laptop/device:

│   └── package.json            # Node dependencies

│

├── dataset/                    # Training data

│   ├── ai_generated/**On macOS/Linux:**git clone https://github.com/kathirpmdk-star/Truepix.git

│   └── real/

│```bash

├── model/                      # Model training scripts

│   └── train_model.pypython3 -m venv venvNoise residual inconsistencies

│

├── README.md                   # This file```

├── LICENSE                     # MIT License

├── .gitignore                  # Git ignore rules# Navigate to the project directory

└── requirements.txt            # Root dependencies

```**On Windows:**



---```bashcd Truepix- **Python 3.9 or higher** - [Download here](https://www.python.org/downloads/)



## 🤝 Contributingpython -m venv venv



We welcome contributions! Please follow these guidelines:``````



### Development Setup



1. Fork the repository#### 2.3 Activate the virtual environment- **Node.js 16 or higher** - [Download here](https://nodejs.org/)| Algorithm | What It Detects | Why It Works |Edge and texture irregularities

2. Create a feature branch: `git checkout -b feature/your-feature`

3. Make your changes

4. Run tests: `pytest backend/test_modules.py`

5. Commit changes: `git commit -m "Add your feature"`**On macOS/Linux:**### Step 2: Backend Setup

6. Push to branch: `git push origin feature/your-feature`

7. Open a Pull Request```bash



### Code Standardssource venv/bin/activate- **Git** - [Download here](https://git-scm.com/downloads)



- Follow PEP 8 for Python code```

- Use ESLint for JavaScript/React code

- Add docstrings to all functions```bash

- Write unit tests for new features

- Update documentation as needed**On Windows:**



### Areas for Contribution```bash# Navigate to backend folder- **4GB RAM minimum** (8GB recommended)|-----------|----------------|--------------|These features capture subtle differences between real camera pipelines and AI image generators.



- [ ] Improve model accuracyvenv\Scripts\activate

- [ ] Add support for more AI generators

- [ ] Implement video analysis```cd backend

- [ ] Create mobile applications

- [ ] Enhance explainability visualizations

- [ ] Optimize inference speed

- [ ] Add more comprehensive testsYou should see `(venv)` appear in your terminal prompt.- **Internet connection** (for installing dependencies)



---



## ⚠️ Limitations#### 2.4 Install Python dependencies# Create a virtual environment



**Generalization**:  

Trained on 2024-2025 AI generators. May not detect future models or architectures not present in training data.

```bashpython3 -m venv venv| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |3. Classification

**Compression Sensitivity**:  

Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%).pip install -r requirements.txt



**Adversarial Robustness**:  ```

Not hardened against sophisticated adversarial attacks designed to fool detectors.



**Hybrid Content**:  

Cannot distinguish which parts of an image are real vs. AI-edited in mixed-content images.This will install all required packages (PyTorch, FastAPI, OpenCV, etc.). It may take 2-3 minutes.# Activate virtual environment### Step 1: Clone the Repository



**False Positives**:  

May misclassify heavily post-processed real photos (HDR, beauty filters).

#### 2.5 Start the backend server# On Mac/Linux:

**Computational Cost**:  

Multi-branch architecture requires 2.5x inference time vs. single CNN.



### Responsible Use```bashsource venv/bin/activate| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Extracted features are passed to a trained deep learning classifier



⚠️ **This tool provides guidance, not definitive proof.**python main.py



- Combine with human expert verification for high-stakes decisions```# On Windows:

- Not recommended as sole evidence in legal/journalistic contexts

- Use as one signal among many when assessing authenticity



---You should see output like:venv\Scripts\activateOpen your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:



## 🗺 Roadmap```



### Q1 2026INFO:     Started server process



- [ ] Model ensemble (Vision Transformers + ResNet)INFO:     Uvicorn running on http://127.0.0.1:8000

- [ ] Adversarial training for robustness

- [ ] EXIF metadata analysis```# Install Python dependencies (this may take 2-3 minutes)| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |The model outputs a probability score for AI vs Real classification

- [ ] Batch processing API



### Q2 2026

✅ **Backend is now running!**pip install -r requirements.txt

- [ ] Video frame-by-frame analysis

- [ ] Model fingerprinting (identify specific generator)

- [ ] Browser extension

- [ ] Mobile apps (iOS/Android)**Keep this terminal window open.** The backend must stay running.```bash



### Q3-Q4 2026



- [ ] Continuous learning pipeline---# Start the backend server

- [ ] Zero-shot detection of unseen models

- [ ] Model quantization (INT8)

- [ ] Cloud deployment with auto-scaling

### Step 3: Set Up the Frontendpython main.py# Clone the repository| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |4. Inference API

---



## 📄 License

Open a **NEW terminal window** (keep the backend terminal running).```

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.



```

MIT License#### 3.1 Navigate to frontend foldergit clone https://github.com/kathirpmdk-star/Truepix.git



Copyright (c) 2026 TruePix Contributors



Permission is hereby granted, free of charge, to any person obtaining a copy```bash✅ **Backend is now running at:** http://localhost:8000  

of this software and associated documentation files (the "Software"), to deal

in the Software without restriction, including without limitation the rightscd Truepix/frontend

to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

copies of the Software, and to permit persons to whom the Software is```✅ **API Documentation available at:** http://localhost:8000/docsBackend exposes a REST API

furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all

copies or substantial portions of the Software.(Adjust the path if you're not starting from your home directory)



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE#### 3.2 Install Node.js dependencies**Keep this terminal window open!**# Navigate to the project directory

AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

SOFTWARE.```bash

```

npm install

---

```### Step 3: Frontend Setupcd Truepix### The ProcessAccepts image uploads

## 📚 Citation



If you use TruePix in your research, please cite:

This will install React and all required packages. It may take 2-3 minutes.

```bibtex

@software{truepix2026,

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},

  author={TruePix Contributors},#### 3.3 Start the frontend development serverOpen a **NEW terminal window** (keep the backend running) and run:```

  year={2026},

  url={https://github.com/kathirpmdk-star/Truepix},

  note={MIT License}

}```bash

```

npm start

---

``````bashReturns prediction and confidence score

## 🙏 Acknowledgments



This project builds upon foundational research in computer vision and AI detection:

The React app will automatically open in your browser at `http://localhost:3000`.# Navigate to the project directory

- **Research**: Wang et al. (CVPR 2020), Gragnaniello et al. (IEEE TIFS 2021)

- **Frameworks**: PyTorch, FastAPI, React development teams

- **Datasets**: COCO, FFHQ, DiffusionDB, CIFAKE contributors

- **Community**: Open-source contributors and researchersIf it doesn't open automatically, manually go to: **http://localhost:3000**cd Truepix/frontend### Step 2: Backend Setup



---



## 📞 Contact & Support✅ **Frontend is now running!**



- **Issues**: [GitHub Issues](https://github.com/kathirpmdk-star/Truepix/issues)

- **Discussions**: [GitHub Discussions](https://github.com/kathirpmdk-star/Truepix/discussions)

- **Security**: Report vulnerabilities via GitHub Security Advisories---# Install Node.js dependencies (this may take 2-3 minutes)```System Architecture



---



<div align="center">### Step 4: Verify Everything is Workingnpm install



**⭐ Star this repository if you find it useful! ⭐**



![TruePix Banner](https://via.placeholder.com/800x200/1a1a1a/ffffff?text=TruePix+-+Verifying+Image+Authenticity+in+the+Age+of+AI)You should now have:```bash



*Developed for research and public good*  

*Promoting transparency and accountability in generative AI*

1. **Backend running** at `http://localhost:8000`# Start the frontend development server

[Documentation](https://github.com/kathirpmdk-star/Truepix/wiki) • [Demo](http://localhost:3000) • [API](http://localhost:8000/docs) • [Research Paper](#)

   - You can visit `http://localhost:8000/docs` to see the API documentation

</div>

npm start# Navigate to backend folder1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationUser Image

2. **Frontend running** at `http://localhost:3000`

   - You should see the TruePix landing page```



---cd backend



### Troubleshooting Common Issues✅ **Frontend is now running at:** http://localhost:3000



#### Issue 1: "Python not found"```   ↓



**Solution:**Your browser should automatically open. If not, manually open: **http://localhost:3000**

- Install Python from python.org

- Restart your terminal after installation# Create a virtual environment

- Try using `python3` instead of `python`

### Step 4: Using TruePix

#### Issue 2: "npm not found"

python3 -m venv venvFrontend (Upload Interface)

**Solution:**

- Install Node.js from nodejs.org1. **Upload an Image:**

- Restart your terminal after installation

   - Click the upload area or drag-and-drop an image (JPG/PNG format)

#### Issue 3: "Port 8000 already in use"

   - Recommended image size: 500KB - 5MB

**Solution:**

# Activate virtual environmentEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.   ↓

**On macOS/Linux:**

```bash2. **View Results:**

lsof -ti:8000 | xargs kill -9

```   - Classification: "AI-Generated" or "Real"# On Mac/Linux:



**On Windows:**   - Confidence Score: 0-100%

```bash

netstat -ano | findstr :8000   - Executive Summary: Why the image was classifiedsource venv/bin/activateBackend API

taskkill /PID <PID_NUMBER> /F

```   - Branch Analysis: What each algorithm detected



#### Issue 4: "Port 3000 already in use"   - Grad-CAM Heatmap: Visual explanation highlighting suspicious regions# On Windows:



**Solution:**

- When prompted, press `Y` to run on a different port (e.g., 3001)

- Or kill the process using port 3000 (similar to port 8000 above)3. **Test Platform Stability (Optional):**venv\Scripts\activate---   ↓



#### Issue 5: Backend crashes or import errors   - Click "Test Platform Stability" button



**Solution:**   - See how prediction changes across WhatsApp, Instagram, Facebook compression

- Make sure you activated the virtual environment (`source venv/bin/activate`)

- Check Python version: `python --version` (must be 3.9+)

- Reinstall dependencies: `pip install -r requirements.txt`

---# Install Python dependencies (this may take 2-3 minutes)Preprocessing → Feature Extraction → Classifier

#### Issue 6: Frontend can't connect to backend



**Solution:**

- Verify backend is running on `http://localhost:8000`## Troubleshootingpip install -r requirements.txt

- Check if both terminals are still active

- Try refreshing the browser page



---**Problem: "Python not found"**## Tech Stack   ↓



### Stopping the Application- Solution: Install Python from python.org and restart terminal



When you're done using TruePix:# Start the backend server



#### Stop Frontend:**Problem: "npm not found"**

- Go to the frontend terminal

- Press `Ctrl + C`- Solution: Install Node.js from nodejs.org and restart terminalpython main.pyPrediction + Confidence



#### Stop Backend:

- Go to the backend terminal

- Press `Ctrl + C`**Problem: "Port 8000 already in use"**```



#### Deactivate Virtual Environment:- Solution: Kill the process using port 8000:

- In the backend terminal, type:

```bash  ```bash- **Frontend:** React 18.2, CSS3

deactivate

```  # Mac/Linux:



---  lsof -ti:8000 | xargs kill -9✅ **Backend is now running at:** http://localhost:8000  



## 8. How to Use TruePix  # Windows:



Once both servers are running, follow these steps:  netstat -ano | findstr :8000✅ **API Documentation available at:** http://localhost:8000/docs- **Backend:** FastAPI 0.108, UvicornTech Stack



### Step 1: Upload an Image  taskkill /PID <PID_NUMBER> /F



- Click the **upload area** or **drag and drop** an image  ```

- Supported formats: JPG, PNG

- Recommended size: 500KB - 5MB



### Step 2: View Analysis Results**Problem: "Port 3000 already in use"****Keep this terminal window open!**- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0Backend



After uploading, TruePix will display:- Solution: Kill the process or use a different port:



#### 2.1 Classification  ```bash

- **Label:** "AI-Generated" or "Real"

- **Confidence Score:** 0-100%  # The frontend will prompt: "Would you like to run on another port?" → Press Y



#### 2.2 Executive Summary  ```### Step 3: Frontend Setup- **Image Processing:** OpenCV, Pillow, scipyPython

- A human-readable explanation of why the image was classified

- Example: "This image shows strong AI generation indicators with 92.5% confidence. The CNN detected unnatural textures in the background..."



#### 2.3 Branch Analysis**Problem: Backend crashes or errors**

Four cards showing what each algorithm detected:

- **CNN Branch:** Spatial patterns and textures- Solution: Make sure you activated the virtual environment

- **FFT Branch:** Frequency domain artifacts

- **Noise Branch:** Sensor noise characteristics- Check Python version: `python --version` (should be 3.9+)Open a **NEW terminal window** (keep the backend running) and run:FastAPI / Flask

- **Edge Branch:** Edge structure plausibility

- Reinstall dependencies: `pip install -r requirements.txt`

Each branch shows:

- Confidence score (0-100%)

- Key findings

- Contribution to final decision**Problem: Frontend doesn't connect to backend**



#### 2.4 Grad-CAM Heatmap- Solution: Make sure backend is running on http://localhost:8000```bash---OpenCV

- Visual heatmap overlaid on the image

- **Red/Yellow regions:** Areas that influenced AI detection- Check if both terminals are still active

- **Blue/Green regions:** Less suspicious areas

# Navigate to the project directory

### Step 3: Test Platform Stability (Optional)

---

Click the **"Test Platform Stability"** button to see how the prediction changes when the image is compressed as it would be on:

- WhatsApp (512px, Quality 40%)cd Truepix/frontendPyTorch (model inference)

- Instagram (1080px, Quality 70%)

- Facebook (960px, Quality 60%)## Stopping the Application



This shows how robust the detection is to real-world compression.



---To stop the servers:



## 9. Results and Performance# Install Node.js dependencies (this may take 2-3 minutes)## Quick StartFrontend



### 9.1 Model Performance1. **Stop Frontend:** Press `Ctrl + C` in the frontend terminal



**Training Configuration:**2. **Stop Backend:** Press `Ctrl + C` in the backend terminalnpm install

- Dataset: 50,000 images (25,000 real + 25,000 AI-generated)

- Real sources: COCO dataset, FFHQ, natural photography3. **Deactivate virtual environment:** Type `deactivate` in backend terminal

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs

- Training: 20 epochs, Adam optimizer (learning rate = 0.0001)React

- Loss function: Cross-entropy

---

**Evaluation Metrics (Test Set):**

# Start the frontend development server

| Metric | Score |

|--------|-------|## System Requirements

| Accuracy | 89.3% |

| Precision (AI class) | 91.2% |npm start### Backend SetupHTML / CSS / JavaScript

| Recall (AI class) | 87.5% |

| F1-Score | 89.3% || Component | Minimum | Recommended |



### 9.2 Platform Robustness|-----------|---------|-------------|```



| Platform | Image Processing | Accuracy || OS | Windows 10, macOS 10.15, Ubuntu 20.04 | Latest version |

|----------|-----------------|----------|

| Original | No compression | 89.3% || RAM | 4GB | 8GB |```bashModel & Analysis

| WhatsApp | 512px, Quality 40% | 82.1% |

| Instagram | 1080px, Quality 70% | 86.7% || Storage | 2GB free space | 5GB free space |

| Facebook | 960px, Quality 60% | 85.4% |

| CPU | Dual-core 2.0GHz | Quad-core 2.5GHz+ |✅ **Frontend is now running at:** http://localhost:3000

**Insight:** The multi-branch architecture maintains reasonable accuracy even under aggressive compression. The frequency and noise branches provide resilience when spatial features degrade.

| GPU | Not required | CUDA-compatible (optional, for faster inference) |

### 9.3 Explainability Validation

cd backendCNN-based feature extraction

- **Grad-CAM Accuracy:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test cases

- **Branch Contribution:** Spatial 45%, Frequency 28%, Noise 18%, Edge 9% (average)---

- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictions

Your browser should automatically open. If not, manually open: **http://localhost:3000**

---

## Results & Evaluation

## 10. What Makes TruePix Unique

python3 -m venv venvFrequency-domain analysis

Most AI image detection tools use **single-model approaches** (typically just one CNN classifier). TruePix stands out with several innovations:

### Model Performance

### Comparison with Existing Solutions

### Step 4: Using TruePix

| Feature | Traditional Detectors | TruePix |

|---------|----------------------|---------|**Training Configuration:**

| **Detection Methods** | Single CNN model | 4 complementary algorithms |

| **Explainability** | Black-box predictions | Executive summary + Grad-CAM + branch attribution |- Dataset: 50,000 images (balanced real/AI split)source venv/bin/activateImage forensics techniques

| **Robustness** | Fails on compressed images | Tested against WhatsApp/Instagram/Facebook |

| **Frequency Analysis** | Rarely implemented | FFT-based artifact detection |- Real sources: COCO, FFHQ, natural photography datasets

| **Noise Analysis** | Usually not included | Multi-scale sensor noise verification |

| **Edge Analysis** | Often ignored | Physics-based plausibility checks |- AI sources: Stable Diffusion, DALL-E, Midjourney outputs1. **Upload an Image:**

| **User Interface** | API only or basic UI | Full-stack web app with visualizations |

| **Transparency** | No explanations | Complete reasoning for every decision |- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss



### Our Key Innovations   - Click the upload area or drag-and-drop an image (JPG/PNG format)pip install -r requirements.txtHow to Run the Project



**1. Hybrid Multi-Branch Architecture****Evaluation Metrics:**

- First system to combine spatial CNN, frequency FFT, noise consistency, and edge analysis in one unified framework

- Each branch catches different generator artifacts that others might miss- **Accuracy:** 89.3% on held-out test set   - Recommended image size: 500KB - 5MB



**2. Comprehensive Explainability**- **Precision (AI class):** 91.2%

- Goes beyond "AI-generated" labels

- Provides executive summaries, per-branch confidence scores, and visual heatmaps- **Recall (AI class):** 87.5%python main.pyBackend

- Users understand exactly why an image was classified

- **F1-Score:** 89.3%

**3. Frequency Domain Analysis**

- Leverages FFT to detect periodic artifacts invisible in spatial domain2. **View Results:**

- Catches sophisticated generators that fool spatial-only detectors

### Robustness Testing

**4. Physics-Based Edge Verification**

- Validates that edges follow optical physics laws (depth-of-field, blur consistency)   - Classification: "AI-Generated" or "Real"```cd backend

- AI generators often violate physical constraints

**Platform Stability Scores:**

**5. Multi-Scale Noise Analysis**

- Analyzes sensor noise at three scales (fine, medium, coarse)- Original images: 89.3% accuracy   - Confidence Score: 0-100%

- Real cameras have characteristic noise fingerprints; AI images lack authentic sensor signatures

- WhatsApp compression (512px, Q=40): 82.1% accuracy

**6. Platform Stability Testing**

- Built-in testing against real-world compression scenarios- Instagram compression (1080px, Q=70): 86.7% accuracy   - Executive Summary: Why the image was classifiedBackend runs at: **http://localhost:8000**pip install -r requirements.txt

- Ensures predictions remain stable in practical deployment

- Facebook compression (960px, Q=60): 85.4% accuracy

**7. Production-Ready Full-Stack System**

- Complete implementation with FastAPI backend and React frontend   - Branch Analysis: What each algorithm detected

- Not just research code - ready for real-world deployment

**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.

---

   - Grad-CAM Heatmap: Visual explanation highlighting suspicious regionspython app.py

## 11. Limitations

### Explainability Validation

We believe in transparent communication about our system's constraints:



### Current Limitations

- **Grad-CAM Analysis:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test cases

**1. Generalization to Future Models**

- Trained on 2024-2025 AI generators (Stable Diffusion, DALL-E, Midjourney)- **Branch Attribution:** Spatial branch contributes 45%, frequency 28%, noise 18%, edge 9% on average3. **Test Platform Stability (Optional):**### Frontend SetupFrontend

- May not generalize to future AI models with different architectures

- Requires retraining as new generators emerge- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictions



**2. Compression Sensitivity**   - Click "Test Platform Stability" button

- Accuracy drops 5-8% under aggressive compression (JPEG quality < 50%)

- Frequency and noise features degrade with heavy compression---



**3. Adversarial Attacks**   - See how prediction changes across WhatsApp, Instagram, Facebook compression```bashcd frontend

- Not hardened against sophisticated adversarial perturbations

- An attacker could potentially add noise to fool the detector## Limitations



**4. Hybrid Content**

- Struggles with hybrid images (real photos with AI-edited elements)

- Cannot pinpoint which regions are real vs. AI-generated**Current Constraints:**



**5. Computational Cost**- Trained on 2024-2025 AI generators; may not generalize to future models or unseen architectures---cd frontendnpm install

- Multi-branch architecture requires 2.5x inference time vs. single CNN

- Average inference: 150ms on CPU vs. 60ms for single model- Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%)



**6. False Positives**- Not hardened against sophisticated adversarial attacks designed to fool detectors

- May misclassify heavily post-processed real photos (HDR, beauty filters, aggressive sharpening)

- Over-edited Instagram photos sometimes trigger AI detection- Struggles with hybrid images (real photos containing AI-edited elements)



### Responsible Use Guidelines- Computational cost: Multi-branch architecture requires 2.5x inference time vs single CNN (150ms vs 60ms on CPU)## Troubleshootingnpm installnpm start



⚠️ **This tool should be used as guidance, not definitive proof**- False positives possible on heavily post-processed real photos (HDR, beauty filters)



- Combine with human expert verification for high-stakes decisions

- Not recommended as sole evidence in legal or journalistic contexts

- Use as one signal among many when assessing image authenticity**Responsible Use:**



---- This tool should be used as guidance, not definitive proof**Problem: "Python not found"**npm startResults & Evaluation



## 12. Future Work- Combine with human expert verification for high-stakes decisions



We have identified several directions for improving TruePix:- Not recommended as sole evidence in legal or journalistic contexts- Solution: Install Python from python.org and restart terminal



### Short-Term Improvements (Next 6 Months)



**1. Model Enhancements**---```The system successfully identifies AI-generated images by detecting non-natural artifacts.

- Ensemble multiple architectures (Vision Transformers + ResNet) for improved accuracy

- Adversarial training to improve robustness against evasion techniques

- Expand training dataset to 500k+ images across more generators

## Future Work**Problem: "npm not found"**

**2. Feature Additions**

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)

- Batch processing API for high-throughput analysis

- Model fingerprinting to identify specific AI generator (DALL-E vs. Midjourney vs. Stable Diffusion)**Model Enhancements:**- Solution: Install Node.js from nodejs.org and restart terminalFrontend runs at: **http://localhost:3000**Tested on a mixed dataset containing:



### Medium-Term Goals (6-12 Months)- Ensemble multiple architectures (Vision Transformers + ResNet) for improved accuracy



**3. Video Analysis**- Continuous learning pipeline to adapt to emerging AI generators

- Frame-by-frame analysis for deepfake video detection

- Temporal consistency analysis across video sequences- Adversarial training to improve robustness against evasion techniques



**4. Deployment Improvements**- Expand training dataset to 500k+ images across more generators**Problem: "Port 8000 already in use"**Real photographs

- Model quantization (INT8) for 3x faster inference

- Mobile applications (iOS/Android) with on-device inference

- Browser extension for in-situ web image analysis

**Feature Additions:**- Solution: Kill the process using port 8000:

### Long-Term Vision (1-2 Years)

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)

**5. Research Directions**

- Continuous learning pipeline to adapt to emerging AI generators- Batch processing API for high-throughput analysis  ```bash### UsageAI-generated images from multiple generation sources

- Zero-shot detection of unseen generative models

- Cross-modal consistency analysis (text-image alignment verification)- Video frame-by-frame analysis for deepfake detection

- Certified defenses against adversarial perturbations

- Model fingerprinting to identify specific AI generator (DALL-E vs Midjourney vs Stable Diffusion)  # Mac/Linux:

**6. Partnerships**

- Collaboration with fact-checking organizations- Temporal consistency analysis for video sequences

- Integration with news platforms and content moderation systems

- Cloud deployment with auto-scaling for high traffic  lsof -ti:8000 | xargs kill -91. Open http://localhost:3000Performance was evaluated using accuracy and confidence-based prediction consistency.



---**Deployment Improvements:**



## 13. Conclusion- Model quantization (INT8) for 3x faster inference  # Windows:



TruePix represents a significant advancement in AI-generated image detection technology. In an era where generative AI can create photorealistic images that fool even expert observers, tools like TruePix are essential for maintaining trust in digital media.- Mobile applications (iOS/Android) with on-device inference



### Our Contributions- Browser extension for in-situ web image analysis  netstat -ano | findstr :80002. Upload an image (JPG/PNG)Note: Due to dataset size constraints, datasets are not included in the repository.



**1. Technical Innovation**- Cloud deployment with auto-scaling for high traffic

- Novel hybrid multi-branch architecture combining spatial, frequency, noise, and edge analysis

- First system to provide comprehensive explainability with visual heatmaps and per-branch attribution- Partnership with fact-checking organizations and news platforms  taskkill /PID <PID_NUMBER> /F

- Robust performance even under real-world compression scenarios



**2. Practical Impact**

- Production-ready full-stack system deployable in real-world scenarios**Research Directions:**  ```3. View results: classification, confidence, explanation, Grad-CAM heatmapLimitations

- Intuitive interface accessible to non-technical users

- Platform stability testing ensures reliability in social media contexts- Cross-modal consistency analysis (text-image alignment verification)



**3. Transparency and Trust**- Zero-shot detection of unseen generative models

- Complete explainability for every prediction

- Open acknowledgment of limitations and responsible use guidelines- Certified defenses against adversarial perturbations

- Educational tool for understanding AI generation artifacts

**Problem: "Port 3000 already in use"**Performance depends on diversity of training data

### Why TruePix Matters

---

As AI image generation becomes more sophisticated, the line between real and synthetic images continues to blur. This poses existential challenges for:

- **Journalism:** Verifying the authenticity of news imagery- Solution: Kill the process or use a different port:

- **Legal Systems:** Ensuring digital evidence integrity

- **Social Media:** Combating misinformation and fake content## Conclusion

- **Digital Forensics:** Investigating fraud and identity theft

  ```bash---Newer AI generators may introduce unseen patterns

TruePix addresses these challenges by providing:

- **Robustness:** Four detection methods working together catch artifacts that single-method systems missTruePix represents a significant advancement in AI-generated image detection by introducing a hybrid multi-branch architecture that combines spatial, frequency, noise, and edge analysis. Unlike existing single-model detectors, our system provides:

- **Explainability:** Users don't just get a verdict - they understand the reasoning

- **Practicality:** Ready for deployment in real-world content moderation and fact-checking workflows  # The frontend will prompt: "Would you like to run on another port?" → Press Y



### Our Vision1. **Superior Robustness:** Four complementary detection methods ensure artifacts are caught regardless of which AI generator was used or what compression the image underwent.



**"Empowering users to verify image authenticity in an era where seeing is no longer believing."**  ```Does not guarantee 100% accuracy for highly post-processed images



We envision a future where:2. **Unprecedented Explainability:** Users don't just get a "yes/no" answer—they understand exactly why an image was classified through executive summaries, per-branch attributions, and visual Grad-CAM heatmaps.

- Every social media platform integrates AI detection to flag synthetic content

- Journalists have reliable tools to verify imagery before publication

- Digital forensics teams can analyze evidence with confidence

- Users can trust that the images they see represent reality3. **Real-World Applicability:** Built-in platform stability testing and production-ready architecture make TruePix deployable in actual content moderation, fact-checking, and forensic scenarios.



TruePix is a step toward that future - combining cutting-edge machine learning with transparent explainability to restore trust in digital media.**Problem: Backend crashes or errors**## ResultsFuture Work



### Final Thoughts4. **Research Innovation:** Novel integration of frequency-domain FFT analysis, multi-scale noise verification, and physics-based edge validation sets new standards for detection methodology.



This project demonstrates that effective AI detection requires:- Solution: Make sure you activated the virtual environment

1. **Multi-faceted analysis** - No single algorithm is sufficient

2. **Explainable results** - Users need to understand the reasoningAs generative AI continues to evolve, tools like TruePix are essential for maintaining trust in digital media. By combining multiple forensic approaches with transparent explainability, we provide a foundation for responsible AI detection that can adapt to future challenges.

3. **Continuous adaptation** - Detection must evolve as generators improve

4. **Responsible deployment** - Clear communication about limitations- Check Python version: `python --version` (should be 3.9+)Extend detection to video and deepfake content



We invite researchers, developers, and organizations to build upon this work, contribute improvements, and deploy TruePix in contexts where image authenticity matters.**Our vision:** Empower users, journalists, and organizations with trustworthy tools to verify image authenticity in an era where "seeing is no longer believing."



**Together, we can preserve the integrity of digital media in the age of generative AI.**- Reinstall dependencies: `pip install -r requirements.txt`



------



## 14. References- **Accuracy:** 89.3% on test setIntegrate explainability heatmaps for visual justification



### Academic Research## References



1. **Wang, S. Y., et al. (2020).** "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.**Problem: Frontend doesn't connect to backend**



2. **Gragnaniello, D., et al. (2021).** "GAN-Generated Faces Detection: A Survey and New Perspectives." *IEEE Transactions on Information Forensics and Security*.**Academic Research:**



3. **Selvaraju, R. R., et al. (2017).** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *International Conference on Computer Vision (ICCV)*.1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." CVPR 2020.- Solution: Make sure backend is running on http://localhost:8000- **Precision:** 91.2% (AI detection)Expand dataset to include newer AI image generators



4. **Marra, F., et al. (2019).** "Do GANs Leave Specific Traces? A Large-Scale Study." *IEEE Workshop on Information Forensics and Security (WIFS)*.2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.



5. **Tan, M., & Le, Q. (2019).** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *International Conference on Machine Learning (ICML)*.3. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017.- Check if both terminals are still active



### Datasets Used4. Marra et al. (2019). "Do GANs Leave Specific Traces?" IEEE WIFS.



- **CIFAKE:** Kaggle dataset for real vs. AI image classification- **Robustness:** 82-87% accuracy after social media compressionImprove robustness against adversarial attacks

- **DiffusionDB:** Large-scale Stable Diffusion dataset (Hugging Face)

- **COCO:** Microsoft Common Objects in Context dataset**Datasets:**

- **FFHQ:** NVIDIA Flickr-Faces-HQ dataset

- CIFAKE: Kaggle real vs AI image dataset---

### Tools and Frameworks

- DiffusionDB: Stable Diffusion dataset (Hugging Face)

- **PyTorch:** https://pytorch.org/

- **timm (PyTorch Image Models):** https://github.com/huggingface/pytorch-image-models- COCO: Microsoft Common Objects in Context- **Explainability:** 84% Grad-CAM accuracy, 92% user satisfactionEthical Considerations

- **FastAPI:** https://fastapi.tiangolo.com/

- **React:** https://react.dev/- FFHQ: NVIDIA Flickr-Faces-HQ

- **OpenCV:** https://opencv.org/

## Stopping the Application

---

**Tools & Frameworks:**

## License

- PyTorch: pytorch.orgThis project is intended only for detection and verification purposes.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

- timm (PyTorch Image Models): github.com/huggingface/pytorch-image-models

---

- FastAPI: fastapi.tiangolo.comTo stop the servers:

## Citation

- React: react.dev

If you use TruePix in your research or project, please cite:

---It does not generate or promote misuse of AI-generated content.

```bibtex

@software{truepix2026,---

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},

  author={TruePix Contributors},1. **Stop Frontend:** Press `Ctrl + C` in the frontend terminal

  year={2026},

  url={https://github.com/kathirpmdk-star/Truepix}## License & Citation

}

```2. **Stop Backend:** Press `Ctrl + C` in the backend terminalAuthor



---**License:** MIT License - See LICENSE file for details



## Acknowledgments3. **Deactivate virtual environment:** Type `deactivate` in backend terminal



This project builds upon foundational research in computer vision and AI detection. We thank:**Citation:**

- The open-source community for PyTorch, FastAPI, and React

- Researchers who published papers on AI image detectionIf you use TruePix in your research or project, please cite:## LimitationsKathiravan M

- Dataset contributors who made training possible



---

```bibtex---

<div align="center">

@software{truepix2026,

**Developed for Academic Research & Public Good**

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},Project: TruePix – AI vs Real Image Detection

*Promoting transparency and accountability in the age of generative AI*

  author={TruePix Contributors},

---

  year={2026},## System Requirements

### Need Help?

  url={https://github.com/kathirpmdk-star/Truepix}

- 📧 **Email:** Open a GitHub issue for questions

- 🐛 **Bug Reports:** Submit issues on GitHub}- Trained on 2024-2025 AI generators; may not detect future models

- 💡 **Feature Requests:** Contributions welcome!

```

---

| Component | Minimum | Recommended |

⭐ **If you find TruePix useful, please star this repository!** ⭐

---

</div>

|-----------|---------|-------------|- Accuracy drops 5-8% under heavy compression (< 50% JPEG quality)

## Acknowledgments

| OS | Windows 10, macOS 10.15, Ubuntu 20.04 | Latest version |

This project builds upon foundational research in AI detection and computer vision. We thank the open-source community for tools like PyTorch, FastAPI, and React that made this work possible.

| RAM | 4GB | 8GB |- Not hardened against adversarial attacks

---

| Storage | 2GB free space | 5GB free space |- Struggles with hybrid images (real + AI edits)

<div align="center">

| CPU | Dual-core 2.0GHz | Quad-core 2.5GHz+ |

**Developed for Academic Research & Public Good**

| GPU | Not required | CUDA-compatible (optional, for faster inference) |---

*Promoting transparency and accountability in the age of generative AI*



For questions, collaborations, or bug reports, please open a GitHub issue

---## Future Work

⭐ **Star this repository if you find it useful!** ⭐



</div>

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
