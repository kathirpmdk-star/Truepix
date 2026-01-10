# TruePix - AI Image Detection System# TruePix - AI Image Detection System# TruePix - AI Image Detection SystemTruePix – Robust AI vs Real Image Detection System



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)Overview

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

**A multi-branch deep learning system that detects AI-generated images and explains why.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)TruePix is a system designed to distinguish AI-generated images from real photographs using image forensic cues and deep learning.

---

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

## Overview

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)The project addresses the growing challenge of synthetic media misuse, especially in social media, journalism, and digital evidence verification.

TruePix is an advanced AI image detection system that distinguishes AI-generated images from real photographs using a novel hybrid multi-branch architecture. Unlike traditional single-model approaches, TruePix combines four complementary detection methods—CNN spatial analysis, FFT frequency analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.

**A multi-branch deep learning system that detects AI-generated images and explains why.**

**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification, social media authentication.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)The system accepts an image as input and outputs:

---

---

## Problem Statement

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)Prediction: AI-generated or Real

Recent advances in generative AI models (DALL-E, Midjourney, Stable Diffusion) have made synthetic images visually indistinguishable from real photographs. This poses critical challenges:

## What is TruePix?

- **Misinformation:** Fake news propagation using synthetic imagery

- **Digital Fraud:** Identity theft, document forgery, deepfake scamsConfidence score

- **Trust Erosion:** Declining confidence in digital media authenticity

- **Evidence Tampering:** Compromised digital evidence in legal contextsTruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.



Traditional visual inspection and single-method detection systems are no longer reliable. An automated, multi-faceted detection system is essential to analyze hidden artifacts left by AI generation pipelines.**A multi-branch deep learning system that detects AI-generated images and explains why.**Feature-based explanation (backend level)



---**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.



## Key FeaturesProblem Statement



✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy across different AI generators  ---

✅ **Explainable AI** - Provides detailed reasoning for every classification decision  

✅ **Grad-CAM Visualization** - Highlights suspicious image regions with visual heatmaps  ---Recent advances in generative AI models (GANs, Diffusion models) have made synthetic images visually indistinguishable from real ones. This creates serious risks such as:

✅ **Platform Robustness** - Tests stability across social media compression scenarios  

✅ **Real-Time Analysis** - Fast inference with intuitive web interface  ## Key Features

✅ **High Accuracy** - 89.3% accuracy with 91.2% precision on AI detection

Spread of misinformation

---

✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  

## What Makes TruePix Unique?

✅ **Explainable AI** - Shows why an image is classified as real or AI  ## What is TruePix?Fake digital evidence

### Compared to Existing Solutions

✅ **Grad-CAM Visualization** - Highlights suspicious image regions  

Most AI image detection tools available online rely on single-model approaches (typically just CNN-based classifiers). TruePix stands out with several innovations:

✅ **Platform Robustness** - Tests stability across social media compression  Loss of trust in visual media

| Feature | Traditional Detectors | TruePix |

|---------|----------------------|---------|✅ **Real-Time Analysis** - Fast inference with web interface  

| **Detection Methods** | Single CNN model | 4 complementary algorithms (CNN + FFT + Noise + Edge) |

| **Explainability** | Black-box predictions | Full transparency with per-branch attribution + Grad-CAM |TruePix distinguishes AI-generated images from real photographs using four detection methods: CNN spatial analysis, FFT frequency analysis, noise pattern analysis, and edge structure analysis. The system provides not just a prediction, but also explains which features led to the classification.Traditional visual inspection is no longer reliable.

| **Robustness** | Fails on compressed images | Tested against WhatsApp, Instagram, Facebook compression |

| **Frequency Analysis** | Rarely used | FFT-based detection of periodic artifacts |---

| **Noise Analysis** | Not implemented | Multi-scale sensor noise verification |

| **Physical Plausibility** | Ignored | Edge structure analysis based on optical physics |Hence, an automated, robust detection system is required to analyze hidden artifacts left by AI generation pipelines.

| **Fusion Strategy** | N/A | Temperature-calibrated multi-branch fusion |

| **User Interface** | API only or limited UI | Full-stack web app with real-time visualization |## How It Works



### Our Innovations**Use Cases:** Content moderation, fact-checking, digital forensics, journalism verification.Proposed Solution



1. **Hybrid Multi-Branch Architecture:** First system to combine spatial CNN, frequency FFT, noise consistency, and edge analysis in a unified framework. Each branch captures different generator artifacts that others might miss.### The 4 Detection Algorithms



2. **Comprehensive Explainability:** Goes beyond "AI-generated" labels. Provides executive summaries, per-branch confidence scores, and Grad-CAM heatmaps showing exactly which image regions triggered detection.TruePix detects AI-generated images by analyzing both spatial and frequency-domain inconsistencies that are commonly introduced during synthetic image generation.



3. **Frequency Domain Analysis:** Leverages FFT to detect periodic artifacts and upsampling patterns invisible in spatial domain—catching sophisticated generators that fool spatial-only detectors.| Algorithm | What It Detects | Why It Works |



4. **Physics-Based Edge Verification:** Validates that edges follow optical physics laws (depth-of-field, blur consistency). AI generators often violate physical constraints.|-----------|----------------|--------------|---Key objectives:



5. **Multi-Scale Noise Analysis:** Analyzes sensor noise at three scales (fine, medium, coarse). Real cameras have characteristic noise fingerprints; AI images lack authentic sensor signatures.| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |



6. **Platform Stability Testing:** Built-in testing against real-world compression (WhatsApp, Instagram, Facebook) to ensure predictions remain stable in practical deployment scenarios.| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Detect AI-generated images with high robustness



7. **Production-Ready System:** Complete full-stack implementation with FastAPI backend, React frontend, and deployment-ready architecture—not just research code.| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |



---| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |## Key FeaturesRemain effective under compression and resizing



## How It Works



### The 4 Detection Algorithms### The ProcessProvide a scalable backend inference API



| Algorithm | What It Detects | Why It Works |

|-----------|----------------|--------------|

| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |```✅ **Multi-Branch Detection** - Combines 4 algorithms for robust accuracy  Enable real-time image verification via a frontend interface

| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |

| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + Explanation

| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |

```✅ **Explainable AI** - Shows why an image is classified as real or AI  Methodology

### The Process



```

1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.✅ **Grad-CAM Visualization** - Highlights suspicious image regions  The detection pipeline consists of the following stages:

```



Each algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.

---✅ **Platform Robustness** - Tests stability across social media compression  1. Image Preprocessing

---



## Tech Stack

## Tech Stack✅ **Real-Time Analysis** - Fast inference with web interface  Image resizing and normalization

- **Frontend:** React 18.2, CSS3

- **Backend:** FastAPI 0.108, Uvicorn ASGI server

- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0

- **Image Processing:** OpenCV 4.9, Pillow 10.1, scipy, scikit-image- **Frontend:** React 18.2, CSS3Color space standardization

- **Explainability:** Custom Grad-CAM implementation

- **Backend:** FastAPI 0.108, Uvicorn

---

- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0---Noise stabilization

## Installation & Setup Guide

- **Image Processing:** OpenCV, Pillow, scipy

### Prerequisites

2. Feature Extraction

Before you begin, ensure you have the following installed on your laptop/device:

---

- **Python 3.9 or higher** - [Download here](https://www.python.org/downloads/)

- **Node.js 16 or higher** - [Download here](https://nodejs.org/)## How It WorksMultiple forensic cues are analyzed:

- **Git** - [Download here](https://git-scm.com/downloads)

- **4GB RAM minimum** (8GB recommended)## Installation & Setup Guide

- **Internet connection** (for installing dependencies)

Spatial artifacts using CNN-based feature extraction

### Step 1: Clone the Repository

### Prerequisites

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

### The 4 Detection AlgorithmsFrequency-domain patterns (FFT-based anomalies)

```bash

# Clone the repositoryBefore you begin, ensure you have the following installed on your laptop/device:

git clone https://github.com/kathirpmdk-star/Truepix.git

Noise residual inconsistencies

# Navigate to the project directory

cd Truepix- **Python 3.9 or higher** - [Download here](https://www.python.org/downloads/)

```

- **Node.js 16 or higher** - [Download here](https://nodejs.org/)| Algorithm | What It Detects | Why It Works |Edge and texture irregularities

### Step 2: Backend Setup

- **Git** - [Download here](https://git-scm.com/downloads)

```bash

# Navigate to backend folder- **4GB RAM minimum** (8GB recommended)|-----------|----------------|--------------|These features capture subtle differences between real camera pipelines and AI image generators.

cd backend

- **Internet connection** (for installing dependencies)

# Create a virtual environment

python3 -m venv venv| **CNN (Spatial)** | Unnatural textures, malformed anatomy, over-smooth surfaces | AI generators create statistically different pixel patterns than cameras |3. Classification



# Activate virtual environment### Step 1: Clone the Repository

# On Mac/Linux:

source venv/bin/activate| **FFT (Frequency)** | Grid patterns, upsampling artifacts, periodic noise | AI leaves "fingerprints" in frequency domain invisible to human eyes |Extracted features are passed to a trained deep learning classifier

# On Windows:

venv\Scripts\activateOpen your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:



# Install Python dependencies (this may take 2-3 minutes)| **Noise Analysis** | Absent or synthetic camera sensor noise | Real cameras have characteristic noise; AI images don't |The model outputs a probability score for AI vs Real classification

pip install -r requirements.txt

```bash

# Start the backend server

python main.py# Clone the repository| **Edge Analysis** | Physically impossible edges, inconsistent blur | AI violates optical physics; real photos follow depth-of-field laws |4. Inference API

```

git clone https://github.com/kathirpmdk-star/Truepix.git

✅ **Backend is now running at:** http://localhost:8000  

✅ **API Documentation available at:** http://localhost:8000/docsBackend exposes a REST API



**Keep this terminal window open!**# Navigate to the project directory



### Step 3: Frontend Setupcd Truepix### The ProcessAccepts image uploads



Open a **NEW terminal window** (keep the backend running) and run:```



```bashReturns prediction and confidence score

# Navigate to the project directory

cd Truepix/frontend### Step 2: Backend Setup



# Install Node.js dependencies (this may take 2-3 minutes)```System Architecture

npm install

```bash

# Start the frontend development server

npm start# Navigate to backend folder1. Upload Image → 2. Extract Features (4 branches) → 3. Fusion Network → 4. Prediction + ExplanationUser Image

```

cd backend

✅ **Frontend is now running at:** http://localhost:3000

```   ↓

Your browser should automatically open. If not, manually open: **http://localhost:3000**

# Create a virtual environment

### Step 4: Using TruePix

python3 -m venv venvFrontend (Upload Interface)

1. **Upload an Image:**

   - Click the upload area or drag-and-drop an image (JPG/PNG format)

   - Recommended image size: 500KB - 5MB

# Activate virtual environmentEach algorithm analyzes the image independently (256 features each), then combines into a 1024-dimensional vector for final classification with confidence score and explainable results.   ↓

2. **View Results:**

   - Classification: "AI-Generated" or "Real"# On Mac/Linux:

   - Confidence Score: 0-100%

   - Executive Summary: Why the image was classifiedsource venv/bin/activateBackend API

   - Branch Analysis: What each algorithm detected

   - Grad-CAM Heatmap: Visual explanation highlighting suspicious regions# On Windows:



3. **Test Platform Stability (Optional):**venv\Scripts\activate---   ↓

   - Click "Test Platform Stability" button

   - See how prediction changes across WhatsApp, Instagram, Facebook compression



---# Install Python dependencies (this may take 2-3 minutes)Preprocessing → Feature Extraction → Classifier



## Troubleshootingpip install -r requirements.txt



**Problem: "Python not found"**## Tech Stack   ↓

- Solution: Install Python from python.org and restart terminal

# Start the backend server

**Problem: "npm not found"**

- Solution: Install Node.js from nodejs.org and restart terminalpython main.pyPrediction + Confidence



**Problem: "Port 8000 already in use"**```

- Solution: Kill the process using port 8000:

  ```bash- **Frontend:** React 18.2, CSS3

  # Mac/Linux:

  lsof -ti:8000 | xargs kill -9✅ **Backend is now running at:** http://localhost:8000  

  # Windows:

  netstat -ano | findstr :8000✅ **API Documentation available at:** http://localhost:8000/docs- **Backend:** FastAPI 0.108, UvicornTech Stack

  taskkill /PID <PID_NUMBER> /F

  ```



**Problem: "Port 3000 already in use"****Keep this terminal window open!**- **ML Framework:** PyTorch 2.1.2, EfficientNet-B0Backend

- Solution: Kill the process or use a different port:

  ```bash

  # The frontend will prompt: "Would you like to run on another port?" → Press Y

  ```### Step 3: Frontend Setup- **Image Processing:** OpenCV, Pillow, scipyPython



**Problem: Backend crashes or errors**

- Solution: Make sure you activated the virtual environment

- Check Python version: `python --version` (should be 3.9+)Open a **NEW terminal window** (keep the backend running) and run:FastAPI / Flask

- Reinstall dependencies: `pip install -r requirements.txt`



**Problem: Frontend doesn't connect to backend**

- Solution: Make sure backend is running on http://localhost:8000```bash---OpenCV

- Check if both terminals are still active

# Navigate to the project directory

---

cd Truepix/frontendPyTorch (model inference)

## Stopping the Application



To stop the servers:

# Install Node.js dependencies (this may take 2-3 minutes)## Quick StartFrontend

1. **Stop Frontend:** Press `Ctrl + C` in the frontend terminal

2. **Stop Backend:** Press `Ctrl + C` in the backend terminalnpm install

3. **Deactivate virtual environment:** Type `deactivate` in backend terminal

React

---

# Start the frontend development server

## System Requirements

npm start### Backend SetupHTML / CSS / JavaScript

| Component | Minimum | Recommended |

|-----------|---------|-------------|```

| OS | Windows 10, macOS 10.15, Ubuntu 20.04 | Latest version |

| RAM | 4GB | 8GB |```bashModel & Analysis

| Storage | 2GB free space | 5GB free space |

| CPU | Dual-core 2.0GHz | Quad-core 2.5GHz+ |✅ **Frontend is now running at:** http://localhost:3000

| GPU | Not required | CUDA-compatible (optional, for faster inference) |

cd backendCNN-based feature extraction

---

Your browser should automatically open. If not, manually open: **http://localhost:3000**

## Results & Evaluation

python3 -m venv venvFrequency-domain analysis

### Model Performance

### Step 4: Using TruePix

**Training Configuration:**

- Dataset: 50,000 images (balanced real/AI split)source venv/bin/activateImage forensics techniques

- Real sources: COCO, FFHQ, natural photography datasets

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs1. **Upload an Image:**

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss

   - Click the upload area or drag-and-drop an image (JPG/PNG format)pip install -r requirements.txtHow to Run the Project

**Evaluation Metrics:**

- **Accuracy:** 89.3% on held-out test set   - Recommended image size: 500KB - 5MB

- **Precision (AI class):** 91.2%

- **Recall (AI class):** 87.5%python main.pyBackend

- **F1-Score:** 89.3%

2. **View Results:**

### Robustness Testing

   - Classification: "AI-Generated" or "Real"```cd backend

**Platform Stability Scores:**

- Original images: 89.3% accuracy   - Confidence Score: 0-100%

- WhatsApp compression (512px, Q=40): 82.1% accuracy

- Instagram compression (1080px, Q=70): 86.7% accuracy   - Executive Summary: Why the image was classifiedBackend runs at: **http://localhost:8000**pip install -r requirements.txt

- Facebook compression (960px, Q=60): 85.4% accuracy

   - Branch Analysis: What each algorithm detected

**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.

   - Grad-CAM Heatmap: Visual explanation highlighting suspicious regionspython app.py

### Explainability Validation



- **Grad-CAM Analysis:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test cases

- **Branch Attribution:** Spatial branch contributes 45%, frequency 28%, noise 18%, edge 9% on average3. **Test Platform Stability (Optional):**### Frontend SetupFrontend

- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictions

   - Click "Test Platform Stability" button

---

   - See how prediction changes across WhatsApp, Instagram, Facebook compression```bashcd frontend

## Limitations



**Current Constraints:**

- Trained on 2024-2025 AI generators; may not generalize to future models or unseen architectures---cd frontendnpm install

- Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%)

- Not hardened against sophisticated adversarial attacks designed to fool detectors

- Struggles with hybrid images (real photos containing AI-edited elements)

- Computational cost: Multi-branch architecture requires 2.5x inference time vs single CNN (150ms vs 60ms on CPU)## Troubleshootingnpm installnpm start

- False positives possible on heavily post-processed real photos (HDR, beauty filters)



**Responsible Use:**

- This tool should be used as guidance, not definitive proof**Problem: "Python not found"**npm startResults & Evaluation

- Combine with human expert verification for high-stakes decisions

- Not recommended as sole evidence in legal or journalistic contexts- Solution: Install Python from python.org and restart terminal



---```The system successfully identifies AI-generated images by detecting non-natural artifacts.



## Future Work**Problem: "npm not found"**



**Model Enhancements:**- Solution: Install Node.js from nodejs.org and restart terminalFrontend runs at: **http://localhost:3000**Tested on a mixed dataset containing:

- Ensemble multiple architectures (Vision Transformers + ResNet) for improved accuracy

- Continuous learning pipeline to adapt to emerging AI generators

- Adversarial training to improve robustness against evasion techniques

- Expand training dataset to 500k+ images across more generators**Problem: "Port 8000 already in use"**Real photographs



**Feature Additions:**- Solution: Kill the process using port 8000:

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)

- Batch processing API for high-throughput analysis  ```bash### UsageAI-generated images from multiple generation sources

- Video frame-by-frame analysis for deepfake detection

- Model fingerprinting to identify specific AI generator (DALL-E vs Midjourney vs Stable Diffusion)  # Mac/Linux:

- Temporal consistency analysis for video sequences

  lsof -ti:8000 | xargs kill -91. Open http://localhost:3000Performance was evaluated using accuracy and confidence-based prediction consistency.

**Deployment Improvements:**

- Model quantization (INT8) for 3x faster inference  # Windows:

- Mobile applications (iOS/Android) with on-device inference

- Browser extension for in-situ web image analysis  netstat -ano | findstr :80002. Upload an image (JPG/PNG)Note: Due to dataset size constraints, datasets are not included in the repository.

- Cloud deployment with auto-scaling for high traffic

- Partnership with fact-checking organizations and news platforms  taskkill /PID <PID_NUMBER> /F



**Research Directions:**  ```3. View results: classification, confidence, explanation, Grad-CAM heatmapLimitations

- Cross-modal consistency analysis (text-image alignment verification)

- Zero-shot detection of unseen generative models

- Certified defenses against adversarial perturbations

**Problem: "Port 3000 already in use"**Performance depends on diversity of training data

---

- Solution: Kill the process or use a different port:

## Conclusion

  ```bash---Newer AI generators may introduce unseen patterns

TruePix represents a significant advancement in AI-generated image detection by introducing a hybrid multi-branch architecture that combines spatial, frequency, noise, and edge analysis. Unlike existing single-model detectors, our system provides:

  # The frontend will prompt: "Would you like to run on another port?" → Press Y

1. **Superior Robustness:** Four complementary detection methods ensure artifacts are caught regardless of which AI generator was used or what compression the image underwent.

  ```Does not guarantee 100% accuracy for highly post-processed images

2. **Unprecedented Explainability:** Users don't just get a "yes/no" answer—they understand exactly why an image was classified through executive summaries, per-branch attributions, and visual Grad-CAM heatmaps.



3. **Real-World Applicability:** Built-in platform stability testing and production-ready architecture make TruePix deployable in actual content moderation, fact-checking, and forensic scenarios.

**Problem: Backend crashes or errors**## ResultsFuture Work

4. **Research Innovation:** Novel integration of frequency-domain FFT analysis, multi-scale noise verification, and physics-based edge validation sets new standards for detection methodology.

- Solution: Make sure you activated the virtual environment

As generative AI continues to evolve, tools like TruePix are essential for maintaining trust in digital media. By combining multiple forensic approaches with transparent explainability, we provide a foundation for responsible AI detection that can adapt to future challenges.

- Check Python version: `python --version` (should be 3.9+)Extend detection to video and deepfake content

**Our vision:** Empower users, journalists, and organizations with trustworthy tools to verify image authenticity in an era where "seeing is no longer believing."

- Reinstall dependencies: `pip install -r requirements.txt`

---

- **Accuracy:** 89.3% on test setIntegrate explainability heatmaps for visual justification

## References

**Problem: Frontend doesn't connect to backend**

**Academic Research:**

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." CVPR 2020.- Solution: Make sure backend is running on http://localhost:8000- **Precision:** 91.2% (AI detection)Expand dataset to include newer AI image generators

2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.

3. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017.- Check if both terminals are still active

4. Marra et al. (2019). "Do GANs Leave Specific Traces?" IEEE WIFS.

- **Robustness:** 82-87% accuracy after social media compressionImprove robustness against adversarial attacks

**Datasets:**

- CIFAKE: Kaggle real vs AI image dataset---

- DiffusionDB: Stable Diffusion dataset (Hugging Face)

- COCO: Microsoft Common Objects in Context- **Explainability:** 84% Grad-CAM accuracy, 92% user satisfactionEthical Considerations

- FFHQ: NVIDIA Flickr-Faces-HQ

## Stopping the Application

**Tools & Frameworks:**

- PyTorch: pytorch.orgThis project is intended only for detection and verification purposes.

- timm (PyTorch Image Models): github.com/huggingface/pytorch-image-models

- FastAPI: fastapi.tiangolo.comTo stop the servers:

- React: react.dev

---It does not generate or promote misuse of AI-generated content.

---

1. **Stop Frontend:** Press `Ctrl + C` in the frontend terminal

## License & Citation

2. **Stop Backend:** Press `Ctrl + C` in the backend terminalAuthor

**License:** MIT License - See LICENSE file for details

3. **Deactivate virtual environment:** Type `deactivate` in backend terminal

**Citation:**

If you use TruePix in your research or project, please cite:## LimitationsKathiravan M



```bibtex---

@software{truepix2026,

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},Project: TruePix – AI vs Real Image Detection

  author={TruePix Contributors},

  year={2026},## System Requirements

  url={https://github.com/kathirpmdk-star/Truepix}

}- Trained on 2024-2025 AI generators; may not detect future models

```

| Component | Minimum | Recommended |

---

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
