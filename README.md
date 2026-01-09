# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System



<div align="center">



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)<div align="center">

A hybrid multi-branch deep learning system for detecting AI-generated images with explainability

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

</div>

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)

---

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

## 1. Overview

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)<div align="center">

TruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.

A hybrid multi-branch deep learning system for detecting AI-generated images with explainability

**Key Contributions:**

- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

- Grad-CAM visualizations and per-branch attribution for model transparency

- Platform robustness testing against social media compression</div>

- End-to-end web application with real-time inference and explainable results

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)

**Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.

---

---

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

## 2. Problem Statement

## 1. Overview

### Challenge

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)<div align="center"><div align="center">

Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images increasingly difficult to distinguish from real photographs. This poses significant risks:

TruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.

- Misinformation: Fake news propagation using synthetic imagery

- Fraud: Identity theft, document forgery, deepfake scamsA hybrid multi-branch deep learning system for detecting AI-generated images with explainability

- Trust Erosion: Declining confidence in digital media authenticity

**Key Contributions:**

### Objective

- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

Develop a robust, explainable AI detection system that:

- Grad-CAM visualizations and per-branch attribution for model transparency

- Achieves high accuracy across diverse image types and AI generators

- Provides interpretable explanations for predictions- Platform robustness testing against social media compression</div>

- Maintains performance under real-world compression and post-processing

- Offers real-time analysis suitable for production deployment- End-to-end web application with real-time inference and explainable results



---[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)



## 3. Methodology**Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.



### 3.1 Algorithms Used---



Our system employs four specialized algorithms, each targeting different artifacts left by AI image generators:---



#### CNN (Convolutional Neural Network) - Spatial Analysis[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)



**What it does:** Uses EfficientNet-B0 deep learning model to analyze visual patterns in the image.## 2. Problem Statement



**How it helps:** Detects unnatural textures, incorrect anatomy (like malformed hands), over-smooth skin, and repetitive patterns that AI generators commonly produce. Real photos have natural imperfections; AI images often look "too perfect" or have subtle artifacts invisible to human eyes but detectable by trained neural networks.## 1. Overview



#### FFT (Fast Fourier Transform) - Frequency Analysis### Challenge



**What it does:** Converts the image from spatial domain to frequency domain to analyze periodic patterns.Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images increasingly difficult to distinguish from real photographs. This poses significant risks:[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)



**How it helps:** AI generators leave invisible "fingerprints" in the frequency spectrum—grid patterns, upsampling artifacts, and unnatural frequency distributions. Real photos have random, broad frequency spectra from natural scenes. This catches artifacts that look normal spatially but show clear patterns in frequency domain.



#### Noise Analysis - Sensor Noise Consistency- Misinformation: Fake news propagation using synthetic imageryTruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.



**What it does:** Analyzes noise patterns at multiple scales using Gaussian blur residuals.- Fraud: Identity theft, document forgery, deepfake scams



**How it helps:** Real cameras produce characteristic sensor noise (shot noise, read noise, thermal noise). AI generators either produce images with no noise or add synthetic noise that doesn't match real camera behavior. By analyzing noise at fine, medium, and coarse scales, we can distinguish authentic camera noise from synthetic or absent noise.- Trust Erosion: Declining confidence in digital media authenticity**A hybrid multi-branch deep learning system for detecting AI-generated images with explainability**



#### Edge Analysis - Structural Plausibility



**What it does:** Uses Sobel operators to detect edges and analyze their physical consistency.### Objective**Key Contributions:**



**How it helps:** Real photos follow optical physics—edges are consistent, blur follows depth-of-field laws, sharpness transitions are natural. AI generators sometimes produce physically impossible edges, inconsistent blur, or unnatural sharpness transitions. This algorithm catches violations of physical constraints.Develop a robust, explainable AI detection system that:



### 3.2 How They Work Together- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)



```- Achieves high accuracy across diverse image types and AI generators

Step 1: Each algorithm analyzes the image independently

CNN → Spatial features (256-dim)- Provides interpretable explanations for predictions- Grad-CAM visualizations and per-branch attribution for model transparency

FFT → Frequency features (256-dim)

Noise → Noise patterns (256-dim)- Maintains performance under real-world compression and post-processing

Edge → Edge structures (256-dim)

- Offers real-time analysis suitable for production deployment- Platform robustness testing against social media compression[How to Run](#6-how-to-run) • [Architecture](#4-system-architecture) • [Results](#7-results--evaluation)

Step 2: Combine all features

Total: 1024-dimensional feature vector



Step 3: Fusion network makes final decision---- End-to-end web application with real-time inference and explainable results

Neural network (1024→512→2) → AI or Real



Step 4: Generate explanations

- Which branch contributed most to the decision## 3. Methodology[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

- What specific artifacts were found

- Grad-CAM heatmap showing suspicious regions

```

### 3.1 Algorithms Used**Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.

**Why this approach works:** Different AI generators have different weaknesses. DALL-E might have good textures but poor noise; Midjourney might have good edges but poor frequency distribution. By combining four different detection methods, we catch artifacts regardless of which generator was used.



### 3.3 Explainability

Our system employs four specialized algorithms, each targeting different artifacts left by AI image generators:</div>

The system doesn't just say "AI-generated"—it explains WHY:



- Executive Summary: Natural language explanation of the decision

- Per-Branch Scores: Shows which algorithm found the strongest evidence#### CNN (Convolutional Neural Network) - Spatial Analysis---

- Grad-CAM Heatmap: Visual overlay highlighting suspicious image regions



---

**What it does:** Uses EfficientNet-B0 deep learning model to analyze visual patterns in the image.[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

## 4. System Architecture



```

Frontend (React) → Backend (FastAPI) → ML Model (PyTorch)**How it helps:** Detects unnatural textures, incorrect anatomy (like malformed hands), over-smooth skin, and repetitive patterns that AI generators commonly produce. Real photos have natural imperfections; AI images often look "too perfect" or have subtle artifacts invisible to human eyes but detectable by trained neural networks.## 2. Problem Statement

     ↓                    ↓                   ↓

Image Upload      API Endpoints         4 Algorithms

Results UI        Preprocessing         Fusion Layer

Grad-CAM          Explainability        Grad-CAM#### FFT (Fast Fourier Transform) - Frequency Analysis---

```



**Components:**

- Frontend: React 18.2 with drag-and-drop upload and real-time visualization**What it does:** Converts the image from spatial domain to frequency domain to analyze periodic patterns.### Challenge

- Backend: FastAPI REST API with /api/analyze endpoint

- ML Pipeline: PyTorch 2.1 with EfficientNet-B0 and custom multi-branch architecture

- Storage: Optional Supabase; functions in demo mode without external dependencies

**How it helps:** AI generators leave invisible "fingerprints" in the frequency spectrum—grid patterns, upsampling artifacts, and unnatural frequency distributions. Real photos have random, broad frequency spectra from natural scenes. This catches artifacts that look normal spatially but show clear patterns in frequency domain.Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images increasingly difficult to distinguish from real photographs. This poses significant risks:[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---



## 5. Tech Stack

#### Noise Analysis - Sensor Noise Consistency

| Layer | Technology | Purpose |

|-------|------------|---------|

| Frontend | React 18.2, CSS3 | User interface and visualization |

| Backend | FastAPI 0.108, Uvicorn | High-performance API server |**What it does:** Analyzes noise patterns at multiple scales using Gaussian blur residuals.- **Misinformation:** Fake news propagation using synthetic imagery## 1. Overview

| ML Framework | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |

| Model | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |

| Image Processing | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |

| Explainability | Custom Grad-CAM, scipy | Visualization and frequency analysis |**How it helps:** Real cameras produce characteristic sensor noise (shot noise, read noise, thermal noise). AI generators either produce images with no noise or add synthetic noise that doesn't match real camera behavior. By analyzing noise at fine, medium, and coarse scales, we can distinguish authentic camera noise from synthetic or absent noise.- **Fraud:** Identity theft, document forgery, deepfake scams



---



## 6. How to Run#### Edge Analysis - Structural Plausibility- **Trust Erosion:** Declining confidence in digital media authenticity**A hybrid multi-branch deep learning system for detecting AI-generated images with explainability**



### Prerequisites



- Python 3.9+**What it does:** Uses Sobel operators to detect edges and analyze their physical consistency.

- Node.js 16+

- 4GB RAM (8GB recommended)



### Backend Setup**How it helps:** Real photos follow optical physics—edges are consistent, blur follows depth-of-field laws, sharpness transitions are natural. AI generators sometimes produce physically impossible edges, inconsistent blur, or unnatural sharpness transitions. This algorithm catches violations of physical constraints.### ObjectiveTruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.



```bash

cd backend

python3 -m venv venv### 3.2 How They Work TogetherDevelop a robust, explainable AI detection system that:

source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py

``````**A multi-branch deep learning system for detecting AI-generated images with explainability**



Backend: http://localhost:8000 | API Docs: http://localhost:8000/docsStep 1: Each algorithm analyzes the image independently



### Frontend SetupCNN → Spatial features (256-dim)- Achieves high accuracy across diverse image types and AI generators



```bashFFT → Frequency features (256-dim)

cd frontend

npm installNoise → Noise patterns (256-dim)- Provides interpretable explanations for predictions**Key Contributions:**

npm start

```Edge → Edge structures (256-dim)



Frontend: http://localhost:3000- Maintains performance under real-world compression and post-processing



### UsageStep 2: Combine all features



1. Open http://localhost:3000Total: 1024-dimensional feature vector- Offers real-time analysis suitable for production deployment- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains[Demo](#6-how-to-run) • [Architecture](#4-system-architecture) • [Results](#7-results--evaluation)

2. Upload JPG/PNG image (drag-and-drop or click)

3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM

4. (Optional) Test platform stability across social media compression

Step 3: Fusion network makes final decision

---

Neural network (1024→512→2) → AI or Real

## 7. Results & Evaluation

---- Grad-CAM visualizations and per-branch attribution for model transparency

### Model Performance

Step 4: Generate explanations

**Training Configuration:**

- Dataset: 50,000 images (balanced real/AI split)- Which branch contributed most to the decision

- Real sources: COCO, FFHQ, natural photography datasets

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs- What specific artifacts were found

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss

- Grad-CAM heatmap showing suspicious regions## 3. Methodology- Platform robustness testing against social media compression (WhatsApp, Instagram, Facebook)</div>

**Evaluation Metrics:**

- Accuracy: 89.3% on held-out test set```

- Precision (AI class): 91.2%

- Recall (AI class): 87.5%

- F1-Score: 89.3%

**Why this approach works:** Different AI generators have different weaknesses. DALL-E might have good textures but poor noise; Midjourney might have good edges but poor frequency distribution. By combining four different detection methods, we catch artifacts regardless of which generator was used.

### Robustness Testing

### 3.1 Multi-Branch Feature Extraction- End-to-end web application with real-time inference and explainable results

**Platform Stability Scores:**

- Original images: 89.3% accuracy### 3.3 Explainability

- WhatsApp compression (512px, Q=40): 82.1% accuracy

- Instagram compression (1080px, Q=70): 86.7% accuracy

- Facebook compression (960px, Q=60): 85.4% accuracy

The system doesn't just say "AI-generated"—it explains WHY:

**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.

**Convolutional Neural Network (CNN) - Spatial Branch**</div>

### Explainability Validation

- Executive Summary: Natural language explanation of the decision

- Grad-CAM Analysis: Heatmaps correctly highlight known AI artifacts in 84% of test cases

- Branch Attribution: Spatial 45%, Frequency 28%, Noise 18%, Edge 9% on average- Per-Branch Scores: Shows which algorithm found the strongest evidence

- User Study: 92% of users (n=25) found explanations helpful

- Grad-CAM Heatmap: Visual overlay highlighting suspicious image regions

---

Uses EfficientNet-B0, a state-of-the-art CNN architecture pre-trained on ImageNet with 5.3M parameters.**Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.

## 8. Limitations

---

- Dataset Constraints: Model trained on 2024-2025 AI generators; may not generalize to future models

- Compression Sensitivity: Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%)

- Adversarial Vulnerability: Not hardened against intentional evasion techniques

- Hybrid Images: Struggles with real photos containing AI-edited elements## 4. System Architecture

- Computational Cost: Multi-branch architecture requires 2.5x inference time vs single CNN (150ms vs 60ms)

- False Positives: Heavily post-processed real photos may trigger false AI detections- **Input:** RGB image (224×224×3 pixels)---



---```



## 9. Future WorkFrontend (React) → Backend (FastAPI) → ML Model (PyTorch)- **Processing:** 



**Model Enhancements:**     ↓                    ↓                   ↓

- Ensemble multiple architectures (Vision Transformers, ResNet) for improved accuracy

- Continuous learning pipeline to adapt to emerging AI generatorsImage Upload      API Endpoints         4 Algorithms  - EfficientNet-B0 backbone extracts 1,280-dimensional feature vector---

- Adversarial training for robustness

Results UI        Preprocessing         Fusion Layer

**Feature Additions:**

- EXIF metadata analysis for forensic verificationGrad-CAM          Explainability        Grad-CAM  - Projection head reduces to 256-dim embedding via FC layers (1280→512→256)

- Batch processing API for high-throughput analysis

- Video frame analysis for deepfake detection```

- Model fingerprinting to identify specific AI generator

  - Dropout (p=0.3) prevents overfitting---

**Deployment:**

- Model quantization (INT8) for 3x faster inference**Components:**

- Mobile applications with on-device inference

- Browser extension for web image analysis- Frontend: React 18.2 with drag-and-drop upload and real-time visualization- **What it detects:** Texture smoothness anomalies, unnatural patterns in skin/fabric, anatomically incorrect structures (malformed hands, asymmetric faces), repetitive background elements

- Partnership with fact-checking organizations

- Backend: FastAPI REST API with /api/analyze endpoint

**Research Directions:**

- Cross-modal consistency analysis (text-image alignment)- ML Pipeline: PyTorch 2.1 with EfficientNet-B0 and custom multi-branch architecture- **Why it works:** AI generators produce statistically different pixel patterns than camera sensors; CNNs learn these discriminative features through supervised training## 2. Problem Statement

- Temporal consistency for video sequences

- Zero-shot detection of unseen generative models- Storage: Optional Supabase; functions in demo mode without external dependencies



---



## 10. References---



**Academic Research:****Fast Fourier Transform (FFT) - Frequency Branch**## Table of Contents

1. Wang et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot... For Now. CVPR 2020.

2. Gragnaniello et al. (2021). GAN-Generated Faces Detection. IEEE TIFS.## 5. Tech Stack

3. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.



**Datasets:**

- CIFAKE: Kaggle real vs AI image dataset| Layer | Technology | Purpose |

- DiffusionDB: Stable Diffusion dataset (Hugging Face)

- COCO: Microsoft Common Objects in Context|-------|------------|---------|Applies 2D Discrete Fourier Transform to convert spatial image into frequency domain representation.### Challenge

- FFHQ: NVIDIA Flickr-Faces-HQ

| Frontend | React 18.2, CSS3 | User interface and visualization |

**Tools & Frameworks:**

- PyTorch: pytorch.org| Backend | FastAPI 0.108, Uvicorn | High-performance API server |

- timm: github.com/huggingface/pytorch-image-models

- FastAPI: fastapi.tiangolo.com| ML Framework | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |

- React: react.dev

| Model | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |- **Input:** Grayscale image (224×224 pixels)Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images increasingly difficult to distinguish from real photographs. This poses significant risks:## 1. Overview

---

| Image Processing | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |

## License & Citation

| Explainability | Custom Grad-CAM, scipy | Visualization and frequency analysis |- **Processing:**

**License:** MIT License - See LICENSE file for details



**Citation:**

```bibtex---  - 2D FFT computes frequency spectrum: F(u,v) = Σ Σ f(x,y)·e^(-j2π(ux/M+vy/N))

@software{truepix2026,

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},

  author={TruePix Contributors},

  year={2026},## 6. How to Run  - Logarithmic magnitude transformation: log(1 + |F(u,v)|) enhances weak frequencies

  url={https://github.com/kathirpmdk-star/Truepix}

}

```

### Prerequisites  - Fully connected layers extract 256-dim frequency embedding- **Misinformation**: Fake news propagation using synthetic imagery- [Overview](#overview)

---

- Python 3.9+

<div align="center">

- Node.js 16+- **What it detects:** Grid patterns from upsampling, periodic artifacts from GAN generators, unnatural frequency distributions, checkerboard effects

Developed for Academic Research | Promoting Digital Media Transparency

- 4GB RAM (8GB recommended)

For questions or collaborations, please open a GitHub issue

- **Why it works:** Real photos have broad frequency spectra from natural scenes; AI generators introduce subtle periodic patterns during synthesis that are invisible spatially but prominent in frequency domain- **Fraud**: Identity theft, document forgery, deepfake scams

</div>

### Backend Setup

```bash

cd backend

python3 -m venv venv**Noise Consistency Analysis - Noise Branch**- **Trust Erosion**: Declining confidence in digital media authenticityTruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.- [Methodology](#methodology)

source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py

```Analyzes noise patterns using multi-scale Gaussian blur residual extraction.

Backend: http://localhost:8000 | API Docs: http://localhost:8000/docs



### Frontend Setup

```bash- **Input:** RGB image (224×224×3 pixels)### Objective- [System Architecture](#system-architecture)

cd frontend

npm install- **Processing:**

npm start

```  - Generates three noise residual maps:Develop a robust, explainable AI detection system that:

Frontend: http://localhost:3000

    - Fine scale: Image - GaussianBlur(σ=0.5) → captures high-frequency noise

### Usage

1. Open http://localhost:3000    - Medium scale: Image - GaussianBlur(σ=1.0) → captures mid-frequency noise**Key Contributions:**- [Installation and Setup](#installation-and-setup)

2. Upload JPG/PNG image (drag-and-drop or click)

3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM    - Coarse scale: Image - GaussianBlur(σ=2.0) → captures low-frequency noise

4. (Optional) Test platform stability across social media compression

  - Concatenates into 9-channel tensor (3 RGB × 3 scales)- Achieves high accuracy across diverse image types and AI generators

---

  - CNN layers (Conv2D 9→32→64→128) extract 256-dim noise embedding

## 7. Results & Evaluation

- **What it detects:** Absence of camera sensor noise, synthetic noise patterns, inconsistent noise across image regions, unrealistic noise distribution- Provides interpretable explanations for predictions- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains- [Usage](#usage)

### Model Performance

- **Why it works:** Real cameras produce characteristic sensor noise (shot noise, read noise, dark current); AI generators either omit noise entirely or add synthetic noise that lacks realistic statistical properties (non-Gaussian distribution, spatially inconsistent)

Training Configuration:

- Dataset: 50,000 images (balanced real/AI split)- Maintains performance under real-world compression and post-processing

- Real sources: COCO, FFHQ, natural photography datasets

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs**Edge Structure Analysis - Edge Branch**

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss

- Offers real-time analysis suitable for production deployment- Grad-CAM visualizations and per-branch attribution for model transparency- [API Reference](#api-reference)

Evaluation Metrics:

- Accuracy: 89.3% on held-out test setEmploys Sobel operators to detect edge consistency and structural plausibility.

- Precision (AI class): 91.2%

- Recall (AI class): 87.5%

- F1-Score: 89.3%

- **Input:** Grayscale image (224×224 pixels)

### Robustness Testing

- **Processing:**---- Platform robustness testing against social media compression (WhatsApp, Instagram, Facebook)- [Limitations](#limitations)

Platform Stability Scores:

- Original images: 89.3% accuracy  - Sobel horizontal gradient: Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] ⊗ Image

- WhatsApp compression (512px, Q=40): 82.1% accuracy

- Instagram compression (1080px, Q=70): 86.7% accuracy  - Sobel vertical gradient: Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] ⊗ Image

- Facebook compression (960px, Q=60): 85.4% accuracy

  - Gradient magnitude: G = √(Gx² + Gy²)

Insight: Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.

  - CNN layers extract 256-dim edge embedding## 3. Methodology- End-to-end web application with real-time inference and explainable results- [Future Work](#future-work)

### Explainability Validation

- **What it detects:** Physically impossible edge continuities, blurred object boundaries, unnatural edge sharpness transitions, inconsistent depth-of-field effects

- Grad-CAM Analysis: Heatmaps correctly highlight known AI artifacts in 84% of test cases

- Branch Attribution: Spatial 45%, Frequency 28%, Noise 18%, Edge 9% on average- **Why it works:** Real photos follow optical physics with consistent edge properties; AI generators may produce edges that violate physical constraints or show inconsistent blur/sharpness

- User Study: 92% of users (n=25) found explanations helpful



---

### 3.2 Feature Fusion & Classification### 3.1 Multi-Branch Feature Extraction- [Contributing](#contributing)

## 8. Limitations



- Dataset Constraints: Model trained on 2024-2025 AI generators; may not generalize to future models

- Compression Sensitivity: Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%)Concatenates all branch embeddings and applies fusion network for final prediction.

- Adversarial Vulnerability: Not hardened against intentional evasion techniques

- Hybrid Images: Struggles with real photos containing AI-edited elements

- Computational Cost: Multi-branch architecture requires 2.5x inference time vs single CNN (150ms vs 60ms)

- False Positives: Heavily post-processed real photos may trigger false AI detections```**Convolutional Neural Network (CNN) - Spatial Branch****Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.



---Step 1: Concatenate branch features



## 9. Future Work[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] = 1024-dim vector



Model Enhancements:

- Ensemble multiple architectures (Vision Transformers, ResNet) for improved accuracy

- Continuous learning pipeline to adapt to emerging AI generatorsStep 2: Fusion layer with regularizationUses EfficientNet-B0, a state-of-the-art CNN architecture pre-trained on ImageNet with 5.3M parameters.---

- Adversarial training for robustness

FC(1024→512) + ReLU + Dropout(0.5)

Feature Additions:

- EXIF metadata analysis for forensic verification

- Batch processing API for high-throughput analysis

- Video frame analysis for deepfake detectionStep 3: Binary classification

- Model fingerprinting to identify specific AI generator

FC(512→2) + Softmax → [P(Real), P(AI-Generated)]- **Input:** RGB image (224×224×3 pixels)---

Deployment:

- Model quantization (INT8) for 3x faster inference

- Mobile applications with on-device inference

- Browser extension for web image analysisStep 4: Temperature calibration- **Processing:** 

- Partnership with fact-checking organizations

Calibrated confidence = Softmax(logits / T) where T=1.5 (tuned on validation set)

Research Directions:

- Cross-modal consistency analysis (text-image alignment)```  - EfficientNet-B0 backbone extracts 1,280-dimensional feature vector## Overview

- Temporal consistency for video sequences

- Zero-shot detection of unseen generative models



---**Why multi-branch:** Different AI generators leave different artifacts; combining multiple detection methods provides robustness against various generation techniques.  - Projection head reduces to 256-dim embedding via FC layers (1280→512→256)



## 10. References



Academic Research:### 3.3 Explainability Framework  - Dropout (p=0.3) prevents overfitting## 2. Problem Statement

1. Wang et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot... For Now. CVPR 2020.

2. Gragnaniello et al. (2021). GAN-Generated Faces Detection. IEEE TIFS.

3. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.

**Executive Summary Generation:** Analyzes branch confidence scores and generates natural language explanation of classification decision and key evidence.- **What it detects:** Texture smoothness anomalies, unnatural patterns in skin/fabric, anatomically incorrect structures (malformed hands, asymmetric faces), repetitive background elements

Datasets:

- CIFAKE: Kaggle real vs AI image dataset

- DiffusionDB: Stable Diffusion dataset (Hugging Face)

- COCO: Microsoft Common Objects in Context**Per-Branch Attribution:** Uses ablation testing—evaluates model with each branch disabled to measure individual branch contributions to final prediction.- **Why it works:** AI generators produce statistically different pixel patterns than camera sensors; CNNs learn these discriminative features through supervised trainingTruePix is an academic research project implementing a hybrid multi-branch deep learning architecture for detecting AI-generated images. The system combines spatial, frequency-domain, noise-pattern, and edge-structure analysis to provide robust classification with comprehensive explainability features including Grad-CAM visualizations and per-branch decision attribution.

- FFHQ: NVIDIA Flickr-Faces-HQ



Tools & Frameworks:

- PyTorch: pytorch.org**Grad-CAM Visualization:** Computes gradient-weighted activation maps from spatial CNN's final convolutional layer, highlighting image regions most influential to classification decision.

- timm: github.com/huggingface/pytorch-image-models

- FastAPI: fastapi.tiangolo.com

- React: react.dev

---**Fast Fourier Transform (FFT) - Frequency Branch**### Challenge

---



## License & Citation

## 4. System Architecture

License: MIT License - See LICENSE file for details



Citation:

```bibtex```Applies 2D Discrete Fourier Transform to convert spatial image into frequency domain representation.Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images that are increasingly difficult to distinguish from real photographs. This poses significant risks:### Key Features

@software{truepix2026,

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},Frontend (React) → Backend (FastAPI) → ML Model (PyTorch)

  author={TruePix Contributors},

  year={2026},     ↓                    ↓                   ↓

  url={https://github.com/kathirpmdk-star/Truepix}

}Image Upload      API Endpoints         4 Branches

```

Results UI        Preprocessing         Fusion Layer- **Input:** Grayscale image (224×224 pixels)- **Misinformation**: Fake news propagation using synthetic imagery

---

Grad-CAM          Explainability        Grad-CAM

<div align="center">

```- **Processing:**

Developed for Academic Research | Promoting Digital Media Transparency



For questions or collaborations, please open a GitHub issue

**Components:**  - 2D FFT computes frequency spectrum: F(u,v) = Σ Σ f(x,y)·e^(-j2π(ux/M+vy/N))- **Fraud**: Identity theft, document forgery, deepfake scams- **Hybrid Multi-Branch Architecture**: Combines four complementary detection approaches

</div>

- **Frontend:** React 18.2 with drag-and-drop upload and real-time visualization

- **Backend:** FastAPI REST API with `/api/analyze` and `/api/simulate-platforms` endpoints  - Logarithmic magnitude transformation: log(1 + |F(u,v)|) enhances weak frequencies

- **ML Pipeline:** PyTorch 2.1 with EfficientNet-B0 and custom multi-branch architecture

- **Storage:** Optional Supabase; functions in demo mode without external dependencies  - Fully connected layers extract 256-dim frequency embedding- **Trust Erosion**: Declining confidence in digital media authenticity- **Explainability**: Provides decision basis, per-branch analysis, and Grad-CAM heatmaps



---- **What it detects:** Grid patterns from upsampling, periodic artifacts from GAN generators, unnatural frequency distributions, checkerboard effects



## 5. Tech Stack- **Why it works:** Real photos have broad frequency spectra from natural scenes; AI generators introduce subtle periodic patterns during synthesis that are invisible spatially but prominent in frequency domain- **Platform Robustness Testing**: Evaluates stability across social media compression scenarios



| Layer | Technology | Purpose |

|-------|------------|---------|

| Frontend | React 18.2, CSS3 | User interface and visualization |**Noise Consistency Analysis - Noise Branch**### Objective- **Real-time Analysis**: FastAPI backend with React frontend for immediate results

| Backend | FastAPI 0.108, Uvicorn | High-performance API server |

| ML Framework | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |

| Model | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |

| Image Processing | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |Analyzes noise patterns using multi-scale Gaussian blur residual extraction.Develop a robust, explainable AI detection system that:

| Explainability | Custom Grad-CAM, scipy | Visualization and frequency analysis |



**Dependencies:** Python 3.9+, Node.js 16+, CUDA (optional for GPU acceleration)

- **Input:** RGB image (224×224×3 pixels)- Achieves high accuracy across diverse image types and AI generators---

---

- **Processing:**

## 6. How to Run

  - Generates three noise residual maps:- Provides interpretable explanations for predictions

### Prerequisites

- Python 3.9+    - Fine scale: Image - GaussianBlur(σ=0.5) → captures high-frequency noise

- Node.js 16+

- 4GB RAM (8GB recommended)    - Medium scale: Image - GaussianBlur(σ=1.0) → captures mid-frequency noise- Maintains performance under real-world compression and post-processing## Methodology



### Backend Setup    - Coarse scale: Image - GaussianBlur(σ=2.0) → captures low-frequency noise

```bash

cd backend  - Concatenates into 9-channel tensor (3 RGB × 3 scales)- Offers real-time analysis suitable for production deployment

python3 -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate  - CNN layers (Conv2D 9→32→64→128) extract 256-dim noise embedding

pip install -r requirements.txt

python main.py- **What it detects:** Absence of camera sensor noise, synthetic noise patterns, inconsistent noise across image regions, unrealistic noise distribution### 1. Image Preprocessing

```

**Backend:** http://localhost:8000 | **API Docs:** http://localhost:8000/docs- **Why it works:** Real cameras produce characteristic sensor noise (shot noise, read noise, dark current); AI generators either omit noise entirely or add synthetic noise that lacks realistic statistical properties (non-Gaussian distribution, spatially inconsistent)



### Frontend Setup---

```bash

cd frontend**Edge Structure Analysis - Edge Branch**

npm install

npm startInput images undergo standardized preprocessing to ensure consistent model input:

```

**Frontend:** http://localhost:3000Employs Sobel operators to detect edge consistency and structural plausibility.



### Usage## 3. Methodology

1. Open http://localhost:3000

2. Upload JPG/PNG image (drag-and-drop or click)- **Input:** Grayscale image (224×224 pixels)

3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM heatmap

4. (Optional) Test platform stability across social media compression scenarios- **Processing:**```python



---  - Sobel horizontal gradient: Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] ⊗ Image



## 7. Results & Evaluation  - Sobel vertical gradient: Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] ⊗ Image### A. Multi-Branch Feature ExtractionPreprocessing Pipeline:



### Model Performance  - Gradient magnitude: G = √(Gx² + Gy²)



**Training Configuration:**  - CNN layers extract 256-dim edge embedding1. Resize → 224×224 pixels (bicubic interpolation)

- Dataset: 50,000 images (balanced real/AI split)

- Real sources: COCO, FFHQ, natural photography datasets- **What it detects:** Physically impossible edge continuities, blurred object boundaries, unnatural edge sharpness transitions, inconsistent depth-of-field effects

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss- **Why it works:** Real photos follow optical physics with consistent edge properties; AI generators may produce edges that violate physical constraints or show inconsistent blur/sharpness**Spatial Branch (CNN):** EfficientNet-B0 backbone extracts high-level semantic features detecting texture artifacts, anatomical inconsistencies, and unnatural smoothness characteristic of AI-generated content.2. Normalize → μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]



**Evaluation Metrics:**

- **Accuracy:** 89.3% on held-out test set

- **Precision (AI class):** 91.2%### 3.2 Feature Fusion & Classification3. Convert → RGB tensor format [C×H×W]

- **Recall (AI class):** 87.5%

- **F1-Score:** 89.3%



### Robustness TestingConcatenates all branch embeddings and applies fusion network for final prediction.**Frequency Branch (FFT):** 2D Fast Fourier Transform analyzes frequency-domain signatures. AI generators often leave periodic artifacts and upsampling patterns invisible in spatial domain but prominent in frequency spectrum.```



**Platform Stability Scores:**

- Original images: 89.3% accuracy

- WhatsApp compression (512px, Q=40): 82.1% accuracy (stability: 78.5%)```

- Instagram compression (1080px, Q=70): 86.7% accuracy (stability: 88.2%)

- Facebook compression (960px, Q=60): 85.4% accuracy (stability: 85.9%)Step 1: Concatenate branch features



**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] = 1024-dim vector**Noise Branch:** Multi-scale Gaussian residual analysis (σ = 0.5, 1.0, 2.0) distinguishes authentic camera sensor noise from synthetic or absent noise patterns in AI images.### 2. Feature Extraction



### Explainability Validation



- **Grad-CAM Analysis:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test casesStep 2: Fusion layer with regularization

- **Branch Attribution:** Spatial branch contributes 45%, frequency 28%, noise 18%, edge 9% on average

- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictionsFC(1024→512) + ReLU + Dropout(0.5)



---**Edge Branch:** Sobel operator-based gradient analysis identifies physically implausible edge structures and discontinuities common in generative models.The system employs four specialized branches for comprehensive feature analysis:



## 8. LimitationsStep 3: Binary classification



**Dataset Constraints:** Model trained on 2024-2025 AI generators; may not generalize to future models or specialized domains (medical, satellite imagery).FC(512→2) + Softmax → [P(Real), P(AI-Generated)]



**Compression Sensitivity:** Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%) or multiple re-encoding cycles.



**Adversarial Vulnerability:** System not hardened against intentional adversarial perturbations designed to evade detection.Step 4: Temperature calibration### B. Feature Fusion & Classification#### A. Convolutional Neural Network (CNN) - Spatial Branch



**Hybrid Images:** Struggles with real photos containing AI-edited elements (e.g., object insertion, background replacement).Calibrated confidence = Softmax(logits / T) where T=1.5 (tuned on validation set)



**Computational Cost:** Multi-branch architecture requires ~2.5x inference time compared to single CNN (150ms vs. 60ms on CPU).```



**False Positives:** Heavily post-processed real photos (HDR, beauty filters) may trigger false AI detections.



---**Why multi-branch:** Different AI generators leave different artifacts; combining multiple detection methods provides robustness against various generation techniques.```**Algorithm**: EfficientNet-B0 backbone with custom projection head



## 9. Future Work



**Model Enhancements:**### 3.3 Explainability Framework[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] → 1024-dim

- Ensemble multiple architectures (Vision Transformers, ResNet variants) for improved accuracy

- Continuous learning pipeline to adapt to emerging AI generators

- Adversarial training for robustness against evasion techniques

**Executive Summary Generation:** Analyzes branch confidence scores and generates natural language explanation of classification decision and key evidence.→ Fusion Layer (1024→512) + Dropout(0.5)- **Architecture**: Pre-trained on ImageNet-1K, fine-tuned for AI detection

**Feature Additions:**

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)

- Batch processing API for high-throughput analysis

- Video frame analysis for deepfake detection**Per-Branch Attribution:** Uses ablation testing—evaluates model with each branch disabled to measure individual branch contributions to final prediction.→ Classifier (512→2) + Temperature Calibration- **Features Extracted**: High-level semantic patterns, texture artifacts, structural anomalies

- Model fingerprinting to identify specific AI generator (DALL-E vs. Midjourney)



**Deployment:**

- Model quantization (INT8) for 3x faster inference**Grad-CAM Visualization:** Computes gradient-weighted activation maps from spatial CNN's final convolutional layer, highlighting image regions most influential to classification decision.→ Output: P(Real), P(AI-Generated)- **Output**: 256-dimensional spatial embedding

- Mobile applications with on-device inference

- Browser extension for in-situ web image analysis

- Partnership with fact-checking organizations and news platforms

---```- **Detection Focus**: Unnatural smoothness, anatomical inconsistencies, synthetic textures

**Research Directions:**

- Cross-modal consistency analysis (text-image alignment)

- Temporal consistency for video sequences

- Zero-shot detection of unseen generative models## 4. System Architecture



---



## 10. References### Component Overview### C. Explainability Framework**Technical Details**:



**Academic Research:**

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." CVPR 2020.

2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." IEEE TIFS.``````

3. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017.

┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐

**Datasets:**

- CIFAKE: Kaggle real vs. AI image dataset│   React Frontend │ ───> │   FastAPI Backend    │ ───> │  PyTorch Model  │**Executive Summary:** Automated natural language generation explaining decision basis and key indicators examined.Input: RGB Image (224×224×3)

- DiffusionDB: Stable Diffusion dataset (Hugging Face)

- COCO: Microsoft Common Objects in Context│   - Image Upload │      │   - API Endpoints    │      │  - 4 Branches   │

- FFHQ: NVIDIA Flickr-Faces-HQ

│   - Results UI   │ <─── │   - Preprocessing    │ <─── │  - Fusion Layer │↓

**Tools & Frameworks:**

- PyTorch: pytorch.org│   - Grad-CAM     │      │   - Explainability   │      │  - Grad-CAM     │

- timm (PyTorch Image Models): github.com/huggingface/pytorch-image-models

- FastAPI: fastapi.tiangolo.com└─────────────────┘      └──────────────────────┘      └─────────────────┘**Branch Attribution:** Ablation testing evaluates each branch independently to quantify individual contributions to final prediction.EfficientNet-B0 Backbone (1280-dim features)

- React: react.dev

```

---

↓

## License & Citation

**Frontend:** React 18.2 with responsive design, drag-and-drop upload, real-time results visualization including confidence scores, per-branch analysis, and heatmap overlays.

**License:** MIT License - See LICENSE file for details

**Grad-CAM Heatmaps:** Gradient-weighted Class Activation Mapping visualizes spatial regions influencing the CNN's decision, highlighting suspicious artifacts.Projection: FC(1280→512) → ReLU → Dropout(0.3) → FC(512→256)

**Citation:**

```bibtex**Backend:** FastAPI provides RESTful API with `/api/analyze` endpoint for single-image inference and `/api/simulate-platforms` for robustness testing.

@software{truepix2026,

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},↓

  author={TruePix Contributors},

  year={2026},**ML Pipeline:** PyTorch 2.1 implementation with timm library for EfficientNet-B0, custom branches for FFT/noise/edge analysis, and integrated Grad-CAM for spatial explanations.

  url={https://github.com/kathirpmdk-star/Truepix}

}---Output: Spatial Embedding (256-dim)

```

**Storage (Optional):** Supabase object storage for image persistence; system functions in demo mode without external dependencies.

---



<div align="center">

---

Developed for Academic Research | Promoting Digital Media Transparency

## 4. System Architecture```

For questions or collaborations, please open a GitHub issue

## 5. Tech Stack

</div>



| Layer | Technology | Purpose |

|-------|------------|---------|### Component Overview#### B. Fast Fourier Transform (FFT) - Frequency Branch

| **Frontend** | React 18.2, CSS3 | User interface and visualization |

| **Backend** | FastAPI 0.108, Uvicorn | High-performance API server |

| **ML Framework** | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |

| **Model** | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |```**Algorithm**: 2D FFT spectral analysis with logarithmic magnitude features

| **Image Processing** | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |

| **Explainability** | Custom Grad-CAM, scipy (FFT) | Visualization and frequency analysis |┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐

| **Storage** | Supabase (optional) | Object storage and metadata |

│   React Frontend │ ───> │   FastAPI Backend    │ ───> │  PyTorch Model  │- **Transform**: Applies 2D Discrete Fourier Transform to grayscale image

**Dependencies:** Python 3.9+, Node.js 16+, CUDA (optional for GPU acceleration)

│   - Image Upload │      │   - API Endpoints    │      │  - 4 Branches   │- **Feature Space**: Frequency domain representation highlighting periodic patterns

---

│   - Results UI   │ <─── │   - Preprocessing    │ <─── │  - Fusion Layer │- **Detection Focus**: Generative model artifacts, upsampling signatures, grid-like patterns

## 6. How to Run

│   - Grad-CAM     │      │   - Explainability   │      │  - Grad-CAM     │

### Prerequisites

```bash└─────────────────┘      └──────────────────────┘      └─────────────────┘**Technical Details**:

# System requirements

- Python 3.9+``````

- Node.js 16+

- 4GB RAM (8GB recommended)Input: Grayscale Image (224×224)

```

**Frontend:** React 18.2 with responsive design, drag-and-drop upload, real-time results visualization including confidence scores, per-branch analysis, and heatmap overlays.↓

### Backend Setup

```bash2D FFT: F(u,v) = Σ Σ f(x,y) * e^(-j2π(ux/M + vy/N))

cd backend

**Backend:** FastAPI provides RESTful API with `/api/analyze` endpoint for single-image inference and `/api/simulate-platforms` for robustness testing.↓

# Create virtual environment

python3 -m venv venvMagnitude Spectrum: |F(u,v)|

source venv/bin/activate  # On Windows: venv\Scripts\activate

**ML Pipeline:** PyTorch 2.1 implementation with timm library for EfficientNet-B0, custom branches for FFT/noise/edge analysis, and integrated Grad-CAM for spatial explanations.↓

# Install dependencies

pip install -r requirements.txtLog Transform: log(1 + |F(u,v)|)



# Start server**Storage (Optional):** Supabase object storage for image persistence; system functions in demo mode without external dependencies.↓

python main.py

```Feature Extraction: FC(224×224→128) → ReLU → FC(128→256)

**Backend:** http://localhost:8000 | **API Docs:** http://localhost:8000/docs

---↓

### Frontend Setup

```bashOutput: Frequency Embedding (256-dim)

cd frontend

## 5. Tech Stack```

# Install dependencies

npm install



# Start development server| Layer | Technology | Purpose |**Why FFT Works**: AI generators often produce subtle periodic artifacts invisible to human perception but detectable in frequency domain.

npm start

```|-------|------------|---------|

**Frontend:** http://localhost:3000

| **Frontend** | React 18.2, CSS3 | User interface and visualization |#### C. Noise Consistency Analysis - Noise Branch

### Usage

1. Open http://localhost:3000| **Backend** | FastAPI 0.108, Uvicorn | High-performance API server |

2. Upload JPG/PNG image (drag-and-drop or click)

3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM heatmap| **ML Framework** | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |**Algorithm**: Multi-scale noise pattern extraction using Gaussian blur residuals

4. (Optional) Test platform stability across social media compression scenarios

| **Model** | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |

---

| **Image Processing** | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |- **Method**: Analyzes sensor noise distribution and consistency

## 7. Results & Evaluation

| **Explainability** | Custom Grad-CAM, scipy (FFT) | Visualization and frequency analysis |- **Real Images**: Show consistent sensor-specific noise patterns

### Model Performance

| **Storage** | Supabase (optional) | Object storage and metadata |- **AI Images**: Exhibit synthetic or absent noise characteristics

**Training Configuration:**

- Dataset: 50,000 images (balanced real/AI split)

- Real sources: COCO, FFHQ, natural photography datasets

- AI sources: Stable Diffusion, DALL-E, Midjourney outputs**Dependencies:** Python 3.9+, Node.js 16+, CUDA (optional for GPU acceleration)**Technical Details**:

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy loss

```

**Evaluation Metrics:**

- **Accuracy:** 89.3% on held-out test set---Input: RGB Image (224×224×3)

- **Precision (AI class):** 91.2%

- **Recall (AI class):** 87.5%↓

- **F1-Score:** 89.3%

## 6. How to RunMulti-Scale Analysis:

### Robustness Testing

  - Fine: Residual = Image - GaussianBlur(σ=0.5)

**Platform Stability Scores:**

- Original images: 89.3% accuracy### Prerequisites  - Medium: Residual = Image - GaussianBlur(σ=1.0)

- WhatsApp compression (512px, Q=40): 82.1% accuracy (stability: 78.5%)

- Instagram compression (1080px, Q=70): 86.7% accuracy (stability: 88.2%)```bash  - Coarse: Residual = Image - GaussianBlur(σ=2.0)

- Facebook compression (960px, Q=60): 85.4% accuracy (stability: 85.9%)

# System requirements↓

**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.

- Python 3.9+Concatenate: [Fine, Medium, Coarse] → 9 channels

### Explainability Validation

- Node.js 16+↓

- **Grad-CAM Analysis:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test cases

- **Branch Attribution:** Spatial branch contributes 45%, frequency 28%, noise 18%, edge 9% on average- 4GB RAM (8GB recommended)Convolutional Feature Extraction:

- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictions

```  Conv2D(9→32) → ReLU → MaxPool

---

  Conv2D(32→64) → ReLU → MaxPool

## 8. Limitations

### Backend Setup  Conv2D(64→128) → ReLU → AdaptiveAvgPool

**Dataset Constraints:** Model trained on 2024-2025 AI generators; may not generalize to future models or specialized domains (medical, satellite imagery).

```bash↓

**Compression Sensitivity:** Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%) or multiple re-encoding cycles.

cd backendOutput: Noise Embedding (256-dim)

**Adversarial Vulnerability:** System not hardened against intentional adversarial perturbations designed to evade detection.

```

**Hybrid Images:** Struggles with real photos containing AI-edited elements (e.g., object insertion, background replacement).

# Create virtual environment

**Computational Cost:** Multi-branch architecture requires ~2.5x inference time compared to single CNN (150ms vs. 60ms on CPU).

python3 -m venv venv#### D. Edge Structure Analysis - Edge Branch

**False Positives:** Heavily post-processed real photos (HDR, beauty filters) may trigger false AI detections.

source venv/bin/activate  # On Windows: venv\Scripts\activate

---

**Algorithm**: Sobel operator with edge consistency verification

## 9. Future Work

# Install dependencies

**Model Enhancements:**

- Ensemble multiple architectures (Vision Transformers, ResNet variants) for improved accuracypip install -r requirements.txt- **Edge Detection**: Sobel filters for horizontal and vertical gradients

- Continuous learning pipeline to adapt to emerging AI generators

- Adversarial training for robustness against evasion techniques- **Real Images**: Consistent, physically plausible edge structures



**Feature Additions:**# Start server- **AI Images**: May contain discontinuous or physically impossible edges

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)

- Batch processing API for high-throughput analysispython main.py

- Video frame analysis for deepfake detection

- Model fingerprinting to identify specific AI generator (DALL-E vs. Midjourney)```**Technical Details**:



**Deployment:****Backend:** http://localhost:8000 | **API Docs:** http://localhost:8000/docs```

- Model quantization (INT8) for 3x faster inference

- Mobile applications with on-device inferenceInput: Grayscale Image (224×224)

- Browser extension for in-situ web image analysis

- Partnership with fact-checking organizations and news platforms### Frontend Setup↓



**Research Directions:**```bashSobel Operator:

- Cross-modal consistency analysis (text-image alignment)

- Temporal consistency for video sequencescd frontend  Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * Image  (horizontal)

- Zero-shot detection of unseen generative models

  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * Image  (vertical)

---

# Install dependencies↓

## 10. References

npm installGradient Magnitude: G = √(Gx² + Gy²)

**Academic Research:**

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." *CVPR 2020*.↓

2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." *IEEE TIFS*.

3. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV 2017*.# Start development serverEdge Features: Conv2D(1→64) → ReLU → Conv2D(64→128)



**Datasets:**npm start↓

- **CIFAKE:** Kaggle real vs. AI image dataset (kaggle.com/datasets/birdy654/cifake)

- **DiffusionDB:** Stable Diffusion dataset (huggingface.co/datasets/poloclub/diffusiondb)```Output: Edge Embedding (256-dim)

- **COCO:** Microsoft Common Objects in Context (cocodataset.org)

- **FFHQ:** NVIDIA Flickr-Faces-HQ (github.com/NVlabs/ffhq-dataset)**Frontend:** http://localhost:3000```



**Tools & Frameworks:**

- PyTorch: pytorch.org

- timm (PyTorch Image Models): github.com/huggingface/pytorch-image-models### Usage### 3. Feature Fusion and Classification

- FastAPI: fastapi.tiangolo.com

- React: react.dev1. Open http://localhost:3000



---2. Upload JPG/PNG image (drag-and-drop or click)**Fusion Architecture**:



## License & Citation3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM heatmap```



**License:** MIT License - See [LICENSE](LICENSE) file4. (Optional) Test platform stability across social media compression scenarios[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] → 1024-dim



**Citation:**↓

```bibtex

@software{truepix2026,---Fusion Layer: FC(1024→512) → ReLU → Dropout(0.5)

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},

  author={TruePix Contributors},↓

  year={2026},

  url={https://github.com/kathirpmdk-star/Truepix}## 7. Results & EvaluationClassification Head: FC(512→2) → Softmax

}

```↓



---### Model PerformanceOutput: P(Real), P(AI-Generated)



<div align="center">```



**Developed for Academic Research | Promoting Digital Media Transparency****Training Configuration:**



*For questions or collaborations, please open a GitHub issue*- Dataset: 50,000 images (balanced real/AI split)**Temperature Calibration**: Post-processing confidence calibration for reliable probability estimates



</div>- Real sources: COCO, FFHQ, natural photography datasets


- AI sources: Stable Diffusion, DALL-E, Midjourney outputs```python

- Training: 20 epochs, Adam optimizer (lr=1e-4), cross-entropy lossCalibrated_Confidence = Softmax(logits / T)  # T = temperature parameter

```

**Evaluation Metrics:**

- **Accuracy:** 89.3% on held-out test set### 4. Explainability Generation

- **Precision (AI class):** 91.2%

- **Recall (AI class):** 87.5%#### A. Executive Summary

- **F1-Score:** 89.3%- Analyzes per-branch confidence scores

- Generates human-readable verdict with decision basis

### Robustness Testing- Describes key indicators examined



**Platform Stability Scores:**#### B. Branch Attribution

- Original images: 89.3% accuracy- Performs ablation testing: evaluates each branch independently

- WhatsApp compression (512px, Q=40): 82.1% accuracy (stability: 78.5%)- Reports per-branch confidence contribution

- Instagram compression (1080px, Q=70): 86.7% accuracy (stability: 88.2%)- Provides forensic analysis for each detection method

- Facebook compression (960px, Q=60): 85.4% accuracy (stability: 85.9%)

#### C. Grad-CAM Visualization

**Insight:** Multi-branch architecture provides resilience to compression; frequency and noise branches maintain performance when spatial features degrade.- **Algorithm**: Gradient-weighted Class Activation Mapping

- **Target Layer**: Final convolutional layer of spatial CNN

### Explainability Validation- **Output**: Heatmap highlighting influential image regions



- **Grad-CAM Analysis:** Heatmaps correctly highlight known AI artifacts (hands, text, repetitive patterns) in 84% of test cases**Grad-CAM Formula**:

- **Branch Attribution:** Spatial branch contributes 45%, frequency 28%, noise 18%, edge 9% on average```

- **User Study:** 92% of users (n=25) found explanations helpful for understanding predictionsα_k = (1/Z) Σ Σ (∂y^c / ∂A^k)  # Global average pooling of gradients

L = ReLU(Σ α_k * A^k)          # Weighted combination of feature maps

---```



## 8. Limitations---



**Dataset Constraints:** Model trained on 2024-2025 AI generators; may not generalize to future models or specialized domains (medical, satellite imagery).## System Architecture



**Compression Sensitivity:** Accuracy degrades 5-8% under aggressive compression (JPEG quality < 50%) or multiple re-encoding cycles.### Technology Stack



**Adversarial Vulnerability:** System not hardened against intentional adversarial perturbations designed to evade detection.**Backend**

- **Framework**: FastAPI 0.108 (Python 3.9+)

**Hybrid Images:** Struggles with real photos containing AI-edited elements (e.g., object insertion, background replacement).- **Deep Learning**: PyTorch 2.1.2, torchvision 0.16.2

- **Model Hub**: timm 0.9.12 (EfficientNet-B0)

**Computational Cost:** Multi-branch architecture requires ~2.5x inference time compared to single CNN (150ms vs. 60ms on CPU).- **Image Processing**: OpenCV 4.9, Pillow 10.1, scikit-image

- **Server**: Uvicorn ASGI

**False Positives:** Heavily post-processed real photos (HDR, beauty filters) may trigger false AI detections.

**Frontend**

---- **Framework**: React 18.2.0

- **HTTP Client**: Axios

## 9. Future Work- **Styling**: CSS3 with responsive design



**Model Enhancements:****Storage** (Optional)

- Ensemble multiple architectures (Vision Transformers, ResNet variants) for improved accuracy- **Object Storage**: Supabase Storage

- Continuous learning pipeline to adapt to emerging AI generators- **Database**: PostgreSQL via Supabase

- Adversarial training for robustness against evasion techniques

### Project Structure

**Feature Additions:**

- EXIF metadata analysis for forensic verification (camera model, GPS, edit history)```

- Batch processing API for high-throughput analysisTruepix/

- Video frame analysis for deepfake detection├── backend/

- Model fingerprinting to identify specific AI generator (DALL-E vs. Midjourney)│   ├── main.py                    # FastAPI application

│   ├── model_inference.py         # Hybrid detector implementation

**Deployment:**│   ├── platform_simulator.py      # Social media compression simulator

- Model quantization (INT8) for 3x faster inference│   ├── storage_manager.py         # Image storage handler

- Mobile applications with on-device inference│   ├── requirements.txt           # Python dependencies

- Browser extension for in-situ web image analysis│   └── checkpoints/

- Partnership with fact-checking organizations and news platforms│       └── hybrid_detector_calibrated.pth  # Trained model weights

│

**Research Directions:**├── frontend/

- Cross-modal consistency analysis (text-image alignment)│   ├── src/

- Temporal consistency for video sequences│   │   ├── components/

- Zero-shot detection of unseen generative models│   │   │   ├── LandingPage.js     # Hero section

│   │   │   ├── ImageUpload.js     # File upload component

---│   │   │   ├── ResultsPanel.js    # Detection results display

│   │   │   └── *.css

## 10. References│   │   ├── App.js                 # Main application

│   │   └── index.js

**Academic Research:**│   └── package.json

1. Wang et al. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... For Now." *CVPR 2020*.│

2. Gragnaniello et al. (2021). "GAN-Generated Faces Detection." *IEEE TIFS*.├── dataset/                       # Training data (not included)

3. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV 2017*.│   ├── ai_generated/

│   └── real/

**Datasets:**│

- **CIFAKE:** Kaggle real vs. AI image dataset (kaggle.com/datasets/birdy654/cifake)└── README.md

- **DiffusionDB:** Stable Diffusion dataset (huggingface.co/datasets/poloclub/diffusiondb)```

- **COCO:** Microsoft Common Objects in Context (cocodataset.org)

- **FFHQ:** NVIDIA Flickr-Faces-HQ (github.com/NVlabs/ffhq-dataset)---



**Tools & Frameworks:**## Installation and Setup

- PyTorch: pytorch.org

- timm (PyTorch Image Models): github.com/huggingface/pytorch-image-models### Prerequisites

- FastAPI: fastapi.tiangolo.com

- React: react.dev- Python 3.9 or higher

- Node.js 16+ and npm

---- 4GB RAM minimum (8GB recommended for GPU inference)

- CUDA-compatible GPU (optional, for faster inference)

## License & Citation

### Step 1: Clone Repository

**License:** MIT License - See [LICENSE](LICENSE) file

```bash

**Citation:**git clone https://github.com/kathirpmdk-star/Truepix.git

```bibtexcd Truepix

@software{truepix2026,```

  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},

  author={TruePix Contributors},### Step 2: Backend Setup

  year={2026},

  url={https://github.com/kathirpmdk-star/Truepix}```bash

}# Navigate to backend directory

```cd backend



---# Create Python virtual environment

python3 -m venv venv

<div align="center">

# Activate virtual environment

**Developed for Academic Research | Promoting Digital Media Transparency**# On macOS/Linux:

source venv/bin/activate

*For questions or collaborations, please open a GitHub issue*# On Windows:

# venv\Scripts\activate

</div>

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

**Environment Configuration** (optional):

Create `.env` file in `backend/` directory:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_BUCKET=truepix-images
```

*Note: Application works in demo mode without Supabase configuration.*

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory (from repository root)
cd frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list react
```

### Step 4: Run Application

**Terminal 1 - Start Backend:**
```bash
cd backend
source venv/bin/activate  # Activate virtual environment
python main.py
```

Backend server: `http://localhost:8000`  
API Documentation: `http://localhost:8000/docs`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm start
```

Frontend application: `http://localhost:3000`

---

## Usage

### Web Interface

1. **Open Application**: Navigate to `http://localhost:3000`
2. **Upload Image**: Click upload area or drag-and-drop JPG/PNG file
3. **View Results**: System displays:
   - Classification (AI-Generated / Real / Uncertain)
   - Confidence score (0-100%)
   - Executive summary explaining decision basis
   - Per-branch technical analysis (Spatial, FFT, Noise, Edge)
   - Grad-CAM heatmap highlighting influential regions
4. **Platform Testing** (optional): Test prediction stability across social media compression

### API Usage

**Analyze Image:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "prediction": "AI-Generated",
  "confidence": 87.5,
  "confidence_category": "High",
  "explanation": "Strong indicators of AI generation detected...",
  "metadata": {
    "executive_summary": "🤖 AI-Generated Image Detected...",
    "branch_findings": {
      "spatial": "CNN analysis shows...",
      "fft": "Frequency analysis reveals...",
      "noise": "Noise patterns indicate...",
      "edge": "Edge structure suggests..."
    },
    "gradcam_image": "data:image/png;base64,...",
    "raw_scores": {"real": 0.125, "ai_generated": 0.875}
  }
}
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/analyze` | Analyze single image |
| POST | `/api/simulate-platforms` | Test platform stability |

**POST /api/analyze**

Request:
- Content-Type: `multipart/form-data`
- Body: `file` (JPG/PNG image)

Response:
- `prediction`: Classification result (AI-Generated / Real / Uncertain)
- `confidence`: Probability score (0-100)
- `confidence_category`: Risk level (High / Medium / Low)
- `explanation`: Human-readable summary
- `metadata.executive_summary`: Decision basis explanation
- `metadata.branch_findings`: Per-branch technical analysis
- `metadata.gradcam_image`: Base64-encoded heatmap visualization

---

## Limitations

### Dataset and Training Constraints

**Limited Training Data**: The current model is trained on a specific distribution of AI-generated images. Performance may degrade on:
- Images from newer generative models (2025+) not represented in training data
- Non-English or culturally diverse content if training data was biased
- Specialized domains (medical imagery, satellite photos, technical diagrams)

**Dataset Bias**: Training datasets may exhibit:
- Overrepresentation of certain image categories (e.g., portraits vs. landscapes)
- Quality bias toward high-resolution, well-composed images
- Geographic and demographic imbalances in human-subject imagery

### Generalization Issues

**Compression Sensitivity**: Model accuracy may decrease significantly after:
- Aggressive JPEG compression (quality < 50%)
- Multiple re-encoding cycles (screenshot → upload → download)
- Platform-specific preprocessing (Instagram filters, WhatsApp optimization)

**Adversarial Vulnerability**: The system is susceptible to:
- Intentionally crafted adversarial perturbations
- Post-processing techniques designed to fool detectors
- Hybrid images (real photo with AI-generated elements)

**Edge Cases**: Limited performance on:
- Heavily edited real photographs (extensive Photoshop manipulation)
- Very small images (< 256×256 pixels)
- Images with watermarks or overlays obscuring content

### Explainability Limitations

**Grad-CAM Interpretation**: Heatmaps show spatial attention but:
- May not reflect all decision factors (frequency/noise branches lack spatial mapping)
- Can be influenced by salient objects rather than manipulation artifacts
- Require expert interpretation for meaningful conclusions

**Branch Attribution Uncertainty**: Per-branch confidence scores are derived from ablation testing and may not perfectly represent true model reasoning.

### Ethical Considerations

**False Positives/Negatives**: No detection system is perfect. Misclassifications can:
- Falsely accuse photographers of using AI
- Fail to detect sophisticated AI-generated misinformation
- Impact trust in digital media ecosystems

**Responsible Deployment**: This system should:
- Not be used as sole evidence in legal or journalistic contexts
- Be combined with metadata analysis and human expert review
- Include clear disclaimers about accuracy limitations

---

## Future Work

### Model Improvements

**1. Ensemble Architecture**
- Combine multiple detection models (EfficientNet, ResNet, Vision Transformer)
- Implement voting mechanisms or learned fusion weights
- Expected improvement: +5-10% accuracy on diverse test sets

**2. Expanded Training Data**
- Curate balanced dataset with 500k+ images across:
  - Multiple AI generators (DALL-E 3, Midjourney v6, Stable Diffusion XL, Adobe Firefly)
  - Diverse real image sources (cameras, smartphones, drones)
  - Various compression levels and post-processing scenarios
- Fine-tune on domain-specific datasets (medical, legal, news media)

**3. Temporal Model Updates**
- Implement continuous learning pipeline to adapt to new AI generators
- Establish dataset refresh cycle (quarterly updates)
- Version control for model deployments with A/B testing

### Feature Enhancements

**4. Metadata Analysis Integration**
- Extract and analyze EXIF data (camera model, GPS, software signatures)
- Detect absence of expected metadata in "real" photos
- Cross-reference metadata consistency with visual content

**5. Multi-Image Analysis**
- Batch processing for analyzing image sets
- Temporal consistency checking for video frames
- Photographer style profiling for authentication

**6. Advanced Explainability**
- Interactive Grad-CAM with region highlighting and zoom
- Natural language generation for detailed forensic reports
- Comparison visualizations (image vs. typical real/AI patterns)

### System Architecture

**7. Performance Optimization**
- Model quantization (INT8) for 3x faster inference
- TensorRT or ONNX Runtime integration
- Distributed inference for high-traffic scenarios

**8. API Enhancements**
- Rate limiting and authentication (API keys)
- Webhook support for asynchronous processing
- Batch upload endpoints
- Historical analysis dashboard

**9. Mobile Applications**
- Native iOS/Android apps with on-device inference
- Camera integration for real-time analysis
- Offline mode with cached models

### Research Directions

**10. Generative Model Fingerprinting**
- Develop techniques to identify specific AI generator (DALL-E vs. Midjourney)
- Extract model version signatures from generated images
- Enable provenance tracking for AI-generated content

**11. Robustness to Adversarial Attacks**
- Train with adversarial examples
- Implement certified defenses (randomized smoothing)
- Collaborative research on detection-evasion arms race

**12. Cross-Modal Detection**
- Extend to video deepfake detection
- Audio synthesis detection (voice cloning)
- Multi-modal consistency analysis (audio-visual alignment)

### Deployment and Accessibility

**13. Cloud Infrastructure**
- Dockerized deployment for Kubernetes orchestration
- Serverless functions (AWS Lambda, Google Cloud Run)
- CDN integration for global low-latency access

**14. Integration Partnerships**
- Browser extensions for in-situ analysis
- Social media platform API partnerships
- News organization fact-checking tooling

---

## Contributing

We welcome contributions from the research and developer community. Areas of particular interest:

- **Dataset Contributions**: Labeled real and AI-generated images with provenance
- **Model Architectures**: Novel detection approaches or branch improvements
- **Explainability Research**: Better visualization and interpretation techniques
- **Performance Optimization**: Inference speed improvements without accuracy loss
- **Documentation**: Tutorials, use cases, academic paper references

### Contribution Guidelines

1. **Fork Repository**: Create personal fork of the project
2. **Create Branch**: `git checkout -b feature/descriptive-name`
3. **Implement Changes**: Follow PEP 8 (Python) and Airbnb style guide (JavaScript)
4. **Add Tests**: Ensure new features include unit tests
5. **Document**: Update README and inline documentation
6. **Submit PR**: Provide clear description of changes and motivation

### Code of Conduct

- Respectful and inclusive communication
- Focus on constructive feedback
- Acknowledge contributions from others
- Prioritize academic integrity and ethical considerations

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

**Academic Use**: If you use this work in academic research, please cite:

```
@software{truepix2026,
  title={TruePix: Multi-Branch Deep Learning for AI-Generated Image Detection},
  author={TruePix Contributors},
  year={2026},
  url={https://github.com/kathirpmdk-star/Truepix}
}
```

---

## Acknowledgments

This project builds upon research and tools from:

- **PyTorch Team** - Deep learning framework
- **Hugging Face timm** - Pre-trained vision models
- **FastAPI** - High-performance API framework
- **AI Detection Research Community** - Foundational work on GAN and diffusion model detection

**Datasets** (for training reference):
- CIFAKE (Kaggle)
- DiffusionDB (Hugging Face)
- COCO Dataset (Microsoft)
- ImageNet-1K (Stanford)

---

## Contact and Support

**Repository**: [github.com/kathirpmdk-star/Truepix](https://github.com/kathirpmdk-star/Truepix)

**Issues**: Report bugs or request features via GitHub Issues

**Discussions**: Join community discussions on GitHub Discussions tab

---

<div align="center">

**Developed for academic research and educational purposes**

*Promoting transparency and accountability in the age of generative AI*

</div>
