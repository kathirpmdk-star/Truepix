# TruePix - AI-Generated Image Detection System# TruePix - AI-Generated Image Detection System



<div align="center"><div align="center">



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)

[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A hybrid multi-branch deep learning system for detecting AI-generated images with explainability**

**A multi-branch deep learning system for detecting AI-generated images with explainability**

[Demo](#6-how-to-run) • [Architecture](#4-system-architecture) • [Results](#7-results--evaluation)

</div>

</div>

---

---

## Table of Contents

## 1. Overview

- [Overview](#overview)

TruePix addresses the growing challenge of distinguishing AI-generated images from authentic photographs using a novel hybrid detection architecture. The system combines four complementary analysis methods—spatial CNN features, frequency-domain analysis, noise pattern consistency, and edge structure verification—to achieve robust classification with comprehensive explainability.- [Methodology](#methodology)

- [System Architecture](#system-architecture)

**Key Contributions:**- [Installation and Setup](#installation-and-setup)

- Hybrid multi-branch architecture leveraging spatial, frequency, noise, and edge domains- [Usage](#usage)

- Grad-CAM visualizations and per-branch attribution for model transparency- [API Reference](#api-reference)

- Platform robustness testing against social media compression (WhatsApp, Instagram, Facebook)- [Limitations](#limitations)

- End-to-end web application with real-time inference and explainable results- [Future Work](#future-work)

- [Contributing](#contributing)

**Use Cases:** Content moderation, journalism verification, digital forensics, academic research on synthetic media detection.

---

---

## Overview

## 2. Problem Statement

TruePix is an academic research project implementing a hybrid multi-branch deep learning architecture for detecting AI-generated images. The system combines spatial, frequency-domain, noise-pattern, and edge-structure analysis to provide robust classification with comprehensive explainability features including Grad-CAM visualizations and per-branch decision attribution.

### Challenge

Modern generative AI models (DALL-E, Midjourney, Stable Diffusion) produce photorealistic images that are increasingly difficult to distinguish from real photographs. This poses significant risks:### Key Features

- **Misinformation**: Fake news propagation using synthetic imagery

- **Fraud**: Identity theft, document forgery, deepfake scams- **Hybrid Multi-Branch Architecture**: Combines four complementary detection approaches

- **Trust Erosion**: Declining confidence in digital media authenticity- **Explainability**: Provides decision basis, per-branch analysis, and Grad-CAM heatmaps

- **Platform Robustness Testing**: Evaluates stability across social media compression scenarios

### Objective- **Real-time Analysis**: FastAPI backend with React frontend for immediate results

Develop a robust, explainable AI detection system that:

- Achieves high accuracy across diverse image types and AI generators---

- Provides interpretable explanations for predictions

- Maintains performance under real-world compression and post-processing## Methodology

- Offers real-time analysis suitable for production deployment

### 1. Image Preprocessing

---

Input images undergo standardized preprocessing to ensure consistent model input:

## 3. Methodology

```python

### A. Multi-Branch Feature ExtractionPreprocessing Pipeline:

1. Resize → 224×224 pixels (bicubic interpolation)

**Spatial Branch (CNN):** EfficientNet-B0 backbone extracts high-level semantic features detecting texture artifacts, anatomical inconsistencies, and unnatural smoothness characteristic of AI-generated content.2. Normalize → μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]

3. Convert → RGB tensor format [C×H×W]

**Frequency Branch (FFT):** 2D Fast Fourier Transform analyzes frequency-domain signatures. AI generators often leave periodic artifacts and upsampling patterns invisible in spatial domain but prominent in frequency spectrum.```



**Noise Branch:** Multi-scale Gaussian residual analysis (σ = 0.5, 1.0, 2.0) distinguishes authentic camera sensor noise from synthetic or absent noise patterns in AI images.### 2. Feature Extraction



**Edge Branch:** Sobel operator-based gradient analysis identifies physically implausible edge structures and discontinuities common in generative models.The system employs four specialized branches for comprehensive feature analysis:



### B. Feature Fusion & Classification#### A. Convolutional Neural Network (CNN) - Spatial Branch



```**Algorithm**: EfficientNet-B0 backbone with custom projection head

[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] → 1024-dim

→ Fusion Layer (1024→512) + Dropout(0.5)- **Architecture**: Pre-trained on ImageNet-1K, fine-tuned for AI detection

→ Classifier (512→2) + Temperature Calibration- **Features Extracted**: High-level semantic patterns, texture artifacts, structural anomalies

→ Output: P(Real), P(AI-Generated)- **Output**: 256-dimensional spatial embedding

```- **Detection Focus**: Unnatural smoothness, anatomical inconsistencies, synthetic textures



### C. Explainability Framework**Technical Details**:

```

**Executive Summary:** Automated natural language generation explaining decision basis and key indicators examined.Input: RGB Image (224×224×3)

↓

**Branch Attribution:** Ablation testing evaluates each branch independently to quantify individual contributions to final prediction.EfficientNet-B0 Backbone (1280-dim features)

↓

**Grad-CAM Heatmaps:** Gradient-weighted Class Activation Mapping visualizes spatial regions influencing the CNN's decision, highlighting suspicious artifacts.Projection: FC(1280→512) → ReLU → Dropout(0.3) → FC(512→256)

↓

---Output: Spatial Embedding (256-dim)



## 4. System Architecture```



### Component Overview#### B. Fast Fourier Transform (FFT) - Frequency Branch



```**Algorithm**: 2D FFT spectral analysis with logarithmic magnitude features

┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐

│   React Frontend │ ───> │   FastAPI Backend    │ ───> │  PyTorch Model  │- **Transform**: Applies 2D Discrete Fourier Transform to grayscale image

│   - Image Upload │      │   - API Endpoints    │      │  - 4 Branches   │- **Feature Space**: Frequency domain representation highlighting periodic patterns

│   - Results UI   │ <─── │   - Preprocessing    │ <─── │  - Fusion Layer │- **Detection Focus**: Generative model artifacts, upsampling signatures, grid-like patterns

│   - Grad-CAM     │      │   - Explainability   │      │  - Grad-CAM     │

└─────────────────┘      └──────────────────────┘      └─────────────────┘**Technical Details**:

``````

Input: Grayscale Image (224×224)

**Frontend:** React 18.2 with responsive design, drag-and-drop upload, real-time results visualization including confidence scores, per-branch analysis, and heatmap overlays.↓

2D FFT: F(u,v) = Σ Σ f(x,y) * e^(-j2π(ux/M + vy/N))

**Backend:** FastAPI provides RESTful API with `/api/analyze` endpoint for single-image inference and `/api/simulate-platforms` for robustness testing.↓

Magnitude Spectrum: |F(u,v)|

**ML Pipeline:** PyTorch 2.1 implementation with timm library for EfficientNet-B0, custom branches for FFT/noise/edge analysis, and integrated Grad-CAM for spatial explanations.↓

Log Transform: log(1 + |F(u,v)|)

**Storage (Optional):** Supabase object storage for image persistence; system functions in demo mode without external dependencies.↓

Feature Extraction: FC(224×224→128) → ReLU → FC(128→256)

---↓

Output: Frequency Embedding (256-dim)

## 5. Tech Stack```



| Layer | Technology | Purpose |**Why FFT Works**: AI generators often produce subtle periodic artifacts invisible to human perception but detectable in frequency domain.

|-------|------------|---------|

| **Frontend** | React 18.2, CSS3 | User interface and visualization |#### C. Noise Consistency Analysis - Noise Branch

| **Backend** | FastAPI 0.108, Uvicorn | High-performance API server |

| **ML Framework** | PyTorch 2.1.2, torchvision 0.16 | Deep learning inference |**Algorithm**: Multi-scale noise pattern extraction using Gaussian blur residuals

| **Model** | EfficientNet-B0 (timm 0.9.12) | Pre-trained CNN backbone |

| **Image Processing** | OpenCV 4.9, Pillow 10.1, scikit-image | Preprocessing and analysis |- **Method**: Analyzes sensor noise distribution and consistency

| **Explainability** | Custom Grad-CAM, scipy (FFT) | Visualization and frequency analysis |- **Real Images**: Show consistent sensor-specific noise patterns

| **Storage** | Supabase (optional) | Object storage and metadata |- **AI Images**: Exhibit synthetic or absent noise characteristics



**Dependencies:** Python 3.9+, Node.js 16+, CUDA (optional for GPU acceleration)**Technical Details**:

```

---Input: RGB Image (224×224×3)

↓

## 6. How to RunMulti-Scale Analysis:

  - Fine: Residual = Image - GaussianBlur(σ=0.5)

### Prerequisites  - Medium: Residual = Image - GaussianBlur(σ=1.0)

```bash  - Coarse: Residual = Image - GaussianBlur(σ=2.0)

# System requirements↓

- Python 3.9+Concatenate: [Fine, Medium, Coarse] → 9 channels

- Node.js 16+↓

- 4GB RAM (8GB recommended)Convolutional Feature Extraction:

```  Conv2D(9→32) → ReLU → MaxPool

  Conv2D(32→64) → ReLU → MaxPool

### Backend Setup  Conv2D(64→128) → ReLU → AdaptiveAvgPool

```bash↓

cd backendOutput: Noise Embedding (256-dim)

```

# Create virtual environment

python3 -m venv venv#### D. Edge Structure Analysis - Edge Branch

source venv/bin/activate  # On Windows: venv\Scripts\activate

**Algorithm**: Sobel operator with edge consistency verification

# Install dependencies

pip install -r requirements.txt- **Edge Detection**: Sobel filters for horizontal and vertical gradients

- **Real Images**: Consistent, physically plausible edge structures

# Start server- **AI Images**: May contain discontinuous or physically impossible edges

python main.py

```**Technical Details**:

**Backend:** http://localhost:8000 | **API Docs:** http://localhost:8000/docs```

Input: Grayscale Image (224×224)

### Frontend Setup↓

```bashSobel Operator:

cd frontend  Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * Image  (horizontal)

  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * Image  (vertical)

# Install dependencies↓

npm installGradient Magnitude: G = √(Gx² + Gy²)

↓

# Start development serverEdge Features: Conv2D(1→64) → ReLU → Conv2D(64→128)

npm start↓

```Output: Edge Embedding (256-dim)

**Frontend:** http://localhost:3000```



### Usage### 3. Feature Fusion and Classification

1. Open http://localhost:3000

2. Upload JPG/PNG image (drag-and-drop or click)**Fusion Architecture**:

3. View results: classification, confidence, executive summary, branch analysis, Grad-CAM heatmap```

4. (Optional) Test platform stability across social media compression scenarios[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] → 1024-dim

↓

---Fusion Layer: FC(1024→512) → ReLU → Dropout(0.5)

↓

## 7. Results & EvaluationClassification Head: FC(512→2) → Softmax

↓

### Model PerformanceOutput: P(Real), P(AI-Generated)

```

**Training Configuration:**

- Dataset: 50,000 images (balanced real/AI split)**Temperature Calibration**: Post-processing confidence calibration for reliable probability estimates

- Real sources: COCO, FFHQ, natural photography datasets

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
