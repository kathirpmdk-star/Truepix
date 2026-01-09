# TruePix - AI-Generated Image Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A multi-branch deep learning system for detecting AI-generated images with explainability**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Overview

TruePix is an academic research project implementing a hybrid multi-branch deep learning architecture for detecting AI-generated images. The system combines spatial, frequency-domain, noise-pattern, and edge-structure analysis to provide robust classification with comprehensive explainability features including Grad-CAM visualizations and per-branch decision attribution.

### Key Features

- **Hybrid Multi-Branch Architecture**: Combines four complementary detection approaches
- **Explainability**: Provides decision basis, per-branch analysis, and Grad-CAM heatmaps
- **Platform Robustness Testing**: Evaluates stability across social media compression scenarios
- **Real-time Analysis**: FastAPI backend with React frontend for immediate results

---

## Methodology

### 1. Image Preprocessing

Input images undergo standardized preprocessing to ensure consistent model input:

```python
Preprocessing Pipeline:
1. Resize → 224×224 pixels (bicubic interpolation)
2. Normalize → μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]
3. Convert → RGB tensor format [C×H×W]
```

### 2. Feature Extraction

The system employs four specialized branches for comprehensive feature analysis:

#### A. Convolutional Neural Network (CNN) - Spatial Branch

**Algorithm**: EfficientNet-B0 backbone with custom projection head

- **Architecture**: Pre-trained on ImageNet-1K, fine-tuned for AI detection
- **Features Extracted**: High-level semantic patterns, texture artifacts, structural anomalies
- **Output**: 256-dimensional spatial embedding
- **Detection Focus**: Unnatural smoothness, anatomical inconsistencies, synthetic textures

**Technical Details**:
```
Input: RGB Image (224×224×3)
↓
EfficientNet-B0 Backbone (1280-dim features)
↓
Projection: FC(1280→512) → ReLU → Dropout(0.3) → FC(512→256)
↓
Output: Spatial Embedding (256-dim)

```

#### B. Fast Fourier Transform (FFT) - Frequency Branch

**Algorithm**: 2D FFT spectral analysis with logarithmic magnitude features

- **Transform**: Applies 2D Discrete Fourier Transform to grayscale image
- **Feature Space**: Frequency domain representation highlighting periodic patterns
- **Detection Focus**: Generative model artifacts, upsampling signatures, grid-like patterns

**Technical Details**:
```
Input: Grayscale Image (224×224)
↓
2D FFT: F(u,v) = Σ Σ f(x,y) * e^(-j2π(ux/M + vy/N))
↓
Magnitude Spectrum: |F(u,v)|
↓
Log Transform: log(1 + |F(u,v)|)
↓
Feature Extraction: FC(224×224→128) → ReLU → FC(128→256)
↓
Output: Frequency Embedding (256-dim)
```

**Why FFT Works**: AI generators often produce subtle periodic artifacts invisible to human perception but detectable in frequency domain.

#### C. Noise Consistency Analysis - Noise Branch

**Algorithm**: Multi-scale noise pattern extraction using Gaussian blur residuals

- **Method**: Analyzes sensor noise distribution and consistency
- **Real Images**: Show consistent sensor-specific noise patterns
- **AI Images**: Exhibit synthetic or absent noise characteristics

**Technical Details**:
```
Input: RGB Image (224×224×3)
↓
Multi-Scale Analysis:
  - Fine: Residual = Image - GaussianBlur(σ=0.5)
  - Medium: Residual = Image - GaussianBlur(σ=1.0)
  - Coarse: Residual = Image - GaussianBlur(σ=2.0)
↓
Concatenate: [Fine, Medium, Coarse] → 9 channels
↓
Convolutional Feature Extraction:
  Conv2D(9→32) → ReLU → MaxPool
  Conv2D(32→64) → ReLU → MaxPool
  Conv2D(64→128) → ReLU → AdaptiveAvgPool
↓
Output: Noise Embedding (256-dim)
```

#### D. Edge Structure Analysis - Edge Branch

**Algorithm**: Sobel operator with edge consistency verification

- **Edge Detection**: Sobel filters for horizontal and vertical gradients
- **Real Images**: Consistent, physically plausible edge structures
- **AI Images**: May contain discontinuous or physically impossible edges

**Technical Details**:
```
Input: Grayscale Image (224×224)
↓
Sobel Operator:
  Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * Image  (horizontal)
  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * Image  (vertical)
↓
Gradient Magnitude: G = √(Gx² + Gy²)
↓
Edge Features: Conv2D(1→64) → ReLU → Conv2D(64→128)
↓
Output: Edge Embedding (256-dim)
```

### 3. Feature Fusion and Classification

**Fusion Architecture**:
```
[Spatial(256) ⊕ FFT(256) ⊕ Noise(256) ⊕ Edge(256)] → 1024-dim
↓
Fusion Layer: FC(1024→512) → ReLU → Dropout(0.5)
↓
Classification Head: FC(512→2) → Softmax
↓
Output: P(Real), P(AI-Generated)
```

**Temperature Calibration**: Post-processing confidence calibration for reliable probability estimates

```python
Calibrated_Confidence = Softmax(logits / T)  # T = temperature parameter
```

### 4. Explainability Generation

#### A. Executive Summary
- Analyzes per-branch confidence scores
- Generates human-readable verdict with decision basis
- Describes key indicators examined

#### B. Branch Attribution
- Performs ablation testing: evaluates each branch independently
- Reports per-branch confidence contribution
- Provides forensic analysis for each detection method

#### C. Grad-CAM Visualization
- **Algorithm**: Gradient-weighted Class Activation Mapping
- **Target Layer**: Final convolutional layer of spatial CNN
- **Output**: Heatmap highlighting influential image regions

**Grad-CAM Formula**:
```
α_k = (1/Z) Σ Σ (∂y^c / ∂A^k)  # Global average pooling of gradients
L = ReLU(Σ α_k * A^k)          # Weighted combination of feature maps
```

---

## System Architecture

### Technology Stack

**Backend**
- **Framework**: FastAPI 0.108 (Python 3.9+)
- **Deep Learning**: PyTorch 2.1.2, torchvision 0.16.2
- **Model Hub**: timm 0.9.12 (EfficientNet-B0)
- **Image Processing**: OpenCV 4.9, Pillow 10.1, scikit-image
- **Server**: Uvicorn ASGI

**Frontend**
- **Framework**: React 18.2.0
- **HTTP Client**: Axios
- **Styling**: CSS3 with responsive design

**Storage** (Optional)
- **Object Storage**: Supabase Storage
- **Database**: PostgreSQL via Supabase

### Project Structure

```
Truepix/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── model_inference.py         # Hybrid detector implementation
│   ├── platform_simulator.py      # Social media compression simulator
│   ├── storage_manager.py         # Image storage handler
│   ├── requirements.txt           # Python dependencies
│   └── checkpoints/
│       └── hybrid_detector_calibrated.pth  # Trained model weights
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LandingPage.js     # Hero section
│   │   │   ├── ImageUpload.js     # File upload component
│   │   │   ├── ResultsPanel.js    # Detection results display
│   │   │   └── *.css
│   │   ├── App.js                 # Main application
│   │   └── index.js
│   └── package.json
│
├── dataset/                       # Training data (not included)
│   ├── ai_generated/
│   └── real/
│
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- Node.js 16+ and npm
- 4GB RAM minimum (8GB recommended for GPU inference)
- CUDA-compatible GPU (optional, for faster inference)

### Step 1: Clone Repository

```bash
git clone https://github.com/kathirpmdk-star/Truepix.git
cd Truepix
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

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
