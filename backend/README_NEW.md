# TruePix Backend - Modular AI Detection System

## 🎯 Overview

This is the updated TruePix backend with a **modular, production-ready architecture** for detecting AI-generated images.

### Key Features

✅ **Sequential Processing Pipeline**
- Store image in SQLite database
- Retrieve and preprocess (resize 224×224, normalize, strip metadata)
- CNN analysis (EfficientNet-B0)
- FFT frequency analysis
- Noise residual analysis
- Weighted score fusion

✅ **Modular Architecture**
- `database.py` - SQLite database for image storage
- `preprocessing.py` - Image preprocessing and normalization
- `cnn_detector.py` - CNN-based detection (weight: 0.6)
- `fft_analyzer.py` - Frequency domain analysis (weight: 0.2)
- `noise_analyzer.py` - Noise pattern analysis (weight: 0.1)
- `score_fusion.py` - Weighted score combination
- `main_new.py` - FastAPI server with /analyze-image endpoint

✅ **Comprehensive Analysis**
- Individual scores from each module
- Weighted fusion with confidence metrics
- Human-readable explanations
- Detailed feature analysis

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
chmod +x start_backend.sh
./start_backend.sh
```

### 2. Or Manual Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main_new.py
```

## 📡 API Endpoints

### POST /analyze-image

Upload an image for AI detection analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze-image" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "image_id": "uuid",
  "final_score": 0.75,
  "prediction": "AI-Generated",
  "confidence": 0.82,
  "individual_scores": {
    "cnn_score": 0.80,
    "fft_score": 0.65,
    "noise_score": 0.70
  },
  "score_breakdown": {
    "cnn_contribution": 0.48,
    "fft_contribution": 0.13,
    "noise_contribution": 0.07
  },
  "explanation": "**AI-Generated** with high confidence (75.0%). Key findings: CNN detected AI-typical patterns; FFT found frequency domain anomalies; Noise analysis revealed synthetic patterns. Detected: unnatural texture smoothness, regular periodic artifacts, abnormal noise variance.",
  "detailed_analysis": {
    "cnn": "High probability of AI generation detected; unnatural texture smoothness; abnormal color saturation patterns; (high confidence)",
    "fft": "Significant frequency domain anomalies detected; unusual frequency patterns; regular periodic artifacts (typical of generators)",
    "noise": "Significant noise anomalies detected; unusual noise variance (over-smoothed or synthetic); low noise entropy (structured rather than random)"
  },
  "processing_time": 2.45,
  "timestamp": "2026-01-09T10:30:00.000Z"
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "database": true,
    "preprocessor": true,
    "cnn_detector": true,
    "fft_analyzer": true,
    "noise_analyzer": true,
    "score_fusion": true
  },
  "timestamp": "2026-01-09T10:30:00.000Z"
}
```

### GET /image/{image_id}

Retrieve information about a previously analyzed image.

**Response:**
```json
{
  "image_info": {
    "id": "uuid",
    "filename": "image.jpg",
    "upload_timestamp": "2026-01-09T10:30:00",
    "file_size": 245678,
    "image_format": "JPEG",
    "width": 1920,
    "height": 1080
  },
  "analysis_result": {
    "cnn_score": 0.80,
    "fft_score": 0.65,
    "noise_score": 0.70,
    "final_score": 0.75,
    "prediction": "AI-Generated",
    "explanation": "...",
    "processing_time": 2.45
  }
}
```

## 🏗️ Architecture

### Processing Pipeline

```
1. Upload Image
   ↓
2. Validate & Store in SQLite Database
   ↓
3. Retrieve Image from Database
   ↓
4. Preprocess
   - Resize to 224×224
   - Normalize compression (JPEG quality=90)
   - Strip metadata (EXIF, IPTC, XMP)
   ↓
5. CNN Analysis (Weight: 0.6)
   - EfficientNet-B0 inference
   - Feature extraction
   - Texture & color analysis
   ↓
6. FFT Analysis (Weight: 0.2)
   - Frequency domain transformation
   - Periodic artifact detection
   - Radial profile analysis
   ↓
7. Noise Analysis (Weight: 0.1)
   - High-frequency residual extraction
   - Variance & entropy analysis
   - Spatial consistency check
   ↓
8. Score Fusion
   - Weighted combination
   - Confidence calculation
   - Explanation generation
   ↓
9. Store Results & Return Response
```

### Module Descriptions

#### 1. Database (`database.py`)
- SQLite database for persistent storage
- Stores uploaded images as BLOBs
- Stores analysis results with scores
- Provides retrieval and query methods

#### 2. Preprocessing (`preprocessing.py`)
- Resizes images to 224×224
- Normalizes JPEG compression (quality=90)
- Strips all metadata
- Converts to numpy arrays

#### 3. CNN Detector (`cnn_detector.py`)
- EfficientNet-B0 pretrained on ImageNet
- Modified for binary classification
- Analyzes texture, color, and patterns
- Returns probability score (0-1)

#### 4. FFT Analyzer (`fft_analyzer.py`)
- 2D Fourier Transform analysis
- Detects periodic artifacts
- Analyzes frequency distribution
- Identifies unnatural patterns

#### 5. Noise Analyzer (`noise_analyzer.py`)
- Extracts high-frequency noise residual
- Analyzes variance and entropy
- Checks spatial consistency
- Detects synthetic noise patterns

#### 6. Score Fusion (`score_fusion.py`)
- Combines scores with weights:
  - CNN: 0.6
  - FFT: 0.2
  - Noise: 0.1
  - Edge: 0.1 (optional)
- Calculates confidence
- Generates explanations

## 📊 Score Interpretation

- **0.0 - 0.3**: Likely Real
- **0.3 - 0.5**: Uncertain / Low Confidence
- **0.5 - 0.7**: Likely AI-Generated
- **0.7 - 1.0**: High Probability AI-Generated

## 🔧 Configuration

Edit `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Supabase Cloud Storage
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=truepix-images
```

## 📦 Dependencies

Key dependencies:
- `fastapi` - Web framework
- `torch` & `torchvision` - Deep learning
- `timm` - EfficientNet model
- `opencv-python` - Image processing
- `scipy` - FFT and scientific computing
- `pillow` - Image handling
- `numpy` - Numerical computing

See `requirements.txt` for complete list.

## 🧪 Testing

Test the API using curl:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Analyze an image
curl -X POST http://localhost:8000/analyze-image \
  -F "file=@test_image.jpg"

# Get image info
curl http://localhost:8000/image/{image_id}
```

Or use the interactive API docs:
- Open browser: `http://localhost:8000/docs`

## 📝 Database

Images and results are stored in `truepix.db` (SQLite database).

**Tables:**
- `images` - Uploaded image data and metadata
- `analysis_results` - Analysis scores and predictions

## 🎓 How It Works

1. **CNN Analysis**: Uses deep learning to detect patterns typical of AI-generated images
2. **FFT Analysis**: Examines frequency domain for artifacts introduced by generative models
3. **Noise Analysis**: Studies noise patterns that differ between real and synthetic images
4. **Score Fusion**: Combines all analyses with weighted averaging for final prediction

## 📞 Support

For issues or questions, check:
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

**Version:** 2.0.0  
**Last Updated:** January 9, 2026
