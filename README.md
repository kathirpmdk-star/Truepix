TruePix – Robust AI vs Real Image Detection System
Overview
TruePix is a system designed to distinguish AI-generated images from real photographs using image forensic cues and deep learning.
The project addresses the growing challenge of synthetic media misuse, especially in social media, journalism, and digital evidence verification.
The system accepts an image as input and outputs:
Prediction: AI-generated or Real
Confidence score
Feature-based explanation (backend level)
Problem Statement
Recent advances in generative AI models (GANs, Diffusion models) have made synthetic images visually indistinguishable from real ones. This creates serious risks such as:
Spread of misinformation
Fake digital evidence
Loss of trust in visual media
Traditional visual inspection is no longer reliable.
Hence, an automated, robust detection system is required to analyze hidden artifacts left by AI generation pipelines.
Proposed Solution
TruePix detects AI-generated images by analyzing both spatial and frequency-domain inconsistencies that are commonly introduced during synthetic image generation.
Key objectives:
Detect AI-generated images with high robustness
Remain effective under compression and resizing
Provide a scalable backend inference API
Enable real-time image verification via a frontend interface
Methodology
The detection pipeline consists of the following stages:
1. Image Preprocessing
Image resizing and normalization
Color space standardization
Noise stabilization
2. Feature Extraction
Multiple forensic cues are analyzed:
Spatial artifacts using CNN-based feature extraction
Frequency-domain patterns (FFT-based anomalies)
Noise residual inconsistencies
Edge and texture irregularities
These features capture subtle differences between real camera pipelines and AI image generators.
3. Classification
Extracted features are passed to a trained deep learning classifier
The model outputs a probability score for AI vs Real classification
4. Inference API
Backend exposes a REST API
Accepts image uploads
Returns prediction and confidence score
System Architecture
User Image
   ↓
Frontend (Upload Interface)
   ↓
Backend API
   ↓
Preprocessing → Feature Extraction → Classifier
   ↓
Prediction + Confidence

Tech Stack
Backend
Python
FastAPI / Flask
OpenCV
PyTorch (model inference)
Frontend
React
HTML / CSS / JavaScript
Model & Analysis
CNN-based feature extraction
Frequency-domain analysis
Image forensics techniques
How to Run the Project
Backend
cd backend
pip install -r requirements.txt
python app.py
Frontend
cd frontend
npm install
npm start
Results & Evaluation
The system successfully identifies AI-generated images by detecting non-natural artifacts.
Tested on a mixed dataset containing:
Real photographs
AI-generated images from multiple generation sources
Performance was evaluated using accuracy and confidence-based prediction consistency.
Note: Due to dataset size constraints, datasets are not included in the repository.
Limitations
Performance depends on diversity of training data
Newer AI generators may introduce unseen patterns
Does not guarantee 100% accuracy for highly post-processed images
Future Work
Extend detection to video and deepfake content
Integrate explainability heatmaps for visual justification
Expand dataset to include newer AI image generators
Improve robustness against adversarial attacks
Ethical Considerations
This project is intended only for detection and verification purposes.
It does not generate or promote misuse of AI-generated content.
Author
Kathiravan M
Project: TruePix – AI vs Real Image Detection


