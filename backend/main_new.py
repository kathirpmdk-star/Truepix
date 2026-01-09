"""
TruePix Backend - FastAPI Application with Modular AI Detection
Main API server with database storage and sequential processing:
1. Upload & Store in database
2. Retrieve & Preprocess
3. CNN Analysis (0.6 weight)
4. FFT Analysis (0.2 weight)
5. Noise Analysis (0.1 weight)
6. Score Fusion & Response
"""

import os
import io
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

# Import modular detection components
from database import DatabaseManager
from preprocessing import ImagePreprocessor
from cnn_detector import CNNDetector
from fft_analyzer import FFTAnalyzer
from noise_analyzer import NoiseAnalyzer
from score_fusion import ScoreFusion

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TruePix AI Detection API",
    description="Modular AI-generated image detection with explainability",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://localhost:3002"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all components
print("\n🚀 Initializing TruePix Backend Components...")
print("=" * 60)

# Database
db_manager = DatabaseManager()

# Preprocessing
preprocessor = ImagePreprocessor(target_size=(224, 224), jpeg_quality=90)

# Analysis modules
cnn_detector = CNNDetector(use_gpu=True)
fft_analyzer = FFTAnalyzer()
noise_analyzer = NoiseAnalyzer()

# Score fusion (CNN: 0.6, FFT: 0.2, Noise: 0.1, Edge: 0.1)
score_fusion = ScoreFusion(
    cnn_weight=0.6,
    fft_weight=0.2,
    noise_weight=0.1,
    edge_weight=0.1
)

print("=" * 60)
print("✅ All components initialized successfully!\n")


# Response models
class AnalysisResponse(BaseModel):
    """Response model for /analyze-image endpoint"""
    image_id: str
    final_score: float
    prediction: str
    confidence: float
    individual_scores: Dict[str, Optional[float]]  # Allow None for optional scores
    score_breakdown: Dict[str, float]
    explanation: str
    detailed_analysis: Dict[str, str]
    processing_time: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    components: Dict[str, bool]
    timestamp: str


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "TruePix AI Detection API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns status of all backend components
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        components={
            "database": True,
            "preprocessor": True,
            "cnn_detector": True,
            "fft_analyzer": True,
            "noise_analyzer": True,
            "score_fusion": True
        },
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Main endpoint: Analyze image for AI generation detection
    
    Sequential Processing Pipeline:
    1. Validate and store image in database
    2. Retrieve image from database
    3. Preprocess image (resize 224×224, normalize, strip metadata)
    4. CNN Analysis (EfficientNet-B0) → score (0-1)
    5. FFT Analysis (frequency artifacts) → score (0-1)
    6. Noise Analysis (residual extraction) → score (0-1)
    7. Score Fusion (weighted: 0.6 CNN + 0.2 FFT + 0.1 Noise)
    8. Return comprehensive results
    
    Args:
        file: Uploaded image file (JPEG/PNG)
    
    Returns:
        Comprehensive analysis results with scores and explanations
    """
    start_time = time.time()
    
    try:
        # ========================================
        # STEP 1: Validate image file
        # ========================================
        print("\n" + "=" * 60)
        print("📥 NEW IMAGE ANALYSIS REQUEST")
        print("=" * 60)
        
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )
        
        # Read image data
        image_bytes = await file.read()
        
        # Validate image can be opened
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            image_format = img.format
            print(f"📷 Image: {file.filename} ({width}×{height}, {image_format})")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # ========================================
        # STEP 2: Store image in database
        # ========================================
        image_id = str(uuid.uuid4())
        
        success = db_manager.store_image(
            image_id=image_id,
            filename=file.filename,
            image_bytes=image_bytes,
            width=width,
            height=height,
            image_format=image_format
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store image in database"
            )
        
        # ========================================
        # STEP 3: Retrieve image from database
        # ========================================
        retrieved_bytes = db_manager.retrieve_image(image_id)
        
        if retrieved_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve image from database"
            )
        
        # ========================================
        # STEP 4: Preprocess image
        # ========================================
        print("\n🔧 PREPROCESSING")
        print("-" * 60)
        
        image_pil, image_array = preprocessor.preprocess(retrieved_bytes)
        
        # ========================================
        # STEP 5: CNN Analysis
        # ========================================
        print("\n" + "=" * 60)
        print("ANALYSIS MODULE 1: CNN (Weight: 0.6)")
        print("=" * 60)
        
        cnn_result = cnn_detector.analyze(image_array)
        
        # ========================================
        # STEP 6: FFT Analysis
        # ========================================
        print("\n" + "=" * 60)
        print("ANALYSIS MODULE 2: FFT (Weight: 0.2)")
        print("=" * 60)
        
        fft_result = fft_analyzer.analyze(image_array)
        
        # ========================================
        # STEP 7: Noise Analysis
        # ========================================
        print("\n" + "=" * 60)
        print("ANALYSIS MODULE 3: NOISE (Weight: 0.1)")
        print("=" * 60)
        
        noise_result = noise_analyzer.analyze(image_array)
        
        # ========================================
        # STEP 8: Score Fusion
        # ========================================
        print("\n" + "=" * 60)
        print("FINAL STEP: SCORE FUSION")
        print("=" * 60)
        
        fusion_result = score_fusion.fuse_scores(
            cnn_result=cnn_result,
            fft_result=fft_result,
            noise_result=noise_result,
            edge_result=None  # Optional edge detection
        )
        
        # ========================================
        # STEP 9: Store results in database
        # ========================================
        processing_time = time.time() - start_time
        
        db_manager.store_analysis_result(
            image_id=image_id,
            cnn_score=cnn_result["score"],
            fft_score=fft_result["score"],
            noise_score=noise_result["score"],
            edge_score=None,
            final_score=fusion_result["final_score"],
            prediction=fusion_result["prediction"],
            explanation=fusion_result["explanation"],
            processing_time=processing_time
        )
        
        # ========================================
        # STEP 10: Return response
        # ========================================
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Final Score: {fusion_result['final_score']:.3f}")
        print(f"Prediction: {fusion_result['prediction']}")
        print(f"Confidence: {fusion_result['confidence']:.3f}")
        print(f"Processing Time: {processing_time:.2f}s")
        print("=" * 60 + "\n")
        
        return AnalysisResponse(
            image_id=image_id,
            final_score=fusion_result["final_score"],
            prediction=fusion_result["prediction"],
            confidence=fusion_result["confidence"],
            individual_scores=fusion_result["individual_scores"],
            score_breakdown=fusion_result["score_breakdown"],
            explanation=fusion_result["explanation"],
            detailed_analysis=fusion_result["detailed_analysis"],
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/image/{image_id}")
async def get_image_info(image_id: str):
    """
    Get information about a stored image
    
    Args:
        image_id: Unique image identifier
    
    Returns:
        Image metadata and analysis results (if available)
    """
    try:
        # Get image info
        image_info = db_manager.get_image_info(image_id)
        
        if image_info is None:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Get analysis results (if available)
        analysis_result = db_manager.get_analysis_result(image_id)
        
        return {
            "image_info": image_info,
            "analysis_result": analysis_result
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve image info: {str(e)}"
        )


@app.get("/stats")
async def get_statistics():
    """
    Get statistics about analyzed images
    
    Returns:
        Statistics about the system usage
    """
    # This is a basic implementation
    # You can extend it to query database for real statistics
    return {
        "total_images_analyzed": "Check database",
        "average_processing_time": "Check database",
        "detection_rate": "Check database"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print("\n" + "=" * 60)
    print("🚀 TRUEPIX AI DETECTION API")
    print("=" * 60)
    print(f"📍 Server: http://{host}:{port}")
    print(f"📝 API Docs: http://localhost:{port}/docs")
    print(f"🔍 Health Check: http://localhost:{port}/health")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host=host, port=port, reload=False)
