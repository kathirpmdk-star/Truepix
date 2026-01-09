"""
TruePix Backend - FastAPI Application
Main API server for AI image detection with platform simulation
"""

import os
import io
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

from storage_manager import StorageManager
from model_inference import AIDetectorModel
from platform_simulator import PlatformSimulator

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TruePix API",
    description="AI-generated image detection with explainability",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
storage_manager = StorageManager()
ai_detector = AIDetectorModel()
platform_simulator = PlatformSimulator()

# Response models
class AnalysisResponse(BaseModel):
    prediction: str  # "AI-Generated" or "Real"
    confidence: float
    risk_level: str  # "High" / "Medium" / "Uncertain"
    explanation: str
    image_url: str
    metadata: Dict[str, Any]

class PlatformSimulationResponse(BaseModel):
    platform: str
    original_result: Dict[str, Any]
    platform_results: Dict[str, Dict[str, Any]]
    stability_score: float


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "TruePix API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.post("/api/upload", response_model=Dict[str, str])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image to object storage and return public URL
    
    Args:
        file: Image file (JPG/PNG)
    
    Returns:
        image_id: Unique identifier
        image_url: Public URL to access the image
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPG and PNG images are supported"
            )
        
        # Read image data
        image_data = await file.read()
        
        # Validate image can be opened
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        filename = f"{image_id}.{file_extension}"
        
        # Upload to storage
        image_url = storage_manager.upload_image(image_data, filename)
        
        return {
            "image_id": image_id,
            "image_url": image_url,
            "filename": filename
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image for AI generation detection
    
    Returns:
        - Prediction (AI-Generated or Real)
        - Confidence percentage
        - Risk level
        - Human-readable explanation
    """
    try:
        # Read and validate image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Upload to storage first
        image_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        filename = f"{image_id}.{file_extension}"
        image_url = storage_manager.upload_image(image_data, filename)
        
        # Run AI detection model
        result = ai_detector.predict(img)
        
        # Determine risk level based on confidence
        confidence = result['confidence']
        if confidence >= 0.85:
            risk_level = "High"
        elif confidence >= 0.65:
            risk_level = "Medium"
        else:
            risk_level = "Uncertain"
        
        return AnalysisResponse(
            prediction=result['prediction'],
            confidence=round(confidence * 100, 2),
            risk_level=risk_level,
            explanation=result['explanation'],
            image_url=image_url,
            metadata={
                "image_id": image_id,
                "filename": filename,
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": ai_detector.model_version
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/simulate-platforms", response_model=PlatformSimulationResponse)
async def simulate_platforms(file: UploadFile = File(...)):
    """
    Simulate how the image appears on different social media platforms
    and analyze prediction stability
    
    Platforms simulated:
    - WhatsApp: 512px, JPEG quality 40
    - Instagram: 1080px, JPEG quality 70
    - Facebook: 960px, JPEG quality 60
    
    Returns:
        - Original prediction
        - Platform-specific predictions
        - Stability score (0-100)
    """
    try:
        # Read original image
        image_data = await file.read()
        original_img = Image.open(io.BytesIO(image_data))
        
        # Analyze original image
        original_result = ai_detector.predict(original_img)
        original_pred = original_result['prediction']
        
        # Define platforms
        platforms = ['whatsapp', 'instagram', 'facebook']
        platform_results = {}
        predictions = [original_result['confidence']]
        
        # Simulate each platform
        for platform in platforms:
            # Transform image
            transformed_img = platform_simulator.transform(original_img, platform)
            
            # Analyze transformed image
            platform_result = ai_detector.predict(transformed_img)
            
            # Store results
            platform_results[platform] = {
                "prediction": platform_result['prediction'],
                "confidence": round(platform_result['confidence'] * 100, 2),
                "explanation": platform_result['explanation']
            }
            
            predictions.append(platform_result['confidence'])
        
        # Calculate stability score
        # Stability = 100 - (std_deviation * 100)
        # Lower variance = higher stability
        import numpy as np
        stability_score = platform_simulator.calculate_stability(predictions)
        
        return PlatformSimulationResponse(
            platform="multi-platform",
            original_result={
                "prediction": original_result['prediction'],
                "confidence": round(original_result['confidence'] * 100, 2),
                "explanation": original_result['explanation']
            },
            platform_results=platform_results,
            stability_score=round(stability_score, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Platform simulation failed: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "components": {
            "storage": storage_manager.is_connected(),
            "model": ai_detector.is_loaded(),
            "simulator": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting TruePix API on {host}:{port}")
    print(f"📝 API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, reload=False)
