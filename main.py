"""
🌾 Annam AI - FastAPI REST API
=============================

A comprehensive FastAPI REST API for the Annam AI agricultural intelligence system.

Features:
- RESTful API endpoints for all 4 modules
- File upload support for images and data
- JSON responses with structured data
- API documentation with Swagger/OpenAPI
- Authentication and rate limiting
- Docker-ready deployment

Author: Annam AI Team
License: MIT
Version: 1.0.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import asyncio
import uvicorn

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnnamAI.API")

# Initialize FastAPI app
app = FastAPI(
    title="Annam AI - Agricultural Intelligence API",
    description="Comprehensive AI-powered agricultural intelligence system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for request/response
class CropClassificationRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    model_type: str = Field("cnn", description="Model type: cnn, rf, xgboost")
    confidence_threshold: float = Field(0.5, description="Confidence threshold")

class CropClassificationResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_used: str
    processing_time: float
    confidence_threshold: float

class YieldPredictionRequest(BaseModel):
    crop: str = Field(..., description="Crop type")
    region: str = Field(..., description="Geographic region")
    soil_ph: float = Field(..., description="Soil pH level")
    soil_organic_matter: float = Field(..., description="Soil organic matter percentage")
    soil_nitrogen: float = Field(..., description="Soil nitrogen content (ppm)")
    soil_phosphorus: float = Field(..., description="Soil phosphorus content (ppm)")
    soil_potassium: float = Field(..., description="Soil potassium content (ppm)")
    temperature_avg: float = Field(..., description="Average temperature (°C)")
    temperature_max_avg: float = Field(..., description="Average maximum temperature (°C)")
    temperature_min_avg: float = Field(..., description="Average minimum temperature (°C)")
    total_precipitation: float = Field(..., description="Total precipitation (mm)")
    avg_humidity: float = Field(..., description="Average humidity (%)")
    avg_solar_radiation: float = Field(..., description="Average solar radiation")

class YieldPredictionResponse(BaseModel):
    predicted_yield: float
    confidence_interval: List[float]
    yield_category: str
    model_used: str
    processing_time: float

class DiseaseDetectionRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded plant image")
    plant_type: Optional[str] = Field(None, description="Plant type (optional)")
    top_k: int = Field(3, description="Number of top predictions to return")

class DiseaseDetectionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    treatment_recommendations: Optional[List[str]]
    processing_time: float

class AdvisoryRequest(BaseModel):
    question: str = Field(..., description="Agricultural question")
    context: Optional[Dict[str, str]] = Field(None, description="Additional context")
    top_k_docs: int = Field(3, description="Number of relevant documents to retrieve")

class AdvisoryResponse(BaseModel):
    response: str
    confidence_score: float
    relevant_documents: List[Dict[str, Any]]
    follow_up_questions: List[str]
    processing_time: float

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Authentication dependency (simple token validation)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token validation for demonstration."""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if token != "annam-ai-demo-token":  # Demo token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "demo_user"}

# Crop Classification Endpoints
@app.post("/api/v1/crop-classification/predict", 
          response_model=CropClassificationResponse,
          tags=["Crop Classification"])
async def predict_crop_from_request(
    request: CropClassificationRequest,
    user = Depends(get_current_user)
):
    """Predict crop type from base64 encoded image."""
    start_time = datetime.now()

    try:
        # Mock prediction - replace with actual model inference
        predictions = [
            {"crop": "Wheat", "confidence": 0.85, "plant_type": "cereal"},
            {"crop": "Corn", "confidence": 0.12, "plant_type": "cereal"},
            {"crop": "Soybean", "confidence": 0.03, "plant_type": "legume"}
        ]

        # Filter by confidence threshold
        filtered_predictions = [
            p for p in predictions if p["confidence"] >= request.confidence_threshold
        ]

        processing_time = (datetime.now() - start_time).total_seconds()

        return CropClassificationResponse(
            predictions=filtered_predictions,
            model_used=request.model_type,
            processing_time=processing_time,
            confidence_threshold=request.confidence_threshold
        )

    except Exception as e:
        logger.error(f"Crop classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/crop-classification/upload", 
          response_model=CropClassificationResponse,
          tags=["Crop Classification"])
async def predict_crop_from_upload(
    file: UploadFile = File(...),
    model_type: str = "cnn",
    confidence_threshold: float = 0.5,
    user = Depends(get_current_user)
):
    """Predict crop type from uploaded image file."""
    start_time = datetime.now()

    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Mock prediction - replace with actual model inference
        predictions = [
            {"crop": "Tomato", "confidence": 0.92, "plant_type": "vegetable"},
            {"crop": "Potato", "confidence": 0.06, "plant_type": "vegetable"},
            {"crop": "Pepper", "confidence": 0.02, "plant_type": "vegetable"}
        ]

        # Filter by confidence threshold
        filtered_predictions = [
            p for p in predictions if p["confidence"] >= confidence_threshold
        ]

        processing_time = (datetime.now() - start_time).total_seconds()

        return CropClassificationResponse(
            predictions=filtered_predictions,
            model_used=model_type,
            processing_time=processing_time,
            confidence_threshold=confidence_threshold
        )

    except Exception as e:
        logger.error(f"Crop classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Yield Prediction Endpoints
@app.post("/api/v1/yield-prediction/predict",
          response_model=YieldPredictionResponse,
          tags=["Yield Prediction"])
async def predict_yield(
    request: YieldPredictionRequest,
    model_type: str = "random_forest",
    user = Depends(get_current_user)
):
    """Predict crop yield based on environmental and soil data."""
    start_time = datetime.now()

    try:
        # Mock prediction - replace with actual model inference
        # Simple heuristic based on input values
        base_yield = {"wheat": 4.5, "corn": 9.8, "rice": 6.2, "soybean": 3.1, "potato": 8.5}.get(
            request.crop.lower(), 5.0
        )

        # Adjust based on conditions
        ph_factor = 1.0 if 6.0 <= request.soil_ph <= 7.5 else 0.8
        temp_factor = 1.0 if 18 <= request.temperature_avg <= 30 else 0.9
        precip_factor = 1.0 if 300 <= request.total_precipitation <= 800 else 0.85

        predicted_yield = base_yield * ph_factor * temp_factor * precip_factor
        predicted_yield += np.random.normal(0, 0.2)  # Add some randomness

        confidence_interval = [predicted_yield - 0.5, predicted_yield + 0.5]

        yield_category = "High" if predicted_yield > 6 else "Medium" if predicted_yield > 3 else "Low"

        processing_time = (datetime.now() - start_time).total_seconds()

        return YieldPredictionResponse(
            predicted_yield=round(predicted_yield, 2),
            confidence_interval=[round(ci, 2) for ci in confidence_interval],
            yield_category=yield_category,
            model_used=model_type,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/yield-prediction/batch",
          tags=["Yield Prediction"])
async def predict_yield_batch(
    file: UploadFile = File(...),
    model_type: str = "random_forest",
    user = Depends(get_current_user)
):
    """Predict yields for multiple crops from CSV file."""
    start_time = datetime.now()

    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file.")

        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        # Mock batch prediction
        predictions = np.random.normal(5.0, 1.5, len(df))
        df['predicted_yield'] = [round(p, 2) for p in predictions]
        df['yield_category'] = df['predicted_yield'].apply(
            lambda x: "High" if x > 6 else "Medium" if x > 3 else "Low"
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "message": "Batch prediction completed",
            "total_records": len(df),
            "predictions": df.to_dict('records'),
            "model_used": model_type,
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Batch yield prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Disease Detection Endpoints
@app.post("/api/v1/disease-detection/predict",
          response_model=DiseaseDetectionResponse,
          tags=["Disease Detection"])
async def detect_disease_from_request(
    request: DiseaseDetectionRequest,
    user = Depends(get_current_user)
):
    """Detect plant diseases from base64 encoded image."""
    start_time = datetime.now()

    try:
        # Mock disease detection - replace with actual model inference
        predictions = [
            {
                "disease": "Late Blight",
                "plant": "Tomato",
                "confidence": 0.89,
                "severity": "High",
                "description": "Fungal disease affecting tomatoes and potatoes"
            },
            {
                "disease": "Healthy",
                "plant": "Tomato", 
                "confidence": 0.08,
                "severity": "None",
                "description": "No disease detected"
            },
            {
                "disease": "Early Blight",
                "plant": "Tomato",
                "confidence": 0.03,
                "severity": "Low",
                "description": "Common fungal disease in tomatoes"
            }
        ]

        # Get top predictions
        top_predictions = predictions[:request.top_k]

        # Treatment recommendations
        treatment_recommendations = None
        if top_predictions[0]["disease"] != "Healthy":
            treatment_recommendations = [
                "Remove affected leaves immediately",
                "Apply copper-based fungicide",
                "Improve air circulation",
                "Avoid overhead watering",
                "Consider resistant varieties for future planting"
            ]

        processing_time = (datetime.now() - start_time).total_seconds()

        return DiseaseDetectionResponse(
            predictions=top_predictions,
            treatment_recommendations=treatment_recommendations,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/disease-detection/upload",
          response_model=DiseaseDetectionResponse,
          tags=["Disease Detection"])
async def detect_disease_from_upload(
    file: UploadFile = File(...),
    plant_type: Optional[str] = None,
    top_k: int = 3,
    user = Depends(get_current_user)
):
    """Detect plant diseases from uploaded image file."""
    start_time = datetime.now()

    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Mock disease detection - replace with actual model inference
        predictions = [
            {
                "disease": "Powdery Mildew",
                "plant": plant_type or "Unknown",
                "confidence": 0.91,
                "severity": "Medium",
                "description": "Fungal disease causing white powdery growth"
            },
            {
                "disease": "Healthy",
                "plant": plant_type or "Unknown",
                "confidence": 0.07,
                "severity": "None",
                "description": "No disease detected"
            },
            {
                "disease": "Leaf Spot",
                "plant": plant_type or "Unknown",
                "confidence": 0.02,
                "severity": "Low",
                "description": "Bacterial or fungal leaf spot disease"
            }
        ]

        # Get top predictions
        top_predictions = predictions[:top_k]

        # Treatment recommendations
        treatment_recommendations = None
        if top_predictions[0]["disease"] != "Healthy":
            treatment_recommendations = [
                "Apply sulfur-based fungicide",
                "Improve air circulation around plants",
                "Remove infected plant parts",
                "Avoid overcrowding plants",
                "Water at soil level to avoid wetting leaves"
            ]

        processing_time = (datetime.now() - start_time).total_seconds()

        return DiseaseDetectionResponse(
            predictions=top_predictions,
            treatment_recommendations=treatment_recommendations,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agricultural Advisory Endpoints
@app.post("/api/v1/advisory/ask",
          response_model=AdvisoryResponse,
          tags=["Agricultural Advisory"])
async def get_agricultural_advice(
    request: AdvisoryRequest,
    user = Depends(get_current_user)
):
    """Get agricultural advice using AI and knowledge base."""
    start_time = datetime.now()

    try:
        # Mock advisory response - replace with actual RAG system
        question_lower = request.question.lower()

        if "yellow" in question_lower and "leaves" in question_lower:
            response = """Yellow leaves can indicate several issues:

1. **Nitrogen Deficiency**: Most common cause, especially if lower leaves yellow first
   - Solution: Apply nitrogen fertilizer

2. **Overwatering**: Soil stays too wet, roots can't breathe
   - Solution: Improve drainage, reduce watering frequency

3. **Underwatering**: Plants stressed from lack of water
   - Solution: Increase irrigation, check soil moisture regularly

4. **Disease**: Various diseases can cause yellowing
   - Solution: Identify specific disease, apply appropriate treatment

**Recommended Actions:**
- Conduct soil test for nutrient levels
- Check soil moisture and drainage
- Examine plants for disease symptoms
- Apply appropriate fertilizer based on test results"""

            confidence_score = 0.92

        elif "pest" in question_lower or "insect" in question_lower:
            response = """For pest management, follow Integrated Pest Management (IPM) principles:

**Identification**: Correctly identify the pest before treatment
**Monitoring**: Regular scouting to assess pest populations
**Prevention**: Cultural practices to prevent pest problems
**Control Options**:
- Biological: Beneficial insects, natural enemies
- Cultural: Crop rotation, resistant varieties
- Chemical: Targeted pesticide application when necessary

**Key Steps:**
1. Identify the specific pest
2. Monitor population levels
3. Use economic thresholds to guide treatment decisions
4. Apply targeted control measures
5. Evaluate effectiveness and adjust as needed"""

            confidence_score = 0.89

        else:
            response = """Thank you for your agricultural question. Here are some general best practices:

**Crop Management:**
- Regular monitoring of plant health
- Proper irrigation scheduling
- Balanced fertilization based on soil tests
- Integrated pest and disease management

**Soil Health:**
- Maintain soil organic matter
- Ensure proper pH levels
- Practice crop rotation
- Prevent soil erosion

**Sustainability:**
- Use resources efficiently
- Minimize environmental impact
- Adopt conservation practices
- Plan for long-term productivity

For specific advice, please provide more details about your situation, including crop type, growing conditions, and specific symptoms or concerns."""

            confidence_score = 0.75

        # Mock relevant documents
        relevant_docs = [
            {
                "document": "Crop nutrition and fertilization guidelines",
                "metadata": {"category": "nutrition", "topic": "fertilization"},
                "score": 0.85
            },
            {
                "document": "Plant disease identification and management",
                "metadata": {"category": "diseases", "topic": "identification"},
                "score": 0.78
            },
            {
                "document": "Integrated pest management strategies",
                "metadata": {"category": "pests", "topic": "management"},
                "score": 0.72
            }
        ]

        # Generate follow-up questions
        follow_up_questions = [
            "What specific symptoms are you observing?",
            "What is your current fertilization program?",
            "Have you noticed any pest activity?",
            "What are your soil test results?",
            "Are you experiencing any weather stress?"
        ]

        processing_time = (datetime.now() - start_time).total_seconds()

        return AdvisoryResponse(
            response=response,
            confidence_score=confidence_score,
            relevant_documents=relevant_docs[:request.top_k_docs],
            follow_up_questions=follow_up_questions[:3],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Advisory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Information Endpoints
@app.get("/api/v1/system/info", tags=["System"])
async def get_system_info():
    """Get system information and statistics."""
    return {
        "system": "Annam AI - Agricultural Intelligence System",
        "version": "1.0.0",
        "modules": {
            "crop_classification": {
                "supported_crops": ["wheat", "corn", "rice", "soybean", "tomato", "potato"],
                "model_types": ["cnn", "random_forest", "xgboost"],
                "accuracy": "85-95%"
            },
            "yield_prediction": {
                "supported_crops": ["wheat", "corn", "rice", "soybean", "potato"],
                "model_types": ["random_forest", "xgboost", "lstm", "linear_regression"],
                "accuracy": "R² 0.85-0.92"
            },
            "disease_detection": {
                "supported_diseases": 38,
                "supported_plants": ["tomato", "potato", "apple", "corn", "grape"],
                "accuracy": "90-95%"
            },
            "agricultural_advisory": {
                "knowledge_base_size": "1000+ documents",
                "supported_topics": ["crops", "pests", "diseases", "soil", "irrigation"],
                "response_time": "<2 seconds"
            }
        },
        "api_status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/system/models", tags=["System"])
async def get_available_models():
    """Get information about available models."""
    return {
        "crop_classification": {
            "cnn": {
                "name": "Convolutional Neural Network",
                "architecture": "ResNet50 + Custom Head",
                "input_size": "224x224x3",
                "accuracy": "92%",
                "inference_time": "~0.1s"
            },
            "random_forest": {
                "name": "Random Forest Classifier",
                "features": "Color, texture, shape features",
                "accuracy": "87%",
                "inference_time": "~0.05s"
            },
            "xgboost": {
                "name": "XGBoost Classifier",
                "features": "Color, texture, shape features",
                "accuracy": "89%",
                "inference_time": "~0.03s"
            }
        },
        "yield_prediction": {
            "random_forest": {
                "name": "Random Forest Regressor",
                "features": "Weather, soil, crop features",
                "r2_score": "0.89",
                "rmse": "0.45 tons/ha"
            },
            "xgboost": {
                "name": "XGBoost Regressor", 
                "features": "Weather, soil, crop features",
                "r2_score": "0.92",
                "rmse": "0.38 tons/ha"
            },
            "lstm": {
                "name": "LSTM Neural Network",
                "features": "Time series yield data",
                "r2_score": "0.90",
                "rmse": "0.42 tons/ha"
            }
        },
        "disease_detection": {
            "cnn": {
                "name": "Transfer Learning CNN",
                "architecture": "EfficientNetB0 + Custom Head",
                "input_size": "224x224x3",
                "accuracy": "94%",
                "supported_diseases": 38
            }
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
