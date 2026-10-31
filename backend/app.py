"""
RAD-X FastAPI Application
High-performance inference API for thoracic multi-label disease classification.
"""

from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.predictor import get_predictor, RadXPredictor
from ml.config import (
    BEST_MODEL_PATH,
    DISEASE_CLASSES,
    VALIDATION_THRESHOLDS,
    TEST_METRICS,
)

# Configuration
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
    "application/octet-stream",  # Sometimes sent for raw image files
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once
    try:
        predictor = get_predictor()
        print(f"RAD-X Backend initialized: Model loaded on {predictor.device} from {predictor.checkpoint_path}")
    except Exception as e:
        print(f"Warning: Failed to load model during startup: {e}")
    yield
    # Shutdown logic if needed

app = FastAPI(
    title="RAD-X Chest X-Ray Diagnostic Assistant API",
    description="Multi-Label Thoracic Disease Classification Research Prototype using DenseNet121.",
    version="1.0.0",
    lifespan=lifespan
)

# In production Render builds the Vite app and FastAPI serves it from the same origin.
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"

# CORS Configuration for React Frontend
allowed_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class DiseasePrediction(BaseModel):
    disease: str
    probability: float
    threshold: float
    is_positive: bool
    test_auc: Optional[float] = None
    test_f1: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None

class ModelInfo(BaseModel):
    architecture: str
    checkpoint: str
    device: str
    classes_count: int

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    positive_findings: List[str]
    is_no_finding: bool
    model_info: ModelInfo
    disclaimer: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint: str
    classes_count: int

class MetadataResponse(BaseModel):
    architecture: str
    num_classes: int
    classes: List[str]
    thresholds: Dict[str, float]
    test_metrics: Dict[str, Dict[str, float]]
    disclaimer: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify server and model status."""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=predictor.is_ready,
            device=str(predictor.device),
            checkpoint=predictor.checkpoint_path.name,
            classes_count=len(predictor.classes),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}"
        )

@app.get("/metadata", response_model=MetadataResponse)
def get_metadata():
    """Returns model metadata, class lists, and validation-calibrated thresholds."""
    return MetadataResponse(
        architecture="DenseNet121",
        num_classes=len(DISEASE_CLASSES),
        classes=DISEASE_CLASSES,
        thresholds=VALIDATION_THRESHOLDS,
        test_metrics=TEST_METRICS,
        disclaimer=(
            "RESEARCH USE ONLY: Metrics are calculated on the 2,121-image untouched test set "
            "with patient-level isolation (no patient overlap with training or validation)."
        )
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_chest_xray(file: UploadFile = File(...)):
    """
    Accepts an uploaded chest X-ray image (PNG, JPEG, WebP)
    and returns 15 multi-label disease probabilities with calibrated thresholds.
    """
    # 1. Content-Type check
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a PNG, JPEG, or WebP image."
        )

    # 2. Read file contents & validate size
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image file."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    # 3. Predict using singleton
    try:
        predictor = get_predictor()
        result = predictor.predict(contents)
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Avoid leaking internal tracebacks
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference processing failed. Please verify that a valid radiograph image was supplied."
        )

if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(_FRONTEND_DIST / "index.html")
