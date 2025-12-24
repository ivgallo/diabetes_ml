"""
Diabetes Prediction API

FastAPI backend service that connects to MLflow for model serving
and provides prediction endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import os

app = FastAPI(
    title="Diabetes Prediction API",
    description="REST API for diabetes risk prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "diabetes-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None
_model_version = None


class DiabetesFeatures(BaseModel):
    """Input features schema for prediction requests."""
    Glucose: float = Field(..., ge=50, le=250, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., ge=40, le=150, description="Diastolic blood pressure (mm Hg)")
    BMI: float = Field(..., ge=15, le=60, description="Body mass index (kg/m2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.05, le=2.5, description="Diabetes pedigree function")
    Age: int = Field(..., ge=18, le=100, description="Age (years)")

    class Config:
        json_schema_extra = {
            "example": {
                "Glucose": 120,
                "BloodPressure": 70,
                "BMI": 25.5,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 35
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: int
    label: str
    probability: float
    risk_level: str
    model_version: str


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    model_name: str
    model_version: str
    run_id: str


def get_model():
    """
    Load the model from MLflow registry with caching.
    
    Returns:
        tuple: (model, version) loaded from MLflow
        
    Raises:
        HTTPException: If model cannot be loaded
    """
    global _model, _model_version
    
    try:
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])
        
        if not latest:
            raise HTTPException(
                status_code=404, 
                detail="No model registered. Run: docker-compose run --rm training"
            )
        
        version = latest[0].version
        
        if _model is None or _model_version != version:
            model_uri = f"models:/{MODEL_NAME}/{version}"
            _model = mlflow.sklearn.load_model(model_uri)
            _model_version = version
        
        return _model, _model_version
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on prediction probability.
    
    Args:
        probability: Probability of positive class (diabetes)
        
    Returns:
        str: Risk level category
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "moderate"
    else:
        return "high"


@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Retrieve model metrics from MLflow.
    
    Returns:
        MetricsResponse: Model performance metrics and metadata
    """
    try:
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])
        
        if not latest:
            raise HTTPException(status_code=404, detail="No model registered")
        
        model_version = latest[0]
        run = client.get_run(model_version.run_id)
        metrics = run.data.metrics
        
        return MetricsResponse(
            accuracy=metrics.get("accuracy", 0),
            precision=metrics.get("precision", 0),
            recall=metrics.get("recall", 0),
            f1_score=metrics.get("f1_score", 0),
            roc_auc=metrics.get("roc_auc", 0),
            model_name=MODEL_NAME,
            model_version=model_version.version,
            run_id=model_version.run_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DiabetesFeatures):
    """
    Generate diabetes risk prediction for given features.
    
    Args:
        features: Clinical indicators for prediction
        
    Returns:
        PredictionResponse: Prediction result with probability and risk level
    """
    model, version = get_model()
    
    df = pd.DataFrame([features.model_dump()])
    
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    diabetes_prob = proba[1]
    
    return PredictionResponse(
        prediction=int(prediction),
        label="diabetic" if prediction == 1 else "non_diabetic",
        probability=float(diabetes_prob),
        risk_level=get_risk_level(diabetes_prob),
        model_version=version
    )


@app.get("/features")
def get_feature_info():
    """
    Get metadata about model input features.
    
    Returns:
        dict: Feature specifications including ranges and defaults
    """
    return {
        "features": [
            {"name": "Glucose", "label": "Glucose", "unit": "mg/dL", "min": 50, "max": 250, "default": 120},
            {"name": "BloodPressure", "label": "BloodPressure", "unit": "mm Hg", "min": 40, "max": 150, "default": 70},
            {"name": "BMI", "label": "BMI", "unit": "kg/m2", "min": 15, "max": 60, "default": 25},
            {"name": "DiabetesPedigreeFunction", "label": "DiabetesPedigreeFunction", "unit": "", "min": 0.05, "max": 2.5, "default": 0.5},
            {"name": "Age", "label": "Age", "unit": "years", "min": 18, "max": 100, "default": 35}
        ]
    }
