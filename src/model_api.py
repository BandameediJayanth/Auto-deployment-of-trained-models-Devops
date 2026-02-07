"""
Model API Server
Auto-Deployment ML Models Project

FastAPI server for serving trained ML models with monitoring and logging.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging
import time
import traceback
import subprocess
import sys

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Prometheus monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import Drift Detector and Reliability Tracker
try:
    from src.drift_detection import DriftDetector
    from src.reliability import ReliabilityTracker
except ImportError:
    # Fallback for when running from different directory structure
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.drift_detection import DriftDetector
    from src.reliability import ReliabilityTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('model_api_request_duration_seconds', 'Request duration in seconds')
PREDICTION_COUNT = Counter('model_predictions_total', 'Total predictions made')
ERROR_COUNT = Counter('model_api_errors_total', 'Total API errors', ['error_type'])
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load the model')
DRIFT_SCORE = Gauge('model_drift_score', 'Fraction of features with detected drift')
PREDICTION_CONFIDENCE = Histogram('model_prediction_confidence', 'Prediction confidence score')
DRIFTED_FEATURES_COUNT = Gauge('model_drifted_features_count', 'Number of drifted features')


# Pydantic models for API
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Input features for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7, 1.9, 3.2]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Model prediction (class)")
    probability: List[float] = Field(..., description="Prediction probabilities for each class")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    features: int
    training_date: str
    accuracy: float

# Global variables
app = FastAPI(
    title="ML Model API",
    description="Auto-Deployment ML Models - Production API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata
model = None
metadata = None
start_time = time.time()

class ModelManager:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.is_loaded = False
        self.drift_detector = DriftDetector()
        self.reliability_tracker = ReliabilityTracker()
        self.request_buffer = []
        self.buffer_size = 50 # Run drift detection every 50 requests
    
    def check_drift_background(self, data_batch):
        """Run drift detection on a batch of data"""
        try:
            if not self.feature_names:
                return
                
            logger.info(f"Running drift detection on batch of {len(data_batch)} samples")
            result = self.drift_detector.check_drift(pd.DataFrame(data_batch, columns=self.feature_names))
            
            if "error" not in result:
                drift_score = result["drifted_feature_count"] / result["total_features"] if result["total_features"] > 0 else 0
                DRIFT_SCORE.set(drift_score)
                DRIFTED_FEATURES_COUNT.set(result["drifted_feature_count"])
                
                if result["drift_detected"]:
                    logger.warning(f"Drift detected! Score: {drift_score:.2f}")
                    self.reliability_tracker.log_event("failure", {"reason": "drift_detected", "score": drift_score})
                    
                    # Trigger retraining
                    try:
                        logger.info("Triggering automated retraining...")
                        subprocess.Popen([sys.executable, "src/trigger_retraining.py"])
                        # Also could trigger rollback here if drift is severe
                    except Exception as e:
                        logger.error(f"Failed to trigger retraining: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")

    def load_model(self):
        """Load the latest trained model"""
        try:
            start_load_time = time.time()
            
            # Load latest model info
            if not os.path.exists('models/latest_model.json'):
                raise FileNotFoundError("No trained model found. Please train a model first.")
            
            with open('models/latest_model.json', 'r') as f:
                latest_info = json.load(f)
            
            model_path = latest_info['latest_model']
            metadata_path = latest_info['latest_metadata']
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            self.is_loaded = True
            
            load_time = time.time() - start_load_time
            MODEL_LOAD_TIME.set(load_time)
            
            logger.info(f"Model loaded successfully: {self.metadata['model_name']} v{self.metadata['version']}")
            logger.info(f"Model load time: {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            ERROR_COUNT.labels(error_type='model_load_error').inc()
            self.is_loaded = False
            return False
    
    def predict(self, features: List[float], background_tasks: BackgroundTasks = None) -> Dict[str, Any]:
        """Make prediction using the loaded model"""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Convert to numpy array with correct shape
            X = np.array(features).reshape(1, -1)
            
            # Validate input dimensions
            expected_features = len(self.feature_names)
            if X.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probabilities if available
            probabilities = []
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0].tolist()
                confidence = float(np.max(probabilities))
                PREDICTION_CONFIDENCE.observe(confidence)
            
            PREDICTION_COUNT.inc()
            
            # Add to buffer for drift detection
            self.request_buffer.append(features)
            if background_tasks and len(self.request_buffer) >= self.buffer_size:
                batch = self.request_buffer.copy()
                self.request_buffer = [] # Reset buffer
                background_tasks.add_task(self.check_drift_background, batch)
            
            return {
                'prediction': int(prediction),
                'probability': probabilities,
                'model_version': self.metadata['version'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            ERROR_COUNT.labels(error_type='prediction_error').inc()
            raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting ML Model API server...")
    success = model_manager.load_model()
    if not success:
        logger.error("Failed to load model on startup!")

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    REQUEST_DURATION.observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = f"""
    <html>
        <head>
            <title>ML Model API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #2E86AB; }}
                .info {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .endpoints {{ margin-top: 20px; }}
                .endpoint {{ margin: 10px 0; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1 class="header">🚀 ML Model API</h1>
            <div class="info">
                <h2>Auto-Deployment ML Models - Production API</h2>
                <p><strong>Status:</strong> {'🟢 Running' if model_manager.is_loaded else '🔴 Model Not Loaded'}</p>
                <p><strong>Model:</strong> {model_manager.metadata['model_name'] if model_manager.is_loaded else 'None'}</p>
                <p><strong>Version:</strong> {model_manager.metadata['version'] if model_manager.is_loaded else 'N/A'}</p>
                <p><strong>Uptime:</strong> {time.time() - start_time:.1f} seconds</p>
            </div>
            
            <div class="endpoints">
                <h3>🔗 Available Endpoints:</h3>
                <div class="endpoint"><strong>GET /health</strong> - Health check</div>
                <div class="endpoint"><strong>POST /predict</strong> - Make predictions</div>
                <div class="endpoint"><strong>GET /model/info</strong> - Model information</div>
                <div class="endpoint"><strong>GET /metrics</strong> - Prometheus metrics</div>
                <div class="endpoint"><strong>GET /docs</strong> - API documentation (Swagger UI)</div>
                <div class="endpoint"><strong>GET /redoc</strong> - API documentation (ReDoc)</div>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.metadata['version'] if model_manager.is_loaded else "none",
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a prediction using the loaded model"""
    try:
        result = model_manager.predict(request.features, background_tasks)
        return PredictionResponse(**result)
    
    except Exception as e:
        ERROR_COUNT.labels(error_type='api_error').inc()
        logger.error(f"Prediction API error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model_manager.metadata
    
    return ModelInfo(
        name=metadata['model_name'],
        version=metadata['version'],
        type=metadata['model_type'],
        features=len(metadata['feature_names']),
        training_date=metadata['metrics']['training_date'],
        accuracy=metadata['metrics']['accuracy']
    )

@app.post("/model/reload")
async def reload_model():
    """Reload the model (useful for model updates)"""
    logger.info("Reloading model...")
    success = model_manager.load_model()
    
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/features")
async def get_model_features():
    """Get the list of expected features"""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": model_manager.feature_names,
        "count": len(model_manager.feature_names),
        "model_version": model_manager.metadata['version']
    }

# Example usage endpoint
@app.get("/example")
async def get_example_request():
    """Get an example prediction request"""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate example features (random values)
    np.random.seed(42)
    example_features = np.random.randn(len(model_manager.feature_names)).tolist()
    
    return {
        "example_request": {
            "features": example_features
        },
        "curl_example": f"""
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"features": {example_features}}}'
        """.strip(),
        "feature_names": model_manager.feature_names
    }

def main():
    """Main function to run the API server"""
    logger.info("Starting ML Model API Server...")
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Debug: {debug}")
    
    # Run the server
    # Use 'app' directly to avoid double-import and registry errors when running as __main__
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
