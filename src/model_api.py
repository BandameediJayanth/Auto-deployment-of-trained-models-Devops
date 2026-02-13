"""
Model API Server - Enhanced with Model Selection
Auto-Deployment ML Models Project

FastAPI server for serving trained ML models with monitoring and logging.
Now supports model selection and loading.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time
import traceback
import argparse

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

# Prometheus monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

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
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7, 1.9, 3.2]
            }
        }
    )

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: int = Field(..., description="Model prediction (class)")
    probability: List[float] = Field(..., description="Prediction probabilities for each class")
    model_version: str = Field(..., description="Version of the model used")
    model_name: str = Field(..., description="Name of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_version: str
    model_name: str
    uptime_seconds: float
    timestamp: str

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = None
start_time = time.time()

class ModelManager:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.is_loaded = False
        self.model_path = model_path
        self.request_buffer = []
        self.buffer_size = 50
        
    def find_model_files(self, models_dir: str = 'models') -> List[Dict[str, Any]]:
        """Find all available model files"""
        model_files = []
        
        if not os.path.exists(models_dir):
            return model_files
        
        # Look for .pkl files
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and not file.startswith('.'):
                model_path = os.path.join(models_dir, file)
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                
                # Check if metadata exists
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            
                        model_files.append({
                            'path': model_path,
                            'metadata_path': metadata_path,
                            'name': metadata.get('model_name', file.replace('.pkl', '')),
                            'version': metadata.get('version', 'unknown'),
                            'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                            'features': len(metadata.get('feature_names', [])),
                            'metadata': metadata
                        })
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {model_path}: {e}")
        
        return model_files
    
    def select_model_interactive(self) -> Optional[str]:
        """Interactive model selection"""
        models = self.find_model_files()
        
        if not models:
            logger.error("No models found in models directory!")
            return None
        
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)
        print(f"{'#':<5} {'Model Name':<30} {'Version':<15} {'Accuracy':<12} {'Features':<10}")
        print("-" * 80)
        
        for idx, model in enumerate(models, 1):
            print(f"{idx:<5} {model['name'][:28]:<30} {model['version']:<15} "
                  f"{model['accuracy']:<12.4f} {model['features']:<10}")
        
        print("=" * 80)
        
        # Check if running in non-interactive mode (Docker, CI/CD, etc.)
        import sys
        if not sys.stdin.isatty():
            # Auto-select first model in non-interactive mode
            selected = models[0]
            print(f"\nAuto-selecting first model (non-interactive mode): {selected['name']} (v{selected['version']})")
            logger.info(f"Auto-selected model: {selected['name']} (v{selected['version']})")
            return selected['path']
        
        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected = models[choice_num - 1]
                    print(f"\nSelected: {selected['name']} (v{selected['version']})")
                    return selected['path']
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n\nCancelled by user")
                return None
            except EOFError:
                # Handle EOF error in non-interactive environments
                selected = models[0]
                print(f"\nAuto-selecting first model (EOF detected): {selected['name']} (v{selected['version']})")
                logger.info(f"Auto-selected model due to EOF: {selected['name']} (v{selected['version']})")
                return selected['path']
    
    def load_model(self, model_path: Optional[str] = None):
        """Load a specific model or prompt for selection"""
        if model_path is None:
            model_path = self.model_path
        
        if model_path is None:
            # Interactive selection
            model_path = self.select_model_interactive()
            if model_path is None:
                return False
        
        try:
            start_load_time = time.time()
            
            # Load model
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            logger.info(f"Loading metadata from {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            self.is_loaded = True
            self.model_path = model_path
            
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
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
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
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0].tolist()
                confidence = float(np.max(probabilities))
                PREDICTION_CONFIDENCE.observe(confidence)
            
            PREDICTION_COUNT.inc()
            
            return {
                'prediction': int(prediction),
                'probability': probabilities,
                'model_version': self.metadata['version'],
                'model_name': self.metadata['model_name'],
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
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(process_time)
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        # Fallback if static file doesn't exist
        return HTMLResponse(content="""
        <html>
            <head><title>ML Model API</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>🏥 ML Model API</h1>
                <p>Web UI not found. Please check the static directory.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.metadata['version'] if model_manager.is_loaded else "none",
        model_name=model_manager.metadata['model_name'] if model_manager.is_loaded else "none",
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the loaded model"""
    try:
        result = model_manager.predict(request.features)
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

@app.get("/models")
async def list_models():
    """List all available models"""
    models = model_manager.find_model_files()
    return {
        "models": models,
        "count": len(models),
        "loaded_model": model_manager.model_path if model_manager.is_loaded else None
    }

@app.post("/model/load")
async def load_model_endpoint(model_path: Optional[str] = None):
    """Load a specific model"""
    success = model_manager.load_model(model_path)
    
    if success:
        return {"status": "success", "message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def main():
    """Main function to run the API server"""
    parser = argparse.ArgumentParser(description='ML Model API Server')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    # Initialize model manager with specified model
    global model_manager
    model_manager = ModelManager(args.model)
    
    # Try to load model
    if args.model:
        success = model_manager.load_model(args.model)
        if not success:
            logger.error(f"Failed to load model: {args.model}")
            exit(1)
    
    print(f"\n🚀 Starting ML Model API Server...")
    print(f"📊 Model: {args.model if args.model else 'Interactive selection'}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()