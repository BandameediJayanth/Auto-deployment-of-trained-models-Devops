"""
Readiness Verifier Script
Auto-Deployment ML Models Project

This script validates the external model for production readiness.
It performs smoke tests, latency checks, and generates a final verdict.
It supports optional 'validation.csv' and 'config.json' for enhanced verification.
"""

import os
import joblib
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('readiness_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_DIR = 'input_models'
REPORT_DIR = 'reports'
VERDICT_FILE = os.path.join(REPORT_DIR, 'final_verdict.md')

# Default Thresholds
DEFAULT_THRESHOLDS = {
    "max_latency_ms": 100.0,
    "min_accuracy": 0.70,
    "min_f1_score": 0.70
}

def load_resources():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.pkl', '.joblib', '.sav'))]
    if not files:
        return None, None, None, None
    
    model_path = os.path.join(INPUT_DIR, files[0])
    validation_path = os.path.join(INPUT_DIR, 'validation.csv')
    config_path = os.path.join(INPUT_DIR, 'config.json')
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None, None

    validation_data = None
    if os.path.exists(validation_path):
        try:
            validation_data = pd.read_csv(validation_path)
        except Exception as e:
            logger.warning(f"Could not load validation data: {e}")

    config = DEFAULT_THRESHOLDS.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")

    return model, model_path, validation_data, config

def run_smoke_test(model):
    """Run a basic prediction to ensure model works"""
    try:
        # Try to infer shape
        n_features = 10 # Default
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
        elif hasattr(model, "n_features_"):
            n_features = model.n_features_
            
        logger.info(f"Running smoke test with {n_features} features...")
        
        # Generate random input
        X_test = np.random.randn(1, n_features)
        
        # Predict
        y_pred = model.predict(X_test)
        
        return True, f"Success. Output: {y_pred[0]}", n_features
    except Exception as e:
        return False, f"Failed: {str(e)}", 0

def run_validation_test(model, data, config):
    """Run validation against provided dataset"""
    if data is None:
        return False, "No validation data provided.", {}
    
    try:
        # Assume last column is target
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]
        
        y_pred = model.predict(X)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
        
        passed = (metrics["accuracy"] >= config["min_accuracy"] and 
                  metrics["f1_score"] >= config["min_f1_score"])
                  
        msg = f"Accuracy: {metrics['accuracy']:.2f}, F1: {metrics['f1_score']:.2f}"
        return passed, msg, metrics
        
    except Exception as e:
        return False, f"Validation Failed: {str(e)}", {}

def measure_latency(model, n_features, n_iterations=100):
    """Measure average inference latency"""
    latencies = []
    X_test = np.random.randn(1, n_features)
    
    # Warmup
    for _ in range(10):
        model.predict(X_test)
        
    for _ in range(n_iterations):
        start = time.time()
        model.predict(X_test)
        latencies.append((time.time() - start) * 1000) # ms
        
    return np.mean(latencies), np.std(latencies)

def generate_verdict(model_path, smoke_pass, smoke_msg, val_pass, val_msg, val_metrics, latency_avg, latency_std, config):
    """Generate the final verdict report"""
    filename = os.path.basename(model_path)
    
    # Logic: 
    # - Must pass smoke test
    # - Must pass latency
    # - If validation data exists, MUST pass validation test
    
    ready = smoke_pass and (latency_avg < config['max_latency_ms'])
    
    if "No validation data" not in val_msg:
        ready = ready and val_pass

    status = "✅ READY FOR PRODUCTION" if ready else "❌ NOT READY"
    
    report = f"""# ⚖️ Model Final Verdict

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**File:** `{filename}`
**Status:** {status}

## 1. Functional Validation
- **Smoke Test:** {'✅ Passed' if smoke_pass else '❌ Failed'}
- **Details:** `{smoke_msg}`

## 2. Performance Validation (Real Data)
- **Validation Test:** {'✅ Passed' if val_pass else '❌ Failed' if "No validation" not in val_msg else '⚠️ Skipped'}
- **Details:** `{val_msg}`
- **Metrics:**
{chr(10).join([f"  - {k}: {v:.4f} (Threshold: {config.get(f'min_{k}', 'N/A')})" for k, v in val_metrics.items()]) if val_metrics else "  - None"}

## 3. Latency Metrics
- **Average Latency:** `{latency_avg:.2f} ms` (Threshold: {config['max_latency_ms']} ms)
- **Jitter (Std Dev):** `{latency_std:.2f} ms`

## 4. Deployment Recommendation
{'Proceed with automatic deployment.' if 'READY' in status else 'Do not deploy. Review errors.'}

---
*Generated by Auto-Deployment Readiness Verifier*
"""
    
    with open(VERDICT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Verdict generated: {VERDICT_FILE}")
    return VERDICT_FILE, "READY FOR PRODUCTION" in status

def main():
    logger.info("Starting Readiness Verification...")
    
    model, path, val_data, config = load_resources()
    if not model:
        logger.error("No model found to verify.")
        return
    
    # 1. Smoke Test
    smoke_pass, smoke_msg, n_features = run_smoke_test(model)
    logger.info(f"Smoke Test: {smoke_pass}")
    
    # 2. Validation Test
    val_pass, val_msg, val_metrics = run_validation_test(model, val_data, config)
    logger.info(f"Validation: {val_msg}")

    # 3. Latency Test
    latency_avg, latency_std = 0.0, 0.0
    if smoke_pass:
        latency_avg, latency_std = measure_latency(model, n_features)
        logger.info(f"Latency: {latency_avg:.2f}ms")
    
    # 4. Generate Report
    report_path, is_ready = generate_verdict(path, smoke_pass, smoke_msg, val_pass, val_msg, val_metrics, latency_avg, latency_std, config)
    
    print(f"✅ Verification Complete. Verdict saved to: {report_path}")
    if is_ready:
        print("🚀 Model is READY for production.")
    else:
        print("🛑 Model is NOT ready.")

if __name__ == "__main__":
    main()
