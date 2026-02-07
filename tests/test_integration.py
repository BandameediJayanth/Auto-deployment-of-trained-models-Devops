"""
Integration Test for MLOps Pipeline
Auto-Deployment ML Models Project

This script tests the end-to-end flow:
1. API Startup
2. Prediction Requests
3. Drift Simulation
4. Automated Retraining Trigger
5. Reliability Logging
"""

import subprocess
import time
import requests
import json
import os
import sys
import signal
import pandas as pd
import numpy as np

# Configuration
API_URL = "http://localhost:8000"
API_PROCESS = None

def start_api():
    """Start the API server as a subprocess"""
    global API_PROCESS
    print("🚀 Starting API server...")
    
    # Ensure we are in the root directory
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = cwd
    
    API_PROCESS = subprocess.Popen(
        [sys.executable, "src/model_api.py"],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    print("Waiting for API to become healthy...")
    for i in range(20):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                print("✅ API is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        if API_PROCESS.poll() is not None:
            print("❌ API process terminated unexpectedly")
            stdout, stderr = API_PROCESS.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
        time.sleep(1)
        print(".", end="", flush=True)
    
    print("\n❌ API failed to start in time")
    return False

def stop_api():
    """Stop the API server"""
    global API_PROCESS
    if API_PROCESS:
        print("\n🛑 Stopping API server...")
        API_PROCESS.terminate()
        try:
            API_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            API_PROCESS.kill()
        print("API server stopped")

def run_prediction_test():
    """Test normal predictions"""
    print("\n🧪 Testing predictions...")
    
    # Get features from API
    response = requests.get(f"{API_URL}/model/features")
    feature_names = response.json()['features']
    print(f"Model expects {len(feature_names)} features")
    
    # Generate random features
    features = np.random.randn(len(feature_names)).tolist()
    
    payload = {"features": features}
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result['prediction']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction request error: {str(e)}")
        return False

def simulate_drift():
    """Simulate data drift to trigger retraining"""
    print("\n🌊 Simulating data drift...")
    
    # Get features
    response = requests.get(f"{API_URL}/model/features")
    feature_names = response.json()['features']
    n_features = len(feature_names)
    
    # Send 60 requests (buffer size is 50) with drifted data
    # We shift the data significantly to ensure KS test fails
    print("Sending 60 drifted requests...")
    
    for i in range(60):
        # Create drifted data (mean + 5)
        features = (np.random.randn(n_features) + 5.0).tolist()
        
        try:
            requests.post(f"{API_URL}/predict", json={"features": features})
        except:
            pass
            
        if i % 10 == 0:
            print(f"{i}/60...", end="", flush=True)
            
    print("Done!")
    
    # Wait for background tasks to process
    print("Waiting for drift detection and retraining trigger (10s)...")
    time.sleep(10)

def verify_retraining():
    """Verify that retraining was triggered"""
    print("\n🔍 Verifying retraining...")
    
    # Check if a new model version file exists
    # We started with 1.0.0 (or 1.0.1 from previous runs). Retraining should create next version.
    
    models_dir = "models"
    files = os.listdir(models_dir)
    model_files = [f for f in files if f.endswith(".pkl") and "ml_classifier" in f]
    model_files.sort()
    
    print(f"Found model files: {model_files}")
    
    if len(model_files) >= 2:
        print("✅ New model file detected!")
        return True
    else:
        print("⚠️ No new model file detected yet (might be still training)")
        return False

def verify_reliability_logs():
    """Verify reliability logs"""
    print("\n📋 Verifying reliability logs...")
    
    log_file = "models/reliability_events.json"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            events = json.load(f)
        
        print(f"Found {len(events)} reliability events")
        for event in events:
            print(f" - {event['datetime']}: {event['type']}")
            
        failures = [e for e in events if e['type'] == 'failure']
        if failures:
            print(f"✅ Failure events logged: {len(failures)}")
            return True
        else:
            print("❌ No failure events logged")
            return False
    else:
        print(f"❌ Log file {log_file} not found")
        return False

def main():
    try:
        if start_api():
            run_prediction_test()
            simulate_drift()
            verify_retraining()
            verify_reliability_logs()
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        stop_api()

if __name__ == "__main__":
    main()
