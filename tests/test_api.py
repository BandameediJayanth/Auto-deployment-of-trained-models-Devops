"""
CLI Tests for ML Model CLI
Auto-Deployment ML Models Project

Tests for the CLI-based train, validate, and predict scripts.
"""

import subprocess
import sys
import pytest
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'ml_classifier_v1.0.0.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset.csv')

def run_cli(cmd):
    """Run a CLI command and return (exit_code, stdout, stderr)"""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr

def test_train_model():
    cmd = [sys.executable, os.path.join(SRC_DIR, 'train_model.py')]
    code, out, err = run_cli(cmd)
    assert code == 0
    assert "Model training completed successfully" in out
    assert os.path.exists(MODEL_PATH)

def test_validate_model():
    cmd = [sys.executable, os.path.join(SRC_DIR, 'validate_model.py'), '--model', MODEL_PATH]
    code, out, err = run_cli(cmd)
    assert code == 0
    assert "MODEL VALIDATION PASSED" in out

def test_predict_valid():
    # Use 10 features for prediction
    features = [str(x) for x in range(1, 11)]
    cmd = [sys.executable, os.path.join(SRC_DIR, 'predict.py'), '--model', MODEL_PATH, '--features'] + features
    code, out, err = run_cli(cmd)
    assert code == 0
    assert "Prediction:" in out

def test_predict_invalid_feature_count():
    # Use wrong number of features
    features = [str(x) for x in range(1, 5)]
    cmd = [sys.executable, os.path.join(SRC_DIR, 'predict.py'), '--model', MODEL_PATH, '--features'] + features
    code, out, err = run_cli(cmd)
    assert code != 0
    assert "Error" in err or "error" in out.lower() or "Expected" in err

def test_predict_non_numeric():
    features = ["a"] + [str(x) for x in range(2, 11)]
    cmd = [sys.executable, os.path.join(SRC_DIR, 'predict.py'), '--model', MODEL_PATH, '--features'] + features
    code, out, err = run_cli(cmd)
    assert code != 0
    assert "Error" in err or "error" in out.lower() or "invalid float value" in err
