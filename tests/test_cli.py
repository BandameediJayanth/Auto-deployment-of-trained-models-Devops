import subprocess
import json
import sys
import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'train_model.py')
VALIDATE_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'validate_model.py')
PREDICT_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'predict.py')

class TestCLIPipeline:
    def test_train(self):
        result = subprocess.run([PYTHON, TRAIN_SCRIPT], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Model training completed' in result.stdout or 'Model training completed' in result.stderr

    def test_validate(self):
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        result = subprocess.run([PYTHON, VALIDATE_SCRIPT, '--model', model_path], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'MODEL VALIDATION PASSED' in result.stdout or 'MODEL VALIDATION PASSED' in result.stderr

    def test_predict_valid(self):
        # Use 10 features as expected by the model
        features = [str(1.0)] * 10
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        result = subprocess.run([
            PYTHON, PREDICT_SCRIPT, '--model', model_path, '--features'] + features,
            capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Prediction:' in result.stdout

    def test_predict_invalid_feature_count(self):
        features = [str(1.0)] * 2  # Too few
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        result = subprocess.run([
            PYTHON, PREDICT_SCRIPT, '--model', model_path, '--features'] + features,
            capture_output=True, text=True)
        assert result.returncode != 0
        assert 'Expected' in result.stderr or 'Error' in result.stderr

    def test_predict_non_numeric(self):
        features = [str(1.0)] * 9 + ['bad']
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        result = subprocess.run([
            PYTHON, PREDICT_SCRIPT, '--model', model_path, '--features'] + features,
            capture_output=True, text=True)
        assert result.returncode != 0

    @pytest.mark.slow
    def test_predict_concurrent(self):
        import concurrent.futures
        features = [str(1.0)] * 10
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        def call_predict():
            result = subprocess.run([
                PYTHON, PREDICT_SCRIPT, '--model', model_path, '--features'] + features,
                capture_output=True, text=True)
            return result.returncode == 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: call_predict(), range(10)))
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.7

    @pytest.mark.slow
    def test_predict_sustained(self):
        features = [str(1.0)] * 10
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ml_classifier_v1.0.0.pkl')
        success_count = 0
        total = 50
        for _ in range(total):
            result = subprocess.run([
                PYTHON, PREDICT_SCRIPT, '--model', model_path, '--features'] + features,
                capture_output=True, text=True)
            if result.returncode == 0:
                success_count += 1
        success_rate = success_count / total
        assert success_rate >= 0.8
