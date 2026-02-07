"""
API Tests for ML Model API
Auto-Deployment ML Models Project

Tests for the FastAPI model serving endpoints.
"""

import requests
import json
import time
import pytest
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestAPIHealth:
    """Test API health and basic functionality"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML"""
        response = requests.get(f"{API_BASE_URL}/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "ML Model API" in response.text
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health")
        
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "model_loaded" in health_data
        assert "uptime_seconds" in health_data
        assert health_data["uptime_seconds"] >= 0
        
    def test_docs_endpoint(self):
        """Test API documentation endpoint"""
        response = requests.get(f"{API_BASE_URL}/docs")
        
        assert response.status_code == 200
        assert "Swagger UI" in response.text or "swagger" in response.text.lower()

class TestModelInfo:
    """Test model information endpoints"""
    
    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        response = requests.get(f"{API_BASE_URL}/model/info")
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 200
        
        info = response.json()
        assert "name" in info
        assert "version" in info
        assert "type" in info
        assert "features" in info
        assert "accuracy" in info
        
    def test_model_features_endpoint(self):
        """Test model features endpoint"""
        response = requests.get(f"{API_BASE_URL}/model/features")
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 200
        
        features = response.json()
        assert "features" in features
        assert "count" in features
        assert isinstance(features["features"], list)
        assert features["count"] == len(features["features"])
        
    def test_example_endpoint(self):
        """Test example request endpoint"""
        response = requests.get(f"{API_BASE_URL}/example")
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 200
        
        example = response.json()
        assert "example_request" in example
        assert "features" in example["example_request"]
        assert "curl_example" in example
        assert isinstance(example["example_request"]["features"], list)

class TestPredictionAPI:
    """Test prediction endpoints"""
    
    def get_example_features(self) -> list:
        """Get example features for testing"""
        try:
            response = requests.get(f"{API_BASE_URL}/example")
            if response.status_code == 200:
                return response.json()["example_request"]["features"]
        except:
            pass
        
        # Fallback: return default test features
        return [1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7, 1.9, 3.2,
                4.5, 6.7, 8.9, 1.0, 2.3, 4.6, 7.9, 5.2, 8.1, 3.7]
    
    def test_prediction_valid_input(self):
        """Test prediction with valid input"""
        features = self.get_example_features()
        
        payload = {"features": features}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 200
        
        prediction = response.json()
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "model_version" in prediction
        assert "timestamp" in prediction
        
        # Validate prediction value
        assert isinstance(prediction["prediction"], int)
        assert prediction["prediction"] in [0, 1]  # Binary classification
        
        # Validate probabilities
        assert isinstance(prediction["probability"], list)
        if len(prediction["probability"]) > 0:
            assert abs(sum(prediction["probability"]) - 1.0) < 0.01  # Sum to ~1
            
    def test_prediction_invalid_input_format(self):
        """Test prediction with invalid input format"""
        # Test with non-list features
        payload = {"features": "invalid"}
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        
        assert response.status_code == 422  # Validation error
        
    def test_prediction_wrong_feature_count(self):
        """Test prediction with wrong number of features"""
        # Test with wrong number of features
        payload = {"features": [1.0, 2.0]}  # Too few features
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 400  # Bad request
        
    def test_prediction_non_numeric_features(self):
        """Test prediction with non-numeric features"""
        features = self.get_example_features()
        features[0] = "invalid"  # Replace first feature with string
        
        payload = {"features": features}
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        # Should either be 400 (bad request) or 422 (validation error)
        assert response.status_code in [400, 422]
        
    def test_prediction_response_time(self):
        """Test prediction response time"""
        features = self.get_example_features()
        payload = {"features": features}
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        response_time = time.time() - start_time
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in some test scenarios")
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds

class TestModelManagement:
    """Test model management endpoints"""
    
    def test_model_reload(self):
        """Test model reload endpoint"""
        response = requests.post(f"{API_BASE_URL}/model/reload")
        
        # Should either succeed (200) or fail gracefully (500)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            result = response.json()
            assert "status" in result
            assert "message" in result

class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = requests.get(f"{API_BASE_URL}/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        
        # Check for some expected Prometheus metrics
        metrics_text = response.text
        assert "model_api_requests_total" in metrics_text
        assert "model_api_request_duration_seconds" in metrics_text

class TestAPILoad:
    """Test API under load"""
    
    @pytest.mark.slow
    def test_concurrent_predictions(self):
        """Test multiple concurrent prediction requests"""
        import concurrent.futures
        import threading
        
        features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        payload = {"features": features}
        
        def make_prediction():
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=10
                )
                return response.status_code == 200
            except:
                return False
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 70% of requests should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.7
        
    @pytest.mark.slow
    def test_sustained_load(self):
        """Test sustained prediction load"""
        features = [1.0] * 20  # Simple feature vector
        payload = {"features": features}
        
        success_count = 0
        total_requests = 50
        
        for _ in range(total_requests):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=5
                )
                if response.status_code == 200:
                    success_count += 1
            except:
                pass
        
        # At least 80% of requests should succeed
        success_rate = success_count / total_requests
        assert success_rate >= 0.8

class TestErrorHandling:
    """Test API error handling"""
    
    def test_malformed_json(self):
        """Test API with malformed JSON"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
        
    def test_missing_content_type(self):
        """Test API with missing content type"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            data='{"features": [1, 2, 3]}'
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 415, 422]
        
    def test_large_payload(self):
        """Test API with very large payload"""
        # Create a very large feature vector
        large_features = [1.0] * 10000
        payload = {"features": large_features}
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 413, 422]

class TestAPIIntegration:
    """Integration tests for the complete API"""
    
    def test_complete_workflow(self):
        """Test complete API workflow"""
        # 1. Check health
        health_response = requests.get(f"{API_BASE_URL}/health")
        assert health_response.status_code == 200
        
        # 2. Get model info (if available)
        info_response = requests.get(f"{API_BASE_URL}/model/info")
        if info_response.status_code == 200:
            model_info = info_response.json()
            
            # 3. Get example features
            example_response = requests.get(f"{API_BASE_URL}/example")
            assert example_response.status_code == 200
            
            example_features = example_response.json()["example_request"]["features"]
            
            # 4. Make prediction
            prediction_response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": example_features}
            )
            assert prediction_response.status_code == 200
            
            prediction = prediction_response.json()
            assert "prediction" in prediction
            assert "model_version" in prediction
            
            # Verify model version matches
            assert prediction["model_version"] == model_info["version"]

# Utility functions for testing
def wait_for_api(timeout=60):
    """Wait for API to become available"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(2)
    
    return False

def create_test_payload(feature_count=20):
    """Create a test payload with specified number of features"""
    import random
    
    features = [random.uniform(-2, 2) for _ in range(feature_count)]
    return {"features": features}

if __name__ == "__main__":
    # Check if API is available before running tests
    print("Checking API availability...")
    if wait_for_api(timeout=30):
        print("API is available, running tests...")
        pytest.main([
            "test_api.py",
            "-v",
            "--tb=short"
        ])
    else:
        print("API is not available. Please start the API server first:")
        print("python src/model_api.py")
