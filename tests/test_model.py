"""
Unit Tests for ML Model
Auto-Deployment ML Models Project

Tests for model training, validation, and core functionality.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append('src')

from train_model import ModelTrainer
from validate_model import ModelValidator
from utils import load_config, validate_data_schema, format_metrics

class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
        
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer("test_model", "1.0.0")
        
        assert trainer.model_name == "test_model"
        assert trainer.version == "1.0.0"
        assert trainer.model is None
        assert trainer.metrics == {}
        
    def test_generate_sample_data(self):
        """Test sample data generation"""
        trainer = ModelTrainer()
        X, y, feature_names = trainer.generate_sample_data(n_samples=100, n_features=10)
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert len(feature_names) == 10
        assert all(name.startswith('feature_') for name in feature_names)
        assert os.path.exists('data/dataset.csv')
        
    def test_train_model(self):
        """Test model training"""
        trainer = ModelTrainer()
        X, y, feature_names = trainer.generate_sample_data(n_samples=100, n_features=5)
        
        model, metrics = trainer.train_model(X, y)
        
        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] > 0
        assert metrics['training_samples'] > 0
        assert metrics['test_samples'] > 0
        
    def test_save_model(self):
        """Test model saving"""
        trainer = ModelTrainer("test_model", "1.0.0")
        X, y, feature_names = trainer.generate_sample_data(n_samples=100, n_features=5)
        trainer.train_model(X, y)
        
        model_path, metadata_path = trainer.save_model(feature_names)
        
        assert os.path.exists(model_path)
        assert os.path.exists(metadata_path)
        assert os.path.exists('models/latest_model.json')
        
        # Verify saved model can be loaded
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None
        
        # Verify metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['model_name'] == "test_model"
        assert metadata['version'] == "1.0.0"
        assert 'metrics' in metadata
        assert 'feature_names' in metadata

class TestModelValidator:
    """Test cases for ModelValidator class"""
    
    def setup_method(self):
        """Setup test environment with a trained model"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Train a test model
        trainer = ModelTrainer("test_model", "1.0.0")
        X, y, feature_names = trainer.generate_sample_data(n_samples=200, n_features=5)
        trainer.train_model(X, y)
        trainer.save_model(feature_names)
        
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
        
    def test_validator_initialization(self):
        """Test ModelValidator initialization"""
        model_path = 'models/test_model_v1.0.0.pkl'
        validator = ModelValidator(model_path=model_path)
        assert validator.model is None
        assert validator.metadata is None
        assert validator.validation_results == {}
        assert 'min_accuracy' in validator.thresholds
        
    def test_load_model(self):
        """Test loading model"""
        model_path = 'models/test_model_v1.0.0.pkl'
        validator = ModelValidator(model_path=model_path)
        success = validator.load_model()
        assert success is True
        assert validator.model is not None
        assert validator.metadata is not None
        assert validator.metadata['model_name'] == "test_model"
        
    def test_validate_model_structure(self):
        """Test model structure validation"""
        model_path = 'models/test_model_v1.0.0.pkl'
        validator = ModelValidator(model_path=model_path)
        validator.load_model()
        is_valid = validator.validate_model_structure()
        assert is_valid is True
        
    def test_validate_model_performance(self):
        """Test model performance validation"""
        model_path = 'models/test_model_v1.0.0.pkl'
        validator = ModelValidator(model_path=model_path)
        validator.load_model()
        X, y = validator.load_test_data()
        validations, metrics = validator.validate_model_performance(X, y)
        assert 'accuracy_threshold' in validations
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0
        
    def test_validate_prediction_capability(self):
        """Test prediction capability validation"""
        model_path = 'models/test_model_v1.0.0.pkl'
        validator = ModelValidator(model_path=model_path)
        validator.load_model()
        X, y = validator.load_test_data()
        validations = validator.validate_prediction_capability(X)
        assert validations['can_predict'] is True
        assert validations['prediction_shape_correct'] is True

class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_load_config_default(self):
        """Test loading default configuration"""
        config = load_config('nonexistent_config.json')
        
        assert 'model' in config
        assert 'training' in config
        assert 'validation' in config
        assert config['model']['name'] == 'ml_model'
        
    def test_validate_data_schema_valid(self):
        """Test data schema validation with valid data"""
        # Create valid test data
        data = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [4.0, 5.0, 6.0],
            'target': [0, 1, 0]
        })
        
        expected_columns = ['feature_1', 'feature_2', 'target']
        required_columns = ['feature_1', 'target']
        
        is_valid, errors = validate_data_schema(data, expected_columns, required_columns)
        
        assert is_valid is True
        assert len(errors) == 0
        
    def test_validate_data_schema_invalid(self):
        """Test data schema validation with invalid data"""
        # Create invalid test data (missing required column)
        data = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'unexpected_col': ['a', 'b', 'c']
        })
        
        expected_columns = ['feature_1', 'feature_2', 'target']
        required_columns = ['feature_1', 'target']
        
        is_valid, errors = validate_data_schema(data, expected_columns, required_columns)
        
        assert is_valid is False
        assert len(errors) > 0
        
    def test_format_metrics(self):
        """Test metrics formatting"""
        metrics = {
            'accuracy': 0.85123456,
            'training_samples': 1000,
            'some_string': 'test'
        }
        
        formatted = format_metrics(metrics, decimal_places=2)
        
        assert formatted['accuracy'] == '0.85'
        assert formatted['training_samples'] == '1,000'
        assert formatted['some_string'] == 'test'

class TestModelPredictions:
    """Test model prediction functionality"""
    
    def setup_method(self):
        """Setup test environment with a trained model"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Train a test model
        trainer = ModelTrainer("test_model", "1.0.0")
        X, y, feature_names = trainer.generate_sample_data(n_samples=200, n_features=5)
        trainer.train_model(X, y)
        trainer.save_model(feature_names)
        
        # Load the model for testing
        self.model = joblib.load('models/test_model_v1.0.0.pkl')
        
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
        
    def test_model_prediction_shape(self):
        """Test model prediction output shape"""
        # Create test input
        test_input = np.random.randn(1, 5)
        
        prediction = self.model.predict(test_input)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]  # Binary classification
        
    def test_model_prediction_probability(self):
        """Test model probability prediction"""
        # Create test input
        test_input = np.random.randn(1, 5)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(test_input)
            
            assert probabilities.shape[0] == 1
            assert probabilities.shape[1] == 2  # Binary classification
            assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
            
    def test_model_batch_prediction(self):
        """Test model batch prediction"""
        # Create batch test input
        batch_input = np.random.randn(10, 5)
        
        predictions = self.model.predict(batch_input)
        
        assert predictions.shape == (10,)
        assert all(pred in [0, 1] for pred in predictions)

# Performance and stress tests
class TestModelPerformance:
    """Test model performance characteristics"""
    
    @pytest.mark.slow
    def test_training_time_reasonable(self):
        """Test that model training completes in reasonable time"""
        import time
        
        trainer = ModelTrainer()
        X, y, feature_names = trainer.generate_sample_data(n_samples=1000, n_features=20)
        
        start_time = time.time()
        trainer.train_model(X, y)
        training_time = time.time() - start_time
        
        # Training should complete within 30 seconds
        assert training_time < 30.0
        
    @pytest.mark.slow
    def test_prediction_latency(self):
        """Test prediction latency for single sample"""
        import time
        
        # Setup
        trainer = ModelTrainer()
        X, y, feature_names = trainer.generate_sample_data(n_samples=200, n_features=10)
        trainer.train_model(X, y)
        
        # Test prediction latency
        test_input = np.random.randn(1, 10)
        
        start_time = time.time()
        prediction = trainer.model.predict(test_input)
        prediction_time = time.time() - start_time
        
        # Prediction should be very fast (< 0.1 seconds)
        assert prediction_time < 0.1
        assert prediction is not None

# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete training -> validation -> prediction pipeline"""
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Create directories
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            # Step 1: Train model
            trainer = ModelTrainer("integration_test", "1.0.0")
            success = trainer.run_training_pipeline()
            assert success is True
            # Step 2: Validate model
            model_path = 'models/integration_test_v1.0.0.pkl'
            validator = ModelValidator(model_path=model_path)
            validator.load_model()
            success = validator.run_validation_pipeline()
            assert success is True
            # Step 3: Test model loading and prediction
            with open('models/latest_model.json', 'r') as f:
                latest_info = json.load(f)
            model = joblib.load(latest_info['latest_model'])
            test_input = np.random.randn(1, 20)  # Default is 20 features
            prediction = model.predict(test_input)
            assert prediction is not None
            assert len(prediction) == 1
        finally:
            os.chdir(original_dir)
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "test_model.py",
        "-v",
        "--cov=../src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
