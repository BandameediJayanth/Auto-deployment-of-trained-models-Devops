"""
Model Testing and Validation Module
Auto-Deployment ML Models Project

This module provides comprehensive testing for ML models to determine
if they are deployment-worthy before canary deployment.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTester:
    """
    Comprehensive model testing and validation for deployment readiness.
    """
    
    def __init__(self, test_data_path: str = 'data/dataset.csv'):
        self.test_data_path = test_data_path
        self.test_results = {}
        self.thresholds = {
            'min_accuracy': 0.75,
            'min_precision': 0.70,
            'min_recall': 0.70,
            'min_f1_score': 0.70,
            'max_latency_ms': 1000,
            'min_cv_mean': 0.70,
            'max_cv_std': 0.10
        }
    
    def load_model(self, model_path: str) -> Tuple[Any, Optional[Dict]]:
        """
        Load model and metadata from file paths.
        
        Returns:
            Tuple of (model, metadata)
        """
        try:
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            
            # Try to find metadata file
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if not os.path.exists(metadata_path):
                # Try alternative naming
                base_name = os.path.splitext(model_path)[0]
                metadata_path = f"{base_name}_metadata.json"
            
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                logger.warning(f"Metadata file not found for {model_path}")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load test data for validation"""
        try:
            if not os.path.exists(self.test_data_path):
                logger.warning(f"Test data not found at {self.test_data_path}")
                logger.info("Generating synthetic test data...")
                return self._generate_test_data()
            
            logger.info(f"Loading test data from {self.test_data_path}")
            df = pd.read_csv(self.test_data_path)
            
            if 'target' not in df.columns:
                raise ValueError("Test data must have 'target' column")
            
            X = df.drop('target', axis=1).values
            y = df['target'].values
            feature_names = df.drop('target', axis=1).columns.tolist()
            
            logger.info(f"Loaded {len(X)} samples with {len(feature_names)} features")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def _generate_test_data(self, n_samples: int = 200, n_features: int = 20):
        """Generate synthetic test data"""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        logger.info(f"Generated {n_samples} synthetic samples")
        return X, y, feature_names
    
    def test_model_structure(self, model: Any, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Test 1: Model structure validation"""
        logger.info("=" * 60)
        logger.info("TEST 1: Model Structure Validation")
        logger.info("=" * 60)
        
        results = {
            "test_name": "Model Structure",
            "passed": True,
            "checks": {},
            "errors": []
        }
        
        # Check if model exists
        if model is None:
            results["passed"] = False
            results["errors"].append("Model is None")
            return results
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            results["passed"] = False
            results["errors"].append("Model does not have predict() method")
        else:
            results["checks"]["has_predict"] = True
        
        # Check if model has predict_proba method (for confidence scores)
        if hasattr(model, 'predict_proba'):
            results["checks"]["has_predict_proba"] = True
        else:
            results["checks"]["has_predict_proba"] = False
            logger.warning("Model does not have predict_proba() method")
        
        # Check metadata if available
        if metadata:
            required_fields = ['model_name', 'version', 'feature_names']
            for field in required_fields:
                if field in metadata:
                    results["checks"][f"metadata_{field}"] = True
                else:
                    results["checks"][f"metadata_{field}"] = False
                    results["errors"].append(f"Metadata missing field: {field}")
        else:
            results["checks"]["metadata_available"] = False
            logger.warning("No metadata available")
        
        # Log results
        for check, passed in results["checks"].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check}: {status}")
        
        if results["errors"]:
            for error in results["errors"]:
                logger.error(f"  Error: {error}")
        
        return results
    
    def test_model_performance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Test 2: Model performance metrics"""
        logger.info("=" * 60)
        logger.info("TEST 2: Model Performance Metrics")
        logger.info("=" * 60)
        
        results = {
            "test_name": "Model Performance",
            "passed": True,
            "metrics": {},
            "thresholds": self.thresholds,
            "violations": []
        }
        
        try:
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            results["metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
            
            # Check against thresholds
            if accuracy < self.thresholds['min_accuracy']:
                results["passed"] = False
                results["violations"].append(f"Accuracy {accuracy:.4f} < {self.thresholds['min_accuracy']}")
            
            if precision < self.thresholds['min_precision']:
                results["passed"] = False
                results["violations"].append(f"Precision {precision:.4f} < {self.thresholds['min_precision']}")
            
            if recall < self.thresholds['min_recall']:
                results["passed"] = False
                results["violations"].append(f"Recall {recall:.4f} < {self.thresholds['min_recall']}")
            
            if f1 < self.thresholds['min_f1_score']:
                results["passed"] = False
                results["violations"].append(f"F1-score {f1:.4f} < {self.thresholds['min_f1_score']}")
            
            # Log results
            logger.info(f"  Accuracy: {accuracy:.4f} (threshold: {self.thresholds['min_accuracy']})")
            logger.info(f"  Precision: {precision:.4f} (threshold: {self.thresholds['min_precision']})")
            logger.info(f"  Recall: {recall:.4f} (threshold: {self.thresholds['min_recall']})")
            logger.info(f"  F1-Score: {f1:.4f} (threshold: {self.thresholds['min_f1_score']})")
            
            if results["violations"]:
                for violation in results["violations"]:
                    logger.error(f"  ❌ {violation}")
            else:
                logger.info("  ✅ All performance metrics passed")
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            logger.error(f"Performance test failed: {str(e)}")
        
        return results
    
    def test_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Test 3: Cross-validation stability"""
        logger.info("=" * 60)
        logger.info("TEST 3: Cross-Validation Stability")
        logger.info("=" * 60)
        
        results = {
            "test_name": "Cross-Validation",
            "passed": True,
            "metrics": {},
            "violations": []
        }
        
        try:
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results["metrics"] = {
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
                "cv_scores": cv_scores.tolist()
            }
            
            # Check thresholds
            if cv_mean < self.thresholds['min_cv_mean']:
                results["passed"] = False
                results["violations"].append(f"CV mean {cv_mean:.4f} < {self.thresholds['min_cv_mean']}")
            
            if cv_std > self.thresholds['max_cv_std']:
                results["passed"] = False
                results["violations"].append(f"CV std {cv_std:.4f} > {self.thresholds['max_cv_std']}")
            
            logger.info(f"  CV Mean: {cv_mean:.4f} (threshold: {self.thresholds['min_cv_mean']})")
            logger.info(f"  CV Std: {cv_std:.4f} (threshold: {self.thresholds['max_cv_std']})")
            
            if results["violations"]:
                for violation in results["violations"]:
                    logger.error(f"  ❌ {violation}")
            else:
                logger.info("  ✅ Cross-validation passed")
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            logger.error(f"Cross-validation test failed: {str(e)}")
        
        return results
    
    def test_prediction_latency(
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """Test 4: Prediction latency"""
        logger.info("=" * 60)
        logger.info("TEST 4: Prediction Latency")
        logger.info("=" * 60)
        
        results = {
            "test_name": "Prediction Latency",
            "passed": True,
            "metrics": {},
            "violations": []
        }
        
        try:
            latencies = []
            
            for i in range(n_iterations):
                sample_idx = i % len(X)
                sample = X[sample_idx:sample_idx+1]
                
                start_time = time.time()
                _ = model.predict(sample)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            results["metrics"] = {
                "avg_latency_ms": float(avg_latency),
                "p95_latency_ms": float(p95_latency),
                "p99_latency_ms": float(p99_latency),
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies))
            }
            
            # Check threshold (using p95)
            if p95_latency > self.thresholds['max_latency_ms']:
                results["passed"] = False
                results["violations"].append(
                    f"P95 latency {p95_latency:.2f}ms > {self.thresholds['max_latency_ms']}ms"
                )
            
            logger.info(f"  Average Latency: {avg_latency:.2f}ms")
            logger.info(f"  P95 Latency: {p95_latency:.2f}ms (threshold: {self.thresholds['max_latency_ms']}ms)")
            logger.info(f"  P99 Latency: {p99_latency:.2f}ms")
            
            if results["violations"]:
                for violation in results["violations"]:
                    logger.error(f"  ❌ {violation}")
            else:
                logger.info("  ✅ Latency test passed")
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            logger.error(f"Latency test failed: {str(e)}")
        
        return results
    
    def test_model_compatibility(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Test 5: Model compatibility and feature matching"""
        logger.info("=" * 60)
        logger.info("TEST 5: Model Compatibility")
        logger.info("=" * 60)
        
        results = {
            "test_name": "Model Compatibility",
            "passed": True,
            "checks": {},
            "errors": []
        }
        
        try:
            # Test prediction with correct number of features
            test_sample = np.random.randn(1, len(feature_names))
            prediction = model.predict(test_sample)
            
            results["checks"]["can_predict"] = True
            results["checks"]["feature_count_match"] = True
            
            # Test with wrong number of features
            try:
                wrong_sample = np.random.randn(1, len(feature_names) + 1)
                model.predict(wrong_sample)
                results["checks"]["feature_validation"] = False
                results["errors"].append("Model does not validate feature count")
            except (ValueError, IndexError):
                results["checks"]["feature_validation"] = True
            
            # Check if predict_proba works if available
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(test_sample)
                    results["checks"]["predict_proba_works"] = True
                    if not np.allclose(proba.sum(axis=1), 1.0):
                        results["errors"].append("Probabilities do not sum to 1")
                except Exception as e:
                    results["checks"]["predict_proba_works"] = False
                    results["errors"].append(f"predict_proba failed: {str(e)}")
            
            if results["errors"]:
                results["passed"] = False
            
            # Log results
            for check, passed in results["checks"].items():
                status = "✅ PASS" if passed else "❌ FAIL"
                logger.info(f"  {check}: {status}")
            
            if results["errors"]:
                for error in results["errors"]:
                    logger.error(f"  Error: {error}")
            
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)
            logger.error(f"Compatibility test failed: {str(e)}")
        
        return results
    
    def run_all_tests(
        self,
        model_path: str,
        test_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all tests on a model and return comprehensive results.
        
        Args:
            model_path: Path to the model file
            test_data_path: Optional path to test data (uses default if not provided)
        
        Returns:
            Dictionary with all test results and deployment recommendation
        """
        if test_data_path:
            self.test_data_path = test_data_path
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE MODEL TESTING FOR DEPLOYMENT READINESS")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Test Data: {self.test_data_path}")
        logger.info("")
        
        all_results = {
            "model_path": model_path,
            "test_timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_passed": True,
            "deployment_recommendation": "NOT READY",
            "summary": {}
        }
        
        try:
            # Load model
            model, metadata = self.load_model(model_path)
            
            # Load test data
            X, y, feature_names = self.load_test_data()
            
            # Run all tests
            test1 = self.test_model_structure(model, metadata)
            all_results["tests"].append(test1)
            if not test1["passed"]:
                all_results["overall_passed"] = False
            
            test2 = self.test_model_performance(model, X, y)
            all_results["tests"].append(test2)
            if not test2["passed"]:
                all_results["overall_passed"] = False
            
            test3 = self.test_cross_validation(model, X, y)
            all_results["tests"].append(test3)
            if not test3["passed"]:
                all_results["overall_passed"] = False
            
            test4 = self.test_prediction_latency(model, X)
            all_results["tests"].append(test4)
            if not test4["passed"]:
                all_results["overall_passed"] = False
            
            test5 = self.test_model_compatibility(model, feature_names)
            all_results["tests"].append(test5)
            if not test5["passed"]:
                all_results["overall_passed"] = False
            
            # Generate summary
            passed_tests = sum(1 for t in all_results["tests"] if t.get("passed", False))
            total_tests = len(all_results["tests"])
            
            all_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            }
            
            # Determine deployment recommendation
            if all_results["overall_passed"]:
                all_results["deployment_recommendation"] = "READY FOR DEPLOYMENT"
            else:
                all_results["deployment_recommendation"] = "NOT READY - FIX ISSUES FIRST"
            
            # Print final summary
            logger.info("")
            logger.info("=" * 80)
            logger.info("TEST SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed: {passed_tests}")
            logger.info(f"Failed: {total_tests - passed_tests}")
            logger.info(f"Pass Rate: {all_results['summary']['pass_rate']:.1%}")
            logger.info("")
            logger.info(f"DEPLOYMENT RECOMMENDATION: {all_results['deployment_recommendation']}")
            logger.info("=" * 80)
            
            # Save results
            results_file = f"models/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Test results saved to: {results_file}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            all_results["error"] = str(e)
            all_results["overall_passed"] = False
            all_results["deployment_recommendation"] = "ERROR - CANNOT TEST"
            return all_results
