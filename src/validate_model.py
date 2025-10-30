"""
Model Validation Script
Auto-Deployment ML Models Project

This script validates trained models for deployment readiness.
"""


import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, model_path=None):
        self.model = None
        self.metadata = None
        self.validation_results = {}
        self.model_path = model_path
        self.metadata_path = None
        # Validation thresholds
        self.thresholds = {
            'min_accuracy': 0.8,
            'min_precision': 0.8,
            'min_recall': 0.8,
            'min_f1_score': 0.8,
            'max_cv_std': 0.05,  # Maximum standard deviation for cross-validation
            'min_cv_mean': 0.8   # Minimum mean for cross-validation
        }

    def load_model(self):
        """Load the specified model and metadata"""
        try:
            if not self.model_path:
                raise ValueError("No model path provided.")
            self.metadata_path = os.path.splitext(self.model_path)[0] + '_metadata.json'
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Model loaded successfully: {self.metadata.get('model_name', 'unknown')} v{self.metadata.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_test_data(self, data_path='data/dataset.csv'):
        """Load test data for validation"""
        try:
            logger.info(f"Loading test data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            logger.info(f"Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X.values, y.values
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return None, None
    
    def validate_model_structure(self):
        """Validate model structure and properties"""
        logger.info("Validating model structure...")
        validations = {
            'model_exists': self.model is not None,
            'metadata_exists': self.metadata is not None,
            'has_feature_names': self.metadata is not None and 'feature_names' in self.metadata,
            'has_metrics': self.metadata is not None and 'metrics' in self.metadata,
            'has_version': self.metadata is not None and 'version' in self.metadata
        }
        all_passed = all(validations.values())
        for check, passed in validations.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check}: {status}")
        return all_passed
    
    def validate_model_performance(self, X, y):
        """Validate model performance metrics"""
        logger.info("Validating model performance...")
        try:
            if self.model is None:
                raise ValueError("Model is not loaded.")
            # Make predictions
            y_pred = self.model.predict(X)
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1_score': f1_score(y, y_pred, average='weighted')
            }
            # Validate against thresholds
            validations = {}
            for metric, value in metrics.items():
                threshold_key = f'min_{metric}'
                if threshold_key in self.thresholds:
                    threshold = self.thresholds[threshold_key]
                    validations[f'{metric}_threshold'] = value >= threshold
                    status = "✅ PASS" if value >= threshold else "❌ FAIL"
                    logger.info(f"  {metric}: {value:.4f} (threshold: {threshold:.2f}) {status}")
            return validations, metrics
        except Exception as e:
            logger.error(f"Error validating performance: {str(e)}")
            return {}, {}
    
    def validate_cross_validation(self, X, y):
        """Perform cross-validation for model stability"""
        logger.info("Performing cross-validation...")
        try:
            if self.model is None:
                raise ValueError("Model is not loaded.")
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            validations = {
                'cv_mean_threshold': cv_mean >= self.thresholds['min_cv_mean'],
                'cv_std_threshold': cv_std <= self.thresholds['max_cv_std']
            }
            status_mean = "✅ PASS" if validations['cv_mean_threshold'] else "❌ FAIL"
            status_std = "✅ PASS" if validations['cv_std_threshold'] else "❌ FAIL"
            logger.info(f"  CV Mean: {cv_mean:.4f} (threshold: {self.thresholds['min_cv_mean']:.2f}) {status_mean}")
            logger.info(f"  CV Std: {cv_std:.4f} (threshold: {self.thresholds['max_cv_std']:.2f}) {status_std}")
            return validations, {'cv_mean': cv_mean, 'cv_std': cv_std, 'cv_scores': cv_scores.tolist()}
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {}, {}
    
    def validate_prediction_capability(self, X):
        """Test model prediction capability"""
        logger.info("Testing prediction capability...")
        try:
            if self.model is None:
                raise ValueError("Model is not loaded.")
            # Test with a single sample
            sample = X[0:1]
            prediction = self.model.predict(sample)
            prediction_proba = None
            # Test probability prediction if available
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(sample)
            validations = {
                'can_predict': True,
                'prediction_shape_correct': prediction.shape[0] == 1,
                'prediction_is_numeric': np.isfinite(prediction).all()
            }
            if prediction_proba is not None:
                validations['can_predict_proba'] = True
                validations['proba_sums_to_one'] = np.allclose(prediction_proba.sum(axis=1), 1.0)
            all_passed = all(validations.values())
            status = "✅ PASS" if all_passed else "❌ FAIL"
            logger.info(f"  Prediction capability: {status}")
            return validations
        except Exception as e:
            logger.error(f"Error testing predictions: {str(e)}")
            return {'can_predict': False}
    
    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Determine overall status from nested validations (the pipeline stores overall_pass inside
        # self.validation_results['validations']). Keep a top-level overall_pass boolean for clarity.
        nested_overall = False
        try:
            nested_overall = bool(self.validation_results.get('validations', {}).get('overall_pass', False))
        except Exception:
            nested_overall = False

        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': self.metadata.get('model_name', 'unknown') if self.metadata else 'unknown',
                'version': self.metadata.get('version', 'unknown') if self.metadata else 'unknown',
                'type': self.metadata.get('model_type', 'unknown') if self.metadata else 'unknown'
            },
            'validation_results': convert_numpy_types(self.validation_results),
            'thresholds_used': self.thresholds,
            'overall_pass': nested_overall,
            'overall_status': 'PASS' if nested_overall else 'FAIL'
        }
        
        # Save validation report
        os.makedirs('models', exist_ok=True)
        report_path = f"models/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        return report, report_path
    
    def run_validation_pipeline(self):
        """Run the complete validation pipeline"""
        try:
            print("=" * 60)
            print("STARTING MODEL VALIDATION PIPELINE")
            print("=" * 60)

            # Load model
            if not self.load_model():
                print("Error: Could not load model or metadata.", file=sys.stderr)
                return False

            # Load test data
            X, y = self.load_test_data()
            if X is None or y is None:
                print("Error: Could not load test data.", file=sys.stderr)
                return False

            all_validations = {}

            # 1. Validate model structure
            print("\n1. MODEL STRUCTURE VALIDATION")
            print("-" * 40)
            structure_valid = self.validate_model_structure()
            all_validations['structure_valid'] = structure_valid

            if not structure_valid:
                print("Model structure validation failed!", file=sys.stderr)
                return False

            # 2. Validate model performance
            print("\n2. MODEL PERFORMANCE VALIDATION")
            print("-" * 40)
            perf_validations, perf_metrics = self.validate_model_performance(X, y)
            all_validations.update(perf_validations)

            # 3. Cross-validation
            print("\n3. CROSS-VALIDATION")
            print("-" * 40)
            cv_validations, cv_metrics = self.validate_cross_validation(X, y)
            all_validations.update(cv_validations)

            # 4. Prediction capability test
            print("\n4. PREDICTION CAPABILITY TEST")
            print("-" * 40)
            pred_validations = self.validate_prediction_capability(X)
            all_validations.update(pred_validations)

            # Determine overall validation status
            overall_pass = all(all_validations.values())
            all_validations['overall_pass'] = overall_pass

            # Store results
            self.validation_results = {
                'validations': all_validations,
                'performance_metrics': perf_metrics,
                'cross_validation_metrics': cv_metrics
            }

            # Generate report
            report, report_path = self.generate_validation_report()

            # Print summary
            print("\n" + "=" * 60)
            print("VALIDATION PIPELINE COMPLETED")
            print("=" * 60)

            if overall_pass:
                print("MODEL VALIDATION PASSED - READY FOR DEPLOYMENT!")
                print(f"Validation report: {report_path}")
            else:
                print("MODEL VALIDATION FAILED - NOT READY FOR DEPLOYMENT", file=sys.stderr)
                print("Check the validation report for details on failed checks", file=sys.stderr)

            return overall_pass

        except Exception as e:
            print(f"Validation pipeline failed: {str(e)}", file=sys.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(description="Validate a trained ML model for deployment readiness.")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.pkl)')
    args = parser.parse_args()

    validator = ModelValidator(model_path=args.model)
    success = validator.run_validation_pipeline()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
