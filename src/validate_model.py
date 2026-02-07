"""
Model Validation Script
Auto-Deployment ML Models Project

This script validates trained models for deployment readiness.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
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
    def __init__(self):
        self.model = None
        self.metadata = None
        self.validation_results = {}
        
        # Validation thresholds
        self.thresholds = {
            'min_accuracy': 0.8,
            'min_precision': 0.8,
            'min_recall': 0.8,
            'min_f1_score': 0.8,
            'max_cv_std': 0.05,  # Maximum standard deviation for cross-validation
            'min_cv_mean': 0.8   # Minimum mean for cross-validation
        }
    
    def load_latest_model(self):
        """Load the latest trained model"""
        try:
            # Load latest model info
            with open('models/latest_model.json', 'r') as f:
                latest_info = json.load(f)
            
            model_path = latest_info['latest_model']
            metadata_path = latest_info['latest_metadata']
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Model loaded successfully: {self.metadata['model_name']} v{self.metadata['version']}")
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
            'has_feature_names': 'feature_names' in self.metadata,
            'has_metrics': 'metrics' in self.metadata,
            'has_version': 'version' in self.metadata
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
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': self.metadata.get('model_name', 'unknown'),
                'version': self.metadata.get('version', 'unknown'),
                'type': self.metadata.get('model_type', 'unknown')
            },
            'validation_results': self.validation_results,
            'thresholds_used': self.thresholds,
            'overall_status': 'PASS' if self.validation_results.get('overall_pass', False) else 'FAIL'
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
            logger.info("=" * 60)
            logger.info("STARTING MODEL VALIDATION PIPELINE")
            logger.info("=" * 60)
            
            # Load model
            if not self.load_latest_model():
                return False
            
            # Load test data
            X, y = self.load_test_data()
            if X is None or y is None:
                return False
            
            all_validations = {}
            
            # 1. Validate model structure
            logger.info("\n1. MODEL STRUCTURE VALIDATION")
            logger.info("-" * 40)
            structure_valid = self.validate_model_structure()
            all_validations['structure_valid'] = structure_valid
            
            if not structure_valid:
                logger.error("Model structure validation failed!")
                return False
            
            # 2. Validate model performance
            logger.info("\n2. MODEL PERFORMANCE VALIDATION")
            logger.info("-" * 40)
            perf_validations, perf_metrics = self.validate_model_performance(X, y)
            all_validations.update(perf_validations)
            
            # 3. Cross-validation
            logger.info("\n3. CROSS-VALIDATION")
            logger.info("-" * 40)
            cv_validations, cv_metrics = self.validate_cross_validation(X, y)
            all_validations.update(cv_validations)
            
            # 4. Prediction capability test
            logger.info("\n4. PREDICTION CAPABILITY TEST")
            logger.info("-" * 40)
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
            logger.info("\n" + "=" * 60)
            logger.info("VALIDATION PIPELINE COMPLETED")
            logger.info("=" * 60)
            
            if overall_pass:
                logger.info("🎉 MODEL VALIDATION PASSED - READY FOR DEPLOYMENT!")
                logger.info(f"📊 Validation report: {report_path}")
                logger.info("\n📋 Next steps:")
                logger.info("   1. Start API server: python src/model_api.py")
                logger.info("   2. Build Docker container: docker build -f docker/Dockerfile .")
                logger.info("   3. Deploy to production environment")
            else:
                logger.error("❌ MODEL VALIDATION FAILED - NOT READY FOR DEPLOYMENT")
                logger.error("Check the validation report for details on failed checks")
            
            return overall_pass
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            return False

def main():
    """Main function to run model validation"""
    validator = ModelValidator()
    
    # Check if model exists
    if not os.path.exists('models/latest_model.json'):
        print("❌ No trained model found!")
        print("Please run model training first: python src/train_model.py")
        exit(1)
    
    # Run validation pipeline
    success = validator.run_validation_pipeline()
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
