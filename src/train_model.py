"""
ML Model Training Script - Breast Cancer Model
Auto-Deployment ML Models Project

This script trains a machine learning model and saves it for deployment.
Now supports model selection via command line arguments.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name="ml_model", version="1.0.0"):
        self.model_name = model_name
        self.version = version
        self.model = None
        self.metrics = {}
        
    def load_breast_cancer_data(self):
        """Load the breast cancer dataset"""
        logger.info("Loading breast cancer dataset...")
        data = load_breast_cancer()
        
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist()
        
        # Create DataFrame for better handling
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/breast_cancer_dataset.csv', index=False)
        logger.info(f"Breast cancer dataset saved to data/breast_cancer_dataset.csv")
        logger.info(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def train_model(self, X, y):
        """Train the machine learning model"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'training_date': datetime.now().isoformat(),
            'model_version': self.version
        }
        
        logger.info(f"Training completed! Accuracy: {self.metrics['accuracy']:.4f}")
        return self.model, self.metrics
    
    def save_model(self, feature_names):
        """Save the trained model and metadata"""
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = f'models/{self.model_name}_v{self.version}.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'version': self.version,
            'model_path': model_path,
            'feature_names': feature_names,
            'metrics': self.metrics,
            'model_type': 'RandomForestClassifier',
            'sklearn_version': joblib.__version__,
            'dataset_info': {
                'name': 'Breast Cancer Wisconsin',
                'source': 'scikit-learn',
                'n_samples': 569,
                'n_features': 30,
                'n_classes': 2,
                'class_names': ['malignant', 'benign']
            }
        }
        
        metadata_path = f'models/{self.model_name}_v{self.version}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save latest model info
        latest_info = {
            'latest_model': model_path,
            'latest_metadata': metadata_path,
            'version': self.version,
            'created_at': datetime.now().isoformat()
        }
        
        with open('models/latest_model.json', 'w') as f:
            json.dump(latest_info, f, indent=2)
            
        # Update model history
        history_path = 'models/model_history.json'
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model history: {str(e)}")
        
        # Add new model to history if not already present
        if not any(entry['version'] == self.version for entry in history):
            history.append(latest_info)
            # Sort by creation date (newest last)
            history.sort(key=lambda x: x.get('created_at', ''))
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Model history updated in {history_path}")
        
        return model_path, metadata_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING ML MODEL TRAINING PIPELINE")
            logger.info("=" * 50)
            
            # Load breast cancer data
            X, y, feature_names = self.load_breast_cancer_data()
            
            # Train model
            model, metrics = self.train_model(X, y)
            
            # Save model
            model_path, metadata_path = self.save_model(feature_names)
            
            # Print summary
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Metadata saved: {metadata_path}")
            logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Model F1-score: {metrics['f1_score']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False

def main():
    """Main function to run model training"""
    parser = argparse.ArgumentParser(description='Train ML Model')
    parser.add_argument('--version', type=str, default="1.0.0", help='Model version')
    parser.add_argument('--name', type=str, default="breast_cancer_model", help='Model name')
    parser.add_argument('--model-type', type=str, default="breast_cancer", help='Model type (breast_cancer)')
    args = parser.parse_args()
    
    MODEL_NAME = args.name
    VERSION = args.version
    MODEL_TYPE = args.model_type
    
    # Validate model type
    if MODEL_TYPE != "breast_cancer":
        logger.error(f"Unsupported model type: {MODEL_TYPE}. Only 'breast_cancer' is supported.")
        exit(1)
    
    # Create trainer instance
    trainer = ModelTrainer(model_name=MODEL_NAME, version=VERSION)
    
    # Run training pipeline
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 Model training completed successfully!")
        print("📁 Check the 'models' directory for trained model files")
        print("📊 Next steps:")
        print("   1. Run model validation: python src/validate_model.py")
        print("   2. Start API server: python src/model_api.py")
        print("   3. Build Docker container: docker build -f docker/Dockerfile .")
    else:
        print("\n❌ Model training failed. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main()