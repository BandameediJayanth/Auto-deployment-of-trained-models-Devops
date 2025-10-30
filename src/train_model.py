"""
ML Model Training Script
Auto-Deployment ML Models Project

This script trains a machine learning model and saves it for deployment.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import argparse
import sys

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
        self.last_model_path = None
        self.last_metadata_path = None
        
    def generate_sample_data(self, n_samples=1000, n_features=20):
        """Generate sample dataset for demonstration"""
        logger.info(f"Generating sample dataset with {n_samples} samples and {n_features} features")
        
        n_informative = max(1, int(n_features * 0.75))
        n_redundant = n_features - n_informative
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/dataset.csv', index=False)
        logger.info("Dataset saved to data/dataset.csv")
        
        return X, y, feature_names
    
    def load_data(self, data_path='data/dataset.csv'):
        """Load data from CSV file"""
        try:
            if not os.path.exists(data_path):
                logger.warning(f"Data file {data_path} not found. Generating sample data...")
                return self.generate_sample_data()
            
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            feature_names = X.columns.tolist()
            
            logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X.values, y.values, feature_names
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.info("Generating sample data instead...")
            return self.generate_sample_data()
    
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
            'sklearn_version': joblib.__version__
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

        # store last saved paths on the trainer instance
        self.last_model_path = model_path
        self.last_metadata_path = metadata_path
        return model_path, metadata_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING ML MODEL TRAINING PIPELINE")
            logger.info("=" * 50)
            
            # Load data
            X, y, feature_names = self.load_data()
            
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
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--model-name', type=str, default='ml_classifier', help='Model name to save')
    parser.add_argument('--version', type=str, default='1.0.0', help='Model version')
    args = parser.parse_args()

    # If running interactively, allow choosing/entering model name and version
    model_name = args.model_name
    version = args.version
    if sys.stdin.isatty():
        try:
            # List existing models as presets if available
            existing = []
            if os.path.exists('models'):
                for fn in os.listdir('models'):
                    if fn.endswith('.pkl'):
                        existing.append(fn)

            if existing:
                print("Existing models:")
                for i, fn in enumerate(existing, start=1):
                    print(f"  {i}. {fn}")
                print("  0. Enter new model name/version")
                choice = input("Select a model to overwrite or 0 to enter new: ")
                if choice.strip().isdigit() and int(choice) > 0 and int(choice) <= len(existing):
                    sel = existing[int(choice) - 1]
                    # derive model name and version from filename if possible
                    base = sel.replace('.pkl', '')
                    if '_v' in base:
                        parts = base.split('_v')
                        model_name = parts[0]
                        version = parts[1]
                    else:
                        model_name = base
                else:
                    user_input = input(f"Model name [{model_name}]: ")
                    if user_input.strip():
                        model_name = user_input.strip()
                    user_input = input(f"Version [{version}]: ")
                    if user_input.strip():
                        version = user_input.strip()
            else:
                user_input = input(f"Model name [{model_name}]: ")
                if user_input.strip():
                    model_name = user_input.strip()
                user_input = input(f"Version [{version}]: ")
                if user_input.strip():
                    version = user_input.strip()
        except Exception:
            # If prompting fails for any reason, fall back to args/defaults
            pass

    # Create trainer instance
    trainer = ModelTrainer(model_name=model_name, version=version)

    # Run training pipeline
    success = trainer.run_training_pipeline()
    
    if success:
        print("\nModel training completed successfully!")
        saved_model = trainer.last_model_path if getattr(trainer, 'last_model_path', None) else f"models/{model_name}_v{version}.pkl"
        saved_meta = trainer.last_metadata_path if getattr(trainer, 'last_metadata_path', None) else f"models/{model_name}_v{version}_metadata.json"
        print(f"Model saved: {saved_model}")
        print(f"Metadata saved: {saved_meta}")
        print("Next steps:")
        print(f"   1. Run model validation: python src/validate_model.py --model {saved_model}")
    else:
        print("\nModel training failed. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main()
