"""
Automatic Model Retraining System
Triggers retraining when drift is detected
"""

import os
import json
import logging
import joblib
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer

from drift_detection_enhanced import EnhancedDriftDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoRetrainingSystem:
    """
    Automatic model retraining system triggered by drift detection
    """
    
    def __init__(self, models_dir='models', drift_threshold=0.2):
        """
        Initialize auto-retraining system
        
        Args:
            models_dir: Directory to save models
            drift_threshold: Drift score threshold to trigger retraining
        """
        self.models_dir = models_dir
        self.drift_threshold = drift_threshold
        self.drift_detector = None
        
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info("Auto-retraining system initialized")
        logger.info(f"Drift threshold: {drift_threshold}")
    
    def get_current_version(self) -> str:
        """Get current model version from latest_model.json"""
        latest_path = os.path.join(self.models_dir, 'latest_model.json')
        
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                data = json.load(f)
                return data.get('version', '1.0.0')
        
        return '1.0.0'
    
    def increment_version(self, current_version: str) -> str:
        """
        Increment model version (minor version)
        
        Args:
            current_version: Current version (e.g., "1.0.0")
            
        Returns:
            New version (e.g., "1.1.0")
        """
        parts = current_version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Increment minor version for retraining
        new_version = f"{major}.{minor + 1}.{patch}"
        
        logger.info(f"Version incremented: {current_version} to {new_version}")
        
        return new_version
    
    def train_model(self, X_train, y_train, X_test, y_test, version: str) -> Dict:
        """
        Train a new model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            version: Model version
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training new model version {version}...")
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted'))
        }
        
        logger.info(f"Model trained successfully")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'version': version,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def save_model(self, model, metadata: Dict, version: str):
        """
        Save model and metadata
        
        Args:
            model: Trained model
            metadata: Model metadata
            version: Model version
        """
        # Save model
        model_filename = f"breast_cancer_model_v{version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save metadata
        metadata_filename = f"breast_cancer_model_v{version}_metadata.json"
        metadata_path = os.path.join(self.models_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Update latest_model.json
        latest_path = os.path.join(self.models_dir, 'latest_model.json')
        latest_data = {
            'version': version,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(latest_path, 'w') as f:
            json.dump(latest_data, f, indent=2)
        logger.info(f"Latest model updated: v{version}")
        
        # Update model history
        self.update_model_history(version, metadata)
    
    def update_model_history(self, version: str, metadata: Dict):
        """Update model version history"""
        history_path = os.path.join(self.models_dir, 'model_history.json')
        
        history = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        history.append({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metadata.get('metrics', {}),
            'trigger': metadata.get('trigger', 'manual')
        })
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Model history updated")
    
    def check_and_retrain(self, current_data=None) -> Dict:
        """
        Check for drift and trigger retraining if needed
        
        Args:
            current_data: Current production data (optional, uses test data if None)
            
        Returns:
            Retraining results
        """
        logger.info("\n" + "="*70)
        logger.info("CHECKING FOR DRIFT AND RETRAINING NEED")
        logger.info("="*70)
        
        # Load data
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize drift detector if not already done
        if self.drift_detector is None:
            import pandas as pd
            ref_df = pd.DataFrame(X_train, columns=feature_names)
            ref_df.to_csv('data/dataset.csv', index=False)
            self.drift_detector = EnhancedDriftDetector()
        
        # Check for drift
        if current_data is None:
            # Use test data for demonstration
            import pandas as pd
            current_data = pd.DataFrame(X_test, columns=feature_names)
        
        drift_results = self.drift_detector.check_drift(current_data)
        
        result = {
            'drift_detected': drift_results['drift_detected'],
            'drift_score': drift_results['drift_score'],
            'retraining_triggered': False,
            'new_version': None,
            'metrics': None
        }
        
        # Check if retraining needed
        if drift_results['drift_detected'] or drift_results['drift_score'] > self.drift_threshold:
            logger.warning(f"DRIFT THRESHOLD EXCEEDED!")
            logger.warning(f"Drift Score: {drift_results['drift_score']:.4f} (threshold: {self.drift_threshold})")
            logger.warning(f"Triggering automatic retraining...")
            
            # Get current version and increment
            current_version = self.get_current_version()
            new_version = self.increment_version(current_version)
            
            # Train new model
            training_results = self.train_model(X_train, y_train, X_test, y_test, new_version)
            
            # Prepare metadata
            metadata = {
                'model_name': 'breast_cancer_model',
                'version': new_version,
                'algorithm': 'RandomForestClassifier',
                'metrics': training_results['metrics'],
                'training_date': training_results['training_date'],
                'training_samples': training_results['training_samples'],
                'test_samples': training_results['test_samples'],
                'feature_names': feature_names,
                'features': len(feature_names),
                'trigger': 'drift_detection',
                'drift_score': drift_results['drift_score'],
                'previous_version': current_version
            }
            
            # Save model
            self.save_model(training_results['model'], metadata, new_version)
            
            result.update({
                'retraining_triggered': True,
                'new_version': new_version,
                'previous_version': current_version,
                'metrics': training_results['metrics']
            })
            
            logger.info(f"\nRETRAINING COMPLETE!")
            logger.info(f"New Version: v{new_version}")
            logger.info(f"Accuracy: {training_results['metrics']['accuracy']:.4f}")
            logger.info(f"Trigger: Drift detection (score: {drift_results['drift_score']:.4f})")
            
        else:
            logger.info(f"No drift detected. Retraining not needed.")
            logger.info(f"Drift Score: {drift_results['drift_score']:.4f} (threshold: {self.drift_threshold})")
        
        logger.info("="*70 + "\n")
        
        return result


def main():
    """Main function to demonstrate auto-retraining"""
    print("\n" + "="*70)
    print("AUTOMATIC RETRAINING DEMONSTRATION")
    print("="*70 + "\n")
    
    # Initialize system
    retraining_system = AutoRetrainingSystem(drift_threshold=0.2)
    
    # Test 1: Check with clean data (no retraining expected)
    print("\n" + "-"*70)
    print("TEST 1: Clean Data (No Retraining Expected)")
    print("-"*70)
    result1 = retraining_system.check_and_retrain()
    print(f"Drift Detected: {result1['drift_detected']}")
    print(f"Retraining Triggered: {result1['retraining_triggered']}")
    
    # Test 2: Simulate drift (retraining expected)
    print("\n" + "-"*70)
    print("TEST 2: Simulated Drift (Retraining Expected)")
    print("-"*70)
    
    # Load data and simulate drift
    from drift_detection_enhanced import simulate_drift
    import pandas as pd
    
    data = load_breast_cancer()
    X = data.data
    feature_names = data.feature_names.tolist()
    
    # Create drifted data
    X_train, X_test, _, _ = train_test_split(
        X, data.target, test_size=0.2, random_state=42
    )
    
    drifted_df = pd.DataFrame(X_test, columns=feature_names)
    drifted_df = simulate_drift(drifted_df, drift_magnitude=0.8)
    
    result2 = retraining_system.check_and_retrain(current_data=drifted_df)
    print(f"Drift Detected: {result2['drift_detected']}")
    print(f"Retraining Triggered: {result2['retraining_triggered']}")
    
    if result2['retraining_triggered']:
        print(f"New Version: v{result2['new_version']}")
        print(f"Previous Version: v{result2['previous_version']}")
        print(f"New Accuracy: {result2['metrics']['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("Auto-retraining demonstration complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
