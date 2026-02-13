"""
Automatic Rollback System
Monitors model performance and rolls back to previous version if degradation detected
"""

import os
import json
import logging
import joblib
import shutil
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rollback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RollbackSystem:
    """
    Automatic rollback system for model deployments
    Monitors performance and rolls back if degradation detected
    """
    
    def __init__(self, models_dir='models', performance_threshold=0.02):
        """
        Initialize rollback system
        
        Args:
            models_dir: Directory containing models
            performance_threshold: Performance drop threshold (e.g., 0.02 = 2% drop)
        """
        self.models_dir = models_dir
        self.performance_threshold = performance_threshold
        self.rollback_history_path = os.path.join(models_dir, 'rollback_history.json')
        
        logger.info("Rollback system initialized")
        logger.info(f"Performance threshold: {performance_threshold * 100}%")
    
    def get_model_history(self) -> List[Dict]:
        """Get model version history"""
        history_path = os.path.join(self.models_dir, 'model_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        
        return []
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get current model information"""
        latest_path = os.path.join(self.models_dir, 'latest_model.json')
        
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_previous_version(self, current_version: str) -> Optional[str]:
        """
        Get previous model version
        
        Args:
            current_version: Current version (e.g., "1.2.0")
            
        Returns:
            Previous version or None
        """
        history = self.get_model_history()
        
        # Find current version index
        for i, entry in enumerate(history):
            if entry['version'] == current_version:
                if i > 0:
                    return history[i - 1]['version']
                break
        
        return None
    
    def load_model(self, version: str):
        """Load a specific model version"""
        model_filename = f"breast_cancer_model_v{version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        
        return None
    
    def evaluate_model(self, model, X_test, y_test) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Performance metrics
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'test_samples': len(y_test)
        }
    
    def check_performance_degradation(self, current_metrics: Dict, 
                                     previous_metrics: Dict) -> bool:
        """
        Check if performance has degraded
        
        Args:
            current_metrics: Current model metrics
            previous_metrics: Previous model metrics
            
        Returns:
            True if degradation detected
        """
        current_acc = current_metrics.get('accuracy', 0)
        previous_acc = previous_metrics.get('accuracy', 0)
        
        degradation = previous_acc - current_acc
        
        logger.info(f"Performance comparison:")
        logger.info(f"  Current accuracy: {current_acc:.4f}")
        logger.info(f"  Previous accuracy: {previous_acc:.4f}")
        logger.info(f"  Degradation: {degradation:.4f} ({degradation * 100:.2f}%)")
        
        if degradation > self.performance_threshold:
            logger.warning(f"Performance degradation detected!")
            logger.warning(f"Drop: {degradation * 100:.2f}% (threshold: {self.performance_threshold * 100}%)")
            return True
        
        return False
    
    def perform_rollback(self, current_version: str, previous_version: str) -> Dict:
        """
        Perform rollback to previous version
        
        Args:
            current_version: Current (failing) version
            previous_version: Previous (stable) version
            
        Returns:
            Rollback results
        """
        logger.warning(f"\n" + "="*70)
        logger.warning(f"PERFORMING ROLLBACK")
        logger.warning(f"="*70)
        logger.warning(f"Rolling back from v{current_version} to v{previous_version}")
        
        # Update latest_model.json to point to previous version
        model_filename = f"breast_cancer_model_v{previous_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        metadata_filename = f"breast_cancer_model_v{previous_version}_metadata.json"
        metadata_path = os.path.join(self.models_dir, metadata_filename)
        
        latest_path = os.path.join(self.models_dir, 'latest_model.json')
        latest_data = {
            'version': previous_version,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'updated_at': datetime.now().isoformat(),
            'rollback_from': current_version
        }
        
        with open(latest_path, 'w') as f:
            json.dump(latest_data, f, indent=2)
        
        # Log rollback event
        rollback_event = {
            'timestamp': datetime.now().isoformat(),
            'from_version': current_version,
            'to_version': previous_version,
            'reason': 'performance_degradation',
            'threshold': self.performance_threshold
        }
        
        self.log_rollback_event(rollback_event)
        
        logger.warning(f"Rollback complete: v{current_version} to v{previous_version}")
        logger.warning(f"="*70 + "\n")
        
        return {
            'rollback_performed': True,
            'from_version': current_version,
            'to_version': previous_version,
            'timestamp': rollback_event['timestamp']
        }
    
    def log_rollback_event(self, event: Dict):
        """Log rollback event to history"""
        history = []
        if os.path.exists(self.rollback_history_path):
            with open(self.rollback_history_path, 'r') as f:
                history = json.load(f)
        
        history.append(event)
        
        with open(self.rollback_history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Rollback event logged")
    
    def monitor_and_rollback(self, X_test, y_test, simulate_degradation=False) -> Dict:
        """
        Monitor current model and rollback if needed
        
        Args:
            X_test: Test features
            y_test: Test labels
            simulate_degradation: Simulate performance drop for testing
            
        Returns:
            Monitoring results
        """
        logger.info("\n" + "="*70)
        logger.info("MONITORING MODEL PERFORMANCE")
        logger.info("="*70)
        
        # Get current model info
        current_info = self.get_current_model_info()
        if not current_info:
            logger.error("No current model found")
            return {'error': 'No current model'}
        
        current_version = current_info['version']
        logger.info(f"Current model version: v{current_version}")
        
        # Load and evaluate current model
        current_model = self.load_model(current_version)
        if not current_model:
            logger.error(f"Could not load model v{current_version}")
            return {'error': 'Could not load current model'}
        
        current_metrics = self.evaluate_model(current_model, X_test, y_test)
        
        # Simulate degradation if requested
        if simulate_degradation:
            logger.warning(f"Simulating performance degradation...")
            current_metrics['accuracy'] -= 0.05  # Simulate 5% drop
        
        # Get previous version
        previous_version = self.get_previous_version(current_version)
        
        result = {
            'current_version': current_version,
            'current_metrics': current_metrics,
            'degradation_detected': False,
            'rollback_performed': False
        }
        
        if previous_version:
            logger.info(f"Previous model version: v{previous_version}")
            
            # Get previous metrics from history
            history = self.get_model_history()
            previous_metrics = None
            
            for entry in history:
                if entry['version'] == previous_version:
                    previous_metrics = entry.get('metrics', {})
                    break
            
            if previous_metrics:
                # Check for degradation
                degradation = self.check_performance_degradation(
                    current_metrics, previous_metrics
                )
                
                result['degradation_detected'] = degradation
                result['previous_version'] = previous_version
                result['previous_metrics'] = previous_metrics
                
                if degradation:
                    # Perform rollback
                    rollback_result = self.perform_rollback(
                        current_version, previous_version
                    )
                    result.update(rollback_result)
                else:
                    logger.info("No performance degradation detected")
            else:
                logger.warning("Could not find previous metrics for comparison")
        else:
            logger.info("No previous version available for comparison")
        
        logger.info("="*70 + "\n")
        
        return result


def main():
    """Main function to demonstrate rollback system"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*70)
    print("AUTOMATIC ROLLBACK DEMONSTRATION")
    print("="*70 + "\n")
    
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    
    # Initialize rollback system
    rollback_system = RollbackSystem(performance_threshold=0.02)
    
    # Test 1: Normal performance (no rollback)
    print("\n" + "-"*70)
    print("TEST 1: Normal Performance (No Rollback Expected)")
    print("-"*70)
    result1 = rollback_system.monitor_and_rollback(X_test, y_test, simulate_degradation=False)
    print(f"Degradation Detected: {result1.get('degradation_detected', False)}")
    print(f"Rollback Performed: {result1.get('rollback_performed', False)}")
    
    # Test 2: Simulated degradation (rollback expected)
    print("\n" + "-"*70)
    print("TEST 2: Simulated Degradation (Rollback Expected)")
    print("-"*70)
    result2 = rollback_system.monitor_and_rollback(X_test, y_test, simulate_degradation=True)
    print(f"Degradation Detected: {result2.get('degradation_detected', False)}")
    print(f"Rollback Performed: {result2.get('rollback_performed', False)}")
    
    if result2.get('rollback_performed'):
        print(f"Rolled back from v{result2['from_version']} to v{result2['to_version']}")
    
    print("\n" + "="*70)
    print("Rollback demonstration complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
