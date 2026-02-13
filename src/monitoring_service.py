"""
Autonomous Drift Monitoring Service

This service runs continuously as a separate Docker container,
monitoring for drift and triggering retraining when needed.

Lifecycle: Tied to docker-compose (starts/stops with the system)
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from drift_detection_enhanced import EnhancedDriftDetector
from auto_retraining import AutoRetrainingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftMonitoringService:
    """
    Autonomous drift monitoring service that runs continuously.
    
    Features:
    - Checks for drift at configurable intervals
    - Triggers retraining when drift detected
    - Logs all activities
    - Can be enabled/disabled via environment variable
    """
    
    def __init__(
        self,
        check_interval: int = 3600,  # 1 hour default
        drift_threshold: float = 0.2,
        reference_data_path: str = 'data/dataset.csv'
    ):
        """
        Initialize the drift monitoring service.
        
        Args:
            check_interval: Seconds between drift checks (default: 3600 = 1 hour)
            drift_threshold: Drift score threshold for triggering retraining
            reference_data_path: Path to reference dataset
        """
        self.check_interval = check_interval
        self.drift_threshold = drift_threshold
        self.reference_data_path = reference_data_path
        
        # Initialize drift detector
        self.drift_detector = EnhancedDriftDetector(
            reference_data_path=reference_data_path,
            thresholds={
                'psi': drift_threshold,
                'ks': 0.05,
                'kl': 0.1
            }
        )
        
        # Initialize retraining system
        self.retraining_system = AutoRetrainingSystem(
            drift_threshold=drift_threshold
        )
        
        logger.info(f"Drift Monitoring Service initialized")
        logger.info(f"Check interval: {check_interval} seconds ({check_interval/3600:.1f} hours)")
        logger.info(f"Drift threshold: {drift_threshold}")
    
    def check_drift_and_retrain(self) -> dict:
        """
        Check for drift and trigger retraining if needed.
        
        Returns:
            dict: Results of drift check and retraining (if triggered)
        """
        logger.info("=" * 70)
        logger.info("DRIFT CHECK INITIATED")
        logger.info("=" * 70)
        
        try:
            # Load some sample data for drift checking
            # In production, this would be recent production data
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            
            # Take a sample for drift checking
            sample_data = df.sample(n=min(100, len(df)))
            
            # Check for drift
            drift_result = self.drift_detector.check_drift(sample_data)
            
            logger.info(f"Drift Score: {drift_result['drift_score']:.4f}")
            logger.info(f"Drift Detected: {drift_result['drift_detected']}")
            logger.info(f"Drifted Features: {drift_result['drifted_features_count']}/{drift_result['total_features']}")
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'drift_score': drift_result['drift_score'],
                'drift_detected': drift_result['drift_detected'],
                'retraining_triggered': False
            }
            
            # Trigger retraining if drift detected
            if drift_result['drift_detected']:
                logger.warning("DRIFT THRESHOLD EXCEEDED - TRIGGERING RETRAINING")
                
                retraining_result = self.retraining_system.check_and_retrain(
                    current_data=sample_data
                )
                
                result['retraining_triggered'] = retraining_result['retraining_triggered']
                
                if retraining_result['retraining_triggered']:
                    logger.info(f"NEW MODEL VERSION: {retraining_result['new_version']}")
                    logger.info(f"NEW ACCURACY: {retraining_result['new_accuracy']:.4f}")
                    result['new_version'] = retraining_result['new_version']
                    result['new_accuracy'] = retraining_result['new_accuracy']
            else:
                logger.info("No drift detected - continuing monitoring")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during drift check: {str(e)}", exc_info=True)
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def run(self):
        """
        Main monitoring loop - runs continuously until stopped.
        """
        logger.info("=" * 70)
        logger.info("DRIFT MONITORING SERVICE STARTED")
        logger.info("=" * 70)
        logger.info(f"Service will check for drift every {self.check_interval} seconds")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"MONITORING ITERATION #{iteration}")
                logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                # Perform drift check and retraining if needed
                result = self.check_drift_and_retrain()
                
                # Log summary
                logger.info(f"\n{'='*70}")
                logger.info("ITERATION SUMMARY")
                logger.info(f"{'='*70}")
                logger.info(f"Drift Score: {result.get('drift_score', 'N/A')}")
                logger.info(f"Retraining Triggered: {result.get('retraining_triggered', False)}")
                logger.info(f"Next check in: {self.check_interval} seconds")
                logger.info(f"{'='*70}\n")
                
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 70)
            logger.info("DRIFT MONITORING SERVICE STOPPED (Ctrl+C)")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Fatal error in monitoring service: {str(e)}", exc_info=True)
            raise


def main():
    """
    Main entry point for the drift monitoring service.
    """
    # Check if automation is enabled
    if os.getenv("ENABLE_AUTOMATION", "true").lower() != "true":
        logger.info("Automation is disabled (ENABLE_AUTOMATION != true)")
        logger.info("Exiting...")
        sys.exit(0)
    
    # Get configuration from environment variables
    check_interval = int(os.getenv("DRIFT_CHECK_INTERVAL", "3600"))  # 1 hour default
    drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    reference_data_path = os.getenv("REFERENCE_DATA_PATH", "data/dataset.csv")
    
    # Create and run the monitoring service
    service = DriftMonitoringService(
        check_interval=check_interval,
        drift_threshold=drift_threshold,
        reference_data_path=reference_data_path
    )
    
    service.run()


if __name__ == "__main__":
    main()
