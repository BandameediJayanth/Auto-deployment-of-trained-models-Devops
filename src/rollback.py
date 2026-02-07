"""
Model Rollback Script
Auto-Deployment ML Models Project

This script handles the rollback of the model to a previous version
in case of performance degradation or drift.
"""

import os
import json
import logging
import requests
import argparse
import sys

# Add src to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reliability import ReliabilityTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rollback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRollback:
    def __init__(self, history_path='models/model_history.json', latest_path='models/latest_model.json'):
        self.history_path = history_path
        self.latest_path = latest_path
        self.reliability_tracker = ReliabilityTracker()
        
    def get_current_version(self):
        """Get the currently deployed version"""
        try:
            with open(self.latest_path, 'r') as f:
                latest = json.load(f)
            return latest.get('version')
        except Exception as e:
            logger.error(f"Error reading latest model info: {str(e)}")
            return None
            
    def rollback(self, target_version=None):
        """
        Rollback to a previous model version.
        
        Args:
            target_version (str, optional): Specific version to rollback to.
                                            If None, rolls back to the immediate previous version.
        """
        try:
            # Load history
            if not os.path.exists(self.history_path):
                logger.error(f"History file {self.history_path} not found.")
                return False
                
            with open(self.history_path, 'r') as f:
                history = json.load(f)
                
            if not history:
                logger.error("Model history is empty.")
                return False
                
            current_version = self.get_current_version()
            logger.info(f"Current version: {current_version}")
            
            target_info = None
            
            if target_version:
                # Find specific version
                for entry in history:
                    if entry['version'] == target_version:
                        target_info = entry
                        break
                if not target_info:
                    logger.error(f"Target version {target_version} not found in history.")
                    return False
            else:
                # Rollback to previous version (n-1)
                # Assuming history is sorted by date (as implemented in train_model.py)
                history.sort(key=lambda x: x.get('created_at', ''))
                
                # Find index of current version
                current_index = -1
                for i, entry in enumerate(history):
                    if entry['version'] == current_version:
                        current_index = i
                        break
                
                if current_index <= 0:
                    logger.warning("No previous version available to rollback to.")
                    return False
                    
                target_info = history[current_index - 1]
            
            logger.info(f"Rolling back to version: {target_info['version']}")
            
            self.reliability_tracker.log_event("recovery_start", {"from_version": current_version, "to_version": target_info['version']})
            
            # Update latest_model.json
            with open(self.latest_path, 'w') as f:
                json.dump(target_info, f, indent=2)
                
            logger.info(f"Updated {self.latest_path} to point to version {target_info['version']}")
            
            # Trigger API reload
            self.trigger_api_reload()
            
            self.reliability_tracker.log_event("recovery_end", {"status": "success"})
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            self.reliability_tracker.log_event("recovery_end", {"status": "failed", "error": str(e)})
            return False

    def trigger_api_reload(self, api_url="http://localhost:8000/model/reload"):
        """Trigger the API to reload the model"""
        try:
            response = requests.post(api_url)
            if response.status_code == 200:
                logger.info("API model reload triggered successfully.")
            else:
                logger.warning(f"Failed to trigger API reload. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.warning(f"Could not connect to API to trigger reload: {str(e)}")
            logger.info("If API is running, it will pick up the change on next restart.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollback ML Model')
    parser.add_argument('--target', type=str, help='Target version to rollback to')
    args = parser.parse_args()
    
    rollbacker = ModelRollback()
    rollbacker.rollback(args.target)
