"""
Retraining Trigger Script
Auto-Deployment ML Models Project

This script handles the automated triggering of model retraining.
It determines the next version number and executes the training pipeline.
"""
import subprocess
import logging
import json
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_next_version(current_version):
    """Calculate the next patch version"""
    try:
        parts = current_version.split('.')
        if len(parts) >= 3:
            major, minor, patch = map(int, parts[:3])
            return f"{major}.{minor}.{patch + 1}"
        else:
            return f"{current_version}.1"
    except:
        return f"{current_version}.1"

def trigger_retraining():
    """Trigger the model training process"""
    logger.info("Triggering automated retraining...")
    
    # Get current version
    current_version = "1.0.0"
    if os.path.exists('models/latest_model.json'):
        try:
            with open('models/latest_model.json', 'r') as f:
                data = json.load(f)
                current_version = data.get('version', "1.0.0")
        except Exception as e:
            logger.warning(f"Could not read latest model version: {e}")
            
    next_version = get_next_version(current_version)
    logger.info(f"Current version: {current_version}, Next version: {next_version}")
    
    # Run training script
    try:
        # We use subprocess to run the training script
        # In a real production system, this would trigger a CI/CD pipeline (e.g., Jenkins job)
        cmd = [sys.executable, 'src/train_model.py', '--version', next_version]
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Retraining completed successfully.")
        logger.info("Training Output:\n" + result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed with exit code {e.returncode}")
        logger.error("Error Output:\n" + e.stderr)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    trigger_retraining()
