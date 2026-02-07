"""
Cleanup and Promote Script
Auto-Deployment ML Models Project

This script handles the final stage of the ingestion pipeline:
1. Checks the verdict.
2. Promotes the model to production if ready.
3. Cleans up temporary reports.
"""

import os
import shutil
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('promotion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_DIR = 'input_models'
PROD_DIR = 'models/production'
REPORT_DIR = 'reports'
VERDICT_FILE = os.path.join(REPORT_DIR, 'final_verdict.md')
INITIAL_REPORT = os.path.join(REPORT_DIR, 'initial_audit.md')
LATEST_MODEL_JSON = 'models/latest_model.json'

def check_verdict():
    """Read the verdict file to see if we can proceed"""
    if not os.path.exists(VERDICT_FILE):
        logger.error("No verdict file found!")
        return False
        
    with open(VERDICT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "✅ READY FOR PRODUCTION" in content:
        return True
    return False

def promote_model():
    """Move model to production folder"""
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.pkl', '.joblib', '.sav'))]
    if not files:
        logger.error("No model found in input directory to promote!")
        return False
        
    model_file = files[0]
    src_path = os.path.join(INPUT_DIR, model_file)
    dst_path = os.path.join(PROD_DIR, f"prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_file}")
    
    # Ensure prod dir exists
    os.makedirs(PROD_DIR, exist_ok=True)
    
    # Move file
    shutil.move(src_path, dst_path)
    logger.info(f"Promoted model to: {dst_path}")
    
    # Update latest_model.json
    model_info = {
        "version": "external_" + datetime.now().strftime('%Y%m%d'),
        "path": dst_path,
        "created_at": datetime.now().isoformat(),
        "metrics": {"source": "external_ingestion"}
    }
    
    with open(LATEST_MODEL_JSON, 'w') as f:
        json.dump(model_info, f, indent=2)
        
    logger.info(f"Updated {LATEST_MODEL_JSON}")
    return True

def cleanup():
    """Delete initial report"""
    if os.path.exists(INITIAL_REPORT):
        os.remove(INITIAL_REPORT)
        logger.info(f"Deleted {INITIAL_REPORT}")
    else:
        logger.warning(f"Could not find {INITIAL_REPORT} to delete")

def main():
    logger.info("Starting Cleanup and Promotion...")
    
    if check_verdict():
        logger.info("Verdict is POSITIVE. Proceeding with promotion.")
        if promote_model():
            cleanup()
            print("✅ Model successfully promoted to production!")
        else:
            print("❌ Promotion failed.")
            exit(1)
    else:
        logger.warning("Verdict is NEGATIVE or MISSING. Aborting promotion.")
        print("🛑 Promotion aborted. Model not ready.")
        exit(1)

if __name__ == "__main__":
    main()
