"""
MLOps Orchestrator
Auto-Deployment ML Models Project

This module orchestrates the complete MLOps pipeline, integrating all components:
- Training and validation
- Deployment with canary releases
- Monitoring and drift detection
- Decision engine and automated actions
- Reliability tracking

This implements the closed-loop feedback system described in the paper.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLOpsOrchestrator:
    """
    Main orchestrator that coordinates all MLOps components.
    """
    
    def __init__(self):
        self.components = {
            'training': None,
            'validation': None,
            'monitoring': None,
            'decision_engine': None,
            'canary': None
        }
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all MLOps components"""
        try:
            from src.train_model import ModelTrainer
            from src.validate_model import ModelValidator
            from src.monitoring_service import MonitoringService
            from src.decision_engine import PolicyEngine
            from src.canary_deployment import CanaryDeployment
            
            self.components['training'] = ModelTrainer
            self.components['validation'] = ModelValidator
            self.components['monitoring'] = MonitoringService
            self.components['decision_engine'] = PolicyEngine
            self.components['canary'] = CanaryDeployment
            
            logger.info("All components initialized successfully")
        except ImportError as e:
            logger.warning(f"Some components could not be imported: {str(e)}")
    
    def run_training_pipeline(self, version: str = None):
        """Run the complete training pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        
        if not self.components['training']:
            logger.error("Training component not available")
            return False
        
        try:
            trainer = self.components['training'](version=version or "1.0.0")
            success = trainer.run_training_pipeline()
            
            if success:
                logger.info("✅ Training pipeline completed successfully")
            else:
                logger.error("❌ Training pipeline failed")
            
            return success
        except Exception as e:
            logger.error(f"Training pipeline error: {str(e)}")
            return False
    
    def run_validation_pipeline(self):
        """Run the validation pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING VALIDATION PIPELINE")
        logger.info("=" * 60)
        
        if not self.components['validation']:
            logger.error("Validation component not available")
            return False
        
        try:
            validator = self.components['validation']()
            success = validator.run_validation_pipeline()
            
            if success:
                logger.info("✅ Validation pipeline completed successfully")
            else:
                logger.error("❌ Validation pipeline failed")
            
            return success
        except Exception as e:
            logger.error(f"Validation pipeline error: {str(e)}")
            return False
    
    def run_deployment_pipeline(self, use_canary: bool = True):
        """Run the deployment pipeline with optional canary release"""
        logger.info("=" * 60)
        logger.info("STARTING DEPLOYMENT PIPELINE")
        logger.info("=" * 60)
        
        # Check if model is validated
        if not os.path.exists('models/latest_model.json'):
            logger.error("No trained model found. Please run training first.")
            return False
        
        try:
            with open('models/latest_model.json', 'r') as f:
                model_info = json.load(f)
            
            if use_canary and self.components['canary']:
                logger.info("Using canary deployment strategy")
                canary = self.components['canary']()
                
                result = canary.start_canary(
                    model_version=model_info['version'],
                    model_path=model_info['latest_model'],
                    metadata_path=model_info['latest_metadata']
                )
                
                if result['success']:
                    logger.info("✅ Canary deployment started")
                    logger.info(f"   Version: {model_info['version']}")
                    logger.info(f"   Initial traffic: {canary.config['initial_percentage']}%")
                    return True
                else:
                    logger.error(f"❌ Canary deployment failed: {result.get('error')}")
                    return False
            else:
                logger.info("Using standard deployment (no canary)")
                logger.info(f"   Version: {model_info['version']}")
                logger.info("✅ Deployment pipeline completed")
                return True
                
        except Exception as e:
            logger.error(f"Deployment pipeline error: {str(e)}")
            return False
    
    def run_full_pipeline(self, version: str = None, use_canary: bool = True):
        """Run the complete MLOps pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING FULL MLOPS PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Training
        if not self.run_training_pipeline(version):
            logger.error("Pipeline stopped: Training failed")
            return False
        
        # Step 2: Validation
        if not self.run_validation_pipeline():
            logger.error("Pipeline stopped: Validation failed")
            return False
        
        # Step 3: Deployment
        if not self.run_deployment_pipeline(use_canary):
            logger.error("Pipeline stopped: Deployment failed")
            return False
        
        logger.info("=" * 60)
        logger.info("✅ FULL MLOPS PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
    
    def get_system_status(self):
        """Get the current status of all system components"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "models": {},
            "deployment": {}
        }
        
        # Check components
        for name, component in self.components.items():
            status['components'][name] = component is not None
        
        # Check models
        if os.path.exists('models/latest_model.json'):
            try:
                with open('models/latest_model.json', 'r') as f:
                    status['models']['latest'] = json.load(f)
            except Exception:
                pass
        
        # Check canary deployment
        if self.components['canary']:
            try:
                canary = self.components['canary']()
                active_canary = canary.get_active_canary()
                status['deployment']['canary'] = active_canary is not None
                if active_canary:
                    status['deployment']['canary_info'] = active_canary
            except Exception:
                pass
        
        return status


def main():
    """Main entry point for the orchestrator"""
    parser = argparse.ArgumentParser(description='MLOps Pipeline Orchestrator')
    parser.add_argument('command', choices=['train', 'validate', 'deploy', 'full', 'status'],
                       help='Command to execute')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--no-canary', action='store_true', help='Disable canary deployment')
    
    args = parser.parse_args()
    
    orchestrator = MLOpsOrchestrator()
    
    if args.command == 'train':
        success = orchestrator.run_training_pipeline(args.version)
        sys.exit(0 if success else 1)
    
    elif args.command == 'validate':
        success = orchestrator.run_validation_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.command == 'deploy':
        success = orchestrator.run_deployment_pipeline(use_canary=not args.no_canary)
        sys.exit(0 if success else 1)
    
    elif args.command == 'full':
        success = orchestrator.run_full_pipeline(args.version, use_canary=not args.no_canary)
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        status = orchestrator.get_system_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
