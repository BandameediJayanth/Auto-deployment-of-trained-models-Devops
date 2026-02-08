"""
Canary Deployment and Controlled Rollout Module
Auto-Deployment ML Models Project

This module implements canary releases and controlled rollout strategies
as mentioned in the paper's policy-based deployment control (Section 6.6).

Enhanced with model selection and comprehensive testing capabilities.
"""

import json
import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import glob

# Import model tester
try:
    from src.model_tester import ModelTester
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model_tester import ModelTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('canary_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CanaryDeployment:
    """
    Manages canary deployments and controlled rollouts of ML models.
    
    A canary deployment gradually routes traffic to a new model version,
    allowing for monitoring and validation before full rollout.
    
    Enhanced with model selection and comprehensive testing capabilities.
    """
    
    def __init__(self, config_path: str = 'config/canary_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.deployment_state_file = 'models/canary_state.json'
        self.deployment_state = self._load_state()
        self.model_tester = ModelTester()
        self.selected_model = None
        self.test_results = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load canary deployment configuration"""
        default_config = {
            "initial_percentage": 10,  # Start with 10% traffic
            "increment_percentage": 10,  # Increase by 10% each step
            "evaluation_duration_minutes": 30,  # Evaluate for 30 minutes
            "success_thresholds": {
                "max_error_rate": 0.05,
                "max_latency_ms": 1000,
                "min_accuracy": 0.80
            },
            "rollback_on_failure": True,
            "auto_promote": True  # Automatically promote if metrics are good
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                logger.info(f"Loaded canary config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load canary config: {e}")
        else:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default canary config at {self.config_path}")
        
        return default_config
    
    def _load_state(self) -> Dict[str, Any]:
        """Load current deployment state"""
        if os.path.exists(self.deployment_state_file):
            try:
                with open(self.deployment_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load deployment state: {e}")
        
        return {
            "active_canary": None,
            "canary_history": []
        }
    
    def _save_state(self):
        """Save deployment state"""
        os.makedirs(os.path.dirname(self.deployment_state_file), exist_ok=True)
        with open(self.deployment_state_file, 'w') as f:
            json.dump(self.deployment_state, f, indent=2)
    
    def start_canary(
        self,
        model_version: str,
        model_path: str,
        metadata_path: str
    ) -> Dict[str, Any]:
        """
        Start a canary deployment for a new model version.
        
        Args:
            model_version: Version identifier for the new model
            model_path: Path to the model file
            metadata_path: Path to the model metadata
            
        Returns:
            Dictionary with canary deployment information
        """
        logger.info(f"Starting canary deployment for version {model_version}")
        
        # Check if there's already an active canary
        if self.deployment_state.get('active_canary'):
            logger.warning("There is already an active canary deployment")
            return {
                "success": False,
                "error": "Active canary deployment already exists"
            }
        
        canary_info = {
            "version": model_version,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "start_time": datetime.now().isoformat(),
            "current_percentage": self.config["initial_percentage"],
            "status": "active",
            "metrics": {
                "request_count": 0,
                "error_count": 0,
                "total_latency_ms": 0,
                "predictions": []
            }
        }
        
        self.deployment_state["active_canary"] = canary_info
        self._save_state()
        
        logger.info(f"Canary deployment started: {model_version} at {self.config['initial_percentage']}% traffic")
        
        return {
            "success": True,
            "canary_info": canary_info
        }
    
    def record_canary_metrics(
        self,
        success: bool,
        latency_ms: float,
        prediction: Optional[Any] = None
    ):
        """Record metrics for the active canary deployment"""
        if not self.deployment_state.get('active_canary'):
            return
        
        canary = self.deployment_state['active_canary']
        metrics = canary['metrics']
        
        metrics['request_count'] += 1
        metrics['total_latency_ms'] += latency_ms
        
        if not success:
            metrics['error_count'] += 1
        
        if prediction is not None:
            metrics['predictions'].append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction
            })
            # Keep only last 1000 predictions
            if len(metrics['predictions']) > 1000:
                metrics['predictions'] = metrics['predictions'][-1000:]
        
        self._save_state()
    
    def evaluate_canary(self) -> Dict[str, Any]:
        """
        Evaluate the active canary deployment against success thresholds.
        
        Returns:
            Dictionary with evaluation results and recommended action
        """
        if not self.deployment_state.get('active_canary'):
            return {
                "success": False,
                "error": "No active canary deployment"
            }
        
        canary = self.deployment_state['active_canary']
        metrics = canary['metrics']
        
        if metrics['request_count'] == 0:
            return {
                "success": False,
                "error": "No metrics collected yet"
            }
        
        # Calculate metrics
        error_rate = metrics['error_count'] / metrics['request_count']
        avg_latency = metrics['total_latency_ms'] / metrics['request_count']
        
        # Evaluate against thresholds
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "version": canary['version'],
            "metrics": {
                "request_count": metrics['request_count'],
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency
            },
            "thresholds": self.config['success_thresholds'],
            "passed": True,
            "violations": []
        }
        
        # Check error rate
        if error_rate > self.config['success_thresholds']['max_error_rate']:
            evaluation['passed'] = False
            evaluation['violations'].append({
                'metric': 'error_rate',
                'value': error_rate,
                'threshold': self.config['success_thresholds']['max_error_rate']
            })
        
        # Check latency
        if avg_latency > self.config['success_thresholds']['max_latency_ms']:
            evaluation['passed'] = False
            evaluation['violations'].append({
                'metric': 'latency',
                'value': avg_latency,
                'threshold': self.config['success_thresholds']['max_latency_ms']
            })
        
        # Determine action
        if evaluation['passed']:
            if canary['current_percentage'] < 100:
                evaluation['action'] = 'increment'
                evaluation['recommendation'] = f"Increase traffic to {canary['current_percentage'] + self.config['increment_percentage']}%"
            else:
                evaluation['action'] = 'promote'
                evaluation['recommendation'] = "Promote to full production"
        else:
            evaluation['action'] = 'rollback'
            evaluation['recommendation'] = "Rollback due to threshold violations"
        
        return evaluation
    
    def increment_canary_traffic(self) -> Dict[str, Any]:
        """Increment the traffic percentage for the active canary"""
        if not self.deployment_state.get('active_canary'):
            return {
                "success": False,
                "error": "No active canary deployment"
            }
        
        canary = self.deployment_state['active_canary']
        new_percentage = min(
            canary['current_percentage'] + self.config['increment_percentage'],
            100
        )
        
        canary['current_percentage'] = new_percentage
        canary['last_increment'] = datetime.now().isoformat()
        self._save_state()
        
        logger.info(f"Incremented canary traffic to {new_percentage}%")
        
        return {
            "success": True,
            "new_percentage": new_percentage
        }
    
    def promote_canary(self) -> Dict[str, Any]:
        """
        Promote the canary deployment to full production.
        
        This completes the canary deployment and makes it the primary model.
        """
        if not self.deployment_state.get('active_canary'):
            return {
                "success": False,
                "error": "No active canary deployment"
            }
        
        canary = self.deployment_state['active_canary']
        
        # Move to history
        canary['status'] = 'promoted'
        canary['promoted_at'] = datetime.now().isoformat()
        self.deployment_state['canary_history'].append(canary)
        
        # Clear active canary
        self.deployment_state['active_canary'] = None
        self._save_state()
        
        logger.info(f"Canary deployment {canary['version']} promoted to production")
        
        return {
            "success": True,
            "version": canary['version']
        }
    
    def rollback_canary(self) -> Dict[str, Any]:
        """
        Rollback the canary deployment.
        
        This aborts the canary and returns to the previous model version.
        """
        if not self.deployment_state.get('active_canary'):
            return {
                "success": False,
                "error": "No active canary deployment"
            }
        
        canary = self.deployment_state['active_canary']
        
        # Move to history as failed
        canary['status'] = 'rolled_back'
        canary['rolled_back_at'] = datetime.now().isoformat()
        self.deployment_state['canary_history'].append(canary)
        
        # Clear active canary
        self.deployment_state['active_canary'] = None
        self._save_state()
        
        logger.info(f"Canary deployment {canary['version']} rolled back")
        
        return {
            "success": True,
            "version": canary['version']
        }
    
    def get_active_canary(self) -> Optional[Dict[str, Any]]:
        """Get information about the active canary deployment"""
        return self.deployment_state.get('active_canary')
    
    def should_route_to_canary(self) -> bool:
        """
        Determine if a request should be routed to the canary deployment.
        
        This implements the traffic splitting logic.
        """
        canary = self.deployment_state.get('active_canary')
        if not canary or canary['status'] != 'active':
            return False
        
        # Simple percentage-based routing (in production, use consistent hashing)
        import random
        return random.random() * 100 < canary['current_percentage']


class ControlledRollout:
    """
    Manages controlled rollouts with multiple stages and validation gates.
    """
    
    def __init__(self, stages: List[Dict[str, Any]] = None):
        self.stages = stages or [
            {"name": "development", "percentage": 0, "duration_minutes": 0},
            {"name": "staging", "percentage": 0, "duration_minutes": 60},
            {"name": "canary", "percentage": 10, "duration_minutes": 30},
            {"name": "production_25", "percentage": 25, "duration_minutes": 60},
            {"name": "production_50", "percentage": 50, "duration_minutes": 60},
            {"name": "production_100", "percentage": 100, "duration_minutes": 0}
        ]
        self.current_stage_index = 0
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get the current rollout stage"""
        if self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return self.stages[-1]  # Return last stage if completed
    
    def advance_stage(self) -> bool:
        """Advance to the next rollout stage"""
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            return True
        return False
    
    def get_traffic_percentage(self) -> int:
        """Get the traffic percentage for the current stage"""
        stage = self.get_current_stage()
        return stage.get('percentage', 0)


    def list_available_models(self, models_dir: str = 'models') -> List[Dict[str, Any]]:
        """
        List all available model files in the models directory.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory {models_dir} does not exist")
            return models
        
        # Find all .pkl files
        model_files = glob.glob(os.path.join(models_dir, '**/*.pkl'), recursive=True)
        
        for model_path in model_files:
            # Skip if in __pycache__ or hidden directories
            if '__pycache__' in model_path or os.path.basename(model_path).startswith('.'):
                continue
            
            model_info = {
                'path': model_path,
                'name': os.path.basename(model_path),
                'directory': os.path.dirname(model_path),
                'size_bytes': os.path.getsize(model_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            }
            
            # Try to find metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if not os.path.exists(metadata_path):
                base_name = os.path.splitext(model_path)[0]
                metadata_path = f"{base_name}_metadata.json"
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_info['metadata'] = metadata
                        model_info['version'] = metadata.get('version', 'unknown')
                        model_info['model_name'] = metadata.get('model_name', 'unknown')
                except Exception as e:
                    logger.warning(f"Could not load metadata for {model_path}: {e}")
            
            models.append(model_info)
        
        return models
    
    def select_model_interactive(self) -> Optional[Dict[str, Any]]:
        """
        Interactive model selection from available models.
        
        Returns:
            Selected model information or None
        """
        models = self.list_available_models()
        
        if not models:
            logger.error("No models found in models directory!")
            logger.info("Please place your .pkl model files in the 'models' folder")
            return None
        
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS FOR DEPLOYMENT")
        print("=" * 80)
        print(f"{'#':<5} {'Model Name':<30} {'Version':<15} {'Size (MB)':<12} {'Modified':<20}")
        print("-" * 80)
        
        for idx, model in enumerate(models, 1):
            size_mb = model['size_bytes'] / (1024 * 1024)
            version = model.get('version', 'N/A')
            name = model.get('model_name', model['name'])
            modified = model['modified'][:19]  # Truncate to date+time
            
            print(f"{idx:<5} {name[:28]:<30} {version:<15} {size_mb:<12.2f} {modified:<20}")
        
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected = models[choice_num - 1]
                    print(f"\n✅ Selected: {selected.get('model_name', selected['name'])} (v{selected.get('version', 'unknown')})")
                    self.selected_model = selected
                    return selected
                else:
                    print(f"❌ Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n\nCancelled by user")
                return None
    
    def test_selected_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive tests on the selected model.
        
        Args:
            model_path: Optional path to model (uses selected_model if not provided)
        
        Returns:
            Test results dictionary
        """
        if model_path is None:
            if self.selected_model is None:
                logger.error("No model selected. Please select a model first.")
                return None
            model_path = self.selected_model['path']
        
        logger.info(f"Running comprehensive tests on: {model_path}")
        
        # Run all tests
        self.test_results = self.model_tester.run_all_tests(model_path)
        
        return self.test_results
    
    def deploy_with_testing(
        self,
        skip_tests: bool = False,
        auto_deploy: bool = False
    ) -> Dict[str, Any]:
        """
        Complete deployment workflow with model selection and testing.
        
        Args:
            skip_tests: If True, skip testing and deploy directly
            auto_deploy: If True, automatically deploy if tests pass
        
        Returns:
            Deployment result dictionary
        """
        result = {
            "success": False,
            "steps": [],
            "model": None,
            "test_results": None,
            "canary_info": None
        }
        
        # Step 1: Select model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: MODEL SELECTION")
        logger.info("=" * 80)
        
        selected = self.select_model_interactive()
        if not selected:
            result["error"] = "No model selected"
            return result
        
        result["model"] = selected
        result["steps"].append("Model selected")
        
        # Step 2: Run tests (unless skipped)
        if not skip_tests:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: COMPREHENSIVE MODEL TESTING")
            logger.info("=" * 80)
            
            test_results = self.test_selected_model(selected['path'])
            result["test_results"] = test_results
            result["steps"].append("Tests completed")
            
            if not test_results or not test_results.get("overall_passed", False):
                logger.error("\n❌ MODEL FAILED TESTS - NOT READY FOR DEPLOYMENT")
                logger.error("Please fix the issues before deploying.")
                
                if not auto_deploy:
                    deploy_anyway = input("\nDeploy anyway? (yes/no): ").strip().lower()
                    if deploy_anyway != 'yes':
                        result["error"] = "Deployment cancelled - model failed tests"
                        return result
            else:
                logger.info("\n✅ MODEL PASSED ALL TESTS - READY FOR DEPLOYMENT")
        else:
            logger.info("Skipping tests (--skip-tests flag)")
            result["steps"].append("Tests skipped")
        
        # Step 3: Start canary deployment
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: CANARY DEPLOYMENT")
        logger.info("=" * 80)
        
        # Extract version and metadata path
        version = selected.get('version', '1.0.0')
        model_path = selected['path']
        
        # Find metadata path
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if not os.path.exists(metadata_path):
            base_name = os.path.splitext(model_path)[0]
            metadata_path = f"{base_name}_metadata.json"
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found at {metadata_path}, creating minimal metadata...")
            metadata_path = None
        
        canary_result = self.start_canary(
            model_version=version,
            model_path=model_path,
            metadata_path=metadata_path or ""
        )
        
        if canary_result.get('success'):
            result["success"] = True
            result["canary_info"] = canary_result.get('canary_info')
            result["steps"].append("Canary deployment started")
            logger.info("\n✅ CANARY DEPLOYMENT STARTED SUCCESSFULLY")
            logger.info(f"   Model: {selected.get('model_name', selected['name'])}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Initial Traffic: {self.config['initial_percentage']}%")
        else:
            result["error"] = canary_result.get('error', 'Unknown error')
            logger.error(f"\n❌ CANARY DEPLOYMENT FAILED: {result['error']}")
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Canary Deployment with Model Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive deployment with testing
  python src/canary_deployment.py
  
  # Skip tests and deploy directly
  python src/canary_deployment.py --skip-tests
  
  # Auto-deploy if tests pass
  python src/canary_deployment.py --auto-deploy
        """
    )
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip model testing and deploy directly')
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Automatically deploy if tests pass')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    canary = CanaryDeployment()
    
    if args.list_models:
        models = canary.list_available_models()
        if models:
            print(f"\nFound {len(models)} model(s):")
            for model in models:
                print(f"  - {model['name']} (v{model.get('version', 'unknown')})")
        else:
            print("\nNo models found in models directory")
        sys.exit(0)
    
    # Run deployment workflow
    result = canary.deploy_with_testing(
        skip_tests=args.skip_tests,
        auto_deploy=args.auto_deploy
    )
    
    if result.get("success"):
        print("\n" + "=" * 80)
        print("DEPLOYMENT WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Monitor canary metrics: python src/canary_deployment.py --evaluate")
        print("2. Check API health: curl http://localhost:8000/health")
        print("3. View monitoring: http://localhost:3000 (Grafana)")
        sys.exit(0)
    else:
        print(f"\n❌ Deployment failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
