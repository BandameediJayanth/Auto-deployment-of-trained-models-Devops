"""
Policy-Based Decision Engine
Auto-Deployment ML Models Project

This module implements the feedback-driven decision engine that maps monitoring
signals and drift indicators to deployment actions, as described in Section 6.6
of the paper.

Formally, the deployment action A_t at time t is defined as:
A_t = f(M_t, D_t, Π)

where:
- M_t represents monitoring metrics
- D_t denotes detected drift signals
- Π is a set of predefined operational policies
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decision_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeploymentAction(Enum):
    """Enumeration of possible deployment actions"""
    CONTINUE = "continue"  # Continue normal operation
    RETRAIN = "retrain"  # Trigger model retraining
    ROLLBACK = "rollback"  # Rollback to previous version
    REDEPLOY = "redeploy"  # Redeploy current version
    CANARY = "canary"  # Deploy using canary strategy
    PAUSE = "pause"  # Pause deployment temporarily


class PolicyEngine:
    """
    Policy-based decision engine that evaluates monitoring signals and drift
    indicators to determine appropriate deployment actions.
    """
    
    def __init__(self, policies_path: str = 'config/deployment_policies.json'):
        self.policies_path = policies_path
        self.policies = self._load_policies()
        self.decision_history = []
        
    def _load_policies(self) -> Dict[str, Any]:
        """Load deployment policies from configuration file"""
        default_policies = {
            "drift_threshold": 0.3,  # 30% of features drifted
            "drift_severity_high": 0.5,  # 50% of features drifted
            "performance_degradation_threshold": 0.1,  # 10% accuracy drop
            "error_rate_threshold": 0.05,  # 5% error rate
            "latency_threshold_ms": 1000,  # 1 second
            "confidence_threshold": 0.7,  # 70% confidence
            "retraining_cooldown_hours": 24,  # Wait 24 hours between retrains
            "rollback_on_severe_drift": True,
            "canary_percentage": 10,  # 10% traffic for canary
            "canary_duration_minutes": 30,  # 30 minutes canary period
            "enable_auto_retraining": True,
            "enable_auto_rollback": True
        }
        
        if os.path.exists(self.policies_path):
            try:
                with open(self.policies_path, 'r') as f:
                    user_policies = json.load(f)
                    default_policies.update(user_policies)
                logger.info(f"Loaded policies from {self.policies_path}")
            except Exception as e:
                logger.warning(f"Could not load policies from {self.policies_path}: {e}")
                logger.info("Using default policies")
        else:
            # Create default policies file
            os.makedirs(os.path.dirname(self.policies_path), exist_ok=True)
            with open(self.policies_path, 'w') as f:
                json.dump(default_policies, f, indent=2)
            logger.info(f"Created default policies file at {self.policies_path}")
        
        return default_policies
    
    def evaluate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate monitoring metrics against policy thresholds.
        
        Args:
            metrics: Dictionary containing monitoring metrics (M_t)
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "metrics_received": metrics,
            "violations": [],
            "severity": "none"
        }
        
        # Check error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.policies['error_rate_threshold']:
            evaluation['violations'].append({
                'metric': 'error_rate',
                'value': error_rate,
                'threshold': self.policies['error_rate_threshold'],
                'severity': 'high' if error_rate > 2 * self.policies['error_rate_threshold'] else 'medium'
            })
        
        # Check latency
        latency_ms = metrics.get('latency_ms', 0)
        if latency_ms > self.policies['latency_threshold_ms']:
            evaluation['violations'].append({
                'metric': 'latency',
                'value': latency_ms,
                'threshold': self.policies['latency_threshold_ms'],
                'severity': 'high' if latency_ms > 2 * self.policies['latency_threshold_ms'] else 'medium'
            })
        
        # Check performance degradation
        accuracy = metrics.get('accuracy', None)
        baseline_accuracy = metrics.get('baseline_accuracy', None)
        if accuracy is not None and baseline_accuracy is not None:
            degradation = baseline_accuracy - accuracy
            if degradation > self.policies['performance_degradation_threshold']:
                evaluation['violations'].append({
                    'metric': 'performance_degradation',
                    'value': degradation,
                    'threshold': self.policies['performance_degradation_threshold'],
                    'severity': 'high' if degradation > 2 * self.policies['performance_degradation_threshold'] else 'medium'
                })
        
        # Check prediction confidence
        avg_confidence = metrics.get('avg_confidence', None)
        if avg_confidence is not None and avg_confidence < self.policies['confidence_threshold']:
            evaluation['violations'].append({
                'metric': 'confidence',
                'value': avg_confidence,
                'threshold': self.policies['confidence_threshold'],
                'severity': 'medium'
            })
        
        # Determine overall severity
        if any(v['severity'] == 'high' for v in evaluation['violations']):
            evaluation['severity'] = 'high'
        elif evaluation['violations']:
            evaluation['severity'] = 'medium'
        
        return evaluation
    
    def evaluate_drift(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate drift detection results against policy thresholds.
        
        Args:
            drift_results: Dictionary containing drift detection results (D_t)
            
        Returns:
            Dictionary with drift evaluation results
        """
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_results.get('drift_detected', False),
            "drift_score": drift_results.get('drifted_feature_count', 0) / max(
                drift_results.get('total_features', 1), 1
            ),
            "severity": "none"
        }
        
        if evaluation['drift_detected']:
            drift_score = evaluation['drift_score']
            
            if drift_score >= self.policies['drift_severity_high']:
                evaluation['severity'] = 'high'
            elif drift_score >= self.policies['drift_threshold']:
                evaluation['severity'] = 'medium'
            else:
                evaluation['severity'] = 'low'
        
        return evaluation
    
    def decide_action(
        self,
        metrics: Dict[str, Any],
        drift_results: Optional[Dict[str, Any]] = None,
        deployment_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main decision function that maps monitoring signals to deployment actions.
        
        Implements: A_t = f(M_t, D_t, Π)
        
        Args:
            metrics: Monitoring metrics (M_t)
            drift_results: Drift detection results (D_t)
            deployment_state: Current deployment state information
            
        Returns:
            Dictionary containing the decided action and reasoning
        """
        decision = {
            "timestamp": datetime.now().isoformat(),
            "action": DeploymentAction.CONTINUE.value,
            "reasoning": [],
            "confidence": 1.0,
            "requires_approval": False
        }
        
        # Evaluate metrics
        metrics_eval = self.evaluate_metrics(metrics)
        
        # Evaluate drift if provided
        drift_eval = None
        if drift_results:
            drift_eval = self.evaluate_drift(drift_results)
        
        # Decision logic based on paper's methodology
        
        # 1. Check for severe drift (high priority)
        if drift_eval and drift_eval['severity'] == 'high':
            if self.policies['rollback_on_severe_drift'] and self.policies['enable_auto_rollback']:
                decision['action'] = DeploymentAction.ROLLBACK.value
                decision['reasoning'].append(
                    f"Severe drift detected (score: {drift_eval['drift_score']:.2f}). "
                    "Rolling back to previous stable version."
                )
                decision['confidence'] = 0.9
                decision['requires_approval'] = False
            else:
                decision['action'] = DeploymentAction.RETRAIN.value
                decision['reasoning'].append(
                    f"Severe drift detected (score: {drift_eval['drift_score']:.2f}). "
                    "Triggering immediate retraining."
                )
                decision['confidence'] = 0.85
        
        # 2. Check for medium drift
        elif drift_eval and drift_eval['severity'] == 'medium':
            if self.policies['enable_auto_retraining']:
                decision['action'] = DeploymentAction.RETRAIN.value
                decision['reasoning'].append(
                    f"Moderate drift detected (score: {drift_eval['drift_score']:.2f}). "
                    "Scheduling retraining."
                )
                decision['confidence'] = 0.75
        
        # 3. Check for high-severity metric violations
        elif metrics_eval['severity'] == 'high':
            # Check if it's a performance issue that might be fixed by retraining
            perf_violations = [v for v in metrics_eval['violations'] 
                             if v['metric'] == 'performance_degradation']
            
            if perf_violations and self.policies['enable_auto_retraining']:
                decision['action'] = DeploymentAction.RETRAIN.value
                decision['reasoning'].append(
                    f"High performance degradation detected. Triggering retraining."
                )
                decision['confidence'] = 0.8
            else:
                # For other high-severity issues, consider rollback
                decision['action'] = DeploymentAction.ROLLBACK.value
                decision['reasoning'].append(
                    f"High-severity metric violations detected: {[v['metric'] for v in metrics_eval['violations']]}"
                )
                decision['confidence'] = 0.85
        
        # 4. Check for medium-severity violations
        elif metrics_eval['severity'] == 'medium':
            # For new deployments, use canary strategy
            if deployment_state and deployment_state.get('is_new_deployment', False):
                decision['action'] = DeploymentAction.CANARY.value
                decision['reasoning'].append(
                    "Medium-severity violations detected. Using canary deployment strategy."
                )
                decision['confidence'] = 0.7
            else:
                decision['action'] = DeploymentAction.RETRAIN.value
                decision['reasoning'].append(
                    "Medium-severity violations detected. Scheduling retraining."
                )
                decision['confidence'] = 0.65
        
        # 5. Low-severity drift - continue but monitor
        elif drift_eval and drift_eval['severity'] == 'low':
            decision['action'] = DeploymentAction.CONTINUE.value
            decision['reasoning'].append(
                f"Low-severity drift detected (score: {drift_eval['drift_score']:.2f}). "
                "Continuing operation with increased monitoring."
            )
            decision['confidence'] = 0.9
        
        # 6. No issues - continue normal operation
        else:
            decision['action'] = DeploymentAction.CONTINUE.value
            decision['reasoning'].append("All metrics within acceptable thresholds.")
            decision['confidence'] = 1.0
        
        # Store decision in history
        decision_record = {
            "decision": decision,
            "metrics_evaluation": metrics_eval,
            "drift_evaluation": drift_eval,
            "policies_used": self.policies
        }
        self.decision_history.append(decision_record)
        
        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        logger.info(f"Decision made: {decision['action']} (confidence: {decision['confidence']:.2f})")
        logger.debug(f"Reasoning: {'; '.join(decision['reasoning'])}")
        
        return decision
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return self.decision_history[-limit:]
    
    def save_decision_history(self, path: str = 'models/decision_history.json'):
        """Save decision history to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
        logger.info(f"Decision history saved to {path}")


if __name__ == "__main__":
    # Test the decision engine
    engine = PolicyEngine()
    
    # Test case 1: Normal operation
    print("\n=== Test Case 1: Normal Operation ===")
    metrics = {
        'error_rate': 0.01,
        'latency_ms': 100,
        'accuracy': 0.95,
        'baseline_accuracy': 0.95,
        'avg_confidence': 0.85
    }
    decision = engine.decide_action(metrics)
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    
    # Test case 2: Drift detected
    print("\n=== Test Case 2: Drift Detected ===")
    drift_results = {
        'drift_detected': True,
        'drifted_feature_count': 8,
        'total_features': 20,
        'drift_score': 0.4
    }
    decision = engine.decide_action(metrics, drift_results)
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    
    # Test case 3: Severe drift
    print("\n=== Test Case 3: Severe Drift ===")
    severe_drift = {
        'drift_detected': True,
        'drifted_feature_count': 12,
        'total_features': 20,
        'drift_score': 0.6
    }
    decision = engine.decide_action(metrics, severe_drift)
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    
    # Test case 4: High error rate
    print("\n=== Test Case 4: High Error Rate ===")
    bad_metrics = {
        'error_rate': 0.15,
        'latency_ms': 2000,
        'accuracy': 0.70,
        'baseline_accuracy': 0.95,
        'avg_confidence': 0.60
    }
    decision = engine.decide_action(bad_metrics)
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
