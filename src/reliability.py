"""
Reliability Modeling Module
Auto-Deployment ML Models Project

This module tracks system reliability metrics such as MTTR, failure rate, and availability.
"""

import json
import os
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EVENTS_FILE = 'models/reliability_events.json'

class ReliabilityTracker:
    def __init__(self):
        self.events_file = EVENTS_FILE
        self._ensure_file_exists()
        
    def _ensure_file_exists(self):
        if not os.path.exists(os.path.dirname(self.events_file)):
            os.makedirs(os.path.dirname(self.events_file), exist_ok=True)
            
        if not os.path.exists(self.events_file):
            with open(self.events_file, 'w') as f:
                json.dump([], f)

    def log_event(self, event_type, details=None):
        """
        Log a reliability event.
        
        Args:
            event_type (str): 'deployment', 'failure', 'recovery_start', 'recovery_end'
            details (dict): Additional details
        """
        event = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'type': event_type,
            'details': details or {}
        }
        
        try:
            with open(self.events_file, 'r') as f:
                events = json.load(f)
            
            events.append(event)
            
            with open(self.events_file, 'w') as f:
                json.dump(events, f, indent=2)
                
            logger.info(f"Logged event: {event_type}")
        except Exception as e:
            logger.error(f"Failed to log event: {str(e)}")

    def calculate_deployment_reliability(self):
        """
        Calculate deployment reliability using formal modeling from paper.
        
        From Section 6.8:
        P_success = 1 - (P_test + P_deploy + P_runtime)
        
        where:
        - P_test: failure probability during validation
        - P_deploy: failure probability during deployment
        - P_runtime: failure probability during runtime execution
        """
        try:
            with open(self.events_file, 'r') as f:
                events = json.load(f)
            
            if not events:
                return {
                    "p_success": 1.0,
                    "p_test": 0.0,
                    "p_deploy": 0.0,
                    "p_runtime": 0.0,
                    "deployment_count": 0,
                    "test_failures": 0,
                    "deploy_failures": 0,
                    "runtime_failures": 0
                }
            
            # Count events by type
            deployments = [e for e in events if e['type'] == 'deployment']
            test_failures = [e for e in events if 'test' in e.get('details', {}).get('stage', '').lower()]
            deploy_failures = [e for e in events if 'deploy' in e.get('details', {}).get('stage', '').lower()]
            runtime_failures = [e for e in events if e['type'] == 'failure' and 
                               'runtime' in e.get('details', {}).get('stage', '').lower()]
            
            deployment_count = len(deployments)
            
            # Calculate failure probabilities
            p_test = len(test_failures) / max(deployment_count, 1)
            p_deploy = len(deploy_failures) / max(deployment_count, 1)
            p_runtime = len(runtime_failures) / max(deployment_count, 1)
            
            # Calculate overall success probability
            p_success = 1 - (p_test + p_deploy + p_runtime)
            # Ensure it's between 0 and 1
            p_success = max(0.0, min(1.0, p_success))
            
            return {
                "p_success": p_success,
                "p_test": p_test,
                "p_deploy": p_deploy,
                "p_runtime": p_runtime,
                "deployment_count": deployment_count,
                "test_failures": len(test_failures),
                "deploy_failures": len(deploy_failures),
                "runtime_failures": len(runtime_failures)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate deployment reliability: {str(e)}")
            return {}
    
    def calculate_metrics(self):
        """
        Calculate reliability metrics including MTTR and failure rates.
        
        From Section 6.8:
        MTTR = (1/N) * Σ t_recovery^(i)
        """
        try:
            with open(self.events_file, 'r') as f:
                events = json.load(f)
            
            if not events:
                return {}
                
            failures = [e for e in events if e['type'] == 'failure']
            recoveries = [e for e in events if e['type'] == 'recovery_end']
            
            # Calculate MTTR (Mean Time to Recovery)
            # We assume every recovery corresponds to the last failure
            recovery_times = []
            
            sorted_events = sorted(events, key=lambda x: x['timestamp'])
            last_failure_time = None
            
            for event in sorted_events:
                if event['type'] == 'failure':
                    last_failure_time = event['timestamp']
                elif event['type'] == 'recovery_end' and last_failure_time:
                    recovery_time = event['timestamp'] - last_failure_time
                    recovery_times.append(recovery_time)
                    last_failure_time = None # Reset
            
            # MTTR calculation: MTTR = (1/N) * Σ t_recovery^(i)
            mttr = sum(recovery_times) / len(recovery_times) if recovery_times else 0
            
            # Calculate Failure Rate (failures per hour)
            total_time_seconds = 0
            if len(sorted_events) > 0:
                total_time_seconds = time.time() - sorted_events[0]['timestamp']
                
            total_hours = total_time_seconds / 3600
            failure_rate = len(failures) / total_hours if total_hours > 0 else 0
            
            # Calculate deployment reliability
            reliability_metrics = self.calculate_deployment_reliability()
            
            # Combine all metrics
            metrics = {
                "mttr_seconds": mttr,
                "mttr_minutes": mttr / 60,
                "mttr_hours": mttr / 3600,
                "failure_count": len(failures),
                "recovery_count": len(recoveries),
                "failure_rate_per_hour": failure_rate,
                "total_tracked_time_seconds": total_time_seconds,
                "total_tracked_time_hours": total_time_seconds / 3600,
                "deployment_reliability": reliability_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            return {}

if __name__ == "__main__":
    tracker = ReliabilityTracker()
    # Test logging
    tracker.log_event("deployment", {"version": "1.0.0"})
    time.sleep(1)
    tracker.log_event("failure", {"error": "Drift detected"})
    time.sleep(2)
    tracker.log_event("recovery_end", {"action": "rollback"})
    
    print(json.dumps(tracker.calculate_metrics(), indent=2))
