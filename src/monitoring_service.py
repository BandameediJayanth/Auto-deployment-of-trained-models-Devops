"""
Comprehensive Monitoring Service
Auto-Deployment ML Models Project

This module implements continuous monitoring and signal collection as described
in Section 6.4 of the paper. It monitors both infrastructure-level and
model-level metrics.

Let M_t denote the set of monitored metrics at time t:
M_t = {m_1(t), m_2(t), ..., m_n(t)}
"""

import json
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Comprehensive monitoring service that collects and aggregates metrics
    from various sources (API, model performance, infrastructure).
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        metrics_file: str = "models/monitoring_metrics.json",
        aggregation_window_seconds: int = 60
    ):
        self.api_url = api_url
        self.metrics_file = metrics_file
        self.aggregation_window = aggregation_window_seconds
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.aggregated_metrics = {}
        
        # Threading
        self.running = False
        self.monitor_thread = None
        
        # Initialize metrics structure
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize the metrics structure"""
        self.aggregated_metrics = {
            "infrastructure": {
                "request_count": 0,
                "error_count": 0,
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "error_rate": 0,
                "requests_per_second": 0
            },
            "model": {
                "prediction_count": 0,
                "avg_confidence": 0,
                "confidence_sum": 0,
                "accuracy": None,  # Will be updated from validation
                "baseline_accuracy": None
            },
            "system": {
                "uptime_seconds": 0,
                "last_update": None,
                "monitoring_active": False
            }
        }
    
    def start_monitoring(self):
        """Start the monitoring service in a background thread"""
        if self.running:
            logger.warning("Monitoring service is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring service started")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background"""
        while self.running:
            try:
                # Collect metrics from API
                self._collect_api_metrics()
                
                # Aggregate metrics
                self._aggregate_metrics()
                
                # Save metrics
                self._save_metrics()
                
                # Sleep for aggregation window
                time.sleep(self.aggregation_window)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_api_metrics(self):
        """Collect metrics from the API endpoint"""
        try:
            # Get Prometheus metrics
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                self._parse_prometheus_metrics(metrics_text)
            
            # Get health check
            health_response = requests.get(f"{self.api_url}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                self._update_health_metrics(health_data)
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not collect API metrics: {str(e)}")
    
    def _parse_prometheus_metrics(self, metrics_text: str):
        """Parse Prometheus metrics format"""
        lines = metrics_text.split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            # Parse metric lines (simplified parser)
            if 'model_api_request_duration_seconds' in line and 'sum' not in line:
                # Extract latency
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        latency_seconds = float(parts[-1])
                        latency_ms = latency_seconds * 1000
                        self._record_metric('latency_ms', latency_ms)
                except (ValueError, IndexError):
                    pass
            
            elif 'model_api_requests_total' in line:
                # Extract request count
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        count = float(parts[-1])
                        self._record_metric('request_count', count)
                except (ValueError, IndexError):
                    pass
            
            elif 'model_api_errors_total' in line:
                # Extract error count
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        errors = float(parts[-1])
                        self._record_metric('error_count', errors)
                except (ValueError, IndexError):
                    pass
            
            elif 'model_prediction_confidence' in line and 'sum' not in line:
                # Extract confidence
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        confidence = float(parts[-1])
                        self._record_metric('confidence', confidence)
                except (ValueError, IndexError):
                    pass
    
    def _update_health_metrics(self, health_data: Dict[str, Any]):
        """Update metrics from health check endpoint"""
        if 'uptime_seconds' in health_data:
            self.aggregated_metrics['system']['uptime_seconds'] = health_data['uptime_seconds']
    
    def _record_metric(self, metric_name: str, value: float):
        """Record a metric value with timestamp"""
        metric_record = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value
        }
        self.metrics_buffer.append(metric_record)
    
    def record_prediction(
        self,
        success: bool,
        latency_ms: float,
        confidence: Optional[float] = None
    ):
        """Record a prediction event"""
        self._record_metric('prediction', 1 if success else 0)
        self._record_metric('latency_ms', latency_ms)
        
        if confidence is not None:
            self._record_metric('confidence', confidence)
    
    def _aggregate_metrics(self):
        """Aggregate metrics from the buffer"""
        if not self.metrics_buffer:
            return
        
        # Get metrics from the last aggregation window
        cutoff_time = time.time() - self.aggregation_window
        
        recent_metrics = [
            m for m in self.metrics_buffer
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return
        
        # Aggregate infrastructure metrics
        latencies = [m['value'] for m in recent_metrics if m['metric'] == 'latency_ms']
        requests = [m for m in recent_metrics if m['metric'] == 'request_count']
        errors = [m for m in recent_metrics if m['metric'] == 'error_count']
        confidences = [m['value'] for m in recent_metrics if m['metric'] == 'confidence']
        
        if latencies:
            self.aggregated_metrics['infrastructure']['total_latency_ms'] = sum(latencies)
            self.aggregated_metrics['infrastructure']['avg_latency_ms'] = sum(latencies) / len(latencies)
        
        if requests:
            request_count = len(requests)
            self.aggregated_metrics['infrastructure']['request_count'] = request_count
            self.aggregated_metrics['infrastructure']['requests_per_second'] = request_count / self.aggregation_window
        
        if errors:
            error_count = len(errors)
            request_count = max(self.aggregated_metrics['infrastructure']['request_count'], 1)
            self.aggregated_metrics['infrastructure']['error_count'] = error_count
            self.aggregated_metrics['infrastructure']['error_rate'] = error_count / request_count
        
        if confidences:
            self.aggregated_metrics['model']['avg_confidence'] = sum(confidences) / len(confidences)
            self.aggregated_metrics['model']['confidence_sum'] = sum(confidences)
            self.aggregated_metrics['model']['prediction_count'] = len(confidences)
        
        # Update timestamp
        self.aggregated_metrics['system']['last_update'] = datetime.now().isoformat()
        self.aggregated_metrics['system']['monitoring_active'] = True
    
    def _save_metrics(self):
        """Save aggregated metrics to file"""
        try:
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            
            # Load existing metrics history
            metrics_history = []
            if os.path.exists(self.metrics_file):
                try:
                    with open(self.metrics_file, 'r') as f:
                        metrics_history = json.load(f)
                except Exception:
                    metrics_history = []
            
            # Add current aggregated metrics
            metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": self.aggregated_metrics.copy()
            })
            
            # Keep only last 1000 entries
            if len(metrics_history) > 1000:
                metrics_history = metrics_history[-1000:]
            
            # Save
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current aggregated metrics (M_t).
        
        Returns the set of monitored metrics at time t:
        M_t = {m_1(t), m_2(t), ..., m_n(t)}
        """
        return self.aggregated_metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics formatted for the decision engine.
        
        Returns metrics in the format expected by the policy engine.
        """
        infra = self.aggregated_metrics.get('infrastructure', {})
        model = self.aggregated_metrics.get('model', {})
        
        return {
            "error_rate": infra.get('error_rate', 0),
            "latency_ms": infra.get('avg_latency_ms', 0),
            "accuracy": model.get('accuracy'),
            "baseline_accuracy": model.get('baseline_accuracy'),
            "avg_confidence": model.get('avg_confidence'),
            "request_count": infra.get('request_count', 0),
            "requests_per_second": infra.get('requests_per_second', 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def set_model_accuracy(self, accuracy: float, baseline_accuracy: Optional[float] = None):
        """Set the current model accuracy (from validation)"""
        self.aggregated_metrics['model']['accuracy'] = accuracy
        if baseline_accuracy is not None:
            self.aggregated_metrics['model']['baseline_accuracy'] = baseline_accuracy


# Global monitoring service instance
_monitoring_service = None


def get_monitoring_service() -> MonitoringService:
    """Get or create the global monitoring service instance"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


if __name__ == "__main__":
    # Test the monitoring service
    print("\n=== Testing Monitoring Service ===")
    
    service = MonitoringService()
    
    # Record some test metrics
    for i in range(10):
        service.record_prediction(
            success=i % 10 != 0,
            latency_ms=50 + i * 10,
            confidence=0.7 + (i % 5) * 0.05
        )
        time.sleep(0.1)
    
    # Aggregate
    service._aggregate_metrics()
    
    # Get metrics
    metrics = service.get_current_metrics()
    print(f"\nCurrent metrics: {json.dumps(metrics, indent=2)}")
    
    # Get summary
    summary = service.get_metrics_summary()
    print(f"\nMetrics summary: {json.dumps(summary, indent=2)}")
