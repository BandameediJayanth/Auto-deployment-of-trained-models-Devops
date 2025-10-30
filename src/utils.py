"""
Utility Functions
Auto-Deployment ML Models Project

Common utilities for model training, validation, and deployment.
"""

import os
import json
import logging
import hashlib
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

def setup_logging(log_file: str = "app.log", log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_file: Name of the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_path}")
    
    return logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    default_config = {
        "model": {
            "name": "ml_model",
            "version": "1.0.0",
            "type": "RandomForestClassifier"
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 10
        },
        "validation": {
            "min_accuracy": 0.8,
            "min_precision": 0.8,
            "min_recall": 0.8,
            "min_f1_score": 0.8,
            "max_cv_std": 0.05,
            "min_cv_mean": 0.8
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False
        },
        "monitoring": {
            "prometheus_port": 9090,
            "grafana_port": 3000
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with default config
            config = deep_merge(default_config, user_config)
            print(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            print("Using default configuration")
            config = default_config
    else:
        print(f"Config file {config_path} not found. Using default configuration")
        config = default_config
        
        # Save default config for future reference
        save_config(config, config_path)
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config file {config_path}: {e}")

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into base
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def calculate_model_hash(model_path: str) -> str:
    """
    Calculate MD5 hash of a model file
    
    Args:
        model_path: Path to the model file
    
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    
    try:
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    except Exception as e:
        print(f"Error calculating hash for {model_path}: {e}")
        return ""

def validate_data_schema(data: pd.DataFrame, expected_columns: List[str], 
                        required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate data schema against expected columns
    
    Args:
        data: DataFrame to validate
        expected_columns: List of expected column names
        required_columns: List of required column names (subset of expected)
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if required columns are present
    if required_columns:
        missing_required = set(required_columns) - set(data.columns)
        if missing_required:
            errors.append(f"Missing required columns: {list(missing_required)}")
    
    # Check for unexpected columns
    unexpected_columns = set(data.columns) - set(expected_columns)
    if unexpected_columns:
        errors.append(f"Unexpected columns found: {list(unexpected_columns)}")
    
    # Check data types and missing values
    for col in data.columns:
        if col in expected_columns:
            # Check for excessive missing values
            missing_pct = data[col].isnull().sum() / len(data) * 100
            if missing_pct > 50:
                errors.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
            
            # Check for non-numeric data in numeric columns
            if col != 'target' and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column '{col}' should be numeric but found {data[col].dtype}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def create_model_version(base_version: str = "1.0.0") -> str:
    """
    Create a new model version based on timestamp
    
    Args:
        base_version: Base version string
    
    Returns:
        New version string with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_version}_{timestamp}"

def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def clean_old_models(models_dir: str = "models", keep_versions: int = 5) -> None:
    """
    Clean up old model versions, keeping only the most recent ones
    
    Args:
        models_dir: Directory containing model files
        keep_versions: Number of versions to keep
    """
    try:
        model_files = []
        
        # Find all model files
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and not file.startswith('.'):
                file_path = os.path.join(models_dir, file)
                model_files.append((file_path, os.path.getctime(file_path)))
        
        # Sort by creation time (newest first)
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        files_to_remove = model_files[keep_versions:]
        
        for file_path, _ in files_to_remove:
            try:
                os.remove(file_path)
                
                # Also remove corresponding metadata file
                metadata_path = file_path.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                print(f"Removed old model: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    except Exception as e:
        print(f"Error cleaning old models: {e}")

def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metric values
        decimal_places: Number of decimal places to show
    
    Returns:
        Dictionary of formatted metric strings
    """
    formatted = {}
    
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            if metric.endswith('_count') or metric.endswith('_samples'):
                formatted[metric] = f"{int(value):,}"
            else:
                formatted[metric] = f"{value:.{decimal_places}f}"
        else:
            formatted[metric] = str(value)
    
    return formatted

def generate_model_report(metadata: Dict[str, Any], validation_results: Optional[Dict] = None) -> str:
    """
    Generate a comprehensive model report
    
    Args:
        metadata: Model metadata
        validation_results: Optional validation results
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("MODEL REPORT")
    report.append("=" * 60)
    
    # Model Info
    report.append("\n📊 MODEL INFORMATION")
    report.append("-" * 30)
    report.append(f"Name: {metadata.get('model_name', 'Unknown')}")
    report.append(f"Version: {metadata.get('version', 'Unknown')}")
    report.append(f"Type: {metadata.get('model_type', 'Unknown')}")
    report.append(f"Features: {len(metadata.get('feature_names', []))}")
    
    # Training Metrics
    if 'metrics' in metadata:
        metrics = metadata['metrics']
        report.append("\n📈 TRAINING METRICS")
        report.append("-" * 30)
        
        formatted_metrics = format_metrics(metrics)
        for metric, value in formatted_metrics.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                report.append(f"{metric.capitalize()}: {value}")
        
        report.append(f"Training Samples: {metrics.get('training_samples', 'Unknown')}")
        report.append(f"Test Samples: {metrics.get('test_samples', 'Unknown')}")
        report.append(f"Training Date: {metrics.get('training_date', 'Unknown')}")
    
    # Validation Results
    if validation_results:
        report.append("\n✅ VALIDATION RESULTS")
        report.append("-" * 30)
        
        validations = validation_results.get('validations', {})
        overall_status = "PASSED" if validations.get('overall_pass', False) else "FAILED"
        report.append(f"Overall Status: {overall_status}")
        
        if 'performance_metrics' in validation_results:
            perf_metrics = validation_results['performance_metrics']
            formatted_perf = format_metrics(perf_metrics)
            
            for metric, value in formatted_perf.items():
                report.append(f"Validation {metric.capitalize()}: {value}")
    
    # Feature Information
    if 'feature_names' in metadata:
        report.append("\n🔧 FEATURES")
        report.append("-" * 30)
        feature_names = metadata['feature_names']
        
        if len(feature_names) <= 10:
            for i, feature in enumerate(feature_names, 1):
                report.append(f"{i:2d}. {feature}")
        else:
            for i, feature in enumerate(feature_names[:5], 1):
                report.append(f"{i:2d}. {feature}")
            report.append(f"... and {len(feature_names) - 5} more features")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

def save_experiment_log(experiment_data: Dict[str, Any], log_file: str = "experiments.jsonl") -> None:
    """
    Save experiment data to a JSONL log file
    
    Args:
        experiment_data: Dictionary containing experiment information
        log_file: Path to log file
    """
    ensure_directory("logs")
    log_path = os.path.join("logs", log_file)
    
    # Add timestamp
    experiment_data['logged_at'] = datetime.now().isoformat()
    
    try:
        with open(log_path, 'a') as f:
            json.dump(experiment_data, f)
            f.write('\n')
        
        print(f"Experiment logged to {log_path}")
    
    except Exception as e:
        print(f"Error logging experiment: {e}")

def load_experiment_logs(log_file: str = "experiments.jsonl") -> List[Dict[str, Any]]:
    """
    Load experiment logs from JSONL file
    
    Args:
        log_file: Path to log file
    
    Returns:
        List of experiment dictionaries
    """
    log_path = os.path.join("logs", log_file)
    experiments = []
    
    if not os.path.exists(log_path):
        return experiments
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    experiments.append(json.loads(line))
        
        print(f"Loaded {len(experiments)} experiments from {log_path}")
    
    except Exception as e:
        print(f"Error loading experiments: {e}")
    
    return experiments

# Environment variable helpers
def get_env_var(var_name: str, default_value: Any = None, var_type: type = str) -> Any:
    """
    Get environment variable with type conversion and default value
    
    Args:
        var_name: Name of environment variable
        default_value: Default value if variable not found
        var_type: Type to convert the value to
    
    Returns:
        Environment variable value or default
    """
    value = os.getenv(var_name, default_value)
    
    if value is None:
        return default_value
    
    try:
        if var_type == bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif var_type == list:
            return str(value).split(',')
        else:
            return var_type(value)
    
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {var_name}='{value}' to {var_type.__name__}. Using default: {default_value}")
        return default_value

# Performance monitoring helpers
class PerformanceMonitor:
    """Simple performance monitoring context manager"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        print(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            print(f"✅ {self.operation_name} completed in {duration:.2f} seconds")
        else:
            print(f"❌ {self.operation_name} failed after {duration:.2f} seconds")
        
        return False
