"""
Drift Detection Module
Auto-Deployment ML Models Project

This module provides functionality to detect data drift between reference data
and new incoming data using statistical tests (Kolmogorov-Smirnov test).
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drift_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data_path='data/dataset.csv', p_value_threshold=0.05):
        self.reference_data_path = reference_data_path
        self.p_value_threshold = p_value_threshold
        self.reference_data = None
        self.feature_names = None
        
        self.load_reference_data()
        
    def load_reference_data(self):
        """Load the training data to serve as reference"""
        try:
            if os.path.exists(self.reference_data_path):
                logger.info(f"Loading reference data from {self.reference_data_path}")
                df = pd.read_csv(self.reference_data_path)
                
                # Drop target if present, we only monitor features
                if 'target' in df.columns:
                    df = df.drop('target', axis=1)
                
                self.reference_data = df
                self.feature_names = df.columns.tolist()
                logger.info(f"Reference data loaded: {df.shape[0]} samples, {df.shape[1]} features")
                return True
            else:
                logger.warning(f"Reference data file {self.reference_data_path} not found.")
                return False
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            return False

    def check_drift(self, new_data):
        """
        Check for drift in new data compared to reference data.
        
        Args:
            new_data (pd.DataFrame or list of lists): New data to check.
            
        Returns:
            dict: Drift detection results including drift detected (bool),
                  drift score (float), and per-feature results.
        """
        if self.reference_data is None:
            if not self.load_reference_data():
                return {"error": "Reference data not available"}
        
        # Convert to DataFrame if necessary
        if not isinstance(new_data, pd.DataFrame):
            try:
                new_data = pd.DataFrame(new_data, columns=self.feature_names)
            except Exception as e:
                logger.error(f"Error converting new data to DataFrame: {str(e)}")
                return {"error": str(e)}
        
        drift_results = {
            "drift_detected": False,
            "drifted_features": [],
            "feature_scores": {},
            "total_features": len(self.feature_names),
            "drifted_feature_count": 0
        }
        
        for feature in self.feature_names:
            if feature not in new_data.columns:
                continue
                
            ref_dist = self.reference_data[feature]
            curr_dist = new_data[feature]
            
            # Perform KS test
            statistic, p_value = ks_2samp(ref_dist, curr_dist)
            
            is_drifted = bool(p_value < self.p_value_threshold)
            
            drift_results["feature_scores"][feature] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "is_drifted": is_drifted
            }
            
            if is_drifted:
                drift_results["drifted_features"].append(feature)
        
        drift_results["drifted_feature_count"] = len(drift_results["drifted_features"])
        
        # If more than 30% of features have drifted, we consider the dataset drifted
        drift_ratio = drift_results["drifted_feature_count"] / drift_results["total_features"]
        if drift_ratio > 0.3: 
            drift_results["drift_detected"] = True
            
        logger.info(f"Drift check complete. Detected: {drift_results['drift_detected']} ({drift_results['drifted_feature_count']}/{drift_results['total_features']} features)")
        
        return drift_results

if __name__ == "__main__":
    # Test run
    detector = DriftDetector()
    if detector.reference_data is not None:
        # Create some synthetic drifted data
        sample_data = detector.reference_data.sample(100).copy()
        # Shift the distribution of the first feature
        first_feature = detector.feature_names[0]
        sample_data[first_feature] = sample_data[first_feature] + 5.0
        
        result = detector.check_drift(sample_data)
        print(json.dumps(result, indent=2))
