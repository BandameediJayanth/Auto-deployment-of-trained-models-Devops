"""
Enhanced Drift Detection Module with Multiple Statistical Methods
Includes PSI, KL Divergence, KS Test, Prometheus Metrics, and Alerting
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.special import kl_div
import logging
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from prometheus_client import Gauge, Counter

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

# Prometheus metrics
drift_score_metric = Gauge('model_drift_score', 'Overall drift score (PSI-based)')
drift_detected_metric = Gauge('model_drift_detected', 'Whether drift is detected (1=yes, 0=no)')
drifted_features_metric = Gauge('model_drifted_features_count', 'Number of features with detected drift')
drift_checks_total = Counter('model_drift_checks_total', 'Total number of drift checks performed')

class EnhancedDriftDetector:
    """
    Enhanced drift detector with multiple statistical methods:
    - PSI (Population Stability Index)
    - KL Divergence (Kullback-Leibler)
    - KS Test (Kolmogorov-Smirnov)
    """
    
    def __init__(self, reference_data_path='data/dataset.csv', 
                 psi_threshold=0.2, ks_threshold=0.05, kl_threshold=0.1):
        """
        Initialize enhanced drift detector
        
        Args:
            reference_data_path: Path to reference/training data
            psi_threshold: PSI threshold (>0.2 = significant drift)
            ks_threshold: KS test p-value threshold (<0.05 = drift)
            kl_threshold: KL divergence threshold (>0.1 = drift)
        """
        self.reference_data_path = reference_data_path
        self.thresholds = {
            'psi': psi_threshold,
            'ks': ks_threshold,
            'kl': kl_threshold
        }
        self.reference_data = None
        self.feature_names = None
        
        self.load_reference_data()
        
        logger.info(f"Enhanced drift detector initialized")
        logger.info(f"Thresholds: PSI={psi_threshold}, KS={ks_threshold}, KL={kl_threshold}")
        
    def load_reference_data(self):
        """Load the training data to serve as reference"""
        try:
            if os.path.exists(self.reference_data_path):
                logger.info(f"Loading reference data from {self.reference_data_path}")
                df = pd.read_csv(self.reference_data_path)
                
                # Drop target if present
                if 'target' in df.columns:
                    df = df.drop('target', axis=1)
                
                self.reference_data = df
                self.feature_names = df.columns.tolist()
                logger.info(f"Reference data loaded: {df.shape[0]} samples, {df.shape[1]} features")
                return True
            else:
                logger.warning(f"Reference data file {self.reference_data_path} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            return False
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI Interpretation:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.2: Small change
        - PSI > 0.2: Significant change (drift)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Calculate distributions
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        ref_percents = (ref_counts + epsilon) / (len(reference) + epsilon * len(breakpoints))
        cur_percents = (cur_counts + epsilon) / (len(current) + epsilon * len(breakpoints))
        
        # Calculate PSI
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
        
        return float(psi)
    
    def calculate_kl_divergence(self, reference: np.ndarray, current: np.ndarray, bins: int = 30) -> float:
        """
        Calculate Kullback-Leibler divergence
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            KL divergence value
        """
        # Create bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins_array = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins_array, density=True)
        cur_hist, _ = np.histogram(current, bins=bins_array, density=True)
        
        # Add epsilon
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        cur_hist = cur_hist + epsilon
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        # Calculate KL divergence
        kl = np.sum(kl_div(cur_hist, ref_hist))
        
        return float(kl)
    
    def check_drift(self, new_data):
        """
        Comprehensive drift check using multiple statistical methods
        
        Args:
            new_data: New data to check (DataFrame or array)
            
        Returns:
            dict: Comprehensive drift detection results
        """
        drift_checks_total.inc()
        
        if self.reference_data is None:
            if not self.load_reference_data():
                return {"error": "Reference data not available"}
        
        # Convert to DataFrame if necessary
        if not isinstance(new_data, pd.DataFrame):
            try:
                new_data = pd.DataFrame(new_data, columns=self.feature_names)
            except Exception as e:
                logger.error(f"Error converting new data: {str(e)}")
                return {"error": str(e)}
        
        logger.info(f"Starting drift detection on {len(new_data)} samples...")
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(new_data),
            "drift_detected": False,
            "drifted_features": [],
            "feature_scores": {},
            "total_features": len(self.feature_names),
            "drifted_feature_count": 0,
            "drift_score": 0.0,
            "drift_percentage": 0.0,
            "thresholds": self.thresholds
        }
        
        total_psi = 0.0
        drift_count = 0
        
        for feature in self.feature_names:
            if feature not in new_data.columns:
                continue
            
            ref_dist = self.reference_data[feature].values
            curr_dist = new_data[feature].values
            
            # Calculate all metrics
            psi = self.calculate_psi(ref_dist, curr_dist)
            kl = self.calculate_kl_divergence(ref_dist, curr_dist)
            ks_stat, ks_pvalue = ks_2samp(ref_dist, curr_dist)
            
            # Determine if drift detected
            drift_detected = (
                psi > self.thresholds['psi'] or
                ks_pvalue < self.thresholds['ks'] or
                kl > self.thresholds['kl']
            )
            
            drift_results["feature_scores"][feature] = {
                "psi": float(psi),
                "kl_divergence": float(kl),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "is_drifted": drift_detected
            }
            
            if drift_detected:
                drift_count += 1
                drift_results["drifted_features"].append(feature)
            
            total_psi += psi
        
        # Calculate overall metrics
        drift_results["drifted_feature_count"] = drift_count
        drift_results["drift_score"] = total_psi / len(self.feature_names)
        drift_results["drift_percentage"] = (drift_count / len(self.feature_names)) * 100
        
        # Overall drift decision (>20% features drifted)
        if drift_count > (len(self.feature_names) * 0.2):
            drift_results["drift_detected"] = True
        
        # Update Prometheus metrics
        drift_score_metric.set(drift_results["drift_score"])
        drift_detected_metric.set(1 if drift_results["drift_detected"] else 0)
        drifted_features_metric.set(drift_count)
        
        # Log results
        logger.info(f"Drift check complete:")
        logger.info(f"  - Drift Score: {drift_results['drift_score']:.4f}")
        logger.info(f"  - Drifted Features: {drift_count}/{len(self.feature_names)} ({drift_results['drift_percentage']:.1f}%)")
        logger.info(f"  - Drift Detected: {drift_results['drift_detected']}")
        
        if drift_results["drift_detected"]:
            logger.warning(f"DRIFT ALERT: Threshold exceeded!")
            logger.warning(f"   Drift Score: {drift_results['drift_score']:.4f} (threshold: {self.thresholds['psi']})")
            logger.warning(f"   Recommended Action: Retrain model")
            
            # Log top drifted features
            if drift_results["drifted_features"]:
                sorted_features = sorted(
                    [(f, drift_results["feature_scores"][f]["psi"]) 
                     for f in drift_results["drifted_features"]],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.warning(f"   Top drifted features:")
                for feat, psi_val in sorted_features:
                    logger.warning(f"     - {feat}: PSI={psi_val:.4f}")
        
        return drift_results
    
    def save_drift_report(self, results: Dict, filepath: str = 'reports/drift_report.json'):
        """Save drift detection results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Drift report saved to {filepath}")


def simulate_drift(data: pd.DataFrame, drift_magnitude: float = 0.5, 
                   drift_features: Optional[list] = None) -> pd.DataFrame:
    """
    Simulate data drift for testing
    
    Args:
        data: Original data
        drift_magnitude: Magnitude of drift (0-1)
        drift_features: Features to drift (None = random 30%)
        
    Returns:
        Drifted data
    """
    drifted_data = data.copy()
    
    if drift_features is None:
        n_drift = int(len(data.columns) * 0.3)
        drift_features = np.random.choice(data.columns, n_drift, replace=False)
    
    for feature in drift_features:
        shift = np.random.randn() * drift_magnitude * data[feature].std()
        scale = 1 + np.random.randn() * drift_magnitude * 0.2
        drifted_data[feature] = drifted_data[feature] * scale + shift
    
    logger.info(f"Simulated drift on {len(drift_features)} features with magnitude {drift_magnitude}")
    
    return drifted_data


if __name__ == "__main__":
    # Test the enhanced drift detector
    print("\n" + "="*70)
    print("ENHANCED DRIFT DETECTION DEMONSTRATION")
    print("="*70 + "\n")
    
    detector = EnhancedDriftDetector()
    
    if detector.reference_data is not None:
        # Test 1: Clean data (no drift)
        print("\n" + "-"*70)
        print("TEST 1: Clean Data (No Drift Expected)")
        print("-"*70)
        sample_clean = detector.reference_data.sample(100)
        result_clean = detector.check_drift(sample_clean)
        print(f"Drift Score: {result_clean['drift_score']:.4f}")
        print(f"Drift Detected: {result_clean['drift_detected']}")
        print(f"Drifted Features: {result_clean['drifted_feature_count']}/{result_clean['total_features']}")
        
        # Test 2: Simulated drift
        print("\n" + "-"*70)
        print("TEST 2: Simulated Drift (Drift Expected)")
        print("-"*70)
        sample_drifted = simulate_drift(detector.reference_data.sample(100), drift_magnitude=0.8)
        result_drift = detector.check_drift(sample_drifted)
        print(f"Drift Score: {result_drift['drift_score']:.4f}")
        print(f"Drift Detected: {result_drift['drift_detected']}")
        print(f"Drifted Features: {result_drift['drifted_feature_count']}/{result_drift['total_features']}")
        
        if result_drift['drifted_features']:
            print(f"\nTop drifted features:")
            sorted_features = sorted(
                result_drift['feature_scores'].items(),
                key=lambda x: x[1]['psi'],
                reverse=True
            )[:5]
            for feat, metrics in sorted_features:
                print(f"  - {feat}: PSI={metrics['psi']:.4f}, KL={metrics['kl_divergence']:.4f}")
        
        # Save reports
        detector.save_drift_report(result_clean, 'reports/drift_report_clean.json')
        detector.save_drift_report(result_drift, 'reports/drift_report_detected.json')
        
        print("\n" + "="*70)
        print("Drift detection complete. Reports saved to 'reports/' directory")
        print("="*70 + "\n")
