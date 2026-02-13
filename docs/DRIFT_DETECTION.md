# Drift Detection System

## Overview

The drift detection system monitors data distribution changes between training (reference) data and production (current) data using multiple statistical methods.

## Statistical Methods

### 1. Population Stability Index (PSI)

**Formula**: PSI = Σ (% current - % reference) × ln(% current / % reference)

**Interpretation**:
- PSI < 0.1: No significant change
- PSI 0.1-0.2: Small change  
- **PSI > 0.2: Significant drift detected** ⚠️

**Threshold**: 0.2

### 2. Kullback-Leibler (KL) Divergence

**Formula**: KL(P||Q) = Σ P(x) × log(P(x) / Q(x))

**Interpretation**:
- Measures how one probability distribution diverges from another
- **KL > 0.1: Drift detected** ⚠️

**Threshold**: 0.1

### 3. Kolmogorov-Smirnov (KS) Test

**Formula**: D = max|F₁(x) - F₂(x)|

**Interpretation**:
- Tests if two samples come from the same distribution
- **p-value < 0.05: Drift detected** ⚠️

**Threshold**: p-value = 0.05

## Usage

### Basic Usage

```python
from src.drift_detection_enhanced import EnhancedDriftDetector

# Initialize detector
detector = EnhancedDriftDetector(
    reference_data_path='data/dataset.csv',
    psi_threshold=0.2,
    ks_threshold=0.05,
    kl_threshold=0.1
)

# Check for drift
results = detector.check_drift(new_data)

# Check if drift detected
if results['drift_detected']:
    print(f"⚠️ Drift detected! Score: {results['drift_score']:.4f}")
    print(f"Drifted features: {results['drifted_features']}")
```

### Command Line

```bash
# Run drift detection test
python src/drift_detection_enhanced.py
```

## Output Format

```json
{
  "timestamp": "2026-02-13T14:00:00",
  "n_samples": 100,
  "drift_detected": true,
  "drift_score": 0.2543,
  "drift_percentage": 35.5,
  "drifted_feature_count": 11,
  "drifted_features": ["feature1", "feature2", ...],
  "total_features": 30,
  "feature_scores": {
    "feature1": {
      "psi": 0.2543,
      "kl_divergence": 0.1234,
      "ks_statistic": 0.3456,
      "ks_pvalue": 0.0123,
      "is_drifted": true
    }
  },
  "thresholds": {
    "psi": 0.2,
    "ks": 0.05,
    "kl": 0.1
  }
}
```

## Prometheus Metrics

The drift detector exports the following Prometheus metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `model_drift_score` | Gauge | Overall drift score (PSI-based) |
| `model_drift_detected` | Gauge | Whether drift is detected (1=yes, 0=no) |
| `model_drifted_features_count` | Gauge | Number of features with detected drift |
| `model_drift_checks_total` | Counter | Total number of drift checks performed |

### Querying in Prometheus

```promql
# Current drift score
model_drift_score

# Drift detection status
model_drift_detected

# Number of drifted features
model_drifted_features_count

# Drift checks over time
rate(model_drift_checks_total[5m])
```

## Alerting

### Alert Conditions

**Drift Alert Triggered When**:
- Overall drift score > 0.2 (PSI threshold)
- OR >20% of features show drift
- OR any individual feature exceeds all three thresholds

### Alert Actions

When drift is detected:
1. ⚠️ Log warning message
2. 📊 Update Prometheus metrics
3. 💾 Save drift report to `reports/drift_report_detected.json`
4. 🔔 Trigger alert (if configured)
5. 📧 Notify team (if configured)

### Recommended Actions

When drift alert fires:
1. **Investigate**: Review drift report to identify drifted features
2. **Analyze**: Determine root cause (data quality, distribution shift, etc.)
3. **Decide**: 
   - If drift is expected: Update reference data
   - If drift is problematic: Retrain model
4. **Monitor**: Track model performance after action taken

## Drift Simulation

For testing purposes, you can simulate drift:

```python
from src.drift_detection_enhanced import simulate_drift

# Simulate drift on 30% of features
drifted_data = simulate_drift(
    data=original_data,
    drift_magnitude=0.8,  # 0-1 scale
    drift_features=None   # None = random 30%
)
```

## Integration with ML API

### Automatic Drift Monitoring

Add drift monitoring to your prediction pipeline:

```python
# In model_api.py
from src.drift_detection_enhanced import EnhancedDriftDetector

# Initialize detector
drift_detector = EnhancedDriftDetector()

# Monitor predictions
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Make prediction
    prediction = model.predict(features)
    
    # Check for drift (periodically)
    if should_check_drift():
        drift_results = drift_detector.check_drift(recent_predictions)
        if drift_results['drift_detected']:
            logger.warning("Drift detected! Consider retraining.")
    
    return prediction
```

## Monitoring Dashboard

### Grafana Dashboard Panels

**Panel 1: Drift Score Over Time**
```promql
model_drift_score
```

**Panel 2: Drift Detection Status**
```promql
model_drift_detected
```

**Panel 3: Drifted Features Count**
```promql
model_drifted_features_count
```

**Panel 4: Drift Check Rate**
```promql
rate(model_drift_checks_total[5m])
```

## Research Paper Documentation

### Screenshots to Capture

1. ✅ Drift detection output (clean data)
2. ✅ Drift detection output (drifted data)
3. ✅ Drift report JSON files
4. ✅ Prometheus metrics showing drift
5. ✅ Grafana dashboard with drift metrics
6. ✅ Alert triggered screenshot

### Metrics to Report

| Metric | Clean Data | Drifted Data |
|--------|------------|--------------|
| Drift Score | ~0.05 | >0.25 |
| Drifted Features | 0-2 | >6 |
| Drift Detected | No | Yes |

### Key Points for Paper

1. **Multi-Method Approach**: Uses PSI, KL, and KS for robust detection
2. **Threshold-Based**: Clear thresholds for automated decision-making
3. **Real-Time Monitoring**: Prometheus integration for live tracking
4. **Actionable Alerts**: Automated alerting when drift exceeds thresholds
5. **Comprehensive Logging**: Detailed logs for debugging and analysis

## Files

- **Implementation**: `src/drift_detection_enhanced.py`
- **Reports**: `reports/drift_report_*.json`
- **Logs**: `drift_detection.log`
- **Documentation**: `docs/DRIFT_DETECTION.md`

## References

- PSI: [Population Stability Index](https://www.listendata.com/2015/05/population-stability-index.html)
- KL Divergence: [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- KS Test: [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)

---

**Last Updated**: 2026-02-13
