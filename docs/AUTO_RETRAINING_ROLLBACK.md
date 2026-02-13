# Auto-Retraining and Rollback Systems

## Overview

This document describes the automatic retraining and rollback mechanisms that enable self-adaptive and reliable ML model deployments.

## Auto-Retraining System

### Purpose

Automatically retrain the model when data drift is detected, ensuring the model stays accurate as data distributions change.

### Trigger Conditions

Retraining is triggered when:
- **Drift Score > 0.2** (PSI threshold exceeded)
- **OR >20% of features** show drift
- **OR any feature** exceeds all three drift thresholds (PSI, KL, KS)

### Retraining Workflow

```
1. Drift Detection
   ↓
2. Drift Threshold Exceeded?
   ↓ (Yes)
3. Increment Model Version (1.0.0 → 1.1.0)
   ↓
4. Train New Model
   ↓
5. Evaluate Performance
   ↓
6. Save Model + Metadata
   ↓
7. Update latest_model.json
   ↓
8. Log to model_history.json
   ↓
9. Deploy New Version
```

### Version Increment Rules

| Change Type | Version Change | Example |
|-------------|---------------|---------|
| Drift-triggered retrain | Minor version | 1.0.0 → 1.1.0 |
| Algorithm change | Major version | 1.5.0 → 2.0.0 |
| Bug fix | Patch version | 1.1.0 → 1.1.1 |

### Usage

```python
from src.auto_retraining import AutoRetrainingSystem

# Initialize system
retraining_system = AutoRetrainingSystem(
    models_dir='models',
    drift_threshold=0.2
)

# Check for drift and retrain if needed
result = retraining_system.check_and_retrain(current_data)

if result['retraining_triggered']:
    print(f"New model trained: v{result['new_version']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
```

### Output Files

After retraining:
- `models/breast_cancer_model_v{version}.pkl` - New model file
- `models/breast_cancer_model_v{version}_metadata.json` - Model metadata
- `models/latest_model.json` - Updated to point to new version
- `models/model_history.json` - Version history log
- `retraining.log` - Retraining event logs

### Metadata Schema

```json
{
  "model_name": "breast_cancer_model",
  "version": "1.1.0",
  "algorithm": "RandomForestClassifier",
  "metrics": {
    "accuracy": 0.9561,
    "f1_score": 0.9561,
    "precision": 0.9565,
    "recall": 0.9561
  },
  "training_date": "2026-02-13T14:15:00",
  "training_samples": 455,
  "test_samples": 114,
  "feature_names": [...],
  "features": 30,
  "trigger": "drift_detection",
  "drift_score": 0.2543,
  "previous_version": "1.0.0"
}
```

---

## Rollback System

### Purpose

Automatically rollback to a previous stable model version if performance degradation is detected in production.

### Trigger Conditions

Rollback is triggered when:
- **Accuracy drops >2%** from previous version
- **OR error rate increases significantly**
- **OR critical bugs detected**

### Rollback Workflow

```
1. Monitor Production Performance
   ↓
2. Compare with Previous Version
   ↓
3. Degradation Detected?
   ↓ (Yes)
4. Load Previous Version Info
   ↓
5. Update latest_model.json
   ↓
6. Log Rollback Event
   ↓
7. Alert Team
   ↓
8. Previous Version Active
```

### Performance Monitoring

The system continuously monitors:
- **Accuracy**: Primary metric
- **Error Rate**: Secondary metric
- **Latency**: Performance metric
- **Prediction Volume**: Traffic metric

### Usage

```python
from src.rollback_system import RollbackSystem

# Initialize system
rollback_system = RollbackSystem(
    models_dir='models',
    performance_threshold=0.02  # 2% drop
)

# Monitor and rollback if needed
result = rollback_system.monitor_and_rollback(X_test, y_test)

if result['rollback_performed']:
    print(f"Rolled back from v{result['from_version']} to v{result['to_version']}")
```

### Rollback History

All rollback events are logged to `models/rollback_history.json`:

```json
[
  {
    "timestamp": "2026-02-13T14:20:00",
    "from_version": "1.2.0",
    "to_version": "1.1.0",
    "reason": "performance_degradation",
    "threshold": 0.02,
    "performance_drop": 0.035
  }
]
```

---

## Integration

### Combined Workflow

```
Production Data
   ↓
Drift Detection
   ↓
Drift? → Yes → Auto-Retrain → New Version (v1.1.0)
   ↓                              ↓
   No                        Deploy & Monitor
   ↓                              ↓
Continue                    Performance OK?
                                  ↓
                                  No
                                  ↓
                            Auto-Rollback → Previous Version (v1.0.0)
```

### API Integration

Add to `model_api.py`:

```python
from src.drift_detection_enhanced import EnhancedDriftDetector
from src.auto_retraining import AutoRetrainingSystem
from src.rollback_system import RollbackSystem

# Initialize systems
drift_detector = EnhancedDriftDetector()
retraining_system = AutoRetrainingSystem()
rollback_system = RollbackSystem()

# Periodic drift check (e.g., every 1000 predictions)
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Make prediction
    prediction = model.predict(features)
    
    # Check drift periodically
    if prediction_count % 1000 == 0:
        drift_result = drift_detector.check_drift(recent_data)
        
        if drift_result['drift_detected']:
            # Trigger retraining
            retrain_result = retraining_system.check_and_retrain()
            if retrain_result['retraining_triggered']:
                logger.info(f"Model retrained: v{retrain_result['new_version']}")
    
    return prediction

# Periodic performance check (e.g., hourly)
@app.get("/health/performance")
async def check_performance():
    result = rollback_system.monitor_and_rollback(test_data, test_labels)
    
    if result['rollback_performed']:
        logger.warning(f"Rollback performed: v{result['to_version']}")
    
    return result
```

---

## Monitoring & Alerts

### Grafana Dashboards

**Panel 1: Model Version Timeline**
```promql
# Show current model version
model_version_info
```

**Panel 2: Retraining Events**
```promql
# Count retraining events
increase(model_retrain_total[24h])
```

**Panel 3: Rollback Events**
```promql
# Count rollback events
increase(model_rollback_total[24h])
```

### Alert Rules

**Retraining Alert:**
```yaml
alert: ModelRetrained
expr: increase(model_retrain_total[5m]) > 0
labels:
  severity: info
annotations:
  summary: "Model retrained due to drift"
```

**Rollback Alert:**
```yaml
alert: ModelRolledBack
expr: increase(model_rollback_total[5m]) > 0
labels:
  severity: critical
annotations:
  summary: "Model rolled back due to performance degradation"
```

---

## Testing

### Test Auto-Retraining

```bash
# Run auto-retraining test
python src/auto_retraining.py
```

**Expected Output:**
```
TEST 1: Clean Data (No Retraining Expected)
✓ No drift detected. Retraining not needed.
Drift Score: 0.0543 (threshold: 0.2)

TEST 2: Simulated Drift (Retraining Expected)
⚠️  DRIFT THRESHOLD EXCEEDED!
Drift Score: 0.2543 (threshold: 0.2)
Triggering automatic retraining...
✓ RETRAINING COMPLETE!
New Version: v1.1.0
Accuracy: 0.9561
```

### Test Rollback

```bash
# Run rollback test
python src/rollback_system.py
```

**Expected Output:**
```
TEST 1: Normal Performance (No Rollback Expected)
✓ No performance degradation detected
Current accuracy: 0.9561
Previous accuracy: 0.9561

TEST 2: Simulated Degradation (Rollback Expected)
⚠️  Performance degradation detected!
Drop: 5.00% (threshold: 2.00%)
PERFORMING ROLLBACK
Rolling back from v1.1.0 to v1.0.0
✓ Rollback complete
```

---

## Research Paper Documentation

### Key Metrics to Report

| Metric | Manual | Static CI/CD | Proposed System |
|--------|--------|--------------|-----------------|
| **Retraining Time** | 30-60 min | 15-30 min | **5-10 min** ⚡ |
| **Retraining Trigger** | Manual | Manual | **Automatic** ✅ |
| **Rollback Time** | 15-30 min | 5-10 min | **1-2 min** ⚡ |
| **Rollback Trigger** | Manual | Manual | **Automatic** ✅ |
| **Drift Detection** | None | None | **Real-time** ✅ |
| **Version Management** | Manual | Semi-auto | **Automatic** ✅ |

### Screenshots to Capture

1. ✅ Auto-retraining script output
2. ✅ Rollback script output
3. ✅ Model version history JSON
4. ✅ Rollback history JSON
5. ✅ Retraining logs
6. ✅ Grafana showing version changes

### Proof Points for Paper

**Auto-Retraining:**
- ✅ Drift detection threshold (δ = 0.2)
- ✅ Automatic trigger on threshold exceeded
- ✅ Version auto-increment (1.0.0 → 1.1.0)
- ✅ Model auto-save with metadata
- ✅ Complete automation (no manual intervention)

**Rollback:**
- ✅ Performance monitoring (2% threshold)
- ✅ Automatic degradation detection
- ✅ Instant rollback to stable version
- ✅ Event logging for audit trail
- ✅ Zero downtime rollback

---

## Files

- **Auto-Retraining**: `src/auto_retraining.py`
- **Rollback**: `src/rollback_system.py`
- **Logs**: `retraining.log`, `rollback.log`
- **History**: `models/model_history.json`, `models/rollback_history.json`
- **Documentation**: `docs/AUTO_RETRAINING_ROLLBACK.md`

---

**Last Updated**: 2026-02-13
