# 📖 User Guide: MLOps Project

Complete guide on how to work with this MLOps project for automated ML model deployment.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Git

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Devops_Project
   ```

2. **Install dependencies**
   ```bash
   # Windows
   .\setup.ps1
   
   # Linux/Mac
   ./setup.sh
   
   # Or manually
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import sklearn, fastapi, prometheus_client; print('✅ All dependencies installed')"
   ```

## 📦 Adding Your ML Model

### Step 1: Prepare Your Model

Place your trained ML model file (`.pkl`, `.joblib`, etc.) in the `models/` folder:

```bash
# Example
cp my_model.pkl models/
```

**Supported formats:**
- `.pkl` (pickle)
- `.joblib` (scikit-learn joblib)
- Models must have a `predict()` method
- Optional: `predict_proba()` method for confidence scores

### Step 2: Create Metadata (Optional but Recommended)

Create a metadata JSON file with the same name as your model:

```json
{
  "model_name": "my_classifier",
  "version": "1.0.0",
  "model_type": "RandomForestClassifier",
  "feature_names": ["feature_1", "feature_2", ...],
  "metrics": {
    "accuracy": 0.95,
    "training_date": "2025-02-08T10:00:00"
  }
}
```

Save as: `models/my_model_metadata.json`

## 🧪 Testing Your Model for Deployment

### Interactive Model Testing and Deployment

Run the canary deployment script to test and deploy your model:

```bash
python src/canary_deployment.py
```

**What happens:**
1. Lists all available models in `models/` folder
2. You select a model to test
3. Runs comprehensive tests:
   - ✅ Model structure validation
   - ✅ Performance metrics (accuracy, precision, recall, F1)
   - ✅ Cross-validation stability
   - ✅ Prediction latency
   - ✅ Model compatibility
4. Shows deployment recommendation
5. Starts canary deployment if tests pass

### Command-Line Options

```bash
# List available models
python src/canary_deployment.py --list-models

# Skip tests and deploy directly
python src/canary_deployment.py --skip-tests

# Auto-deploy if tests pass (no confirmation)
python src/canary_deployment.py --auto-deploy
```

### Understanding Test Results

The testing process evaluates your model against these criteria:

| Test | What It Checks | Threshold |
|------|---------------|------------|
| **Structure** | Model has required methods | Must have `predict()` |
| **Performance** | Accuracy, Precision, Recall, F1 | Min 0.70-0.75 |
| **Stability** | Cross-validation consistency | CV mean ≥ 0.70, std ≤ 0.10 |
| **Latency** | Prediction speed | P95 latency ≤ 1000ms |
| **Compatibility** | Feature validation, probability checks | Must validate inputs |

**Output Example:**
```
DEPLOYMENT RECOMMENDATION: READY FOR DEPLOYMENT
✅ All tests passed - Model is deployment-worthy
```

## 🔄 Complete Workflow Examples

### Example 1: Deploy a New Model

```bash
# 1. Place your model in models folder
cp my_new_model.pkl models/

# 2. Run deployment workflow
python src/canary_deployment.py

# 3. Select your model from the list
# 4. Review test results
# 5. Confirm deployment
```

### Example 2: Train and Deploy Pipeline

```bash
# 1. Train a new model
python src/train_model.py --version 2.0.0

# 2. Validate the model
python src/validate_model.py

# 3. Deploy with canary
python src/canary_deployment.py --auto-deploy
```

### Example 3: Using the Orchestrator

```bash
# Run complete pipeline (train → validate → deploy)
python src/orchestrator.py full --version 2.0.0

# Or step by step
python src/orchestrator.py train --version 2.0.0
python src/orchestrator.py validate
python src/orchestrator.py deploy
```

## 🌐 Running the API Server

### Start the Model API

```bash
python src/model_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Make Predictions

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7, 1.9, 3.2]}'

# Using Python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.2, 3.4, 5.6, 7.8, 2.1, 4.3, 6.5, 8.7, 1.9, 3.2]}
)
print(response.json())
```

## 📊 Monitoring and Dashboards

### Start Monitoring Stack

```bash
# Using Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Access dashboards:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
```

### View Metrics via API

```bash
# Current metrics
curl http://localhost:8000/monitoring/metrics

# Metrics summary
curl http://localhost:8000/monitoring/summary

# Decision history
curl http://localhost:8000/decision/history
```

## 🔧 Configuration

### Deployment Policies

Edit `config/deployment_policies.json` to customize:

```json
{
  "drift_threshold": 0.3,
  "error_rate_threshold": 0.05,
  "latency_threshold_ms": 1000,
  "enable_auto_retraining": true,
  "enable_auto_rollback": true
}
```

### Canary Configuration

Edit `config/canary_config.json` to customize canary deployment:

```json
{
  "initial_percentage": 10,
  "increment_percentage": 10,
  "evaluation_duration_minutes": 30,
  "success_thresholds": {
    "max_error_rate": 0.05,
    "max_latency_ms": 1000,
    "min_accuracy": 0.80
  }
}
```

## 🔄 Model Management

### List All Models

```bash
python src/canary_deployment.py --list-models
```

### Check Model Status

```bash
python src/orchestrator.py status
```

### Rollback to Previous Version

```bash
python src/rollback.py

# Or to specific version
python src/rollback.py --target 1.0.0
```

## 🐛 Troubleshooting

### Model Not Found

**Problem:** "No models found in models directory"

**Solution:**
- Ensure your model file is in the `models/` folder
- Check file extension is `.pkl` or `.joblib`
- Verify file permissions

### Tests Failing

**Problem:** Model fails deployment tests

**Solutions:**
- Check test output for specific failures
- Improve model performance (accuracy, etc.)
- Optimize prediction latency
- Ensure model has required methods (`predict()`)

### API Not Starting

**Problem:** API server fails to start

**Solutions:**
- Check if port 8000 is already in use
- Verify model file exists and is valid
- Check logs in `api_server.log`

### Import Errors

**Problem:** Module import errors

**Solution:**
```bash
# Ensure you're in the project root
cd Devops_Project

# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

## 📝 Best Practices

1. **Version Your Models**
   - Use semantic versioning (e.g., 1.0.0, 1.0.1)
   - Include version in metadata

2. **Always Test Before Deploying**
   - Run comprehensive tests
   - Review test results carefully
   - Fix issues before deployment

3. **Use Canary Deployments**
   - Start with 10% traffic
   - Monitor metrics
   - Gradually increase if successful

4. **Monitor Continuously**
   - Check Grafana dashboards
   - Review API metrics
   - Watch for drift detection alerts

5. **Keep Metadata Updated**
   - Include feature names
   - Record training metrics
   - Document model type

## 🎓 Learning Resources

- **Paper**: See `paper.md` for research background
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md` for technical details
- **API Docs**: http://localhost:8000/docs (when API is running)

## 💡 Tips

- **Quick Test**: Use `--skip-tests` only for development/testing
- **Production**: Always run full tests before production deployment
- **Monitoring**: Keep Grafana dashboard open during deployments
- **Logs**: Check log files in project root for detailed information

## 🆘 Getting Help

1. Check logs: `*.log` files in project root
2. Review test results: `models/test_results_*.json`
3. Check API health: `curl http://localhost:8000/health`
4. Review documentation: `README.md`, `IMPLEMENTATION_SUMMARY.md`

## 📋 Common Commands Cheat Sheet

```bash
# Model Testing & Deployment
python src/canary_deployment.py              # Interactive deployment
python src/canary_deployment.py --list-models  # List models

# Training & Validation
python src/train_model.py --version 1.0.0    # Train model
python src/validate_model.py                 # Validate model

# API & Monitoring
python src/model_api.py                      # Start API server
docker-compose -f docker/docker-compose.yml up  # Start monitoring

# Orchestration
python src/orchestrator.py full              # Complete pipeline
python src/orchestrator.py status            # System status

# Model Management
python src/rollback.py                       # Rollback model
python src/trigger_retraining.py             # Trigger retraining
```

---

**Happy Deploying! 🚀**
