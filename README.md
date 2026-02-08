# 🚀 Auto-Deployment of Trained ML Models

An end-to-end MLOps project implementing automated deployment of machine learning models using DevOps and CI/CD principles.

## 📋 Project Overview

This project creates a fully automated pipeline for deploying ML models from training to production, eliminating manual deployment steps and ensuring fast, reliable, and scalable delivery.

### 🎯 Core Objectives
- ✅ Automate ML Model Deployment Process
- ✅ Implement CI/CD Pipeline
- ✅ Ensure Scalability and Reusability
- ✅ Monitor and Log Model Performance
- ✅ Enable Version Control and Rollbacks
- ✅ **Feedback-Driven Decision Engine** (NEW)
- ✅ **Policy-Based Deployment Control** (NEW)
- ✅ **Canary Releases & Controlled Rollouts** (NEW)
- ✅ **Comprehensive Monitoring Service** (NEW)
- ✅ **Formal Reliability Modeling** (NEW)

## 🏗️ Project Structure

```
Devops_Project/
├── 📁 src/                    # Source code
│   ├── train_model.py         # Model training script
│   ├── validate_model.py      # Model validation
│   ├── model_api.py          # FastAPI serving with monitoring
│   ├── drift_detection.py    # Data drift detection
│   ├── reliability.py        # Reliability metrics (MTTR, failure rates)
│   ├── decision_engine.py    # Policy-based decision engine (NEW)
│   ├── monitoring_service.py  # Comprehensive monitoring (NEW)
│   ├── canary_deployment.py   # Canary releases (NEW)
│   ├── rollback.py           # Model rollback
│   ├── trigger_retraining.py # Automated retraining
│   ├── orchestrator.py      # Pipeline orchestrator (NEW)
│   └── utils.py              # Utility functions
├── 📁 models/                 # Trained models storage
├── 📁 data/                   # Dataset storage
├── 📁 config/                 # Configuration files
│   ├── config.json           # Main configuration
│   ├── deployment_policies.json  # Deployment policies (NEW)
│   └── canary_config.json    # Canary deployment config (NEW)
├── 📁 docker/                 # Docker configurations
│   ├── Dockerfile            # Model serving container
│   ├── docker-compose.yml    # Multi-service setup
│   ├── prometheus.yml        # Prometheus config
│   └── grafana/              # Grafana dashboards (NEW)
│       ├── provisioning/     # Auto-provisioning configs
│       └── dashboards/       # Dashboard definitions
├── 📁 ci-cd/                  # CI/CD pipeline configs
│   ├── Jenkinsfile           # Jenkins pipeline
│   ├── github-actions.yml    # GitHub Actions workflow
│   └── deploy.ps1            # Deployment scripts
├── 📁 tests/                  # Test suite
│   ├── test_model.py         # Model tests
│   ├── test_api.py           # API tests
│   └── test_integration.py   # Integration tests
├── requirements.txt           # Python dependencies
├── setup.ps1                 # Windows setup script
├── setup.sh                  # Linux/Mac setup script
├── paper.md                  # Research paper
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Docker (for containerization)
- PowerShell (Windows) or Bash (Linux/Mac)

### Setup Instructions

#### Windows (PowerShell)
```powershell
# Clone or navigate to project directory
cd "C:\Users\banda\OneDrive\Desktop\Devops_Project"

# Run setup script
.\setup.ps1 -GitUserName "Your Name" -GitUserEmail "your@email.com"

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Linux/Mac (Bash)
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

## 📊 Project Phases

### Phase 1: Planning and Environment Setup ✅
- [x] Project structure created
- [x] Dependencies defined
- [x] Setup scripts prepared

### Phase 2: Model Development and Packaging
```bash
python src/train_model.py      # Train the model
python src/validate_model.py   # Validate model performance
```

### Phase 3: Building the CI/CD Pipeline
```bash
# Build Docker image
docker build -f docker/Dockerfile -t ml-model-api .

# Run containerized model
docker-compose -f docker/docker-compose.yml up
```

### Phase 4: Model Serving
```bash
python src/model_api.py        # Start API server
```

### Phase 5: Monitoring and Logging
```bash
# Start monitoring stack (Prometheus + Grafana)
docker-compose -f docker/docker-compose.yml up

# Access dashboards:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
# - API Docs: http://localhost:8000/docs
```

### Phase 6: Feedback-Driven MLOps (NEW!)
```bash
# Start API with integrated monitoring and decision engine
python src/model_api.py

# The system automatically:
# - Monitors metrics in real-time
# - Detects data drift
# - Makes decisions using policy engine
# - Triggers retraining or rollback as needed

# View monitoring metrics
curl http://localhost:8000/monitoring/metrics
curl http://localhost:8000/monitoring/summary
curl http://localhost:8000/decision/history
```

## 🎯 New Features (Based on Paper Implementation)

### 1. Policy-Based Decision Engine
The decision engine implements the closed-loop control system described in the paper:
- Maps monitoring signals (M_t) and drift indicators (D_t) to deployment actions
- Configurable policies for retraining, rollback, and redeployment
- Formal decision function: A_t = f(M_t, D_t, Π)

**Usage:**
```python
from src.decision_engine import PolicyEngine

engine = PolicyEngine()
decision = engine.decide_action(metrics=metrics, drift_results=drift_results)
print(f"Action: {decision['action']}")
```

### 2. Comprehensive Monitoring Service
Continuous monitoring of infrastructure and model-level metrics:
- Real-time metric collection (M_t = {m_1(t), m_2(t), ..., m_n(t)})
- Aggregation and storage of metrics
- Integration with Prometheus and Grafana

**Usage:**
```python
from src.monitoring_service import get_monitoring_service

monitoring = get_monitoring_service()
monitoring.start_monitoring()
metrics = monitoring.get_metrics_summary()
```

### 3. Canary Deployment & Controlled Rollouts
Gradual traffic routing for safe model deployments:
- Percentage-based traffic splitting
- Automatic evaluation against success thresholds
- Auto-promotion or rollback based on metrics

**Usage:**
```python
from src.canary_deployment import CanaryDeployment

canary = CanaryDeployment()
canary.start_canary(model_version="1.0.1", model_path="...", metadata_path="...")
evaluation = canary.evaluate_canary()
```

### 4. Formal Reliability Modeling
Implementation of reliability equations from the paper:
- P_success = 1 - (P_test + P_deploy + P_runtime)
- MTTR = (1/N) * Σ t_recovery^(i)
- Failure rate calculations

**Usage:**
```python
from src.reliability import ReliabilityTracker

tracker = ReliabilityTracker()
metrics = tracker.calculate_metrics()
reliability = tracker.calculate_deployment_reliability()
```

### 5. Integrated Pipeline Orchestrator
Complete pipeline management:
```bash
# Run full pipeline
python src/orchestrator.py full --version 1.0.1

# Individual steps
python src/orchestrator.py train --version 1.0.1
python src/orchestrator.py validate
python src/orchestrator.py deploy
python src/orchestrator.py status
```

## 🎯 Quick Start: Deploy Your Model

### Step 1: Add Your Model

Place your trained ML model (`.pkl` file) in the `models/` folder:

```bash
cp your_model.pkl models/
```

### Step 2: Test and Deploy

Run the interactive deployment script:

```bash
python src/canary_deployment.py
```

**What happens:**
1. ✅ Lists all available models
2. ✅ You select your model
3. ✅ Runs comprehensive tests (structure, performance, latency, compatibility)
4. ✅ Shows deployment recommendation
5. ✅ Starts canary deployment if tests pass

### Step 3: Monitor Deployment

```bash
# Start API server
python src/model_api.py

# View monitoring dashboards
docker-compose -f docker/docker-compose.yml up
# Access: http://localhost:3000 (Grafana)
```

**📖 For detailed instructions, see [USER_GUIDE.md](USER_GUIDE.md)**

## 🛡️ Model Verification

We have streamlined the process for verifying external models (e.g., from Kaggle).

### 🚀 How to Verify a Model

**Step 1: Input Your Model**
Copy your trained model (e.g., `my_model.pkl`) into the `input_models/` folder.
*(Optional: Add `validation.csv` and `config.json` there for stricter checks).*

**Step 2: Run the Checker**
Run the readiness checker tool to analyze and verify the model without deploying it:
```powershell
.\check_model.ps1
```

**Step 3: View Results**
The tool will print a summary in the terminal. For full details, open the generated reports:
*   **Audit Report:** `reports/initial_audit.md` (What is this model?)
*   **Verdict Report:** `reports/final_verdict.md` (Is it ready?)

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Programming** | Python | Model development & APIs |
| **Containerization** | Docker | Model packaging |
| **CI/CD** | Jenkins, GitHub Actions | Automation pipeline |
| **API Framework** | Flask/FastAPI | Model serving |
| **Monitoring** | Prometheus + Grafana | Performance tracking |
| **Version Control** | Git + DVC | Code & model versioning |
| **Testing** | Pytest | Quality assurance |

## 📈 Success Metrics

| Metric | Target | Description |
|--------|---------|-------------|
| 🔁 **Deployment Time** | < 10 minutes | Training to production |
| ✅ **Success Rate** | > 95% | Automated deployments |
| 🧪 **Validation Accuracy** | > 90% | Model quality checks |
| 🔍 **Monitoring Coverage** | 100% | Key metrics tracked |
| 🔄 **Rollback Speed** | < 2 minutes | Failure recovery |
| 📊 **Model Uptime** | > 99.5% | Service availability |

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_NAME=ml_model
MODEL_VERSION=1.0.0
MODEL_PATH=models/trained_model.pkl

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Cloud Configuration (Optional)
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-model-bucket
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_model.py      # Model tests
pytest tests/test_api.py        # API tests
pytest tests/test_integration.py # Integration tests
```

## 📝 API Documentation

Once the API is running, access documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")

# Model prediction
data = {"features": [1.2, 3.4, 5.6, 7.8]}
response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
```

## 🐳 Docker Usage

```bash
# Build the model API image
docker build -f docker/Dockerfile -t ml-model-api:latest .

# Run the container
docker run -p 8000:8000 ml-model-api:latest

# Using docker-compose for full stack
docker-compose up --build
```

## 📊 Monitoring

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔄 CI/CD Pipeline

The pipeline automatically:
1. 🧪 Runs tests on code changes
2. 🏗️ Builds Docker images
3. ✅ Validates model performance
4. 🚀 Deploys to staging/production
5. 📊 Monitors deployment health

## 📚 Next Steps

1. **Customize the Model**: Replace the example model in `src/train_model.py`
2. **Configure CI/CD**: Set up Jenkins or GitHub Actions
3. **Add Monitoring**: Configure Prometheus alerts
4. **Scale**: Deploy to Kubernetes for production
5. **Security**: Add authentication and HTTPS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues:
- 📧 Create an issue in this repository
- 💬 Contact the development team
- 📖 Check the documentation in `/docs`

---

**🎯 Goal**: Create a robust, automated ML deployment pipeline that scales with your needs!
