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

## 🏗️ Project Structure

```
Devops_Project/
├── 📁 src/                    # Source code
│   ├── train_model.py         # Model training script
│   ├── validate_model.py      # Model validation
│   ├── model_api.py          # Flask/FastAPI serving
│   └── utils.py              # Utility functions
├── 📁 models/                 # Trained models storage
├── 📁 data/                   # Dataset storage
├── 📁 docker/                 # Docker configurations
│   ├── Dockerfile            # Model serving container
│   └── docker-compose.yml    # Multi-service setup
├── 📁 ci-cd/                  # CI/CD pipeline configs
│   ├── Jenkinsfile           # Jenkins pipeline
│   ├── github-actions.yml    # GitHub Actions workflow
│   └── deploy.sh             # Deployment scripts
├── 📁 monitoring/             # Monitoring & logging
│   ├── prometheus.yml        # Prometheus config
│   ├── grafana/              # Grafana dashboards
│   └── logging_config.py     # Logging setup
├── 📁 tests/                  # Test suite
│   ├── test_model.py         # Model tests
│   ├── test_api.py           # API tests
│   └── test_integration.py   # Integration tests
├── requirements.txt           # Python dependencies
├── setup.ps1                 # Windows setup script
├── setup.sh                  # Linux/Mac setup script
├── .gitignore                # Git ignore rules
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
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up
```

## 🛡️ Model Verification (New!)

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
