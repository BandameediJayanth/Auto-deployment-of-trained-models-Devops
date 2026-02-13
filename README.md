# 🏥 Breast Cancer Prediction API - MLOps Deployment

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning API for breast cancer prediction with complete MLOps pipeline, monitoring, and professional web interface.

![Dashboard Preview](docs/images/dashboard.png)

## 🌟 Features

### Core Functionality
- 🤖 **ML Model API** - Random Forest classifier for breast cancer prediction (95.6% accuracy)
- 🎨 **Professional Web Dashboard** - Interactive UI with real-time predictions and charts
- 📊 **Monitoring Stack** - Prometheus + Grafana for metrics and visualization
- 🐳 **Docker Deployment** - Multi-container setup with Docker Compose
- 🔄 **CI/CD Pipeline** - Automated testing and deployment with GitHub Actions
- 📚 **API Documentation** - Interactive Swagger UI and ReDoc

### Technical Highlights
- ⚡ **High Performance** - Async FastAPI with Redis caching
- 🔒 **Security** - Non-root containers, health checks, input validation
- 📈 **Scalability** - Nginx load balancer, horizontal scaling ready
- 🔍 **Observability** - Comprehensive metrics and logging
- 🧪 **Testing** - Automated validation and model testing

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local development)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ml-api-deployment.git
cd ml-api-deployment
```

### 2. Start with Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 3. Access the Services
- **ML API Dashboard**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8000/docs

## 📖 Documentation

- [Setup Guide](docs/SETUP.md) - Detailed installation instructions
- [API Documentation](docs/API.md) - API endpoints and usage
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Architecture](docs/ARCHITECTURE.md) - System architecture overview

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Nginx (Port 80)                      │
│                    Reverse Proxy / Load Balancer             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   ML API     │  │   ML API     │  │   ML API     │
│  (Port 8000) │  │  (Port 8001) │  │  (Port 8002) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Prometheus  │  │    Redis     │  │   Grafana    │
│  (Port 9090) │  │  (Port 6379) │  │ (Port 3000)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 🎯 Usage

### Web Interface

1. Visit http://localhost:8000
2. Click **"New Prediction"**
3. Load sample data or enter custom values
4. View results in the predictions table

### API Endpoints

#### Make a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 
                 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 
                 0.2654, 0.4601, 0.1189]
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

### Python Client Example
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 
                 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 
                 0.2654, 0.4601, 0.1189]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {max(result['probability']) * 100:.2f}%")
```

## 🛠️ Development

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_model.py

# Run the API
python src/model_api.py
```

### Run Tests
```bash
# Validate model
python src/validate_model.py

# Run with pytest (if configured)
pytest tests/
```

### Docker Development
```bash
# Build image
docker build -f docker/Dockerfile -t ml-model-api:latest .

# Run container
docker run -p 8000:8000 ml-model-api:latest

# View logs
docker logs ml-model-api
```

## 📊 Monitoring

### Prometheus Metrics
Access Prometheus at http://localhost:9090

**Key Metrics:**
- `model_api_requests_total` - Total API requests
- `model_api_predictions_total` - Total predictions made
- `model_api_request_duration_seconds` - Request latency
- `model_api_requests_in_progress` - Active requests

**Example Queries:**
```promql
# Request rate per second
rate(model_api_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(model_api_request_duration_seconds_bucket[5m]))

# API uptime
up{job="ml-model-api"}
```

### Grafana Dashboards
1. Access Grafana at http://localhost:3000
2. Login with `admin/admin123`
3. Add Prometheus data source: `http://ml-prometheus:9090`
4. Import pre-built dashboards from `grafana/dashboards/`

## 🔧 Configuration

### Environment Variables
```bash
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
MODEL_PATH=models/breast_cancer_model.pkl
```

### Docker Compose
Edit `docker/docker-compose.yml` to customize:
- Port mappings
- Resource limits
- Volume mounts
- Environment variables

## 🚢 Deployment

### Production Deployment

**Option 1: Docker Compose**
```bash
docker-compose -f docker/docker-compose.yml up -d
```

**Option 2: Kubernetes**
```bash
kubectl apply -f k8s/
```

**Option 3: Cloud Platforms**
- AWS ECS/EKS
- Google Cloud Run/GKE
- Azure Container Instances/AKS

See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.

## 📁 Project Structure

```
.
├── src/
│   ├── model_api.py          # FastAPI application
│   ├── train_model.py        # Model training script
│   └── validate_model.py     # Model validation
├── static/
│   └── index.html            # Web dashboard
├── models/                   # Trained models
├── docker/
│   ├── Dockerfile            # Multi-stage build
│   ├── docker-compose.yml    # Service orchestration
│   ├── prometheus.yml        # Prometheus config
│   └── nginx.conf            # Nginx config
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml   # CI/CD pipeline
├── data/                     # Training data
├── tests/                    # Test files
└── requirements.txt          # Python dependencies
```

## 🧪 Model Details

- **Algorithm**: Random Forest Classifier
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Features**: 30 numerical features
- **Accuracy**: ~95.6%
- **Classes**: Benign (0) / Malignant (1)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Scikit-learn for ML capabilities
- Prometheus & Grafana for monitoring
- Docker for containerization

## 📧 Contact

**Project Maintainer**: Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🔬 Reproducibility

This project is designed to be fully reproducible. Follow these steps to replicate the results:

### **Prerequisites**
- Python 3.9+
- Docker & Docker Compose
- Git
- 4GB+ RAM

### **Step-by-Step Reproduction**

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/ml-api-deployment.git
cd ml-api-deployment
```

**2. Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train the model:**
```bash
python src/train_model.py
```
Expected output:
- Model file: `models/breast_cancer_model.pkl`
- Accuracy: ~95.6%
- Training time: <30 seconds

**4. Validate the model:**
```bash
python src/validate_model.py
```
Expected metrics:
- Accuracy: 95.61%
- F1 Score: 0.9561
- Precision: 95.65%
- Recall: 95.61%

**5. Generate visualizations (for research paper):**
```bash
python src/generate_visualizations.py
```
Output: Confusion matrix, ROC curve, PR curve in `reports/`

**6. Start the API (Docker):**
```bash
docker-compose -f docker/docker-compose.yml up -d
```

**7. Verify deployment:**
```bash
# Health check
curl http://localhost:8000/health

# Make a test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

### **Expected Results**

| Component | Expected Outcome |
|-----------|------------------|
| Model Training | Accuracy: 95.61% ± 0.5% |
| API Response Time | <100ms (p95) |
| Docker Build | Success in <2 minutes |
| Container Startup | <10 seconds |
| Health Check | Status: healthy |

### **Dataset Information**

- **Source**: Scikit-learn's built-in Breast Cancer Wisconsin dataset
- **Samples**: 569 total (455 training, 114 testing)
- **Features**: 30 numerical features
- **Classes**: Binary (Benign/Malignant)
- **Split**: 80/20 train/test, stratified, random_state=42

### **Reproducibility Checklist**

- ✅ Fixed random seeds (`random_state=42`)
- ✅ Pinned dependencies in `requirements.txt`
- ✅ Documented all hyperparameters
- ✅ Included dataset source
- ✅ Docker ensures environment consistency
- ✅ CI/CD pipeline validates reproducibility

### **Troubleshooting**

**Issue**: Different accuracy results
- **Solution**: Ensure scikit-learn version matches `requirements.txt`
- **Note**: Minor variations (<0.5%) are normal due to system differences

**Issue**: Docker build fails
- **Solution**: Ensure Docker has sufficient resources (4GB+ RAM)

**Issue**: API returns 500 errors
- **Solution**: Check model file exists: `ls -la models/`

### **Citation**

If you use this project in your research, please cite:

```bibtex
@misc{ml-api-deployment-2026,
  author = {Your Name},
  title = {Production-Ready ML API Deployment with MLOps},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ml-api-deployment}
}
```

### **Research Paper Artifacts**

For academic use, all research artifacts are available:
- **Visualizations**: `reports/` directory
- **Model Metrics**: `models/*_metadata.json`
- **Version History**: `docs/MODEL_VERSIONS.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Deployment Logs**: Available via `docker logs`



## 🔗 Links

- [Live Demo](https://your-demo-url.com) (if available)
- [Documentation](https://docs.your-project.com)
- [Issue Tracker](https://github.com/yourusername/ml-api-deployment/issues)

---

**⭐ If you find this project useful, please consider giving it a star!**
