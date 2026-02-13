# Setup Guide

## Quick Start

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose
- Git

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/ml-api-deployment.git
cd ml-api-deployment
```

**2. Choose your setup method:**

#### Option A: Docker (Recommended)
```bash
docker-compose -f docker/docker-compose.yml up -d
```

#### Option B: Local Development
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_model.py

# Start the API
python src/model_api.py
```

**3. Verify installation:**
- Visit http://localhost:8000
- You should see the ML API dashboard

## Detailed Setup

### 1. System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB free space
- OS: Windows 10+, Ubuntu 20.04+, macOS 10.15+

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 20+ GB free space
- SSD for better performance

### 2. Install Dependencies

#### Python Dependencies
```bash
pip install -r requirements.txt
```

**Main dependencies:**
- FastAPI - Web framework
- Uvicorn - ASGI server
- Scikit-learn - ML library
- Pandas - Data manipulation
- Prometheus-client - Metrics
- Pydantic - Data validation

#### Docker Setup

**Windows:**
1. Download Docker Desktop from docker.com
2. Install and restart
3. Verify: `docker --version`

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify
docker --version
docker-compose --version
```

**macOS:**
1. Download Docker Desktop from docker.com
2. Install and start
3. Verify: `docker --version`

### 3. Configuration

#### Environment Variables

Create a `.env` file (optional):
```bash
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
MODEL_PATH=models/breast_cancer_model.pkl
LOG_LEVEL=INFO
```

#### Docker Compose Configuration

Edit `docker/docker-compose.yml` to customize:
- Port mappings
- Resource limits
- Volume mounts
- Environment variables

### 4. Training the Model

**Run training script:**
```bash
python src/train_model.py
```

**Output:**
- Model file: `models/breast_cancer_model.pkl`
- Metadata: `models/breast_cancer_model_metadata.json`
- Training log: `training.log`

**Validation:**
```bash
python src/validate_model.py
```

### 5. Starting Services

#### Docker Compose (All Services)
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop all services
docker-compose -f docker/docker-compose.yml down
```

#### Individual Services

**ML API only:**
```bash
python src/model_api.py
```

**With custom port:**
```bash
python src/model_api.py --port 8080
```

**With specific model:**
```bash
python src/model_api.py --model models/breast_cancer_model.pkl
```

### 6. Accessing Services

Once started, access:
- **ML API Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/metrics

### 7. Verification

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

**Check Docker containers:**
```bash
docker ps
```

You should see:
- ml-model-api
- ml-prometheus
- ml-grafana
- ml-nginx
- ml-redis

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

**2. Docker Permission Denied (Linux)**
```bash
sudo usermod -aG docker $USER
# Logout and login again
```

**3. Model Not Found**
```bash
# Ensure model is trained
python src/train_model.py

# Check model file exists
ls -la models/
```

**4. Container Won't Start**
```bash
# Check logs
docker logs ml-model-api

# Rebuild image
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
```

**5. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. Check documentation in `docs/`
2. Review logs: `docker logs ml-model-api`
3. Open an issue on GitHub
4. Check existing issues for solutions

## Next Steps

After setup:
1. ✅ Test predictions via web UI
2. ✅ Set up Grafana dashboards
3. ✅ Review API documentation
4. ✅ Configure monitoring alerts
5. ✅ Plan production deployment

## Development Setup

For development:

**1. Install dev dependencies:**
```bash
pip install -r requirements-dev.txt  # if exists
```

**2. Enable auto-reload:**
```bash
python src/model_api.py --reload
```

**3. Run tests:**
```bash
pytest tests/
```

**4. Code formatting:**
```bash
black src/
flake8 src/
```

## Production Setup

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment instructions.

## Updates

**Update code:**
```bash
git pull origin main
docker-compose -f docker/docker-compose.yml up -d --build
```

**Update dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

## Uninstall

**Remove Docker containers:**
```bash
docker-compose -f docker/docker-compose.yml down -v
```

**Remove virtual environment:**
```bash
# Windows:
rmdir /s venv
# Linux/Mac:
rm -rf venv
```

**Remove project:**
```bash
cd ..
rm -rf ml-api-deployment
```
