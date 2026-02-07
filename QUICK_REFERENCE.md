# Quick Reference Guide

A handy reference for common commands and workflows in the Auto-Deployment ML Models project.

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/BandameediJayanth/Aegis_Code_AI.git
cd Aegis_Code_AI

# Windows Setup
.\ci-cd\setup.ps1

# Linux/Mac Setup
chmod +x ci-cd/setup.sh
./ci-cd/setup.sh

# Activate environment
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🤖 Model Training & Validation

```bash
# Train a new model
python src/train_model.py --model-name my_model --version 1.0.0

# Train with custom parameters
python src/train_model.py --model-name my_model --version 1.0.0 --n-estimators 200

# Validate a model
python src/validate_model.py --model models/my_model_v1.0.0.pkl

# Make a prediction
python src/predict.py --model models/my_model_v1.0.0.pkl --features 1.0 2.0 3.0 4.0 5.0
```

## 📊 Model Analysis & Verification

```bash
# Analyze an external model (e.g., from Kaggle)
python src/analyze_model.py --model input_models/external_model.pkl

# Verify deployment readiness
python src/verify_readiness.py --model models/my_model_v1.0.0.pkl

# Promote model to production
python src/cleanup_and_promote.py --model models/my_model_v1.0.0.pkl
```

## 🔄 Drift Detection & Maintenance

```bash
# Check for data drift
python src/drift_detection.py --reference-data data/reference.csv --current-data data/current.csv

# Trigger retraining (usually automated)
python src/trigger_retraining.py

# Rollback to previous model
python src/rollback.py
```

## 🌐 API Server

```bash
# Start the API server
python src/model_api.py

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'

# With PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}' -ContentType "application/json"
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -v

# Run and stop on first failure
pytest tests/ -x

# Run tests matching pattern
pytest tests/ -k "test_train"
```

## 🐳 Docker

```bash
# Build Docker image
docker build -f docker/Dockerfile -t ml-model-api:latest .

# Run container
docker run -p 8000:8000 ml-model-api:latest

# Using docker-compose
docker-compose -f docker/docker-compose.yml up --build

# Stop containers
docker-compose -f docker/docker-compose.yml down

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

## 📈 Monitoring

```bash
# Check reliability metrics
python src/reliability.py

# View Prometheus metrics (after starting monitoring stack)
# Open browser: http://localhost:9090

# View Grafana dashboards
# Open browser: http://localhost:3000
# Default credentials: admin / admin
```

## 🔧 CI/CD

```bash
# Run deployment pipeline (PowerShell)
.\ci-cd\deploy.ps1

# Run Jenkins pipeline (if Jenkins is set up)
# Configure Jenkins to use ci-cd/Jenkinsfile

# GitHub Actions runs automatically on push to main/develop
```

## 📁 File Management

```bash
# Clean up old models
# (This is done automatically by cleanup_and_promote.py)

# Create a backup
# Windows
Copy-Item models\ models_backup\ -Recurse
# Linux/Mac
cp -r models/ models_backup/

# View model history
cat models/model_history.json

# View reliability events
cat models/reliability_events.json
```

## 🔍 Debugging

```bash
# Check Python environment
python --version
pip list

# View logs
# Windows
Get-Content logs\training.log -Tail 50
Get-Content logs\api_server.log -Tail 50

# Linux/Mac
tail -f logs/training.log
tail -f logs/api_server.log

# Check model metadata
cat models/my_model_v1.0.0_metadata.json

# Validate Python syntax
flake8 src/

# Format code
black src/

# Type checking
mypy src/
```

## 🌿 Git Workflow

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Stage changes
git add .

# Commit with conventional message
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update README"

# Push to remote
git push origin feature/my-feature

# Update from main
git fetch origin
git rebase origin/main

# Clean up local branches
git branch -d feature/old-feature
```

## 📊 Common Issues & Solutions

### Issue: Model not found
```bash
# Check if model exists
dir models\  # Windows
ls models/   # Linux/Mac

# Verify model path in latest_model.json
cat models/latest_model.json
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: Port already in use
```bash
# Windows - find process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Issue: Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -f docker/Dockerfile -t ml-model-api:latest .
```

## 📚 Useful Resources

- **Documentation**: `/docs` directory
- **Examples**: `/tests` directory for usage examples
- **API Docs**: http://localhost:8000/docs (when server is running)
- **GitHub**: https://github.com/BandameediJayanth/Aegis_Code_AI
- **Issues**: Report issues on GitHub

## 💡 Pro Tips

1. **Use virtual environments**: Always work within a virtual environment
2. **Run tests before committing**: `pytest tests/`
3. **Check code quality**: Use flake8 and black
4. **Read the logs**: Most issues can be diagnosed from log files
5. **Keep dependencies updated**: Regularly run `pip list --outdated`
6. **Backup models**: Before major changes, backup the models/ directory
7. **Monitor drift**: Regularly check drift_detection logs
8. **Document changes**: Update CHANGELOG.md for significant changes

---

**Need more help?** Check [README.md](README.md) or [CONTRIBUTING.md](CONTRIBUTING.md)
