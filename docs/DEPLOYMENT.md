# Deployment Guide

## Overview

This guide covers deploying the ML API to various environments.

## Prerequisites

- Docker & Docker Compose installed
- Access to deployment environment
- Model files trained and validated

## Local Deployment

### Using Docker Compose

**1. Build and start all services:**
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

**2. Verify services are running:**
```bash
docker ps
```

**3. Access the services:**
- ML API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

**4. Stop services:**
```bash
docker-compose -f docker/docker-compose.yml down
```

## Cloud Deployment

### AWS Deployment

#### Option 1: ECS (Elastic Container Service)

**1. Push image to ECR:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag ml-model-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-model-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-model-api:latest
```

**2. Create ECS task definition and service**
**3. Configure load balancer**
**4. Set up auto-scaling**

#### Option 2: EKS (Elastic Kubernetes Service)

```bash
# Create EKS cluster
eksctl create cluster --name ml-api-cluster --region us-east-1

# Deploy application
kubectl apply -f k8s/

# Expose service
kubectl expose deployment ml-api --type=LoadBalancer --port=8000
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-model-api

# Deploy to Cloud Run
gcloud run deploy ml-api \
  --image gcr.io/PROJECT_ID/ml-model-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### GKE (Google Kubernetes Engine)

```bash
# Create GKE cluster
gcloud container clusters create ml-api-cluster --num-nodes=3

# Deploy application
kubectl apply -f k8s/

# Get external IP
kubectl get service ml-api
```

### Azure

#### Azure Container Instances

```bash
# Create resource group
az group create --name ml-api-rg --location eastus

# Create container
az container create \
  --resource-group ml-api-rg \
  --name ml-api \
  --image ml-model-api:latest \
  --dns-name-label ml-api-demo \
  --ports 8000
```

#### AKS (Azure Kubernetes Service)

```bash
# Create AKS cluster
az aks create \
  --resource-group ml-api-rg \
  --name ml-api-cluster \
  --node-count 3 \
  --enable-addons monitoring

# Deploy application
kubectl apply -f k8s/
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (minikube, EKS, GKE, AKS)
- kubectl configured
- Docker images pushed to registry

### Deploy

**1. Update image in k8s manifests:**
```yaml
# k8s/deployment.yaml
spec:
  containers:
  - name: ml-api
    image: your-registry/ml-model-api:latest
```

**2. Apply manifests:**
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

**3. Verify deployment:**
```bash
kubectl get pods -n ml-api
kubectl get services -n ml-api
```

**4. Access the application:**
```bash
# Get external IP
kubectl get service ml-api -n ml-api

# Or use port-forward for testing
kubectl port-forward service/ml-api 8000:8000 -n ml-api
```

## Production Considerations

### Security

**1. Enable HTTPS:**
- Use Let's Encrypt for SSL certificates
- Configure nginx with SSL
- Redirect HTTP to HTTPS

**2. Authentication:**
- Implement API key authentication
- Use JWT tokens for user sessions
- Consider OAuth 2.0 for third-party access

**3. Network Security:**
- Use VPC/VNet for isolation
- Configure security groups/firewall rules
- Enable DDoS protection

### Monitoring

**1. Set up alerts:**
```yaml
# prometheus/alerts.yml
groups:
  - name: ml_api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(model_api_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

**2. Configure Grafana dashboards**
**3. Set up log aggregation (ELK, CloudWatch, Stackdriver)**

### Scaling

**1. Horizontal Pod Autoscaler (Kubernetes):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**2. Load Balancer Configuration:**
- Use cloud provider load balancers
- Configure health checks
- Set up SSL termination

### Backup & Recovery

**1. Model Backup:**
```bash
# Backup models to S3/GCS/Azure Blob
aws s3 sync models/ s3://ml-models-backup/
```

**2. Database Backup (if using):**
- Automated daily backups
- Point-in-time recovery
- Cross-region replication

**3. Disaster Recovery Plan:**
- Document recovery procedures
- Test recovery regularly
- Maintain backup deployment in different region

## Environment Variables

### Required
```bash
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/breast_cancer_model.pkl
```

### Optional
```bash
DEBUG=false
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## Health Checks

Configure health checks for your deployment:

**Liveness Probe:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Readiness Probe:**
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs ml-model-api

# Check container status
docker inspect ml-model-api
```

### API Returns 500 Errors
- Check model file exists
- Verify model path in environment variables
- Check application logs

### High Memory Usage
- Reduce model size
- Implement model caching
- Scale horizontally

## Rollback Procedure

**Docker Compose:**
```bash
# Stop current version
docker-compose -f docker/docker-compose.yml down

# Deploy previous version
docker-compose -f docker/docker-compose.yml up -d
```

**Kubernetes:**
```bash
# Rollback to previous deployment
kubectl rollout undo deployment/ml-api -n ml-api

# Check rollout status
kubectl rollout status deployment/ml-api -n ml-api
```

## CI/CD Integration

The project includes GitHub Actions workflow for automated deployment:

**Workflow triggers:**
- Push to main branch
- Pull request to main
- Manual trigger

**Workflow steps:**
1. Run tests
2. Build Docker image
3. Push to registry
4. Deploy to staging
5. Run integration tests
6. Deploy to production (manual approval)

## Support

For deployment issues:
1. Check logs first
2. Review documentation
3. Open an issue on GitHub
4. Contact maintainers
