# Architecture

## System Overview

The ML API Deployment system is a production-ready, containerized machine learning service with comprehensive monitoring, automated deployment, and operational intelligence.

## High-Level Architecture

![System Architecture](../C:/Users/banda/.gemini/antigravity/brain/8f3038db-c229-43bc-abf0-d8fcfebbdb09/architecture_diagram.png)

## Component Architecture

### 1. Application Layer

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                     │
│                   (Web Browser / API Clients)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Nginx Load Balancer                       │
│                      (Port 80/443)                           │
│  • SSL Termination                                           │
│  • Load Distribution                                         │
│  • Request Routing                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   ML API     │  │   ML API     │  │   ML API     │
│  Instance 1  │  │  Instance 2  │  │  Instance 3  │
│ (Port 8000)  │  │ (Port 8001)  │  │ (Port 8002)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │    Redis    │
                  │   Cache     │
                  │ (Port 6379) │
                  └─────────────┘
```

### 2. ML API Service

**Technology Stack:**
- **Framework**: FastAPI (async Python web framework)
- **Model**: Scikit-learn Random Forest Classifier
- **Server**: Uvicorn (ASGI server)
- **Metrics**: Prometheus client

**Key Features:**
- RESTful API endpoints
- Async request handling
- Model versioning
- Health checks
- Metrics export

**Endpoints:**
- `GET /` - Web dashboard
- `POST /predict` - Make predictions
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation

### 3. Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Monitoring Layer                        │
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Prometheus  │ ◄────── │   ML API     │                  │
│  │  (Port 9090) │         │   Metrics    │                  │
│  │              │         │   Exporter   │                  │
│  │ • Metrics DB │         └──────────────┘                  │
│  │ • Alerting   │                                            │
│  │ • Queries    │                                            │
│  └──────┬───────┘                                            │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐                                            │
│  │   Grafana    │                                            │
│  │  (Port 3000) │                                            │
│  │              │                                            │
│  │ • Dashboards │                                            │
│  │ • Alerts     │                                            │
│  │ • Analytics  │                                            │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

**Metrics Collected:**
- Request count and rate
- Prediction count
- Request duration (histogram)
- Active requests
- Model load time
- Error rates
- System resources

### 4. CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Repository                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Actions Workflow                    │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │   Test   │ → │  Train   │ → │  Build   │ → │  Deploy  │ │
│  │          │   │  Model   │   │  Docker  │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                               │
│  • Code quality checks                                       │
│  • Unit tests                                                │
│  • Model validation                                          │
│  • Docker image build                                        │
│  • Container registry push                                   │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline Stages:**
1. **Test**: Run unit tests, validate code
2. **Train**: Train ML model, validate performance
3. **Build**: Build Docker images
4. **Deploy**: Push to registry, deploy to environment

### 5. Data Flow

```
User Request
    │
    ▼
Nginx (Load Balancer)
    │
    ▼
ML API Instance
    │
    ├─► Check Redis Cache ──► Cache Hit? ──► Return Result
    │                              │
    │                              ▼ (Cache Miss)
    │                         Load Model
    │                              │
    │                              ▼
    │                      Make Prediction
    │                              │
    │                              ▼
    │                      Store in Cache
    │                              │
    │                              ▼
    └─────────────────────► Return Result
                                   │
                                   ▼
                          Update Metrics
                                   │
                                   ▼
                            Prometheus Scrape
                                   │
                                   ▼
                            Grafana Display
```

## Deployment Architecture

### Docker Compose Setup

```yaml
Services:
  - ml-api:        ML Model API (FastAPI)
  - prometheus:    Metrics collection
  - grafana:       Monitoring dashboards
  - nginx:         Load balancer
  - redis:         Caching layer

Networks:
  - ml-network:    Internal bridge network

Volumes:
  - prometheus_data:  Persistent metrics storage
  - grafana_data:     Dashboard configurations
  - redis_data:       Cache persistence
```

### Container Details

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| ml-api | ml-model-api:latest | 8000 | ML predictions |
| prometheus | prom/prometheus:latest | 9090 | Metrics storage |
| grafana | grafana/grafana:latest | 3000 | Visualization |
| nginx | nginx:alpine | 80, 443 | Load balancing |
| redis | redis:7-alpine | 6379 | Caching |

## Security Architecture

### Container Security
- Non-root user execution
- Read-only file systems where possible
- Resource limits (CPU, memory)
- Health checks
- Minimal base images

### Network Security
- Internal bridge network
- No direct external access to services
- Nginx as single entry point
- TLS/SSL termination at load balancer

### API Security
- Input validation (Pydantic models)
- Request size limits
- Rate limiting (planned)
- CORS configuration

## Scalability

### Horizontal Scaling
- Multiple ML API instances behind Nginx
- Stateless API design
- Shared Redis cache
- Load balancer distribution

### Vertical Scaling
- Resource limits configurable
- Auto-scaling based on metrics
- Kubernetes HPA support

## Monitoring & Observability

### Metrics
- **Application**: Request rate, latency, errors
- **Business**: Predictions made, model accuracy
- **Infrastructure**: CPU, memory, disk

### Logging
- Structured JSON logs
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Centralized log aggregation (planned)

### Alerting
- Prometheus alert rules
- Grafana notifications
- Email/Slack integration (planned)

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| **ML Framework** | Scikit-learn |
| **API Framework** | FastAPI |
| **Server** | Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes (optional) |
| **Load Balancer** | Nginx |
| **Caching** | Redis |
| **Monitoring** | Prometheus |
| **Visualization** | Grafana |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.9+ |

## Design Principles

1. **Microservices**: Separate concerns into independent services
2. **Containerization**: All services run in Docker containers
3. **Observability**: Comprehensive metrics and monitoring
4. **Automation**: CI/CD pipeline for deployment
5. **Scalability**: Horizontal scaling capability
6. **Reliability**: Health checks, auto-restart, rollback
7. **Security**: Non-root containers, input validation
8. **Performance**: Async API, caching, load balancing

## Future Enhancements

- Model versioning and A/B testing
- Drift detection and auto-retraining
- Advanced alerting and anomaly detection
- Multi-model serving
- Authentication and authorization
- Rate limiting and throttling
- Distributed tracing
- Log aggregation (ELK stack)
