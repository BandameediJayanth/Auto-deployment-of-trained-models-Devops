# API Documentation

## Overview

The ML Model API provides RESTful endpoints for breast cancer prediction using a trained Random Forest classifier.

**Base URL**: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## Endpoints

### Health Check

Check if the API is running and healthy.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "breast_cancer_model",
  "model_version": "1.0.0",
  "timestamp": "2026-02-13T12:00:00"
}
```

---

### Make Prediction

Submit features for breast cancer prediction.

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "features": [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189
  ]
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": [0.05, 0.95],
  "model_version": "1.0.0",
  "model_name": "breast_cancer_model",
  "timestamp": "2026-02-13T12:00:00"
}
```

**Prediction Values**:
- `0`: Benign
- `1`: Malignant

**Probability Array**:
- Index 0: Probability of benign
- Index 1: Probability of malignant

---

### Model Information

Get detailed information about the loaded model.

**Endpoint**: `GET /model/info`

**Response**:
```json
{
  "name": "breast_cancer_model",
  "version": "1.0.0",
  "type": "RandomForestClassifier",
  "accuracy": 0.956,
  "features": 30,
  "training_date": "2026-02-13",
  "description": "Breast cancer prediction model"
}
```

---

### List Available Models

Get a list of all available models.

**Endpoint**: `GET /models`

**Response**:
```json
{
  "models": [
    {
      "name": "breast_cancer_model",
      "version": "1.0.0",
      "path": "models/breast_cancer_model.pkl",
      "accuracy": 0.956,
      "features": 30
    }
  ],
  "count": 1,
  "loaded_model": "models/breast_cancer_model.pkl"
}
```

---

### Load Specific Model

Load a specific model by path.

**Endpoint**: `POST /model/load`

**Request Body**:
```json
{
  "model_path": "models/breast_cancer_model.pkl"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model loaded successfully"
}
```

---

### Prometheus Metrics

Get Prometheus-formatted metrics for monitoring.

**Endpoint**: `GET /metrics`

**Response**: Plain text Prometheus metrics
```
# HELP model_api_requests_total Total number of API requests
# TYPE model_api_requests_total counter
model_api_requests_total{endpoint="/predict",method="POST"} 42.0

# HELP model_api_predictions_total Total number of predictions made
# TYPE model_api_predictions_total counter
model_api_predictions_total 42.0

# HELP model_api_request_duration_seconds Request duration in seconds
# TYPE model_api_request_duration_seconds histogram
model_api_request_duration_seconds_bucket{le="0.005"} 10.0
model_api_request_duration_seconds_bucket{le="0.01"} 25.0
...
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid input: Expected 30 features, got 10"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to load model"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Consider implementing rate limiting per IP
- Use Redis for distributed rate limiting
- Set appropriate limits based on your infrastructure

---

## Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'

# Get model info
curl http://localhost:8000/model/info
```

### Python

```python
import requests

# Make prediction
url = "http://localhost:8000/predict"
data = {
    "features": [
        17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
        0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
        0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
        0.2654, 0.4601, 0.1189
    ]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {'Malignant' if result['prediction'] == 1 else 'Benign'}")
print(f"Confidence: {max(result['probability']) * 100:.2f}%")
```

### JavaScript

```javascript
const url = 'http://localhost:8000/predict';
const data = {
  features: [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189
  ]
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log('Prediction:', result.prediction === 1 ? 'Malignant' : 'Benign');
    console.log('Confidence:', Math.max(...result.probability) * 100 + '%');
  });
```

---

## Interactive Documentation

For interactive API testing, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
