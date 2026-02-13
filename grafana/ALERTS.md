# Grafana Alert Rules Configuration

This directory contains Grafana alert configurations for monitoring the ML API.

## Alert Rules

### 1. High Error Rate Alert

**Trigger**: When error rate exceeds 5% over 5 minutes

**Severity**: Critical

**Configuration**:
```yaml
alert: HighErrorRate
expr: rate(model_api_requests_total{status="error"}[5m]) > 0.05
for: 5m
labels:
  severity: critical
annotations:
  summary: "High error rate detected in ML API"
  description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
```

### 2. High Latency Alert

**Trigger**: When p95 latency exceeds 500ms

**Severity**: Warning

**Configuration**:
```yaml
alert: HighLatency
expr: histogram_quantile(0.95, rate(model_api_request_duration_seconds_bucket[5m])) > 0.5
for: 5m
labels:
  severity: warning
annotations:
  summary: "High API latency detected"
  description: "P95 latency is {{ $value }}s (threshold: 0.5s)"
```

### 3. API Down Alert

**Trigger**: When API is unreachable

**Severity**: Critical

**Configuration**:
```yaml
alert: APIDown
expr: up{job="ml-model-api"} == 0
for: 1m
labels:
  severity: critical
annotations:
  summary: "ML API is down"
  description: "ML API has been unreachable for more than 1 minute"
```

### 4. High Memory Usage Alert

**Trigger**: When memory usage exceeds 80%

**Severity**: Warning

**Configuration**:
```yaml
alert: HighMemoryUsage
expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.8
for: 5m
labels:
  severity: warning
annotations:
  summary: "High memory usage in ML API container"
  description: "Memory usage is {{ $value | humanizePercentage }}"
```

### 5. Low Prediction Rate Alert

**Trigger**: When prediction rate drops significantly

**Severity**: Warning

**Configuration**:
```yaml
alert: LowPredictionRate
expr: rate(model_api_predictions_total[5m]) < 0.1
for: 10m
labels:
  severity: warning
annotations:
  summary: "Low prediction rate detected"
  description: "Prediction rate is {{ $value }} predictions/sec"
```

## Setting Up Alerts in Grafana

### Step 1: Access Grafana
1. Go to http://localhost:3000
2. Login with `admin` / `admin123`

### Step 2: Create Contact Point
1. Navigate to **Alerting** → **Contact points**
2. Click **New contact point**
3. Configure your notification channel:
   - **Email**: Enter email address
   - **Slack**: Enter webhook URL
   - **PagerDuty**: Enter integration key

### Step 3: Create Alert Rules

**For High Error Rate:**
1. Go to **Alerting** → **Alert rules**
2. Click **New alert rule**
3. Set name: `High Error Rate`
4. Add query:
   ```promql
   rate(model_api_requests_total{status="error"}[5m]) > 0.05
   ```
5. Set condition: `WHEN last() OF query(A) IS ABOVE 0.05`
6. Set evaluation: `For: 5m`
7. Add labels: `severity=critical`
8. Save

**For High Latency:**
1. Create new alert rule
2. Set name: `High Latency`
3. Add query:
   ```promql
   histogram_quantile(0.95, rate(model_api_request_duration_seconds_bucket[5m]))
   ```
4. Set condition: `WHEN last() OF query(A) IS ABOVE 0.5`
5. Set evaluation: `For: 5m`
6. Add labels: `severity=warning`
7. Save

**For API Down:**
1. Create new alert rule
2. Set name: `API Down`
3. Add query:
   ```promql
   up{job="ml-model-api"}
   ```
4. Set condition: `WHEN last() OF query(A) IS BELOW 1`
5. Set evaluation: `For: 1m`
6. Add labels: `severity=critical`
7. Save

### Step 4: Create Notification Policy
1. Go to **Alerting** → **Notification policies**
2. Edit default policy or create new
3. Set contact point
4. Configure routing:
   - Critical alerts → Immediate notification
   - Warning alerts → Grouped notification (5 min)

### Step 5: Test Alerts
1. Trigger a test alert
2. Verify notification received
3. Check alert history in Grafana

## Alert Thresholds

| Alert | Metric | Threshold | Duration | Severity |
|-------|--------|-----------|----------|----------|
| High Error Rate | Error rate | >5% | 5 min | Critical |
| High Latency | P95 latency | >500ms | 5 min | Warning |
| API Down | Uptime | 0 | 1 min | Critical |
| High Memory | Memory usage | >80% | 5 min | Warning |
| Low Prediction Rate | Predictions/sec | <0.1 | 10 min | Warning |

## Prometheus Alert Rules File

For Prometheus-native alerting, create `prometheus/alerts.yml`:

```yaml
groups:
  - name: ml_api_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(model_api_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in ML API"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(model_api_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "P95 latency is {{ $value }}s"

      - alert: APIDown
        expr: up{job="ml-model-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML API is down"
          description: "API has been unreachable for 1+ minute"
```

Then update `docker/prometheus.yml`:
```yaml
rule_files:
  - "/etc/prometheus/alerts.yml"
```

## Notification Templates

### Email Template
```
Subject: [{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}

Alert: {{ .GroupLabels.alertname }}
Severity: {{ .CommonLabels.severity }}
Status: {{ .Status }}

{{ range .Alerts }}
  Description: {{ .Annotations.description }}
  Started: {{ .StartsAt }}
{{ end }}
```

### Slack Template
```json
{
  "text": "🚨 Alert: {{ .GroupLabels.alertname }}",
  "attachments": [
    {
      "color": "{{ if eq .Status \"firing\" }}danger{{ else }}good{{ end }}",
      "title": "{{ .CommonAnnotations.summary }}",
      "text": "{{ .CommonAnnotations.description }}",
      "fields": [
        {
          "title": "Severity",
          "value": "{{ .CommonLabels.severity }}",
          "short": true
        },
        {
          "title": "Status",
          "value": "{{ .Status }}",
          "short": true
        }
      ]
    }
  ]
}
```

## Best Practices

1. **Start with critical alerts only** - Avoid alert fatigue
2. **Set appropriate thresholds** - Based on baseline metrics
3. **Use proper evaluation periods** - Avoid false positives
4. **Test alerts regularly** - Ensure notifications work
5. **Document runbooks** - What to do when alert fires
6. **Review and adjust** - Tune thresholds based on experience

## Runbooks

### High Error Rate Runbook
1. Check API logs: `docker logs ml-model-api`
2. Verify model is loaded correctly
3. Check recent deployments
4. Review error patterns in Grafana
5. Rollback if necessary

### High Latency Runbook
1. Check system resources
2. Review concurrent requests
3. Check database/cache performance
4. Scale horizontally if needed
5. Optimize slow endpoints

### API Down Runbook
1. Check container status: `docker ps`
2. Restart container if needed
3. Check logs for crash reason
4. Verify health check endpoint
5. Escalate if persistent

---

**Last Updated**: 2026-02-13
