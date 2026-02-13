# Autonomous Drift Monitoring Service

## Overview

The drift monitoring service runs as a **separate Docker container** that continuously monitors for data drift and triggers retraining when needed.

## Architecture

```
docker-compose up → All services start (including drift-monitor)
docker-compose down → All services stop (including drift-monitor)
```

**Clean lifecycle management**: Automation is tied to application lifecycle.

## Service Configuration

### Docker Compose Service

```yaml
drift-monitor:
  build:
    context: ..
    dockerfile: docker/Dockerfile
  container_name: ml-drift-monitor
  command: python src/monitoring_service.py
  environment:
    - ENABLE_AUTOMATION=true           # Enable/disable automation
    - DRIFT_CHECK_INTERVAL=3600        # Check every hour
    - DRIFT_THRESHOLD=0.2              # Drift threshold
    - REFERENCE_DATA_PATH=data/dataset.csv
  volumes:
    - ../models:/app/models            # Read/write for model updates
    - ../data:/app/data:ro             # Read-only for reference data
    - ../logs:/app/logs                # Write for logging
  restart: unless-stopped
  depends_on:
    - ml-api
    - prometheus
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTOMATION` | `true` | Enable/disable drift monitoring |
| `DRIFT_CHECK_INTERVAL` | `3600` | Seconds between checks (1 hour) |
| `DRIFT_THRESHOLD` | `0.2` | PSI threshold for triggering retraining |
| `REFERENCE_DATA_PATH` | `data/dataset.csv` | Path to reference dataset |

## How It Works

### 1. Continuous Monitoring Loop

```python
while True:
    check_drift()
    if drift_detected:
        trigger_retraining()
    sleep(check_interval)
```

### 2. Drift Detection

- Loads reference data
- Compares with recent production data
- Calculates drift metrics (PSI, KL, KS)
- Checks against threshold

### 3. Auto-Retraining

If drift exceeds threshold:
1. Increment model version (v1.0.0 → v1.1.0)
2. Train new model
3. Validate performance
4. Deploy if acceptable
5. Log event

### 4. Logging

All activities logged to:
- Console (visible in `docker logs ml-drift-monitor`)
- Log files in `logs/` directory

## Usage

### Start the System

```bash
docker-compose -f docker/docker-compose.yml up -d
```

**Result**: All services start, including drift-monitor

### View Drift Monitor Logs

```bash
docker logs -f ml-drift-monitor
```

### Stop the System

```bash
docker-compose -f docker/docker-compose.yml down
```

**Result**: All services stop, including drift-monitor

### Disable Automation

Edit `docker-compose.yml`:

```yaml
environment:
  - ENABLE_AUTOMATION=false
```

Then restart:

```bash
docker-compose -f docker/docker-compose.yml restart drift-monitor
```

### Change Check Interval

Edit `docker-compose.yml`:

```yaml
environment:
  - DRIFT_CHECK_INTERVAL=1800  # Check every 30 minutes
```

Then restart:

```bash
docker-compose -f docker/docker-compose.yml restart drift-monitor
```

## Why This Architecture?

### ✅ Correct Design

**Separate container for monitoring**:
- Clean separation of concerns
- Avoids race conditions when scaling API
- Single monitoring instance (not 3 if you scale to 3 API instances)
- Lifecycle tied to docker-compose

### ❌ What NOT to Do

**Don't run inside FastAPI**:
```python
# BAD: Don't do this
@app.on_event("startup")
async def start_monitoring():
    # This creates 3 monitors if you scale to 3 instances!
    asyncio.create_task(monitor_drift())
```

**Don't use cron**:
```bash
# BAD: External dependency, not tied to application
0 * * * * python src/monitoring_service.py
```

**Don't use OS-level service**:
```bash
# BAD: Orphaned process, hard to manage
systemctl start drift-monitor
```

## Monitoring the Monitor

### Check if Running

```bash
docker ps | grep drift-monitor
```

### View Real-time Logs

```bash
docker logs -f ml-drift-monitor
```

### Check Resource Usage

```bash
docker stats ml-drift-monitor
```

## Production Considerations

### 1. Adjust Check Interval

For production, you might want:
- **High-traffic**: Check every 30 minutes
- **Low-traffic**: Check every 4-6 hours
- **Critical systems**: Check every 15 minutes

### 2. Alert Integration

Add Slack/email alerts when drift detected:

```python
if drift_detected:
    send_slack_alert(f"Drift detected: {drift_score}")
    trigger_retraining()
```

### 3. Resource Limits

Add resource limits in docker-compose:

```yaml
drift-monitor:
  deploy:
    resources:
      limits:
        cpus: '0.5'
        memory: 512M
```

### 4. Health Checks

Add health check:

```yaml
drift-monitor:
  healthcheck:
    test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
    interval: 60s
    timeout: 10s
    retries: 3
```

## Troubleshooting

### Service Not Starting

```bash
# Check logs
docker logs ml-drift-monitor

# Check if automation is enabled
docker exec ml-drift-monitor env | grep ENABLE_AUTOMATION
```

### High Memory Usage

Reduce check frequency or add memory limits:

```yaml
environment:
  - DRIFT_CHECK_INTERVAL=7200  # 2 hours
deploy:
  resources:
    limits:
      memory: 512M
```

### Retraining Not Triggering

Check drift threshold:

```bash
# View current drift scores in logs
docker logs ml-drift-monitor | grep "Drift Score"
```

## Summary

**Architecture**: Separate Docker service for drift monitoring  
**Lifecycle**: Tied to docker-compose (starts/stops with system)  
**Scalability**: Single instance, avoids race conditions  
**Configuration**: Environment variables for easy tuning  
**Production-Ready**: Clean, maintainable, professional design  

**This is the correct way to implement autonomous ML lifecycle management.** ✅
