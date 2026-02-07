# Deployment Script for ML Model API
# PowerShell script for Windows deployment

param(
    [string]$Environment = "development",
    [string]$ImageTag = "latest",
    [switch]$Build = $false,
    [switch]$Push = $false,
    [switch]$Deploy = $false,
    [switch]$Rollback = $false,
    [switch]$Ingest = $false,
    [switch]$All = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$DOCKER_IMAGE = "ml-model-api"
$REGISTRY = "your-registry.com"
$NAMESPACE = "ml-$Environment"

Write-Host "🚀 ML Model Deployment Script" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Cyan
Write-Host "Image Tag: $ImageTag" -ForegroundColor Cyan

if ($All) {
    $Build = $true
    $Push = $true
    $Deploy = $true
}

# Function to check if command exists
function Test-Command($Command) {
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "docker")) {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "kubectl") -and $Deploy) {
    Write-Host "⚠️  kubectl not found. Skipping Kubernetes deployment." -ForegroundColor Yellow
    $Deploy = $false
}

Write-Host "✅ Prerequisites check passed" -ForegroundColor Green

# Ingestion Pipeline (New Feature)
if ($Ingest) {
    Write-Host "📥 Starting Model Ingestion Pipeline..." -ForegroundColor Cyan
    
    # 1. Analyze
    Write-Host "🧐 Analyzing model in input_models/..." -ForegroundColor Yellow
    python src/analyze_model.py
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Analysis failed"; exit 1 }
    
    # 2. Verify
    Write-Host "🧪 Verifying readiness..." -ForegroundColor Yellow
    python src/verify_readiness.py
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Verification failed"; exit 1 }
    
    # 3. Promote
    Write-Host "🚀 Promoting to production..." -ForegroundColor Yellow
    python src/cleanup_and_promote.py
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Promotion failed"; exit 1 }
    
    Write-Host "✅ Ingestion Complete! Proceeding to build/deploy..." -ForegroundColor Green
    
    # Auto-enable build and deploy if ingestion succeeds
    $Build = $true
    $Deploy = $true
}

# Build Docker image
if ($Build) {
    Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
    
    # Ensure model exists
    if (-not (Test-Path "models/latest_model.json")) {
        Write-Host "📊 No trained model found. Training model first..." -ForegroundColor Yellow
        python src/train_model.py
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Model training failed" -ForegroundColor Red
            exit 1
        }
    }
    
    # Build image
    docker build -f docker/Dockerfile -t "${DOCKER_IMAGE}:${ImageTag}" .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker build failed" -ForegroundColor Red
        exit 1
    }
    
    # Tag for registry
    docker tag "${DOCKER_IMAGE}:${ImageTag}" "${REGISTRY}/${DOCKER_IMAGE}:${ImageTag}"
    
    Write-Host "✅ Docker image built successfully" -ForegroundColor Green
}

# Push to registry
if ($Push) {
    Write-Host "📤 Pushing image to registry..." -ForegroundColor Yellow
    
    docker push "${REGISTRY}/${DOCKER_IMAGE}:${ImageTag}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker push failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Image pushed successfully" -ForegroundColor Green
}

# Rollback
if ($Rollback) {
    Write-Host "🔙 Rolling back to previous model version..." -ForegroundColor Yellow
    
    python src/rollback.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Rollback failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Rollback completed successfully" -ForegroundColor Green
}

# Deploy using Docker Compose (local deployment)
if ($Deploy -and $Environment -eq "development") {
    Write-Host "🚀 Deploying locally with Docker Compose..." -ForegroundColor Yellow
    
    # Set environment variables
    $env:DOCKER_IMAGE_TAG = $ImageTag
    $env:API_PORT = "8000"
    
    # Deploy with docker-compose
    docker-compose -f docker/docker-compose.yml up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker Compose deployment failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Local deployment successful" -ForegroundColor Green
    Write-Host "🌐 API available at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📊 Prometheus: http://localhost:9090" -ForegroundColor Cyan
    Write-Host "📈 Grafana: http://localhost:3000" -ForegroundColor Cyan
}

# Deploy to Kubernetes (staging/production)
if ($Deploy -and $Environment -ne "development") {
    Write-Host "🚀 Deploying to Kubernetes ($Environment)..." -ForegroundColor Yellow
    Write-Host "⚠️  Kubernetes deployment logic is currently disabled for debugging." -ForegroundColor Yellow
    # Logic to be re-enabled after Here-String fix
}

# Health check
if ($Deploy) {
    Write-Host "🏥 Performing health check..." -ForegroundColor Yellow
    
    $healthUrl = "http://localhost:8000/health"
    if ($Environment -ne "development" -and $serviceUrl) {
         $healthUrl = "http://$serviceUrl/health"
    }
    
    $maxAttempts = 30
    $attempt = 0
    
    do {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
            if ($response.status -eq "healthy") {
                Write-Host "✅ Health check passed" -ForegroundColor Green
                break
            }
        } catch {
            # Ignore errors and retry
        }
        
        $attempt++
        Start-Sleep -Seconds 2
        Write-Host "Waiting for service to be ready... ($attempt/$maxAttempts)" -ForegroundColor Yellow
        
    } while ($attempt -lt $maxAttempts)
    
    if ($attempt -ge $maxAttempts) {
        Write-Host "❌ Health check failed - service not ready" -ForegroundColor Red
        exit 1
    }
}

Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green

# Display next steps
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Test the API: Invoke-RestMethod -Uri http://localhost:8000/health" -ForegroundColor White
Write-Host "2. View API docs: Start-Process http://localhost:8000/docs" -ForegroundColor White
Write-Host "3. Monitor metrics: Start-Process http://localhost:9090" -ForegroundColor White
Write-Host "4. View dashboards: Start-Process http://localhost:3000" -ForegroundColor White
