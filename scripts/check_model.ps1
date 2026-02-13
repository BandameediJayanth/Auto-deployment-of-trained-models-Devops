# Check Model Readiness Script
Write-Host "Running Model Readiness Check..." -ForegroundColor Cyan

# 1. Analyze
Write-Host "Analyzing model in input_models/..." -ForegroundColor Yellow
python src/analyze_model.py
if ($LASTEXITCODE -ne 0) { 
    Write-Host "Analysis failed" -ForegroundColor Red
    exit 1 
}

# 2. Verify
Write-Host "Verifying readiness..." -ForegroundColor Yellow
python src/verify_readiness.py
if ($LASTEXITCODE -ne 0) { 
    Write-Host "Verification failed" -ForegroundColor Red
    exit 1 
}

Write-Host "Check Complete! Review reports/final_verdict.md for results." -ForegroundColor Green
exit 0
