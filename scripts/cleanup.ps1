# Cleanup Script for GitHub Repository
# Removes temporary files, logs, and cache before pushing to GitHub

Write-Host "Cleaning up codebase for GitHub..." -ForegroundColor Cyan

# Remove log files
Write-Host "Removing log files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter "*.log" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Remove Python cache
Write-Host "Removing Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Remove test results (keep structure)
Write-Host "Cleaning test results..." -ForegroundColor Yellow
if (Test-Path "models") {
    Get-ChildItem -Path "models" -Filter "test_results_*.json" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

# Remove temporary files
Write-Host "Removing temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter "*.tmp" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*.temp" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Remove OS-specific files
Write-Host "Removing OS-specific files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter ".DS_Store" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "Thumbs.db" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Create .gitkeep files for important directories
Write-Host "Creating .gitkeep files..." -ForegroundColor Yellow
@("models", "data", "logs", "reports", "input_models") | ForEach-Object {
    if (Test-Path $_) {
        $gitkeep = Join-Path $_ ".gitkeep"
        if (-not (Test-Path $gitkeep)) {
            New-Item -ItemType File -Path $gitkeep -Force | Out-Null
        }
    }
}

Write-Host "Cleanup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Review changes: git status" -ForegroundColor White
Write-Host "2. Add files: git add ." -ForegroundColor White
Write-Host "3. Commit: git commit -m 'Initial commit: MLOps project'" -ForegroundColor White
Write-Host "4. Push: git push origin main" -ForegroundColor White
