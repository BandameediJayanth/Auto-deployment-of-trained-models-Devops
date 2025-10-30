# Auto-Deployment ML Models - Windows Setup Script
# PowerShell Script for Windows Environment

param(
    [string]$GitUserName = "Your Name",
    [string]$GitUserEmail = "your@email.com",
    [string]$ProjectName = "ml-auto-deployment"
)

Write-Host "🚀 Setting up Auto-Deployment ML Models Project..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if Git is installed
try {
    $gitVersion = git --version 2>$null
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found. Please install Git first." -ForegroundColor Red
    exit 1
}

# Create Python virtual environment
Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install essential packages
Write-Host "📦 Installing essential Python packages..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Initialize Git repository
Write-Host "🔧 Initializing Git repository..." -ForegroundColor Yellow
git init
git config user.name $GitUserName
git config user.email $GitUserEmail

# Create initial commit
Write-Host "📝 Creating initial commit..." -ForegroundColor Yellow
git add .
git commit -m "Initial project setup - Auto-Deployment ML Models"

Write-Host "✅ Setup completed successfully!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Start with Phase 1: Model Development" -ForegroundColor White
Write-Host "3. Run: python src/train_model.py" -ForegroundColor White

