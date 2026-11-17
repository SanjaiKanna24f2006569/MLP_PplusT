# =============================================================================
# Automated Virtual Environment Setup Script (PowerShell)
# =============================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Setting up Python Virtual Environment for IITM BS MLP Project..." -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Check Python
try {
    $pyVer = python --version
    Write-Host "Found: $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10-3.12."
    Exit 1
}

# Create .venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Virtual environment .venv already exists." -ForegroundColor Green
}

# Activate
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Register Jupyter kernel
Write-Host "Registering Jupyter kernel 'Python (MLP)'..." -ForegroundColor Yellow
python -m ipykernel install --user --name=mlp-env --display-name="Python (MLP)"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Setup Complete! Virtual Environment is ready." -ForegroundColor Green
Write-Host "To activate anytime: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
