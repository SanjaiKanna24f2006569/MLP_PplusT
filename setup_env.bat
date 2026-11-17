@echo off
echo =====================================================================
echo Setting up Python Virtual Environment for IITM BS MLP Project...
echo =====================================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Registering Jupyter kernel...
python -m ipykernel install --user --name=mlp-env --display-name="Python (MLP)"

echo =====================================================================
echo Setup Complete!
echo To activate the virtual environment anytime in terminal:
echo   cmd:         .venv\Scripts\activate.bat
echo   powershell:  .\.venv\Scripts\Activate.ps1
echo =====================================================================
pause
