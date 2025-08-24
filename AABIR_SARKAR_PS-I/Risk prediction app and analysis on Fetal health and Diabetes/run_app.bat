@echo off
echo ================================================
echo Diabetes Prediction Application Startup (Windows)
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app3.py" (
    echo Error: app3.py not found
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo The app will be available at: http://127.0.0.1:5000/
echo Press Ctrl+C to stop the application
echo ------------------------------------------------

python app3.py

pause