@echo off
REM VS Code Setup Script for PS1 Healthcare Apps (Windows)
REM This script sets up the development environment for both applications

echo 🚀 Setting up VS Code development environment for PS1 Healthcare Apps
echo ==================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo ✅ Python and pip are installed

REM Get the script directory
set "PROJECT_ROOT=%~dp0"

REM Setup Healthcare App (CONT)
echo.
echo 🏥 Setting up Healthcare App (CONT)...
echo ======================================

set "CONT_DIR=%PROJECT_ROOT%CONT"

if exist "%CONT_DIR%" (
    cd /d "%CONT_DIR%"
    
    REM Create virtual environment
    echo 📦 Creating virtual environment...
    python -m venv venv
    
    REM Activate virtual environment
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
    
    REM Install dependencies
    echo ⬇️  Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Initialize database
    echo 🗄️  Initializing database...
    python run.py init-db
    
    REM Seed database (optional)
    echo 🌱 Seeding database with sample data...
    python run.py seed-db
    
    echo ✅ Healthcare App setup complete!
    
    REM Deactivate virtual environment
    call venv\Scripts\deactivate.bat
) else (
    echo ❌ CONT directory not found!
)

REM Setup Risk Prediction App
echo.
echo 📊 Setting up Risk Prediction App...
echo ====================================

set "RISK_APP_DIR=%PROJECT_ROOT%AABIR_SARKAR_PS-I\Risk prediction app and analysis on Fetal health and Diabetes"

if exist "%RISK_APP_DIR%" (
    cd /d "%RISK_APP_DIR%"
    
    REM Create virtual environment
    echo 📦 Creating virtual environment...
    python -m venv venv
    
    REM Activate virtual environment
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
    
    REM Install dependencies
    echo ⬇️  Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Install additional dependencies for data science
    echo 📊 Installing additional data science packages...
    pip install pandas scikit-learn numpy matplotlib seaborn jupyter
    
    echo ✅ Risk Prediction App setup complete!
    
    REM Deactivate virtual environment
    call venv\Scripts\deactivate.bat
) else (
    echo ❌ Risk Prediction App directory not found!
)

REM Return to project root
cd /d "%PROJECT_ROOT%"

REM Create .env files if they don't exist
echo.
echo 📝 Creating environment configuration files...

if not exist "%CONT_DIR%\.env" (
    (
        echo FLASK_ENV=development
        echo FLASK_DEBUG=1
        echo SECRET_KEY=your-secret-key-change-in-production
        echo DATABASE_URL=sqlite:///healthcare_app.db
    ) > "%CONT_DIR%\.env"
    echo ✅ Created .env file for Healthcare App
)

if not exist "%RISK_APP_DIR%\.env" (
    (
        echo FLASK_ENV=development
        echo FLASK_DEBUG=1
        echo YOUTUBE_API_KEY=your-youtube-api-key-here
    ) > "%RISK_APP_DIR%\.env"
    echo ✅ Created .env file for Risk Prediction App
)

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo 📚 Next Steps:
echo 1. Open VS Code: code .
echo 2. Install recommended extensions when prompted
echo 3. Select Python interpreter from the virtual environments
echo 4. Use F5 to run either application in debug mode
echo.
echo 🏥 Healthcare App: http://localhost:5000 (login: doctor@example.com / password123)
echo 📊 Risk Prediction App: http://localhost:5000 (after running app3.py)
echo.
echo 📖 For detailed instructions, see: VSCODE_SETUP_GUIDE.md
echo.
echo Happy coding! 🚀
pause