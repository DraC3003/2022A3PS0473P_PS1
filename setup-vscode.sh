#!/bin/bash

# VS Code Setup Script for PS1 Healthcare Apps
# This script sets up the development environment for both applications

echo "🚀 Setting up VS Code development environment for PS1 Healthcare Apps"
echo "=================================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

if ! command_exists pip; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ Python and pip are installed"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Setup Healthcare App (CONT)
echo ""
echo "🏥 Setting up Healthcare App (CONT)..."
echo "======================================"

CONT_DIR="$PROJECT_ROOT/CONT"

if [ -d "$CONT_DIR" ]; then
    cd "$CONT_DIR"
    
    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
        PYTHON_PATH="$CONT_DIR/venv/Scripts/python"
    else
        source venv/bin/activate
        PYTHON_PATH="$CONT_DIR/venv/bin/python"
    fi
    
    # Install dependencies
    echo "⬇️  Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Initialize database
    echo "🗄️  Initializing database..."
    python run.py init-db
    
    # Seed database (optional)
    echo "🌱 Seeding database with sample data..."
    python run.py seed-db
    
    echo "✅ Healthcare App setup complete!"
    
    # Deactivate virtual environment
    deactivate
else
    echo "❌ CONT directory not found!"
fi

# Setup Risk Prediction App
echo ""
echo "📊 Setting up Risk Prediction App..."
echo "===================================="

RISK_APP_DIR="$PROJECT_ROOT/AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"

if [ -d "$RISK_APP_DIR" ]; then
    cd "$RISK_APP_DIR"
    
    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Install dependencies
    echo "⬇️  Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Install additional dependencies for data science
    echo "📊 Installing additional data science packages..."
    pip install pandas scikit-learn numpy matplotlib seaborn jupyter
    
    echo "✅ Risk Prediction App setup complete!"
    
    # Deactivate virtual environment
    deactivate
else
    echo "❌ Risk Prediction App directory not found!"
fi

# Return to project root
cd "$PROJECT_ROOT"

# Create .env files if they don't exist
echo ""
echo "📝 Creating environment configuration files..."

if [ ! -f "$CONT_DIR/.env" ]; then
    cat > "$CONT_DIR/.env" << EOL
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///healthcare_app.db
EOL
    echo "✅ Created .env file for Healthcare App"
fi

if [ ! -f "$RISK_APP_DIR/.env" ]; then
    cat > "$RISK_APP_DIR/.env" << EOL
FLASK_ENV=development
FLASK_DEBUG=1
YOUTUBE_API_KEY=your-youtube-api-key-here
EOL
    echo "✅ Created .env file for Risk Prediction App"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📚 Next Steps:"
echo "1. Open VS Code: code ."
echo "2. Install recommended extensions when prompted"
echo "3. Select Python interpreter from the virtual environments"
echo "4. Use F5 to run either application in debug mode"
echo ""
echo "🏥 Healthcare App: http://localhost:5000 (login: doctor@example.com / password123)"
echo "📊 Risk Prediction App: http://localhost:5000 (after running app3.py)"
echo ""
echo "📖 For detailed instructions, see: VSCODE_SETUP_GUIDE.md"
echo ""
echo "Happy coding! 🚀"