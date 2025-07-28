#!/bin/bash

echo "================================================"
echo "Diabetes Prediction Application Startup (Unix/Linux/macOS)"
echo "================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app3.py" ]; then
    echo "Error: app3.py not found"
    echo "Please make sure you're in the correct directory:"
    echo "cd 'AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes'"
    exit 1
fi

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Starting the application..."
echo "The app will be available at: http://127.0.0.1:5000/"
echo "Press Ctrl+C to stop the application"
echo "------------------------------------------------"

python3 app3.py