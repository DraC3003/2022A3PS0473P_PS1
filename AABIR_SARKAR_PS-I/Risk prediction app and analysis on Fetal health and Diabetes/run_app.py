#!/usr/bin/env python3
"""
Simple startup script for the Diabetes Prediction Application
This script checks dependencies and starts the application
"""
import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def check_required_files():
    """Check if required files exist"""
    required_files = ["app3.py", "database.py", "diabetes.csv", "requirements.txt"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        return False
    return True

def main():
    """Main startup function"""
    print("=" * 50)
    print("Diabetes Prediction Application Startup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    if not check_required_files():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    print("\nStarting the application...")
    print("The app will be available at: http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Import and run the app
        import app3
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()