# 2022A3PS0473P_PS1 - Practice School I Projects

This repository contains all evaluative components for PS-I (Practice School I) submissions, featuring two main healthcare applications:

## 🏥 Applications Overview

1. **Healthcare App** (`/CONT`) - Advanced prescription management and pregnancy risk prediction
2. **Risk Prediction App** (`/AABIR_SARKAR_PS-I`) - Diabetes and fetal health risk analysis

## 🚀 Quick Start with Visual Studio Code

### Method 1: Automated Setup (Recommended)

**Windows:**
```cmd
setup-vscode.bat
```

**macOS/Linux:**
```bash
./setup-vscode.sh
```

**Then open in VS Code:**
```bash
code .
```

### Method 2: Manual Setup

For the simple risk prediction app (app3.py):

```bash
cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
python app3.py
```

## 📚 Documentation

- **[📖 Complete VSCode Setup Guide](VSCODE_SETUP_GUIDE.md)** - Comprehensive guide for using this repository in Visual Studio Code
- **[🏥 Healthcare App Documentation](CONT/README.md)** - Advanced healthcare application details
- **[📊 Risk Prediction App](AABIR_SARKAR_PS-I/)** - Simple ML-based risk prediction

## 🎯 VSCode Features Included

- **Debug Configurations** - Pre-configured launch settings for both apps
- **Python Environment** - Virtual environment setup and management  
- **Extensions** - Recommended extensions for Python/Flask development
- **Tasks** - Automated build and setup tasks
- **IntelliSense** - Enhanced code completion and navigation
- **Workspace** - Multi-folder workspace configuration

## 🔧 Development Environment

- **Python 3.8+** required
- **Flask** web framework
- **Machine Learning** with scikit-learn
- **Data Analysis** with pandas, numpy
- **Database** SQLite/SQLAlchemy

## 🌐 Access Applications

After setup:
- **Healthcare App**: http://localhost:5000 (login: doctor@example.com / password123)
- **Risk Prediction App**: http://localhost:5000 (check terminal for exact port)

---

*For detailed setup instructions and VSCode configuration, see [VSCODE_SETUP_GUIDE.md](VSCODE_SETUP_GUIDE.md)*
