# Visual Studio Code Setup Guide

This guide will help you set up and use this repository in Visual Studio Code (VSCode) effectively.

## 📋 Overview

This repository contains two main Python Flask applications:

1. **Healthcare App** (`/CONT` directory) - Advanced prescription management and pregnancy risk prediction
2. **Risk Prediction App** (`/AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes` directory) - Simple diabetes and fetal health risk prediction

## 🚀 Quick Setup

### Prerequisites

1. **Install Visual Studio Code**: Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. **Install Python**: Version 3.8 or higher from [https://python.org](https://python.org)
3. **Install Git**: From [https://git-scm.com/](https://git-scm.com/)

### Essential VSCode Extensions

Install these extensions for the best development experience:

1. **Python** (by Microsoft) - Python language support
2. **Python Debugger** (by Microsoft) - Debugging support
3. **Flask Snippets** - Flask code snippets
4. **autoDocstring** - Auto-generate docstrings
5. **Python Indent** - Better indentation
6. **GitLens** - Enhanced Git capabilities
7. **Thunder Client** - API testing (alternative to Postman)
8. **Prettier** - Code formatting
9. **Python Type Hint** - Type hints support

## 🎯 Project Setup

### 1. Clone and Open Repository

```bash
git clone <repository-url>
cd 2022A3PS0473P_PS1
code .
```

### 2. Set Up Python Environment

#### For Healthcare App (CONT):

```bash
# Navigate to CONT directory
cd CONT

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python run.py init-db

# (Optional) Seed with sample data
python run.py seed-db
```

#### For Risk Prediction App:

```bash
# Navigate to the app directory
cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure VSCode Python Interpreter

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your virtual environment:
   - For CONT app: `./CONT/venv/Scripts/python.exe` (Windows) or `./CONT/venv/bin/python` (macOS/Linux)
   - For Risk Prediction app: `./AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes/venv/Scripts/python.exe`

## 🔧 Running the Applications

### Healthcare App (CONT)

1. Open integrated terminal (`Ctrl+`` ` ` or View > Terminal)
2. Navigate to CONT directory: `cd CONT`
3. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Run the application: `python run.py`
5. Open browser to: `http://localhost:5000`
6. Login with demo credentials: `doctor@example.com` / `password123`

### Risk Prediction App

1. Open new integrated terminal
2. Navigate to app directory: `cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"`
3. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Run the application: `python app3.py`
5. Open browser to: `http://localhost:5000` (or the port shown in terminal)

## 🐛 Debugging in VSCode

### Using the Debugger

1. Set breakpoints by clicking on the line numbers
2. Press `F5` to start debugging
3. Use the debug configuration in `.vscode/launch.json` (will be created automatically)

### Debug Configurations

The repository includes debug configurations for both applications:

- **Healthcare App**: Debugs `CONT/run.py`
- **Risk Prediction App**: Debugs the Flask app in the AABIR_SARKAR_PS-I directory

## 📁 Project Structure

```
2022A3PS0473P_PS1/
├── CONT/                                    # Healthcare App
│   ├── app/                                # Main application package
│   │   ├── routes/                         # URL routes
│   │   ├── templates/                      # HTML templates
│   │   ├── static/                         # CSS, JS, images
│   │   └── ml/                            # Machine learning models
│   ├── config.py                          # Configuration
│   ├── run.py                             # Application entry point
│   └── requirements.txt                   # Dependencies
├── AABIR_SARKAR_PS-I/                     # Student work directory
│   └── Risk prediction app and analysis on Fetal health and Diabetes/
│       ├── app3.py                        # Simple Flask app
│       ├── database.py                    # Database operations
│       ├── templates/                     # HTML templates
│       ├── static/                        # Static files
│       └── *.csv                          # Datasets
├── .vscode/                               # VSCode configuration
│   ├── settings.json                      # Workspace settings
│   ├── launch.json                        # Debug configurations
│   └── tasks.json                         # Build tasks
└── VSCODE_SETUP_GUIDE.md                 # This guide
```

## 🔍 Working with the Code

### Code Navigation

- **Go to Definition**: `F12` or `Ctrl+Click`
- **Find All References**: `Shift+F12`
- **Symbol Search**: `Ctrl+Shift+O`
- **File Search**: `Ctrl+P`
- **Global Search**: `Ctrl+Shift+F`

### Code Formatting

- **Format Document**: `Shift+Alt+F`
- **Format Selection**: `Ctrl+K Ctrl+F`

### IntelliSense and Autocomplete

VSCode provides excellent Python IntelliSense with:
- Auto-completion for variables, functions, and modules
- Parameter hints for functions
- Type information
- Import suggestions

## 🧪 Testing and Development

### Running Tests

```bash
# If tests exist, run them with:
python -m pytest
# or
python -m unittest discover
```

### API Testing

Use Thunder Client extension or curl to test API endpoints:

```bash
# Test healthcare app API
curl http://localhost:5000/api/patients

# Test risk prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "bmi": 25, "glucose": 120}'
```

## 📊 Working with Jupyter Notebooks

The repository contains Jupyter notebooks for data analysis:

1. Install Jupyter extension for VSCode
2. Open `.ipynb` files directly in VSCode
3. Select the correct Python interpreter
4. Run cells with `Shift+Enter`

## 🚀 Deployment and Production

### Environment Variables

Create `.env` files for different environments:

```bash
# .env
FLASK_ENV=development
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///app.db
```

### Build and Deploy

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## 🛠 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the correct Python interpreter is selected
2. **Module Not Found**: Ensure virtual environment is activated and dependencies are installed
3. **Port Already in Use**: Change the port in the Flask app or kill the existing process
4. **Database Errors**: Run database initialization commands

### VSCode Issues

1. **Python Not Detected**: Install Python extension and restart VSCode
2. **IntelliSense Not Working**: Reload window (`Ctrl+Shift+P` > "Developer: Reload Window")
3. **Debugging Not Working**: Check launch.json configuration

## 📚 Additional Resources

### Learning Resources

- [VSCode Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

### VSCode Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Quick Open | `Ctrl+P` | `Cmd+P` |
| Terminal | `Ctrl+`` | `Cmd+`` |
| Run Code | `F5` | `F5` |
| Format Document | `Shift+Alt+F` | `Shift+Option+F` |

## 🎯 Development Workflow

1. **Start Development**:
   - Open VSCode
   - Select appropriate Python interpreter
   - Activate virtual environment in terminal

2. **Code Changes**:
   - Make changes to Python files
   - Use IntelliSense for assistance
   - Format code regularly

3. **Testing**:
   - Run application to test changes
   - Use debugger for troubleshooting
   - Test API endpoints with Thunder Client

4. **Version Control**:
   - Use Source Control panel in VSCode
   - Commit changes with meaningful messages
   - Use GitLens for advanced Git features

## 🤝 Contributing

When contributing to this repository:

1. Create a new branch for your feature
2. Follow Python PEP 8 style guidelines
3. Add docstrings to functions and classes
4. Test your changes thoroughly
5. Update documentation as needed

## 📞 Support

If you encounter issues:

1. Check this guide first
2. Search for solutions in VSCode documentation
3. Check the Flask and Python documentation
4. Create an issue in the repository with details about the problem

---

*Happy coding! 🎉*