# 🚨 Troubleshooting Guide for VSCode Setup

This guide helps you resolve common issues when setting up and using this repository in Visual Studio Code.

## 🔧 Common Setup Issues

### Problem: "Python not found" or "python command not recognized"

**Solution:**
1. Install Python 3.8+ from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Restart your terminal/VSCode
4. Verify: `python --version` should show Python 3.8+

### Problem: "pip not found" or "pip command not recognized"

**Solution:**
1. Python installation should include pip
2. Try: `python -m pip --version`
3. If still not working, reinstall Python with pip included

### Problem: Virtual environment activation fails

**Windows:**
```cmd
# Try these alternatives:
venv\Scripts\activate.bat
# or
venv\Scripts\activate.ps1
# or
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
# Try these alternatives:
source venv/bin/activate
# or
. venv/bin/activate
```

### Problem: Permission denied when running setup scripts

**Windows (run as Administrator):**
```cmd
# Right-click Command Prompt -> "Run as administrator"
setup-vscode.bat
```

**macOS/Linux:**
```bash
chmod +x setup-vscode.sh
./setup-vscode.sh
```

## 🐍 Python Environment Issues

### Problem: Wrong Python interpreter selected in VSCode

**Solution:**
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your virtual environment:
   - CONT app: `./CONT/venv/Scripts/python.exe` (Windows) or `./CONT/venv/bin/python` (Mac/Linux)
   - Risk app: `./AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes/venv/Scripts/python.exe`

### Problem: ModuleNotFoundError when running code

**Solution:**
1. Ensure virtual environment is activated
2. Check if you're in the correct directory
3. Reinstall dependencies: `pip install -r requirements.txt`
4. Verify Python interpreter is from virtual environment

### Problem: Flask app won't start

**Check these:**
1. Virtual environment is activated
2. All dependencies are installed: `pip list`
3. You're in the correct directory (CONT for healthcare app, or the long path for risk prediction app)
4. No other process is using port 5000: `netstat -an | grep 5000`

## 🔌 VSCode Extension Issues

### Problem: Python extension not working

**Solution:**
1. Install Python extension: `ms-python.python`
2. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check extension is enabled: Extensions panel → Search "Python"

### Problem: IntelliSense not working

**Solution:**
1. Ensure correct Python interpreter is selected
2. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Clear cache: `Ctrl+Shift+P` → "Python: Clear Cache and Reload Window"

### Problem: Debugger not working

**Solution:**
1. Check launch.json configuration
2. Ensure correct Python interpreter
3. Verify working directory is correct in launch configuration
4. Try "Python: Current File" debug configuration first

## 🗄️ Database Issues

### Problem: Database initialization fails (Healthcare app)

**Solution:**
```bash
cd CONT
python run.py init-db --force  # Force recreate
```

### Problem: SQLite database locked

**Solution:**
1. Close all applications using the database
2. Delete the database file: `rm healthcare_app.db` (in CONT directory)
3. Reinitialize: `python run.py init-db`

## 🌐 Web Application Issues

### Problem: "Port 5000 already in use"

**Solution:**
1. Find the process: `lsof -i :5000` (Mac/Linux) or `netstat -ano | findstr :5000` (Windows)
2. Kill the process or change the port in the app

### Problem: Application runs but can't access it in browser

**Solution:**
1. Check the terminal output for the correct URL
2. Try `http://127.0.0.1:5000` instead of `localhost:5000`
3. Disable firewall temporarily to test
4. Check if Flask is binding to correct interface

### Problem: Login fails (Healthcare app)

**Solution:**
1. Use demo credentials: `doctor@example.com` / `password123`
2. If still fails, reseed the database: `python run.py seed-db`

## 📊 Machine Learning Issues

### Problem: Model training fails

**Solution:**
1. Check if datasets exist (diabetes.csv, fetal_health.csv)
2. Verify pandas and scikit-learn are installed: `pip install pandas scikit-learn`
3. Check for sufficient memory

### Problem: Prediction results are unexpected

**Solution:**
1. Verify input data format
2. Check if model file exists and is not corrupted
3. Retrain the model by restarting the application

## 🔄 Git and Version Control Issues

### Problem: Git push fails

**Solution:**
1. Check if you have write permissions to the repository
2. Pull latest changes first: `git pull`
3. Resolve any merge conflicts

### Problem: Large files not tracking properly

**Solution:**
1. Add files to .gitignore if they're build artifacts
2. Use git LFS for large datasets if needed

## 💻 Operating System Specific Issues

### Windows Issues

**PowerShell Execution Policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Long path names:**
```cmd
# Enable long paths in Windows registry or use shorter directory names
```

### macOS Issues

**Xcode Command Line Tools:**
```bash
xcode-select --install
```

**Permission issues with Python:**
```bash
# Use homebrew Python instead of system Python
brew install python
```

### Linux Issues

**Missing development headers:**
```bash
# Ubuntu/Debian:
sudo apt-get install python3-dev python3-pip python3-venv

# CentOS/RHEL:
sudo yum install python3-devel python3-pip
```

## 🆘 Getting Help

If you're still having issues:

1. **Check the logs:** Look at terminal output for specific error messages
2. **Search online:** Copy the exact error message and search for solutions
3. **Documentation:** Review the full [VSCODE_SETUP_GUIDE.md](VSCODE_SETUP_GUIDE.md)
4. **Community:** Ask on Stack Overflow with tags: `python`, `flask`, `vscode`

## 📋 Quick Diagnostic Commands

Run these to check your setup:

```bash
# Check Python
python --version
python -m pip --version

# Check virtual environment (after activation)
which python  # macOS/Linux
where python   # Windows

# Check installed packages
pip list

# Check Flask installation
python -c "import flask; print(flask.__version__)"

# Check current directory
pwd          # macOS/Linux
cd           # Windows

# Check if port is free
netstat -an | grep 5000   # macOS/Linux
netstat -an | findstr 5000  # Windows
```

## 🎯 Best Practices to Avoid Issues

1. **Always activate virtual environment** before running commands
2. **Use correct paths** - pay attention to spaces in directory names
3. **Keep dependencies updated** but test after updates
4. **Commit working states** to git frequently
5. **Use absolute paths** when in doubt
6. **Read error messages carefully** - they usually contain the solution

---

*If this guide doesn't solve your issue, create a new issue in the repository with:*
- Your operating system
- Python version
- Exact error message
- Steps you tried