# Quick Start Guide - Diabetes Prediction App

## Fastest Way to Run the App

### Method 1: Using Startup Scripts (Recommended)

**Windows Users:**
1. Double-click `run_app.bat`
2. Wait for dependencies to install
3. Open browser to `http://127.0.0.1:5000/`

**Mac/Linux Users:**
```bash
cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"
./run_app.sh
```

**Cross-Platform Python Script:**
```bash
cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"
python3 run_app.py
```

### Method 2: Manual Steps

1. **Navigate to app directory:**
   ```bash
   cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app3.py
   ```
   or
   ```bash
   python3 app3.py
   ```

4. **Open in browser:**
   Go to `http://127.0.0.1:5000/` or `http://localhost:5000/`

## What You'll See

- A form asking for patient medical parameters
- Input fields for glucose, blood pressure, BMI, age, etc.
- After submitting, you'll get a diabetes risk prediction
- You can view consultation history and statistics

## Need Help?

- Check the main README.md for detailed instructions
- Ensure Python 3.7+ is installed
- Make sure you're in the correct directory
- All files (app3.py, database.py, diabetes.csv) should be present

## Common Issues

- **Port 5000 busy**: Try closing other applications or modify the port in app3.py
- **Module not found**: Run `pip install -r requirements.txt` again
- **Permission denied**: Try running as administrator/sudo (on some systems)