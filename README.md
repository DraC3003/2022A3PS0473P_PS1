# 2022A3PS0473P_PS1 - Diabetes Prediction Web Application

This repository contains a Flask-based web application for diabetes risk prediction using machine learning. The application uses a logistic regression model trained on diabetes data to predict patient risk levels.

## Application Features

- **Diabetes Risk Prediction**: Input patient medical parameters to get diabetes risk predictions
- **Consultation Database**: All predictions are stored in a SQLite database for historical tracking
- **Web Interface**: User-friendly web interface for data input and result visualization
- **Consultation History**: View past consultations and statistics
- **YouTube Integration**: Provides relevant health and lifestyle videos based on prediction results

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## How to Run the Application

### Step 1: Navigate to the Application Directory

```bash
cd "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes"
```

### Step 2: Install Dependencies

#### Option A: Using pip directly
```bash
pip install -r requirements.txt
```

#### Option B: Using virtual environment (recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app3.py
```

**On some systems, you may need to use:**
```bash
python3 app3.py
```

### Step 4: Access the Application

1. Open your web browser
2. Navigate to: `http://127.0.0.1:5000/` or `http://localhost:5000/`
3. You should see the Diabetes Prediction System interface

## Using the Application

1. **Make a Prediction:**
   - Fill in the patient medical parameters in the form
   - Click "Predict" to get the diabetes risk assessment
   - View the result and any recommended videos

2. **View Consultation History:**
   - Click "View Consultation History" to see past predictions
   - Access detailed statistics about consultations

## Input Parameters

The application requires the following medical parameters:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function score
- **Age**: Age in years

## File Structure

```
AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes/
├── app3.py                 # Main Flask application
├── database.py             # Database operations
├── requirements.txt        # Python dependencies
├── diabetes.csv           # Training dataset
├── templates/             # HTML templates
│   ├── index.html         # Main input form
│   ├── result.html        # Prediction results
│   ├── consultations.html # Consultation history
│   └── consultation_detail.html
├── static/               # Static files (CSS, JS)
└── *.ipynb              # Jupyter notebooks for analysis
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install pandas scikit-learn flask requests
   ```

2. **Port Already in Use**: If port 5000 is busy, the app will show an error. Try:
   - Close other applications using port 5000
   - Or modify `app3.py` to use a different port:
     ```python
     app.run(debug=True, port=5001)
     ```

3. **Permission Issues**: On some systems, you may need to run with elevated privileges

4. **Python Version**: Ensure you're using Python 3.7 or higher
   ```bash
   python --version
   ```

### Database Issues

The application automatically creates a SQLite database (`consultations.db`) to store prediction history. If you encounter database errors:

1. Delete the existing database file (if present): `consultations.db`
2. Restart the application - it will create a new database

## Development Notes

- The application runs in debug mode by default
- Database file (`consultations.db`) is created automatically
- YouTube API integration requires an API key (currently set to placeholder)
- The model is trained each time the application starts using the included dataset

## YouTube API Configuration (Optional)

To enable YouTube video recommendations:
1. Get a YouTube Data API key from Google Cloud Console
2. Replace `'enter ur api key'` in `app3.py` with your actual API key

## Support

If you encounter any issues:
1. Check that all prerequisites are met
2. Ensure you're in the correct directory
3. Verify all dependencies are installed
4. Check the terminal/console for error messages
