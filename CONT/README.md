# Healthcare App - Prescription Management & Pregnancy Risk Prediction

A comprehensive healthcare application that combines prescription management with AI-powered pregnancy risk prediction using machine learning.

## Features

### 🏥 Core Healthcare Features

- **Prescription Management**: Store, track, and manage doctor prescriptions
- **Patient Records**: Comprehensive patient information and medical history
- **Doctor Dashboard**: Interface for healthcare providers with analytics
- **Medical History Tracking**: Complete patient medical record management

### 🤖 AI-Powered Risk Assessment

- **Pregnancy Risk Prediction**: ML-powered disease risk assessment during pregnancy
- **Timeline Analysis**: Week-by-week risk progression throughout pregnancy
- **Smart Scheduling**: Automated appointment recommendations based on risk levels
- **Multi-condition Prediction**: Preeclampsia, gestational diabetes, and preterm birth risks

### 📊 Analytics & Insights

- **Risk Timeline Visualization**: Interactive charts showing risk progression
- **High-Risk Patient Monitoring**: Automated alerts for patients requiring attention
- **Treatment Recommendations**: AI-generated care suggestions
- **Statistical Dashboard**: Healthcare provider analytics and insights

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLAlchemy with SQLite
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Authentication**: Flask-Login with session management
- **Data Visualization**: Chart.js for interactive charts

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download the project**

   ```bash
   git clone <repository-url>
   cd healthcare-app
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**

   ```bash
   python run.py init-db
   ```

4. **Seed with sample data (optional)**

   ```bash
   python run.py seed-db
   ```

5. **Run the application**

   ```bash
   python run.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Login with demo account: `doctor@example.com` / `password123`

## Usage Guide

### Getting Started

1. **Login/Register**: Create an account or use the demo credentials
2. **Add Patients**: Register new patients with comprehensive medical information
3. **Manage Prescriptions**: Add and track medication prescriptions
4. **Risk Assessment**: Perform pregnancy risk assessments using ML models
5. **Monitor Progress**: Track risk progression and schedule follow-ups

### Key Workflows

#### Patient Management

- Add new patients with medical history
- View comprehensive patient profiles
- Track pregnancy status and complications
- Manage emergency contacts and medical alerts

#### Prescription Management

- Create detailed prescription records
- Track medication history and interactions
- Monitor pregnancy-safe medications
- Set duration and dosage instructions

#### Risk Prediction Workflow

1. **Assessment**: Input patient data (age, BMI, blood pressure, etc.)
2. **Prediction**: ML model calculates risk scores for multiple conditions
3. **Timeline**: View week-by-week risk progression
4. **Recommendations**: Receive AI-generated care suggestions
5. **Scheduling**: Automated next appointment recommendations

## Machine Learning Models

### Risk Prediction Features

The ML model uses the following patient data:

- Maternal age
- BMI (Body Mass Index)
- Blood pressure (systolic/diastolic)
- Glucose levels
- Protein in urine
- Previous pregnancy complications
- Smoking and alcohol status
- Current pregnancy week

### Predicted Conditions

- **Preeclampsia**: High blood pressure during pregnancy
- **Gestational Diabetes**: Diabetes developed during pregnancy
- **Preterm Birth**: Early delivery risk assessment

### Model Performance

- Uses Random Forest algorithms for classification
- Trained on synthetic medical data following clinical guidelines
- Provides confidence scores and risk levels (Low/Medium/High)
- Week-specific risk adjustments based on pregnancy stage

## API Endpoints

### Patient Management

- `GET /api/patients` - List all patients
- `POST /api/patients` - Create new patient
- `GET /api/patients/<id>/prescriptions` - Get patient prescriptions

### Prescription Management

- `POST /api/prescriptions` - Create new prescription
- `GET /api/prescriptions/<id>` - Get prescription details

### Risk Assessment

- `POST /api/risk-assessment` - Create risk assessment
- `GET /api/patients/<id>/risk-timeline` - Get risk timeline
- `POST /api/risk/predict` - Real-time risk prediction

### Analytics

- `GET /api/stats` - Application statistics
- `GET /api/high-risk-patients` - List high-risk patients

## Project Structure

```
healthcare-app/
├── app/                          # Main application package
│   ├── __init__.py              # Flask app initialization
│   ├── models.py                # Database models
│   ├── ml/                      # Machine learning modules
│   │   ├── __init__.py
│   │   └── risk_predictor.py    # Pregnancy risk prediction model
│   ├── routes/                  # Application routes
│   │   ├── main.py             # Dashboard and main pages
│   │   ├── auth.py             # Authentication routes
│   │   ├── patients.py         # Patient management
│   │   ├── prescriptions.py    # Prescription management
│   │   ├── risk_prediction.py  # Risk assessment routes
│   │   └── api.py              # REST API endpoints
│   ├── templates/               # HTML templates
│   │   ├── base.html           # Base template
│   │   ├── index.html          # Landing page
│   │   ├── dashboard.html      # Main dashboard
│   │   ├── auth/               # Authentication templates
│   │   ├── patients/           # Patient management templates
│   │   ├── prescriptions/      # Prescription templates
│   │   └── risk/               # Risk assessment templates
│   └── static/                  # Static files
│       ├── css/                # Custom CSS
│       ├── js/                 # JavaScript files
│       └── uploads/            # File uploads
├── migrations/                  # Database migrations
├── config.py                   # Configuration settings
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Database Schema

### Core Models

- **User**: Healthcare providers (doctors, nurses, admins)
- **Patient**: Patient information and medical history
- **Prescription**: Medication prescriptions and instructions
- **RiskAssessment**: Pregnancy risk evaluation records
- **MedicalHistory**: Patient medical history records

### Key Relationships

- Users can create multiple prescriptions and risk assessments
- Patients can have multiple prescriptions and risk assessments
- Risk assessments track pregnancy progression over time

## Security Features

- **Password Hashing**: Secure password storage using Werkzeug
- **Session Management**: Flask-Login for user session handling
- **Input Validation**: Form validation and sanitization
- **CSRF Protection**: Cross-site request forgery protection
- **Data Encryption**: Secure data storage and transmission

## Deployment

### Development

The application runs in development mode by default with debug features enabled.

### Production Deployment

For production deployment:

1. Set environment variables for security
2. Use a production WSGI server (Gunicorn)
3. Configure a production database (PostgreSQL)
4. Set up SSL/HTTPS encryption
5. Implement proper logging and monitoring

## Contributing

### Development Guidelines

1. Follow PEP 8 coding standards
2. Write comprehensive docstrings
3. Include unit tests for new features
4. Validate all medical data inputs
5. Ensure HIPAA compliance considerations

### Adding New Features

1. Create feature branch from main
2. Implement feature with tests
3. Update documentation
4. Submit pull request for review

## Medical Disclaimer

This application is designed for educational and demonstration purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, questions, or feature requests:

- Create an issue in the repository
- Contact the development team
- Review the documentation and API guides

## Changelog

### Version 1.0.0

- Initial release with core features
- Patient and prescription management
- ML-powered pregnancy risk prediction
- Interactive dashboard and analytics
- REST API for external integrations
