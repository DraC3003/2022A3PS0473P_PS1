"""
Database Models for Healthcare App
Defines the data structure for users, patients, prescriptions, and risk assessments.
"""

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db

class User(UserMixin, db.Model):
    """User model for healthcare providers."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='doctor')  # doctor, nurse, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    prescriptions = db.relationship('Prescription', backref='doctor', lazy='dynamic')
    risk_assessments = db.relationship('RiskAssessment', backref='assessed_by', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Patient(db.Model):
    """Patient model for storing patient information."""
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True)
    phone = db.Column(db.String(20))
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10))
    address = db.Column(db.Text)
    emergency_contact = db.Column(db.String(100))
    emergency_phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Medical information
    blood_type = db.Column(db.String(5))
    allergies = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    current_medications = db.Column(db.Text)
    
    # Pregnancy information
    is_pregnant = db.Column(db.Boolean, default=False)
    pregnancy_start_date = db.Column(db.Date)
    expected_due_date = db.Column(db.Date)
    pregnancy_complications = db.Column(db.Text)
    
    # Relationships
    prescriptions = db.relationship('Prescription', backref='patient', lazy='dynamic')
    risk_assessments = db.relationship('RiskAssessment', backref='patient', lazy='dynamic')
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self):
        if self.date_of_birth:
            today = datetime.today().date()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None
    
    @property
    def pregnancy_week(self):
        """Calculate current pregnancy week."""
        if self.is_pregnant and self.pregnancy_start_date:
            days_pregnant = (datetime.today().date() - self.pregnancy_start_date).days
            return days_pregnant // 7
        return None
    
    def __repr__(self):
        return f'<Patient {self.full_name}>'

class Prescription(db.Model):
    """Prescription model for storing medication prescriptions."""
    __tablename__ = 'prescriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Prescription details
    medication_name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    duration = db.Column(db.String(50))
    instructions = db.Column(db.Text)
    
    # Dates
    prescribed_date = db.Column(db.DateTime, default=datetime.utcnow)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    
    # Additional information
    diagnosis = db.Column(db.String(200))
    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    
    # Pregnancy safety
    pregnancy_category = db.Column(db.String(5))  # A, B, C, D, X
    pregnancy_safe = db.Column(db.Boolean)
    
    def __repr__(self):
        return f'<Prescription {self.medication_name} for {self.patient.full_name}>'

class RiskAssessment(db.Model):
    """Risk assessment model for pregnancy disease prediction."""
    __tablename__ = 'risk_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    assessed_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Assessment details
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    pregnancy_week = db.Column(db.Integer)
    
    # Risk factors
    maternal_age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    glucose_level = db.Column(db.Float)
    protein_in_urine = db.Column(db.Boolean)
    previous_complications = db.Column(db.Boolean)
    family_history = db.Column(db.Text)
    smoking_status = db.Column(db.String(20))
    alcohol_consumption = db.Column(db.String(20))
    
    # Predictions
    preeclampsia_risk = db.Column(db.Float)
    gestational_diabetes_risk = db.Column(db.Float)
    preterm_birth_risk = db.Column(db.Float)
    overall_risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))  # Low, Medium, High
    
    # Recommendations
    recommendations = db.Column(db.Text)
    next_checkup_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    
    def __repr__(self):
        return f'<RiskAssessment for {self.patient.full_name} - Week {self.pregnancy_week}>'

class MedicalHistory(db.Model):
    """Medical history records for patients."""
    __tablename__ = 'medical_history'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # History details
    condition = db.Column(db.String(200), nullable=False)
    diagnosis_date = db.Column(db.Date)
    treatment = db.Column(db.Text)
    outcome = db.Column(db.String(100))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    patient = db.relationship('Patient', backref='medical_history_records')
    
    def __repr__(self):
        return f'<MedicalHistory {self.condition} for {self.patient.full_name}>'
