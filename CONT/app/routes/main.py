"""
Main routes for healthcare app.
Handles dashboard and general pages.
"""

from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models import Patient, Prescription, RiskAssessment

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard for healthcare providers."""
    # Get statistics for dashboard
    total_patients = Patient.query.count()
    total_prescriptions = Prescription.query.count()
    total_assessments = RiskAssessment.query.count()
    
    # Recent patients
    recent_patients = Patient.query.order_by(Patient.created_at.desc()).limit(5).all()
    
    # Recent prescriptions
    recent_prescriptions = Prescription.query.order_by(
        Prescription.prescribed_date.desc()
    ).limit(5).all()
    
    # High-risk pregnancies
    high_risk_assessments = RiskAssessment.query.filter(
        RiskAssessment.risk_level == 'High'
    ).order_by(RiskAssessment.assessment_date.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                         total_patients=total_patients,
                         total_prescriptions=total_prescriptions,
                         total_assessments=total_assessments,
                         recent_patients=recent_patients,
                         recent_prescriptions=recent_prescriptions,
                         high_risk_assessments=high_risk_assessments)

@main_bp.route('/about')
def about():
    """About page."""
    return render_template('about.html')
