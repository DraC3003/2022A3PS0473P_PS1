"""
Patient management routes.
Handles patient registration, viewing, and updating.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from app import db
from app.models import Patient, Prescription, RiskAssessment

patients_bp = Blueprint('patients', __name__)

@patients_bp.route('/')
@login_required
def list_patients():
    """List all patients."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    query = Patient.query
    
    if search:
        query = query.filter(
            Patient.first_name.contains(search) |
            Patient.last_name.contains(search) |
            Patient.email.contains(search)
        )
    
    patients = query.order_by(Patient.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    
    return render_template('patients/list.html', patients=patients, search=search)

@patients_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_patient():
    """Add new patient."""
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        date_of_birth = request.form.get('date_of_birth')
        gender = request.form.get('gender')
        address = request.form.get('address')
        emergency_contact = request.form.get('emergency_contact')
        emergency_phone = request.form.get('emergency_phone')
        
        # Medical information
        blood_type = request.form.get('blood_type')
        allergies = request.form.get('allergies')
        medical_history = request.form.get('medical_history')
        current_medications = request.form.get('current_medications')
        
        # Pregnancy information
        is_pregnant = bool(request.form.get('is_pregnant'))
        pregnancy_start_date = request.form.get('pregnancy_start_date')
        expected_due_date = request.form.get('expected_due_date')
        pregnancy_complications = request.form.get('pregnancy_complications')
        
        # Validation
        if not first_name or not last_name or not date_of_birth:
            flash('First name, last name, and date of birth are required.', 'error')
            return render_template('patients/add.html')
        
        # Check for existing email
        if email and Patient.query.filter_by(email=email).first():
            flash('Patient with this email already exists.', 'error')
            return render_template('patients/add.html')
        
        # Create new patient
        patient = Patient(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            date_of_birth=datetime.strptime(date_of_birth, '%Y-%m-%d').date(),
            gender=gender,
            address=address,
            emergency_contact=emergency_contact,
            emergency_phone=emergency_phone,
            blood_type=blood_type,
            allergies=allergies,
            medical_history=medical_history,
            current_medications=current_medications,
            is_pregnant=is_pregnant,
            pregnancy_start_date=datetime.strptime(pregnancy_start_date, '%Y-%m-%d').date() if pregnancy_start_date else None,
            expected_due_date=datetime.strptime(expected_due_date, '%Y-%m-%d').date() if expected_due_date else None,
            pregnancy_complications=pregnancy_complications
        )
        
        db.session.add(patient)
        db.session.commit()
        
        flash(f'Patient {patient.full_name} added successfully!', 'success')
        return redirect(url_for('patients.view_patient', id=patient.id))
    
    return render_template('patients/add.html')

@patients_bp.route('/<int:id>')
@login_required
def view_patient(id):
    """View patient details."""
    patient = Patient.query.get_or_404(id)
    
    # Get patient's prescriptions
    prescriptions = Prescription.query.filter_by(patient_id=id).order_by(
        Prescription.prescribed_date.desc()
    ).all()
    
    # Get patient's risk assessments
    risk_assessments = RiskAssessment.query.filter_by(patient_id=id).order_by(
        RiskAssessment.assessment_date.desc()
    ).all()
    
    return render_template('patients/view.html',
                         patient=patient,
                         prescriptions=prescriptions,
                         risk_assessments=risk_assessments)

@patients_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(id):
    """Edit patient information."""
    patient = Patient.query.get_or_404(id)
    
    if request.method == 'POST':
        # Update patient information
        patient.first_name = request.form.get('first_name')
        patient.last_name = request.form.get('last_name')
        patient.email = request.form.get('email')
        patient.phone = request.form.get('phone')
        patient.gender = request.form.get('gender')
        patient.address = request.form.get('address')
        patient.emergency_contact = request.form.get('emergency_contact')
        patient.emergency_phone = request.form.get('emergency_phone')
        
        # Medical information
        patient.blood_type = request.form.get('blood_type')
        patient.allergies = request.form.get('allergies')
        patient.medical_history = request.form.get('medical_history')
        patient.current_medications = request.form.get('current_medications')
        
        # Pregnancy information
        patient.is_pregnant = bool(request.form.get('is_pregnant'))
        pregnancy_start_date = request.form.get('pregnancy_start_date')
        expected_due_date = request.form.get('expected_due_date')
        patient.pregnancy_start_date = datetime.strptime(pregnancy_start_date, '%Y-%m-%d').date() if pregnancy_start_date else None
        patient.expected_due_date = datetime.strptime(expected_due_date, '%Y-%m-%d').date() if expected_due_date else None
        patient.pregnancy_complications = request.form.get('pregnancy_complications')
        
        db.session.commit()
        flash('Patient information updated successfully!', 'success')
        return redirect(url_for('patients.view_patient', id=patient.id))
    
    return render_template('patients/edit.html', patient=patient)

@patients_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete_patient(id):
    """Delete patient (soft delete)."""
    patient = Patient.query.get_or_404(id)
    
    # In a real application, you might want to implement soft delete
    # For now, we'll just remove the patient
    db.session.delete(patient)
    db.session.commit()
    
    flash(f'Patient {patient.full_name} has been deleted.', 'info')
    return redirect(url_for('patients.list_patients'))
