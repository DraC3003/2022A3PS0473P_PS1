"""
Prescription management routes.
Handles prescription creation, viewing, and management.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from datetime import datetime, date
from app import db
from app.models import Patient, Prescription, User

prescriptions_bp = Blueprint('prescriptions', __name__)

@prescriptions_bp.route('/')
@login_required
def list_prescriptions():
    """List all prescriptions."""
    page = request.args.get('page', 1, type=int)
    patient_id = request.args.get('patient_id', type=int)
    
    query = Prescription.query
    
    if patient_id:
        query = query.filter_by(patient_id=patient_id)
    
    prescriptions = query.order_by(Prescription.prescribed_date.desc()).paginate(
        page=page, per_page=15, error_out=False
    )
    
    return render_template('prescriptions/list.html', prescriptions=prescriptions)

@prescriptions_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_prescription():
    """Add new prescription."""
    patient_id = request.args.get('patient_id', type=int)
    patients = Patient.query.all()
    
    if request.method == 'POST':
        # Get form data
        patient_id = request.form.get('patient_id', type=int)
        medication_name = request.form.get('medication_name')
        dosage = request.form.get('dosage')
        frequency = request.form.get('frequency')
        duration = request.form.get('duration')
        instructions = request.form.get('instructions')
        diagnosis = request.form.get('diagnosis')
        notes = request.form.get('notes')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        pregnancy_category = request.form.get('pregnancy_category')
        pregnancy_safe = request.form.get('pregnancy_safe') == 'true'
        
        # Validation
        if not patient_id or not medication_name or not dosage or not frequency:
            flash('Patient, medication name, dosage, and frequency are required.', 'error')
            return render_template('prescriptions/add.html', patients=patients, patient_id=patient_id)
        
        patient = Patient.query.get(patient_id)
        if not patient:
            flash('Invalid patient selected.', 'error')
            return render_template('prescriptions/add.html', patients=patients)
        
        # Check pregnancy safety
        if patient.is_pregnant and not pregnancy_safe:
            flash('Warning: This medication may not be safe during pregnancy. Please review.', 'warning')
        
        # Create new prescription
        prescription = Prescription(
            patient_id=patient_id,
            doctor_id=current_user.id,
            medication_name=medication_name,
            dosage=dosage,
            frequency=frequency,
            duration=duration,
            instructions=instructions,
            diagnosis=diagnosis,
            notes=notes,
            start_date=datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None,
            end_date=datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None,
            pregnancy_category=pregnancy_category,
            pregnancy_safe=pregnancy_safe
        )
        
        db.session.add(prescription)
        db.session.commit()
        
        flash(f'Prescription for {medication_name} added successfully!', 'success')
        return redirect(url_for('patients.view_patient', id=patient_id))
    
    return render_template('prescriptions/add.html', patients=patients, patient_id=patient_id)

@prescriptions_bp.route('/<int:id>')
@login_required
def view_prescription(id):
    """View prescription details."""
    prescription = Prescription.query.get_or_404(id)
    return render_template('prescriptions/view.html', prescription=prescription)

@prescriptions_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_prescription(id):
    """Edit prescription."""
    prescription = Prescription.query.get_or_404(id)
    patients = Patient.query.all()
    
    if request.method == 'POST':
        # Update prescription
        prescription.patient_id = request.form.get('patient_id', type=int)
        prescription.medication_name = request.form.get('medication_name')
        prescription.dosage = request.form.get('dosage')
        prescription.frequency = request.form.get('frequency')
        prescription.duration = request.form.get('duration')
        prescription.instructions = request.form.get('instructions')
        prescription.diagnosis = request.form.get('diagnosis')
        prescription.notes = request.form.get('notes')
        
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        prescription.start_date = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
        prescription.end_date = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None
        
        prescription.pregnancy_category = request.form.get('pregnancy_category')
        prescription.pregnancy_safe = request.form.get('pregnancy_safe') == 'true'
        prescription.is_active = request.form.get('is_active') == 'true'
        
        db.session.commit()
        flash('Prescription updated successfully!', 'success')
        return redirect(url_for('prescriptions.view_prescription', id=prescription.id))
    
    return render_template('prescriptions/edit.html', prescription=prescription, patients=patients)

@prescriptions_bp.route('/<int:id>/toggle', methods=['POST'])
@login_required
def toggle_prescription(id):
    """Toggle prescription active status."""
    prescription = Prescription.query.get_or_404(id)
    prescription.is_active = not prescription.is_active
    db.session.commit()
    
    status = 'activated' if prescription.is_active else 'deactivated'
    flash(f'Prescription {status} successfully.', 'success')
    return redirect(url_for('prescriptions.view_prescription', id=id))

@prescriptions_bp.route('/<int:id>/delete', methods=['DELETE'])
@login_required
def delete_prescription(id):
    """Delete a prescription."""
    prescription = Prescription.query.get_or_404(id)
    db.session.delete(prescription)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Prescription deleted successfully'})

@prescriptions_bp.route('/<int:id>/deactivate', methods=['POST'])
@login_required
def deactivate_prescription(id):
    """Deactivate a prescription."""
    prescription = Prescription.query.get_or_404(id)
    prescription.is_active = False
    db.session.commit()
    
    flash('Prescription deactivated successfully.', 'info')
    return redirect(url_for('prescriptions.view_prescription', id=id))

@prescriptions_bp.route('/patient/<int:patient_id>')
@login_required
def patient_prescriptions(patient_id):
    """Get all prescriptions for a specific patient."""
    patient = Patient.query.get_or_404(patient_id)
    prescriptions = Prescription.query.filter_by(patient_id=patient_id).order_by(
        Prescription.prescribed_date.desc()
    ).all()
    
    return render_template('prescriptions/patient_prescriptions.html',
                         patient=patient,
                         prescriptions=prescriptions)

@prescriptions_bp.route('/check-drug-interactions')
@login_required
def check_drug_interactions():
    """Check for potential drug interactions."""
    patient_id = request.args.get('patient_id', type=int)
    
    if not patient_id:
        return jsonify({'error': 'Patient ID required'}), 400
    
    # Get active prescriptions for patient
    active_prescriptions = Prescription.query.filter_by(
        patient_id=patient_id,
        is_active=True
    ).all()
    
    # In a real application, you would integrate with a drug interaction API
    # For now, we'll return a simple response
    medications = [p.medication_name for p in active_prescriptions]
    
    return jsonify({
        'patient_id': patient_id,
        'active_medications': medications,
        'interactions': [],  # Would be populated by drug interaction service
        'warnings': []
    })
