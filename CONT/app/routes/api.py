"""
API routes for healthcare app.
Provides REST API endpoints for external integrations.
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from app import db
from app.models import Patient, Prescription, RiskAssessment, User

api_bp = Blueprint('api', __name__)

@api_bp.route('/patients', methods=['GET'])
@login_required
def get_patients():
    """Get all patients."""
    patients = Patient.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.full_name,
        'email': p.email,
        'phone': p.phone,
        'age': p.age,
        'is_pregnant': p.is_pregnant,
        'pregnancy_week': p.pregnancy_week
    } for p in patients])

@api_bp.route('/patients', methods=['POST'])
@login_required
def create_patient():
    """Create new patient via API."""
    data = request.get_json()
    
    if not data or not data.get('first_name') or not data.get('last_name'):
        return jsonify({'error': 'First name and last name are required'}), 400
    
    patient = Patient(
        first_name=data.get('first_name'),
        last_name=data.get('last_name'),
        email=data.get('email'),
        phone=data.get('phone'),
        date_of_birth=datetime.strptime(data.get('date_of_birth'), '%Y-%m-%d').date() if data.get('date_of_birth') else None,
        gender=data.get('gender'),
        address=data.get('address'),
        is_pregnant=data.get('is_pregnant', False)
    )
    
    db.session.add(patient)
    db.session.commit()
    
    return jsonify({
        'id': patient.id,
        'message': 'Patient created successfully'
    }), 201

@api_bp.route('/patients/<int:patient_id>/prescriptions', methods=['GET'])
@login_required
def get_patient_prescriptions(patient_id):
    """Get prescriptions for a specific patient."""
    patient = Patient.query.get_or_404(patient_id)
    prescriptions = Prescription.query.filter_by(patient_id=patient_id).all()
    
    return jsonify([{
        'id': p.id,
        'medication_name': p.medication_name,
        'dosage': p.dosage,
        'frequency': p.frequency,
        'prescribed_date': p.prescribed_date.isoformat() if p.prescribed_date else None,
        'is_active': p.is_active,
        'doctor': p.doctor.username if p.doctor else None
    } for p in prescriptions])

@api_bp.route('/prescriptions', methods=['POST'])
@login_required
def create_prescription():
    """Create new prescription via API."""
    data = request.get_json()
    
    required_fields = ['patient_id', 'medication_name', 'dosage', 'frequency']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': 'Required fields: patient_id, medication_name, dosage, frequency'}), 400
    
    # Verify patient exists
    patient = Patient.query.get(data['patient_id'])
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    prescription = Prescription(
        patient_id=data['patient_id'],
        doctor_id=current_user.id,
        medication_name=data['medication_name'],
        dosage=data['dosage'],
        frequency=data['frequency'],
        duration=data.get('duration'),
        instructions=data.get('instructions'),
        diagnosis=data.get('diagnosis'),
        notes=data.get('notes')
    )
    
    db.session.add(prescription)
    db.session.commit()
    
    return jsonify({
        'id': prescription.id,
        'message': 'Prescription created successfully'
    }), 201

@api_bp.route('/risk-assessment', methods=['POST'])
@login_required
def create_risk_assessment():
    """Create risk assessment via API."""
    data = request.get_json()
    
    if not data or not data.get('patient_id'):
        return jsonify({'error': 'Patient ID is required'}), 400
    
    # Import here to avoid circular imports
    from app.ml.risk_predictor import PregnancyRiskPredictor
    
    try:
        predictor = PregnancyRiskPredictor()
        predictions = predictor.predict_risks(data)
        recommendations = predictor.generate_recommendations(predictions, data)
        
        assessment = RiskAssessment(
            patient_id=data['patient_id'],
            assessed_by_id=current_user.id,
            pregnancy_week=data.get('pregnancy_week'),
            maternal_age=data.get('maternal_age'),
            bmi=data.get('bmi'),
            systolic_bp=data.get('systolic_bp'),
            diastolic_bp=data.get('diastolic_bp'),
            glucose_level=data.get('glucose_level'),
            protein_in_urine=data.get('protein_in_urine', False),
            previous_complications=data.get('previous_complications', False),
            preeclampsia_risk=predictions['preeclampsia_risk'],
            gestational_diabetes_risk=predictions['gestational_diabetes_risk'],
            preterm_birth_risk=predictions['preterm_birth_risk'],
            overall_risk_score=predictions['overall_risk'],
            risk_level=predictions['risk_level'],
            recommendations=recommendations
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        return jsonify({
            'id': assessment.id,
            'predictions': predictions,
            'recommendations': recommendations,
            'message': 'Risk assessment completed successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/patients/<int:patient_id>/risk-timeline', methods=['GET'])
@login_required
def get_risk_timeline(patient_id):
    """Get risk timeline for a patient."""
    patient = Patient.query.get_or_404(patient_id)
    
    if not patient.is_pregnant:
        return jsonify({'error': 'Patient is not currently pregnant'}), 400
    
    assessments = RiskAssessment.query.filter_by(patient_id=patient_id).order_by(
        RiskAssessment.assessment_date.asc()
    ).all()
    
    timeline = []
    for assessment in assessments:
        timeline.append({
            'week': assessment.pregnancy_week,
            'date': assessment.assessment_date.isoformat(),
            'preeclampsia_risk': assessment.preeclampsia_risk,
            'gestational_diabetes_risk': assessment.gestational_diabetes_risk,
            'preterm_birth_risk': assessment.preterm_birth_risk,
            'overall_risk': assessment.overall_risk_score,
            'risk_level': assessment.risk_level
        })
    
    return jsonify({
        'patient_id': patient_id,
        'patient_name': patient.full_name,
        'pregnancy_week': patient.pregnancy_week,
        'timeline': timeline
    })

@api_bp.route('/stats', methods=['GET'])
@login_required
def get_stats():
    """Get application statistics."""
    stats = {
        'total_patients': Patient.query.count(),
        'pregnant_patients': Patient.query.filter_by(is_pregnant=True).count(),
        'total_prescriptions': Prescription.query.count(),
        'active_prescriptions': Prescription.query.filter_by(is_active=True).count(),
        'total_assessments': RiskAssessment.query.count(),
        'high_risk_patients': RiskAssessment.query.filter_by(risk_level='High').count(),
        'total_doctors': User.query.filter_by(role='doctor').count()
    }
    
    return jsonify(stats)

@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500
