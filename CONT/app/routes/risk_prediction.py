"""
Risk prediction routes using machine learning.
Handles pregnancy disease risk assessment and prediction.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from datetime import datetime, date
import numpy as np
from app import db
from app.models import Patient, RiskAssessment
from app.ml.risk_predictor import PregnancyRiskPredictor

risk_bp = Blueprint('risk', __name__)

@risk_bp.route('/')
@login_required
def list_assessments():
    """List all risk assessments."""
    page = request.args.get('page', 1, type=int)
    patient_id = request.args.get('patient_id', type=int)
    
    query = RiskAssessment.query
    
    if patient_id:
        query = query.filter_by(patient_id=patient_id)
    
    assessments = query.order_by(RiskAssessment.assessment_date.desc()).paginate(
        page=page, per_page=15, error_out=False
    )
    
    # Get all patients for the filter dropdown
    patients = Patient.query.all()
    
    return render_template('risk/list.html', assessments=assessments, patients=patients)

@risk_bp.route('/assess', methods=['GET', 'POST'])
@login_required
def assess_risk():
    """Assess pregnancy risk for a patient."""
    patient_id = request.args.get('patient_id', type=int)
    patients = Patient.query.filter_by(is_pregnant=True).all()
    
    if request.method == 'POST':
        # Get form data
        patient_id = request.form.get('patient_id', type=int)
        pregnancy_week = request.form.get('pregnancy_week', type=int)
        maternal_age = request.form.get('maternal_age', type=int)
        bmi = request.form.get('bmi', type=float)
        systolic_bp = request.form.get('systolic_bp', type=int)
        diastolic_bp = request.form.get('diastolic_bp', type=int)
        glucose_level = request.form.get('glucose_level', type=float)
        protein_in_urine = bool(request.form.get('protein_in_urine'))
        previous_complications = bool(request.form.get('previous_complications'))
        family_history = request.form.get('family_history')
        smoking_status = request.form.get('smoking_status')
        alcohol_consumption = request.form.get('alcohol_consumption')
        
        # Validation
        if not patient_id or not pregnancy_week:
            flash('Patient and pregnancy week are required.', 'error')
            return render_template('risk/assess.html', patients=patients, patient_id=patient_id)
        
        patient = Patient.query.get(patient_id)
        if not patient:
            flash('Invalid patient selected.', 'error')
            return render_template('risk/assess.html', patients=patients)
        
        if not patient.is_pregnant:
            flash('Selected patient is not currently pregnant.', 'error')
            return render_template('risk/assess.html', patients=patients)
        
        # Use ML model for risk prediction
        try:
            predictor = PregnancyRiskPredictor()
            
            # Prepare input data for ML model
            input_data = {
                'pregnancy_week': pregnancy_week,
                'maternal_age': maternal_age or patient.age,
                'bmi': bmi or 25.0,  # Default BMI if not provided
                'systolic_bp': systolic_bp or 120,
                'diastolic_bp': diastolic_bp or 80,
                'glucose_level': glucose_level or 100.0,
                'protein_in_urine': int(protein_in_urine),
                'previous_complications': int(previous_complications),
                'smoking_status': 1 if smoking_status == 'current' else 0,
                'alcohol_consumption': 1 if alcohol_consumption == 'regular' else 0
            }
            
            # Get predictions
            predictions = predictor.predict_risks(input_data)
            
            # Generate recommendations
            recommendations = predictor.generate_recommendations(predictions, input_data)
            
            # Calculate next checkup date
            next_checkup = predictor.recommend_next_checkup(predictions, pregnancy_week)
            
        except Exception as e:
            flash(f'Error in risk prediction: {str(e)}', 'error')
            return render_template('risk/assess.html', patients=patients, patient_id=patient_id)
        
        # Create risk assessment record
        assessment = RiskAssessment(
            patient_id=patient_id,
            assessed_by_id=current_user.id,
            pregnancy_week=pregnancy_week,
            maternal_age=maternal_age or patient.age,
            bmi=bmi,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            glucose_level=glucose_level,
            protein_in_urine=protein_in_urine,
            previous_complications=previous_complications,
            family_history=family_history,
            smoking_status=smoking_status,
            alcohol_consumption=alcohol_consumption,
            preeclampsia_risk=predictions['preeclampsia_risk'],
            gestational_diabetes_risk=predictions['gestational_diabetes_risk'],
            preterm_birth_risk=predictions['preterm_birth_risk'],
            overall_risk_score=predictions['overall_risk'],
            risk_level=predictions['risk_level'],
            recommendations=recommendations,
            next_checkup_date=next_checkup
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        flash('Risk assessment completed successfully!', 'success')
        return redirect(url_for('risk.view_assessment', id=assessment.id))
    
    return render_template('risk/assess.html', patients=patients, patient_id=patient_id)

@risk_bp.route('/<int:id>')
@login_required
def view_assessment(id):
    """View risk assessment details."""
    assessment = RiskAssessment.query.get_or_404(id)
    return render_template('risk/view.html', assessment=assessment)

@risk_bp.route('/timeline/<int:patient_id>')
@login_required
def risk_timeline(patient_id):
    """Show risk timeline for a pregnant patient."""
    patient = Patient.query.get_or_404(patient_id)
    
    if not patient.is_pregnant:
        flash('Patient is not currently pregnant.', 'error')
        return redirect(url_for('patients.view_patient', id=patient_id))
    
    # Get all assessments for this patient
    assessments = RiskAssessment.query.filter_by(patient_id=patient_id).order_by(
        RiskAssessment.pregnancy_week.asc()
    ).all()
    
    # Generate week-by-week risk predictions
    predictor = PregnancyRiskPredictor()
    weekly_predictions = []
    
    current_week = patient.pregnancy_week or 1
    
    for week in range(1, 41):  # Full pregnancy term
        # Use latest assessment data or defaults
        latest_assessment = assessments[-1] if assessments else None
        
        input_data = {
            'pregnancy_week': week,
            'maternal_age': latest_assessment.maternal_age if latest_assessment else patient.age,
            'bmi': latest_assessment.bmi if latest_assessment else 25.0,
            'systolic_bp': latest_assessment.systolic_bp if latest_assessment else 120,
            'diastolic_bp': latest_assessment.diastolic_bp if latest_assessment else 80,
            'glucose_level': latest_assessment.glucose_level if latest_assessment else 100.0,
            'protein_in_urine': int(latest_assessment.protein_in_urine) if latest_assessment else 0,
            'previous_complications': int(latest_assessment.previous_complications) if latest_assessment else 0,
            'smoking_status': 1 if latest_assessment and latest_assessment.smoking_status == 'current' else 0,
            'alcohol_consumption': 1 if latest_assessment and latest_assessment.alcohol_consumption == 'regular' else 0
        }
        
        try:
            predictions = predictor.predict_risks(input_data)
            weekly_predictions.append({
                'week': week,
                'predictions': predictions,
                'is_current': week == current_week
            })
        except:
            # Skip week if prediction fails
            continue
    
    return render_template('risk/timeline.html',
                         patient=patient,
                         assessments=assessments,
                         weekly_predictions=weekly_predictions)

@risk_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict_risk():
    """API endpoint for risk prediction."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        predictor = PregnancyRiskPredictor()
        predictions = predictor.predict_risks(data)
        recommendations = predictor.generate_recommendations(predictions, data)
        
        return jsonify({
            'predictions': predictions,
            'recommendations': recommendations,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@risk_bp.route('/high-risk-patients')
@login_required
def high_risk_patients():
    """List patients with high risk assessments."""
    high_risk_assessments = db.session.query(RiskAssessment).filter(
        RiskAssessment.risk_level == 'High'
    ).order_by(RiskAssessment.assessment_date.desc()).all()
    
    return render_template('risk/high_risk.html', assessments=high_risk_assessments)

@risk_bp.route('/analytics')
@login_required
def analytics():
    """Risk assessment analytics dashboard."""
    # Get statistics
    total_assessments = RiskAssessment.query.count()
    high_risk_count = RiskAssessment.query.filter_by(risk_level='High').count()
    medium_risk_count = RiskAssessment.query.filter_by(risk_level='Medium').count()
    low_risk_count = RiskAssessment.query.filter_by(risk_level='Low').count()
    
    # Risk distribution
    risk_distribution = {
        'High': high_risk_count,
        'Medium': medium_risk_count,
        'Low': low_risk_count
    }
    
    # Recent assessments
    recent_assessments = RiskAssessment.query.order_by(
        RiskAssessment.assessment_date.desc()
    ).limit(10).all()
    
    return render_template('risk/analytics.html',
                         total_assessments=total_assessments,
                         risk_distribution=risk_distribution,
                         recent_assessments=recent_assessments)
