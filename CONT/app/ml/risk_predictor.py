"""
Pregnancy Risk Predictor using Machine Learning.
Predicts risk of various pregnancy complications and optimal checkup timing.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta

class PregnancyRiskPredictor:
    """Machine learning model for predicting pregnancy risks."""
    
    def __init__(self):
        """Initialize the risk predictor."""
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'pregnancy_week', 'maternal_age', 'bmi', 'systolic_bp', 'diastolic_bp',
            'glucose_level', 'protein_in_urine', 'previous_complications',
            'smoking_status', 'alcohol_consumption'
        ]
        
        # Initialize or load models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with sample data or load existing models."""
        model_path = 'app/ml/models'
        
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Try to load existing models
        try:
            self.models['preeclampsia'] = joblib.load(f'{model_path}/preeclampsia_model.pkl')
            self.models['gestational_diabetes'] = joblib.load(f'{model_path}/diabetes_model.pkl')
            self.models['preterm_birth'] = joblib.load(f'{model_path}/preterm_model.pkl')
            self.scalers['features'] = joblib.load(f'{model_path}/feature_scaler.pkl')
            print("Loaded existing ML models.")
        except FileNotFoundError:
            # Create and train new models with synthetic data
            print("Creating new ML models with synthetic data...")
            self._create_synthetic_data_and_train()
            self._save_models(model_path)
    
    def _create_synthetic_data_and_train(self):
        """Create synthetic training data and train models."""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate synthetic pregnancy data
        data = {
            'pregnancy_week': np.random.randint(1, 41, n_samples),
            'maternal_age': np.random.normal(28, 6, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'systolic_bp': np.random.normal(120, 15, n_samples),
            'diastolic_bp': np.random.normal(80, 10, n_samples),
            'glucose_level': np.random.normal(100, 20, n_samples),
            'protein_in_urine': np.random.binomial(1, 0.1, n_samples),
            'previous_complications': np.random.binomial(1, 0.15, n_samples),
            'smoking_status': np.random.binomial(1, 0.1, n_samples),
            'alcohol_consumption': np.random.binomial(1, 0.05, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['maternal_age'] = np.clip(df['maternal_age'], 15, 50)
        df['bmi'] = np.clip(df['bmi'], 15, 50)
        df['systolic_bp'] = np.clip(df['systolic_bp'], 90, 180)
        df['diastolic_bp'] = np.clip(df['diastolic_bp'], 60, 120)
        df['glucose_level'] = np.clip(df['glucose_level'], 70, 200)
        
        # Generate realistic target variables based on risk factors
        # Preeclampsia risk (higher with age, high BP, protein in urine)
        preeclampsia_prob = (
            0.05 +  # Base rate
            0.002 * (df['maternal_age'] - 25).clip(0, 20) +  # Age factor
            0.003 * (df['systolic_bp'] - 120).clip(0, 60) +  # BP factor
            0.15 * df['protein_in_urine'] +  # Protein factor
            0.05 * df['previous_complications']  # History factor
        )
        df['preeclampsia'] = np.random.binomial(1, np.clip(preeclampsia_prob, 0, 1))
        
        # Gestational diabetes risk (higher with age, BMI, glucose)
        diabetes_prob = (
            0.08 +  # Base rate
            0.003 * (df['maternal_age'] - 25).clip(0, 20) +  # Age factor
            0.002 * (df['bmi'] - 25).clip(0, 25) +  # BMI factor
            0.002 * (df['glucose_level'] - 100).clip(0, 100) +  # Glucose factor
            0.03 * df['previous_complications']  # History factor
        )
        df['gestational_diabetes'] = np.random.binomial(1, np.clip(diabetes_prob, 0, 1))
        
        # Preterm birth risk (higher with smoking, complications, extreme ages)
        preterm_prob = (
            0.10 +  # Base rate
            0.002 * np.abs(df['maternal_age'] - 30) +  # Age extremes
            0.08 * df['smoking_status'] +  # Smoking factor
            0.06 * df['alcohol_consumption'] +  # Alcohol factor
            0.10 * df['previous_complications'] +  # History factor
            0.002 * (df['systolic_bp'] - 120).clip(0, 60)  # BP factor
        )
        df['preterm_birth'] = np.random.binomial(1, np.clip(preterm_prob, 0, 1))
        
        # Prepare features
        X = df[self.feature_names]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Train models
        self.models['preeclampsia'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['preeclampsia'].fit(X_scaled, df['preeclampsia'])
        
        self.models['gestational_diabetes'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['gestational_diabetes'].fit(X_scaled, df['gestational_diabetes'])
        
        self.models['preterm_birth'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['preterm_birth'].fit(X_scaled, df['preterm_birth'])
        
        print("Trained ML models with synthetic data.")
    
    def _save_models(self, model_path):
        """Save trained models to disk."""
        joblib.dump(self.models['preeclampsia'], f'{model_path}/preeclampsia_model.pkl')
        joblib.dump(self.models['gestational_diabetes'], f'{model_path}/diabetes_model.pkl')
        joblib.dump(self.models['preterm_birth'], f'{model_path}/preterm_model.pkl')
        joblib.dump(self.scalers['features'], f'{model_path}/feature_scaler.pkl')
        print("Saved ML models to disk.")
    
    def predict_risks(self, input_data):
        """
        Predict pregnancy risks for given input data.
        
        Args:
            input_data (dict): Dictionary containing feature values
            
        Returns:
            dict: Predictions for various risks
        """
        # Prepare input features
        features = []
        for feature in self.feature_names:
            value = input_data.get(feature, 0)
            features.append(float(value))
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Get predictions and probabilities
        preeclampsia_prob = self.models['preeclampsia'].predict_proba(X_scaled)[0][1]
        diabetes_prob = self.models['gestational_diabetes'].predict_proba(X_scaled)[0][1]
        preterm_prob = self.models['preterm_birth'].predict_proba(X_scaled)[0][1]
        
        # Calculate overall risk score
        overall_risk = (preeclampsia_prob * 0.4 + diabetes_prob * 0.3 + preterm_prob * 0.3)
        
        # Determine risk level
        if overall_risk < 0.2:
            risk_level = 'Low'
        elif overall_risk < 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Adjust risk based on pregnancy week (some complications more likely at certain times)
        week = input_data.get('pregnancy_week', 20)
        
        # Preeclampsia typically develops after 20 weeks
        if week < 20:
            preeclampsia_prob *= 0.3
        elif week > 32:
            preeclampsia_prob *= 1.5
        
        # Gestational diabetes typically screened 24-28 weeks
        if week > 24:
            diabetes_prob *= 1.2
        
        # Preterm birth risk increases in later weeks if other factors present
        if week > 30 and overall_risk > 0.3:
            preterm_prob *= 1.3
        
        return {
            'preeclampsia_risk': round(preeclampsia_prob, 3),
            'gestational_diabetes_risk': round(diabetes_prob, 3),
            'preterm_birth_risk': round(preterm_prob, 3),
            'overall_risk': round(overall_risk, 3),
            'risk_level': risk_level
        }
    
    def generate_recommendations(self, predictions, input_data):
        """
        Generate medical recommendations based on risk predictions.
        
        Args:
            predictions (dict): Risk predictions
            input_data (dict): Input data used for predictions
            
        Returns:
            str: Formatted recommendations
        """
        recommendations = []
        
        week = input_data.get('pregnancy_week', 20)
        
        # General recommendations
        if predictions['overall_risk'] > 0.4:
            recommendations.append("Schedule frequent monitoring appointments")
            recommendations.append("Consider consultation with maternal-fetal medicine specialist")
        
        # Preeclampsia specific
        if predictions['preeclampsia_risk'] > 0.3:
            recommendations.append("Monitor blood pressure daily")
            recommendations.append("Watch for symptoms: severe headaches, vision changes, upper abdominal pain")
            recommendations.append("Reduce sodium intake")
            if week > 34:
                recommendations.append("Consider delivery planning discussion")
        
        # Gestational diabetes specific
        if predictions['gestational_diabetes_risk'] > 0.25:
            recommendations.append("Schedule glucose tolerance test")
            recommendations.append("Monitor diet and carbohydrate intake")
            recommendations.append("Regular blood glucose monitoring")
            recommendations.append("Consider nutritionist consultation")
        
        # Preterm birth specific
        if predictions['preterm_birth_risk'] > 0.2:
            recommendations.append("Monitor for signs of preterm labor")
            recommendations.append("Reduce physical activity and stress")
            recommendations.append("Consider progesterone supplementation")
            if week > 32:
                recommendations.append("Discuss steroid injections for fetal lung maturity")
        
        # Lifestyle recommendations
        if input_data.get('smoking_status', 0):
            recommendations.append("URGENT: Smoking cessation counseling and support")
        
        if input_data.get('alcohol_consumption', 0):
            recommendations.append("Completely avoid alcohol consumption")
        
        bmi = input_data.get('bmi', 25)
        if bmi > 30:
            recommendations.append("Nutritional counseling for weight management")
        elif bmi < 18.5:
            recommendations.append("Nutritional support for healthy weight gain")
        
        # Blood pressure recommendations
        systolic = input_data.get('systolic_bp', 120)
        if systolic > 140:
            recommendations.append("Antihypertensive medication evaluation")
            recommendations.append("Daily blood pressure monitoring")
        
        if not recommendations:
            recommendations.append("Continue routine prenatal care")
            recommendations.append("Maintain healthy diet and regular exercise")
            recommendations.append("Take prenatal vitamins as prescribed")
        
        return "; ".join(recommendations)
    
    def recommend_next_checkup(self, predictions, current_week):
        """
        Recommend when the next checkup should be scheduled.
        
        Args:
            predictions (dict): Risk predictions
            current_week (int): Current pregnancy week
            
        Returns:
            datetime.date: Recommended next checkup date
        """
        base_interval = 4  # weeks
        
        # Adjust interval based on risk level
        if predictions['risk_level'] == 'High':
            interval = 1  # Weekly
        elif predictions['risk_level'] == 'Medium':
            interval = 2  # Bi-weekly
        else:
            # Standard care intervals
            if current_week < 28:
                interval = 4  # Monthly
            elif current_week < 36:
                interval = 2  # Bi-weekly
            else:
                interval = 1  # Weekly
        
        # Calculate next checkup date
        next_date = datetime.now().date() + timedelta(weeks=interval)
        return next_date
    
    def get_feature_importance(self, condition='preeclampsia'):
        """
        Get feature importance for a specific condition.
        
        Args:
            condition (str): Condition to analyze ('preeclampsia', 'gestational_diabetes', 'preterm_birth')
            
        Returns:
            dict: Feature importance scores
        """
        if condition not in self.models:
            return {}
        
        model = self.models[condition]
        importance_scores = model.feature_importances_
        
        return dict(zip(self.feature_names, importance_scores))
    
    def explain_prediction(self, input_data, predictions):
        """
        Provide explanation for predictions.
        
        Args:
            input_data (dict): Input data
            predictions (dict): Predictions
            
        Returns:
            str: Explanation text
        """
        explanations = []
        
        # Analyze key risk factors
        age = input_data.get('maternal_age', 25)
        if age > 35:
            explanations.append(f"Advanced maternal age ({age} years) increases risk")
        elif age < 18:
            explanations.append(f"Young maternal age ({age} years) may increase risk")
        
        bmi = input_data.get('bmi', 25)
        if bmi > 30:
            explanations.append(f"High BMI ({bmi:.1f}) is a risk factor")
        
        bp_systolic = input_data.get('systolic_bp', 120)
        if bp_systolic > 140:
            explanations.append(f"High blood pressure ({bp_systolic} mmHg) increases risk")
        
        if input_data.get('protein_in_urine', 0):
            explanations.append("Protein in urine is concerning for preeclampsia")
        
        if input_data.get('previous_complications', 0):
            explanations.append("Previous pregnancy complications increase risk")
        
        if input_data.get('smoking_status', 0):
            explanations.append("Smoking significantly increases pregnancy risks")
        
        week = input_data.get('pregnancy_week', 20)
        if week > 34 and predictions['overall_risk'] > 0.3:
            explanations.append("Late pregnancy with risk factors requires close monitoring")
        
        if not explanations:
            explanations.append("No significant risk factors identified")
        
        return "; ".join(explanations)
