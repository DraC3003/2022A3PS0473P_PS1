#!/usr/bin/env python3
"""
Healthcare App Entry Point
Run this file to start the Flask application.
"""

import os
import sys
from flask.cli import FlaskGroup
from app import create_app, db
from app.models import User, Patient, Prescription, RiskAssessment

# Create Flask app instance
app = create_app(os.getenv('FLASK_CONFIG') or 'default')

def init_db():
    """Initialize the database with tables."""
    print("Creating database tables...")
    db.create_all()
    print("Database initialized successfully!")

def seed_db():
    """Seed the database with sample data."""
    print("Seeding database with sample data...")
    
    # Create a sample doctor user
    if not User.query.filter_by(email='doctor@example.com').first():
        doctor = User(
            username='Dr. Smith',
            email='doctor@example.com',
            role='doctor'
        )
        doctor.set_password('password123')
        db.session.add(doctor)
    
    # Create sample patients
    if not Patient.query.first():
        from datetime import date
        
        patient1 = Patient(
            first_name='Jane',
            last_name='Doe',
            email='jane.doe@example.com',
            phone='555-0101',
            date_of_birth=date(1990, 5, 15),
            gender='Female',
            is_pregnant=True,
            pregnancy_start_date=date(2025, 1, 1),
            expected_due_date=date(2025, 10, 1)
        )
        
        patient2 = Patient(
            first_name='Mary',
            last_name='Johnson',
            email='mary.johnson@example.com',
            phone='555-0102',
            date_of_birth=date(1985, 8, 22),
            gender='Female',
            is_pregnant=False
        )
        
        db.session.add(patient1)
        db.session.add(patient2)
    
    db.session.commit()
    print("Database seeded successfully!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'init-db':
            with app.app_context():
                init_db()
        elif sys.argv[1] == 'seed-db':
            with app.app_context():
                seed_db()
        else:
            print("Available commands:")
            print("  init-db  : Initialize database tables")
            print("  seed-db  : Seed database with sample data")
    else:
        # Run the Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
