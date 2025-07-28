import sqlite3
import json
from datetime import datetime
import os

DATABASE_PATH = 'consultations.db'

def init_database():
    """Initialize the database with the consultations table."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create consultations table to store doctor consultation factors
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pregnancies INTEGER,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            diabetes_pedigree_function REAL,
            age INTEGER,
            prediction_result INTEGER,
            prediction_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_consultation(consultation_data, prediction_result, prediction_probability):
    """Save consultation data and prediction result to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO consultations 
        (pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
         bmi, diabetes_pedigree_function, age, prediction_result, prediction_probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        consultation_data[0],  # pregnancies
        consultation_data[1],  # glucose
        consultation_data[2],  # blood_pressure
        consultation_data[3],  # skin_thickness
        consultation_data[4],  # insulin
        consultation_data[5],  # bmi
        consultation_data[6],  # diabetes_pedigree_function
        consultation_data[7],  # age
        int(prediction_result),  # Convert to int explicitly
        float(prediction_probability)  # Convert to float explicitly
    ))
    
    consultation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return consultation_id

def get_all_consultations():
    """Retrieve all consultations from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, consultation_date, pregnancies, glucose, blood_pressure, 
               skin_thickness, insulin, bmi, diabetes_pedigree_function, age,
               prediction_result, prediction_probability, created_at
        FROM consultations 
        ORDER BY created_at DESC
    ''')
    
    consultations = cursor.fetchall()
    conn.close()
    
    return consultations

def get_consultation_by_id(consultation_id):
    """Retrieve a specific consultation by ID."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, consultation_date, pregnancies, glucose, blood_pressure, 
               skin_thickness, insulin, bmi, diabetes_pedigree_function, age,
               prediction_result, prediction_probability, created_at
        FROM consultations 
        WHERE id = ?
    ''', (consultation_id,))
    
    consultation = cursor.fetchone()
    conn.close()
    
    return consultation

def get_consultation_stats():
    """Get basic statistics about consultations."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM consultations')
    total_consultations = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM consultations WHERE CAST(prediction_result AS INTEGER) = 1')
    positive_predictions = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(prediction_probability) FROM consultations')
    avg_probability = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_consultations': total_consultations,
        'positive_predictions': positive_predictions,
        'negative_predictions': total_consultations - positive_predictions,
        'avg_probability': avg_probability or 0
    }