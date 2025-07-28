from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import requests
from database import init_database, save_consultation, get_all_consultations, get_consultation_by_id, get_consultation_stats

app = Flask(__name__)

# Initialize the database
init_database()

# Load the dataset and extract feature names
data = pd.read_csv('diabetes.csv')
feature_names = data.columns[:-1]  # Exclude the 'Outcome' column

# Separate features and target
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Define function to get YouTube videos
def get_youtube_videos(query, api_key):
    try:
        url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5&q={query}&key={api_key}'
        response = requests.get(url, timeout=5)
        videos = response.json().get('items', [])
        
        video_details = []
        for video in videos:
            video_id = video['id'].get('videoId')
            if video_id:
                video_details.append({
                    'title': video['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                })
        return video_details
    except Exception as e:
        # Return empty list if YouTube API fails
        print(f"YouTube API error: {e}")
        return []

# Define function to predict and recommend
def predict_and_recommend(features, model, scaler, api_key):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Make a prediction
    risk = model.predict(features_scaled)[0]
    
    # Get the probability of the positive class (index 1)
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Determine the query based on the risk
    if risk == 1:
        query ='lenest diabetes'
    else:
        query = 'lenest'
    
    # Get relevant YouTube videos
    videos = get_youtube_videos(query, api_key)
    
    return risk, probability, videos

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[feature]) for feature in feature_names]
    api_key = 'enter ur api key'  # Replace with your actual API key
    risk, probability, videos = predict_and_recommend(features, model, scaler, api_key)
    
    # Save consultation data to database
    consultation_id = save_consultation(features, risk, probability)
    
    return render_template('result.html', risk=risk, probability=probability, videos=videos, consultation_id=consultation_id)

@app.route('/consultations')
def consultations():
    """Display all stored consultations."""
    consultations = get_all_consultations()
    stats = get_consultation_stats()
    return render_template('consultations.html', consultations=consultations, stats=stats, feature_names=feature_names)

@app.route('/consultation/<int:consultation_id>')
def consultation_detail(consultation_id):
    """Display details of a specific consultation."""
    consultation = get_consultation_by_id(consultation_id)
    if consultation:
        return render_template('consultation_detail.html', consultation=consultation, feature_names=feature_names)
    else:
        return redirect(url_for('consultations'))

if __name__ == "__main__":
    app.run(debug=True)
