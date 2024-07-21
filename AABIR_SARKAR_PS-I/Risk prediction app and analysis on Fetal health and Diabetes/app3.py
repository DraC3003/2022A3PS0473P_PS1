from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import requests

app = Flask(__name__)

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
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5&q={query}&key={api_key}'
    response = requests.get(url)
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
    return render_template('result.html', risk=risk, probability=probability, videos=videos)

if __name__ == "__main__":
    app.run(debug=True)
