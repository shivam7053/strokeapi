from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Random Forest Classifier
rf_classifier = joblib.load('stroke_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json

    # Extract features from JSON data
    features = data['features']

    # Convert features to a numerical array-like object
    feature_values = [features[key] for key in features]

    # Make prediction
    prediction = rf_classifier.predict([feature_values])

    # Return prediction as JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
