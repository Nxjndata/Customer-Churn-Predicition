from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Customer Churn Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['feature1'], data['feature2']])  # Adjust based on your model
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
