from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app) # Allows your Android app to connect

# Load the trained model and tokenizer
model = tf.keras.models.load_model('restaurant_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Mock Database for the results
DATABASE = {
    0: {"name": "L'Amour Bistro", "cuisine": "French", "rating": 4.9, "match": "98%"},
    1: {"name": "Fire & Spice", "cuisine": "Thai/Indian", "rating": 4.3, "match": "92%"},
    2: {"name": "The Green Garden", "cuisine": "Vegan/Organic", "rating": 4.7, "match": "95%"},
    3: {"name": "Brew & Work", "cuisine": "Cafe/Bakery", "rating": 4.5, "match": "89%"}
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        prompt = user_data.get('prompt', '')

        # ML Prediction
        seq = tokenizer.texts_to_sequences([prompt])
        padded = pad_sequences(seq, maxlen=10)
        prediction = model.predict(padded)[0] # Get the first result array

        # Get all results where AI confidence is > 10%
        results = []
        for i, confidence in enumerate(prediction):
            if confidence > 0.1:
                res = DATABASE[i].copy()
                res['match'] = f"{int(confidence * 100)}%"
                results.append(res)
        
        # Sort by highest match first
        results = sorted(results, key=lambda x: x['match'], reverse=True)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)