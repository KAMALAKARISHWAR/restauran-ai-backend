import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
# Use standard TensorFlow Keras imports
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def get_resources():
    global model, tokenizer
    if model is None:
        print("--- Loading AI Model (.keras format) ---")
        # Modern Keras files load directly with standard tf.keras
        # This handles all the InputLayer config automatically
        model = tf.keras.models.load_model('restaurant_model.keras', compile=False)
        print("--- Model Loaded Successfully ---")
        
    if tokenizer is None:
        print("--- Loading Tokenizer ---")
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    return model, tokenizer

DATABASE = {
    0: {"name": "L'Amour Bistro", "cuisine": "French", "rating": 4.9},
    1: {"name": "Fire & Spice", "cuisine": "Thai/Indian", "rating": 4.3},
    2: {"name": "The Green Garden", "cuisine": "Vegan/Organic", "rating": 4.7},
    3: {"name": "Brew & Work", "cuisine": "Cafe/Bakery", "rating": 4.5}
}

@app.route('/')
def health_check():
    return "AI Server is Live", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ai_model, ai_tokenizer = get_resources()
        user_data = request.json
        prompt = user_data.get('prompt', '')

        # Preprocessing
        seq = ai_tokenizer.texts_to_sequences([prompt])
        padded = pad_sequences(seq, maxlen=10)
        
        # Prediction
        prediction = ai_model.predict(padded)[0]

        results = []
        for i, confidence in enumerate(prediction):
            if confidence > 0.1 and i in DATABASE:
                res = DATABASE[i].copy()
                res['match'] = f"{int(confidence * 100)}%"
                results.append(res)
        
        results = sorted(results, key=lambda x: x['match'], reverse=True)
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
