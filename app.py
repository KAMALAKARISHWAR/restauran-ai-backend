import os
import pickle
import numpy as np
import tensorflow as tf
import tf_keras as keras
from flask import Flask, request, jsonify
from flask_cors import CORS
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def get_resources():
    global model, tokenizer
    if model is None:
        print("--- Manually Reconstructing Model Structure ---")
        try:
            # 1. Define the layers manually to match your training exactly.
            # Based on your error log (maxlen=10), this is the standard structure:
            model = Sequential([
                Input(shape=(10,), name="input_layer"),
                Embedding(input_dim=1000, output_dim=16), 
                GlobalAveragePooling1D(),
                Dense(24, activation='relu'),
                Dense(4, activation='softmax') # 4 classes for your 4 restaurants
            ])
            
            # 2. Load ONLY the weights. This bypasses the InputLayer config error.
            model.load_weights('restaurant_model.h5')
            print("--- Weights Loaded Successfully ---")
            
        except Exception as e:
            print(f"Weight load failed: {e}")
            # Fallback: try one more time with standard load if reconstruction fails
            model = keras.models.load_model('restaurant_model.h5', compile=False)
            
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

        seq = ai_tokenizer.texts_to_sequences([prompt])
        padded = pad_sequences(seq, maxlen=10)
        
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
