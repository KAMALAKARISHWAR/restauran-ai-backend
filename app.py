import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def get_resources():
    global model, tokenizer
    if model is None:
        print("--- Manually Reconstructing Model ---")
        # We build the structure EXACTLY as shown in your error log
        model = Sequential([
            Input(shape=(10,), name="input_layer"),
            Embedding(input_dim=1000, output_dim=16), 
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'), # Changed to 16 to match your log
            Dense(4, activation='softmax')
        ])
        
        # Load weights only. This ignores 'quantization_config' and all other errors.
        # We use the .h5 file for weights as it is more stable for this method.
        try:
            model.load_weights('restaurant_model.h5')
            print("--- Weights Loaded Successfully ---")
        except:
            # Fallback for the .keras file if you deleted the .h5
            model.load_weights('restaurant_model.keras')
            print("--- Weights Loaded from .keras Successfully ---")
            
    if tokenizer is None:
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
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
