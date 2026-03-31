from flask import Flask, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved model and data
with open('restaurant_model.pkl', 'rb') as f:
    tfidf, tfidf_matrix, df = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_prompt = data.get('prompt', '')

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # 1. Transform user prompt
    user_vec = tfidf.transform([user_prompt])

    # 2. Calculate Similarity
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    
    # 3. Get top 5 recommendations
    similar_indices = similarity_scores.argsort()[0][-5:][::-1]
    
    results = []
    for i in similar_indices:
        results.append({
            "name": df.iloc[i]['Restaurant Name'],
            "cuisine": df.iloc[i]['Cuisines'],
            "location": f"{df.iloc[i]['Address']}, {df.iloc[i]['City']}",
            "rating": df.iloc[i]['Aggregate rating']
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)