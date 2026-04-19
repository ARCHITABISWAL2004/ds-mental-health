from flask import Flask, request, jsonify
import pickle
import os
from utils import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    text = clean_text(data['text'])
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return jsonify({"prediction": pred})


# IMPORTANT: only run locally, NOT on Render/Gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)