from flask import Flask, request, jsonify
import pickle
from utils import clean_text

app = Flask(__name__)

model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = clean_text(data['text'])
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({"prediction": pred})

app.run()