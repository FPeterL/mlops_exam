# app/app.py
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# 1. Modell betöltése induláskor
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/spam_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    vectorizer = data['vectorizer']
    model = data['model']

@app.route('/health', methods=['GET'])
def health_check():
    """
    Egyszerű health check, pl. readiness probe-hoz Kubernetesben
    """
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON bemenet: {"message": "itt az SMS szöveg"}
    Kimenet: {"prediction": "spam" vagy "ham", "score": valami pl. valószínűség}
    """
    if not request.is_json:
        return jsonify({'error': 'JSON bemenet szükséges'}), 400

    req_data = request.get_json()
    message = req_data.get('message', None)
    if message is None:
        return jsonify({'error': 'Az üzenet mező (message) nincs megadva'}), 400

    # 2. Vektorizálás + predikció
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    # Ha szeretnél probabilisztikus kimenetet:
    proba = model.predict_proba(X)[0][pred]

    label = 'spam' if pred == 1 else 'ham'
    return jsonify({'prediction': label, 'probability': float(proba)}), 200

if __name__ == '__main__':
    # Lokális teszteléshez: flask run helyett:
    app.run(host='0.0.0.0', port=5000, debug=True)
