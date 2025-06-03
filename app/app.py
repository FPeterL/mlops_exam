from flask import Flask, request, jsonify, render_template
import pickle
import os
import logging
import json
from datetime import datetime
import time
import threading
from collections import defaultdict, deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# Globális változók monitoringhoz
request_stats = {
    'total_requests': 0,
    'total_predictions': 0,
    'spam_predictions': 0,
    'ham_predictions': 0,
    'errors': 0,
    'response_times': deque(maxlen=1000)  # Utolsó 1000 response time
}

# Thread-safe counter
stats_lock = threading.Lock()

def load_model():
    """Model betöltése induláskor"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../artifacts/spam_model.pkl')

        if not os.path.exists(model_path):
            model_path = os.path.join('artifacts', 'spam_model.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model fájl nem található: {model_path}")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'vectorizer' in data and 'model' in data:
            vectorizer = data['vectorizer']
            model = data['model']
            metadata = data.get('metadata', {})

            logger.info(f"Model sikeresen betöltve: {model_path}")
            if metadata:
                logger.info(f"Model verzió: {metadata.get('model_version', 'N/A')}")
                logger.info(f"Training accuracy: {metadata.get('accuracy', 'N/A')}")
                logger.info(f"Tanítás dátuma: {metadata.get('trained_at', 'N/A')}")

            return vectorizer, model, metadata
        else:

            logger.warning("Régi model formátum észlelve")
            return data['vectorizer'], data['model'], {}

    except Exception as e:
        logger.error(f"Hiba a model betöltése során: {str(e)}")
        raise

def update_stats(prediction=None, response_time=None, error=False):
    """Thread-safe statisztika frissítés"""
    with stats_lock:
        request_stats['total_requests'] += 1

        if error:
            request_stats['errors'] += 1
        elif prediction is not None:
            request_stats['total_predictions'] += 1
            if prediction == 'spam':
                request_stats['spam_predictions'] += 1
            else:
                request_stats['ham_predictions'] += 1

        if response_time is not None:
            request_stats['response_times'].append(response_time)


def validate_input(data):
    """Bemenet validáció"""
    if not isinstance(data, dict):
        return False, "JSON objektum szükséges"

    message = data.get('message')
    if message is None:
        return False, "A 'message' mező kötelező"

    if not isinstance(message, str):
        return False, "A 'message' mező string típusú kell legyen"

    if len(message.strip()) == 0:
        return False, "Üres üzenet nem fogadható el"

    if len(message) > 1000:
        return False, "Az üzenet túl hosszú (max 1000 karakter)"

    return True, None

try:
    vectorizer, model, model_metadata = load_model()
    logger.info("Flask alkalmazás sikeresen inicializálva")
except Exception as e:
    logger.error(f"KRITIKUS HIBA: {str(e)}")
    vectorizer, model, model_metadata = None, None, {}

@app.route('/', methods=['GET', 'POST'])
def index():
    """Főoldal - HTML interface"""
    try:
        if request.method == 'GET':
            return render_template('index.html')

        elif request.method == 'POST':
            start_time = time.time()

            message = request.form.get('message', '').strip()

            if not message:
                return render_template('index.html', result={
                    'error': 'Kérlek írj be egy üzenetet!'
                })

            if len(message) > 1000:
                return render_template('index.html', result={
                    'error': 'Az üzenet túl hosszú (max 1000 karakter)!'
                })

            if model is None or vectorizer is None:
                return render_template('index.html', result={
                    'error': 'Model nem elérhető!'
                })

            try:
                X = vectorizer.transform([message])
                pred = model.predict(X)[0]
                pred_proba = model.predict_proba(X)[0]

                label = 'spam' if pred == 1 else 'ham'
                confidence = float(pred_proba[pred])

                response_time = time.time() - start_time

                logger.info(f"Web prediction: {label} (confidence: {confidence:.3f}) - "
                            f"Processing time: {response_time:.3f}s")

                update_stats(prediction=label, response_time=response_time)

                result = {
                    'message': message,
                    'label': label,
                    'probability': f"{confidence:.4f}",
                    'processing_time': f"{response_time:.3f}"
                }

                return render_template('index.html', result=result)

            except Exception as e:
                logger.error(f"Web prediction error: {str(e)}")
                update_stats(error=True)
                return render_template('index.html', result={
                    'error': f'Predikciós hiba: {str(e)}'
                })

    except Exception as e:
        logger.error(f"Template rendering error: {str(e)}")
        return jsonify({'error': 'Template nem található'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    start_time = time.time()

    try:
        if model is None or vectorizer is None:
            return jsonify({
                'status': 'unhealthy',
                'reason': 'Model not loaded'
            }), 503

        test_message = "test message"
        X_test = vectorizer.transform([test_message])
        _ = model.predict(X_test)

        response_time = time.time() - start_time
        update_stats(response_time=response_time)

        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_version': model_metadata.get('model_version', 'unknown'),
            'response_time_ms': round(response_time * 1000, 2)
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        update_stats(error=True)
        return jsonify({
            'status': 'unhealthy',
            'reason': str(e)
        }), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Predikciós endpoint"""
    start_time = time.time()

    try:
        if not request.is_json:
            update_stats(error=True)
            return jsonify({'error': 'Content-Type: application/json szükséges'}), 400

        req_data = request.get_json()
        is_valid, error_msg = validate_input(req_data)

        if not is_valid:
            update_stats(error=True)
            return jsonify({'error': error_msg}), 400

        message = req_data['message'].strip()

        if model is None or vectorizer is None:
            update_stats(error=True)
            return jsonify({'error': 'Model nem elérhető'}), 503

        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]

        label = 'spam' if pred == 1 else 'ham'
        confidence = float(pred_proba[pred])

        response_time = time.time() - start_time

        logger.info(f"Prediction: {label} (confidence: {confidence:.3f}) - "
                    f"Processing time: {response_time:.3f}s - "
                    f"Message length: {len(message)}")

        update_stats(prediction=label, response_time=response_time)

        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'ham': float(pred_proba[0]),
                'spam': float(pred_proba[1])
            },
            'processing_time_ms': round(response_time * 1000, 2),
            'message_length': len(message)
        }), 200

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Prediction error: {str(e)} - Processing time: {response_time:.3f}s")
        update_stats(error=True, response_time=response_time)

        return jsonify({
            'error': 'Belső szerver hiba',
            'processing_time_ms': round(response_time * 1000, 2)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Statisztikák lekérdezése"""
    try:
        with stats_lock:
            stats_copy = request_stats.copy()
            response_times = list(stats_copy['response_times'])

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0

        return jsonify({
            'total_requests': stats_copy['total_requests'],
            'total_predictions': stats_copy['total_predictions'],
            'spam_predictions': stats_copy['spam_predictions'],
            'ham_predictions': stats_copy['ham_predictions'],
            'errors': stats_copy['errors'],
            'error_rate': stats_copy['errors'] / max(stats_copy['total_requests'], 1),
            'spam_rate': stats_copy['spam_predictions'] / max(stats_copy['total_predictions'], 1),
            'response_times': {
                'avg_ms': round(avg_response_time * 1000, 2),
                'max_ms': round(max_response_time * 1000, 2),
                'min_ms': round(min_response_time * 1000, 2),
                'samples': len(response_times)
            },
            'model_info': {
                'version': model_metadata.get('model_version', 'unknown'),
                'accuracy': model_metadata.get('accuracy', 'unknown'),
                'trained_at': model_metadata.get('trained_at', 'unknown')
            }
        }), 200

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Statisztika lekérdezési hiba'}), 500

@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    """Statisztikák nullázása"""
    try:
        with stats_lock:
            request_stats.update({
                'total_requests': 0,
                'total_predictions': 0,
                'spam_predictions': 0,
                'ham_predictions': 0,
                'errors': 0,
                'response_times': deque(maxlen=1000)
            })

        logger.info("Statisztikák nullázva")
        return jsonify({'message': 'Statisztikák sikeresen nullázva'}), 200

    except Exception as e:
        logger.error(f"Stats reset error: {str(e)}")
        return jsonify({'error': 'Statisztika nullázási hiba'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint nem található'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Belső szerver hiba'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'

    logger.info(f"Flask alkalmazás indítása - Port: {port}, Debug: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)