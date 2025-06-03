import os
import pandas as pd
import pickle
import logging
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_data(df):
    """Adatminőség ellenőrzése"""
    logger.info("Adat validáció indítása...")

    if df.isnull().any().any():
        raise ValueError("Hiányzó értékek találhatók az adatban!")

    if len(df) < 1000:
        raise ValueError(f"Az adathalmaz túl kicsi: {len(df)} sor")

    spam_ratio = (df['label'] == 'spam').mean()
    if not (0.05 < spam_ratio < 0.5):
        logger.warning(f"Szokatlan spam arány: {spam_ratio:.3f}")

    logger.info(f"Validáció sikeres. Adatméret: {len(df)}, Spam arány: {spam_ratio:.3f}")
    return True

def save_metrics(metrics, filepath='artifacts/training_metrics.json'):
    """Metrikák mentése JSON fájlba"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Timestamp hozzáadása
    metrics['timestamp'] = datetime.now().isoformat()

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrikák mentve: {filepath}")

def main():
    start_time = datetime.now()
    logger.info("Training script indítása")

    try:
        os.makedirs('artifacts', exist_ok=True)

        data_path = os.path.join('data', 'spam.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Nem található a dataset: {data_path}. "
                "Töltsd le a Kaggle-ről, és helyezd a data/ mappába."
            )
        logger.info(f"Adat betöltése: {data_path}")

        df = pd.read_csv(
            data_path,
            encoding='latin-1',
            usecols=['v1', 'v2']
        )
        df.columns = ['label', 'text']

        logger.info(f"Adat betöltve: {len(df)} sor")

        validate_data(df)

        # Label encoding
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

        # Train/test split
        X = df['text']
        y = df['label_num']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train méret: {X_train.shape[0]}, Test méret: {X_test.shape[0]}")
        logger.info(f"Label eloszlás:\n{df['label'].value_counts().to_string()}")

        # Vektorizálás
        logger.info("Vektorizálás...")
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 1),
            max_features=10000
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        logger.info(f"Vocabulary méret: {len(vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix alakja: {X_train_vec.shape}")

        # Model training
        logger.info("Model tanítása...")
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_vec, y_train)

        # Predikció és kiértékelés
        logger.info("Model kiértékelése...")
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)

        # Metrikák számolása
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)

        print("\n=== TRAINING EREDMÉNYEK ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
        print(f"\nConfusion Matrix:")
        print(f"[[TN: {conf_matrix[0, 0]}, FP: {conf_matrix[0, 1]}],")
        print(f" [FN: {conf_matrix[1, 0]}, TP: {conf_matrix[1, 1]}]]")

        # Metrikák mentése
        metrics = {
            'model_version': f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'accuracy': float(accuracy),
            'auc_roc': float(auc_score),
            'precision_spam': float(class_report['spam']['precision']),
            'recall_spam': float(class_report['spam']['recall']),
            'f1_spam': float(class_report['spam']['f1-score']),
            'confusion_matrix': conf_matrix.tolist(),
            'train_size': int(X_train.shape[0]),
            'test_size': int(X_test.shape[0]),
            'vocabulary_size': int(len(vectorizer.vocabulary_)),
            'training_time_seconds': (datetime.now() - start_time).total_seconds()
        }

        save_metrics(metrics)

        # Model mentése
        artifact_path = os.path.join('artifacts', 'spam_model.pkl')
        model_data = {
            'vectorizer': vectorizer,
            'model': model,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'accuracy': accuracy,
                'auc_roc': auc_score,
                'model_version': metrics['model_version']
            }
        }

        with open(artifact_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model sikeresen mentve: {artifact_path}")
        logger.info(f"Training befejezve. Összes idő: {(datetime.now() - start_time).total_seconds():.2f} másodperc")

        return metrics

    except Exception as e:
        logger.error(f"Hiba a training során: {str(e)}")
        raise



if __name__ == '__main__':
    main()
