import os
import json
import pickle
import logging
import warnings
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score


warnings.filterwarnings('ignore')

# Logging setup
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
    metrics['timestamp'] = datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrikák mentve: {filepath}")


def compare_vectorizers(X_train, X_test, y_train, y_test):
    """Különböző vektorizálók összehasonlítása"""
    logger.info("Vektorizálók összehasonlítása...")

    vectorizers = {
        'CountVectorizer': CountVectorizer(lowercase=True, stop_words='english', max_features=10000),
        'TfidfVectorizer': TfidfVectorizer(lowercase=True, stop_words='english', max_features=10000)
    }

    vectorizer_results = {}

    for name, vectorizer in vectorizers.items():
        logger.info(f"Tesztelés: {name}")

        # Transform data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Quick test with Naive Bayes
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        vectorizer_results[name] = {
            'accuracy': accuracy,
            'vectorizer': vectorizer
        }
        logger.info(f"{name} accuracy: {accuracy:.4f}")

    # Legjobb vektorizáló kiválasztása
    best_vectorizer_name = max(vectorizer_results.keys(),
                               key=lambda x: vectorizer_results[x]['accuracy'])
    best_vectorizer = vectorizer_results[best_vectorizer_name]['vectorizer']

    logger.info(f"Legjobb vektorizáló: {best_vectorizer_name}")
    return best_vectorizer, vectorizer_results


def compare_models_with_hyperparameter_tuning(X_train_vec, X_test_vec, y_train, y_test):
    """Különböző modellek összehasonlítása hiperparaméter-optimalizálással"""
    logger.info("Modellek összehasonlítása hiperparaméter-optimalizálással...")

    # Modellek és hiperparamétereik
    models_config = {
        'MultinomialNB': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    }

    model_results = {}

    for model_name, config in models_config.items():
        logger.info(f"Optimalizálás: {model_name}")

        try:
            # Grid Search Cross Validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train_vec, y_train)

            # Legjobb modell kiértékelése
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_vec)
            y_pred_proba = best_model.predict_proba(X_test_vec)

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train_vec, y_train, cv=5, scoring='accuracy')

            model_results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            logger.info(
                f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            logger.info(f"Legjobb paraméterek: {grid_search.best_params_}")

        except Exception as e:
            logger.error(f"Hiba {model_name} modellnél: {str(e)}")
            continue

    return model_results


def main():
    start_time = datetime.now()
    logger.info("Enhanced Training script indítása")

    try:
        os.makedirs('artifacts', exist_ok=True)

        # Adat betöltése
        data_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(data_dir, 'data', 'spam.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Nem található a dataset: {data_path}. "
                "Töltsd le a Kaggle-ről, és helyezd a data/ mappába."
            )

        logger.info(f"Adat betöltése: {data_path}")
        df = pd.read_csv(data_path, encoding='latin-1', usecols=['v1', 'v2'])
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

        # 1. VEKTORIZÁLÓK ÖSSZEHASONLÍTÁSA
        best_vectorizer, vectorizer_results = compare_vectorizers(X_train, X_test, y_train, y_test)

        # Legjobb vektorizálóval történő transzformáció
        X_train_vec = best_vectorizer.fit_transform(X_train)
        X_test_vec = best_vectorizer.transform(X_test)

        logger.info(f"Vocabulary méret: {len(best_vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix alakja: {X_train_vec.shape}")

        # 2. MODELLEK ÖSSZEHASONLÍTÁSA HIPERPARAMÉTER-OPTIMALIZÁLÁSSAL
        model_results = compare_models_with_hyperparameter_tuning(X_train_vec, X_test_vec, y_train, y_test)

        # Legjobb modell kiválasztása
        if not model_results:
            raise ValueError("Egyik modell sem futott le sikeresen!")

        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        best_model_info = model_results[best_model_name]
        best_model = best_model_info['model']

        logger.info(f"LEGJOBB MODELL: {best_model_name}")
        logger.info(f"Accuracy: {best_model_info['accuracy']:.4f}")
        logger.info(f"AUC-ROC: {best_model_info['auc_score']:.4f}")
        logger.info(f"Cross-validation: {best_model_info['cv_mean']:.4f}±{best_model_info['cv_std']:.4f}")

        # Részletes kiértékelés a legjobb modellre
        y_pred = best_model_info['y_pred']
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)

        print("\n" + "=" * 50)
        print("VÉGSŐ EREDMÉNYEK")
        print("=" * 50)
        print(f"Legjobb modell: {best_model_name}")
        print(f"Legjobb paraméterek: {best_model_info['best_params']}")
        print(f"Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"AUC-ROC: {best_model_info['auc_score']:.4f}")
        print(f"Cross-validation: {best_model_info['cv_mean']:.4f}±{best_model_info['cv_std']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
        print(f"\nConfusion Matrix:")
        print(f"[[TN: {conf_matrix[0, 0]}, FP: {conf_matrix[0, 1]}],")
        print(f" [FN: {conf_matrix[1, 0]}, TP: {conf_matrix[1, 1]}]]")

        # Összes modell összehasonlítása
        print("\n" + "=" * 50)
        print("ÖSSZES MODELL ÖSSZEHASONLÍTÁSA")
        print("=" * 50)
        for model_name, results in model_results.items():
            print(
                f"{model_name:20s} | Accuracy: {results['accuracy']:.4f} | AUC: {results['auc_score']:.4f} | CV: {results['cv_mean']:.4f}±{results['cv_std']:.4f}")

        # Részletes metrikák mentése
        detailed_metrics = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'training_time_seconds': (datetime.now() - start_time).total_seconds(),
                'dataset_size': len(df),
                'train_size': int(X_train.shape[0]),
                'test_size': int(X_test.shape[0])
            },
            'vectorizer_comparison': {
                name: {'accuracy': float(results['accuracy'])}
                for name, results in vectorizer_results.items()
            },
            'best_vectorizer': type(best_vectorizer).__name__,
            'vocabulary_size': int(len(best_vectorizer.vocabulary_)),
            'model_comparison': {
                name: {
                    'accuracy': float(results['accuracy']),
                    'auc_score': float(results['auc_score']),
                    'cv_mean': float(results['cv_mean']),
                    'cv_std': float(results['cv_std']),
                    'best_params': results['best_params']
                }
                for name, results in model_results.items()
            },
            'best_model': {
                'name': best_model_name,
                'accuracy': float(best_model_info['accuracy']),
                'auc_roc': float(best_model_info['auc_score']),
                'cv_mean': float(best_model_info['cv_mean']),
                'cv_std': float(best_model_info['cv_std']),
                'best_params': best_model_info['best_params'],
                'precision_spam': float(class_report['spam']['precision']),
                'recall_spam': float(class_report['spam']['recall']),
                'f1_spam': float(class_report['spam']['f1-score']),
                'confusion_matrix': conf_matrix.tolist()
            },
            'model_version': f"v2.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        save_metrics(detailed_metrics)

        # Legjobb modell mentése
        artifact_path = os.path.join('artifacts', 'spam_model.pkl')
        model_data = {
            'vectorizer': best_vectorizer,
            'model': best_model,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'model_name': best_model_name,
                'best_params': best_model_info['best_params'],
                'accuracy': best_model_info['accuracy'],
                'auc_roc': best_model_info['auc_score'],
                'cv_score': best_model_info['cv_mean'],
                'model_version': detailed_metrics['model_version']
            }
        }

        with open(artifact_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Legjobb modell sikeresen mentve: {artifact_path}")
        logger.info(f"Training befejezve. Összes idő: {(datetime.now() - start_time).total_seconds():.2f} másodperc")

        return detailed_metrics

    except Exception as e:
        logger.error(f"Hiba a training során: {str(e)}")
        raise


if __name__ == '__main__':
    main()