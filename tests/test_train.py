import os
import pickle
from train import main

def test_training_output():
    metrics = main()

    assert 'best_model' in metrics

    best_model_info = metrics['best_model']
    assert 'accuracy' in best_model_info
    assert best_model_info['accuracy'] > 0.8

    assert os.path.exists('artifacts/spam_model.pkl')

    with open('artifacts/spam_model.pkl', 'rb') as f:
        data = pickle.load(f)
        assert 'model' in data
        assert 'vectorizer' in data
