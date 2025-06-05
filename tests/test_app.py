import os
import json
import pytest
from app.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get('/health')
    assert response.status_code in (200, 503)
    assert 'status' in response.json

def test_predict_valid(client):
    response = client.post('/predict', json={"message": "Congratulations, you've won a prize!"})
    assert response.status_code == 200
    assert 'prediction' in response.json
    assert response.json['prediction'] in ['spam', 'ham']

def test_predict_invalid_no_json(client):
    response = client.post('/predict', data="This is not JSON", content_type='text/plain')
    assert response.status_code == 400
    assert 'error' in response.json

def test_predict_invalid_empty(client):
    response = client.post('/predict', json={"message": ""})
    assert response.status_code == 400
    assert 'error' in response.json

def test_stats(client):
    response = client.get('/stats')
    assert response.status_code == 200
    assert 'total_requests' in response.json

def test_reset_stats(client):
    response = client.post('/reset-stats')
    assert response.status_code == 200
    assert 'message' in response.json
