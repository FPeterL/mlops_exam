import os
import train
import pytest
import importlib
import app.app as flask_app_module

@pytest.mark.order(10)
def test_train_and_reload_model(tmp_path, monkeypatch):

    # Projekt gyökérkönyvtár választás
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    monkeypatch.chdir(project_root)

    # Training futtatása
    metrics = train.main()

    # Metrics ellenőrizése
    assert 'best_model' in metrics, "A train.main() nem adott vissza 'best_model' kulcsot"
    best_model_info = metrics['best_model']
    assert 'accuracy' in best_model_info, "A 'best_model' nem tartalmazza az 'accuracy' kulcsot"
    assert best_model_info['accuracy'] > 0.8, "A legjobb modell pontossága nem éri el a 0.8-at"

    #  artifacts/spam_model.pkl fájl létrejöttének ellenőrzése
    model_path = os.path.join(project_root, "artifacts", "spam_model.pkl")
    assert os.path.exists(model_path), "A tréning nem hozta létre az artifacts/spam_model.pkl fájlt"

    # Friss modell betöltése a Flask alkalmazásból
    importlib.reload(flask_app_module)
    app = flask_app_module.app
    app.config["TESTING"] = True
    client = app.test_client()

    # „ham” üzenet teszt
    ham_message = "Hello, how are you doing today?"
    resp_ham = client.post("/predict", json={"message": ham_message})
    assert resp_ham.status_code == 200, "A /predict nem 200-as kódot adott ham üzenetre"
    assert "prediction" in resp_ham.json, "Nincs 'prediction' mező a ham válaszban"
    assert resp_ham.json["prediction"] == "ham", f"Várt 'ham'-et, de '{resp_ham.json['prediction']}' jött"

    # „spam” üzenet teszt
    spam_message = "Congratulations! You have won $1000. Click here to claim your prize now!"
    resp_spam = client.post("/predict", json={"message": spam_message})
    assert resp_spam.status_code == 200, "A /predict nem 200-as kódot adott spam üzenetre"
    assert "prediction" in resp_spam.json, "Nincs 'prediction' mező a spam válaszban"
    assert resp_spam.json["prediction"] == "spam", f"Várt 'spam'-et, de '{resp_spam.json['prediction']}' jött"
