# Machine Learning modellek fejlesztése tanfolyam 
## MLOps vizsga – Fehér Péter

A projekt egy egyszerű spam-detekciós rendszer, angol nyelvű szövegekre, magyar nyelvű, nyílt, szabadon felhasználható datasetek hiányában , amely Python/Flask alapú webes felületet és egy Naive Bayes modellre épülő tanító scriptet tartalmaz. 

Projekthez használt dataset : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

A cél, hogy bemutassam a gépi tanulási modell fejlesztésének, kiértékelésének és élesítésének teljes folyamatait MLOps-szemléletben. A rendszer tartalmazza a következő fő részeket:


1. **`train.py`**  
   - Betölti a `data/spam.csv` fájlt, előfeldolgozza a szövegeket,  
   - Vektorizálja őket `CountVectorizer`-rel,  
   - Naive Bayes modellel megtanítja a spam/ham osztályozót,  
   - Kiértékeli a modellt (acc, ROC-AUC, konfúziós mátrix, classification report),  
   - Mentésre kerül a tanított modell (`artifacts/spam_model.pkl`) és a metrikák (`artifacts/training_metrics.json`).  
   - A futás közben naplózza a lépéseket a `logs/training.log` fájlba.

2. **`app/app.py`**  
   - Flask-alapú API és egyszerű HTML front-end:  
     - `/` – főoldal, ahol beírható egy üzenet, és a rendszer kiszámolja, spam-e vagy sem;  
     - `/predict` – JSON alapú API-végpont, amely POST-kérésben vár egy `{ "message": "<szöveg>" }` formátumú kérést, és visszaadja a predikciót;  
     - `/stats` – lekérhető a szolgáltatásra vonatkozó statisztika (kérésszám, spam/ham arány, átlagos válaszidő, hibaarány, stb.);  
     - `/reset-stats` – statisztikák nullázása.  
   - A Flask-alkalmazás betölti a korábban mentett modellt (`artifacts/spam_model.pkl`) a konténer indításakor, és minden kérést naplóz (`logs/app.log`).

3. **Docker & Docker Compose**  
   - A projekt tartalmaz egy `Dockerfile`-t, amely a Python 3.10-es slim-alapú image-en építi fel a környezetet, telepíti a függőségeket (a `app/requirements.txt` alapján), bemásolja a kódot, és elindítja a Flask szervert.  
   - A `docker-compose.yml` két szolgáltatást definiál:
     - **`web`** – a Flask-alapú API/GUI, amely a 5000-es porton fut.  
     - **`trainer`** – a `train.py` tanító script futtatása külön konténerben, csak akkor indul el, ha a `training` profillal hívjuk (például `docker-compose --profile training up trainer`).  
   - A `volumes` részben hivatkozunk a host-oldali mappákra (`./logs`, `./data`, `./artifacts`), hogy a konténerben keletkezett mentések és naplók visszakerüljenek a fejlesztői környezetbe.

---

## Projekt áttekintése

Ez a repository egy példaprojekt, amely demonstrálja:

1. **Adatbetöltés és előfeldolgozás**  
   - A CSV-adat (`data/spam.csv`) betöltése, oszlopok átnevezése, hiányzó értékek ellenőrzése, label encode (ham=0, spam=1), stb.

2. **Gépi tanulási pipeline**  
   - Szövegből feature-mátrix előállítása `CountVectorizer` segítségével,  
   - Modell tanítása `MultinomialNB` osztályozóval,  
   - Teljesítménymutatók kiszámítása (accuracy, precision, recall, f1, ROC-AUC, konfúziós mátrix),  
   - Metrikák JSON-formátumban kiírása (`artifacts/training_metrics.json`), modellek picklezése (`artifacts/spam_model.pkl`).

3. **API és webes felület**  
   - Felhasználóbarát HTML/Flask form, ahol a user beírhat egy üzenetet, és azonnal látja a spam/ham eredményt,  
   - REST-ful végpont (`/predict`) kliensprogramok, script-ek vagy más mikroszolgáltatások számára,  
   - Egyszerű health check (`/health`) és statisztika-visszaadó endpointok (`/stats`, `/reset-stats`).

4. **Konténerizálás Dockerrel és Docker Compose-al**  
   - A teljes környezet egyszerűen építhető Docker image-ként,  
   - Több konténer egymásra épülve: web-alkalmazás és opcionális tréner,  
   - Könnyen skálázható, hordozható – bármely Docker-kompatibilis környezetben futtatható.

---

## Követelmények

- **Docker** (20.x vagy újabb)  
- **Docker Compose** (v2.x vagy újabb)  
- (Lokális fejlesztéshez) Python 3.10, pip, virtualenv (ha nem konténerben szeretnéd futtatni)

---

## Függőségek telepítése

A kódot konténer nélkül is kipróbálható, Python 3.10 szükséges hozzá:

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Parancsok
```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tesztek futtatása
```bash
pytest tests/
```
## Docker parancsok
```bash
docker-compose build
docker-compose up -d web #Web-szerver indítása 
docker-compose --profile training up trainer #Model újratanítása

docker-compose down
```
http://localhost:5000

---
# Api dokumentáció
### 1. `/health`  
**Módszer:** `GET`  
**Leírás:** Ellenőrzi, hogy a modell betöltődött-e, és válaszol egy egyszerű JSON állapottal. Gyors health-check, amely futtat egy kis próbapredikciót.  

- **Kérés:**  
    ```http
  GET /health HTTP/1.1
  Host: localhost:5000
  
### 2. `/predict`  
**Módszer:** `POST`  
**Leírás:**  A spam/ham predikciós endpoint. Egy JSON-t vár a kérés törzsében, amely tartalmaz egy message mezőt.  
**Content-Type:** application/json

- **Kérés:**  
    ```http
  {
  "message": "Ez itt egy próbalevél, spam?"
  }

### 3. `/stats`  
**Módszer:** `GET`  
**Leírás:** Visszaadja az eddigi kérésekre és predikciókra vonatkozó statisztikákat.

- **Kérés:**  
    ```http
    GET /stats HTTP/1.1
    Host: localhost:5000

### 4. `/reset-stats`  
**Módszer:** `POST`  
**Leírás:** Nullázza a belső statisztikákat `(request_stats)`, mintha újraindult volna az alkalmazás.  
**Content-Type:** application/json

Nem kötelező mező. Üres POST is elegendő

### 5. `/` (HTML)
**Módszer:** `GET,POST`  
**Leírás:**
`GET` kérésre visszaadja a `templates/index.html`-t, 
és az űrlapon keresztül lekérdezhető az üzenetet, és „Predikció” gombra kattintva kiadja az eredményt.  
`POST` kérés esetén a form mező `(message)` ellenőrzése után lefuttatja a modell-predikciót,
és ugyanarra az oldalra rendereli az eredményt (`spam/ham` vagy hibaüzenet).
