# Dockerfile

# 1) Kiválasztjuk az alap Python image-et
FROM python:3.10-slim

# 2) Nem szeretnénk, hogy a Python buffering miatt késve írja ki a logokat
ENV PYTHONUNBUFFERED=1

# 3) Munkakönyvtár a konténeren belül
WORKDIR /usr/src/app

# 4) Először csak a requirements fájlt másoljuk, és telepítjük a függőségeket
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5) Majd a teljes forráskódot és a szükséges mappákat másoljuk be
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY train.py ./train.py

# 6) Készítsünk log könyvtárat (ahová a Flask app írni fogja a logokat)
RUN mkdir -p /usr/src/app/logs

# 7) Megnyitjuk a 5000-es portot, mert itt fut majd a Flask
EXPOSE 5000

# 8) Alapértelmezett parancs a Flask app indításához
CMD ["python", "app/app.py"]
