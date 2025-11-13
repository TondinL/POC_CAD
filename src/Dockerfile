FROM python:3.10-slim

# --- Dipendenze di sistema per OpenCV / ffmpeg ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# --- Directory di lavoro ---
WORKDIR /app

# Copia l'intero progetto (inclusi DataHandler, model.py, FineTuned_Models, ecc.)
COPY . /app

# Installa le dipendenze della webapp
COPY src/requirements.txt /app/src/requirements.txt
RUN pip install --no-cache-dir -r /app/src/requirements.txt

# Imposta il path per importare i tuoi moduli
ENV PYTHONPATH=/app

# Porta esposta
EXPOSE 8000

# Comando di avvio: esegue il server FastAPI
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "9000"]
