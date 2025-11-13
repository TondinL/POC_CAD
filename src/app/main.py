# src/app/main.py
import os
from pathlib import Path
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .runtime import InferenceEngine

# Config tramite env:
#   MODEL_PATH (obbligatorio): path del best.pth
#   MODEL_DEPTH (default 50)
#   DEVICE (default cpu)
#   TEMPORAL_SIZE (default 8)
#   IMAGE_SIZE (default 224)
#   THRESHOLD (default 0.5)
MODEL_PATH = os.getenv("MODEL_PATH", "")
MODEL_DEPTH = int(os.getenv("MODEL_DEPTH", "50"))
DEVICE = os.getenv("DEVICE", "cpu")
TEMPORAL_SIZE = int(os.getenv("TEMPORAL_SIZE", "8"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

if not MODEL_PATH:
    raise RuntimeError("MODEL_PATH non impostato. Passa -e MODEL_PATH=/models/best.pth al docker run.")

app = FastAPI(title="POC_CAD Inference API")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

engine = InferenceEngine(
    ckpt_path=MODEL_PATH,
    depth=MODEL_DEPTH,
    device=DEVICE,
    temporal_size=TEMPORAL_SIZE,
    image_size=IMAGE_SIZE,
    threshold=THRESHOLD
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_depth": MODEL_DEPTH}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Form(None)):
    # override soglia facoltativo
    thr = THRESHOLD if threshold is None else float(threshold)

    # salva temporaneamente il video
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # esegui inferenza
    try:
        # hack veloce: aggiorna soglia runtime (non thread-safe in parallelo, ok per POC)
        engine.threshold = thr
        out = engine.predict_video(tmp_path)
        return JSONResponse(out)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
