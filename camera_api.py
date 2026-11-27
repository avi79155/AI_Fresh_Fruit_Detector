from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import io
import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fruit_detector.keras"
CLASS_PATH = "class_names.json"
HISTORY_PATH = "prediction_history.csv"

app = FastAPI(title="FreshFruit AI Camera API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local development only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = None
history_lock = threading.Lock()
last_logged = {"key": None, "time": 0.0}

@app.on_event("startup")
def startup():
    global model, class_names
    if not Path(MODEL_PATH).exists() or not Path(CLASS_PATH).exists():
        raise RuntimeError("Model and class_names.json must be beside camera_api.py")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        class_names = raw
    elif isinstance(raw, dict):
        try:
            class_names = [v for k, v in sorted(raw.items(), key=lambda x: int(x[0]))]
        except (ValueError, TypeError):
            class_names = list(raw.values())
    else:
        raise RuntimeError("Unsupported class_names.json format")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        arr = np.asarray(image, dtype=np.float32)
        arr = preprocess_input(arr)
        pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(pred))
        predicted_class = class_names[idx]
        confidence = float(pred[idx]) * 100
        low = predicted_class.lower()
        if low.startswith("fresh"):
            freshness, fruit = "Fresh", predicted_class[5:]
        elif low.startswith("rotten"):
            freshness, fruit = "Rotten", predicted_class[6:]
        else:
            freshness, fruit = "Unknown", predicted_class
        fruit = fruit.replace("_", " ").title()

        # Save only when the predicted class changes, with a 5-second cooldown.
        now = datetime.now().timestamp()
        key = predicted_class
        with history_lock:
            if key != last_logged["key"] or now - last_logged["time"] >= 5:
                row = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fruit": fruit,
                    "freshness": freshness,
                    "confidence": round(confidence, 2),
                    "source": "Live Webcam"
                }])
                path = Path(HISTORY_PATH)
                if path.exists():
                    try:
                        old = pd.read_csv(path)
                        data = pd.concat([old, row], ignore_index=True)
                    except Exception:
                        data = row
                else:
                    data = row
                data.to_csv(path, index=False)
                last_logged["key"] = key
                last_logged["time"] = now

        return {
            "fruit": fruit,
            "freshness": freshness,
            "confidence": confidence,
            "predicted_class": predicted_class
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
