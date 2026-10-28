
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Project folder
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "signvision_model.keras"
LABELS_PATH = BASE_DIR / "class_names.json"

# Check required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

# Load trained AI model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Create API
app = FastAPI(
    title="SignVision AI",
    description="ASL Alphabet Hand Gesture Recognition",
    version="1.0"
)

# Allow frontend to connect to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SignVision AI",
        "status": "Backend is running",
        "classes": len(class_names)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((160, 160))

        # Convert image into a NumPy array
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict hand sign
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        best_index = int(np.argmax(predictions))

        return {
            "predicted_sign": class_names[best_index],
            "confidence": round(
                float(predictions[best_index]) * 100,
                2
            )
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )
