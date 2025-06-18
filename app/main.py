# app/main.py

import io
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image

from .model import load_model
from .utils import (
    inference_transform, 
    generate_grad_cam, 
    overlay_cam_on_image, 
    CLASS_NAMES
)

# --- App Configuration ---
app = FastAPI(
    title="Pneumonia Detection API",
    description="An API to classify chest X-ray images and provide visual explanations (Grad-CAM).",
    version="1.1.0" # Version updated for new feature
)

# --- Model Loading (Eager Loading) ---
model = None

@app.on_event("startup")
def startup_event():
    """Load the model when the server starts."""
    global model
    try:
        model_path = "model/pneumonia_classifier_weights.pth"
        print(f"--- Loading model from {model_path}... ---")
        model = load_model(model_path, num_classes=2)
        print("--- Model loaded successfully and is ready for predictions. ---")
    except Exception as e:
        print(f"!!! CRITICAL: Model loading failed: {e} !!!")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome! Navigate to /docs to use the API."}

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    generate_cam: bool = Query(False, description="Set to true to generate and include the Grad-CAM image.")
):
    """
    Receives an image, predicts its class (Normal/Pneumonia), and optionally
    returns a Grad-CAM visualization.
    """
    if model is None:
        print("Error: /predict called but model is not loaded.")
        return JSONResponse(status_code=503, content={"error": "Model is not available. Please check server logs."})

    # 1. Read and process image
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = inference_transform(pil_image).unsqueeze(0)

    # 2. Get model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

    # Prepare the base response
    response_data = {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": float(f"{confidence_score * 100:.4f}")
    }


    # Only perform the expensive CAM generation if the user requests it.
    # This makes the default endpoint fast and avoids timeouts.
    if generate_cam:
        print("--- CAM generation requested. This may be slow. ---")
        try:
            target_layer = model.densenet.features.denseblock4
            cam_heatmap = generate_grad_cam(model, input_tensor, target_layer)
            
            original_image_np = np.array(pil_image)
            superimposed_img = overlay_cam_on_image(original_image_np, cam_heatmap)
            
            superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
            _, img_encoded = cv2.imencode(".jpg", superimposed_img_bgr)
            img_base64 = base64.b64encode(img_encoded).decode("utf-8")
            
            response_data["cam_image"] = "data:image/jpeg;base64," + img_base64
        except Exception as e:
            print(f"!!! CAM generation failed: {e} !!!")
            response_data["cam_error"] = "Failed to generate CAM visualization."

    return response_data

