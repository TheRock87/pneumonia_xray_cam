import io
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

# Assuming main.py is inside an 'app' directory, alongside model.py and utils.py
# This is a standard and clean project structure.
from .model import load_model
from .utils import (
    inference_transform, 
    generate_grad_cam, 
    overlay_cam_on_image, 
    CLASS_NAMES
)


app = FastAPI(
    title="Pneumonia Detection API",
    description="An API to classify chest X-ray images as Normal or Pneumonia.",
    version="0.1.0"
)

# Initialize a global variable `model` to None. This will hold our loaded,
# trained model, making it accessible to our prediction endpoint.
model = None

@app.on_event("startup")
def startup_event():

    global model

    model = load_model(
        model_path= "model/pneumonia_classifier_weights.pth",
        num_classes=2
        )
    
    print("--- Model loaded successfully and is ready to make predictions. ---")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Pneumonia Detection API!"
    }

@app.get("/health", status_code=200)
def health_check():
    """A simple health check endpoint that responds immediately."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded. Please try again in a few moments."}

    content = await file.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")

    # Transform the image to the same format as the training images.
    image_tensor = inference_transform(pil_image)
    if not isinstance(image_tensor, torch.Tensor):  # Ensure it's a tensor
        image_tensor = torch.tensor(image_tensor)
    input_batch = image_tensor.unsqueeze(0)

    # Make a prediction.
    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities,1)
        predicted_class = CLASS_NAMES[int(predicted_idx.item())]
        confidence_score = confidence.item()


    # 4. Generate Grad-CAM
    # Identify the target layer in your DenseNet model
    target_layer = model.densenet.features.denseblock4
    cam_heatmap = generate_grad_cam(model, input_batch, target_layer)

    # 5. Overlay CAM on the original image
    # Convert PIL image to numpy array for OpenCV
    original_image_np = np.array(pil_image)
    superimposed_img = overlay_cam_on_image(original_image_np, cam_heatmap)

    # 6. Encode the result image to Base64
    # Convert the numpy array (RGB) to BGR for OpenCV encoding
    superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode(".jpg", superimposed_img_bgr)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
    
    # Prepend the data URI scheme for easy use in HTML/JS
    cam_image_uri = "data:image/jpeg;base64," + img_base64

    # 7. Return the full result
    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": float(f"{confidence_score*100:.4f}"),
        "cam_image": cam_image_uri
    }

