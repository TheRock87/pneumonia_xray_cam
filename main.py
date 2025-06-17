import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import torch.nn.functional as F

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and preprocessing setup
MODEL_PATH = "pneumonia_classifier_weights.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.densenet201(weights="DEFAULT")
    model.classifier = nn.Sequential(
        nn.Linear(1920, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layer = model.features.denseblock4

def generate_grad_cam(model, input_tensor, target_layer, target_class=None):
    activations = dict()
    def save_activation(module, input, output):
        activations['feature_map'] = output.detach()
    def save_gradient(module, grad_in, grad_out):
        activations['gradient'] = grad_out[0].detach()
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)
    try:
        logits = model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
    finally:
        handle_forward.remove()
        handle_backward.remove()
    if 'gradient' not in activations or 'feature_map' not in activations:
        return None
    feature_maps = activations['feature_map']
    gradients = activations['gradient']
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    feature_maps = feature_maps.squeeze(0)
    for i in range(feature_maps.shape[0]):
        feature_maps[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(feature_maps, dim=0).squeeze()
    heatmap = F.relu(heatmap)
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    return heatmap.cpu().numpy()

def overlay_cam_on_image(original_img, cam):
    heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

@app.post("https://pneumonia-xray-cam.onrender.com")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        predicted_label = CLASS_NAMES[predicted_class.item()]
    # Grad-CAM
    cam = generate_grad_cam(model, input_tensor, target_layer, target_class=predicted_class.item())
    # Prepare images for response
    original_img = np.array(pil_image)
    if cam is not None:
        cam_overlay = overlay_cam_on_image(original_img, cam)
        _, cam_img_bytes = cv2.imencode('.png', cam_overlay)
        cam_img_b64 = base64.b64encode(cam_img_bytes).decode('utf-8')
    else:
        cam_img_b64 = None
    # Original image as base64
    _, orig_img_bytes = cv2.imencode('.png', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    orig_img_b64 = base64.b64encode(orig_img_bytes).decode('utf-8')
    return {
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "original_image": orig_img_b64,
        "cam_image": cam_img_b64
    }

@app.get("/metrics")
def get_metrics():
    return {
        "accuracy": 85.53,
        "precision": 88.40,
        "recall": 93.02,
        "f1_score": 90.65
    } 