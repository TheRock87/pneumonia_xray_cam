import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import numpy as np

inference_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def generate_grad_cam(model, input_tensor, target_layer, target_class=None):
    """
    Generates a Grad-CAM heatmap.
    This function captures the gradients and activations from the target layer.
    """
    activations = dict()
    
    # --- START OF FIX ---
    # Store original requires_grad states and temporarily set to True
    # This is necessary to allow gradients to flow back to the target layer,
    # even if the model's layers were frozen during setup.
    original_grad_states = {}
    for name, param in model.named_parameters():
        original_grad_states[name] = param.requires_grad
        param.requires_grad = True
    # --- END OF FIX ---

    def save_activation(module, input, output):
        activations['feature_map'] = output.detach()

    def save_gradient(module, grad_in, grad_out):
        activations['gradient'] = grad_out[0].detach()

    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.eval()
        logits = model(input_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # This key should now exist
        if 'gradient' not in activations:
             print("Warning: Gradient not captured. CAM might be incorrect.")
             return None

        feature_maps = activations['feature_map']
        gradients = activations['gradient']
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(feature_maps.shape[1]):
            feature_maps[0, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

    finally:
        # CRITICAL: Always remove hooks and restore original gradient states
        handle_forward.remove()
        handle_backward.remove()
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_states[name]


def overlay_cam_on_image(img, cam):
    """
    Overlays the CAM heatmap on the original image.
    """
    if cam is None:
        return img # Return original image if CAM generation failed

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

