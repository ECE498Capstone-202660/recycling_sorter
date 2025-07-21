import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import io
from .material_classifier import MaterialClassifier

CATEGORIES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

def load_model():
    device = torch.device("mps")
    model_path = os.path.join(os.path.dirname(__file__), "best_cm_model.pth")
    model = MaterialClassifier(num_classes=len(CATEGORIES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, device

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def run_inference(input_tensor, weight_grams):
    model, device = load_model()
    input_tensor = input_tensor.to(device)
    weight_tensor = torch.tensor([[weight_grams]], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(input_tensor, weight_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        predicted_class = CATEGORIES[pred_idx.item()]
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence.item()),
            "raw_output": outputs.cpu().numpy().tolist()
        } 