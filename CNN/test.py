import torch
from torchvision import transforms
from PIL import Image
import os
from model import MaterialClassifier  # assuming your model class is saved in model.py

# === 1. Setup ===
CATEGORIES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
MODEL_PATH = "best_model.pth"  # your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # or whatever was used in training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Load model ===
def load_model():
    model = MaterialClassifier(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === 4. Predict one image + weight ===
def predict_image(model, image_path, weight_grams):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Normalize or scale the weight if needed (ensure training scale is matched!)
    weight_tensor = torch.tensor([[weight_grams]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor, weight_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    return CATEGORIES[pred_idx], confidence

# === 5. Predict multiple with static weight (or modify to use a CSV input) ===
def predict_folder(model, folder_path, weight=100.0):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, file)
            label, conf = predict_image(model, full_path, weight)
            print(f"{file} -> {label} ({conf*100:.2f}%)")

# === 6. Main ===
if __name__ == "__main__":
    model = load_model()

    # Example: Single prediction
    test_img = "data/test/images/glass_478.jpg"
    test_weight = 150.0  # grams, for example
    label, conf = predict_image(model, test_img, test_weight)
    print(f"{test_img} -> {label} ({conf*100:.2f}%)")

    # Example: Predict all images in folder with same weight
    # predict_folder(model, "inference_images/", weight=120.0)
