import torch
from torchvision import transforms, models
from PIL import Image
import os
import time
import torch.nn as nn

# === 1. Setup ===
CATEGORIES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
MODEL_PATH = "best_mobile_model.pth"  # Your trained MobileNetV2 model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 default input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Load model ===
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(CATEGORIES))  # 6 output classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === 4. Predict one image (inference time included) ===
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()
    end_time = time.time()

    inference_time = end_time - start_time
    return CATEGORIES[pred_idx], confidence, inference_time

# === 5. Predict multiple images in folder ===
def predict_folder(model, folder_path):
    times = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, file)
            label, conf, infer_time = predict_image(model, full_path)
            times.append(infer_time)
            print(f"{file} -> {label} ({conf*100:.2f}%), Time: {infer_time*1000:.2f} ms")
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage inference time: {avg_time*1000:.2f} ms")

# === 6. Main ===
if __name__ == "__main__":
    model = load_model()

    # --- Single image test ---
    test_img = "data/test/images/glass_478.jpg"
    label, conf, infer_time = predict_image(model, test_img)
    print(f"{test_img} -> {label} ({conf*100:.2f}%), Time: {infer_time*1000:.2f} ms")

    # --- Folder test (optional) ---
    # predict_folder(model, "inference_images/")
