import torch
from torchvision import transforms, models
from PIL import Image
import os
import time
import torch.nn as nn
import pandas as pd

# === 1. Setup ===
CATEGORIES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
MODEL_PATH = "best_mobile_model.pth"
DEVICE = torch.device( "cpu")
print("Using device:", DEVICE)

# === 2. Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Load model ===
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(CATEGORIES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# === 4. Predict one image (timed) ===
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

# === 5. Evaluate test set from CSV ===
def evaluate_test_set(model, label_csv_path, image_folder):
    df = pd.read_csv(label_csv_path)
    correct = 0
    total = 0
    total_time = 0.0

    for idx, row in df.iterrows():
        filename, true_label = row["filename"], row["label"]
        image_path = os.path.join(image_folder, filename)

        if not os.path.exists(image_path):
            print(f"Warning: File {image_path} not found, skipping.")
            continue

        pred_label, conf, infer_time = predict_image(model, image_path)

        # Convert int label to class name
        true_label_name = CATEGORIES[int(true_label)]
        is_correct = (pred_label.lower() == true_label_name.lower())

        total += 1
        total_time += infer_time
        correct += int(is_correct)

        print(f"{filename} -> Predicted: {pred_label} ({conf*100:.2f}%), "
              f"True: {true_label_name}, Correct: {is_correct}, Time: {infer_time*1000:.2f} ms")

    accuracy = correct / total * 100 if total > 0 else 0
    avg_time = total_time / total * 1000 if total > 0 else 0

    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
    print(f"⏱️ Average Inference Time: {avg_time:.2f} ms over {total} images")

# === 6. Main ===
if __name__ == "__main__":
    model = load_model()

    # Optional warm-up to avoid CUDA first-run delay
    _ = predict_image(model, "data/test/images/glass_478.jpg")

    # Run full test evaluation
    evaluate_test_set(
        model,
        label_csv_path="data/test/test_labels.csv",
        image_folder="data/test/images"
    )
