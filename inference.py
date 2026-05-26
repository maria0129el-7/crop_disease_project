

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# ===================================
# Config
# ===================================
WEIGHTS_PATH = "potato_model_best.pth"
CLASS_NAMES  = ["early_blight", "healthy", "late_blight"]

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===================================
# Build model — must match train.py
# ===================================
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


# ===================================
# Inference
# ===================================
def predict(image_path):
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)        # (1, 3, 128, 128)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx].item() * 100

    return pred_class, confidence


# ===================================
# Main
# ===================================
if len(sys.argv) != 2:
    print("Usage: python inference.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
model      = load_model()

predicted_class, confidence = predict(image_path)

print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")