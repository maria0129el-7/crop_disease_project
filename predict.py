import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import argparse
import os


# ===================================
# Config
# ===================================
CLASS_NAMES = ['early_blight', 'healthy', 'late_blight']  # alphabetical

ADVICE = {
    'early_blight': (
        "Early Blight detected.\n"
        "  Cause   : Alternaria solani fungus\n"
        "  Symptoms: Dark brown spots with concentric rings (target-board pattern)\n"
        "  Action  : Apply copper-based or mancozeb fungicide. Remove infected leaves.\n"
        "            Avoid overhead irrigation. Rotate crops next season."
    ),
    'late_blight': (
        "Late Blight detected.\n"
        "  Cause   : Phytophthora infestans (water mould)\n"
        "  Symptoms: Water-soaked grey-green lesions, white mould on underside\n"
        "  Action  : Apply metalaxyl or chlorothalonil fungicide immediately.\n"
        "            Destroy heavily infected plants. Do NOT compost infected material."
    ),
    'healthy': (
        "Healthy leaf — no disease detected.\n"
        "  Action  : Continue regular monitoring. Maintain good air circulation\n"
        "            and avoid excessive moisture on leaves."
    ),
}

# Must match val_transform in train.py exactly
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===================================
# Task 5: Load ResNet18 — must match train.py architecture
# ===================================
def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Run train.py first to generate the checkpoint."
        )

    # Build the exact same architecture as train.py
    model = models.resnet18(weights=None)           # no pretrained weights here
    in_features = model.fc.in_features              # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, len(CLASS_NAMES))
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded ResNet18 weights: {weights_path}")
    return model


# ===================================
# Core predict function
# ===================================
def predict(image_path: str, model: nn.Module, device: torch.device) -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 128, 128)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)                        # (1, 3)
        probs  = F.softmax(logits, dim=1)[0]          # (3,)

    pred_idx   = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    prob_dict = {cls: round(probs[i].item() * 100, 2)
                 for i, cls in enumerate(CLASS_NAMES)}

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 2),
        "probabilities":   prob_dict,
        "advice":          ADVICE[pred_class],
    }


def print_result(image_path: str, result: dict):
    bar_len = 30
    print("\n" + "="*60)
    print(f"  Image      : {os.path.basename(image_path)}")
    print("="*60)
    print(f"  Prediction : {result['predicted_class'].upper()}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print("\n  Class probabilities:")
    for cls, pct in result['probabilities'].items():
        filled = int(bar_len * pct / 100)
        bar    = "█" * filled + "░" * (bar_len - filled)
        marker = " ◀" if cls == result['predicted_class'] else ""
        print(f"    {cls:<15} {bar}  {pct:6.2f}%{marker}")
    print("\n  Advice:")
    for line in result['advice'].splitlines():
        print(f"    {line}")
    print("="*60 + "\n")


# ===================================
# CLI entry-point
# ===================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Potato leaf disease predictor using ResNet18"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to one or more leaf images (jpg/png)"
    )
    parser.add_argument(
        "--model",
        default="potato_model_best.pth",
        help="Path to model weights (default: potato_model_best.pth)"
    )
    parser.add_argument(
        "--use-final",
        action="store_true",
        help="Use potato_model_final.pth instead of best checkpoint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights_path = "potato_model_final.pth" if args.use_final else args.model
    model        = load_model(weights_path, device)

    print(f"\nRunning predictions on {len(args.images)} image(s)...")

    for img_path in args.images:
        try:
            result = predict(img_path, model, device)
            print_result(img_path, result)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")