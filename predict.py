import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os


# ===================================
# PotatoCNN — must match train.py exactly
# ===================================
class PotatoCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(PotatoCNN, self).__init__()

        def conv_block(in_ch, out_ch, pool=True, dropout=0.25):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.block1 = conv_block(3,   32,  pool=True,  dropout=0.25)
        self.block2 = conv_block(32,  64,  pool=True,  dropout=0.25)
        self.block3 = conv_block(64,  128, pool=True,  dropout=0.25)
        self.block4 = conv_block(128, 256, pool=False, dropout=0.0)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ===================================
# Config
# ===================================
CLASS_NAMES = ['early_blight', 'healthy', 'late_blight']   # alphabetical — must match training

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

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===================================
# Core predict function
# ===================================
def predict(image_path: str, model: nn.Module, device: torch.device) -> dict:
    """
    Run inference on a single image.

    Returns a dict with:
      - predicted_class (str)
      - confidence      (float, 0-100)
      - probabilities   (dict  class → %)
      - advice          (str)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)   # (1, 3, 128, 128)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)                          # (1, 3)
        probs  = F.softmax(logits, dim=1)[0]            # (3,)

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
    print(f"  Image : {os.path.basename(image_path)}")
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
        description="Potato leaf disease predictor using PotatoCNN"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to one or more leaf images (jpg/png)"
    )
    parser.add_argument(
        "--model",
        default="potato_model_best.pth",
        help="Path to saved model weights (default: potato_model_best.pth)"
    )
    parser.add_argument(
        "--use-final",
        action="store_true",
        help="Load potato_model_final.pth instead of the best checkpoint"
    )
    return parser.parse_args()


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Run train.py first to generate the checkpoint."
        )
    model = PotatoCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded weights: {weights_path}")
    return model


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