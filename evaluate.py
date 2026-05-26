import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ===================================
# Config
# ===================================
CLASS_NAMES  = ['early_blight', 'healthy', 'late_blight']
WEIGHTS_PATH = 'potato_model_best.pth'
DATA_VAL_DIR = 'data/val'
BATCH_SIZE   = 32


# ===================================
# Build model — must match train.py exactly
# ===================================
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ===================================
# Run inference on full val set
# ===================================
def get_predictions(model, loader, device):
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ===================================
# Plot single confusion matrix (normalised %)
# Normalised is better for imbalanced classes —
# 31 healthy vs 200 others makes raw counts misleading
# ===================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Annotation: show both % and raw count in each cell
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i,j]:.1f}%\n({cm[i,j]})"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm,
        annot=annot, fmt='', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
        vmin=0, vmax=100
    )
    ax.set_title('Confusion Matrix — ResNet18 Potato Disease Classifier',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return cm


# ===================================
# Print metrics report
# ===================================
def print_metrics(y_true, y_pred, class_names):
    print("\n" + "="*65)
    print("  Task 6 — Evaluation Report")
    print("="*65)

    accuracy  = (y_true == y_pred).mean() * 100
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, average=None, zero_division=0)
    support   = np.bincount(y_true, minlength=len(class_names))

    print(f"\n  Overall Accuracy : {accuracy:.2f}%")
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*55}")

    for i, cls in enumerate(class_names):
        print(f"  {cls:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")

    print(f"  {'-'*55}")

    macro_p  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_r  = recall_score(   y_true, y_pred, average='macro',    zero_division=0)
    macro_f1 = f1_score(       y_true, y_pred, average='macro',    zero_division=0)
    weight_p  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weight_r  = recall_score(   y_true, y_pred, average='weighted', zero_division=0)
    weight_f1 = f1_score(       y_true, y_pred, average='weighted', zero_division=0)

    print(f"  {'Macro avg':<15} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {sum(support):>10}")
    print(f"  {'Weighted avg':<15} {weight_p:>10.4f} {weight_r:>10.4f} {weight_f1:>10.4f} {sum(support):>10}")

    print(f"\n  {'='*65}")
    print("  Per-class verdict:")
    for i, cls in enumerate(class_names):
        if f1[i] < 0.7:
            print(f"  ⚠  {cls:<15} F1={f1[i]:.2f}  — needs improvement")
        else:
            print(f"  ✔  {cls:<15} F1={f1[i]:.2f}  — good")
    print("="*65 + "\n")


# ===================================
# Plot metrics bar chart
# ===================================
def plot_metrics_bar(y_true, y_pred, class_names, save_path='metrics_bar.png'):
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, average=None, zero_division=0)

    x     = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue',  alpha=0.85)
    ax.bar(x,         recall,    width, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + width, f1,        width, label='F1-Score',  color='seagreen',   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Precision / Recall / F1-Score per Class\nResNet18 — Potato Disease Classifier',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)

    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Metrics bar chart saved to {save_path}")


# ===================================
# Main
# ===================================
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    print(f"Loading model: {WEIGHTS_PATH}")

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(root=DATA_VAL_DIR, transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Val images   : {len(val_dataset)}")
    print(f"Classes      : {val_dataset.classes}")

    model = build_model(num_classes=len(CLASS_NAMES), device=device)
    print("Model loaded.\n")

    print("Running inference on validation set...")
    y_true, y_pred = get_predictions(model, val_loader, device)

    print_metrics(y_true, y_pred, CLASS_NAMES)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    plot_metrics_bar(y_true, y_pred, CLASS_NAMES)

    print("Task 6 complete. Files saved:")
    print("  confusion_matrix.png")
    print("  metrics_bar.png")