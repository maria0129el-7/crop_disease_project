import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ===================================
# Custom CNN Architecture
# ===================================
class PotatoCNN(nn.Module):
    """
    Custom CNN for 3-class potato leaf disease classification.
    Input: (B, 3, 128, 128)

    Architecture:
      Block 1: Conv(3->32)   -> BN -> ReLU -> Conv(32->32)  -> BN -> ReLU -> MaxPool -> Dropout
      Block 2: Conv(32->64)  -> BN -> ReLU -> Conv(64->64)  -> BN -> ReLU -> MaxPool -> Dropout
      Block 3: Conv(64->128) -> BN -> ReLU -> Conv(128->128)-> BN -> ReLU -> MaxPool -> Dropout
      Block 4: Conv(128->256)-> BN -> ReLU -> Conv(256->256)-> BN -> ReLU -> AdaptiveAvgPool
      Classifier: FC(256->128) -> ReLU -> Dropout -> FC(128->num_classes)
    """

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
                layers.append(nn.MaxPool2d(2, 2))   # halves spatial dims
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.block1 = conv_block(3,   32,  pool=True, dropout=0.25)  # -> (32, 64, 64)
        self.block2 = conv_block(32,  64,  pool=True, dropout=0.25)  # -> (64, 32, 32)
        self.block3 = conv_block(64,  128, pool=True, dropout=0.25)  # -> (128, 16, 16)
        self.block4 = conv_block(128, 256, pool=False, dropout=0.0)  # -> (256, 16, 16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)                   # -> (256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ===================================
# Main Training Script
# ===================================
if __name__ == '__main__':

    # -----------------------------------
    # Device
    # -----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------
    # Transforms
    # -----------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------------
    # Datasets & DataLoaders
    # -----------------------------------
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    val_dataset   = datasets.ImageFolder(root='data/val',   transform=val_transform)

    NUM_CLASSES = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)
    print("Class-to-index:", train_dataset.class_to_idx)
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

    # -----------------------------------
    # Model
    # -----------------------------------
    model = PotatoCNN(num_classes=NUM_CLASSES).to(device)

    # ── Pretty-print the full architecture ──────────────────────────────────
    print("\n" + "="*60)
    print("         PotatoCNN — Model Architecture")
    print("="*60)
    print(model)
    print("="*60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:>10,}")
    print(f"  Trainable parameters : {trainable:>10,}")
    print(f"  Output classes       : {NUM_CLASSES}  →  {train_dataset.classes}")
    print("="*60 + "\n")

    # -----------------------------------
    # Loss & Optimizer
    # -----------------------------------
    # Equal weights; adjust e.g. [2.0, 1.0, 2.0] if classes are imbalanced
    weights   = torch.ones(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # -----------------------------------
    # Tracking
    # -----------------------------------
    train_losses   = []
    val_losses     = []
    val_accuracies = []
    best_val_loss  = float('inf')

    # -----------------------------------
    # Helper: one epoch
    # -----------------------------------
    def run_epoch(loader, optimizer=None):
        training = optimizer is not None
        model.train() if training else model.eval()
        total_loss = 0.0
        correct = 0
        total   = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(loader), 100.0 * correct / total

    # ===================================
    # Phase 1: Warm-up — classifier head only (5 epochs)
    # Trains only the FC layers; conv blocks stay frozen.
    # ===================================
    print("\n--- Phase 1: Warm-up classifier head (5 epochs) ---")
    PHASE1_EPOCHS = 5

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', patience=2, factor=0.5, verbose=True
    )

    for epoch in range(PHASE1_EPOCHS):
        tr_loss, _       = run_epoch(train_loader, optimizer1)
        vl_loss, vl_acc  = run_epoch(val_loader)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_accuracies.append(vl_acc)
        scheduler1.step(vl_loss)

        print(f"Epoch [{epoch+1:02d}/{PHASE1_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  |  "
              f"Val Acc: {vl_acc:.2f}%")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "potato_model_best.pth")
            print("  -> Best model saved!")

    # ===================================
    # Phase 2: Full fine-tune — all layers (15 epochs)
    # Lower LR to avoid destroying early features.
    # ===================================
    print("\n--- Phase 2: Full fine-tune all layers (15 epochs) ---")
    PHASE2_EPOCHS = 15

    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=PHASE2_EPOCHS, eta_min=1e-6
    )

    for epoch in range(PHASE2_EPOCHS):
        tr_loss, _       = run_epoch(train_loader, optimizer2)
        vl_loss, vl_acc  = run_epoch(val_loader)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_accuracies.append(vl_acc)
        scheduler2.step()

        print(f"Epoch [{epoch+1:02d}/{PHASE2_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  |  "
              f"Val Acc: {vl_acc:.2f}%  |  "
              f"LR: {scheduler2.get_last_lr()[0]:.2e}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "potato_model_best.pth")
            print("  -> Best model saved!")

    # -----------------------------------
    # Final Metrics
    # -----------------------------------
    print(f"\nFinal Validation Accuracy : {val_accuracies[-1]:.2f}%")
    print(f"Best Validation Loss      : {best_val_loss:.4f}")

    # -----------------------------------
    # Plot
    # -----------------------------------
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
    x = list(range(1, TOTAL_EPOCHS + 1))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses,   label='Val Loss')
    plt.axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Full fine-tune start')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, val_accuracies, color='green', label='Val Accuracy')
    plt.axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Full fine-tune start')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Training curves saved to training_curves.png")

    torch.save(model.state_dict(), "potato_model_final.pth")
    print("Final model saved to potato_model_final.pth")