import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===================================
    # Task 4: Augmentation Pipeline
    # ===================================
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.RandomResizedCrop(size=128, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # ===================================
    # Datasets
    # ===================================
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    val_dataset   = datasets.ImageFolder(root='data/val',   transform=val_transform)
    NUM_CLASSES   = len(train_dataset.classes)

    print("\nClasses     :", train_dataset.classes)
    print("Class→index :", train_dataset.class_to_idx)

    # Class counts
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    print("\nTraining images per class:")
    for cls, cnt in zip(train_dataset.classes, class_counts):
        print(f"  {cls:<15}: {cnt}")

    # ===================================
    # WeightedRandomSampler (fixes class imbalance)
    # ===================================
    class_weights  = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,   num_workers=0)

    # ===================================
    # Weighted Loss (extra penalty for minority class)
    # ===================================
    total_samples = sum(class_counts)
    loss_weights  = torch.tensor(
        [total_samples / (NUM_CLASSES * c) for c in class_counts],
        dtype=torch.float
    ).to(device)

    print("\nLoss weights:")
    for cls, w in zip(train_dataset.classes, loss_weights.tolist()):
        print(f"  {cls:<15}: {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # ===================================
    # Task 5: Transfer Learning — ResNet18
    # ===================================
    print("\n" + "="*60)
    print("  Task 5 — Transfer Learning: ResNet18")
    print("="*60)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze ALL base layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final FC layer with our 3-class head
    in_features  = model.fc.in_features          # 512 for ResNet18
    model.fc     = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_now   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Pretrained backbone : ResNet18 (ImageNet weights)")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable (Phase 1) : {trainable_now:,}  ← classifier head only")
    print(f"  Frozen (Phase 1)    : {total_params - trainable_now:,}  ← ResNet18 backbone")
    print(f"  Output classes      : {NUM_CLASSES} → {train_dataset.classes}")
    print("="*60)

    # ===================================
    # Tracking
    # ===================================
    train_losses   = []
    val_losses     = []
    val_accuracies = []
    best_val_loss  = float('inf')

    def run_epoch(loader, optimizer=None):
        training = optimizer is not None
        model.train() if training else model.eval()
        total_loss = correct = total = 0
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
    # Phase 1: Train classifier head only (7 epochs)
    # Backbone frozen — ResNet18 features used as-is
    # ===================================
    print("\n--- Phase 1: Classifier head only — backbone FROZEN (7 epochs) ---")
    PHASE1_EPOCHS = 7

    optimizer1 = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', patience=2, factor=0.5, verbose=True)

    for epoch in range(PHASE1_EPOCHS):
        tr_loss, _      = run_epoch(train_loader, optimizer1)
        vl_loss, vl_acc = run_epoch(val_loader)
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
    # Phase 2: Unfreeze last ResNet block + train (10 epochs)
    # Gradually unfreeze — avoids destroying pretrained features
    # ===================================
    print("\n--- Phase 2: Unfreeze layer4 + classifier (10 epochs) ---")
    PHASE2_EPOCHS = 10

    # Only unfreeze the last residual block (layer4) + fc
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_p2:,}  (layer4 + classifier)")

    optimizer2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(PHASE2_EPOCHS):
        tr_loss, _      = run_epoch(train_loader, optimizer2)
        vl_loss, vl_acc = run_epoch(val_loader)
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

    # ===================================
    # Phase 3: Full fine-tune ALL layers (8 epochs)
    # Very low LR to fine-tune without destroying ImageNet features
    # ===================================
    print("\n--- Phase 3: Full fine-tune ALL layers (8 epochs) ---")
    PHASE3_EPOCHS = 8

    for param in model.parameters():
        param.requires_grad = True

    trainable_p3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_p3:,}  (entire network)")

    optimizer3 = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer3, T_max=PHASE3_EPOCHS, eta_min=1e-7)

    for epoch in range(PHASE3_EPOCHS):
        tr_loss, _      = run_epoch(train_loader, optimizer3)
        vl_loss, vl_acc = run_epoch(val_loader)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_accuracies.append(vl_acc)
        scheduler3.step()
        print(f"Epoch [{epoch+1:02d}/{PHASE3_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  |  "
              f"Val Acc: {vl_acc:.2f}%  |  "
              f"LR: {scheduler3.get_last_lr()[0]:.2e}")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "potato_model_best.pth")
            print("  -> Best model saved!")

    # ===================================
    # Final per-class accuracy
    # ===================================
    print(f"\nFinal Validation Accuracy : {val_accuracies[-1]:.2f}%")
    print(f"Best Validation Loss      : {best_val_loss:.4f}")

    model.load_state_dict(torch.load("potato_model_best.pth", map_location=device))
    model.eval()
    per_class = {i: {'c': 0, 't': 0} for i in range(NUM_CLASSES)}
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            for p, l in zip(preds, labels):
                per_class[l.item()]['t'] += 1
                if p == l:
                    per_class[l.item()]['c'] += 1

    print("\nPer-class accuracy (best model):")
    for i, cls in enumerate(train_dataset.classes):
        t  = per_class[i]['t']
        cc = per_class[i]['c']
        bar = '█' * int(30 * cc / t) + '░' * (30 - int(30 * cc / t))
        print(f"  {cls:<15}: {bar}  {cc}/{t} = {100*cc/t if t>0 else 0:.1f}%")

    # ===================================
    # Plot
    # ===================================
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS
    x = list(range(1, TOTAL_EPOCHS + 1))

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses,   label='Val Loss')
    plt.axvline(x=PHASE1_EPOCHS + 0.5,                color='orange', linestyle='--', label='Unfreeze layer4')
    plt.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS + 0.5, color='red',   linestyle='--', label='Full fine-tune')
    plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, val_accuracies, color='green', label='Val Accuracy')
    plt.axvline(x=PHASE1_EPOCHS + 0.5,                color='orange', linestyle='--', label='Unfreeze layer4')
    plt.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS + 0.5, color='red',   linestyle='--', label='Full fine-tune')
    plt.title("Validation Accuracy"); plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)"); plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("\nTraining curves saved to training_curves.png")

    torch.save(model.state_dict(), "potato_model_final.pth")
    print("Final model saved to potato_model_final.pth")