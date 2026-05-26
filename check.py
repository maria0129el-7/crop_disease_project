import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Dataset sizes
for split in ['train', 'val']:
    print(f'\n{split}:')
    for cls in sorted(os.listdir(f'data/{split}')):
        n = len(os.listdir(f'data/{split}/{cls}'))
        print(f'  {cls}: {n} images')

# Per-class accuracy
class PotatoCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        def cb(i, o, pool=True, drop=0.25):
            l = [nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                 nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
            if pool: l.append(nn.MaxPool2d(2,2))
            if drop > 0: l.append(nn.Dropout2d(drop))
            return nn.Sequential(*l)
        self.block1 = cb(3,32);   self.block2 = cb(32,64)
        self.block3 = cb(64,128); self.block4 = cb(128,256,pool=False,drop=0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Flatten(), nn.Linear(256,128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(128,num_classes))
    def forward(self, x):
        return self.classifier(self.global_pool(
            self.block4(self.block3(self.block2(self.block1(x))))))

val_tf = transforms.Compose([
    transforms.Resize((128,128)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_ds = datasets.ImageFolder('data/val', transform=val_tf)
val_dl = DataLoader(val_ds, batch_size=32)
classes = ['early_blight', 'healthy', 'late_blight']

print()
for ckpt in ['potato_model_best.pth', 'potato_model_final.pth']:
    m = PotatoCNN()
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    correct = total = 0
    per_class = {i: {'c':0,'t':0} for i in range(3)}
    with torch.no_grad():
        for imgs, labels in val_dl:
            out = m(imgs)
            _, pred = torch.max(out, 1)
            for p, l in zip(pred, labels):
                per_class[l.item()]['t'] += 1
                if p == l: per_class[l.item()]['c'] += 1
            correct += (pred==labels).sum().item()
            total   += labels.size(0)
    print(f'{ckpt}:  Overall = {100*correct/total:.2f}%')
    for i, c in enumerate(classes):
        t  = per_class[i]['t']
        cc = per_class[i]['c']
        print(f'  {c:<15}: {cc}/{t} = {100*cc/t if t>0 else 0:.1f}%')
    print()