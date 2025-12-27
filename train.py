import torch
import yaml
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
from PIL import Image
import random

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FishDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            image = Image.new('RGB', (160, 160))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row['label']]
        return image, label

def set_trainable(model: nn.Module, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable

def freeze_for_warmup(model: nn.Module):
    set_trainable(model, False)
    for p in model.fc.parameters():
        p.requires_grad = True

def unfreeze_for_finetune(model: nn.Module):
    set_trainable(model, False)
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    losses = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys_true, ys_pred = [], []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        
        ys_true.extend(labels.cpu().numpy().tolist())
        ys_pred.extend(pred.cpu().numpy().tolist())
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(ys_true, ys_pred)
    f1 = f1_score(ys_true, ys_pred, average="macro")
    return acc, f1

def main():
    import yaml
    
    # Загрузка параметров из params.yaml
    if Path("params.yaml").exists():
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        SEED = params.get("SEED", 42)
        IMAGE_SIZE = params.get("IMAGE_SIZE", 160)
        BATCH_SIZE = params.get("BATCH_SIZE", 8)
        WARMUP_EPOCHS = params.get("WARMUP_EPOCHS", 3)
        FINETUNE_EPOCHS = params.get("FINETUNE_EPOCHS", 10)
        WARMUP_LR = params.get("WARMUP_LR", 1e-3)
        FINETUNE_LR = params.get("FINETUNE_LR", 2e-4)
        WEIGHT_DECAY = params.get("WEIGHT_DECAY", 0.01)
    else:
        SEED = 42
        IMAGE_SIZE = 160
        BATCH_SIZE = 8
        WARMUP_EPOCHS = 3
        FINETUNE_EPOCHS = 10
        WARMUP_LR = 1e-3
        FINETUNE_LR = 2e-4
        WEIGHT_DECAY = 0.01
    
    set_seed(SEED)
    torch.set_num_threads(1)
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРА РЫБ (TWO-STAGE)")
    print("="*60)
    
    tfm_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    tfm_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    print("\nЗагружаю датасеты...")
    train_ds = FishDataset("data/processed/train.csv", transform=tfm_train)
    val_ds = FishDataset("data/processed/val.csv", transform=tfm_val)
    
    classes = train_ds.classes
    num_classes = len(classes)
    
    print(f"Классы ({num_classes}): {classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nИнициализирую ResNet18 (device: {DEVICE})...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    mlflow.set_experiment("Fish_Classification_TwoStage")
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": "ResNet18",
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "warmup_lr": WARMUP_LR,
            "finetune_lr": FINETUNE_LR,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "device": DEVICE,
            "seed": SEED
        })
        
        best_val_f1 = -1.0
        global_step = 0
        
        print("\n" + "="*60)
        print("STAGE A: WARMUP (обучаем только голову)")
        print("="*60 + "\n")
        
        freeze_for_warmup(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=WARMUP_LR,
            weight_decay=WEIGHT_DECAY
        )
        
        for epoch in range(1, WARMUP_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, DEVICE, optimizer, criterion)
            val_acc, val_f1 = evaluate(model, val_loader, DEVICE)
            
            global_step += 1
            mlflow.log_metric("train_loss", train_loss, step=global_step)
            mlflow.log_metric("val_acc", float(val_acc), step=global_step)
            mlflow.log_metric("val_f1", float(val_f1), step=global_step)
            mlflow.log_metric("stage", 0, step=global_step)
            
            print(f"[Warmup] Epoch {epoch}/{WARMUP_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
                print(f"  ✓ Новая лучшая модель! F1={best_val_f1:.4f}")
        
        print("\n" + "="*60)
        print("STAGE B: FINETUNE (обучаем layer4 + голову)")
        print("="*60 + "\n")
        
        unfreeze_for_finetune(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR,
            weight_decay=WEIGHT_DECAY
        )
        
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, DEVICE, optimizer, criterion)
            val_acc, val_f1 = evaluate(model, val_loader, DEVICE)
            
            global_step += 1
            mlflow.log_metric("train_loss", train_loss, step=global_step)
            mlflow.log_metric("val_acc", float(val_acc), step=global_step)
            mlflow.log_metric("val_f1", float(val_f1), step=global_step)
            mlflow.log_metric("stage", 1, step=global_step)
            
            print(f"[Finetune] Epoch {epoch}/{FINETUNE_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
                print(f"  ✓ Новая лучшая модель! F1={best_val_f1:.4f}")
        
        metadata = {
            "classes": classes,
            "num_classes": num_classes,
            "best_val_f1": float(best_val_f1),
            "image_size": IMAGE_SIZE
        }
        
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        metrics = {
            "best_val_f1": float(best_val_f1),
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "total_epochs": WARMUP_EPOCHS + FINETUNE_EPOCHS,
            "final_lr": FINETUNE_LR,
            "num_train": len(train_ds),
            "num_val": len(val_ds)
        }
        
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        mlflow.pytorch.log_model(model, "model")
    
    print("\n" + "="*60)
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший F1-score: {best_val_f1:.4f}")
    print(f"Модель: {MODEL_DIR / 'best_model.pth'}")
    print(f"Метрики: {MODEL_DIR / 'metrics.json'}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
