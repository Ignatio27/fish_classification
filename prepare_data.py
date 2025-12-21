import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

DATA_DIR = Path("data/fish_dataset")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# Чтение CSV
df = pd.read_csv(DATA_DIR / "dataset.csv")

# Проверка баланса классов
print("\n📊 Баланс классов в датасете:")
print(df['label'].value_counts())

# Разделение на train/val/test (70/15/15)
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

# Сохранение
train.to_csv(PROCESSED_DIR / "train.csv", index=False)
val.to_csv(PROCESSED_DIR / "val.csv", index=False)
test.to_csv(PROCESSED_DIR / "test.csv", index=False)

print(f"\n✅ Подготовка данных завершена:")
print(f"  • Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"  • Val: {len(val)} ({len(val)/len(df)*100:.1f}%)")
print(f"  • Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")

