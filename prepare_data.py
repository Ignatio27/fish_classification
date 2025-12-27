import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

NEW_DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FISH_SPECIES = {
    "goldfish": "Goldfish",
    "perch": "Perch",
    "pike": "Pike",
    "smelt": "Smelt",
    "zander": "Zander"
}

def create_dataset_from_folders():
    """Создает CSV датасет из папок с фото"""
    
    images_data = []
    
    for folder_name, label in FISH_SPECIES.items():
        folder_path = NEW_DATA_DIR / folder_name
        
        if not folder_path.exists():
            print(f"⚠ Папка {folder_path} не найдена!")
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.is_file() and f.suffix in image_extensions]
        
        print(f"\n{label}: найдено {len(image_files)} изображений")
        
        valid_count = 0
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                img.verify()
                
                images_data.append({
                    'image_path': str(img_path.absolute()),
                    'label': label
                })
                valid_count += 1
                
            except Exception as e:
                print(f"  ✗ Битое изображение: {img_path.name}")
                continue
        
        print(f"  ✓ Валидных: {valid_count}")
    
    if not images_data:
        print("\n Ошибка: Не найдено изображений!")
        return
    
    df = pd.DataFrame(images_data)
    
    print(f"\n{'='*60}")
    print(f"Всего изображений: {len(df)}")
    print(f"\nБаланс классов:")
    print(df['label'].value_counts())
    print(f"{'='*60}")
    
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print(f"\n✓ Подготовка данных завершена:")
    print(f"  • Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  • Val: {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"  • Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"\n✓ Сохранено в: {PROCESSED_DIR}")

if __name__ == "__main__":
    create_dataset_from_folders()
