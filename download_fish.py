import requests
import csv
import os
import time
from pathlib import Path
from urllib.parse import urlparse

# 5 видов рыб
FISH_SPECIES = [
    {"name": "Carassius_auratus", "russian": "Золотая рыбка", "taxon_id": 58612},
    {"name": "Esox_lucius", "russian": "Щука", "taxon_id": 55387},
    {"name": "Perca_fluviatilis", "russian": "Окунь", "taxon_id": 109062},
    {"name": "Sander_lucioperca", "russian": "Судак", "taxon_id": 48156},
    {"name": "Osmerus_mordax", "russian": "Корюшка", "taxon_id": 179504},
]

DATASET_DIR = Path("fish_dataset")
MAX_IMAGES = 350

def setup():
    DATASET_DIR.mkdir(exist_ok=True)
    for sp in FISH_SPECIES:
        (DATASET_DIR / sp["name"]).mkdir(exist_ok=True)
    print(f"Директории созданы в {DATASET_DIR}\n")

def download_images(taxon_id, species_name, max_images=MAX_IMAGES):
    """Скачивает изображения рыб"""
    species_dir = DATASET_DIR / species_name
    count = 0
    page = 1
    
    print(f"Скачиваю {species_name}...")
    
    while count < max_images:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "has_photos": True,
            "per_page": 200,
            "page": page,
            "order_by": "created_at",
            "order": "desc"
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("results"):
                break
            
            for obs in data["results"]:
                if count >= max_images:
                    break
                if not obs.get("photos"):
                    continue
                
                photo = obs["photos"][0]
                photo_url = photo.get("url")
                if not photo_url:
                    continue
                
                # Берём medium размер
                photo_url = photo_url.replace("thumb", "medium")
                
                try:
                    img_resp = requests.get(photo_url, timeout=10)
                    img_resp.raise_for_status()
                    
                    ext = Path(urlparse(photo_url).path).suffix or ".jpg"
                    filename = f"{species_name}_{count:04d}{ext}"
                    filepath = species_dir / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(img_resp.content)
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"   {count}/{max_images}")
                    
                    time.sleep(0.1)
                except Exception as e:
                    pass
            
            page += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Ошибка: {e}")
            break
    
    print(f"  Готово: {count} изображений\n")
    return count

def create_csv():
    """Создаёт CSV манифест"""
    print("Создаю CSV манифест...")
    
    csv_path = DATASET_DIR / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "species_latin", "taxon_id"])
        
        for sp in FISH_SPECIES:
            sp_dir = DATASET_DIR / sp["name"]
            for img in sorted(sp_dir.glob("*")):
                if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    writer.writerow([
                        str(img),
                        sp["russian"],
                        sp["name"],
                        sp["taxon_id"]
                    ])
    
    print(f" {csv_path}\n")

def main():
    print("="*60)
    print(" СКАЧИВАНИЕ ДАТАСЕТА РЫБ")
    print("="*60 + "\n")
    
    setup()
    
    total = 0
    for sp in FISH_SPECIES:
        count = download_images(sp["taxon_id"], sp["name"])
        total += count
    
    create_csv()
    
    print("="*60)
    print(f" ГОТОВО!")
    print("="*60)
    print(f"\n Датасет: {DATASET_DIR}/")
    print(f"  Всего изображений: {total}")
    
    for sp in FISH_SPECIES:
        imgs = len(list((DATASET_DIR / sp["name"]).glob("*")))
        print(f"  • {sp['russian']}: {imgs}")
    
    print(f"\n CSV: {DATASET_DIR}/dataset.csv")

if __name__ == "__main__":
    main()
