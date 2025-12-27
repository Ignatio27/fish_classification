import requests
import json
from pathlib import Path
import time
from PIL import Image
from io import BytesIO

NEW_DATA_DIR = Path("new_dataset")
NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)

FISH_SPECIES = {
    "goldfish": 58612,
    "pike": 55387,
    "smelt": 179504,
    "perch": 109062,
    "zander": 48156
}

TARGET_COUNT = 300

def download_inaturalist_hd(species_name, taxon_id, target_count=300):
    """Скачивает полномасштабные фото с iNaturalist в хорошем качестве"""
    
    output_dir = NEW_DATA_DIR / species_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Скачиваю {species_name.upper()}")
    print(f"{'='*60}")
    
    downloaded = 0
    page = 1
    per_page = 50
    
    while downloaded < target_count:
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_id={taxon_id}&"
            f"quality_grade=research&"
            f"page={page}&"
            f"per_page={per_page}&"
            f"order_by=id&"
            f"order=desc"
        )
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                print(f"Конец данных на странице {page}")
                break
            
            for obs in data['results']:
                if downloaded >= target_count:
                    break
                
                if not obs.get('photos'):
                    continue
                
                photo = obs['photos'][0]
                
                photo_url = photo.get('url')
                if not photo_url:
                    continue
                
                photo_url = photo_url.replace('square', 'large')
                photo_url = photo_url.replace('thumb', 'large')
                
                try:
                    img_response = requests.get(photo_url, timeout=15)
                    img_response.raise_for_status()
                    
                    img = Image.open(BytesIO(img_response.content))
                    
                    if img.size[0] < 400 or img.size[1] < 300:
                        continue
                    
                    filename = f"{species_name}_{downloaded:03d}.jpg"
                    img_path = output_dir / filename
                    
                    img.save(img_path, 'JPEG', quality=95)
                    
                    downloaded += 1
                    
                    print(f"  [{downloaded:3d}/{target_count}] {filename} ({img.size[0]}x{img.size[1]})")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    continue
            
            page += 1
            time.sleep(2)
            
        except Exception as e:
            print(f"Ошибка API: {e}")
            time.sleep(5)
            continue
    
    total = len(list(output_dir.glob("*.jpg")))
    print(f"\n✓ {species_name}: {total} фото скачано")
    return total


def main():
    print("\n" + "="*60)
    print("СКАЧИВАНИЕ HD ФОТО С iNATURALIST")
    print("="*60)
    
    total_downloaded = 0
    
    for species_name, taxon_id in FISH_SPECIES.items():
        count = download_inaturalist_hd(species_name, taxon_id, TARGET_COUNT)
        total_downloaded += count
    
    print("\n" + "="*60)
    print("ИТОГОВОЕ КОЛИЧЕСТВО ФОТО:")
    print("="*60)
    
    for species_name in FISH_SPECIES.keys():
        count = len(list((NEW_DATA_DIR / species_name).glob("*.jpg")))
        print(f"{species_name}: {count} фото")
        
        size_mb = sum(f.stat().st_size for f in (NEW_DATA_DIR / species_name).glob("*.jpg")) / (1024*1024)
        print(f"  Размер: {size_mb:.1f} МБ")
    
    print(f"\nВсего: {total_downloaded} фото")
    print(f"Папка: {NEW_DATA_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
