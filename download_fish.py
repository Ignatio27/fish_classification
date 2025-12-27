import os
from pathlib import Path
from bing_image_downloader import downloader
import time

DATA_DIR = Path("data/raw_fish")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FISH_SPECIES = {
    "goldfish": "Carassius auratus",
    "pike": "Esox lucius",
    "smelt": "Osmerus mordax",
    "perch": "Perca fluviatilis",
    "zander": "Sander lucioperca"
}

for folder_name, scientific_name in FISH_SPECIES.items():
    output_dir = DATA_DIR / folder_name
    
    print(f"\nСкачиваю {scientific_name}...")
    print(f"Папка: {output_dir}")
    
    downloader.download(
        scientific_name,
        limit=300,
        output_dir="dataset",
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=False
    )
    
    time.sleep(3)

print("Готово!")
