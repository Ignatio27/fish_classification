import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models, transforms
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def make_id(path: str) -> int:
    h = hashlib.md5(path.encode('utf-8')).hexdigest()
    return int(h[:16], 16)

def build_embedder(device: str = "cpu"):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    return model, preprocess

@torch.no_grad()
def embed_image(img: Image.Image, model, preprocess, device: str) -> np.ndarray:
    x = preprocess(img.convert('RGB')).unsqueeze(0).to(device)
    vec = model(x).cpu().numpy().reshape(-1)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32)

def main():
    data_root = Path("data")
    storage_path = Path("qdrant_storage")
    collection = "fish_images"
    
    image_paths = []
    for folder in ["goldfish", "pike", "smelt", "perch", "zander"]:
        folder_path = data_root / folder
        if folder_path.exists():
            for p in folder_path.glob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    image_paths.append(p)
    
    if not image_paths:
        raise SystemExit(f"No images found in {data_root.resolve()}")
    
    print(f"Found {len(image_paths)} images to index.")
    
    client = QdrantClient(path=str(storage_path))
    
    if collection in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection)
    
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_embedder(device=device)
    
    points = []
    for p in tqdm(image_paths, desc="Embedding"):
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        
        vec = embed_image(img, model, preprocess, device)
        rel_path = p.as_posix()
        label = p.parent.name
        pid = make_id(rel_path)
        
        points.append(PointStruct(
            id=pid,
            vector=vec.tolist(),
            payload={"path": rel_path, "label": label}
        ))
        
        if len(points) >= 128:
            client.upsert(collection_name=collection, points=points)
            points = []
    
    if points:
        client.upsert(collection_name=collection, points=points)
    
    print(f"Indexing done. Total: {len(image_paths)} images")

if __name__ == "__main__":
    main()
