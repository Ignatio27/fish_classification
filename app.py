import streamlit as st
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import models, transforms
import pandas as pd
from qdrant_client import QdrantClient

st.set_page_config(page_title="🐟 Fish Classifier & Search", layout="wide")

@st.cache_resource
def load_classifier():
    model_path = Path("models/best_model.pth")
    metadata_path = Path("models/metadata.json")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    num_classes = metadata["num_classes"]
    classes = metadata["classes"]
    image_size = metadata.get("image_size", 160)
    
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, classes, image_size

@st.cache_resource
def load_embedder():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

@st.cache_resource
def load_qdrant():
    return QdrantClient(path="qdrant_storage")

def get_classifier_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

def get_embedder_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def embed_image(img: Image.Image, model, transform):
    x = transform(img.convert('RGB')).unsqueeze(0)
    vec = model(x).cpu().numpy().reshape(-1)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32)

st.title("🐟 Классификатор и Поиск Похожих Рыб")

tab1, tab2 = st.tabs(["🎯 Классификация", "🔍 Поиск Похожих"])

with tab1:
    st.subheader("Определение вида рыбы")
    
    uploaded_file = st.file_uploader("Загрузи изображение рыбы", 
                                     type=["jpg", "jpeg", "png"],
                                     key="classify")
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        with col2:
            model, classes, image_size = load_classifier()
            transform = get_classifier_transform(image_size)
            
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            predicted_class = classes[probabilities.argmax().item()]
            confidence = probabilities.max().item()
            
            st.metric("🎯 Предсказание", predicted_class, f"{confidence*100:.1f}%")
            
            results_df = pd.DataFrame({
                "Вид рыбы": classes,
                "Вероятность": [f"{p*100:.2f}%" for p in probabilities.cpu().numpy()]
            }).sort_values("Вероятность", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
            
            st.dataframe(results_df, use_container_width=True)

with tab2:
    st.subheader("Найти визуально похожие изображения")
    
    uploaded_file2 = st.file_uploader("Загрузи изображение для поиска", 
                                      type=["jpg", "jpeg", "png"],
                                      key="search")
    
    if uploaded_file2:
        query_image = Image.open(uploaded_file2).convert('RGB')
        st.image(query_image, caption="Запрос", width=300)
        
        embedder_model = load_embedder()
        embedder_transform = get_embedder_transform()
        
        query_vec = embed_image(query_image, embedder_model, embedder_transform)
        
        client = load_qdrant()
        
        try:
            results = client.search(
                collection_name="fish_images",
                query_vector=query_vec.tolist(),
                limit=10
            )
        except AttributeError:
            results = client.query_points(
                collection_name="fish_images",
                query=query_vec.tolist(),
                limit=10
            ).points
        
        st.subheader("🔝 ТОП-10 похожих изображений:")
        
        cols = st.columns(5)
        for idx, hit in enumerate(results):
            col = cols[idx % 5]
            
            img_path = Path(hit.payload["path"])
            label = hit.payload["label"]
            score = hit.score
            
            if img_path.exists():
                with col:
                    img = Image.open(img_path)
                    st.image(img, caption=f"{label}\n{score:.3f}", use_container_width=True)

st.markdown("---")
st.markdown("**Модель:** ResNet18 | **F1-score:** 74.77%")
