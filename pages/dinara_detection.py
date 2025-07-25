import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Детекция опухолей мозга (YOLOv11)")

# --- Конфигурация ---
MODEL_DIR = "models"
MODEL_NAME = "dinara_yolov11_best.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


# --- Загрузка обученной модели ---
@st.cache_resource # Кэшировать модель, чтобы избежать перезагрузки при каждом запуске
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Ошибка: Файл модели не найден по пути {path}. Убедитесь, что '{MODEL_NAME}' находится в каталоге '{MODEL_DIR}'.")
        return None
    try:
        model = YOLO(path)
        st.success(f"Модель '{MODEL_NAME}' успешно загружена! Можно приступать к использованию.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

model = load_model(MODEL_PATH)

# ===== Загрузка изображений =====
tab1, tab2 = st.tabs(["📂 Загрузить изображение", "🌐 По ссылке"])

imgs = []

with tab1:
    uploaded_files = st.file_uploader("Загрузите одно или несколько изображений:", type=["jpg", "png"], accept_multiple_files=True)
    for file in uploaded_files:
        imgs.append(Image.open(file))

with tab2:
    url = st.text_input("Вставьте ссылку на изображение (jpeg/png):")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            imgs.append(img)
        except:
            st.error("Ошибка загрузки изображения по ссылке.")

# ===== Инференс и отображение =====
if imgs:
    for img in imgs:
        st.markdown("---")
        st.image(img, caption="Оригинал", width=400)

        # YOLOv11 предсказание
        results = model.predict(img, conf=0.25)

        for r in results:
            annotated_img = r.plot()  # изображение с бокcами
            st.image(annotated_img, caption="Результат детекции", width=600)

# ===== Информация о модели =====
with st.expander("ℹ️ Информация о модели и метрики"):
    st.markdown("""
    **YOLOv11** использовалась для детекции опухолей на MR-снимках.

    - 📦 **Размер обучающей выборки:** 310 изображений  
    - 📦 **Размер валидационной выборки:** 75 изображений  
    - 🔁 **Эпохи обучения:** 50 эпох *3 модели = 150  
    - 🎯 **mAP50:** 0.88  
    - 🎯 **mAP50-95:** 0.60
    """)


    # Загрузка MaP
    st.markdown("**box_f1_curve - детекция опухолей мозга (YOLOv11):**")
    cm_image = Image.open("images/dinara_map.jpg")
    st.image(cm_image, caption="Confusion Matrix", use_container_width=True)

    # Загрузка PR-кривой из файла
    st.markdown("**PR-кривая - детекция опухолей мозга (YOLOv11):**")
    pr_curve = Image.open("images/dinara_BoxPR_curve.png")
    st.image(pr_curve, caption="PR-кривая", use_container_width=True)

    # Загрузка confusion matrix из файла
    st.markdown("**Матрица ошибок - детекция опухолей мозга (YOLOv11):**")
    cm_image = Image.open("images/dinara_confusion_matrix.png")
    st.image(cm_image, caption="Confusion Matrix", use_container_width=True)


