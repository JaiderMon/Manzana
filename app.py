import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# Cargar el modelo
model= joblib.load("modelo_manzana.bin")

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))  # Ajusta al tamaño de entrada del modelo
    image = np.array(image) / 255.0   # Normalización
    image = np.expand_dims(image, axis=0)  # Agregar dimensión batch
    return image

# Predicción
def predict_disease(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ["Sana", "Roya", "Mildiu", "Tizón"]  # Ajusta con las clases de tu modelo
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Interfaz con Streamlit
st.set_page_config(page_title="Detección de Enfermedades en Hojas", layout="centered")
st.title("🍏 Detección de Enfermedades en Hojas de Manzana")
st.write("Sube una imagen de una hoja de manzana para analizarla.")

uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    if st.button("🔍 Analizar Imagen"):
        predicted_class, confidence = predict_disease(image)
        st.success(f"**Enfermedad detectada:** {predicted_class} ({confidence:.2f}%)")
