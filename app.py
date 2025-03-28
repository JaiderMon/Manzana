
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

# URL de descarga desde Google Drive (cambia el ID del archivo)
FILE_ID = "1a2b3c4d5e6f7g8h9i0j"
MODEL_PATH = "ciencia_de_datos.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Cargar modelo
model = download_and_load_model()

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
    class_names = ['Apple___scab', 'Apple___black_rot', 'Apple___rust', 'Apple___healthy',
                   'Apple___alternaria_leaf_spot', 'Apple___brown_spot', 'Apple___gray_spot']  
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

