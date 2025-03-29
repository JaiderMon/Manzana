
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Detección de Enfermedades en Hojas 🍏", layout="centered")

MODEL_PATH = os.path.join(os.getcwd(), "ciencia_de_datos3.tflite")  # Ajusta según tu carpeta

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ No se encontró el modelo en {MODEL_PATH}. Verifica la ruta.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Modelo cargado exitosamente.")




def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Asegurar que tenga 3 canales (RGB)
        image = image.resize((256, 256))  # Cambiar tamaño a 256x256
        image = np.array(image) / 255.0   # Normalizar a rango [0,1]
        image = np.expand_dims(image, axis=0)  # Agregar dimensión batch (1, 256, 256, 3)
        return image
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
        return None



def predict_disease(image):
    if model is None:
        st.error("⚠️ No se pudo cargar el modelo.")
        return None, None

    processed_image = preprocess_image(image)
    if processed_image is None:
        return None, None

    prediction = model.predict(processed_image)

    class_names = [
        'Apple___scab', 'Apple___black_rot', 'Apple___rust', 'Apple___healthy',
        'Apple___alternaria_leaf_spot', 'Apple___brown_spot', 'Apple___gray_spot'
    ]  

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

st.title("🍏 Detección de Enfermedades en Hojas de Manzana")
st.write("Sube una imagen de una hoja de manzana para analizarla.")
uploaded_image = st.file_uploader("📤 Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="🖼️ Imagen cargada", use_column_width=True)
        
        if st.button("🔍 Analizar Imagen"):
            predicted_class, confidence = predict_disease(image)

            if predicted_class and confidence:
                st.success(f"✅ **Enfermedad detectada:** {predicted_class} ({confidence:.2f}%)")
            else:
                st.error("❌ No se pudo realizar la predicción.")
    except Exception as e:
        st.error(f"❌ Error al cargar la imagen: {e}")

