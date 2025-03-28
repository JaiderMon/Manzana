
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ⚠️ Configurar la página debe ser lo PRIMERO que hace Streamlit
st.set_page_config(page_title="Detección de Enfermedades en Hojas 🍏", layout="centered")

# **Carga del modelo**
MODEL_PATH = r"C:\Users\JAIDER\Downloads\Ciencias de datos\ciencia_de_datos.h5"

@st.cache_resource
def load_model():
    """Carga el modelo desde la ruta local."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ No se encontró el modelo en {MODEL_PATH}. Verifica la ruta.")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Modelo cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

# **Cargar modelo**
model = load_model()

# **Preprocesamiento de imagen**
def preprocess_image(image):
    """Preprocesa la imagen para que coincida con la entrada del modelo."""
    try:
        image = image.convert("RGB")  # Asegurar que tenga 3 canales (RGB)
        image = image.resize((256, 256))  # Cambiar tamaño a 256x256
        image = np.array(image) / 255.0   # Normalizar a rango [0,1]
        image = np.expand_dims(image, axis=0)  # Agregar dimensión batch (1, 256, 256, 3)
        return image
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
        return None


# **Función de predicción**
def predict_disease(image):
    """Realiza la predicción de la enfermedad en la hoja."""
    if model is None:
        st.error("⚠️ No se pudo cargar el modelo.")
        return None, None

    processed_image = preprocess_image(image)
    if processed_image is None:
        return None, None

    prediction = model.predict(processed_image)

    # ⚠️ Lista de clases (ajustar según el modelo)
    class_names = [
        'Apple___scab', 'Apple___black_rot', 'Apple___rust', 'Apple___healthy',
        'Apple___alternaria_leaf_spot', 'Apple___brown_spot', 'Apple___gray_spot'
    ]  

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# **Interfaz de la aplicación**
st.title("🍏 Detección de Enfermedades en Hojas de Manzana")
st.write("Sube una imagen de una hoja de manzana para analizarla.")

# **Subida de imagen**
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

