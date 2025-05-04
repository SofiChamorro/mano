import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# ----- Configuración de la página -----
st.set_page_config(page_title="Reconocimiento de Imágenes", layout="centered")

# ----- Cabecera -----
st.title("🧠 Reconocimiento de Imágenes")
st.markdown("Esta aplicación usa un modelo entrenado en [Teachable Machine](https://teachablemachine.withgoogle.com/) para identificar imágenes en tiempo real.")

# Mostrar versión de Python
st.caption(f"Versión de Python: {platform.python_version()}")

# Imagen de portada
st.image("https://img.asmedia.epimg.net/resizer/v2/OVU7WXOSXNKQTCQQKSQ4HOU5IQ.jpg?auth=dadbc4415df04b6490883046278fb5d4e504b4be1edc614aa4fcccbbc30267fb&width=360&height=203&smart=true")

# Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ----- Barra lateral -----
with st.sidebar:
    st.header("📸 Instrucciones")
    st.markdown("Toma una foto con tu cámara para que la IA la analice y reconozca una mano abierta o cerrada.")

# ----- Entrada de cámara -----
img_file_buffer = st.camera_input("Toma una foto con tu cámara")

# ----- Procesamiento de imagen -----
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Normalizar imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized


