import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model (relative path from this script)
try:
    model = tf.keras.models.load_model("src/CNNCrashCarModel.keras")
except Exception as e:
    st.error(f"Model input error: {e}")

# Class names
class_names = ['01–minor', '02–moderate', '03–severe']

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Main function
def predict():
    st.markdown("<h1 style='text-align: center;'>Deteksi Kerusakan Mobil</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Folder containing predefined images
    image_folder = "src/Visualization"

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        st.warning("Tidak ada gambar ditemukan dalam folder 'deployment/'.")
        return

    selected_image_name = st.selectbox("Pilih gambar mobil rusak:", image_files)

    if selected_image_name:
        image_path = os.path.join(image_folder, selected_image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            st.image(image, caption=f"Gambar: {selected_image_name}", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            return

        # Predict button
        if st.button("Predict"):
            try:
                with st.spinner("Sedang memproses gambar..."):
                    img_array = preprocess_image(image)
                    prediction = model.predict(img_array)
                    predicted_class = class_names[np.argmax(prediction)]

                st.markdown(f"<h2 style='text-align: center; color: blue;'>Prediksi: {predicted_class}</h2>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

if __name__ == '__main__':
    predict()
