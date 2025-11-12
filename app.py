import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="🧠 Image Intelligence App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# HEADER
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #3E64FF;'>🧠 Image Detection & Classification App</h1>
    <p style='text-align: center; font-size: 18px;'>Gunakan YOLO untuk deteksi objek atau model klasifikasi berbasis CNN untuk mengenali gambar.</p>
    <hr>
    """, unsafe_allow_html=True
)

# ==========================
# SIDEBAR MENU
# ==========================
menu = st.sidebar.radio("🔍 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.info("Unggah gambar dan pilih mode untuk melihat hasil deteksi atau klasifikasi.")

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MAIN LAYOUT
# ==========================
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            with st.spinner("🔎 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="✅ Hasil Deteksi", use_container_width=True)
                st.success("Deteksi selesai!")

        elif menu == "Klasifikasi Gambar":
            with st.spinner("🧮 Mengklasifikasikan gambar..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                st.markdown(f"### 🔤 Hasil Prediksi: **{class_index}**")
                st.progress(float(confidence) / 100)
                st.write(f"**Probabilitas:** {confidence:.2f}%")
                st.success("Klasifikasi selesai!")

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk memulai.")

# ==========================
# FOOTER
# ==========================
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>
    Dibuat dengan ❤️ menggunakan Streamlit, YOLOv8, dan TensorFlow.
    </p>
    """, unsafe_allow_html=True
)
