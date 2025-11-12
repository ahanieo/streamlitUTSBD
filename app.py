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
    page_title="♻️ E-Waste Detection App",
    page_icon="♻️",
    layout="wide"
)

# ==========================
# CUSTOM THEME (CSS)
# ==========================
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
        color: #004d40;
        font-family: "Poppins", sans-serif;
    }

    /* Title styling */
    h1 {
        color: #1b5e20;
        text-align: center;
        font-weight: 700;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #a5d6a7;
    }

    /* Buttons */
    .stButton button {
        background-color: #388e3c;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
    }
    .stButton button:hover {
        background-color: #2e7d32;
        transform: scale(1.03);
    }

    /* Upload box */
    div[data-testid="stFileUploader"] {
        background-color: #c8e6c9;
        padding: 1em;
        border-radius: 10px;
    }

    /* Progress bar color */
    div[data-testid="stProgressBar"] > div > div > div {
        background-color: #4caf50;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown("""
<h1>♻️ E-Waste Detection & Classification App</h1>
<p style="text-align:center; font-size:18px; color:#1b5e20;">
Gunakan YOLO untuk mendeteksi komponen e-waste atau model klasifikasi CNN untuk mengenali jenis limbah elektronik.
</p>
<hr style="border:1px solid #66bb6a;">
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi E-Waste"])
st.sidebar.info("Unggah gambar e-waste dan pilih mode deteksi atau klasifikasi.")

# ==========================
# MAIN APP
# ==========================
uploaded_file = st.file_uploader("📸 Unggah Gambar E-Waste", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            with st.spinner("🔍 Sedang mendeteksi e-waste..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi E-Waste", use_container_width=True)
                st.success("✅ Deteksi selesai!")

        elif menu == "Klasifikasi E-Waste":
            with st.spinner("🧠 Mengklasifikasikan jenis e-waste..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                # Contoh label kelas
                class_labels = [
    "Battery",
    "Keyboard",
    "Microwave",
    "Mobile",
    "Mouse",
    "PCB",
    "Player",
    "Printer",
    "Television",
    "Washing Machine"
]
                predicted_label = class_labels[class_index] if class_index < len(class_labels) else f"Class {class_index}"

                st.markdown(f"### 🔤 Jenis E-Waste: **{predicted_label}**")
                st.progress(float(confidence) / 100)
                st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                st.success("✅ Klasifikasi selesai!")

else:
    st.warning("📂 Silakan unggah gambar e-waste terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
🌱 Dibuat untuk mendukung pengelolaan limbah elektronik berkelanjutan menggunakan AI & Streamlit.
</p>
""", unsafe_allow_html=True)
