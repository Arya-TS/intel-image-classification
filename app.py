import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Scene Classification",
    page_icon="🕸️",
    layout="wide"
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("intel_cnn_model.keras")

model = load_model()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = (150, 150)  # change if your model expects different size

# ---------- TITLE ----------
st.title("🕸️ Scene Classification App")
st.markdown("Upload an image and let the CNN decide the scene category.")

# ---------- COLUMNS ----------
col1, col2 = st.columns([1, 1])

# ---------- IMAGE UPLOAD ----------
with col1:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------- PREDICTION ----------
with col2:
    if uploaded_file:
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        st.subheader("Prediction")
        st.metric(
            label="Predicted Class",
            value=class_names[predicted_index]
        )

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.subheader("Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
