import streamlit as st

# ✅ MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Road Sign Classifier", layout="centered")

import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# --- Load Model and Class Names ---
MODEL_PATH = r"C:\My Files\Deep Learning project\road_sign_model_tuned.h5"
CLASS_NAMES_PATH = r"C:\My Files\Deep Learning project\class_names.pkl"
IMG_SIZE = 160

@st.cache_resource
def load_model_and_classes():
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

# --- Streamlit UI ---
st.title("🚦 Road Sign Classifier")
st.markdown("Upload a road sign image and get a prediction from the trained EfficientNetB3 model.")

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess ---
    img = np.array(image)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_preprocessed = preprocess_input(img_resized.astype("float32"))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # --- Predict ---
    preds = model.predict(img_batch)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # --- Display Result ---
    st.markdown("### 📌 Prediction:")
    st.success(f"**{predicted_class}** ({confidence:.2f}% confidence)")
