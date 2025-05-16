import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Load model and class indices once
@st.cache_resource
def load_model_and_labels():
    model = load_model("custard_apple_disease_model.keras")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}
    return model, class_labels

model, class_labels = load_model_and_labels()

# UI Layout
st.title("🍏 Custard apple Disease Detection       ")
st.write("Upload an image of a custard apple, and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
print("   ")
print("   ")
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Resize for model input (128x128)
    resized_img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display a resized version for the UI (optional: limit display size)
    display_image = image.copy()
    display_image.thumbnail((400, 400))  # Limit display size to 400x400
    st.image(display_image, caption='Uploaded Image', use_container_width=False)

    # Show prediction
    st.markdown(f"###  Predicted Disease: `{predicted_label}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}`")