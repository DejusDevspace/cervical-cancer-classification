import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  #type:ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img #type:ignore
from PIL import Image

# Load your trained model
@st.cache_resource
def load_trained_model():
    return load_model("models/cervical_cancer_classifier2.keras")

model = load_trained_model()

# Define the class names
class_names = ["type1", "type2", "type3"]

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB just in case
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # normalize if your model was trained that way
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit UI
st.title("Cervical Cancer Classification")
st.write("Upload a cervical cell image to classify it as Type 1, Type 2, or Type 3.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
