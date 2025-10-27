"""
Simple Streamlit app to upload a handwritten digit image and get prediction from saved MNIST model.
Run:
    pip install streamlit tensorflow pillow
    python streamlit_mnist_app.py
Then open the local Streamlit URL shown in the terminal (http://localhost:8501).
Note: Ensure mnist_cnn_model.h5 is in the same folder (created by task2_mnist_cnn.py).
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.title("MNIST Digit Classifier (CNN)")
st.write("Upload a 28x28 grayscale or color image of a handwritten digit.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

uploaded = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
if uploaded is not None:
    image = Image.open(uploaded).convert("L")  # convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Preprocess: resize to 28x28, invert if needed, normalize
    img_resized = ImageOps.fit(image, (28,28), Image.ANTIALIAS)
    img_arr = np.array(img_resized).astype("float32")/255.0
    img_arr = np.expand_dims(img_arr, axis=(0,-1))  # (1,28,28,1)
    preds = model.predict(img_arr)
    pred_label = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    st.write(f"Predicted digit: {pred_label}  —  Confidence: {conf:.3f}")
    st.bar_chart(preds[0])
else:
    st.info("Upload an image to get predictions.")