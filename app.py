import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title="Flower Classification", layout="centered")
st.title('🌸 Flower Classification CNN Model')

# Class names
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load model
model = load_model('Flower_Recog_Model.h5')

# Prediction function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class = flower_names[np.argmax(result)]
    confidence = np.max(result) * 100
    return predicted_class, confidence, result

# Upload
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_path = os.path.join('upload', uploaded_file.name)
    os.makedirs('upload', exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=250, caption="Uploaded Image")

    predicted_class, confidence, result = classify_images(file_path)

    st.success(f"🌼 This image belongs to **{predicted_class}** with **{confidence:.2f}%** confidence.")

    # Plot probabilities
    fig, ax = plt.subplots()
    ax.bar(flower_names, result)
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Flower Classes")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
